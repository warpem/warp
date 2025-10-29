using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{

    public void ProcessCTFSimultaneous(ProcessingOptionsMovieCTF options)
    {
        IsProcessing = true;

        if (!Directory.Exists(PowerSpectrumDir))
            Directory.CreateDirectory(PowerSpectrumDir);

        int2 DimsFrame;
        {
            Movie FirstMovie = new Movie(System.IO.Path.Combine(DataOrProcessingDirectoryName, TiltMoviePaths[0]));
            MapHeader HeaderMovie = MapHeader.ReadFromFile(FirstMovie.DataPath);
            DimsFrame = new int2(new float2(HeaderMovie.Dimensions.X, HeaderMovie.Dimensions.Y) / (float)options.DownsampleFactor + 1) / 2 * 2;
        }

        #region Dimensions and grids

        int NFrames = NTilts;
        int2 DimsImage = DimsFrame;
        int2 DimsRegionBig = new int2(1536);
        int2 DimsRegion = new int2(options.Window, options.Window);

        float OverlapFraction = 0.5f;
        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, new int2(DimsRegionBig.X, DimsRegionBig.Y), OverlapFraction, out DimsPositionGrid);
        float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X + DimsRegionBig.X / 2 - DimsImage.X / 2,
                                                                     v.Y + DimsRegionBig.Y / 2 - DimsImage.Y / 2,
                                                                     0) *
                                                                 (float)options.BinnedPixelSizeMean * 1e-4f).ToArray();
        int NPositions = (int)DimsPositionGrid.Elements();

        int3 CTFSpectraGrid = new int3(DimsPositionGrid.X, DimsPositionGrid.Y, NFrames);

        int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
        int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
        int NFreq = MaxFreqExclusive - MinFreqInclusive;

        #endregion

        #region Allocate memory

        // GPU
        Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
        Image CTFMean = new Image(new int3(DimsRegion), true);
        Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
        Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

        // CPU
        float2[] GlobalPS1D = null;
        float[][] LocalPS1D = new float[NPositions * NFrames][];
        Cubic1D GlobalBackground = null, GlobalScale = null;
        CTF GlobalCTF = null;
        float2 GlobalPlaneAngle = new float2();

        #endregion

        #region Helper methods

        Func<float[], float[]> GetDefocusGrid = (defoci) =>
        {
            float[] Result = new float[NPositions * NFrames];

            for (int t = 0; t < NFrames; t++)
            {
                float3 Normal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);
                Normal = Matrix3.Euler(0, Angles[t] * (AreAnglesInverted ? -1 : 1) * Helper.ToRad, 0) * Normal;
                Normal = Matrix3.Euler(0, 0, -TiltAxisAngles[t] * Helper.ToRad) * Normal;
                for (int i = 0; i < NPositions; i++)
                    Result[t * NPositions + i] = defoci[t] - float3.Dot(Normal, PositionGridPhysical[i]) / Normal.Z;
            }

            return Result;
        };

        #region Background fitting methods

        Action UpdateBackgroundFit = () =>
        {
            float2[] ForPS1D = GlobalPS1D.Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray();
            Cubic1D.FitCTF(ForPS1D,
                GlobalCTF.Get1D(GlobalPS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 1)).ToArray(),
                GlobalCTF.GetZeros(),
                GlobalCTF.GetPeaks(),
                out GlobalBackground,
                out GlobalScale);
        };

        Action<bool> UpdateRotationalAverage = keepbackground =>
        {
            float[] MeanData = CTFMean.GetHost(Intent.Read)[0];

            Image CTFMeanCorrected = new Image(new int3(DimsRegion), true);
            float[] MeanCorrectedData = CTFMeanCorrected.GetHost(Intent.Write)[0];

            // Subtract current background estimate from spectra, populate coords.
            Helper.ForEachElementFT(DimsRegion,
                (x, y, xx, yy, r, a) =>
                {
                    int i = y * (DimsRegion.X / 2 + 1) + x;
                    MeanCorrectedData[i] = MeanData[i] - GlobalBackground.Interp(r / DimsRegion.X);
                });

            Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

            GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                CTFCoordsCart.GetDevice(Intent.Read),
                (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                (uint)DimsRegion.X,
                new[] { GlobalCTF.ToStruct() },
                GlobalCTF.ToStruct(),
                0,
                (uint)DimsRegion.X / 2,
                1,
                CTFAverage1D.GetDevice(Intent.Write));

            //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

            float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
            float2[] ForPS1D = new float2[GlobalPS1D.Length];
            if (keepbackground)
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + GlobalBackground.Interp((float)i / DimsRegion.X));
            else
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
            MathHelper.UnNaN(ForPS1D);

            GlobalPS1D = ForPS1D;

            CTFMeanCorrected.Dispose();
            CTFAverage1D.Dispose();
        };

        #endregion

        #endregion

        // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

        #region Create spectra

        int PlanForw = GPU.CreateFFTPlan(new int3(DimsRegionBig), (uint)NPositions);
        int PlanBack = GPU.CreateIFFTPlan(new int3(DimsRegion), (uint)NPositions);

        Movie[] TiltMovies;
        Image[] TiltMovieData;
        LoadMovieData(options, out TiltMovies, out TiltMovieData, false, out _, out _);

        for (int t = 0; t < NTilts; t++)
        {
            Image TiltMovieAverage = TiltMovieData[t];

            GPU.Normalize(TiltMovieAverage.GetDevice(Intent.Read),
                TiltMovieAverage.GetDevice(Intent.Write),
                (uint)TiltMovieAverage.ElementsReal,
                1);

            Image MovieCTFMean = new Image(new int3(DimsRegion), true);

            GPU.CreateSpectra(TiltMovieAverage.GetDevice(Intent.Read),
                DimsImage,
                TiltMovieAverage.Dims.Z,
                PositionGrid,
                NPositions,
                DimsRegionBig,
                CTFSpectraGrid.Slice(),
                DimsRegion,
                CTFSpectra.GetDeviceSlice(t * (int)CTFSpectraGrid.ElementsSlice(), Intent.Write),
                MovieCTFMean.GetDevice(Intent.Write),
                PlanForw,
                PlanBack);

            CTFMean.Add(MovieCTFMean);

            MovieCTFMean.Dispose();
            TiltMovieAverage.FreeDevice();
        }

        GPU.DestroyFFTPlan(PlanBack);
        GPU.DestroyFFTPlan(PlanForw);

        CTFMean.Multiply(1f / NTilts);

        #endregion

        // Populate address arrays for later.

        #region Init addresses

        {
            float2[] CoordsData = new float2[CTFCoordsCart.ElementsSliceComplex];

            Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => CoordsData[y * (DimsRegion.X / 2 + 1) + x] = new float2(r, a));
            CTFCoordsCart.UpdateHostWithComplex(new[] { CoordsData });

            CoordsData = new float2[NFreq * DimsRegion.X];
            Helper.ForEachElement(CTFCoordsPolarTrimmed.DimsSlice, (x, y) =>
            {
                float Angle = (float)y / DimsRegion.X * (float)Math.PI;
                float Ny = 1f / DimsRegion.X;
                CoordsData[y * NFreq + x] = new float2((x + MinFreqInclusive) * Ny, Angle);
            });
            CTFCoordsPolarTrimmed.UpdateHostWithComplex(new[] { CoordsData });
        }

        #endregion

        #region Initial 1D spectra

        // Mean spectrum to fit background
        {
            Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

            GPU.CTFMakeAverage(CTFMean.GetDevice(Intent.Read),
                CTFCoordsCart.GetDevice(Intent.Read),
                (uint)CTFMean.ElementsSliceReal,
                (uint)DimsRegion.X,
                new[] { new CTF().ToStruct() },
                new CTF().ToStruct(),
                0,
                (uint)DimsRegion.X / 2,
                1,
                CTFAverage1D.GetDevice(Intent.Write));

            //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

            float[] CTFAverage1DData = CTFAverage1D.GetHost(Intent.Read)[0];
            float2[] ForPS1D = new float2[DimsRegion.X / 2];
            for (int i = 0; i < ForPS1D.Length; i++)
                ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(CTFAverage1DData[i], 4));
            GlobalPS1D = ForPS1D;

            CTFAverage1D.Dispose();
        }

        // Individual 1D spectra for initial grid search below
        {
            Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

            for (int s = 0; s < NPositions * NFrames; s++)
            {
                GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(s, Intent.Read),
                    CTFCoordsCart.GetDevice(Intent.Read),
                    (uint)CTFMean.ElementsSliceReal,
                    (uint)DimsRegion.X,
                    new[] { new CTF().ToStruct() },
                    new CTF().ToStruct(),
                    0,
                    (uint)DimsRegion.X / 2,
                    1,
                    CTFAverage1D.GetDevice(Intent.Write));

                //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

                LocalPS1D[s] = CTFAverage1D.GetHostContinuousCopy();
            }

            CTFAverage1D.Dispose();
        }

        #endregion

        #region Do initial fit on mean 1D PS

        {
            float2[] ForPS1D = GlobalPS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();

            float[] CurrentBackground;

            // Get a very rough background spline fit with 3-5 nodes
            int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M * 2 / 3));
            GlobalBackground = Cubic1D.Fit(ForPS1D, NumNodes);

            CurrentBackground = GlobalBackground.Interp(ForPS1D.Select(p => p.X).ToArray());
            float[][] SubtractedLocal1D = new float[LocalPS1D.Length][];
            for (int s = 0; s < LocalPS1D.Length; s++)
            {
                SubtractedLocal1D[s] = new float[NFreq * 2 / 3];
                for (int f = 0; f < NFreq * 2 / 3; f++)
                    SubtractedLocal1D[s][f] = LocalPS1D[s][f + MinFreqInclusive] - CurrentBackground[f];
            }

            float[] GridDeltas = GetDefocusGrid(Helper.ArrayOfConstant(0f, NFrames));

            float ZMin = (float)options.ZMin;
            float ZMax = (float)options.ZMax;
            float PhaseMin = 0f;
            float PhaseMax = options.DoPhase ? 1f : 0f;

            float ZStep = Math.Max(0.01f, (ZMax - ZMin) / 200f);

            float BestZ = 0, BestPhase = 0, BestScore = -999;
            Parallel.For(0, (int)((ZMax - ZMin + ZStep - 1e-6f) / ZStep), zi =>
            {
                float z = ZMin + zi * ZStep;

                for (float p = PhaseMin; p <= PhaseMax; p += 0.01f)
                {
                    float Score = 0;

                    for (int s = 0; s < NPositions * NFrames; s++)
                    {
                        CTF CurrentParams = new CTF
                        {
                            PixelSize = options.BinnedPixelSizeMean,

                            Defocus = (decimal)(z + GridDeltas[s]),
                            PhaseShift = (decimal)p,

                            Cs = options.Cs,
                            Voltage = options.Voltage,
                            Amplitude = options.Amplitude
                        };
                        float[] SimulatedCTF = CurrentParams.Get1D(GlobalPS1D.Length, true).Skip(MinFreqInclusive).Take(Math.Max(2, NFreq * 2 / 3)).ToArray();
                        MathHelper.NormalizeInPlace(SimulatedCTF);

                        Score += MathHelper.CrossCorrelate(SubtractedLocal1D[s], SimulatedCTF);
                    }

                    lock(ForPS1D)
                        if (Score > BestScore)
                        {
                            BestScore = Score;
                            BestZ = z;
                            BestPhase = p;
                        }
                }
            });

            GlobalCTF = new CTF
            {
                PixelSize = options.BinnedPixelSizeMean,

                Defocus = (decimal)BestZ,
                PhaseShift = (decimal)BestPhase,

                Cs = options.Cs,
                Voltage = options.Voltage,
                Amplitude = options.Amplitude
            };

            //UpdateRotationalAverage(true);  // This doesn't have a nice background yet.

            // Scale everything to one common defocus value
            {
                CTFStruct[] LocalParams = GridDeltas.Select((v, i) =>
                {
                    CTF Local = GlobalCTF.GetCopy();
                    Local.Defocus += (decimal)v;
                    Local.Scale = (decimal)Math.Pow(1 - Math.Abs(Math.Sin(Angles[i / NPositions] * Helper.ToRad)), 2);

                    return Local.ToStruct();
                }).ToArray();

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                CTF CTFAug = GlobalCTF.GetCopy();

                GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                    CTFCoordsCart.GetDevice(Intent.Read),
                    (uint)CTFSpectra.ElementsSliceReal,
                    (uint)DimsRegion.X,
                    LocalParams,
                    CTFAug.ToStruct(),
                    0,
                    (uint)DimsRegion.X / 2,
                    (uint)LocalParams.Length,
                    CTFAverage1D.GetDevice(Intent.Write));

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                for (int i = 0; i < RotationalAverageData.Length; i++)
                    GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(GlobalPS1D);

                CTFAverage1D.Dispose();
                CTFSpectra.FreeDevice();
            }

            UpdateBackgroundFit(); // Now get a reasonably nice background.

            #region For debug purposes, check what the background-subtracted average looks like at this point

            // Scale everything to one common defocus value
            if (false)
            {
                Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                // Construct background in Cartesian coordinates.
                Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => { CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X); });

                CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                CTFStruct[] LocalParams = GridDeltas.Select(v =>
                {
                    CTF Local = GlobalCTF.GetCopy();
                    Local.Defocus += (decimal)v;

                    return Local.ToStruct();
                }).ToArray();

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                CTF CTFAug = GlobalCTF.GetCopy();

                GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                    CTFCoordsCart.GetDevice(Intent.Read),
                    (uint)CTFSpectra.ElementsSliceReal,
                    (uint)DimsRegion.X,
                    LocalParams,
                    CTFAug.ToStruct(),
                    0,
                    (uint)DimsRegion.X / 2,
                    (uint)LocalParams.Length,
                    CTFAverage1D.GetDevice(Intent.Write));

                float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                for (int i = 0; i < RotationalAverageData.Length; i++)
                    GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                MathHelper.UnNaN(GlobalPS1D);

                CTFSpectra.AddToSlices(CTFSpectraBackground);

                CTFSpectraBackground.Dispose();
                CTFAverage1D.Dispose();
                CTFSpectra.FreeDevice();
            }

            #endregion
        }

        #endregion

        // Do BFGS optimization of defocus, astigmatism and phase shift,
        // using 2D simulation for comparison

        double[] DefociFromMovies = TiltMoviePaths.Select(p => (double)(new Movie(System.IO.Path.Combine(DataOrProcessingDirectoryName, p)).CTF.Defocus)).ToArray();

        double[] StartParams = new double[5];
        StartParams[0] = 0;
        StartParams[1] = 0;
        StartParams[2] = (double)GlobalCTF.DefocusDelta;
        StartParams[3] = (double)GlobalCTF.DefocusDelta;
        StartParams[4] = (double)GlobalCTF.DefocusAngle / 20 * Helper.ToRad;
        StartParams = Helper.Combine(StartParams,
            DefociFromMovies, //Helper.ArrayOfConstant((double)GlobalCTF.Defocus, NFrames),
            Helper.ArrayOfConstant((double)GlobalCTF.PhaseShift, Math.Max(1, NFrames / 3)));

        float3[] GridCoordsByAngle = Helper.ArrayOfFunction(i => new float3((float)i / (NFrames - 1), 0, 0), NFrames);
        float3[] GridCoordsByDose = Helper.ArrayOfFunction(i => new float3(Dose[i] / MathHelper.Max(Dose), 0, 0), NFrames);

        #region BFGS

        {
            // Second iteration will have a nicer background
            for (int opt = 0; opt < 1; opt++)
            {
                if (opt > 0)
                    NFreq = Math.Min(NFreq + 10, DimsRegion.X / 2 - MinFreqInclusive - 1);

                Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
                CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

                #region Create background and scale

                float[] CurrentScale = Helper.ArrayOfConstant(1f, GlobalPS1D.Length); // GlobalScale.Interp(GlobalPS1D.Select(p => p.X).ToArray());

                Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
                float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

                // Trim polar to relevant frequencies, and populate coordinates.
                Parallel.For(0, DimsRegion.X, y =>
                {
                    for (int x = 0; x < NFreq; x++)
                        CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
                });
                //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

                // Background is just 1 line since we're in polar.
                Image CurrentBackground = new Image(GlobalBackground.Interp(GlobalPS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

                CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
                CurrentBackground.Dispose();

                //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

                #endregion

                #region Eval and Gradient methods

                // Helper method for getting CTFStructs for the entire spectra grid.
                Func<double[], CTF, float[], float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, phaseValues, defocusValues, defocusDeltaValues) =>
                {
                    CTF Local = ctf.GetCopy();
                    Local.DefocusAngle = (decimal)(input[4] * 20 / (Math.PI / 180));

                    CTFStruct LocalStruct = Local.ToStruct();
                    CTFStruct[] LocalParams = new CTFStruct[defocusValues.Length];
                    for (int f = 0; f < NFrames; f++)
                    for (int p = 0; p < NPositions; p++)
                    {
                        LocalParams[f * NPositions + p] = LocalStruct;
                        LocalParams[f * NPositions + p].Defocus = defocusValues[f * NPositions + p] * -1e-6f;
                        LocalParams[f * NPositions + p].DefocusDelta = defocusDeltaValues[f] * -1e-6f;
                        LocalParams[f * NPositions + p].PhaseShift = phaseValues[f] * (float)Math.PI;
                    }

                    return LocalParams;
                };

                Func<double[], double> Eval = input =>
                {
                    GlobalPlaneAngle = new float2((float)input[0], (float)input[1]) * Helper.ToDeg;

                    CubicGrid TempGridPhase = new CubicGrid(new int3(Math.Max(1, NFrames / 3), 1, 1), input.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v).ToArray());
                    CubicGrid TempGridDefocus = new CubicGrid(new int3(NFrames, 1, 1), input.Skip(5).Take(NFrames).Select(v => (float)v).ToArray());
                    CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)input[2] });

                    float[] PhaseValues = TempGridPhase.GetInterpolated(GridCoordsByDose);
                    float[] DefocusValues = GetDefocusGrid(TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                    float[] DefocusDeltaValues = TempGridDefocusDelta.GetInterpolated(GridCoordsByDose);

                    CTFStruct[] LocalParams = EvalGetCTF(input, GlobalCTF, PhaseValues, DefocusValues, DefocusDeltaValues);

                    float[] Result = new float[LocalParams.Length];

                    GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                        CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                        CTFSpectraScale.GetDevice(Intent.Read),
                        (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                        LocalParams,
                        Result,
                        (uint)LocalParams.Length);

                    float Score = Result.Sum();

                    if (float.IsNaN(Score) || float.IsInfinity(Score))
                        throw new Exception("Bad score.");

                    return Score;
                };

                Func<double[], double[]> Gradient = input =>
                {
                    const float Step = 0.0025f;
                    double[] Result = new double[input.Length];

                    for (int i = 0; i < input.Length; i++)
                    {
                        if (!options.DoPhase && i >= 5 + NFrames)
                            continue;

                        double[] UpperInput = new double[input.Length];
                        input.CopyTo(UpperInput, 0);
                        UpperInput[i] += Step;
                        double UpperValue = Eval(UpperInput);

                        double[] LowerInput = new double[input.Length];
                        input.CopyTo(LowerInput, 0);
                        LowerInput[i] -= Step;
                        double LowerValue = Eval(LowerInput);

                        Result[i] = (UpperValue - LowerValue) / (2f * Step);
                    }

                    if (Result.Any(i => double.IsNaN(i) || double.IsInfinity(i)))
                        throw new Exception("Bad score.");

                    return Result;
                };

                #endregion

                #region Do optimization

                // StartParams are initialized above, before the optimization loop

                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
                {
                    MaxIterations = 15
                };
                Optimizer.Maximize(StartParams);

                #endregion

                #region Retrieve parameters

                GlobalCTF.Defocus = (decimal)Optimizer.Solution.Skip(5).Take(NFrames).Select(v => (float)v).Average();
                GlobalCTF.PhaseShift = (decimal)Optimizer.Solution.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v).Average();
                GlobalCTF.DefocusDelta = (decimal)(Optimizer.Solution[2]) / 1;
                GlobalCTF.DefocusAngle = (decimal)(Optimizer.Solution[4] * 20 * Helper.ToDeg);

                if (GlobalCTF.DefocusDelta < 0)
                {
                    GlobalCTF.DefocusAngle += 90;
                    GlobalCTF.DefocusDelta *= -1;
                }

                GlobalCTF.DefocusAngle = ((int)GlobalCTF.DefocusAngle + 180 * 99) % 180;

                GlobalPlaneAngle = new float2((float)Optimizer.Solution[0],
                    (float)Optimizer.Solution[1]) * Helper.ToDeg;

                {
                    CubicGrid TempGridPhase = new CubicGrid(new int3(Math.Max(1, NFrames / 3), 1, 1), StartParams.Skip(5 + NFrames).Take(Math.Max(1, NFrames / 3)).Select(v => (float)v).ToArray());
                    CubicGrid TempGridDefocusDelta = new CubicGrid(new int3(1, 1, 1), new[] { (float)GlobalCTF.DefocusDelta });
                    CubicGrid TempGridDefocus = new CubicGrid(new int3(NFrames, 1, 1), StartParams.Skip(5).Take(NFrames).Select(v => (float)v).ToArray());

                    GridCTFDefocus = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocus.GetInterpolated(GridCoordsByAngle));
                    GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), TempGridDefocusDelta.GetInterpolated(GridCoordsByDose));
                    GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), Helper.ArrayOfConstant((float)GlobalCTF.DefocusAngle, NTilts));
                    GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), TempGridPhase.GetInterpolated(GridCoordsByDose));
                }

                #endregion

                // Dispose GPU resources manually because GC can't be bothered to do it in time.
                CTFSpectraPolarTrimmed.Dispose();
                CTFSpectraScale.Dispose();

                #region Get nicer envelope fit

                // Scale everything to one common defocus value
                {
                    float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

                    float[] DefocusValues = GetDefocusGrid(GridCTFDefocus.GetInterpolated(GridCoords));
                    float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
                    float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
                    float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);

                    CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                    for (int f = 0; f < NFrames; f++)
                    {
                        for (int p = 0; p < NPositions; p++)
                        {
                            CTF Local = GlobalCTF.GetCopy();
                            Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                            Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                            Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                            Local.PhaseShift = (decimal)PhaseValues[f];

                            LocalParams[f * NPositions + p] = Local.ToStruct();
                        }
                    }

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                    CTF CTFAug = GlobalCTF.GetCopy();

                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                        CTFCoordsCart.GetDevice(Intent.Read),
                        (uint)CTFSpectra.ElementsSliceReal,
                        (uint)DimsRegion.X,
                        LocalParams,
                        CTFAug.ToStruct(),
                        0,
                        (uint)DimsRegion.X / 2,
                        (uint)LocalParams.Length,
                        CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();

                    UpdateBackgroundFit(); // Now get a nice background.
                }

                #endregion
            }
        }

        #endregion

        #region Create global, and per-tilt average spectra

        {
            TiltPS1D = new ObservableCollection<float2[]>();
            TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
            TiltSimulatedScale = new ObservableCollection<Cubic1D>();
            Image AllPS2D = new Image(new int3(DimsRegion.X, DimsRegion.X / 2, NTilts));

            float3[] GridCoords = Helper.ArrayOfFunction(i => new float3(0, 0, (float)i / (NFrames - 1)), NFrames);

            float[] DefocusValues = GetDefocusGrid(GridCTFDefocus.GetInterpolated(GridCoords));
            float[] DefocusDeltaValues = GridCTFDefocusDelta.GetInterpolated(GridCoords);
            float[] DefocusAngleValues = GridCTFDefocusAngle.GetInterpolated(GridCoords);
            float[] PhaseValues = GridCTFPhase.GetInterpolated(GridCoords);

            // Scale everything to one common defocus value
            {
                Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                // Construct background in Cartesian coordinates.
                Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => { CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = GlobalBackground.Interp(r / DimsRegion.X); });

                CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                CTFStruct[] LocalParams = new CTFStruct[DefocusValues.Length];
                for (int f = 0; f < NFrames; f++)
                {
                    for (int p = 0; p < NPositions; p++)
                    {
                        CTF Local = GlobalCTF.GetCopy();
                        Local.Defocus = (decimal)DefocusValues[f * NPositions + p];
                        Local.DefocusDelta = (decimal)DefocusDeltaValues[f];
                        Local.DefocusAngle = (decimal)DefocusAngleValues[f];
                        Local.PhaseShift = (decimal)PhaseValues[f];
                        Local.Scale = (decimal)Math.Pow(1 - Math.Abs(Math.Sin(Angles[f] * Helper.ToRad)), 2);

                        LocalParams[f * NPositions + p] = Local.ToStruct();
                    }
                }

                Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));
                CTF CTFAug = GlobalCTF.GetCopy();

                {
                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                        CTFCoordsCart.GetDevice(Intent.Read),
                        (uint)CTFSpectra.ElementsSliceReal,
                        (uint)DimsRegion.X,
                        LocalParams,
                        CTFAug.ToStruct(),
                        0,
                        (uint)DimsRegion.X / 2,
                        (uint)LocalParams.Length,
                        CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    PS1D = GlobalPS1D.ToArray();
                }

                #region Now go through all tilts

                for (int t = 0; t < NTilts; t++)
                {
                    CTFAug.Defocus = (decimal)GridCTFDefocus.FlatValues[t];

                    GPU.CTFMakeAverage(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                        CTFCoordsCart.GetDevice(Intent.Read),
                        (uint)CTFSpectra.ElementsSliceReal,
                        (uint)DimsRegion.X,
                        LocalParams.Skip(t * NPositions).Take(NPositions).ToArray(),
                        CTFAug.ToStruct(),
                        0,
                        (uint)DimsRegion.X / 2,
                        (uint)NPositions,
                        CTFAverage1D.GetDevice(Intent.Write));

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    for (int i = 0; i < RotationalAverageData.Length; i++)
                        GlobalPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
                    MathHelper.UnNaN(GlobalPS1D);

                    TiltPS1D.Add(GlobalPS1D.ToArray());
                    TiltSimulatedBackground.Add(new Cubic1D(GlobalBackground.Data.ToArray()));
                    TiltSimulatedScale.Add(new Cubic1D(GlobalScale.Data.ToArray()));

                    #region Make 2D power spectrum for display

                    Image Sum2D = new Image(CTFSpectra.Dims.Slice(), true);
                    GPU.ReduceMean(CTFSpectra.GetDeviceSlice(t * NPositions, Intent.Read),
                        Sum2D.GetDevice(Intent.Write),
                        (uint)Sum2D.ElementsReal,
                        (uint)NPositions,
                        1);

                    float[] Sum2DData = Sum2D.GetHostContinuousCopy();
                    Sum2D.Dispose();

                    int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
                    float[] Average2DData = new float[DimsAverage.Elements()];
                    int DimHalf = DimsRegion.X / 2;

                    for (int y = 0; y < DimsAverage.Y; y++)
                    {
                        int yy = y * y;
                        for (int x = 0; x < DimHalf; x++)
                        {
                            int xx = x;
                            xx *= xx;
                            float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                            Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = Sum2DData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x];
                        }

                        for (int x = 1; x < DimHalf; x++)
                        {
                            int xx = -(x - DimHalf);
                            float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                            Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = Sum2DData[y * (DimsRegion.X / 2 + 1) + xx];
                        }
                    }

                    AllPS2D.GetHost(Intent.Write)[t] = Average2DData;

                    #endregion
                }

                #endregion

                AllPS2D.WriteMRC(PowerSpectrumPath, true);

                CTFSpectraBackground.Dispose();
                CTFAverage1D.Dispose();
                CTFSpectra.FreeDevice();
            }
        }

        #endregion

        CTF = GlobalCTF;
        SimulatedScale = GlobalScale;
        PlaneNormal = (Matrix3.RotateX(GlobalPlaneAngle.X * Helper.ToRad) * Matrix3.RotateY(GlobalPlaneAngle.Y * Helper.ToRad)) * new float3(0, 0, 1);

        #region Estimate fittable resolution

        {
            float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(),
                SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()),
                (float)options.RangeMin, 16);
            int FirstFreq = 0;
            while((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
                FirstFreq++;

            int LastFreq = FirstFreq;
            while(!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                LastFreq++;

            CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
        }

        #endregion

        CTFSpectra.Dispose();
        CTFMean.Dispose();
        CTFCoordsCart.Dispose();
        CTFCoordsPolarTrimmed.Dispose();

        Simulated1D = GetSimulated1D();

        OptionsCTF = options;

        SaveMeta();

        IsProcessing = false;
        TiltCTFProcessed?.Invoke();
    }
}