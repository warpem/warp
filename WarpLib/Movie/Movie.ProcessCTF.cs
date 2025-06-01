using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Tools;

namespace Warp;

public partial class Movie
{
    public virtual void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options)
    {
        IsProcessing = true;

        if (!Directory.Exists(PowerSpectrumDir))
            Directory.CreateDirectory(PowerSpectrumDir);

        //CTF = new CTF();
        PS1D = null;
        _SimulatedBackground = null;
        _SimulatedScale = new Cubic1D(new[] { new float2(0, 1), new float2(1, 1) });

        #region Dimensions and grids

        int NFrames = options.UseMovieSum ? 1 : originalStack.Dims.Z;
        int2 DimsImage = new int2(originalStack.Dims);
        int2 DimsRegion = new int2(options.Window);
        int2 DimsRegionLarge = DimsRegion * 2;

        float OverlapFraction = 0.5f;
        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(DimsImage, DimsRegionLarge, OverlapFraction, out DimsPositionGrid);
        int NPositions = (int)DimsPositionGrid.Elements();

        // Auto grid dims
        if (options.GridDims.Elements() == 0)
        {
            float OverallDose = (float)(options.DosePerAngstromFrame < 0 ? -options.DosePerAngstromFrame : options.DosePerAngstromFrame * NFrames);

            int AutoZ = (int)MathF.Ceiling(Math.Max(1, OverallDose));
            int AutoX, AutoY;

            // For a FoV of 4000 Angstrom, use a grid side length of 5
            // Scale up linearly for larger FoV
            // Scale down linearly for lower dose than 30 e/A^2
            // At least 2x2 because we want a gradient even for low-dose tilt movies
            float ShortAngstrom = Math.Min(originalStack.Dims.X, originalStack.Dims.Y) * (float)options.BinnedPixelSizeMean;
            int ShortGrid = (int)Math.Max(2, MathF.Round(5f * Math.Min(1, OverallDose / 30f) * (ShortAngstrom / 4000f)));

            if (originalStack.Dims.X <= originalStack.Dims.Y)
            {
                AutoX = ShortGrid;
                AutoY = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.Y / originalStack.Dims.X);
            }
            else
            {
                AutoY = ShortGrid;
                AutoX = (int)MathF.Round(ShortGrid * (float)originalStack.Dims.X / originalStack.Dims.Y);
            }

            options.GridDims = new int3(AutoX, AutoY, AutoZ);
        }

        int CTFGridX = Math.Min(DimsPositionGrid.X, options.GridDims.X);
        int CTFGridY = Math.Min(DimsPositionGrid.Y, options.GridDims.Y);
        int CTFGridZ = Math.Min(NFrames, options.GridDims.Z);
        GridCTFDefocus = new CubicGrid(new int3(CTFGridX, CTFGridY, CTFGridZ));
        GridCTFPhase = new CubicGrid(new int3(1, 1, CTFGridZ));

        bool CTFSpace = CTFGridX * CTFGridY > 1;
        bool CTFTime = CTFGridZ > 1;
        int3 CTFSpectraGrid = new int3(CTFSpace ? DimsPositionGrid.X : 1,
            CTFSpace ? DimsPositionGrid.Y : 1,
            CTFTime ? CTFGridZ : 1);

        int MinFreqInclusive = (int)(options.RangeMin * DimsRegion.X / 2);
        int MaxFreqExclusive = (int)(options.RangeMax * DimsRegion.X / 2);
        int NFreq = MaxFreqExclusive - MinFreqInclusive;

        if (MaxFreqExclusive > DimsRegion.X / 2)
            throw new Exception("Max frequency to fit is higher than the Nyquist frequency");

        #endregion

        var Timer0 = CTFTimers[0].Start();

        #region Allocate GPU memory

        Image CTFSpectra = new Image(IntPtr.Zero, new int3(DimsRegion.X, DimsRegion.X, (int)CTFSpectraGrid.Elements()), true);
        Image CTFMean = new Image(IntPtr.Zero, new int3(DimsRegion), true);
        Image CTFCoordsCart = new Image(new int3(DimsRegion), true, true);
        Image CTFCoordsPolarTrimmed = new Image(new int3(NFreq, DimsRegion.X, 1), false, true);

        #endregion

        CTFTimers[0].Finish(Timer0);

        // Extract movie regions, create individual spectra in Cartesian coordinates and their mean.

        var Timer1 = CTFTimers[1].Start();

        #region Create spectra

        if (options.UseMovieSum)
        {
            Image StackAverage = null;
            if (!File.Exists(AveragePath))
                StackAverage = originalStack.AsReducedAlongZ();
            else
                StackAverage = Image.FromFile(AveragePath);

            originalStack?.FreeDevice();

            GPU.CreateSpectra(StackAverage.GetDevice(Intent.Read),
                DimsImage,
                1,
                PositionGrid,
                NPositions,
                DimsRegionLarge,
                CTFSpectraGrid,
                DimsRegion,
                CTFSpectra.GetDevice(Intent.Write),
                CTFMean.GetDevice(Intent.Write),
                0,
                0);

            StackAverage.Dispose();
        }
        else
        {
            GPU.CreateSpectra(originalStack.GetDevice(Intent.Read),
                DimsImage,
                NFrames,
                PositionGrid,
                NPositions,
                DimsRegionLarge,
                CTFSpectraGrid,
                DimsRegion,
                CTFSpectra.GetDevice(Intent.Write),
                CTFMean.GetDevice(Intent.Write),
                0,
                0);

            originalStack.FreeDevice(); // Won't need it in this method anymore.
        }

        //CTFSpectra.WriteMRC("d_spectra.mrc", true);

        #endregion

        CTFTimers[1].Finish(Timer1);

        // Populate address arrays for later.

        var Timer2 = CTFTimers[2].Start();

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

        CTFTimers[2].Finish(Timer2);

        // Retrieve average 1D spectrum from CTFMean (not corrected for astigmatism yet).

        var Timer3 = CTFTimers[3].Start();

        #region Initial 1D spectrum

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

            //CTFAverage1D.WriteMRC("d_CTFAverage1D.mrc");

            float[] CTFAverage1DData = CTFAverage1D.GetHost(Intent.Read)[0];
            float2[] ForPS1D = new float2[DimsRegion.X / 2];
            for (int i = 0; i < ForPS1D.Length; i++)
                ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(CTFAverage1DData[i], 4));
            _PS1D = ForPS1D;

            CTFAverage1D.Dispose();
        }

        #endregion

        CTFTimers[3].Finish(Timer3);

        var Timer4 = CTFTimers[4].Start();

        #region Background fitting methods

        Action UpdateBackgroundFit = () =>
        {
            float2[] ForPS1D = PS1D.Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray();
            Cubic1D.FitCTF(ForPS1D,
                CTF.Get1D(PS1D.Length, true, true).Skip(Math.Max(5, MinFreqInclusive / 2)).ToArray(),
                CTF.GetZeros(),
                CTF.GetPeaks(),
                out _SimulatedBackground,
                out _SimulatedScale);
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
                    MeanCorrectedData[i] = MeanData[i] - _SimulatedBackground.Interp(r / DimsRegion.X);
                });

            Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

            GPU.CTFMakeAverage(CTFMeanCorrected.GetDevice(Intent.Read),
                CTFCoordsCart.GetDevice(Intent.Read),
                (uint)CTFMeanCorrected.DimsEffective.ElementsSlice(),
                (uint)DimsRegion.X,
                new[] { CTF.ToStruct() },
                CTF.ToStruct(),
                0,
                (uint)DimsRegion.X / 2,
                1,
                CTFAverage1D.GetDevice(Intent.Write));

            //CTFAverage1D.WriteMRC("CTFAverage1D.mrc");

            float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
            float2[] ForPS1D = new float2[PS1D.Length];
            if (keepbackground)
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i] + _SimulatedBackground.Interp((float)i / DimsRegion.X));
            else
                for (int i = 0; i < ForPS1D.Length; i++)
                    ForPS1D[i] = new float2((float)i / DimsRegion.X, RotationalAverageData[i]);
            MathHelper.UnNaN(ForPS1D);

            _PS1D = ForPS1D;

            CTFMeanCorrected.Dispose();
            CTFAverage1D.Dispose();
        };

        #endregion

        // Fit background to currently best average (not corrected for astigmatism yet).
        {
            float2[] ForPS1D = PS1D.Skip(MinFreqInclusive).Take(Math.Max(2, NFreq)).ToArray();

            int NumNodes = Math.Max(3, (int)((options.RangeMax - options.RangeMin) * 5M));
            _SimulatedBackground = Cubic1D.Fit(ForPS1D, NumNodes); // This won't fit falloff and scale, because approx function is 0

            float[] CurrentBackground = _SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray();
            float[] Subtracted1D = Helper.ArrayOfFunction(i => ForPS1D[i].Y - CurrentBackground[i], ForPS1D.Length);
            MathHelper.NormalizeInPlace(Subtracted1D);

            float ZMin = (float)options.ZMin;
            float ZMax = (float)options.ZMax;
            float ZStep = (ZMax - ZMin) / 200f;

            float BestZ = 0, BestIceOffset = 0, BestPhase = 0, BestScore = -float.MaxValue;

            int NThreads = 8;
            float[] ZValues = Helper.ArrayOfFunction(i => i * 0.01f + ZMin, (int)((ZMax + 1e-5f - ZMin) / 0.01f));
            float[] MTBestZ = new float[NThreads];
            float[] MTBestPhase = new float[NThreads];
            float[] MTBestScore = Helper.ArrayOfConstant(-float.MaxValue, NThreads);

            Helper.ForCPU(0, ZValues.Length, NThreads, null, (i, threadID) =>
            {
                float z = ZValues[i];

                for (float p = 0; p <= (options.DoPhase ? 1f : 0f); p += 0.01f)
                {
                    CTF CurrentParams = new CTF
                    {
                        PixelSize = options.BinnedPixelSizeMean,

                        Defocus = (decimal)z,
                        PhaseShift = (decimal)p,

                        Cs = options.Cs,
                        Voltage = options.Voltage,
                        Amplitude = options.Amplitude
                    };

                    float[] SimulatedCTFFull = CurrentParams.Get1D(PS1D.Length, true);
                    float[] SimulatedCTF = Helper.Subset(SimulatedCTFFull, MinFreqInclusive, MinFreqInclusive + NFreq);

                    MathHelper.NormalizeInPlace(SimulatedCTF);
                    float Score = MathHelper.CrossCorrelate(Subtracted1D, SimulatedCTF);

                    ArrayPool<float>.Return(SimulatedCTFFull);
                    ArrayPool<float>.Return(SimulatedCTF);

                    if (Score > MTBestScore[threadID])
                    {
                        MTBestScore[threadID] = Score;
                        MTBestZ[threadID] = z;
                        MTBestPhase[threadID] = p;
                    }
                }
            }, null);

            for (int i = 0; i < NThreads; i++)
            {
                if (MTBestScore[i] > BestScore)
                {
                    BestScore = MTBestScore[i];
                    BestZ = MTBestZ[i];
                    BestPhase = MTBestPhase[i];
                }
            }

            CTF = new CTF
            {
                PixelSize = options.BinnedPixelSizeMean,

                Defocus = (decimal)BestZ,
                PhaseShift = (decimal)BestPhase,

                Cs = options.Cs,
                Voltage = options.Voltage,
                Amplitude = options.Amplitude
            };

            UpdateRotationalAverage(true); // This doesn't have a nice background yet.
            UpdateBackgroundFit(); // Now get a reasonably nice background.
        }
        CTFTimers[4].Finish(Timer4);

        // Do BFGS optimization of defocus, astigmatism and phase shift,
        // using 2D simulation for comparison

        var Timer5 = CTFTimers[5].Start();

        #region BFGS

        GridCTFDefocus = new CubicGrid(GridCTFDefocus.Dimensions, (float)CTF.Defocus, (float)CTF.Defocus, Dimension.X);
        GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, (float)CTF.PhaseShift, (float)CTF.PhaseShift, Dimension.X);

        {
            NFreq = MaxFreqExclusive - MinFreqInclusive;

            Image CTFSpectraPolarTrimmed = CTFSpectra.AsPolar((uint)MinFreqInclusive, (uint)(MinFreqInclusive + NFreq));
            CTFSpectra.FreeDevice(); // This will only be needed again for the final PS1D.

            #region Create background and scale

            float[] CurrentScale = _SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray());

            Image CTFSpectraScale = new Image(new int3(NFreq, DimsRegion.X, 1));
            float[] CTFSpectraScaleData = CTFSpectraScale.GetHost(Intent.Write)[0];

            // Trim polar to relevant frequencies, and populate coordinates.
            Parallel.For(0, DimsRegion.X, y =>
            {
                float Angle = ((float)y / DimsRegion.X + 0.5f) * (float)Math.PI;
                for (int x = 0; x < NFreq; x++)
                    CTFSpectraScaleData[y * NFreq + x] = CurrentScale[x + MinFreqInclusive];
            });
            //CTFSpectraScale.WriteMRC("ctfspectrascale.mrc");

            // Background is just 1 line since we're in polar.
            Image CurrentBackground = new Image(_SimulatedBackground.Interp(PS1D.Select(p => p.X).ToArray()).Skip(MinFreqInclusive).Take(NFreq).ToArray());

            #endregion

            CTFSpectraPolarTrimmed.SubtractFromLines(CurrentBackground);
            CurrentBackground.Dispose();

            // Normalize background-subtracted spectra.
            GPU.Normalize(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                CTFSpectraPolarTrimmed.GetDevice(Intent.Write),
                (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                (uint)CTFSpectraGrid.Elements());
            //CTFSpectraPolarTrimmed.WriteMRC("ctfspectrapolartrimmed.mrc");

            // Wiggle weights show how the defocus on the spectra grid is altered 
            // by changes in individual anchor points of the spline grid.
            // They are used later to compute the dScore/dDefocus values for each spectrum 
            // only once, and derive the values for each anchor point from them.
            float[][] WiggleWeights = GridCTFDefocus.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));
            float[][] WiggleWeightsPhase = GridCTFPhase.GetWiggleWeights(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, 1f / (CTFGridZ + 1)));

            // Helper method for getting CTFStructs for the entire spectra grid.
            Func<double[], CTF, float[], float[], CTFStruct[]> EvalGetCTF = (input, ctf, defocusValues, phaseValues) =>
            {
                decimal AlteredDelta = (decimal)input[input.Length - 2];
                decimal AlteredAngle = (decimal)(input[input.Length - 1] * 20 * Helper.ToDeg);

                CTF Local = ctf.GetCopy();
                Local.DefocusDelta = AlteredDelta;
                Local.DefocusAngle = AlteredAngle;

                CTFStruct LocalStruct = Local.ToStruct();
                CTFStruct[] LocalParams = ArrayPool<CTFStruct>.Rent(defocusValues.Length);
                for (int i = 0; i < LocalParams.Length; i++)
                {
                    LocalParams[i] = LocalStruct;
                    LocalParams[i].Defocus = defocusValues[i] * -1e-6f;
                    LocalParams[i].PhaseShift = phaseValues[i] * (float)Math.PI;
                }

                return LocalParams;
            };

            // Simulate with adjusted CTF, compare to originals

            #region Eval and Gradient methods

            float BorderZ = 0.5f / CTFGridZ;

            Func<double[], double> Eval = input =>
            {
                using CubicGrid Altered = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
                float[] DefocusValues = Altered.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                using CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                float IceOffset = (float)input[input.Length - 6] * (-1e4f);

                float[] Result = new float[LocalParams.Length];

                GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                    CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                    CTFSpectraScale.GetDevice(Intent.Read),
                    (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                    LocalParams,
                    Result,
                    (uint)LocalParams.Length);

                float Score = Result.Sum();

                ArrayPool<CTFStruct>.Return(LocalParams);

                if (float.IsNaN(Score) || float.IsInfinity(Score))
                    throw new Exception("Bad score.");

                return Score;
            };

            Func<double[], double[]> Gradient = input =>
            {
                const float Step = 0.005f;
                double[] Result = new double[input.Length];

                // In 0D grid case, just get gradient for all 4 parameters.
                // In 1+D grid case, do simple gradient for ice ring, astigmatism, phase, ...
                int StartComponent = input.Length - 2;
                //int StartComponent = 0;
                for (int i = StartComponent; i < input.Length; i++)
                {
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

                float IceIntensity = 1 / (1 + (float)Math.Exp(-input[input.Length - 3] * 10));
                float2 IceStd = new float2((float)Math.Exp(input[input.Length - 5] * 10), (float)Math.Exp(input[input.Length - 4] * 10));
                float IceOffset = (float)input[input.Length - 6] * (-1e4f);

                float[] ResultPlus = new float[CTFSpectraGrid.Elements()];
                float[] ResultMinus = new float[CTFSpectraGrid.Elements()];

                // ... take shortcut for defoci, ...
                {
                    using CubicGrid AlteredPhase = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] PhaseValues = AlteredPhase.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                    {
                        using CubicGrid AlteredPlus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                        float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                            CTFSpectraScale.GetDevice(Intent.Read),
                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                            LocalParams,
                            ResultPlus,
                            (uint)LocalParams.Length);

                        ArrayPool<CTFStruct>.Return(LocalParams);
                    }
                    {
                        using CubicGrid AlteredMinus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                        float[] DefocusValues = AlteredMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                        CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                            CTFSpectraScale.GetDevice(Intent.Read),
                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                            LocalParams,
                            ResultMinus,
                            (uint)LocalParams.Length);

                        ArrayPool<CTFStruct>.Return(LocalParams);
                    }
                    float[] LocalGradients = new float[ResultPlus.Length];
                    for (int i = 0; i < LocalGradients.Length; i++)
                        LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                    // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                    Parallel.For(0, GridCTFDefocus.Dimensions.Elements(), i => Result[i] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeights[i]) / (2f * Step));
                }

                // ... and take shortcut for phases.
                if (options.DoPhase)
                {
                    using CubicGrid AlteredPlus = new CubicGrid(GridCTFDefocus.Dimensions, input.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
                    float[] DefocusValues = AlteredPlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));

                    {
                        using CubicGrid AlteredPhasePlus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v + Step).ToArray());
                        float[] PhaseValues = AlteredPhasePlus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                        CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                            CTFSpectraScale.GetDevice(Intent.Read),
                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                            LocalParams,
                            ResultPlus,
                            (uint)LocalParams.Length);

                        ArrayPool<CTFStruct>.Return(LocalParams);
                    }
                    {
                        using CubicGrid AlteredPhaseMinus = new CubicGrid(GridCTFPhase.Dimensions, input.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v - Step).ToArray());
                        float[] PhaseValues = AlteredPhaseMinus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                        CTFStruct[] LocalParams = EvalGetCTF(input, CTF, DefocusValues, PhaseValues);

                        GPU.CTFCompareToSim(CTFSpectraPolarTrimmed.GetDevice(Intent.Read),
                            CTFCoordsPolarTrimmed.GetDevice(Intent.Read),
                            CTFSpectraScale.GetDevice(Intent.Read),
                            (uint)CTFSpectraPolarTrimmed.ElementsSliceReal,
                            LocalParams,
                            ResultMinus,
                            (uint)LocalParams.Length);

                        ArrayPool<CTFStruct>.Return(LocalParams);
                    }
                    float[] LocalGradients = new float[ResultPlus.Length];
                    for (int i = 0; i < LocalGradients.Length; i++)
                        LocalGradients[i] = ResultPlus[i] - ResultMinus[i];

                    // Now compute gradients per grid anchor point using the precomputed individual gradients and wiggle factors.
                    Parallel.For(0, GridCTFPhase.Dimensions.Elements(), i => Result[i + GridCTFDefocus.Dimensions.Elements()] = MathHelper.ReduceWeighted(LocalGradients, WiggleWeightsPhase[i]) / (2f * Step));
                }

                foreach (var i in Result)
                    if (double.IsNaN(i) || double.IsInfinity(i))
                        throw new Exception("Bad score.");

                return Result;
            };

            #endregion

            #region Optimize

            double[] StartParams = new double[GridCTFDefocus.Dimensions.Elements() + GridCTFPhase.Dimensions.Elements() + 6];
            for (int i = 0; i < GridCTFDefocus.Dimensions.Elements(); i++)
                StartParams[i] = GridCTFDefocus.FlatValues[i];
            for (int i = 0; i < GridCTFPhase.Dimensions.Elements(); i++)
                StartParams[i + GridCTFDefocus.Dimensions.Elements()] = GridCTFPhase.FlatValues[i];

            StartParams[StartParams.Length - 2] = (double)CTF.DefocusDelta;
            StartParams[StartParams.Length - 1] = (double)CTF.DefocusAngle / 20 * Helper.ToRad;

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, Eval, Gradient)
            {
                MaxIterations = 15
            };
            Optimizer.Maximize(StartParams);

            #endregion

            #region Retrieve parameters

            CTF.Defocus = (decimal)MathHelper.Mean(Optimizer.Solution.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v));
            CTF.DefocusDelta = (decimal)Optimizer.Solution[StartParams.Length - 2];
            CTF.DefocusAngle = (decimal)(Optimizer.Solution[StartParams.Length - 1] * 20 * Helper.ToDeg);
            CTF.PhaseShift = (decimal)MathHelper.Mean(Optimizer.Solution.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v));

            if (CTF.DefocusDelta < 0)
            {
                CTF.DefocusAngle += 90;
                CTF.DefocusDelta *= -1;
            }

            CTF.DefocusAngle = ((int)CTF.DefocusAngle + 180 * 99) % 180;

            GridCTFDefocus = new CubicGrid(GridCTFDefocus.Dimensions, Optimizer.Solution.Take((int)GridCTFDefocus.Dimensions.Elements()).Select(v => (float)v).ToArray());
            GridCTFPhase = new CubicGrid(GridCTFPhase.Dimensions, Optimizer.Solution.Skip((int)GridCTFDefocus.Dimensions.Elements()).Take((int)GridCTFPhase.Dimensions.Elements()).Select(v => (float)v).ToArray());

            #endregion

            // Dispose GPU resources manually because GC can't be bothered to do it in time.
            CTFSpectraPolarTrimmed.Dispose();
            CTFSpectraScale.Dispose();

            #region Get nicer envelope fit

            {
                if (!CTFSpace && !CTFTime)
                {
                    UpdateRotationalAverage(true);
                }
                else
                {
                    Image CTFSpectraBackground = new Image(new int3(DimsRegion), true);
                    float[] CTFSpectraBackgroundData = CTFSpectraBackground.GetHost(Intent.Write)[0];

                    // Construct background in Cartesian coordinates.
                    Helper.ForEachElementFT(DimsRegion, (x, y, xx, yy, r, a) => { CTFSpectraBackgroundData[y * CTFSpectraBackground.DimsEffective.X + x] = _SimulatedBackground.Interp(r / DimsRegion.X); });

                    CTFSpectra.SubtractFromSlices(CTFSpectraBackground);

                    float[] DefocusValues = GridCTFDefocus.GetInterpolatedNative(CTFSpectraGrid, new float3(DimsRegion.X / 2f / DimsImage.X, DimsRegion.Y / 2f / DimsImage.Y, BorderZ));
                    CTFStruct[] LocalParams = DefocusValues.Select(v =>
                    {
                        CTF Local = CTF.GetCopy();
                        Local.Defocus = (decimal)v + 0.0M;

                        return Local.ToStruct();
                    }).ToArray();

                    Image CTFAverage1D = new Image(IntPtr.Zero, new int3(DimsRegion.X / 2, 1, 1));

                    CTF CTFAug = CTF.GetCopy();
                    CTFAug.Defocus += 0.0M;
                    GPU.CTFMakeAverage(CTFSpectra.GetDevice(Intent.Read),
                        CTFCoordsCart.GetDevice(Intent.Read),
                        (uint)CTFSpectra.ElementsSliceReal,
                        (uint)DimsRegion.X,
                        LocalParams,
                        CTFAug.ToStruct(),
                        0,
                        (uint)DimsRegion.X / 2,
                        (uint)CTFSpectraGrid.Elements(),
                        CTFAverage1D.GetDevice(Intent.Write));

                    CTFSpectra.AddToSlices(CTFSpectraBackground);

                    float[] RotationalAverageData = CTFAverage1D.GetHost(Intent.Read)[0];
                    float2[] ForPS1D = new float2[PS1D.Length];
                    for (int i = 0; i < ForPS1D.Length; i++)
                        ForPS1D[i] = new float2((float)i / DimsRegion.X, (float)Math.Round(RotationalAverageData[i], 4) + _SimulatedBackground.Interp((float)i / DimsRegion.X));
                    MathHelper.UnNaN(ForPS1D);
                    _PS1D = ForPS1D;

                    CTFSpectraBackground.Dispose();
                    CTFAverage1D.Dispose();
                    CTFSpectra.FreeDevice();
                }

                //CTF.Defocus = Math.Max(CTF.Defocus, 0);
                UpdateBackgroundFit();
            }

            #endregion
        }

        #endregion

        CTFTimers[5].Finish(Timer5);

        // Subtract background from 2D average and write it to disk. 
        // This image is used for quick visualization purposes only.

        var Timer6 = CTFTimers[6].Start();

        #region PS2D update

        {
            int3 DimsAverage = new int3(DimsRegion.X, DimsRegion.X / 2, 1);
            float[] Average2DData = new float[DimsAverage.Elements()];
            float[] OriginalAverageData = CTFMean.GetHost(Intent.Read)[0];
            int DimHalf = DimsRegion.X / 2;

            for (int y = 0; y < DimsAverage.Y; y++)
            {
                int yy = y * y;
                for (int x = 0; x < DimHalf; x++)
                {
                    int xx = x;
                    xx *= xx;
                    float r = (float)Math.Sqrt(xx + yy) / DimsRegion.X;
                    Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x + DimHalf] = OriginalAverageData[(DimsRegion.X - 1 - y) * (DimsRegion.X / 2 + 1) + x] - SimulatedBackground.Interp(r);
                }

                for (int x = 1; x < DimHalf; x++)
                {
                    int xx = -(x - DimHalf);
                    float r = (float)Math.Sqrt(xx * xx + yy) / DimsRegion.X;
                    Average2DData[(DimsAverage.Y - 1 - y) * DimsAverage.X + x] = OriginalAverageData[y * (DimsRegion.X / 2 + 1) + xx] - SimulatedBackground.Interp(r);
                }
            }

            IOHelper.WriteMapFloat(PowerSpectrumPath,
                new HeaderMRC
                {
                    Dimensions = DimsAverage,
                    MinValue = MathHelper.Min(Average2DData),
                    MaxValue = MathHelper.Max(Average2DData)
                },
                Average2DData);
        }

        #endregion

        CTFTimers[6].Finish(Timer6);

        var Timer7 = CTFTimers[7].Start();
        for (int i = 0; i < PS1D.Length; i++)
            PS1D[i].Y -= SimulatedBackground.Interp(PS1D[i].X);
        SimulatedBackground = new Cubic1D(SimulatedBackground.Data.Select(v => new float2(v.X, 0f)).ToArray());

        CTFSpectra.Dispose();
        CTFMean.Dispose();
        CTFCoordsCart.Dispose();
        CTFCoordsPolarTrimmed.Dispose();

        Simulated1D = GetSimulated1D();
        //CTFQuality = GetCTFQuality();

        #region Estimate fittable resolution

        {
            float[] Quality = CTF.EstimateQuality(PS1D.Select(p => p.Y).ToArray(), SimulatedScale.Interp(PS1D.Select(p => p.X).ToArray()), (float)options.RangeMin, 16);
            int FirstFreq = MinFreqInclusive + NFreq / 2;
            //while ((float.IsNaN(Quality[FirstFreq]) || Quality[FirstFreq] < 0.8f) && FirstFreq < Quality.Length - 1)
            //    FirstFreq++;

            int LastFreq = FirstFreq;
            while(!float.IsNaN(Quality[LastFreq]) && Quality[LastFreq] > 0.3f && LastFreq < Quality.Length - 1)
                LastFreq++;

            CTFResolutionEstimate = Math.Round(options.BinnedPixelSizeMean / ((decimal)LastFreq / options.Window), 1);
        }

        #endregion

        OptionsCTF = options;

        SaveMeta();

        IsProcessing = false;
        CTFTimers[7].Finish(Timer7);

        //lock (CTFTimers)
        //{
        //    if (CTFTimers[0].NItems > 5)
        //        using (TextWriter Writer = File.CreateText("d_ctftimers.txt"))
        //            foreach (var timer in CTFTimers)
        //            {
        //                Debug.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
        //                Writer.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
        //            }
        //}
    }
}

[Serializable]
public class ProcessingOptionsMovieCTF : ProcessingOptionsBase
{
    [WarpSerializable] public int Window { get; set; }
    [WarpSerializable] public decimal RangeMin { get; set; }
    [WarpSerializable] public decimal RangeMax { get; set; }
    [WarpSerializable] public int Voltage { get; set; }
    [WarpSerializable] public decimal Cs { get; set; }
    [WarpSerializable] public decimal Cc { get; set; }
    [WarpSerializable] public decimal Amplitude { get; set; }
    [WarpSerializable] public bool DoPhase { get; set; }
    [WarpSerializable] public bool UseMovieSum { get; set; }
    [WarpSerializable] public decimal ZMin { get; set; }
    [WarpSerializable] public decimal ZMax { get; set; }
    [WarpSerializable] public int3 GridDims { get; set; }
    [WarpSerializable] public decimal DosePerAngstromFrame { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsMovieCTF)obj);
    }

    protected bool Equals(ProcessingOptionsMovieCTF other)
    {
        return base.Equals(other) &&
               Window == other.Window &&
               RangeMin == other.RangeMin &&
               RangeMax == other.RangeMax &&
               Voltage == other.Voltage &&
               Cs == other.Cs &&
               Cc == other.Cc &&
               Amplitude == other.Amplitude &&
               DoPhase == other.DoPhase &&
               UseMovieSum == other.UseMovieSum &&
               ZMin == other.ZMin &&
               ZMax == other.ZMax &&
               GridDims == other.GridDims &&
               DosePerAngstromFrame == other.DosePerAngstromFrame;
    }

    public static bool operator ==(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
    {
        return !Equals(left, right);
    }
}