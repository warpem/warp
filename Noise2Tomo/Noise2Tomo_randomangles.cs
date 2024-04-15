//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Globalization;
//using System.IO;
//using System.Linq;
//using System.Text;
//using System.Threading;
//using System.Threading.Tasks;
//using CommandLine;
//using CommandLine.Text;
//using Warp;
//using Warp.Tools;

//namespace Noise2Tomo
//{
//    class Program
//    {
//        static void Main(string[] args)
//        {
//            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
//            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

//            #region Command line options

//            Options Options = new Options();
//            string WorkingDirectory;

//            string ProgramFolder = System.Reflection.Assembly.GetEntryAssembly().Location;
//            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

//            {
//                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
//                WorkingDirectory = Environment.CurrentDirectory + "/";
//            }

//            bool DebugFromScratch = false;

//            int NThreads = 1;

//            int3 DimsTrainingSource = new int3(128, 128, 128);
//            int3 DimsTrainingTarget = new int3(64, 64, 64);
//            int3 DimsTrainingCTF = DimsTrainingSource * 2;
//            int DimCTF = 512;

//            float AngularRange = 1.5f;
//            float LowpassFraction = Options.LowpassStart;

//            Random Rand = new Random(123);

//            Queue<BatchData3D> BatchDataPreprocessed = new Queue<BatchData3D>();

//            #endregion

//            List<int> GPUsNetwork = Options.GPUNetwork.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s)).ToList();

//            GPU.SetDevice(Options.GPUPreprocess);

//            #region Load and prepare data

//            Console.WriteLine("Preparing data:");

//            List<Matrix3> RotationsRedundant = new List<Matrix3>();
//            RotationsRedundant.Add(Matrix3.RotateY(0));
//            RotationsRedundant.Add(Matrix3.RotateY(180 * Helper.ToRad));
//            RotationsRedundant.Add(Matrix3.RotateZ(180 * Helper.ToRad));
//            RotationsRedundant.Add(Matrix3.RotateY(180 * Helper.ToRad) * Matrix3.RotateZ(180 * Helper.ToRad));
//            for (int z = 0; z < 4; z++)
//            {
//                for (int y = 0; y < 4; y++)
//                {
//                    for (int x = 0; x < 4; x++)
//                    {
//                        Matrix3 R = Matrix3.RotateZ(90 * z * Helper.ToRad) * 
//                                    Matrix3.RotateY(90 * y * Helper.ToRad) * 
//                                    Matrix3.RotateX(90 * x * Helper.ToRad);
//                        float[] RA = R.ToArray();

//                        bool DuplicateFound = false;
//                        foreach (var R2 in RotationsRedundant)
//                        {
//                            bool IsDuplicate = true;
//                            float[] R2A = R2.ToArray();
//                            for (int i = 0; i < RA.Length; i++)
//                                if (Math.Abs(RA[i] - R2A[i]) > 1e-4f)
//                                {
//                                    IsDuplicate = false;
//                                    break;
//                                }
//                            if (IsDuplicate)
//                            {
//                                DuplicateFound = true;
//                                break;
//                            }
//                        }

//                        if (!DuplicateFound)
//                            RotationsRedundant.Add(R);
//                    }
//                }
//            }
//            RotationsRedundant = RotationsRedundant.Skip(4).ToList();

//            Matrix3[] Rotations = new Matrix3[]
//            {
//                Matrix3.RotateZ(90 * Helper.ToRad),
//                Matrix3.RotateZ(-90 * Helper.ToRad),
//                Matrix3.RotateY(90 * Helper.ToRad),
//                Matrix3.RotateY(-90 * Helper.ToRad),
//                Matrix3.RotateY(90 * Helper.ToRad) * Matrix3.RotateZ(90 * Helper.ToRad),
//                Matrix3.RotateY(-90 * Helper.ToRad) * Matrix3.RotateZ(90 * Helper.ToRad),
//                Matrix3.RotateY(90 * Helper.ToRad) * Matrix3.RotateZ(-90 * Helper.ToRad),
//                Matrix3.RotateY(-90 * Helper.ToRad) * Matrix3.RotateZ(-90 * Helper.ToRad),
//            };
//            Rotations = RotationsRedundant.ToArray();
//            int NRotations = Rotations.Length;

//            List<TiltSeries> LoadedSeries = new List<TiltSeries>();
//            List<Image> OriTomos = new List<Image>();
//            List<Image> PrefilteredTomos = new List<Image>();
//            // PSFs are centered for easier rotation, and need to be decentered later for conversion to CTFs
//            List<Image> OriPSFs = new List<Image>();
//            List<Image> PrefilteredPSFs = new List<Image>();
//            List<Image> OriCTFs = new List<Image>();
//            List<Image> ImprovedTomos = new List<Image>();
//            List<Image> GroundTruthTomos = new List<Image>();

//            List<float3[]> MaskPositions = new List<float3[]>();
//            List<float3> MaskCellSizes = new List<float3>();

//            Projector AFProjector;
//            {
//                Image AF = Image.FromFile(@"D:\workshop\emd_11603.mrc");
//                AF.MaskSpherically(AF.Dims.X - 16, 16, true);
//                AF = AF.AsScaled(new int3(new float3(AF.Dims) * AF.PixelSize / Options.PixelSize / 2 + 1) * 2).AndDisposeParent();
//                AF = AF.AsPadded(AF.Dims * 2).AndDisposeParent();
//                AF.WriteMRC("d_af.mrc", true);

//                AFProjector = new Projector(AF, 2);
//                AF.FreeDevice();
//                AF.Dispose();
//            }

//            foreach (var file in Directory.EnumerateFiles(WorkingDirectory, "*.tomostar"))
//            {
//                TiltSeries Series = new TiltSeries(file);

//                Console.Write($"Preparing {Series.Name}...");

//                string DeconvPath = Path.Combine(Series.ReconstructionDir, "denoised", Series.RootName + $"_{Options.PixelSize:F2}Apx.mrc");
//                if (!File.Exists(DeconvPath))
//                {
//                    Console.WriteLine(" No data found, skipping.");
//                    continue;
//                }

//                LoadedSeries.Add(Series);

//                Image OriTomo = Image.FromFile(DeconvPath);

//                {
//                    Image TomoCenter = OriTomo.AsPadded(OriTomo.Dims / 2);
//                    float2 MeanStd = MathHelper.MeanAndStd(TomoCenter.GetHostContinuousCopy());
//                    TomoCenter.Dispose();

//                    OriTomo.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
//                }

//                OriTomo = OriTomo.AsPadded(int3.Max(DimsTrainingSource, OriTomo.Dims)).AndDisposeParent();
//                OriTomo.FreeDevice();
//                //Image ImprovedTomo = OriTomo.GetCopy();

//                //OriTomos.Add(OriTomo);
//                //ImprovedTomos.Add(ImprovedTomo);

//                #region CTF

//                Series.VolumeDimensionsPhysical = new float3(OriTomo.Dims) * Options.PixelSize;
//                float3[] Angles = Series.GetAngleInAllTilts(Series.VolumeDimensionsPhysical / 2);
//                //Angles = Helper.ArrayOfFunction(i => new float3(0, -60 + i * 2, 0) * Helper.ToRad, 61);
//                //Angles = Angles.Select(v => new float3(0, 0, 0)).ToArray();
//                float[] Occupancy = Series.UseTilt.Select(b => b ? 1f : 0f).ToArray();

//                Image TiltPoints = new Image(new int3(DimCTF, DimCTF, Angles.Length));
//                for (int t = 0; t < Angles.Length; t++)
//                    TiltPoints.GetHost(Intent.ReadWrite)[t][0] = Occupancy[t];

//                Image TiltPointsFT = TiltPoints.AsFFT().AndDisposeParent();
//                Image TiltWeights = TiltPointsFT.AsReal();
//                //TiltWeights.WriteMRC("d_tiltweights.mrc", true);

//                Image ProjData = new Image(new int3(DimCTF), true, true);
//                Image ProjWeights = new Image(new int3(DimCTF), true);
//                //for (int a = 0; a < 24; a++)
//                {
//                    GPU.ProjectBackward(ProjData.GetDevice(Intent.ReadWrite),
//                                        ProjWeights.GetDevice(Intent.ReadWrite),
//                                        new int3(DimCTF),
//                                        TiltPointsFT.GetDevice(Intent.Read),
//                                        TiltWeights.GetDevice(Intent.Read),
//                                        TiltPointsFT.DimsSlice,
//                                        TiltPointsFT.Dims.X / 2,
//                                        Helper.ToInterleaved(Angles),//.Select(v => Matrix3.EulerFromMatrix(Matrix3.Euler(v) * Matrix3.RotateY((a - 12) / 12f * Helper.ToRad))).ToArray()),
//                                        null,
//                                        Matrix2.Identity().ToVec(),
//                                        0,
//                                        1,
//                                        true,
//                                        false,
//                                        (uint)Angles.Length);
//                }
//                ProjWeights.Max(1);
//                ProjData.Divide(ProjWeights);
//                ProjWeights.Dispose();

//                {
//                    Image ProjDataReal = ProjData.AsReal();
//                    ProjDataReal.WriteMRC("d_datareal.mrc", true);
//                    ProjDataReal.Dispose();
//                }

//                Image PSF = ProjData.AsIFFT(true);
//                ProjData.Dispose();
//                PSF.Multiply(1f / (PSF.ElementsReal));
//                PSF.RemapFromFT(true);
//                PSF.MaskSpherically(DimCTF - 20, 10, true);
//                //PSF = PSF.AsPadded(OriTomo.Dims).AndDisposeParent();
//                //PSF.WriteMRC("d_psf.mrc", true);
//                TiltPointsFT.Dispose();
//                TiltWeights.Dispose();

//                Image PSFPrefiltered = PSF.GetCopy();
//                GPU.PrefilterForCubic(PSFPrefiltered.GetDevice(Intent.ReadWrite), PSFPrefiltered.Dims);
//                PSFPrefiltered.FreeDevice();

//                {
//                    Image CTF = PSF.AsFFT(true).AsAmplitudes().AndDisposeParent();
//                    CTF.Multiply(1f / CTF.ElementsReal);
//                    CTF.FreeDevice();
//                    OriCTFs.Add(CTF);
//                    CTF.WriteMRC("d_ctf.mrc", true);
//                }

//                PSF.FreeDevice();
//                OriPSFs.Add(PSF);
//                PrefilteredPSFs.Add(PSFPrefiltered);

//                if (false)
//                {
//                    int DimPoint = 128;
//                    int Super = 8;
//                    Image Points = new Image(new int3(DimPoint, DimPoint, Angles.Length));
//                    for (int a = 0; a < Points.Dims.Z; a++)
//                        Points.GetHost(Intent.ReadWrite)[a][0] = 1f;
//                    Points.RemapFromFT();
//                    Points.Bandpass(0, 0.98f, false, 0.02f);
//                    Points = Points.AsScaled(new int2(DimPoint * Super)).AndDisposeParent();

//                    Image PSFBP = new Image(new int3(DimCTF));
//                    GPU.RealspaceProjectBackward(PSFBP.GetDevice(Intent.ReadWrite),
//                                                 PSFBP.Dims,
//                                                 Points.GetDevice(Intent.Read),
//                                                 new int2(DimPoint * Super),
//                                                 Super,
//                                                 Helper.ToInterleaved(Angles),
//                                                 false,
//                                                 Points.Dims.Z);
//                    PSFBP.Multiply(1f / DimCTF);
//                    PSFBP.MaskSpherically(DimCTF - 32, 16, true);
//                    Image CTFBP = PSFBP.AsFFT(true).AsAmplitudes();
//                    CTFBP.Min(1);
//                    CTFBP.WriteMRC("d_ctfbp.mrc");

//                    Image CTF = PSF.AsFFT(true);
//                    CTF.AsAmplitudes().WriteMRC("d_ctf.mrc", true);
//                    CTF.Dispose();
//                }

//                if (false)
//                {
//                    GPU.PrefilterForCubic(PSF.GetDevice(Intent.ReadWrite), PSF.Dims);
//                    ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
//                    GPU.CreateTexture3D(PSF.GetDevice(Intent.Read), PSF.Dims, Texture, TextureArray, true);

//                    Image PSFRotated = new Image(PSF.Dims);

//                    GPU.Rotate3DExtractAt(Texture[0],
//                                                  PSF.Dims,
//                                                  PSFRotated.GetDevice(Intent.Write),
//                                                  PSFRotated.Dims,
//                                                  Helper.ToInterleaved(new float3[] { new float3(0, 1.5f, 0) * Helper.ToRad}),
//                                                  Helper.ToInterleaved(new float3[] { new float3(PSF.Dims / 2)}),
//                                                  (uint)1);

//                    PSFRotated.WriteMRC("d_psfrotated.mrc", true);
//                    PSFRotated.AsFFT(true).AsAmplitudes().WriteMRC("d_ctfrotated.mrc", true);
//                }

//                //Image CTFNeg = new Image(CTF.Dims, true, true);
//                //CTFNeg.Fill(new float2(1, 0));
//                //CTFNeg.Subtract(CTF);

//                //Image PSFNeg = CTFNeg.AsIFFT(true).AndDisposeParent();
//                //PSFNeg.Bandpass(0, 0.99f, true, 0.01f);
//                //CTFNeg = PSFNeg.AsFFT(true).AndDisposeParent();
//                //CTFNeg.Multiply(1f / (PSF.ElementsReal));
//                //CTFNeg.AsImaginary().WriteMRC("d_ctfneg.mrc", true);

//                #endregion

//                #region Fake AF tomo

//                if (false)
//                {
//                    Image GroundtruthTomo;

//                    if (DebugFromScratch)
//                    {
//                        OriTomo.Fill(0);
//                        int DimAF = AFProjector.Dims.X / 2 + 4 + 4;
//                        int2 GridStep = new int2(DimAF, DimAF * 2 / 3);
//                        {
//                            Rand = new Random(123 + OriTomos.Count);

//                            for (int gy = 0; gy < OriTomo.Dims.Y / GridStep.Y; gy++)
//                            {
//                                for (int gx = 0; gx < OriTomo.Dims.X / GridStep.X; gx++)
//                                {
//                                    int3 Pos = new int3(gx * GridStep.X + (gy % 2 == 0 ? GridStep.X / 2 : 0) + Rand.Next(4) - 2,
//                                                        gy * GridStep.Y + Rand.Next(4) - 2,
//                                                        OriTomo.Dims.Z / 2 - DimAF / 2 + Rand.Next(10) - 5);

//                                    Image AFFT = AFProjector.Project(AFProjector.Dims, new[] { new float3((float)Rand.NextDouble() * 360 * Helper.ToRad,
//                                                                                                  (float)Rand.NextDouble() * 360 * Helper.ToRad,
//                                                                                                  (float)Rand.NextDouble() * 360 * Helper.ToRad)});
//                                    Image AF = AFFT.AsIFFT(true, 0, true).AndDisposeParent();
//                                    AF.RemapFromFT(true);
//                                    AF = AF.AsPadded(new int3(DimAF)).AndDisposeParent();
//                                    //AF.WriteMRC("d_projaf.mrc", true);

//                                    OriTomo.TransformRegionValues(AF.Dims, Pos + DimAF / 2, (coord, coordCentered, v) =>
//                                    {
//                                        int3 PosTomo = coordCentered + DimAF / 2;
//                                        return v + AF.GetHost(Intent.Read)[PosTomo.Z][PosTomo.Y * DimAF + PosTomo.X];
//                                    });

//                                    AF.Dispose();
//                                }
//                            }
//                        }
//                        GroundtruthTomo = OriTomo.GetCopy();
//                        GroundtruthTomo.WriteMRC($"d_faketomo_nowedge{OriTomos.Count:D2}.mrc", true);

//                        Image PSFRemapped = PSF.GetCopyGPU().AndFreeParent();
//                        PSFRemapped.RemapToFT(true);
//                        Image CTF = PSFRemapped.AsFFT(true).AndDisposeParent().AsReal().AndDisposeParent();

//                        int3 DimsTomoConv = OriTomo.Dims;
//                        DimsTomoConv.Z *= 2;
//                        Image CTFTomoShaped = RescaleCTF(CTF, DimsTomoConv);
//                        //CTFTomoShaped.WriteMRC("d_ctftomoshaped.mrc", true);
//                        CTFTomoShaped.FreeDevice();

//                        Image TomoPadFT = OriTomo.AsPadded(DimsTomoConv).AndDisposeParent().AsFFT(true).AndDisposeParent();
//                        TomoPadFT.Multiply(CTFTomoShaped);
//                        OriTomo = TomoPadFT.AsIFFT(true).AndDisposeParent().AsPadded(OriTomo.Dims).AndDisposeParent();

//                        OriTomo.WriteMRC($"d_faketomo{OriTomos.Count:D2}.mrc", true);
//                    }
//                    else
//                    {
//                        OriTomo = Image.FromFile($"d_faketomo{OriTomos.Count:D2}.mrc");
//                        GroundtruthTomo = Image.FromFile($"d_faketomo_nowedge{OriTomos.Count:D2}.mrc");
//                    }


//                    {
//                        Image TomoCenter = OriTomo.AsPadded(OriTomo.Dims / 2);
//                        float2 MeanStd = MathHelper.MeanAndStd(TomoCenter.GetHostContinuousCopy());
//                        TomoCenter.Dispose();

//                        OriTomo.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
//                    }

//                    GroundTruthTomos.Add(GroundtruthTomo);
//                }

//                #endregion

//                OriTomo.FreeDevice();
//                OriTomos.Add(OriTomo);
//                ImprovedTomos.Add(OriTomo.GetCopy());

//                Image PrefilteredTomo = OriTomo.GetCopy();
//                GPU.PrefilterForCubic(PrefilteredTomo.GetDevice(Intent.ReadWrite), PrefilteredTomo.Dims);
//                PrefilteredTomo.FreeDevice();
//                PrefilteredTomos.Add(PrefilteredTomo);

//                #region Mask

//                float ScaleFactor = Options.PixelSize / Options.PixelSizeMask;
//                int3 MaskDims = new int3((new float3(OriTomo.Dims) * ScaleFactor / 2).Round()) * 2;
//                float3 ScaleBack = new float3(OriTomo.Dims) / new float3(MaskDims);

//                Image MaskTomo = OriTomo.AsScaled(MaskDims * 2).AndFreeParent();
//                Image MaskTomoConv = MaskTomo.AsConvolvedRaisedCosine(5, 1);
//                //MaskTomoConv.WriteMRC("d_masktomoconv.mrc", true);

//                MaskTomo.Subtract(MaskTomoConv);
//                MaskTomo.Multiply(MaskTomo);
//                //MaskTomo.WriteMRC("d_maskdiffsquared.mrc", true);
//                Image MaskTomoVar = MaskTomo.AsConvolvedRaisedCosine(5, 1).AndDisposeParent();
//                MaskTomoConv.Dispose();

//                MaskTomo = MaskTomoVar.AsScaled(MaskDims).AndDisposeParent();

//                int3 SafeBorder = new int3((new float3(DimsTrainingTarget / 2) * ScaleFactor).Ceil());
//                int3 DimsSafe = MaskDims - SafeBorder * 2 - 2;
//                if (DimsSafe.Z <= 0)
//                    throw new Exception("Tomogram is too small in Z to accomodate the training volume dimensions. Please reconstruct bigger tomograms.");

//                MaskTomo = MaskTomo.AsPadded(MaskDims - SafeBorder * 2 - 2).AndDisposeParent();
//                MaskTomo.WriteMRC("d_masktomovar.mrc", true);

//                List<(int, float)> ScoredPositions = new List<(int, float)>();

//                MaskTomo.TransformValues((x, y, z, v) =>
//                {
//                    ScoredPositions.Add((((z + SafeBorder.Z + 1) * MaskDims.Y + (y + SafeBorder.Y + 1)) * MaskDims.X + (x + SafeBorder.X + 1), v));

//                    return v;
//                });
//                MaskTomo.Dispose();

//                ScoredPositions.Sort((a, b) => -a.Item2.CompareTo(b.Item2));
//                ScoredPositions = ScoredPositions.Take((int)(Options.MaskPercentage / 100f * ScoredPositions.Count)).ToList();

//                float3[] Positions = ScoredPositions.Select(v =>
//                {
//                    int p = v.Item1;

//                    int Z = p / (MaskDims.X * MaskDims.Y);
//                    p -= Z * (MaskDims.X * MaskDims.Y);
//                    int Y = p / MaskDims.X;
//                    int X = p - Y * MaskDims.X;

//                    return new float3(X, Y, Z) * ScaleBack;
//                }).ToArray();

//                MaskPositions.Add(Positions);
//                MaskCellSizes.Add(ScaleBack);

//                if (DebugFromScratch)
//                {
//                    Rand = new Random(123);
//                    Image SampleTest = new Image(OriTomo.Dims);
//                    float[][] SampleData = SampleTest.GetHost(Intent.ReadWrite);
//                    for (int i = 0; i < 1000000; i++)
//                    {
//                        float3 PosCenter = Positions[Rand.Next(Positions.Length)];
//                        int3 Pos = new int3((PosCenter + new float3((float)Rand.NextDouble() * ScaleBack.X - ScaleBack.X / 2,
//                                                                    (float)Rand.NextDouble() * ScaleBack.Y - ScaleBack.Y / 2,
//                                                                    (float)Rand.NextDouble() * ScaleBack.Z - ScaleBack.Z / 2)).Round());

//                        SampleData[Pos.Z][Pos.Y * SampleTest.Dims.X + Pos.X] += 1f;
//                    }
//                    SampleTest.WriteMRC("d_sampletest.mrc", true);
//                }

//                #endregion

//                Console.WriteLine($" Done.");// {GPU.GetFreeMemory(GPU.GetDevice())} MB");
//                GPU.CheckGPUExceptions();

//                //if (OriTomos.Count >= 1)
//                //    break;
//            }
//            int NTomos = OriTomos.Count;
//            int NLeaveValidation = 0;
//            NTomos -= NLeaveValidation;

//            if (LoadedSeries.Count == 0)
//                throw new Exception("No tilt series were found. Please make sure the directory contains at least one .tomostar file with associated Warp/M metadata.");

//            Console.WriteLine("");

//            #endregion

//            TomoNet TrainModel = null;
//            string NameTrainedModel = Options.StartModelName;

//            #region Figure out device IDs and batch size

//            int NDevices = GPU.GetDeviceCount();
//            if (GPUsNetwork.Any(id => id >= NDevices) || Options.GPUPreprocess >= NDevices)
//            {
//                Console.WriteLine("Requested GPU ID that isn't present on this system. Defaulting to highest ID available.\n");

//                GPUsNetwork.RemoveAll(id => id >= NDevices);
//                Options.GPUPreprocess = Math.Min(Options.GPUPreprocess, NDevices - 1);

//                if (GPUsNetwork.Count == 0)
//                    throw new Exception("No more GPU IDs left after removing invalid ones, exiting.\n");
//            }

//            if (Options.BatchSize != 4 || LoadedSeries.Count > 1)
//            {
//                if (Options.BatchSize < 1)
//                    throw new Exception("Batch size must be at least 1.");

//                if (Options.BatchSize % GPUsNetwork.Count != 0)
//                    throw new Exception("Batch size must be divisible by the number of training GPUs.\n");

//                Options.NIterations = Options.NIterations * 4 / Options.BatchSize;
//                Console.WriteLine($"Adjusting the number of iterations to {Options.NIterations} to match batch size.\n");
//            }

//            #endregion

//            Rand = new Random(123);
//            Image.PrintObjectIDs();

//            for (int iepoch = 0; iepoch < Options.NEpochs; iepoch++)
//            {
//                TorchSharp.Torch.CudaEmptyCache();

//                #region Create source and target tomos from current best deconvolved versions

//                Console.Write("Preparing training data...");

//                foreach (var tomo in ImprovedTomos)
//                {
//                    tomo.Bandpass(0, LowpassFraction, true, 1f - LowpassFraction);
//                    GPU.PrefilterForCubic(tomo.GetDevice(Intent.ReadWrite), tomo.Dims);
//                    tomo.FreeDevice();
//                }

//                for (int itomo = 0; itomo < PrefilteredTomos.Count; itomo++)
//                {
//                    PrefilteredTomos[itomo].Dispose();

//                    Image Copy = OriTomos[itomo].GetCopy();
//                    Copy.Bandpass(0, LowpassFraction, true, 1f - LowpassFraction);
//                    GPU.PrefilterForCubic(Copy.GetDevice(Intent.ReadWrite), Copy.Dims);

//                    PrefilteredTomos[itomo] = Copy;
//                }

//                //Image.PrintObjectIDs();

//                Console.WriteLine(" Done.");

//                #endregion

//                #region Load model

//                TorchSharp.Torch.SetSeed(123);
//                if (TrainModel == null)
//                {
//                    Console.Write("Initializing model...");
//                    TrainModel = new TomoNet(DimsTrainingSource, DimsTrainingCTF, DimsTrainingTarget, GPUsNetwork.ToArray(), Options.BatchSize);
//                    Console.WriteLine(" Done.");
//                }
//                string ModelPath = Path.Combine(WorkingDirectory, NameTrainedModel);
//                if (File.Exists(ModelPath) && iepoch == 0)
//                {
//                    Console.Write($"Loading model parameters from {ModelPath}...");
//                    TrainModel.Load(ModelPath);
//                    Console.WriteLine(" Done.");
//                }

//                #endregion

//                #region Training

//                int NMaps = OriTomos.Count;
//                int NMapsPerBatch = Math.Min(8, NMaps);
//                int MapSamples = Options.BatchSize;

//                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(DimsTrainingSource.MultZ(MapSamples)), NMapsPerBatch);
//                Image[] ExtractedSourceRand = Helper.ArrayOfFunction(i => new Image(DimsTrainingSource.MultZ(MapSamples)), NMapsPerBatch);
//                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(DimsTrainingTarget.MultZ(MapSamples)), NMapsPerBatch);
//                Image[] ExtractedTargetRand = Helper.ArrayOfFunction(i => new Image(DimsTrainingTarget.MultZ(MapSamples)), NMapsPerBatch);
//                Image[] ExtractedCTF = Helper.ArrayOfFunction(i => new Image(DimsTrainingCTF.MultZ(MapSamples), true), NMapsPerBatch);
//                Image[] ExtractedCTFRand = Helper.ArrayOfFunction(i => new Image(DimsTrainingCTF.MultZ(MapSamples), true), NMapsPerBatch);

//                Stopwatch Watch = new Stopwatch();
//                Watch.Start();

//                Queue<float> Losses = new Queue<float>();

//                Image PredictedData = null;
//                float[] Loss = null;

//                for (int iter = 0, iterFine = 0; iter < Options.NIterations; iter++)
//                {
//                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999999));

//                    for (int m = 0; m < NMapsPerBatch; m++)
//                    {
//                        int MapID = ShuffledMapIDs[m];

//                        Image MapSource = ImprovedTomos[MapID];
//                        Image MapTarget = PrefilteredTomos[MapID];
//                        Image MapPSF = PrefilteredPSFs[MapID];
//                        Image MapCTF = OriCTFs[MapID];


//                        int3 DimsOriMap = MapSource.Dims;
//                        int3 Margin = DimsTrainingSource / 2;

//                        float3[] Positions = MaskPositions[MapID];
//                        float3 CellSize = MaskCellSizes[MapID];

//                        float3[] ExtractionPositions = new float3[MapSamples];
//                        float3[] ExtractionAngles = new float3[MapSamples];

//                        for (int i = 0; i < MapSamples; i++)
//                        {
//                            float3 CellCenter = Positions[Rand.Next(Positions.Length)];
//                            int3 Pos = new int3((CellCenter + new float3((float)Rand.NextDouble() * CellSize.X - CellSize.X / 2,
//                                                                         (float)Rand.NextDouble() * CellSize.Y - CellSize.Y / 2,
//                                                                         (float)Rand.NextDouble() * CellSize.Z - CellSize.Z / 2)).Round());

//                            Pos = int3.Max(Margin, int3.Min(Pos, DimsOriMap - Margin));
//                            ExtractionPositions[i] = new float3(Pos);

//                            decimal AngleStep = 1.5M;
//                            int MaxSteps = (int)(45M / AngleStep);
//                            int CurrentSteps = Math.Min(iepoch + 1, MaxSteps);

//                            float3 Angle = new float3(0);
//                            while (true)
//                            {
//                                int AngleX = Rand.Next(CurrentSteps * 2 + 1) - CurrentSteps;
//                                int AngleY = Rand.Next(CurrentSteps * 2 + 1) - CurrentSteps;
//                                int AngleZ = Rand.Next(CurrentSteps * 2 + 1) - CurrentSteps;

//                                if (AngleX == 0 && AngleY == 0 && AngleZ == 0)
//                                    continue;

//                                Angle = new float3(AngleX, AngleY, AngleZ) * (float)AngleStep * Helper.ToRad;
//                                break;
//                            }
//                            ExtractionAngles[i] = Angle;
//                        }

//                        if (true)
//                        {
//                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
//                            GPU.CreateTexture3D(MapSource.GetDevice(Intent.Read), MapSource.Dims, Texture, TextureArray, true);

//                            int PlanForw = GPU.CreateFFTPlan(new int3(DimCTF), 1);
//                            int PlanBack = GPU.CreateIFFTPlan(new int3(DimCTF), 1);

//                            for (int i = 0; i < MapSamples; i++)
//                            {
//                                Image Extracted = new Image(IntPtr.Zero, new int3(DimCTF));

//                                GPU.Rotate3DExtractAt(Texture[0],
//                                                              MapSource.Dims,
//                                                              Extracted.GetDevice(Intent.Write),
//                                                              Extracted.Dims,
//                                                              Helper.ToInterleaved(new float3[] { ExtractionAngles[i] }),
//                                                              Helper.ToInterleaved(new float3[] { ExtractionPositions[i] }),
//                                                              (uint)1);

//                                //Extracted.MaskRectangularly(Extracted.Dims / 2, Extracted.Dims.X / 4, true);
//                                //Extracted.WriteMRC("d_extracted.mrc", true);

//                                Image ExtractedFT = Extracted.AsFFT(true, PlanForw).AndDisposeParent();
//                                ExtractedFT.Multiply(MapCTF);

//                                Extracted = ExtractedFT.AsIFFT(true, PlanBack).AndDisposeParent().
//                                                        AsPadded(DimsTrainingSource).AndDisposeParent();

//                                GPU.CopyDeviceToDevice(Extracted.GetDevice(Intent.Read), 
//                                                       ExtractedSource[m].GetDeviceSlice(i * DimsTrainingSource.Z, Intent.Write), 
//                                                       Extracted.ElementsReal);
//                                Extracted.Dispose();
//                            }

//                            GPU.DestroyFFTPlan(PlanForw);
//                            GPU.DestroyFFTPlan(PlanBack);

//                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
//                            MapSource.FreeDevice();
//                            MapCTF.FreeDevice();

//                            //ExtractedSource[m].WriteMRC("d_extractedsource.mrc", true);
//                        }
//                        else
//                        {
//                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
//                            GPU.CreateTexture3D(MapSource.GetDevice(Intent.Read), MapSource.Dims, Texture, TextureArray, true);

//                            GPU.Rotate3DExtractAt(Texture[0],
//                                                  MapSource.Dims,
//                                                  ExtractedSource[m].GetDevice(Intent.Write),
//                                                  DimsTrainingSource,
//                                                  Helper.ToInterleaved(ExtractionAngles),
//                                                  Helper.ToInterleaved(ExtractionPositions),
//                                                  (uint)MapSamples);

//                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
//                            MapSource.FreeDevice();
//                        }

//                        {
//                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
//                            GPU.CreateTexture3D(MapTarget.GetDevice(Intent.Read), MapTarget.Dims, Texture, TextureArray, true);

//                            GPU.Rotate3DExtractAt(Texture[0],
//                                                  MapTarget.Dims,
//                                                  ExtractedTarget[m].GetDevice(Intent.Write),
//                                                  DimsTrainingTarget,
//                                                  Helper.ToInterleaved(ExtractionAngles),
//                                                  Helper.ToInterleaved(ExtractionPositions),
//                                                  (uint)MapSamples);

//                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
//                            MapTarget.FreeDevice();
//                        }

//                        {
//                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
//                            GPU.CreateTexture3D(MapPSF.GetDevice(Intent.Read), MapPSF.Dims, Texture, TextureArray, true);

//                            Image Extracted = new Image(IntPtr.Zero, new int3(DimsTrainingCTF.X, DimsTrainingCTF.Y, DimsTrainingCTF.Z * MapSamples));

//                            GPU.Rotate3DExtractAt(Texture[0],
//                                                  MapPSF.Dims,
//                                                  Extracted.GetDevice(Intent.Write),
//                                                  DimsTrainingCTF,
//                                                  Helper.ToInterleaved(ExtractionAngles),
//                                                  Helper.ToInterleaved(Helper.ArrayOfConstant(new float3(MapPSF.Dims / 2), MapSamples)),
//                                                  (uint)MapSamples);

//                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
//                            MapPSF.FreeDevice();

//                            Image ExtractedFT = new Image(IntPtr.Zero, Extracted.Dims, true, true);
//                            GPU.FFT(Extracted.GetDevice(Intent.Read),
//                                    ExtractedFT.GetDevice(Intent.Write),
//                                    DimsTrainingCTF,
//                                    (uint)MapSamples,
//                                    -1);
//                            GPU.Amplitudes(ExtractedFT.GetDevice(Intent.Read),
//                                           ExtractedCTF[m].GetDevice(Intent.Write),
//                                           ExtractedFT.ElementsComplex);
//                            ExtractedFT.Dispose();
//                            Extracted.Dispose();

//                            //ExtractedCTF[m].WriteMRC("d_extractedctf.mrc", true);
//                        }
//                    }

//                    // Shuffle individual examples between batches so each batch doesn't source from only one map
//                    for (int b = 0; b < MapSamples; b++)
//                    {
//                        int[] Order = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMapsPerBatch, 1), NMapsPerBatch, Rand.Next(9999999));
//                        for (int i = 0; i < Order.Length; i++)
//                        {
//                            GPU.CopyDeviceToDevice(ExtractedSource[i].GetDeviceSlice(b * DimsTrainingSource.Z, Intent.Read),
//                                                   ExtractedSourceRand[Order[i]].GetDeviceSlice(b * DimsTrainingSource.Z, Intent.Write),
//                                                   DimsTrainingSource.Elements());
//                            GPU.CopyDeviceToDevice(ExtractedTarget[i].GetDeviceSlice(b * DimsTrainingTarget.Z, Intent.Read),
//                                                   ExtractedTargetRand[Order[i]].GetDeviceSlice(b * DimsTrainingTarget.Z, Intent.Write),
//                                                   DimsTrainingTarget.Elements());
//                            GPU.CopyDeviceToDevice(ExtractedCTF[i].GetDeviceSlice(b * DimsTrainingCTF.Z, Intent.Read),
//                                                   ExtractedCTFRand[Order[i]].GetDeviceSlice(b * DimsTrainingCTF.Z, Intent.Write),
//                                                   DimsTrainingCTF.ElementsFFT());
//                        }
//                    }

//                    {
//                        //double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
//                        //                                                      (float)Math.Log(Options.LearningRateFinish),
//                        //                                                      iter / (float)Options.NIterations));
//                        double CurrentLearningRate = MathHelper.Lerp((float)Options.LearningRateStart,
//                                                                              (float)Options.LearningRateFinish,
//                                                                              iepoch / (float)(Options.NEpochs - 1));

//                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
//                        {

//                            int MapID = m;

//                            TrainModel.TrainDeconv(ExtractedSourceRand[MapID],
//                                                   ExtractedTargetRand[MapID],
//                                                   ExtractedCTFRand[MapID],
//                                                   (float)CurrentLearningRate,
//                                                   out PredictedData,
//                                                   out Loss);

//                            Losses.Enqueue(Loss[0]);
//                            if (Losses.Count > 40)
//                                Losses.Dequeue();

//                            if (true && iterFine % (iterFine < 100 ? 100 : 100) == 0)
//                            {
//                                ExtractedSourceRand[MapID].WriteMRC(Path.Combine(WorkingDirectory, $"d_source_{iterFine:D6}.mrc"), true);
//                                ExtractedTargetRand[MapID].WriteMRC(Path.Combine(WorkingDirectory, $"d_target_{iterFine:D6}.mrc"), true);
//                                PredictedData.WriteMRC(Path.Combine(WorkingDirectory, $"d_predicted_{iterFine:D6}.mrc"), true);
//                            }

//                            iterFine++;
//                        }
//                    }

//                    double TicksPerIteration = Watch.ElapsedTicks;// / (double)(iter + 1);
//                    TimeSpan TimeRemaining = new TimeSpan((long)(TicksPerIteration * (Options.NIterations - 1 - iter)));

//                    {
//                        double CurrentLearningRate = MathHelper.Lerp((float)Options.LearningRateStart,
//                                                                              (float)Options.LearningRateFinish,
//                                                                              iepoch / (float)(Options.NEpochs - 1));

//                        ClearCurrentConsoleLine();
//                        Console.Write($"{iter + 1}/{Options.NIterations}, " +
//                                      (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
//                                      $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " +
//                                      $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
//                                      $"lr = {CurrentLearningRate:F6}, " +
//                                      $"{GPU.GetFreeMemory(GPUsNetwork[0])} MB free");
//                    }

//                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
//                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

//                    GPU.CheckGPUExceptions();
//                    Watch.Restart();
//                }

//                Watch.Stop();

//                NameTrainedModel = $"TomoNet_epoch{(iepoch + 1):D2}.pt";
//                TrainModel.Save(Path.Combine(WorkingDirectory, NameTrainedModel));

//                Console.WriteLine($"\nEpoch {iepoch + 1} finished!");

//                #endregion

//                //for (int itomo = 0; itomo < NTomos; itomo++)
//                //{
//                //    for (int irot = 0; irot < NRotations; irot++)
//                //    {
//                //        Samplers[itomo * NRotations + irot].WriteMRC($"d_sampling_tomo{itomo:D2}_rot{irot:D2}.mrc", true);
//                //    }
//                //}

//                #region Clean up

//                TorchSharp.Torch.CudaEmptyCache();

//                foreach (var item in ExtractedSource)
//                    item.Dispose();
//                foreach (var item in ExtractedSourceRand)
//                    item.Dispose();
//                foreach (var item in ExtractedTarget)
//                    item.Dispose();
//                foreach (var item in ExtractedTargetRand)
//                    item.Dispose();
//                foreach (var item in ExtractedCTF)
//                    item.Dispose();
//                foreach (var item in ExtractedCTFRand)
//                    item.Dispose();

//                #endregion

//                #region Prepare for next epoch

//                Console.Write("Deconvolving all tomograms using updated model... ");

//                for (int itomo = 0; itomo < NTomos + NLeaveValidation; itomo++)
//                {
//                    Image OriTomo = OriTomos[itomo];
//                    Image OriPSF = OriPSFs[itomo];

//                    ImprovedTomos[itomo].Dispose();

//                    {
//                        Image DeconvTomo = OriTomo.GetCopyGPU().AndFreeParent();
//                        DeconvTomo.Bandpass(0, LowpassFraction, true, 1f - LowpassFraction);

//                        TomoNet.Deconvolve(DeconvTomo, new[] { TrainModel });

//                        // Pad tomo for convolution
//                        Image DeconvTomoPadded = DeconvTomo.AsPaddedClamped(DeconvTomo.Dims * 2);
//                        DeconvTomo.Dispose();

//                        // Get tomo-sized 1 - CTF
//                        Image PSFRemapped = OriPSF.GetCopyGPU().AndFreeParent();
//                        PSFRemapped.RemapToFT(true);
//                        Image CTFPadded = PSFRemapped.AsPadded(DeconvTomoPadded.Dims, true).AndDisposeParent().
//                                                      AsFFT(true).AndDisposeParent().
//                                                      AsReal().AndDisposeParent();

//                        float[][] CTFPaddedData = CTFPadded.GetHost(Intent.ReadWrite);
//                        Helper.ForEachElementFT(CTFPadded.Dims, (x, y, z, xx, yy, zz, r) =>
//                        {
//                            float fx = xx / (CTFPadded.Dims.X / 2f);
//                            float fy = yy / (CTFPadded.Dims.Y / 2f);
//                            float fz = zz / (CTFPadded.Dims.Z / 2f);
//                            float fr = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);

//                            CTFPaddedData[z][y * (CTFPadded.Dims.X / 2 + 1) + x] = (1f - CTFPaddedData[z][y * (CTFPadded.Dims.X / 2 + 1) + x]) * (fr >= 1 ? 0 : 1);
//                        });
//                        //CTFPadded.WriteMRC("d_oneminusctf.mrc", true);

//                        // Convolve prediction with 1 - CTF before adding it to the original tomo
//                        Image DeconvTomoPaddedFT = DeconvTomoPadded.AsFFT(true).AndDisposeParent();
//                        DeconvTomoPaddedFT.Multiply(CTFPadded);
//                        DeconvTomoPaddedFT.Multiply(1f / DeconvTomoPadded.ElementsReal);
//                        CTFPadded.Dispose();

//                        Image ImprovedTomo = DeconvTomoPaddedFT.AsIFFT(true, 0, false).AndDisposeParent().
//                                                                AsPadded(DeconvTomo.Dims).AndDisposeParent();

//                        //ImprovedTomo.Fill(0);
//                        ImprovedTomo.Add(OriTomo);
//                        OriTomo.FreeDevice();
//                        ImprovedTomo.FreeDevice();

//                        ImprovedTomos[itomo] = ImprovedTomo;

//                        ImprovedTomo.WriteMRC(Path.Combine(WorkingDirectory, $"{LoadedSeries[itomo].RootName}_{(iepoch + 1):D2}.mrc"), Options.PixelSize, true);

//                        if (false)
//                        {
//                            Image TruthConv = GroundTruthTomos[itomo].GetCopy();
//                            TruthConv.MaskRectangularly(new int3(300, 300, 2), 80, true);

//                            Image Predicted = ImprovedTomo.GetCopy();
//                            Predicted.MaskRectangularly(new int3(300, 300, 2), 80, true);

//                            Predicted.WriteMRC("d_predicted.mrc", true);
//                            TruthConv.WriteMRC("d_truth.mrc", true);

//                            float[] FSCPrediction = FSC.GetFSCNonCubic(Predicted, TruthConv, null, Predicted.Dims.Z / 2);
//                            new Star(FSCPrediction, "wrpFSC").Save($"d_fsc_{itomo}_e{(iepoch + 1):D2}.star");

//                            TruthConv.Dispose();
//                            Predicted.Dispose();
//                        }

//                        //Image.PrintObjectIDs();
//                    }

//                    OriTomo.FreeDevice();
//                    OriPSF.FreeDevice();
//                }

//                Console.WriteLine("Done.\n");

//                LowpassFraction = Math.Max(0, Math.Min(1, MathHelper.Lerp(Options.LowpassStart, Options.LowpassEnd, (iepoch + 1) / (float)Options.NEpochs)));

//                #endregion
//            }


//            //if (string.IsNullOrEmpty(Options.OldModelName))
//            //{
//            //    #region Load model

//            //    string ModelPath = Options.StartModelName;
//            //    if (!string.IsNullOrEmpty(ModelPath))
//            //    {
//            //        if (!File.Exists(ModelPath))
//            //            ModelPath = Path.Combine(ProgramFolder, Options.StartModelName);
//            //        if (!File.Exists(ModelPath))
//            //            throw new Exception($"Could not find initial model '{Options.StartModelName}'. Please make sure it can be found either here, or in the installation directory.");
//            //    }

//            //    Console.WriteLine("Loading model, " + GPU.GetFreeMemory(GPUsNetwork[0]) + " MB free.");
//            //    TrainModel = new NoiseNet3DTorch(DimsTrainingSource, DimsTrainingTarget, DimsTrainingCTF, true, false, GPUsNetwork.ToArray(), Options.BatchSize);
//            //    if (!string.IsNullOrEmpty(ModelPath))
//            //        TrainModel.Load(ModelPath);
//            //    Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(GPUsNetwork[0]) + " MB remaining.\n");

//            //    #endregion

//            //    #region Data loading and preprocessing

//            //    GPU.SetDevice(Options.GPUPreprocess);

//            //    Semaphore ReloadBlock = new Semaphore(1, 1);
//            //    List<int> AvailableDataIndices = Helper.ArrayOfSequence(0, LoadedSeries.Count, 1).ToList();
//            //    bool CreateData = true;

//            //    ParameterizedThreadStart ReloadLambda = (par) =>
//            //    {
//            //        #region Allocate resources

//            //        GPU.SetDevice(Options.GPUPreprocess);

//            //        Random TRand = new Random((int)par);
//            //        RandomNormal TRandN = new RandomNormal((int)par);

//            //        Image[] TVolumeFT = new[] { new Image(IntPtr.Zero, DimsReconstruction, true, true), new Image(IntPtr.Zero, DimsReconstruction, true, true) };
//            //        Image[] TVolumeCTF = new[] { null, new Image(IntPtr.Zero, DimsReconstruction, true, true) };
//            //        Image TVolumeWeights = new Image(IntPtr.Zero, DimsReconstruction, true);
//            //        Image TVolumeSampling = new Image(IntPtr.Zero, DimsReconstruction, true);

//            //        Image TCTFCoords = CTF.GetCTFCoords(DimsReconstruction.X, DimsReconstruction.X);

//            //        Image TParticleImages = new Image(IntPtr.Zero, new int3(DimsReconstruction.X, DimsReconstruction.Y, MaxNTilts));
//            //        Image TParticleImagesFT = new Image(IntPtr.Zero, new int3(DimsReconstruction.X, DimsReconstruction.Y, MaxNTilts), true, true);
//            //        Image TParticleCTFWeighted = new Image(IntPtr.Zero, new int3(DimsReconstruction.X, DimsReconstruction.Y, MaxNTilts), true);
//            //        Image TParticleCTFUnweighted = new Image(IntPtr.Zero, new int3(DimsReconstruction.X, DimsReconstruction.Y, MaxNTilts), true);
//            //        Image TParticleCTFComplex = new Image(IntPtr.Zero, new int3(DimsReconstruction.X, DimsReconstruction.Y, MaxNTilts), true, true);

//            //        int PlanParticleForw = GPU.CreateFFTPlan(new int3(DimsReconstruction.X, DimsReconstruction.Y, 1), (uint)MaxNTilts);

//            //        int PlanVolumeBack = GPU.CreateIFFTPlan(DimsReconstruction, 1);
//            //        int PlanCTFForw = GPU.CreateFFTPlan(DimsTrainingCTF, 1);

//            //        Image TVolumeIFT = new Image(IntPtr.Zero, DimsReconstruction);
//            //        Image TVolumeCropped = new Image(IntPtr.Zero, DimsTrainingCTF);
//            //        Image TVolumeCTFFT = new Image(IntPtr.Zero, DimsTrainingCTF, true, true);

//            //        Image TOnesComplex = new Image(IntPtr.Zero, DimsReconstruction, true, true);
//            //        TOnesComplex.Fill(new float2(1, 0));
//            //        Image TLerpA = new Image(IntPtr.Zero, DimsReconstruction, true, true);
//            //        Image THighpass = new Image(DimsReconstruction, true, true);
//            //        {
//            //            float[][] HighpassData = THighpass.GetHost(Intent.ReadWrite);
//            //            Helper.ForEachElementFT(DimsReconstruction, (x, y, z, xx, yy, zz, r) =>
//            //            {
//            //                float w1 = (float)Math.Cos(Math.Max(0, Math.Min((r - 4) / 8, 1)) * Math.PI) * 0.5f + 0.5f;
//            //                float w2 = 1f - ((float)Math.Cos(Math.Max(0, Math.Min((r - DimsReconstruction.X / 2 + 4) / 4, 1)) * Math.PI) * 0.5f + 0.5f);

//            //                HighpassData[z][(y * (DimsReconstruction.X / 2 + 1) + x) * 2 + 0] = Math.Max(w1, w2);
//            //            });
//            //            THighpass.WriteMRC("d_highpass.mrc", true);
//            //        }

//            //        #endregion

//            //        BatchData3D Workload = null;

//            //        while (CreateData)
//            //        {
//            //            // If this thread succeeded at pushing its previously loaded batch to processing
//            //            if (Workload == null)
//            //            {
//            //                int SelectedIndex;

//            //                lock (AvailableDataIndices)
//            //                {
//            //                    if (AvailableDataIndices.Count == 0)
//            //                    {
//            //                        Thread.Sleep(10);
//            //                        continue;
//            //                    }

//            //                    SelectedIndex = AvailableDataIndices[TRand.Next(AvailableDataIndices.Count)];
//            //                    AvailableDataIndices.Remove(SelectedIndex);
//            //                }

//            //                TiltSeries BatchSeries = LoadedSeries[SelectedIndex];
//            //                Image[] BatchDataFull = TiltDataFull[SelectedIndex];
//            //                Image[][] BatchDataOddEven = TiltDataOddEven[SelectedIndex];

//            //                Workload = new BatchData3D();

//            //                Image OutSourceVol = new Image(IntPtr.Zero, new int3(DimsTrainingSource.X, DimsTrainingSource.Y, DimsTrainingSource.Z * Options.BatchSize));
//            //                Image OutTargetVol = new Image(IntPtr.Zero, new int3(DimsTrainingTarget.X, DimsTrainingTarget.Y, DimsTrainingTarget.Z * Options.BatchSize));
//            //                Image OutTargetCTF = new Image(IntPtr.Zero, new int3(DimsTrainingCTF.X, DimsTrainingCTF.Y, DimsTrainingCTF.Z * Options.BatchSize), true);

//            //                for (int b = 0; b < Options.BatchSize; b++)
//            //                {
//            //                    int3 DimsVolSafe = DimsVolBinned - DimsTrainingSource;
//            //                    float3 Position = new float3(new int3(TRand.Next(DimsVolSafe.X), TRand.Next(DimsVolSafe.Y), TRand.Next(DimsVolSafe.Z)) + DimsTrainingSource / 2) * Options.PixelSize;
//            //                    float3 Angle = new float3((float)TRand.Next(4) * 90, (float)TRand.Next(4) * 90, (float)TRand.Next(4) * 90);

//            //                    BatchSeries.ReconstructDenoisingSubtomos(OptionsRec,
//            //                                                             BatchDataFull,
//            //                                                             BatchDataOddEven,
//            //                                                             new[] { Position },
//            //                                                             new[] { Angle },
//            //                                                             1,
//            //                                                             false,
//            //                                                             1,
//            //                                                             TRand.Next(),
//            //                                                             TVolumeFT,
//            //                                                             TVolumeCTF,
//            //                                                             TVolumeWeights,
//            //                                                             TVolumeSampling,
//            //                                                             TCTFCoords,
//            //                                                             TParticleImages,
//            //                                                             TParticleImagesFT,
//            //                                                             TParticleCTFWeighted,
//            //                                                             TParticleCTFUnweighted,
//            //                                                             TParticleCTFComplex,
//            //                                                             PlanParticleForw);

//            //                    #region Source volume

//            //                    GPU.IFFT(TVolumeFT[0].GetDevice(Intent.Read),
//            //                             TVolumeIFT.GetDevice(Intent.Write),
//            //                             DimsReconstruction,
//            //                             1,
//            //                             PlanVolumeBack,
//            //                             false);

//            //                    GPU.CropFTFull(TVolumeIFT.GetDevice(Intent.Read),
//            //                                   TVolumeCropped.GetDevice(Intent.Write),
//            //                                   DimsReconstruction,
//            //                                   DimsTrainingSource,
//            //                                   1);

//            //                    GPU.RemapFullFromFTFloat(TVolumeCropped.GetDevice(Intent.Read),
//            //                                             OutSourceVol.GetDeviceSlice(b * DimsTrainingSource.Z, Intent.Write),
//            //                                             DimsTrainingSource,
//            //                                             1);

//            //                    #endregion

//            //                    #region Target volume

//            //                    GPU.IFFT(TVolumeFT[1].GetDevice(Intent.Read),
//            //                             TVolumeIFT.GetDevice(Intent.Write),
//            //                             DimsReconstruction,
//            //                             1,
//            //                             PlanVolumeBack,
//            //                             false);

//            //                    GPU.CropFTFull(TVolumeIFT.GetDevice(Intent.Read),
//            //                                   TVolumeCropped.GetDevice(Intent.Write),
//            //                                   DimsReconstruction,
//            //                                   DimsTrainingTarget,
//            //                                   1);

//            //                    GPU.RemapFullFromFTFloat(TVolumeCropped.GetDevice(Intent.Read),
//            //                                             OutTargetVol.GetDeviceSlice(b * DimsTrainingTarget.Z, Intent.Write),
//            //                                             DimsTrainingTarget,
//            //                                             1);

//            //                    #endregion

//            //                    #region Target CTF

//            //                    //TVolumeCTF[1].WriteMRC("d_ctf1.mrc", true);

//            //                    GPU.SubtractFromSlices(TOnesComplex.GetDevice(Intent.Read),
//            //                                           TVolumeCTF[1].GetDevice(Intent.Read),
//            //                                           TLerpA.GetDevice(Intent.Write),
//            //                                           TOnesComplex.ElementsReal,
//            //                                           1);
//            //                    GPU.MultiplySlices(TLerpA.GetDevice(Intent.Read),
//            //                                       THighpass.GetDevice(Intent.Read),
//            //                                       TLerpA.GetDevice(Intent.Write),
//            //                                       TLerpA.ElementsReal,
//            //                                       1);
//            //                    TVolumeCTF[1].Add(TLerpA);

//            //                    GPU.IFFT(TVolumeCTF[1].GetDevice(Intent.Read),
//            //                             TVolumeIFT.GetDevice(Intent.Write),
//            //                             DimsReconstruction,
//            //                             1,
//            //                             PlanVolumeBack,
//            //                             false);

//            //                    GPU.CropFTFull(TVolumeIFT.GetDevice(Intent.Read),
//            //                                   TVolumeCropped.GetDevice(Intent.Write),
//            //                                   DimsReconstruction,
//            //                                   DimsTrainingCTF,
//            //                                   1);

//            //                    GPU.FFT(TVolumeCropped.GetDevice(Intent.Read),
//            //                            TVolumeCTFFT.GetDevice(Intent.Write),
//            //                            DimsTrainingCTF,
//            //                            1,
//            //                            PlanCTFForw);

//            //                    GPU.Real(TVolumeCTFFT.GetDevice(Intent.Read),
//            //                             OutTargetCTF.GetDeviceSlice(b * DimsTrainingCTF.Z, Intent.Write),
//            //                             TVolumeCTFFT.ElementsComplex);

//            //                    #endregion
//            //                }

//            //                OutTargetCTF.Multiply(1f / DimsReconstruction.Elements());

//            //                OutSourceVol.Multiply(1 / 10f);
//            //                OutTargetVol.Multiply(1 / 10f);

//            //                Workload.SourceVol = OutSourceVol;
//            //                Workload.TargetVol = OutTargetVol;
//            //                Workload.TargetCTF = OutTargetCTF;

//            //                foreach (var item in BatchDataFull)
//            //                    item.FreeDevice();

//            //                foreach (var oddeven in BatchDataOddEven)
//            //                    if (oddeven != null)
//            //                        foreach (var item in oddeven)
//            //                            item.FreeDevice();

//            //                lock (AvailableDataIndices)
//            //                {
//            //                    AvailableDataIndices.Add(SelectedIndex);
//            //                }
//            //            }

//            //            ReloadBlock.WaitOne();
//            //            if (BatchDataPreprocessed.Count < NThreads && Workload != null)
//            //            {
//            //                BatchDataPreprocessed.Enqueue(Workload);

//            //                Workload = null;
//            //                //Debug.WriteLine($"Using T {par}");
//            //            }
//            //            ReloadBlock.Release();
//            //        }

//            //        #region Waste disposal

//            //        Workload?.Dispose();

//            //        THighpass.Dispose();
//            //        TLerpA.Dispose();
//            //        TOnesComplex.Dispose();

//            //        TVolumeCTFFT.Dispose();
//            //        TVolumeCropped.Dispose();
//            //        TVolumeIFT.Dispose();

//            //        GPU.DestroyFFTPlan(PlanCTFForw);
//            //        GPU.DestroyFFTPlan(PlanVolumeBack);
//            //        GPU.DestroyFFTPlan(PlanParticleForw);

//            //        TParticleCTFComplex.Dispose();
//            //        TParticleCTFUnweighted.Dispose();
//            //        TParticleCTFWeighted.Dispose();
//            //        TParticleImagesFT.Dispose();
//            //        TParticleImages.Dispose();

//            //        TCTFCoords.Dispose();

//            //        TVolumeSampling.Dispose();
//            //        TVolumeWeights.Dispose();
//            //        foreach (var item in TVolumeCTF)
//            //            item?.Dispose();
//            //        foreach (var item in TVolumeFT)
//            //            item?.Dispose();

//            //        #endregion
//            //    };
//            //    Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
//            //    for (int i = 0; i < NThreads; i++)
//            //        ReloadThreads[i].Start(i);

//            //    #endregion

//            //    #region Training loop

//            //    Stopwatch Watch = new Stopwatch();
//            //    Watch.Start();

//            //    Queue<float> Losses = new Queue<float>();

//            //    long IterationsDone = 0;

//            //    while (IterationsDone < Options.NIterations)
//            //    {
//            //        if (BatchDataPreprocessed.Count == 0)
//            //            continue;

//            //        double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
//            //                                                              (float)Math.Log(Options.LearningRateFinish),
//            //                                                              IterationsDone / (float)Options.NIterations));

//            //        BatchData3D Workload = null;
//            //        ReloadBlock.WaitOne();
//            //        {
//            //            Workload = BatchDataPreprocessed.Dequeue();
//            //        }
//            //        ReloadBlock.Release();

//            //        float[] Loss = null;
//            //        Image Prediction, PredictionDeconv;

//            //        if (true)
//            //        {
//            //            TrainModel.TrainDeconv(Workload.SourceVol,
//            //                                   Workload.TargetVol,
//            //                                   Workload.TargetCTF,
//            //                                   (float)CurrentLearningRate,
//            //                                   out Prediction,
//            //                                   out PredictionDeconv,
//            //                                   out Loss);

//            //            Losses.Enqueue(Loss[0]);

//            //            GPU.CheckGPUExceptions();
//            //        }

//            //        if (IterationsDone % 100 == 0)
//            //        {
//            //            Workload.SourceVol.WriteMRC($"d_source_{IterationsDone:D6}.mrc", true);
//            //            Workload.TargetVol.WriteMRC($"d_target_{IterationsDone:D6}.mrc", true);
//            //            Workload.TargetCTF.WriteMRC($"d_targetctf_{IterationsDone:D6}.mrc", true);

//            //            Prediction.WriteMRC($"d_prediction_{IterationsDone:D6}.mrc", true);
//            //            PredictionDeconv.WriteMRC($"d_predictiondeconv_{IterationsDone:D6}.mrc", true);

//            //            GPU.CheckGPUExceptions();
//            //        }

//            //        Workload.Dispose();

//            //        GPU.CheckGPUExceptions();



//            //        double TicksPerIteration = Watch.ElapsedTicks;
//            //        TimeSpan TimeRemaining = new TimeSpan((long)(TicksPerIteration * (Options.NIterations - 1 - IterationsDone)));

//            //        if (true)
//            //        {
//            //            ClearCurrentConsoleLine();
//            //            Console.Write($"{IterationsDone + 1}/{Options.NIterations}, " +
//            //                          (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
//            //                          $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " +
//            //                          $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
//            //                          $"lr = {CurrentLearningRate:F6}, " +
//            //                          $"{GPU.GetFreeMemory(GPUsNetwork[0])} MB free");
//            //        }

//            //        if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
//            //            throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

//            //        GPU.CheckGPUExceptions();
//            //        Watch.Restart();

//            //        IterationsDone++;
//            //    }

//            //    Watch.Stop();
//            //    CreateData = false;

//            //    NameTrainedModel = "NoiseNet3D_" + (!string.IsNullOrEmpty(Options.StartModelName) ? (Options.StartModelName + "_") : "") +
//            //                       DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
//            //    TrainModel.Save(Path.Combine(WorkingDirectory, NameTrainedModel));

//            //    Thread.Sleep(500);

//            //    foreach (var item in BatchDataPreprocessed)
//            //        item.Dispose();

//            //    Console.WriteLine("\nDone training!\n");

//            //    #endregion
//            //}

//            //#region Denoise

//            //Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(GPUsNetwork[0]) + " MB free.");
//            //if (TrainModel == null)
//            //    TrainModel = new NoiseNet3DTorch(DimsTrainingSource, DimsTrainingTarget, DimsTrainingCTF, false, false, GPUsNetwork.ToArray(), Options.BatchSize);
//            //if (!File.Exists(Path.Combine(WorkingDirectory, NameTrainedModel)))
//            //    throw new Exception("Old model could not be found.");
//            //TrainModel.Load(Path.Combine(WorkingDirectory, NameTrainedModel));
//            ////TrainModel = new NoiseNet3D(@"H:\denoise_refine\noisenet3d_64_20180808_010023", new int3(Dim), 1, Options.BatchSize, false, Options.GPUNetwork);
//            //Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(GPUsNetwork[0]) + " MB remaining.\n");

//            ////Directory.Delete(NameTrainedModel, true);

//            //Directory.CreateDirectory(Path.Combine(WorkingDirectory, "denoisedtomograms"));

//            //GPU.SetDevice(Options.GPUPreprocess);

//            //for (int iseries = 0; iseries < LoadedSeries.Count; iseries++)
//            //{
//            //    Console.Write($"Denoising {LoadedSeries[iseries].Name}... ");

//            //    Image Denoised = NoiseNet3DTorch.DenoiseTiltSeries(DimsVolBinned, LoadedSeries[iseries], OptionsRec, DimsReconstruction, TiltDataFull[iseries], TrainModel);
//            //    Denoised.WriteMRC(Path.Combine(WorkingDirectory, "denoisedtomograms", $"{LoadedSeries[iseries].RootName}_{Options.PixelSize:F1}Apx.mrc"), Options.PixelSize, true);
//            //    Denoised.Dispose();

//            //    foreach (var item in TiltDataFull[iseries])
//            //        item?.Dispose();
//            //    if (TiltDataOddEven[iseries][0] != null)
//            //        foreach (var item in TiltDataOddEven[iseries][0])
//            //            item?.Dispose();
//            //    if (TiltDataOddEven[iseries][1] != null)
//            //        foreach (var item in TiltDataOddEven[iseries][1])
//            //            item?.Dispose();
//            //}

//            //Console.WriteLine("\nAll done!");

//            //#endregion
//        }

//        public static Image RescaleCTF(Image ctf, int3 dimsNew)
//        {
//            Image CTFComplex = new Image(ctf.Dims, true, true);
//            CTFComplex.Fill(new float2(1, 0));
//            CTFComplex.Multiply(ctf);

//            Image PSF = CTFComplex.AsIFFT(true).AndDisposeParent().AsPadded(dimsNew, true).AndDisposeParent();
//            PSF.Multiply(1f / dimsNew.Elements());

//            return PSF.AsFFT(true).AndDisposeParent().AsReal().AndDisposeParent();
//        }

//        private static void ClearCurrentConsoleLine()
//        {
//            int currentLineCursor = Console.CursorTop;
//            Console.SetCursorPosition(0, Console.CursorTop);
//            Console.Write(new string(' ', Console.WindowWidth));
//            Console.SetCursorPosition(0, currentLineCursor);
//        }
//    }

//    class BatchData3D
//    {
//        public Image SourceVol;
//        public Image TargetVol;
//        public Image TargetCTF;

//        public void Dispose()
//        {
//            SourceVol?.Dispose();
//            TargetVol?.Dispose();
//            TargetCTF?.Dispose();
//        }
//    }
//}
