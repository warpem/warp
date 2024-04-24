using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace Noise2Mic
{
    class Noise2Mic
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;
            VirtualConsole.AttachToConsole();

            #region Command line options

            Options Options = new Options();
            string WorkingDirectory;

            string ProgramFolder = System.AppContext.BaseDirectory;
            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

            {
                var Result = Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);

                if (Result.Tag == ParserResultType.NotParsed ||
                    Result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError ||
                                           e.Tag == ErrorType.HelpRequestedError))
                    return;

                WorkingDirectory = Environment.CurrentDirectory + "/";
            }

            int2 TrainingDims = new int2(Options.WindowSize);

            #endregion

            int NDevices = GPU.GetDeviceCount();
            if (Options.GPUNetwork.Any(id => id >= NDevices))
            {
                Console.WriteLine($"Requested GPU ID ({Options.GPUNetwork.First(id => id >= NDevices)}) that isn't present on this system.");
            }
            if (Options.GPUPreprocess >= NDevices)
            {
                Options.GPUPreprocess = Math.Min(Options.GPUPreprocess, NDevices - 1);
            }

            GPU.SetDevice(Options.GPUPreprocess);

            #region Load and prepare data

            Console.WriteLine("Preparing data:");

            List<Image> Mics1 = new List<Image>();
            List<Image> Mics2 = new List<Image>();
            List<Image> MicsForDenoising = new List<Image>();
            List<Movie> OriginalMovies = new List<Movie>();
            List<int2> OriginalDims = new List<int2>();
            List<float2> MeanStdForDenoising = new List<float2>();
            List<float> PixelSizeForDenoising = new List<float>();

            foreach (var file in Directory.EnumerateFiles(Path.Combine(WorkingDirectory, Options.ProcessingDirPath), "*.xml"))
            {
                //if (MicsForDenoising.Count > 50)
                //    break;

                string MapName = Helper.PathToName(file);
                string PathDir = Helper.PathToFolder(file);

                Movie M;

                try
                {
                    M = new Movie(file);
                    if (!File.Exists(M.AverageEvenPath) || !File.Exists(M.AverageOddPath))
                    {
                        Console.WriteLine($"No odd/even averages found for {MapName}, skipping");
                        continue;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    continue;
                }

                Console.Write($"Preparing {MapName}... ");

                Image Map1 = Image.FromFile(M.AverageEvenPath);
                Image Map2 = Image.FromFile(M.AverageOddPath);

                float MapPixelSize = Map1.PixelSize;

                int2 DimsOri = new int2(Map1.Dims);
                int2 DimsScaled = Options.PixelSize > 0 ? new int2((new float2(DimsOri) * MapPixelSize / Options.PixelSize + 1) / 2) * 2 : DimsOri;

                Image MapScaled1 = Map1.AsScaled(DimsScaled).AndDisposeParent();
                Image MapScaled2 = Map2.AsScaled(DimsScaled).AndDisposeParent();

                if (File.Exists(M.MaskPath))
                {
                    Image Mask = LoadAndScaleMask(M.MaskPath, DimsScaled);
                    TiltSeries.EraseDirt(MapScaled1, Mask);
                    TiltSeries.EraseDirt(MapScaled2, Mask);
                }

                //if (Options.Lowpass > 0)
                {
                    float BPScaling = 8f / Options.PixelSize;
                    float LP = Options.Lowpass > 0 ? Options.PixelSize * 2 / Options.Lowpass : 1f;
                    int3 OriDims = MapScaled1.Dims;

                    MapScaled1 = MapScaled1.AsPadded(new int2(MapScaled1.Dims) - 16).AndDisposeParent();
                    MapScaled1 = MapScaled1.AsPaddedClamped(new int2(OriDims) * 2).AndDisposeParent();
                    MapScaled1.Bandpass(1f / 32f / BPScaling, LP, false, 1f / 32f / BPScaling);
                    MapScaled1 = MapScaled1.AsPadded(new int2(OriDims)).AndDisposeParent();

                    MapScaled2 = MapScaled2.AsPadded(new int2(MapScaled1.Dims) - 16).AndDisposeParent();
                    MapScaled2 = MapScaled2.AsPaddedClamped(new int2(OriDims) * 2).AndDisposeParent();
                    MapScaled2.Bandpass(1f / 32f / BPScaling, LP, false, 1f / 32f / BPScaling);
                    MapScaled2 = MapScaled2.AsPadded(new int2(OriDims)).AndDisposeParent();

                    //MapScaled1.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    //MapScaled2.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                }

                OriginalDims.Add(DimsOri);
                OriginalMovies.Add(M);

                float2 MeanStd;
                {
                    Image Map1Center = MapScaled1.AsPadded(DimsScaled / 2);
                    Image Map2Center = MapScaled2.AsPadded(DimsScaled / 2);
                    MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1Center.GetHostContinuousCopy(), Map2Center.GetHostContinuousCopy()));

                    Map1Center.Dispose();
                    Map2Center.Dispose();
                }
                MeanStdForDenoising.Add(MeanStd);

                float MaxStd = 30;
                MapScaled1.TransformValues(v => Math.Max(-MaxStd, Math.Min(MaxStd, (v - MeanStd.X) / MeanStd.Y)));
                MapScaled2.TransformValues(v => Math.Max(-MaxStd, Math.Min(MaxStd, (v - MeanStd.X) / MeanStd.Y)));

                Image ForDenoising = MapScaled1.GetCopyGPU();
                ForDenoising.Add(MapScaled2);
                ForDenoising.Multiply(0.5f);

                GPU.PrefilterForCubic(MapScaled1.GetDevice(Intent.ReadWrite), MapScaled1.Dims);
                MapScaled1.FreeDevice();
                Mics1.Add(MapScaled1);

                GPU.PrefilterForCubic(MapScaled2.GetDevice(Intent.ReadWrite), MapScaled2.Dims);
                MapScaled2.FreeDevice();
                Mics2.Add(MapScaled2);

                ForDenoising.FreeDevice();
                MicsForDenoising.Add(ForDenoising);

                PixelSizeForDenoising.Add(MapPixelSize);

                Console.WriteLine($" Done.");// {GPU.GetFreeMemory(GPU.GetDevice())} MB");
                GPU.CheckGPUExceptions();
            }

            if (Mics1.Count == 0)
                throw new Exception("No micrographs were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine("");

            #endregion

            NoiseNet2DTorch TrainModel = null;
            string NameTrainedModel = Options.OldModelName;
            int2 Dim = TrainingDims;

            if (Options.BatchSize != 64 || Mics1.Count > 1)
            {
                if (Options.BatchSize < 1)
                    throw new Exception("Batch size must be at least 1.");

                int NMapsPerBatch = Math.Min(8, Mics1.Count);

                Options.NIterations = Options.NIterations * 64 / Options.BatchSize / NMapsPerBatch;
                Console.WriteLine($"Adjusting the number of iterations to {Options.NIterations * NMapsPerBatch} to match batch size and number of micrographs.\n");
            }


            if (string.IsNullOrEmpty(Options.OldModelName))
            {
                #region Load model

                string ModelPath = Options.StartModelName;
                if (!string.IsNullOrEmpty(ModelPath))
                {
                    if (File.Exists(Path.Combine(WorkingDirectory, Options.StartModelName)))
                        ModelPath = Path.Combine(WorkingDirectory, Options.StartModelName);
                    else if (File.Exists(Path.Combine(ProgramFolder, Options.StartModelName)))
                        ModelPath = Path.Combine(ProgramFolder, Options.StartModelName);

                    if (!File.Exists(ModelPath))
                        throw new Exception($"Could not find initial model '{Options.StartModelName}'. Please make sure it can be found either here, or in the installation directory.");
                }

                Console.WriteLine("Loading model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB free.");
                TrainModel = new NoiseNet2DTorch(Dim, Options.GPUNetwork.ToArray(), Options.BatchSize);
                if (!string.IsNullOrEmpty(ModelPath))
                    TrainModel.Load(ModelPath);
                Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB remaining.\n");

                #endregion

                GPU.SetDevice(Options.GPUPreprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Mics1.Count;
                int NMapsPerBatch = Math.Min(8, NMaps);
                int MapSamples = Options.BatchSize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, MapSamples)), NMapsPerBatch);
                Image[] ExtractedSourceRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, MapSamples)), NMapsPerBatch);
                Image[] ExtractedTargetRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, MapSamples)), NMapsPerBatch);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                Image PredictedData = null;
                float[] Loss = null;

                for (int iter = 0, iterFine = 0; iter < Options.NIterations; iter++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999999));

                    for (int m = 0; m < NMapsPerBatch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image Map1 = Mics1[MapID];
                        Image Map2 = Mics2[MapID];
                        //ulong Texture1 = Textures1[MapID][0];
                        //ulong Texture2 = Textures2[MapID][0];

                        int3 DimsMap = Map1.Dims;

                        int2 Margin = Dim / 2;
                        //Margin.Z = 0;
                        float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                   0), MapSamples);

                        float3[] Angle = Helper.ArrayOfFunction(i => new float3(0, 0, (float)Rand.NextDouble() * (Options.DontAugment ? 0 : 360)) * Helper.ToRad, MapSamples);

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map1.GetDevice(Intent.Read), Map1.Dims, Texture, TextureArray, true);
                            Map1.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map1.Dims,
                                                  ExtractedSource[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map2.GetDevice(Intent.Read), Map2.Dims, Texture, TextureArray, true);
                            Map2.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map2.Dims,
                                                  ExtractedTarget[m].GetDevice(Intent.Write),
                                                  new int3(Dim),
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        Map1.FreeDevice();
                        Map2.FreeDevice();
                    }

                    // Shuffle individual examples between batches so each batch doesn't source from only one map
                    for (int b = 0; b < MapSamples; b++)
                    {
                        int[] Order = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMapsPerBatch, 1), NMapsPerBatch, Rand.Next(9999999));
                        for (int i = 0; i < Order.Length; i++)
                        {
                            GPU.CopyDeviceToDevice(ExtractedSource[i].GetDeviceSlice(b, Intent.Read),
                                                   ExtractedSourceRand[Order[i]].GetDeviceSlice(b, Intent.Write),
                                                   Dim.Elements());
                            GPU.CopyDeviceToDevice(ExtractedTarget[i].GetDeviceSlice(b, Intent.Read),
                                                   ExtractedTargetRand[Order[i]].GetDeviceSlice(b, Intent.Write),
                                                   Dim.Elements());
                        }
                    }

                    {
                        double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
                                                                              (float)Math.Log(Options.LearningRateFinish),
                                                                              iter / (float)Options.NIterations));

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            TrainModel.Train((Twist ? ExtractedSourceRand : ExtractedTargetRand)[MapID],
                                                (Twist ? ExtractedTargetRand : ExtractedSourceRand)[MapID],
                                                (float)CurrentLearningRate,
                                                out PredictedData,
                                                out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 10)
                                Losses.Dequeue();

                            if (false && iterFine % (iterFine < 100 ? 10 : 100) == 0)
                            {
                                ExtractedSourceRand[MapID].WriteMRC(Path.Combine(WorkingDirectory, $"d_source_{iterFine:D6}.mrc"), true);
                                PredictedData.WriteMRC(Path.Combine(WorkingDirectory, $"d_predicted_{iterFine:D6}.mrc"), true);
                            }

                            {
                                TimeSpan TimeRemaining = Watch.Elapsed * ((Options.NIterations * NMapsPerBatch) - 1 - iterFine);

                                string ToWrite = $"{iterFine + 1}/{Options.NIterations * NMapsPerBatch}, " +
                                                 (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
                                                 $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " +
                                                 $"log(loss) = {Math.Log(MathHelper.Mean(Losses)):F4}, " +
                                                 $"lr = {CurrentLearningRate:F6}, " +
                                                 $"{GPU.GetFreeMemory(Options.GPUNetwork.First())} MB free";

                                try
                                {
                                    VirtualConsole.ClearLastLine();
                                    Console.Write(ToWrite);
                                }
                                catch
                                {
                                    // When we're outputting to a text file when launched on HPC cluster
                                    Console.WriteLine(ToWrite);
                                }

                                Watch.Restart();
                            }

                            iterFine++;
                        }
                    }

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();
                }

                Watch.Stop();

                NameTrainedModel = "NoiseNet2D_" + (!string.IsNullOrEmpty(Options.StartModelName) ? (Options.StartModelName + "_") : "") +
                                   DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
                TrainModel.Save(Path.Combine(WorkingDirectory, NameTrainedModel));
                TrainModel.Dispose();

                Console.WriteLine("\nDone training!\n");

                #endregion
            }

            #region Denoise

            Options.BatchSize = Options.GPUNetwork.Count();

            Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB free.");
            TrainModel = new NoiseNet2DTorch(TrainingDims, Options.GPUNetwork.ToArray(), Options.BatchSize);
            if (!File.Exists(Path.Combine(WorkingDirectory, NameTrainedModel)))
                throw new Exception("Old model could not be found.");
            TrainModel.Load(Path.Combine(WorkingDirectory, NameTrainedModel));
            Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB remaining.\n");

            //Directory.Delete(NameTrainedModel, true);

            Directory.CreateDirectory(OriginalMovies.First().AverageDenoisedDir);

            GPU.SetDevice(Options.GPUPreprocess);

            for (int imap = 0; imap < MicsForDenoising.Count; imap++)
            {
                Console.Write($"Denoising {OriginalMovies[imap]}... ");

                Image Map1 = MicsForDenoising[imap];
                NoiseNet2DTorch.Denoise(Map1, new NoiseNet2DTorch[] { TrainModel });

                float2 MeanStd = MeanStdForDenoising[imap];

                Map1.TransformValues(v => v * MeanStd.Y + MeanStd.X);

                if (new int2(Map1.Dims) != OriginalDims[imap])
                    Map1 = Map1.AsScaled(OriginalDims[imap]).AndDisposeParent();

                string SavePath1 = OriginalMovies[imap].AverageDenoisedPath;
                Map1.WriteMRC(SavePath1, true);
                Map1.Dispose();

                Console.WriteLine("Done. Saved to " + SavePath1);
            }

            Console.WriteLine("\nAll done!");

            #endregion
        }

        static Image[] RawMaskBuffers = new Image[GPU.GetDeviceCount()];
        static Image[] ScaledMaskBuffers = new Image[GPU.GetDeviceCount()];
        public static Image LoadAndScaleMask(string maskPath, int2 dimsTarget)
        {
            int2 DimsScaled = dimsTarget;

            int CurrentDevice = GPU.GetDevice();

            #region Make sure reusable buffers are there and have correct dimensions

            if (ScaledMaskBuffers[CurrentDevice] == null ||
                ScaledMaskBuffers[CurrentDevice].ElementsReal != DimsScaled.Elements())
            {
                if (ScaledMaskBuffers[CurrentDevice] != null)
                    ScaledMaskBuffers[CurrentDevice].Dispose();

                ScaledMaskBuffers[CurrentDevice] = new Image(new int3(DimsScaled));
            }

            #endregion

            string MaskPath = maskPath;

            MapHeader MaskHeader = MapHeader.ReadFromFile(MaskPath);

            if (RawMaskBuffers[CurrentDevice] == null || RawMaskBuffers[CurrentDevice].ElementsReal != MaskHeader.Dimensions.Elements())
            {
                if (RawMaskBuffers[CurrentDevice] != null)
                    RawMaskBuffers[CurrentDevice].Dispose();

                RawMaskBuffers[CurrentDevice] = new Image(MaskHeader.Dimensions);
            }

            TiffNative.ReadTIFFPatient(50, 500, MaskPath, 0, true, RawMaskBuffers[CurrentDevice].GetHost(Intent.Write)[0]);

            #region Rescale and re-binarize

            GPU.Scale(RawMaskBuffers[CurrentDevice].GetDevice(Intent.Read),
                        ScaledMaskBuffers[CurrentDevice].GetDevice(Intent.Write),
                        MaskHeader.Dimensions,
                        new int3(DimsScaled),
                        1,
                        0,
                        0,
                        IntPtr.Zero,
                        IntPtr.Zero);

            ScaledMaskBuffers[CurrentDevice].Binarize(0.7f);
            ScaledMaskBuffers[CurrentDevice].FreeDevice();

            #endregion

            return ScaledMaskBuffers[CurrentDevice];
        }
    }
}
