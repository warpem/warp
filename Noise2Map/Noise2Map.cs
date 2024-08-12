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

namespace Noise2Map
{
    class Noise2Map
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

                if (args.Length == 0 ||
                    Result.Tag == ParserResultType.NotParsed ||
                    Result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError ||
                                           e.Tag == ErrorType.HelpRequestedError))
                    return;

                WorkingDirectory = Environment.CurrentDirectory + "/";
            }

            if ((!string.IsNullOrEmpty(Options.Observation1Path) || !string.IsNullOrEmpty(Options.Observation1Path)) &&
                (!string.IsNullOrEmpty(Options.HalfMap1Path) || !string.IsNullOrEmpty(Options.HalfMap1Path)))
                throw new ArgumentException("Can't use --observation1/2 and --half1/2 at the same time");

            if (string.IsNullOrEmpty(Options.Observation1Path) && string.IsNullOrEmpty(Options.HalfMap1Path))
                throw new ArgumentException("You need to specify either two folders with half-maps (--observation1/2) or two single half-maps (--half1/2)");

            if (!string.IsNullOrEmpty(Options.Observation1Path) && string.IsNullOrEmpty(Options.Observation2Path))
                throw new ArgumentException("When specifying --observation1, you also need to specify --observation2");

            if (!string.IsNullOrEmpty(Options.HalfMap1Path) && string.IsNullOrEmpty(Options.HalfMap2Path))
                throw new ArgumentException("When specifying --half1, you also need to specify --half2");

            int3 TrainingDims = new int3(Options.WindowSize);
            bool IsTomo = !string.IsNullOrEmpty(Options.CTFPath);

            //if (!Options.DontFlatten && Options.PixelSize < 0)
            //    throw new Exception("Flattening requested, but pixel size not specified.");

            if (!Options.DontAugment && !string.IsNullOrEmpty(Options.CTFPath))
                throw new ArgumentException("3D CTF cannot be combined with data augmentation.");

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
            //Console.WriteLine($"{NDevices} devices");
            //Console.WriteLine($"{string.Join(",", Options.GPUNetwork.Select(v => v.ToString()))} training");
            //Console.WriteLine($"{Options.GPUPreprocess} preprocessing");

            GPU.SetDevice(Options.GPUPreprocess);

            #region Mask

            Console.Write("Loading mask... ");

            Image Mask = null;
            int3 CropBox = new int3(-1);
            int3 BoundsMin = new int3(0);
            int3 BoundsMax = new int3(10000);
            if (!string.IsNullOrEmpty(Options.MaskPath))
            {
                Mask = Image.FromFile(Path.Combine(WorkingDirectory, Options.MaskPath));

                if (!Options.DontKeepDimensions)
                {
                    BoundsMin = Mask.Dims;
                    BoundsMax = new int3(0);
                }

                Mask.TransformValues((x, y, z, v) =>
                {
                    if (v > 1e-3f)
                    {
                        CropBox.X = Math.Max(CropBox.X, Math.Abs(x - Mask.Dims.X / 2) * 2);
                        CropBox.Y = Math.Max(CropBox.Y, Math.Abs(y - Mask.Dims.Y / 2) * 2);
                        CropBox.Z = Math.Max(CropBox.Z, Math.Abs(z - Mask.Dims.Z / 2) * 2);

                        if (!Options.DontKeepDimensions)
                        {
                            BoundsMin = int3.Min(BoundsMin, new int3(x, y, z));
                            BoundsMax = int3.Max(BoundsMax, new int3(x, y, z));
                        }
                    }

                    return v;
                });

                if (CropBox.X < 2)
                    throw new Exception("Mask does not seem to contain any non-zero values.");

                CropBox += 64;

                CropBox.X = Math.Min(CropBox.X, Mask.Dims.X);
                CropBox.Y = Math.Min(CropBox.Y, Mask.Dims.Y);
                CropBox.Z = Math.Min(CropBox.Z, Mask.Dims.Z);
            }

            Console.WriteLine("done.\n");

            #endregion

            #region Load and prepare data

            Console.WriteLine("Preparing data:");

            List<Image> Maps1 = new List<Image>();
            List<Image> Maps2 = new List<Image>();
            List<Image> MapCTFs = new List<Image>();
            List<ulong[]> Textures1 = new List<ulong[]>();
            List<ulong[]> Textures2 = new List<ulong[]>();
            List<Image> MapsForDenoising = new List<Image>();
            List<Image> MapsForDenoising2 = new List<Image>();
            List<string> NamesForDenoising = new List<string>();
            List<int3> DimensionsForDenoising = new List<int3>();
            List<int3> OriginalBoxForDenoising = new List<int3>();
            List<float2> MeanStdForDenoising = new List<float2>();
            List<float> PixelSizeForDenoising = new List<float>();

            string[] OddMapPaths = new string[0];
            if (!string.IsNullOrEmpty(Options.Observation1Path))
                OddMapPaths = Directory.EnumerateFiles(Path.Combine(WorkingDirectory, Options.Observation1Path), "*.mrc").ToArray();
            else if (!string.IsNullOrEmpty(Options.HalfMap1Path))
                OddMapPaths = new string[] { Path.Combine(WorkingDirectory, Options.HalfMap1Path) };
            else 
                throw new Exception("Shouldn't be here!");

            foreach (var file in OddMapPaths)
            {
                string MapName = Helper.PathToName(file);
                string[] Map2Paths = null;
                if (!string.IsNullOrEmpty(Options.Observation2Path))
                    Map2Paths = Directory.EnumerateFiles(Path.Combine(WorkingDirectory, Options.Observation2Path), MapName + ".mrc").ToArray();
                else if (!string.IsNullOrEmpty(Options.HalfMap1Path))
                    Map2Paths = new string[] { Path.Combine(WorkingDirectory, Options.HalfMap2Path) };
                else
                    throw new Exception("Shouldn't be here!");

                if (Map2Paths == null || Map2Paths.Length == 0)
                    continue;

                string MapCombinedPath = null;
                if (!string.IsNullOrEmpty(Options.ObservationCombinedPath))
                {
                    string[] MapCombinedPaths = Directory.EnumerateFiles(Path.Combine(WorkingDirectory, Options.ObservationCombinedPath), MapName + ".mrc").ToArray();
                    if (MapCombinedPaths == null || MapCombinedPaths.Length == 0)
                        continue;
                    MapCombinedPath = MapCombinedPaths.First();
                }

                if (Options.PixelSize < 0 && Maps1.Count == 0)
                {
                    MapHeader Header = MapHeader.ReadFromFile(file);
                    Options.PixelSize = Header.PixelSize.X;

                    Console.WriteLine($"Set pixel size to {Options.PixelSize} based on map header.");
                }

                if (!Options.DontFlatten && Maps1.Count == 0)
                {
                    MapHeader Header = MapHeader.ReadFromFile(file);
                    if (!Header.Dimensions.IsCubic)
                    {
                        Console.WriteLine("Map is not cubic and thus likely a tomogram. Enabling --dont_flatten_spectrum because flattening only works on cubic volumes");
                        Options.DontFlatten = true;
                    }
                }

                Console.Write($"Preparing {MapName}... ");

                Image Map1 = Image.FromFile(file);
                Image Map2 = Image.FromFile(Map2Paths.First());
                Image MapCombined = MapCombinedPath == null ? null : Image.FromFile(MapCombinedPath);

                float MapPixelSize = Map1.PixelSize;

                if (!Options.DontFlatten)
                {
                    Image Average = Map1.GetCopy();
                    Average.Add(Map2);

                    if (Mask != null)
                        Average.Multiply(Mask);

                    float[] Spectrum = Average.AsAmplitudes1D(true, 1, (Average.Dims.X + Average.Dims.Y + Average.Dims.Z) / 6);
                    Average.Dispose();

                    int i10A = (int)(Options.PixelSize * 2 / 10 * Spectrum.Length);
                    float Amp10A = Spectrum[i10A];

                    for (int i = 0; i < Spectrum.Length; i++)
                        Spectrum[i] = i < i10A ? 1 : (Amp10A / Spectrum[i] * Options.Overflatten);

                    Image Map1Flat = Map1.AsSpectrumMultiplied(true, Spectrum);
                    Map1.Dispose();
                    Map1 = Map1Flat;
                    Map1.FreeDevice();

                    Image Map2Flat = Map2.AsSpectrumMultiplied(true, Spectrum);
                    Map2.Dispose();
                    Map2 = Map2Flat;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedFlat = MapCombined.AsSpectrumMultiplied(true, Spectrum);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedFlat;
                        MapCombined.FreeDevice();
                    }
                }

                if (Options.Lowpass > 0)
                {
                    Map1.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    Map2.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    MapCombined?.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                }

                OriginalBoxForDenoising.Add(Map1.Dims);

                if (!Options.DontKeepDimensions)
                    BoundsMax = int3.Min(BoundsMax, Map1.Dims);

                if (Options.DontKeepDimensions && CropBox.X > 0)
                {
                    Image Map1Cropped = Map1.AsPadded(CropBox);
                    Map1.Dispose();
                    Map1 = Map1Cropped;
                    Map1.FreeDevice();

                    Image Map2Cropped = Map2.AsPadded(CropBox);
                    Map2.Dispose();
                    Map2 = Map2Cropped;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedCropped = MapCombined.AsPadded(CropBox);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedCropped;
                        MapCombined.FreeDevice();
                    }
                }

                float2 MeanStd;
                {
                    Image Map1Center = Map1.AsPadded(Map1.Dims / 2);
                    Image Map2Center = Map2.AsPadded(Map2.Dims / 2);
                    MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1Center.GetHostContinuousCopy(), Map2Center.GetHostContinuousCopy()));

                    Map1Center.Dispose();
                    Map2Center.Dispose();
                }
                MeanStdForDenoising.Add(MeanStd);

                float MaxStd = 30;
                Map1.TransformValues(v => Math.Max(-MaxStd, Math.Min(MaxStd, (v - MeanStd.X) / MeanStd.Y)));
                Map2.TransformValues(v => Math.Max(-MaxStd, Math.Min(MaxStd, (v - MeanStd.X) / MeanStd.Y)));
                MapCombined?.TransformValues(v => Math.Max(-MaxStd, Math.Min(MaxStd, (v - MeanStd.X) / MeanStd.Y)));

                Image ForDenoising = (MapCombined == null || Options.DenoiseSeparately) ? Map1.GetCopy() : MapCombined;
                Image ForDenoising2 = Options.DenoiseSeparately ? Map2.GetCopy() : null;

                if (!Options.DenoiseSeparately)
                {
                    ForDenoising.Add(Map2);
                    ForDenoising.Multiply(0.5f);
                }

                GPU.PrefilterForCubic(Map1.GetDevice(Intent.ReadWrite), Map1.Dims);
                GPU.PrefilterForCubic(Map2.GetDevice(Intent.ReadWrite), Map2.Dims);

                Map1.FreeDevice();
                Maps1.Add(Map1);
                Map2.FreeDevice();
                Maps2.Add(Map2);

                ForDenoising.FreeDevice();
                MapsForDenoising.Add(ForDenoising);
                NamesForDenoising.Add(MapName);

                PixelSizeForDenoising.Add(MapPixelSize);

                if (Options.DenoiseSeparately)
                {
                    ForDenoising2.FreeDevice();
                    MapsForDenoising2.Add(ForDenoising2);
                }

                if (!string.IsNullOrEmpty(Options.CTFPath) &&
                    File.Exists(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + ".mrc")))
                {
                    Image MapCTF = Image.FromFile(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + ".mrc"));
                    {
                        int DimCTF = MapCTF.Dims.Y;
                        MapCTF.Dims = new int3(DimCTF);
                        MapCTF.IsFT = true;
                        Image CTFComplex = new Image(MapCTF.Dims, true, true);
                        CTFComplex.Fill(new float2(1, 0));
                        CTFComplex.Multiply(MapCTF);
                        MapCTF.Dispose();
                        Image CTFReal = CTFComplex.AsIFFT(true).AndDisposeParent();
                        Image CTFPadded = CTFReal.AsPadded(TrainingDims * 2, true).AndDisposeParent();
                        CTFComplex = CTFPadded.AsFFT(true).AndDisposeParent();
                        MapCTF = CTFComplex.AsReal().AndDisposeParent();
                        MapCTF.Multiply(1f / (DimCTF * DimCTF * DimCTF));
                    }

                    float[][] CTFData = MapCTF.GetHost(Intent.ReadWrite);
                    Helper.ForEachElementFT(TrainingDims * 2, (x, y, z, xx, yy, zz, r) =>
                    {
                        float xxx = xx / (float)TrainingDims.X;
                        float yyy = yy / (float)TrainingDims.Y;
                        float zzz = zz / (float)TrainingDims.Z;

                        r = (float)Math.Sqrt(xxx * xxx + yyy * yyy + zzz * zzz);

                        float b = Math.Min(Math.Max(0, r - 0.98f) / 0.02f, 1);

                        r = Math.Min(1, r / 0.05f);
                        r = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                        float a = 90;
                        if (zzz != 0)
                            a = (float)Math.Atan(Math.Abs(xxx / zzz)) * Helper.ToDeg;
                        a = Math.Max(0, Math.Min(1, (a - 20) / 5));
                        a = 1;

                        int i = y * (MapCTF.Dims.X / 2 + 1) + x;
                        CTFData[z][i] = MathHelper.Lerp(MathHelper.Lerp(MathHelper.Lerp(CTFData[z][i], 1, r), 1, b), 1, 1 - a);
                    });

                    MapCTF.WriteMRC(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + "_scaled.mrc"), true);
                    MapCTFs.Add(MapCTF);
                    Console.WriteLine("Found CTF");
                }
                else
                {
                    Image MapCTF = new Image(new int3(128), true);
                    MapCTF.TransformValues(v => 1f);
                    MapCTFs.Add(MapCTF);
                }

                Console.WriteLine($" Done.");// {GPU.GetFreeMemory(GPU.GetDevice())} MB");
                GPU.CheckGPUExceptions();
            }

            Mask?.FreeDevice();

            if (Maps1.Count == 0)
                throw new Exception("No maps were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine("");

            #endregion

            NoiseNet3DTorch TrainModel = null;
            string NameTrainedModel = Options.OldModelName;
            int3 Dim = TrainingDims;
            int3 Dim2 = Dim * 2;

            if (Options.BatchSize != 4 || Maps1.Count > 1)
            {
                if (Options.BatchSize < 1)
                    throw new Exception("Batch size must be at least 1.");

                Options.NIterations = Options.NIterations * 4 / Options.BatchSize / Math.Min(8, Maps1.Count);
                Console.WriteLine($"Adjusting the number of iterations to {Options.NIterations} to match batch size and number of maps.\n");
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
                TrainModel = new NoiseNet3DTorch(Dim, 
                                                 Options.GPUNetwork.ToArray(), 
                                                 Options.BatchSize, 
                                                 depth: Options.MiniModel ? 1 : 2, 
                                                 progressiveDepth: !Options.MiniModel, 
                                                 maxWidth: Options.MiniModel ? 64 : 99999);
                if (!string.IsNullOrEmpty(ModelPath))
                    TrainModel.Load(ModelPath);
                Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB remaining.\n");

                #endregion

                GPU.SetDevice(Options.GPUPreprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Maps1.Count;
                int NMapsPerBatch = Math.Min(8, NMaps);
                int MapSamples = Options.BatchSize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedSourceRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTargetRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim2.X, Dim2.Y, Dim2.Z * MapSamples), true), NMapsPerBatch);
                Image[] ExtractedCTFRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim2.X, Dim2.Y, Dim2.Z * MapSamples), true), NMapsPerBatch);

                foreach (var item in MapCTFs)
                    item.GetDevice(Intent.Read);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                Image PredictedData = null;
                float[] Loss = null;

                ulong[] Texture1 = new ulong[1];
                ulong[] Texture2 = new ulong[1];
                ulong[] TextureArray1 = new ulong[1];
                ulong[] TextureArray2 = new ulong[1];

                for (int iter = 0, iterFine = 0; iter < Options.NIterations; iter++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999999));

                    for (int m = 0; m < NMapsPerBatch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image Map1 = Maps1[MapID];
                        Image Map2 = Maps2[MapID];

                        int3 DimsMap = Map1.Dims;

                        int3 Margin = Dim / 2;
                        //Margin.Z = 0;
                        float3[] Position = null;
                        if (BoundsMin == new int3(0))
                            Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                              (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                              (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), MapSamples);
                        else
                            Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (BoundsMax - BoundsMin).X + BoundsMin.X,
                                                                              (float)Rand.NextDouble() * (BoundsMax - BoundsMin).Y + BoundsMin.Y,
                                                                              (float)Rand.NextDouble() * (BoundsMax - BoundsMin).Z + BoundsMin.Z), MapSamples);

                        float3[] Angle;
                        if (Options.DontAugment)
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Math.Round(Rand.NextDouble()) * 0,
                                                                           (float)Math.Round(Rand.NextDouble()) * 0,
                                                                           (float)Math.Round(Rand.NextDouble()) * 0) * Helper.ToRad, MapSamples);
                        else
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360) * Helper.ToRad, MapSamples);

                        {
                            if (NMaps > 1 || iterFine == 0)
                                GPU.CreateTexture3D(Map1.GetDevice(Intent.Read), Map1.Dims, Texture1, TextureArray1, true);
                            if (NMaps > 4)
                                Map1.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture1[0],
                                                  Map1.Dims,
                                                  ExtractedSource[m].GetDevice(Intent.Write),
                                                  Dim,
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);

                            if (NMaps > 1)
                                GPU.DestroyTexture(Texture1[0], TextureArray1[0]);
                        }

                        {
                            if (NMaps > 1 || iterFine == 0)
                                GPU.CreateTexture3D(Map2.GetDevice(Intent.Read), Map2.Dims, Texture2, TextureArray2, true);
                            if (NMaps > 4)
                                Map2.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture2[0],
                                                  Map2.Dims,
                                                  ExtractedTarget[m].GetDevice(Intent.Write),
                                                  Dim,
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);

                            if (NMaps > 1)
                                GPU.DestroyTexture(Texture2[0], TextureArray2[0]);
                        }

                        {
                            for (int i = 0; i < MapSamples; i++)
                                GPU.CopyDeviceToDevice(MapCTFs[MapID].GetDevice(Intent.Read),
                                                       ExtractedCTF[m].GetDeviceSlice(i * Dim2.Z, Intent.Write),
                                                       MapCTFs[MapID].ElementsReal);
                        }

                        //Map1.FreeDevice();
                        //Map2.FreeDevice();
                    }

                    // Shuffle individual examples between batches so each batch doesn't source from only one map
                    for (int b = 0; b < MapSamples; b++)
                    {
                        int[] Order = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMapsPerBatch, 1), NMapsPerBatch, Rand.Next(9999999));
                        for (int i = 0; i < Order.Length; i++)
                        {
                            GPU.CopyDeviceToDevice(ExtractedSource[i].GetDeviceSlice(b * Dim.Z, Intent.Read),
                                                   ExtractedSourceRand[Order[i]].GetDeviceSlice(b * Dim.Z, Intent.Write),
                                                   Dim.Elements());
                            GPU.CopyDeviceToDevice(ExtractedTarget[i].GetDeviceSlice(b * Dim.Z, Intent.Read),
                                                   ExtractedTargetRand[Order[i]].GetDeviceSlice(b * Dim.Z, Intent.Write),
                                                   Dim.Elements());
                            GPU.CopyDeviceToDevice(ExtractedCTF[i].GetDeviceSlice(b * Dim2.Z, Intent.Read),
                                                   ExtractedCTFRand[Order[i]].GetDeviceSlice(b * Dim2.Z, Intent.Write),
                                                   (Dim2.X / 2 + 1) * Dim2.Y * Dim2.Z);
                        }
                    }

                    double CurrentLearningRate = MathHelper.Lerp((float)Options.LearningRateStart,
                                                                    (float)Options.LearningRateFinish,
                                                                    iter / (float)Options.NIterations);

                    {
                        //double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
                        //                                                      (float)Math.Log(Options.LearningRateFinish),
                        //                                                      iter / (float)Options.NIterations));

                        if (iterFine < 100)
                            CurrentLearningRate = MathHelper.Lerp(0, (float)CurrentLearningRate, iterFine / 99f);

                        Image NoiseMask = new Image(IntPtr.Zero, ExtractedSourceRand[0].Dims);

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            float[] AdversarialAngles = Helper.ArrayOfFunction(v => ((float)Rand.NextDouble() - 0.5f) * 2 * Helper.ToRad * 90, MapSamples);
                            //float[] AdversarialAngles = Helper.ArrayOfFunction(v => (Rand.Next(2) == 0 ? 1 : -1) * 1.5f * Helper.ToRad, MapSamples);
                            float2[] AdversarialShifts = Helper.ArrayOfFunction(v => new float2(((float)Rand.NextDouble() - 0.5f) * 2, ((float)Rand.NextDouble() - 0.5f) * 2), MapSamples);
                            //float2[] AdversarialShifts = Helper.ArrayOfFunction(v => new float2(0, 0), MapSamples);

                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            if (IsTomo)
                                TrainModel.TrainDeconv((Twist ? ExtractedSourceRand : ExtractedTargetRand)[MapID],
                                                       (Twist ? ExtractedTargetRand : ExtractedSourceRand)[MapID],
                                                       ExtractedCTFRand[MapID],
                                                       (float)CurrentLearningRate,
                                                       false,
                                                       null,
                                                       null,
                                                       out PredictedData,
                                                       out _,
                                                       out _,
                                                       out Loss,
                                                       out _);
                            else
                                TrainModel.Train((Twist ? ExtractedSourceRand : ExtractedTargetRand)[MapID],
                                                 (Twist ? ExtractedTargetRand : ExtractedSourceRand)[MapID],
                                                 (float)CurrentLearningRate,
                                                 out PredictedData,
                                                 out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 10)
                                Losses.Dequeue();

                            //if (false && iterFine % (iterFine < 100 ? 10 : 100) == 0)
                            //{
                            //    ExtractedSourceRand[MapID].WriteMRC(Path.Combine(WorkingDirectory, $"d_source_{iterFine:D6}.mrc"), true);
                            //    PredictedData.WriteMRC(Path.Combine(WorkingDirectory, $"d_predicted_{iterFine:D6}.mrc"), true);
                            //}

                            iterFine++;
                        }

                        NoiseMask.Dispose();
                    }

                    TimeSpan TimeRemaining = Watch.Elapsed * (Options.NIterations - 1 - iter);

                    {
                        string ToWrite = $"{iter + 1}/{Options.NIterations}, " +
                                         (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
                                         $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " +
                                         $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
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
                    }

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();
                    Watch.Restart();
                }

                if (NMaps == 1)
                {
                    GPU.DestroyTexture(Texture1[0], TextureArray1[0]);
                    GPU.DestroyTexture(Texture2[0], TextureArray2[0]);
                }

                Watch.Stop();

                NameTrainedModel = "NoiseNet3D_" + (!string.IsNullOrEmpty(Options.StartModelName) ? (Options.StartModelName + "_") : "") +
                                   DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
                TrainModel.Save(Path.Combine(WorkingDirectory, NameTrainedModel));
                TrainModel.Dispose();

                Console.WriteLine("\nDone training!\n");

                #endregion
            }

            #region Denoise

            Options.BatchSize = Options.GPUNetwork.Count();

            Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB free.");
            TrainModel = new NoiseNet3DTorch(TrainingDims, 
                                             Options.GPUNetwork.ToArray(), 
                                             Options.BatchSize, 
                                             depth: Options.MiniModel ? 1 : 2, 
                                             progressiveDepth: !Options.MiniModel, 
                                             maxWidth: Options.MiniModel ? 64 : 99999);
            if (!File.Exists(Path.Combine(WorkingDirectory, NameTrainedModel)))
                throw new Exception("Old model could not be found.");
            TrainModel.Load(Path.Combine(WorkingDirectory, NameTrainedModel));
            //TrainModel = new NoiseNet3D(@"H:\denoise_refine\noisenet3d_64_20180808_010023", new int3(Dim), 1, Options.BatchSize, false, Options.GPUNetwork);
            Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(Options.GPUNetwork.First()) + " MB remaining.\n");

            //Directory.Delete(NameTrainedModel, true);

            Directory.CreateDirectory(Path.Combine(WorkingDirectory, "denoised"));

            GPU.SetDevice(Options.GPUPreprocess);

            for (int imap = 0; imap < MapsForDenoising.Count; imap++)
            {
                Console.Write($"Denoising {NamesForDenoising[imap]}... ");

                Image Map1 = MapsForDenoising[imap];
                NoiseNet3DTorch.Denoise(Map1, new NoiseNet3DTorch[] { TrainModel });

                float2 MeanStd = MeanStdForDenoising[imap];
                Map1.TransformValues(v => v * MeanStd.Y + MeanStd.X);

                Map1.PixelSize = PixelSizeForDenoising[imap];

                if (Options.MaskOutput)
                    Map1.Multiply(Mask);
                else if (Mask != null && !Options.DontKeepDimensions)
                    Map1.MaskSpherically(Map1.Dims.X - 32, 16, true);

                string SavePath1 = Path.Combine(WorkingDirectory, "denoised", NamesForDenoising[imap] + (Options.DenoiseSeparately ? "_1" : "") + ".mrc");
                Map1.WriteMRC16b(SavePath1, true);
                Map1.Dispose();

                Console.WriteLine("Done. Saved to " + SavePath1);

                if (Options.DenoiseSeparately)
                {
                    Console.Write($"Denoising {NamesForDenoising[imap]} (2nd observation)... ");

                    Image Map2 = MapsForDenoising2[imap];
                    NoiseNet3DTorch.Denoise(Map2, new NoiseNet3DTorch[] { TrainModel });

                    Map2.TransformValues(v => v * MeanStd.Y + MeanStd.X);

                    Map2.PixelSize = PixelSizeForDenoising[imap];

                    if (Options.MaskOutput)
                        Map2.Multiply(Mask);
                    else if (Mask != null && !Options.DontKeepDimensions)
                        Map2.MaskSpherically(Map2.Dims.X - 32, 16, true);

                    string SavePath2 = Path.Combine(WorkingDirectory, "denoised",  NamesForDenoising[imap] + "_2" + ".mrc");
                    Map2.WriteMRC16b(SavePath2, true);
                    Map2.Dispose();

                    Console.WriteLine("Done. Saved to " + SavePath2);
                }
            }

            Console.WriteLine("\nAll done!");

            #endregion
        }
    }
}
