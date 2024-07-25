using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using System.Diagnostics;
using System.Threading;
using MathNet.Numerics.Statistics;
using TorchSharp;

namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_boxnet_train", HelpText = "(Re)train a BoxNet model on image/label pairs, producing a new model")]
    [CommandRunner(typeof(BoxNetTrainFrameseries))]
    class BoxNetTrainFrameseriesOptions
    {
        [Option("model_in", HelpText = "Path to the .pt file containing the old model weights; model will be initialized from scratch if this is left empty")]
        public string ModelIn { get; set; }

        [Option("model_out", Required = true, HelpText = "Path to the .pt file where the new model weights will be saved")]
        public string ModelOut { get; set; }

        [Option("examples_pick", HelpText = "Path to a folder containing TIFF files with particle picking examples prepared with boxnet_examples_frameseries")]
        public string ExamplesNewPicking { get; set; }

        [Option("examples_general_pick", HelpText = "Path to a folder containing TIFF files with particle picking examples used to train a general model, which will be mixed 1:1 with new examples to reduce overfitting")]
        public string ExamplesGeneralPicking { get; set; }

        [Option("examples_denoise", HelpText = "Path to a folder containing MRC files with denoising examples prepared with boxnet_examples_frameseries")]
        public string ExamplesNewDenoising { get; set; }

        [Option("examples_general_denoise", HelpText = "Path to a folder containing MRC files with denoising examples used to train a general model, which will be mixed 1:1 with new examples to reduce overfitting")]
        public string ExamplesGeneralDenoising { get; set; }

        [Option("no_mask", HelpText = "Don't consider mask labels in training; they will be converted to background labels")]
        public bool NoMask { get; set; }

        [Option("patchsize", Default = 512, HelpText = "Size of the BoxNet input window, a multiple of 256; remember to use the same window with fs_boxnet_infer")]
        public int PatchSize { get; set; }

        [Option("batchsize", Default = 8, HelpText = "Size of the minibatches used in training; larger batches require more GPU memory; must be divisible by number of devices")]
        public int BatchSize { get; set; }

        [Option("lr_start", Default = 5e-5, HelpText = "Learning rate at training start")]
        public double LearningRateStart { get; set; }

        [Option("lr_end", Default = 1e-5, HelpText = "Learning rate at training end, with linear interpolation in-between")]
        public double LearningRateEnd { get; set; }

        [Option("lr_denoise", HelpText = "Optionally, specify a different, constant learning rate for the denoiser")]
        public double? LearningRateDenoise { get; set; }

        [Option("epochs", Default = 100, HelpText = "Number of training epochs")]
        public int NEpochs { get; set; }

        [Option("checkpoints", Default = 0, HelpText = "Save checkpoints every N minutes; set to 0 to disable")]
        public int Checkpoints { get; set; }

        [Option("device_data", HelpText = "GPU ID for storing raw data and preparing examples")]
        public int DeviceData { get; set; }

        [Option("devices_model", HelpText = "Space-separated list of GPU IDs to be used for training")]
        public IEnumerable<int> DevicesModel { get; set; }
    }

    class BoxNetTrainFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            BoxNetTrainFrameseriesOptions CLI = options as BoxNetTrainFrameseriesOptions;

            #region Validate options

            if (!string.IsNullOrEmpty(CLI.ModelIn) && !File.Exists(CLI.ModelIn))
                throw new Exception($"--model_in file {CLI.ModelIn} does not exist");

            if (!string.IsNullOrEmpty(CLI.ExamplesNewPicking) && !Directory.Exists(CLI.ExamplesNewPicking))
                throw new Exception($"--examples_pick folder {CLI.ExamplesNewPicking} does not exist");

            if (!string.IsNullOrEmpty(CLI.ExamplesGeneralPicking) && !Directory.Exists(CLI.ExamplesGeneralPicking))
                throw new Exception($"--examples_general_pick folder {CLI.ExamplesGeneralPicking} does not exist");

            if (!string.IsNullOrEmpty(CLI.ExamplesNewDenoising) && !Directory.Exists(CLI.ExamplesNewDenoising))
                throw new Exception($"--examples_denoise folder {CLI.ExamplesNewDenoising} does not exist");

            if (!string.IsNullOrEmpty(CLI.ExamplesGeneralDenoising) && !Directory.Exists(CLI.ExamplesGeneralDenoising))
                throw new Exception($"--examples_general_denoise folder {CLI.ExamplesGeneralDenoising} does not exist");

            if (CLI.PatchSize % 256 != 0 || CLI.PatchSize <= 0)
                throw new Exception("--patchsize must be positive and a multiple of 256");

            if (CLI.BatchSize <= 0)
                throw new Exception("--batchsize must be positive");

            if (CLI.LearningRateStart < 0.0)
                throw new Exception("--lr_start can't be negative");

            if (CLI.LearningRateEnd < 0.0)
                throw new Exception("--lr_end can't be negative");

            if (CLI.LearningRateDenoise != null && CLI.LearningRateDenoise < 0.0)
                throw new Exception("--lr_denoise can't be negative");

            if (CLI.LearningRateStart < CLI.LearningRateEnd)
                throw new Exception("--lr_start must be greater than or equal to --lr_end");

            if (CLI.NEpochs <= 0)
                throw new Exception("--epochs must be positive");

            if (CLI.DevicesModel == null || !CLI.DevicesModel.Any())
                CLI.DevicesModel = new[] { 0 };

            if (CLI.DeviceData < 0 || CLI.DeviceData >= GPU.GetDeviceCount())
                throw new Exception($"--device_data must be a GPU ID between 0 and {GPU.GetDeviceCount() - 1} (inclusive)");

            if (CLI.DevicesModel.Any(d => d < 0 || d >= GPU.GetDeviceCount()))
                throw new Exception($"--devices must be a list of GPU IDs between 0 and {GPU.GetDeviceCount() - 1} (inclusive)");

            if (CLI.BatchSize % CLI.DevicesModel.Count() != 0)
                throw new Exception($"--batchsize must be divisible by the number of devices ({CLI.DevicesModel.Count()})");

            #endregion

            GPU.SetDevice(CLI.DeviceData);

            #region Load data

            bool UseCorpusPick = CLI.ExamplesGeneralPicking != null;
            bool UseCorpusDenoise = CLI.ExamplesGeneralDenoising != null;

            int2 DimsLargest = new int2(1);

            List<string[]> AllPathsPick = new();
            if (!string.IsNullOrWhiteSpace(CLI.ExamplesNewPicking))
            {
                if (Directory.EnumerateFiles(CLI.ExamplesNewPicking, "*.tif").Any())
                    AllPathsPick.Add(Directory.EnumerateFiles(CLI.ExamplesNewPicking, "*.tif").ToArray());
                else
                    Console.Error.WriteLine($"No TIFF files found in {CLI.ExamplesNewPicking}");
            }
            if (!string.IsNullOrWhiteSpace(CLI.ExamplesGeneralPicking))
            {
                if (Directory.EnumerateFiles(CLI.ExamplesGeneralPicking, "*.tif").Any())
                    AllPathsPick.Add(Directory.EnumerateFiles(CLI.ExamplesGeneralPicking, "*.tif").ToArray());
                else
                    Console.Error.WriteLine($"No TIFF files found in {CLI.ExamplesGeneralPicking}");
            }

            List<string[]> AllPathsDenoise = new();
            if (!string.IsNullOrWhiteSpace(CLI.ExamplesNewDenoising))
            {
                if (Directory.EnumerateFiles(CLI.ExamplesNewDenoising, "*.mrc").Any())
                    AllPathsDenoise.Add(Directory.EnumerateFiles(CLI.ExamplesNewDenoising, "*.mrc").ToArray());
                else
                    Console.Error.WriteLine($"No MRC files found in {CLI.ExamplesNewDenoising}");
            }
            if (!string.IsNullOrWhiteSpace(CLI.ExamplesGeneralDenoising))
            {
                if (Directory.EnumerateFiles(CLI.ExamplesGeneralDenoising, "*.mrc").Any())
                    AllPathsDenoise.Add(Directory.EnumerateFiles(CLI.ExamplesGeneralDenoising, "*.mrc").ToArray());
                else
                    Console.Error.WriteLine($"No MRC files found in {CLI.ExamplesGeneralDenoising}");
            }

            List<ulong[]>[] AllMicrographsPick = Helper.ArrayOfFunction(i => new List<ulong[]>(), AllPathsPick.Count);
            List<Image>[] AllLabelsPick = Helper.ArrayOfFunction(i => new List<Image>(), AllPathsPick.Count);
            List<int2>[] AllDimsPick = Helper.ArrayOfFunction(i => new List<int2>(), AllPathsPick.Count);
            float[] LabelWeightsPick = { 1f, 1f, 1f };

            List<ExampleDenoise>[] ExamplesDenoise = Helper.ArrayOfFunction(i => new List<ExampleDenoise>(), Math.Max(AllPathsDenoise.Count, AllPathsPick.Count));
            List<ExampleDeconv>[] ExamplesDeconv = Helper.ArrayOfFunction(i => new List<ExampleDeconv>(), Math.Max(AllPathsDenoise.Count, AllPathsPick.Count));

            long[] ClassHist = new long[3];
            Image SoftMask = null;

            if (AllPathsPick.Any())
            {
                Random Rng = new Random(123);

                Console.WriteLine("Loading picking examples...");
                for (int icorpus = 0; icorpus < AllPathsPick.Count; icorpus++)
                {
                    foreach (var examplePath in AllPathsPick[icorpus])
                    {
                        Image ExampleImage = Image.FromFile(examplePath);
                        int N = ExampleImage.Dims.Z / 3;

                        if (ExampleImage.Dims.Z % 3 != 0)
                            throw new Exception($"Image {examplePath} has {ExampleImage.Dims.Z} layers, which is not a multiple of 3");

                        Image OnlyImages = new Image(Helper.ArrayOfSequence(0, N, 1).Select(i => ExampleImage.GetHost(Intent.Read)[i * 3]).ToArray(), new int3(ExampleImage.Dims.X, ExampleImage.Dims.Y, N));
                        int2 DimsOri = new int2(OnlyImages.Dims);

                        if (SoftMask == null || SoftMask.Dims != new int3(OnlyImages.Dims.X * 2, OnlyImages.Dims.Y * 2, 1))
                        {
                            SoftMask?.Dispose();

                            SoftMask = new Image(new int3(OnlyImages.Dims.X * 2, OnlyImages.Dims.Y * 2, 1));
                            SoftMask.Fill(1f);
                            SoftMask.MaskRectangularly(OnlyImages.Dims.Slice(), Math.Min(OnlyImages.Dims.X, OnlyImages.Dims.Y) / 2f, false);
                        }

                        float2[] MedianPercentileOffset;
                        int[] CountsPerPixel = Helper.ArrayOfFunction(i => (int)((20 + Rng.Next(20)) * BoxNetMM.PixelSize * BoxNetMM.PixelSize), N);
                        float[] PercentileOffset = Helper.ArrayOfFunction(i => CountsPerPixel[i] * (float)(0.02 + Rng.NextDouble() * 0.04), N);
                        Image FakeOdd = new Image(OnlyImages.Dims);
                        Image FakeEven = new Image(OnlyImages.Dims);
                        {
                            Image ImagesCenter = OnlyImages.AsPadded(DimsOri / 2);
                            MedianPercentileOffset = ImagesCenter.GetHost(Intent.Read).Select(a => MathHelper.MedianAndPercentileDiff(a, 68)).ToArray();
                            ImagesCenter.Dispose();

                            OnlyImages.TransformValues((x, y, z, v) => ((v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y * PercentileOffset[z]) + CountsPerPixel[z]);

                            int[] Seeds = Helper.ArrayOfFunction(i => Rng.Next(), N);
                            Parallel.For(0, N, z =>
                            {
                                Random RngZ = new Random(Seeds[z]);
                                float[] OriData = OnlyImages.GetHost(Intent.Read)[z];
                                float[] OddData = FakeOdd.GetHost(Intent.ReadWrite)[z];
                                float[] EvenData = FakeEven.GetHost(Intent.ReadWrite)[z];

                                for (int i = 0; i < OriData.Length; i++)
                                {
                                    int Counts = (int)OriData[i];
                                    int FakeOdd = MathHelper.DrawBinomial(Counts, 0.5f, RngZ);
                                    int FakeEven = Counts - FakeOdd;

                                    OddData[i] = FakeOdd * 2;
                                    EvenData[i] = FakeEven * 2;
                                }
                            });
                        }

                        OnlyImages = OnlyImages.AsPaddedClampedSoft(new int2(OnlyImages.Dims) * 2, 8).AndDisposeParent();
                        OnlyImages.MultiplySlices(SoftMask);
                        OnlyImages.Bandpass(2f * 8f / 1000f, 1, false, 2f * 8f / 1000f);

                        OnlyImages = OnlyImages.AsPadded(DimsOri).AndDisposeParent();
                        {
                            Image ImagesCenter = OnlyImages.AsPadded((DimsOri + 2) / 4 * 2);
                            MedianPercentileOffset = ImagesCenter.GetHost(Intent.Read).Select(a => MathHelper.MedianAndPercentileDiff(a, 68)).ToArray();
                            ImagesCenter.Dispose();

                            OnlyImages.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);
                        }

                        {
                            FakeOdd = FakeOdd.AsPaddedClampedSoft(new int2(OnlyImages.Dims) * 2, 8).AndDisposeParent();
                            FakeOdd.MultiplySlices(SoftMask);
                            FakeOdd.Bandpass(2f * 8f / 1000f, 1, false, 2f * 8f / 1000f);
                            FakeOdd = FakeOdd.AsPadded(DimsOri).AndDisposeParent();

                            FakeOdd.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);
                        }

                        {
                            FakeEven = FakeEven.AsPaddedClampedSoft(new int2(OnlyImages.Dims) * 2, 8).AndDisposeParent();
                            FakeEven.MultiplySlices(SoftMask);
                            FakeEven.Bandpass(2f * 8f / 1000f, 1, false, 2f * 8f / 1000f);
                            FakeEven = FakeEven.AsPadded(DimsOri).AndDisposeParent();

                            FakeEven.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);
                        }

                        for (int n = 0; n < N; n++)
                        {
                            ulong TextureOdd = 0, TextureEven = 0;

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(FakeOdd.GetDeviceSlice(n, Intent.Read), new int2(FakeOdd.Dims), h_Texture, h_Array, false);

                                //AllMicrographsOddDenoise[icorpus].Add(h_Texture[0]);
                                TextureOdd = h_Texture[0];
                            }

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(FakeEven.GetDeviceSlice(n, Intent.Read), new int2(FakeEven.Dims), h_Texture, h_Array, false);

                                //AllMicrographsEvenDenoise[icorpus].Add(h_Texture[0]);
                                TextureEven = h_Texture[0];
                            }

                            ExamplesDenoise[icorpus].Add(new ExampleDenoise(TextureOdd, 
                                                                            TextureEven, 
                                                                            new int2(ExampleImage.Dims)));

                            //AllDimsDenoise[icorpus].Add(new int2(FakeOdd.Dims));
                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(OnlyImages.GetDeviceSlice(n, Intent.Read), new int2(OnlyImages.Dims), h_Texture, h_Array, false);

                                AllMicrographsPick[icorpus].Add([h_Texture[0], TextureOdd, TextureEven]);

                                AllDimsPick[icorpus].Add(new int2(ExampleImage.Dims));

                                float[] Labels = ExampleImage.GetHost(Intent.ReadWrite)[n * 3 + 1];
                                for (int i = 0; i < Labels.Length; i++)
                                {
                                    int Label = (int)Labels[i];
                                    if (CLI.NoMask && Label == 2)
                                    {
                                        Label = 0;
                                        Labels[i] = 0;
                                    }
                                    ClassHist[Label]++;
                                }

                                AllLabelsPick[icorpus].Add(new Image(Labels, ExampleImage.Dims.Slice()));
                            }
                        }

                        ExampleImage.Dispose();
                        OnlyImages.Dispose();
                        FakeOdd.Dispose();
                        FakeEven.Dispose();

                        DimsLargest.X = Math.Max(DimsLargest.X, ExampleImage.Dims.X);
                        DimsLargest.Y = Math.Max(DimsLargest.Y, ExampleImage.Dims.Y);

                        GPU.CheckGPUExceptions();

                        Console.WriteLine($"Loaded {Helper.PathToNameWithExtension(examplePath)}");
                    }
                }

                for (int i = 0; i < AllLabelsPick.Length; i++)
                    foreach (var mic in AllLabelsPick[i])
                        mic.GetDevice(Intent.Read);

                SoftMask.Dispose();

                {
                    if (ClassHist[1] > 0)
                    {
                        //LabelWeights[0] = Math.Min((float)Math.Pow((float)ClassHist[1] / ClassHist[0], 1 / 3.0) * 0.5f, 1);
                        //LabelWeights[2] = 1;//(float)Math.Sqrt((float)ClassHist[1] / ClassHist[2]);
                        double Beta = 0.9;
                        double HistNorm0 = (double)ClassHist[0] / ClassHist[1];
                        double HistNorm2 = (double)ClassHist[2] / ClassHist[1];

                        LabelWeightsPick[0] = (float)((1 - Beta) / (1 - Math.Pow(Beta, HistNorm0)));
                        LabelWeightsPick[1] = 1;
                        LabelWeightsPick[2] = Math.Min(1, (float)(ClassHist[2] == 0 ? 1 : (1 - Beta) / (1 - Math.Pow(Beta, HistNorm2))));
                    }
                    else
                    {
                        //LabelWeights[0] = (float)Math.Pow((float)ClassHist[2] / ClassHist[0], 1 / 3.0);
                        double Beta = 0.9;
                        double HistNorm0 = (double)ClassHist[0] / ClassHist[1];

                        LabelWeightsPick[0] = (float)((1 - Beta) / (1 - Math.Pow(Beta, HistNorm0)));
                        LabelWeightsPick[1] = 1;
                        LabelWeightsPick[2] = 1;
                    }
                }
                Console.WriteLine(" Done");

                Console.WriteLine($"{string.Join(", ", AllMicrographsPick.Select(l => l.Count))} examples for picking");
                Console.WriteLine($"Class histogram: {ClassHist[0]:E1} background, {ClassHist[1]:E1} particle, {ClassHist[2]:E1} mask");
                Console.WriteLine($"Using class weights: {LabelWeightsPick[0]:F3} background, {LabelWeightsPick[1]:F3} particle, {LabelWeightsPick[2]:F3} mask");
            }

            if (AllPathsDenoise.Any())
            {
                Console.WriteLine("Loading denoising examples...");
                for (int icorpus = 0; icorpus < AllPathsDenoise.Count; icorpus++)
                {
                    foreach (var examplePath in AllPathsDenoise[icorpus].Skip(0))
                    {
                        Image TriplesStack = Image.FromFile(examplePath);
                        int N = TriplesStack.Dims.Z / 3;
                        int2 DimsOri = new int2(TriplesStack.Dims);

                        Image Odds = new Image(Helper.ArrayOfSequence(0, N, 1).Select(i => TriplesStack.GetHost(Intent.Read)[i * 3]).ToArray(), 
                                               new int3(TriplesStack.Dims.X, TriplesStack.Dims.Y, N));
                        Image Evens = new Image(Helper.ArrayOfSequence(0, N, 1).Select(i => TriplesStack.GetHost(Intent.Read)[i * 3 + 1]).ToArray(),
                                                new int3(TriplesStack.Dims.X, TriplesStack.Dims.Y, N));
                        Image Spectrals = new Image(Helper.ArrayOfSequence(0, N, 1).Select(i => TriplesStack.GetHost(Intent.Read)[i * 3 + 2]).ToArray(),
                                                    new int3(TriplesStack.Dims.X, TriplesStack.Dims.Y, N));

                        float2[] MedianPercentileOffset;
                        {
                            Image CombinedCenter = Odds.GetCopyGPU();
                            CombinedCenter.Add(Evens);
                            CombinedCenter.Multiply(0.5f);
                            CombinedCenter = CombinedCenter.AsPadded(new int2(Odds.Dims + 2) / 4 * 2).AndDisposeParent();
                            MedianPercentileOffset = CombinedCenter.GetHost(Intent.Read).Select(a => MathHelper.MedianAndPercentileDiff(a, 68)).ToArray();
                            CombinedCenter.Dispose();
                        }

                        #region Calculate Wiener filter

                        Image CTFsScaled;
                        {
                            Image CTFs = new Image(Spectrals.Dims, true, false);
                            for (int z = 0; z < Spectrals.Dims.Z; z++)
                            {
                                float[] SpectralsData = Spectrals.GetHost(Intent.Read)[z];
                                float[] CTFsData = CTFs.GetHost(Intent.ReadWrite)[z];

                                for (int y = 0; y < Spectrals.Dims.Y; y++)
                                    for (int x = 0; x < Spectrals.Dims.X / 2 + 1; x++)
                                        CTFsData[y * (CTFs.Dims.X / 2 + 1) + x] = SpectralsData[y * Spectrals.Dims.X + Math.Min(Spectrals.Dims.X / 2 - 1, x)];
                            }

                            CTFsScaled = CTFs.AsComplex().AndDisposeParent()
                                             .AsIFFT().AndDisposeParent()
                                             .AsPadded(new int2(CTFs.Dims) * 2, true).AndDisposeParent()
                                             .AsFFT().AndDisposeParent()
                                             .AsReal().AndDisposeParent();
                            CTFsScaled.Multiply(1f / Odds.ElementsSliceReal);
                        }

                        for (int z = 0; z < CTFsScaled.Dims.Z; z++)
                        {
                            float[] CTFsData = CTFsScaled.GetHost(Intent.ReadWrite)[z];

                            for (int i = 0; i < CTFsData.Length; i++)
                            {
                                float CTF = CTFsData[i];

                                CTFsData[i] = i == 0 ? 1 : CTF / (CTF * CTF + 0.01f);
                                //CTFsData[i] = CTF / (CTF * CTF + Math.Max(0.02f, 1 / Math.Max(1e-6f, SNR)));
                            }
                        }

                        Directory.CreateDirectory("debug");
                        CTFsScaled.WriteMRC16b($"debug/{Helper.PathToName(examplePath)}_wiener.mrc");

                        #endregion

                        if (SoftMask == null || SoftMask.Dims != new int3(Odds.Dims.X * 2, Odds.Dims.Y * 2, 1))
                        {
                            SoftMask?.Dispose();

                            SoftMask = new Image(new int3(Odds.Dims.X * 2, Odds.Dims.Y * 2, 1));
                            SoftMask.Fill(1f);
                            SoftMask.MaskRectangularly(Odds.Dims.Slice(), Math.Min(Odds.Dims.X, Odds.Dims.Y) / 2f, false);
                        }

                        #region Normalize and high-pass filter
                        {
                            Odds.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);

                            Image OddsPadded = Odds.AsPaddedClampedSoft(new int2(Odds.Dims) * 2, 8).AndDisposeParent();
                            OddsPadded.MultiplySlices(SoftMask);
                            OddsPadded.Bandpass(2f * 8f / 1000f, 2, false, 2f * 8f / 1000f);
                            Odds = OddsPadded.AsPadded(DimsOri).AndDisposeParent();
                        }
                        {
                            Evens.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);

                            Image EvensPadded = Evens.AsPaddedClampedSoft(new int2(Odds.Dims) * 2, 8).AndDisposeParent();
                            EvensPadded.MultiplySlices(SoftMask);
                            EvensPadded.Bandpass(2f * 8f / 1000f, 2, false, 2f * 8f / 1000f);
                            Evens = EvensPadded.AsPadded(DimsOri);
                        }
                        #endregion

                        #region Re-normalize after filtering
                        {
                            Image CombinedCenter = Odds.GetCopyGPU();
                            CombinedCenter.Add(Evens);
                            CombinedCenter.Multiply(0.5f);
                            CombinedCenter = CombinedCenter.AsPadded(new int2(Odds.Dims + 2) / 4 * 2).AndDisposeParent();
                            MedianPercentileOffset = CombinedCenter.GetHost(Intent.Read).Select(a => MathHelper.MedianAndPercentileDiff(a, 68)).ToArray();
                            CombinedCenter.Dispose();

                            Odds.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);
                            Evens.TransformValues((x, y, z, v) => (v - MedianPercentileOffset[z].X) / MedianPercentileOffset[z].Y);
                        }
                        #endregion

                        #region Deconvolve odd and even

                        Image OddsDeconv, EvensDeconv;
                        {
                            Image OddsPadded = Odds.AsPaddedClampedSoft(new int2(Odds.Dims) * 2, 8);
                            OddsPadded.MultiplySlices(SoftMask);

                            Image OddsPaddedFT = OddsPadded.AsFFT().AndDisposeParent();
                            OddsPaddedFT.Multiply(CTFsScaled);
                            OddsPadded = OddsPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();

                            OddsDeconv = OddsPadded.AsPadded(DimsOri).AndDisposeParent();
                        }
                        {
                            Image EvensPadded = Evens.AsPaddedClampedSoft(new int2(Odds.Dims) * 2, 8);
                            EvensPadded.MultiplySlices(SoftMask);
                            EvensPadded.Bandpass(2f * 8f / 1000f, 2, false, 2f * 8f / 1000f);
                            Evens = EvensPadded.AsPadded(DimsOri);

                            Image EvensPaddedFT = EvensPadded.AsFFT().AndDisposeParent();
                            EvensPaddedFT.Multiply(CTFsScaled);
                            EvensPadded = EvensPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();

                            EvensDeconv = EvensPadded.AsPadded(DimsOri).AndDisposeParent();
                        }

                        CTFsScaled.Dispose();

                        #endregion

                        #region Re-normalize after deconv
                        {
                            Image CombinedCenter = OddsDeconv.GetCopyGPU();
                            CombinedCenter.Add(EvensDeconv);
                            CombinedCenter.Multiply(0.5f);
                            CombinedCenter = CombinedCenter.AsPadded(new int2(Odds.Dims + 2) / 4 * 2).AndDisposeParent();
                            MedianPercentileOffset = CombinedCenter.GetHost(Intent.Read).Select(a => MathHelper.MedianAndPercentileDiff(a, 68)).ToArray();
                            CombinedCenter.Dispose();

                            OddsDeconv.TransformValues((x, y, z, v) => v / MedianPercentileOffset[z].Y);
                            EvensDeconv.TransformValues((x, y, z, v) => v / MedianPercentileOffset[z].Y);
                        }
                        #endregion

                        #region Convert to textures


                        for (int n = 0; n < N; n++)
                        {
                            ExampleDeconv Example = new();
                            Example.Dims = new int2(Odds.Dims);

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(Odds.GetDeviceSlice(n, Intent.Read), new int2(Odds.Dims), h_Texture, h_Array, false);

                                Example.t_Odd = h_Texture[0];
                            }

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(Evens.GetDeviceSlice(n, Intent.Read), new int2(Evens.Dims), h_Texture, h_Array, false);

                                Example.t_Even = h_Texture[0];
                            }

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(OddsDeconv.GetDeviceSlice(n, Intent.Read), new int2(Odds.Dims), h_Texture, h_Array, false);

                                Example.t_OddDeconv = h_Texture[0];
                            }

                            {
                                ulong[] h_Array = new ulong[1];
                                ulong[] h_Texture = new ulong[1];
                                GPU.CreateTexture2D(EvensDeconv.GetDeviceSlice(n, Intent.Read), new int2(Evens.Dims), h_Texture, h_Array, false);

                                Example.t_EvenDeconv = h_Texture[0];
                            }

                            ExamplesDeconv[icorpus].Add(Example);
                        }

                        #endregion

                        if (true)
                        {
                            Directory.CreateDirectory("debug");
                            Odds.WriteMRC16b($"debug/{Helper.PathToName(examplePath)}_odd.mrc");
                            Evens.WriteMRC16b($"debug/{Helper.PathToName(examplePath)}_even.mrc");
                            OddsDeconv.WriteMRC16b($"debug/{Helper.PathToName(examplePath)}_odd_deconv.mrc");
                            EvensDeconv.WriteMRC16b($"debug/{Helper.PathToName(examplePath)}_even_deconv.mrc");
                        }

                        OddsDeconv.Dispose();
                        EvensDeconv.Dispose();
                        Odds.Dispose();
                        Evens.Dispose();
                        TriplesStack.Dispose();
                        Spectrals.Dispose();
                        GPU.CheckGPUExceptions();

                        Console.WriteLine($"Loaded {Helper.PathToNameWithExtension(examplePath)}");
                    }
                }
                Console.WriteLine(" Done");

                Console.WriteLine($"{string.Join(", ", ExamplesDeconv.Select(l => l.Count))} examples for denoising");
            }

            #endregion

            #region Load model

            Console.Write("Loading model...");

            BoxNetMM NetworkTrain = new BoxNetMM(new int2(CLI.PatchSize), LabelWeightsPick, CLI.DevicesModel.ToArray(), CLI.BatchSize, BoxNetOptimizer.Adam);
            if (CLI.ModelIn != null)
                NetworkTrain.Load(CLI.ModelIn);

            Console.WriteLine(" Done");

            #endregion

            #region Training

            Console.WriteLine("Started training");

            int NThreads = 1;

            int2 DimsAugmented = new int2(CLI.PatchSize);
            int Border = CLI.PatchSize / 8;
            int BatchSize = NetworkTrain.BatchSize;
            int PlotEveryN = 10;
            int SmoothN = 100;

            Queue<float>[] LastLossesPick = { new Queue<float>(SmoothN), new Queue<float>(SmoothN) };
            Queue<float>[] CheckpointLossesPick = { new Queue<float>(), new Queue<float>() };
            Queue<float>[] LastLossesDenoise = { new Queue<float>(SmoothN), new Queue<float>(SmoothN) };
            Queue<float>[] CheckpointLossesDenoise = { new Queue<float>(), new Queue<float>() };
            Queue<float>[] LastLossesFill = { new Queue<float>(SmoothN), new Queue<float>(SmoothN) };
            Queue<float>[] CheckpointLossesFill = { new Queue<float>(), new Queue<float>() };

            GPU.SetDevice(CLI.DeviceData);

            Image[] d_AugmentedDataPick = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedLabelsPick = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize * 3)), NThreads);
            IntPtr[] d_AugmentedWeightsPick = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsAugmented.Elements() * BatchSize), NThreads);

            Image[] d_AugmentedOddDenoise = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedEvenDenoise = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedOddDenoiseDeconv = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedEvenDenoiseDeconv = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            //Image[] d_AugmentedMaskDenoise = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);

            Image[] d_AugmentedFillSource = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedFillTarget = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Random[] RG = Helper.ArrayOfFunction(i => new Random(i), NThreads);
            RandomNormal[] RGN = Helper.ArrayOfFunction(i => new RandomNormal(i), NThreads);

            int MaxExamples = 0;
            if (AllMicrographsPick.Any())
                MaxExamples = Math.Max(MaxExamples, AllMicrographsPick[0].Count);
            if (ExamplesDeconv.Any())
                MaxExamples = Math.Max(MaxExamples, ExamplesDeconv[0].Count);
            
            Stopwatch CheckpointWatch = new Stopwatch();
            CheckpointWatch.Start();
            int NCheckpoints = 0;

            Action<int, int, int, bool, Random> MakeExamplesPick = (threadID, icorpus, subBatch, doAugment, rng) =>
            {
                for (int ib = 0; ib < BatchSize; ib += subBatch)
                {
                    int CurBatch = Math.Min(subBatch, BatchSize - ib);

                    int ExampleID = rng.Next(AllMicrographsPick[icorpus].Count);
                    int2 Dims = AllDimsPick[icorpus][ExampleID];

                    int2 DimsValid = new int2(Math.Max(0, Dims.X - Border * 4), Math.Max(0, Dims.Y - Border * 4));
                    int2 DimsBorder = (Dims - DimsValid) / 2;

                    float2[] Translations = doAugment ? Helper.ArrayOfFunction(x => new float2(rng.Next(DimsValid.X) + DimsBorder.X,
                                                                                    rng.Next(DimsValid.Y) + DimsBorder.Y), CurBatch) :
                                                        Helper.ArrayOfFunction(x => new float2(DimsValid / 2 + DimsBorder), CurBatch);

                    float[] Rotations = doAugment ? Helper.ArrayOfFunction(i => (float)(rng.NextDouble() * Math.PI * 2), CurBatch) :
                                                    Helper.ArrayOfFunction(i => 0f, CurBatch);
                    float3[] Scales = doAugment ? Helper.ArrayOfFunction(i => new float3(0.8f + (float)rng.NextDouble() * 0.4f,
                                                                                         0.8f + (float)rng.NextDouble() * 0.4f,
                                                                                         (float)(rng.NextDouble() * Math.PI * 2)), CurBatch) :
                                                  Helper.ArrayOfFunction(i => new float3(1, 1, 0), CurBatch);
                    float OffsetMean = 0;// doAugment ? rng.NextSingle() * 0.4f - 0.2f : 0;
                    float OffsetScale = 1;// doAugment ? rng.NextSingle() * 0.4f + 0.8f : 1;

                    GPU.BoxNetMMAugmentPicking(AllMicrographsPick[icorpus][ExampleID][doAugment ? rng.Next(3) : 0],
                                               AllLabelsPick[icorpus][ExampleID].GetDevice(Intent.Read),
                                               Dims,
                                               d_AugmentedDataPick[threadID].GetDeviceSlice(ib, Intent.Write),
                                               d_AugmentedLabelsPick[threadID].GetDeviceSlice(ib * 3, Intent.Write),
                                               DimsAugmented,
                                               Helper.ToInterleaved(Translations),
                                               Rotations,
                                               Helper.ToInterleaved(Scales),
                                               OffsetMean,
                                               OffsetScale,
                                               rng.Next(99999),
                                               false,
                                               (uint)CurBatch);
                }

                Image AugmentedDenoised;
                NetworkTrain.PredictDenoiseDeconv(d_AugmentedDataPick[threadID], out AugmentedDenoised);
                GPU.CopyDeviceToDevice(AugmentedDenoised.GetDevice(Intent.Read), 
                                       d_AugmentedDataPick[threadID].GetDevice(Intent.Write), 
                                       d_AugmentedDataPick[threadID].ElementsReal);

                if (doAugment)
                {
                    d_AugmentedDataPick[threadID].Multiply(Helper.ArrayOfFunction(i => rng.NextSingle() * 0.4f + 0.8f, d_AugmentedDataPick[threadID].Dims.Z));
                    d_AugmentedDataPick[threadID].Add(rng.NextSingle() * 0.4f - 0.2f);
                }
            };

            Action<int, int, int, bool, bool, Random> MakeExamplesDenoise = (threadID, icorpus, subBatch, useDeconv, doAugment, rng) =>
            {
                for (int ib = 0; ib < BatchSize; ib += subBatch)
                {
                    int CurBatch = Math.Min(subBatch, BatchSize - ib);

                    int ExampleID = rng.Next(useDeconv ? ExamplesDeconv[icorpus].Count : 
                                                         ExamplesDenoise[icorpus].Count);
                    int2 Dims = useDeconv ? ExamplesDeconv[icorpus][ExampleID].Dims :
                                            ExamplesDenoise[icorpus][ExampleID].Dims;

                    int2 DimsValid = new int2(Math.Max(0, Dims.X - Border * 4), Math.Max(0, Dims.Y - Border * 4));
                    int2 DimsBorder = (Dims - DimsValid) / 2;

                    float2[] Translations = doAugment ? Helper.ArrayOfFunction(x => new float2(rng.Next(DimsValid.X) + DimsBorder.X,
                                                                                    rng.Next(DimsValid.Y) + DimsBorder.Y), CurBatch) :
                                                        Helper.ArrayOfFunction(x => new float2(DimsValid / 2 + DimsBorder), CurBatch);

                    float[] Rotations = doAugment ? Helper.ArrayOfFunction(i => (float)(rng.NextDouble() * Math.PI * 2), CurBatch) :
                                                    Helper.ArrayOfFunction(i => 0f, CurBatch);
                    float3[] Scales = doAugment ? Helper.ArrayOfFunction(i => new float3(0.8f + (float)rng.NextDouble() * 0.4f,
                                                                                         0.8f + (float)rng.NextDouble() * 0.4f,
                                                                                         (float)(rng.NextDouble() * Math.PI * 2)), CurBatch) :
                                                  Helper.ArrayOfFunction(i => new float3(1, 1, 0), CurBatch);
                    float StdDev = doAugment ? (float)Math.Abs(RGN[threadID].NextSingle(0, 0.3f)) : 0;
                    float OffsetMean = doAugment ? rng.NextSingle() * 0.1f - 0.2f : 0;
                    float OffsetScale = doAugment ? rng.NextSingle() * 0.2f + 0.9f : 1;

                    bool SwapOddEven = rng.Next(2) == 0;
                    ulong t_Odd, t_Even, t_OddDeconv, t_EvenDeconv;
                    if (useDeconv)
                    {
                        t_Odd = SwapOddEven ? ExamplesDeconv[icorpus][ExampleID].t_Even : ExamplesDeconv[icorpus][ExampleID].t_Odd;
                        t_Even = SwapOddEven ? ExamplesDeconv[icorpus][ExampleID].t_Odd : ExamplesDeconv[icorpus][ExampleID].t_Even;
                        t_OddDeconv = SwapOddEven ? ExamplesDeconv[icorpus][ExampleID].t_EvenDeconv : ExamplesDeconv[icorpus][ExampleID].t_OddDeconv;
                        t_EvenDeconv = SwapOddEven ? ExamplesDeconv[icorpus][ExampleID].t_OddDeconv : ExamplesDeconv[icorpus][ExampleID].t_EvenDeconv;
                    }
                    else
                    {
                        t_Odd = SwapOddEven ? ExamplesDenoise[icorpus][ExampleID].t_Even : ExamplesDenoise[icorpus][ExampleID].t_Odd;
                        t_Even = SwapOddEven ? ExamplesDenoise[icorpus][ExampleID].t_Odd : ExamplesDenoise[icorpus][ExampleID].t_Even;
                        t_OddDeconv = SwapOddEven ? ExamplesDenoise[icorpus][ExampleID].t_Even : ExamplesDenoise[icorpus][ExampleID].t_Odd;
                        t_EvenDeconv = SwapOddEven ? ExamplesDenoise[icorpus][ExampleID].t_Odd : ExamplesDenoise[icorpus][ExampleID].t_Even;
                    }

                    GPU.BoxNetMMAugmentDenoising(t_Odd,
                                                t_Even,
                                                t_OddDeconv,
                                                t_EvenDeconv,
                                                Dims,
                                                d_AugmentedOddDenoise[threadID].GetDeviceSlice(ib, Intent.Write),
                                                d_AugmentedEvenDenoise[threadID].GetDeviceSlice(ib, Intent.Write),
                                                d_AugmentedOddDenoiseDeconv[threadID].GetDeviceSlice(ib, Intent.Write),
                                                d_AugmentedEvenDenoiseDeconv[threadID].GetDeviceSlice(ib, Intent.Write),
                                                DimsAugmented,
                                                Helper.ToInterleaved(Translations),
                                                Rotations,
                                                Helper.ToInterleaved(Scales),
                                                OffsetMean,
                                                OffsetScale,
                                                (uint)CurBatch);
                }

                GPU.CopyDeviceToDevice(d_AugmentedOddDenoise[threadID].GetDevice(Intent.Read), 
                                       d_AugmentedFillSource[threadID].GetDevice(Intent.Write), 
                                       d_AugmentedOddDenoise[threadID].ElementsReal);
                d_AugmentedFillSource[threadID].Add(d_AugmentedEvenDenoise[threadID]);
                d_AugmentedFillSource[threadID].Multiply(0.5f);

                NetworkTrain.PredictDenoiseDeconv(d_AugmentedFillSource[threadID], out Image AugmentedDenoised);

                GPU.CopyDeviceToDevice(AugmentedDenoised.GetDevice(Intent.Read),
                                       d_AugmentedFillSource[threadID].GetDevice(Intent.Write),
                                       d_AugmentedFillSource[threadID].ElementsReal);

                GPU.CopyDeviceToDevice(d_AugmentedFillSource[threadID].GetDevice(Intent.Read), 
                                       d_AugmentedFillTarget[threadID].GetDevice(Intent.Write), 
                                       d_AugmentedFillSource[threadID].ElementsReal);

                RandomNormal RandN = new RandomNormal(rng.Next());

                for (int z = 0; z < d_AugmentedFillSource[threadID].Dims.Z; z++)
                {
                    float[] AugmentedData = d_AugmentedFillSource[threadID].GetHost(Intent.ReadWrite)[z];
                    int NRects = 20 + rng.Next(80);

                    for (int irect = 0; irect < NRects; irect++)
                    {
                        int2 DimsRect = new int2(rng.Next(6) + 6, rng.Next(6) + 6);
                        int2 DimsCalc = DimsRect * 1;
                        int2 PosRect = new int2(rng.Next(DimsAugmented.X - DimsCalc.X), rng.Next(DimsAugmented.Y - DimsCalc.Y));
                        double Sum = 0, Sum2 = 0;

                        for (int y = 0; y < DimsCalc.Y; y++)
                            for (int x = 0; x < DimsCalc.X; x++)
                            {
                                float Val = AugmentedData[(PosRect.Y + y) * DimsAugmented.X + PosRect.X + x];
                                Sum += Val;
                                Sum2 += Val * Val;
                            }

                        float Mean = (float)(Sum / DimsCalc.Elements());
                        float StdDev = (float)Math.Sqrt(Math.Max(0, Sum2 * DimsCalc.Elements() - Sum * Sum)) / DimsCalc.Elements();

                        for (int y = 0; y < DimsRect.Y; y++)
                            for (int x = 0; x < DimsRect.X; x++)
                                AugmentedData[(PosRect.Y + (DimsCalc.Y - DimsRect.Y) / 2 + y) * DimsAugmented.X + PosRect.X + (DimsCalc.X - DimsRect.X) / 2 + x] = 0;// RandN.NextSingle(Mean, StdDev);
                    }
                }

                //int Grid = 4;
                //for (int z = 0; z < d_AugmentedOddDenoise[threadID].Dims.Z; z++)
                //{
                //    float[] OddData = d_AugmentedOddDenoise[threadID].GetHost(Intent.ReadWrite)[z];
                //    float[] EvenData = d_AugmentedEvenDenoise[threadID].GetHost(Intent.ReadWrite)[z];
                //    float[] MaskData = d_AugmentedMaskDenoise[threadID].GetHost(Intent.ReadWrite)[z];

                //    int OffsetX = rng.Next(Grid);
                //    int OffsetY = rng.Next(Grid);

                //    bool PureN2N = true;// rng.Next(10) == 0;

                //    for (int i = 0; i < OddData.Length; i++)
                //    {
                //        int X = i % DimsAugmented.X;
                //        int Y = i / DimsAugmented.X;

                //        float Odd = OddData[i];
                //        float Even = EvenData[i];
                //        bool Mask = PureN2N || ((X + OffsetX) % Grid == 0 && (Y + OffsetY) % Grid == 0);

                //        EvenData[i] = Mask ? Even : (Odd + Even) * 0.5f;
                //        OddData[i] = Mask ? Odd : (Odd + Even) * 0.5f;
                //        MaskData[i] = Mask ? 1 : 0;
                //    }
                //}
            };

            int NIterations = MaxExamples * CLI.NEpochs;
            int NDone = 0;
            Helper.ForCPUGreedy(0, NIterations, NThreads,

                threadID => GPU.SetDevice(CLI.DeviceData),

                (b, threadID) =>
                {

                    float LearningRate = MathHelper.Lerp((float)CLI.LearningRateStart, (float)CLI.LearningRateEnd, (float)NDone / NIterations);
                    LearningRate = MathF.Min(LearningRate, MathHelper.Lerp(0, (float)CLI.LearningRateStart, (float)NDone / 100));   // Warm-up

                    float LearningRateDenoise = CLI.LearningRateDenoise == null ? LearningRate : (float)CLI.LearningRateDenoise.Value;
                    LearningRateDenoise = MathF.Min(LearningRateDenoise, MathHelper.Lerp(0, (float)CLI.LearningRateDenoise.Value, (float)NDone / 100));   // Warm-up


                    if (AllMicrographsPick.Any() && LearningRate > 0)
                    {
                        int icorpus = b % AllMicrographsPick.Length;
                        float[] Loss;

                        MakeExamplesPick(threadID, icorpus, 8, true, RG[threadID]);

                        lock (NetworkTrain)
                            NetworkTrain.TrainPick(d_AugmentedDataPick[threadID],
                                                   d_AugmentedLabelsPick[threadID],
                                                   LearningRate,
                                                   false,
                                                   false,
                                                   out _,
                                                   out Loss);

                        if (float.IsNaN(Loss[0]))
                            throw new Exception("Something went wrong with picker training because loss = NaN");

                        lock (Watch) 
                        { 
                            LastLossesPick[icorpus].Enqueue(Loss[0]);
                            if (LastLossesPick[icorpus].Count > SmoothN)
                                LastLossesPick[icorpus].Dequeue();
                            CheckpointLossesPick[icorpus].Enqueue(Loss[0]);
                        }
                    }

                    if (ExamplesDeconv.Any())
                    {
                        int icorpus = b % ExamplesDeconv.Length;
                        bool UseDeconv = ExamplesDeconv[icorpus].Any() && RG[threadID].Next(2) == 0;

                        float[] LossDenoise;
                        float[] LossFill;

                        MakeExamplesDenoise(threadID, icorpus, 8, UseDeconv, true, RG[threadID]);

                        if (LearningRateDenoise > 0)
                        {
                            lock (NetworkTrain)
                                NetworkTrain.TrainDenoise(d_AugmentedOddDenoise[threadID],
                                                          d_AugmentedEvenDenoise[threadID],
                                                          UseDeconv ? d_AugmentedEvenDenoiseDeconv[threadID] : null,
                                                          LearningRateDenoise,
                                                          false,
                                                          out _,
                                                          out _,
                                                          out LossDenoise);

                            if (float.IsNaN(LossDenoise[0]))
                                throw new Exception("Something went wrong with denoiser training because loss = NaN");

                            lock (Watch)
                            {
                                LastLossesDenoise[icorpus].Enqueue(LossDenoise[0]);
                                if (LastLossesDenoise[icorpus].Count > SmoothN)
                                    LastLossesDenoise[icorpus].Dequeue();
                                CheckpointLossesDenoise[icorpus].Enqueue(LossDenoise[0]);
                            }
                        }

                        if (LearningRate > 0)
                        {
                            lock (NetworkTrain)
                                NetworkTrain.TrainFill(d_AugmentedFillSource[threadID],
                                                       d_AugmentedFillTarget[threadID],
                                                       LearningRate,
                                                       false,
                                                       false,
                                                       out _,
                                                       out LossFill);

                            if (float.IsNaN(LossFill[0]))
                                throw new Exception("Something went wrong with filler training because loss = NaN");

                            lock (Watch)
                            {
                                LastLossesFill[icorpus].Enqueue(LossFill[0]);
                                if (LastLossesFill[icorpus].Count > SmoothN)
                                    LastLossesFill[icorpus].Dequeue();
                                CheckpointLossesFill[icorpus].Enqueue(LossFill[0]);
                            }
                        }
                    }

                    lock (Watch)
                    {
                        NDone++;

                        if (NDone % PlotEveryN == 0)
                        {
                            long Elapsed = Watch.ElapsedMilliseconds;
                            Watch.Restart();
                            double Estimated = (double)Elapsed / PlotEveryN * (NIterations - NDone);
                            int Remaining = (int)Estimated;
                            TimeSpan SpanRemaining = new TimeSpan(0, 0, 0, 0, Remaining);

                            {
                                VirtualConsole.ClearLastLine();
                                string TimeString = SpanRemaining.ToString((int)SpanRemaining.TotalDays > 0 ? @"dd\.hh\:mm\:ss" : ((int)SpanRemaining.TotalHours > 0 ? @"hh\:mm\:ss" : @"mm\:ss"));
                                Console.Write($"{NDone}/{NIterations}, picking = {MathHelper.Mean(LastLossesPick[0]):E2}, denoising = {MathHelper.Mean(LastLossesDenoise[0]):E2}, filling = {MathHelper.Mean(LastLossesFill[0]):E2}, lr = {LearningRate:E2}, {TimeString} remaining");
                            }
                        }

                        if ((CLI.Checkpoints > 0 && CheckpointWatch.Elapsed.TotalMinutes > CLI.Checkpoints) || b == NIterations - 1)
                        {
                            CheckpointWatch.Restart();
                            string CheckpointName = CLI.ModelOut + $".{NCheckpoints:D3}" + ".checkpoint";
                            NetworkTrain.Save(CheckpointName);

                            Random RGPrediction = new(123);

                            lock (NetworkTrain)
                            {
                                if (AllMicrographsPick.Any())
                                {
                                    MakeExamplesPick(threadID, 0, 1, false, new Random(123));

                                    Image Prediction, Probabilities;
                                    NetworkTrain.PredictPick(d_AugmentedDataPick[threadID], out Prediction, out Probabilities);
                                    Image Merged = new Image(Prediction.Dims.MultZ(2));
                                    for (int z = 0; z < Prediction.Dims.Z; z++)
                                    {
                                        float[] Labels = Prediction.GetHost(Intent.Read)[z];
                                        for (int i = 0; i < Labels.Length; i++)
                                        {
                                            int Label = (int)Labels[i];
                                            float Probability = Probabilities.GetHost(Intent.Read)[z * 3 + Label][i];
                                            if (Label == 1 && Probability < 0.5f)
                                                Labels[i] = 0;
                                            if (Label == 2 && Probability < 0.1f)
                                                Labels[i] = 0;
                                        }

                                        Merged.GetHost(Intent.Write)[z * 2 + 0] = d_AugmentedDataPick[threadID].GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 2 + 1] = Labels;
                                    }

                                    Merged.WriteMRC(CLI.ModelOut + $".{NCheckpoints:D3}" + ".pick.checkpoint.mrc", true);
                                    Merged.Dispose();
                                }

                                if (ExamplesDenoise[0].Any())
                                {
                                    MakeExamplesDenoise(threadID, 0, 1, false, false, new Random(123));
                                    d_AugmentedOddDenoise[threadID].Add(d_AugmentedEvenDenoise[threadID]);
                                    d_AugmentedOddDenoise[threadID].Multiply(0.5f);

                                    Image PredictionDenoise, PredictionDeconv;
                                    NetworkTrain.PredictDenoise(d_AugmentedOddDenoise[threadID], out PredictionDenoise);
                                    NetworkTrain.PredictDenoiseDeconv(d_AugmentedOddDenoise[threadID], out PredictionDeconv);
                                    Image Merged = new Image(PredictionDenoise.Dims.MultZ(3));
                                    for (int z = 0; z < PredictionDenoise.Dims.Z; z++)
                                    {
                                        Merged.GetHost(Intent.Write)[z * 3 + 0] = d_AugmentedOddDenoise[threadID].GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 3 + 1] = PredictionDenoise.GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 3 + 2] = PredictionDeconv.GetHost(Intent.Read)[z];
                                    }

                                    Merged.WriteMRC(CLI.ModelOut + $".{NCheckpoints:D3}" + ".denoise.checkpoint.mrc", true);
                                    Merged.Dispose();
                                }

                                if (ExamplesDenoise.Any())
                                {
                                    Image Prediction;
                                    NetworkTrain.PredictFill(d_AugmentedFillSource[threadID], out Prediction);
                                    Image Merged = new Image(Prediction.Dims.MultZ(2));
                                    for (int z = 0; z < Prediction.Dims.Z; z++)
                                    {
                                        Merged.GetHost(Intent.Write)[z * 2 + 0] = d_AugmentedFillSource[threadID].GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 2 + 1] = Prediction.GetHost(Intent.Read)[z];
                                    }

                                    Merged.WriteMRC(CLI.ModelOut + $".{NCheckpoints:D3}" + ".fill.checkpoint.mrc", true);
                                    Merged.Dispose();
                                }

                                if (ExamplesDeconv[0].Any())
                                {
                                    MakeExamplesDenoise(threadID, 0, 1, true, false, new Random(123));
                                    d_AugmentedOddDenoise[threadID].Add(d_AugmentedEvenDenoise[threadID]);
                                    d_AugmentedOddDenoise[threadID].Multiply(0.5f);

                                    Image PredictionDenoise, PredictionDeconv;
                                    NetworkTrain.PredictDenoise(d_AugmentedOddDenoise[threadID], out PredictionDenoise);
                                    NetworkTrain.PredictDenoiseDeconv(d_AugmentedOddDenoise[threadID], out PredictionDeconv);
                                    Image Merged = new Image(PredictionDenoise.Dims.MultZ(3));
                                    for (int z = 0; z < PredictionDenoise.Dims.Z; z++)
                                    {
                                        Merged.GetHost(Intent.Write)[z * 3 + 0] = d_AugmentedOddDenoise[threadID].GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 3 + 1] = PredictionDenoise.GetHost(Intent.Read)[z];
                                        Merged.GetHost(Intent.Write)[z * 3 + 2] = PredictionDeconv.GetHost(Intent.Read)[z];
                                    }

                                    Merged.WriteMRC(CLI.ModelOut + $".{NCheckpoints:D3}" + ".deconv.checkpoint.mrc", true);
                                    Merged.Dispose();
                                }
                            }

                            Console.WriteLine();
                            Console.WriteLine($"Average loss since previous checkpoint: {CheckpointLossesPick[0].Mean():E2} picking, {CheckpointLossesDenoise[0].Mean():E2} denoising, {CheckpointLossesFill[0].Mean():E2} filling");
                            Console.WriteLine($"Saved checkpoint to {CheckpointName}");

                            CheckpointLossesPick[0].Clear();
                            CheckpointLossesPick[1].Clear();
                            CheckpointLossesDenoise[0].Clear();
                            CheckpointLossesDenoise[1].Clear();
                            CheckpointLossesFill[0].Clear();
                            CheckpointLossesFill[1].Clear();
                            NCheckpoints++;
                        }
                    }
                },

                null);
            Console.WriteLine("");

            #endregion

            Console.Write($"Saving model to {CLI.ModelOut}...");

            NetworkTrain.Save(CLI.ModelOut);

            Console.WriteLine(" Done");
        }
    }

    struct ExampleDenoise
    {
        public ulong t_Odd;
        public ulong t_Even;
        public int2 Dims;

        public ExampleDenoise(ulong t_odd, ulong t_even, int2 dims)
        {
            t_Odd = t_odd;
            t_Even = t_even;
            Dims = dims;
        }
    }

    struct ExampleDeconv
    {
        public ulong t_Odd;
        public ulong t_Even;
        public ulong t_OddDeconv;
        public ulong t_EvenDeconv;
        public int2 Dims;

        public ExampleDeconv(ulong t_odd, ulong t_even, ulong t_odd_deconv, ulong t_even_deconv, int2 dims)
        {
            t_Odd = t_odd;
            t_Even = t_even;
            t_OddDeconv = t_odd_deconv;
            t_EvenDeconv = t_even_deconv;
            Dims = dims;
        }
    }
}
