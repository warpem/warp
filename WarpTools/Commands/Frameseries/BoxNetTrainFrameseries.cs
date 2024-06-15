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

        [Option("examples", Required = true, HelpText = "Path to a folder containing TIFF files with examples prepared with boxnet_examples_frameseries")]
        public string ExamplesNew { get; set; }

        [Option("examples_general", HelpText = "Path to a folder containing TIFF files with examples used to train a more general model, which will be mixed 1:1 with new examples to reduce overfitting")]
        public string ExamplesGeneral { get; set; }

        [Option("no_mask", HelpText = "Don't consider mask labels in training; they will be converted to background labels")]
        public bool NoMask { get; set; }

        [Option("patchsize", Default = 512, HelpText = "Size of the BoxNet input window, a multiple of 256; remember to use the same window with boxnet_infer_frameseries")]
        public int PatchSize { get; set; }

        [Option("batchsize", Default = 8, HelpText = "Size of the minibatches used in training; larger batches require more GPU memory; must be divisible by number of devices")]
        public int BatchSize { get; set; }

        [Option("lr_start", Default = 5e-5, HelpText = "Learning rate at training start")]
        public double LearningRateStart { get; set; }

        [Option("lr_end", Default = 1e-5, HelpText = "Learning rate at training end, with linear interpolation in-between")]
        public double LearningRateEnd { get; set; }

        [Option("epochs", Default = 100, HelpText = "Number of training epochs")]
        public int NEpochs { get; set; }

        [Option("checkpoints", Default = 0, HelpText = "Save checkpoints every N minutes; set to 0 to disable")]
        public int Checkpoints { get; set; }

        [Option("devices", HelpText = "Space-separated list of GPU IDs to be used for training")]
        public IEnumerable<int> Devices { get; set; }
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

            if (!Directory.Exists(CLI.ExamplesNew))
                throw new Exception($"--examples folder {CLI.ExamplesNew} does not exist");

            if (!Directory.EnumerateFiles(CLI.ExamplesNew, "*.tif").Any())
                throw new Exception($"--examples folder {CLI.ExamplesNew} does not contain any .tif files");

            if (!string.IsNullOrEmpty(CLI.ExamplesGeneral) && !Directory.Exists(CLI.ExamplesGeneral))
                throw new Exception($"--examples_general folder {CLI.ExamplesGeneral} does not exist");

            if (!string.IsNullOrEmpty(CLI.ExamplesGeneral) && !Directory.EnumerateFiles(CLI.ExamplesGeneral, "*.tif").Any())
                throw new Exception($"--examples_general folder {CLI.ExamplesGeneral} does not contain any .tif files");

            if (CLI.ExamplesNew == CLI.ExamplesGeneral)
                throw new Exception("--examples and --examples_general must be different");

            if (CLI.PatchSize % 256 != 0 || CLI.PatchSize <= 0)
                throw new Exception("--patchsize must be positive and a multiple of 256");

            if (CLI.BatchSize <= 0)
                throw new Exception("--batchsize must be positive");

            if (CLI.LearningRateStart <= 0.0)
                throw new Exception("--lr_start must be positive");

            if (CLI.LearningRateEnd <= 0.0)
                throw new Exception("--lr_end must be positive");

            if (CLI.LearningRateStart < CLI.LearningRateEnd)
                throw new Exception("--lr_start must be greater than or equal to --lr_end");

            if (CLI.NEpochs <= 0)
                throw new Exception("--epochs must be positive");

            if (CLI.Devices == null || !CLI.Devices.Any())
                CLI.Devices = new[] { 0 };

            if (CLI.Devices.Any(d => d < 0 || d >= GPU.GetDeviceCount()))
                throw new Exception($"--devices must be a list of GPU IDs between 0 and {GPU.GetDeviceCount() - 1} (inclusive)");

            if (CLI.BatchSize % CLI.Devices.Count() != 0)
                throw new Exception($"--batchsize must be divisible by the number of devices ({CLI.Devices.Count()})");

            #endregion

            GPU.SetDevice(CLI.Devices.First());

            #region Load data

            Console.Write("Loading data...");

            bool UseCorpus = CLI.ExamplesGeneral != null;

            int2 DimsLargest = new int2(1);

            List<Image>[] AllMicrographs = { new List<Image>(), new List<Image>() };
            List<Image>[] AllLabels = { new List<Image>(), new List<Image>() };
            List<int2>[] AllDims = { new List<int2>(), new List<int2>() };
            List<float3>[] AllLabelWeights = { new List<float3>(), new List<float3>() };

            string[][] AllPaths = UseCorpus
                                      ? new[]
                                      {
                                              Directory.EnumerateFiles(CLI.ExamplesNew, "*.tif").ToArray(),
                                              Directory.EnumerateFiles(CLI.ExamplesGeneral, "*.tif").ToArray()
                                      }
                                      : new[] { Directory.EnumerateFiles(CLI.ExamplesNew, "*.tif").ToArray() };

            long[] ClassHist = new long[3];
            Image SoftMask = null;

            for (int icorpus = 0; icorpus < AllPaths.Length; icorpus++)
            {
                foreach (var examplePath in AllPaths[icorpus])
                {
                    Image ExampleImage = Image.FromFile(examplePath);
                    int N = ExampleImage.Dims.Z / 3;

                    if (ExampleImage.Dims.Z % 3 != 0)
                        throw new Exception($"Image {examplePath} has {ExampleImage.Dims.Z} layers, which is not a multiple of 3");

                    Image OnlyImages = new Image(Helper.ArrayOfSequence(0, N, 1).Select(i => ExampleImage.GetHost(Intent.Read)[i * 3]).ToArray(), new int3(ExampleImage.Dims.X, ExampleImage.Dims.Y, N));
                    int2 DimsOri = new int2(OnlyImages.Dims);
                    OnlyImages = OnlyImages.AsPadded(DimsOri - 2).AndDisposeParent();

                    if (SoftMask == null || SoftMask.Dims != new int3(OnlyImages.Dims.X * 2, OnlyImages.Dims.Y * 2, 1))
                    {
                        SoftMask?.Dispose();

                        SoftMask = new Image(new int3(OnlyImages.Dims.X * 2, OnlyImages.Dims.Y * 2, 1));
                        SoftMask.Fill(1f);
                        SoftMask.MaskRectangularly(OnlyImages.Dims.Slice(), Math.Min(OnlyImages.Dims.X, OnlyImages.Dims.Y) / 2f, false);
                    }

                    Image OriginalPadded = OnlyImages.AsPaddedClamped(new int2(OnlyImages.Dims) * 2).AndDisposeParent();
                    OriginalPadded.MultiplySlices(SoftMask);
                    //OriginalPadded.WriteMRC("d_oripadded.mrc", true);
                    OriginalPadded.Bandpass(2f * 8f / 500f, 1, false, 2f * 8f / 500f);
                    //OriginalPadded.WriteMRC("d_oripadded_bp.mrc", true);

                    OnlyImages = OriginalPadded.AsPadded(DimsOri).AndDisposeParent();
                    //OnlyImages.Bandpass(2f / 32, 1, false, 2f / 32);
                    //OnlyImages.WriteMRC("d_ori_nopad_bp.mrc", true);
                    OnlyImages.Normalize();
                    //OnlyImages.WriteMRC("d_onlyimages.mrc", true);

                    if (OnlyImages.GetHost(Intent.Read).Any(a => a.All(v => v == 0)))
                    {
                        OnlyImages.WriteMRC("d_stackzeros.mrc", true);
                        Console.WriteLine(examplePath);
                        throw new Exception("All zeros");
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float[] Mic = OnlyImages.GetHost(Intent.Read)[n];

                        AllMicrographs[icorpus].Add(new Image(Mic, ExampleImage.Dims.Slice()));
                        AllLabels[icorpus].Add(new Image(ExampleImage.GetHost(Intent.Read)[n * 3 + 1], ExampleImage.Dims.Slice()));

                        AllDims[icorpus].Add(new int2(ExampleImage.Dims));

                        float[] Labels = ExampleImage.GetHost(Intent.Read)[n * 3 + 1];
                        float[] Uncertains = ExampleImage.GetHost(Intent.Read)[n * 3 + 2];
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
                    }

                    ExampleImage.Dispose();
                    OnlyImages.Dispose();

                    DimsLargest.X = Math.Max(DimsLargest.X, ExampleImage.Dims.X);
                    DimsLargest.Y = Math.Max(DimsLargest.Y, ExampleImage.Dims.Y);

                    GPU.CheckGPUExceptions();
                }
            }

            for (int i = 0; i < AllPaths.Length; i++)
            {
                foreach (var mic in AllMicrographs[i])
                    mic.GetDevice(Intent.Read);
                foreach (var mic in AllLabels[i])
                    mic.GetDevice(Intent.Read);
            }

            SoftMask.Dispose();

            {
                float[] LabelWeights = { 1f, 1f, 1f };
                if (ClassHist[1] > 0)
                {
                    LabelWeights[0] = Math.Min((float)Math.Pow((float)ClassHist[1] / ClassHist[0], 1 / 3.0) * 0.5f, 1);
                    LabelWeights[2] = 1;//(float)Math.Sqrt((float)ClassHist[1] / ClassHist[2]);
                }
                else
                {
                    LabelWeights[0] = (float)Math.Pow((float)ClassHist[2] / ClassHist[0], 1 / 3.0);
                }

                for (int icorpus = 0; icorpus < AllPaths.Length; icorpus++)
                    for (int i = 0; i < AllMicrographs[icorpus].Count; i++)
                        AllLabelWeights[icorpus].Add(new float3(LabelWeights[0], LabelWeights[1], LabelWeights[2]));
            }

            int NNewExamples = AllMicrographs[0].Count;
            int NOldExamples = UseCorpus ? AllMicrographs[1].Count : 0;

            Console.WriteLine($" Done");
            Console.WriteLine($"{NNewExamples} new examples, {NOldExamples} general examples");
            Console.WriteLine($"Class histogram: {ClassHist[0]:E1} background, {ClassHist[1]:E1} particle, {ClassHist[2]:E1} mask");
            Console.WriteLine($"Using class weights: {AllLabelWeights[0][0].X:F3} background, {AllLabelWeights[0][0].Y:F3} particle, {AllLabelWeights[0][0].Z:F3} mask");

            #endregion

            #region Load model

            Console.Write("Loading model...");

            BoxNetTorch NetworkTrain = new BoxNetTorch(new int2(CLI.PatchSize), AllLabelWeights[0][0].ToArray(), CLI.Devices.ToArray(), CLI.BatchSize);
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

            Queue<float>[] LastAccuracies = { new Queue<float>(SmoothN), new Queue<float>(SmoothN) };
            Queue<float>[] CheckpointAccuracies = { new Queue<float>(), new Queue<float>() };
            List<float>[] LastBaselines = { new List<float>(), new List<float>() };
            GPU.SetDevice(CLI.Devices.First());

            Image[] d_AugmentedData = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize)), NThreads);
            Image[] d_AugmentedLabels = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, BatchSize * 3)), NThreads);
            IntPtr[] d_AugmentedWeights = Helper.ArrayOfFunction(i => GPU.MallocDevice(DimsAugmented.Elements() * BatchSize), NThreads);

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Random[] RG = Helper.ArrayOfFunction(i => new Random(i), NThreads);
            RandomNormal[] RGN = Helper.ArrayOfFunction(i => new RandomNormal(i), NThreads);

            int NIterations = NNewExamples * CLI.NEpochs * AllMicrographs.Length;
            Stopwatch CheckpointWatch = new Stopwatch();
            CheckpointWatch.Start();
            int NCheckpoints = 0;

            int NDone = 0;
            Helper.ForCPUGreedy(0, NIterations, NThreads,

                threadID => GPU.SetDevice(CLI.Devices.First()),

                (b, threadID) =>
                {
                    int icorpus;
                    lock (Watch)
                        icorpus = NDone % AllPaths.Length;

                    for (int ib = 0; ib < BatchSize; ib++)
                    {
                        int ExampleID = RG[threadID].Next(AllMicrographs[icorpus].Count);
                        int2 Dims = AllDims[icorpus][ExampleID];

                        int2 DimsValid = new int2(Math.Max(0, Dims.X - Border * 4), Math.Max(0, Dims.Y - Border * 4));
                        int2 DimsBorder = (Dims - DimsValid) / 2;

                        float2[] Translations = Helper.ArrayOfFunction(x => new float2(RG[threadID].Next(DimsValid.X) + DimsBorder.X,
                                                                                        RG[threadID].Next(DimsValid.Y) + DimsBorder.Y), 1);

                        float[] Rotations = Helper.ArrayOfFunction(i => (float)(RG[threadID].NextDouble() * Math.PI * 2), 1);
                        float3[] Scales = Helper.ArrayOfFunction(i => new float3(0.8f + (float)RG[threadID].NextDouble() * 0.4f,
                                                                                0.8f + (float)RG[threadID].NextDouble() * 0.4f,
                                                                                (float)(RG[threadID].NextDouble() * Math.PI * 2)), 1);
                        float StdDev = (float)Math.Abs(RGN[threadID].NextSingle(0, 0.3f));
                        float OffsetMean = RG[threadID].NextSingle() * 0.8f - 0.4f;
                        float OffsetScale = RG[threadID].NextSingle() + 0.5f;

                        GPU.BoxNet2Augment(AllMicrographs[icorpus][ExampleID].GetDevice(Intent.Read),
                                            AllLabels[icorpus][ExampleID].GetDevice(Intent.Read),
                                            Dims,
                                            d_AugmentedData[threadID].GetDeviceSlice(ib, Intent.Write),
                                            d_AugmentedLabels[threadID].GetDeviceSlice(ib * 3, Intent.Write),
                                            DimsAugmented,
                                            Helper.ToInterleaved(Translations),
                                            Rotations,
                                            Helper.ToInterleaved(Scales),
                                            OffsetMean,
                                            OffsetScale,
                                            StdDev,
                                            RG[threadID].Next(99999),
                                            false,
                                            (uint)1);
                    }

                    float LearningRate = MathHelper.Lerp((float)CLI.LearningRateStart, (float)CLI.LearningRateEnd, (float)NDone / NIterations);
                    LearningRate = MathF.Min(LearningRate, MathHelper.Lerp(0, (float)CLI.LearningRateStart, (float)NDone / 100));   // Warm-up

                    //long[][] ResultLabels = new long[2][];
                    float[][] ResultProbabilities = new float[2][];

                    float[] Loss;

                    lock (NetworkTrain)
                        NetworkTrain.Train(d_AugmentedData[threadID],
                                            d_AugmentedLabels[threadID],
                                            LearningRate,
                                            false,
                                            out _,
                                            out Loss);

                    if (float.IsNaN(Loss[0]))
                        throw new Exception("Something went wrong because loss = NaN");

                    lock (Watch)
                    {
                        NDone++;

                        //if (!float.IsNaN(AccuracyParticles[0]))
                        {
                            LastAccuracies[icorpus].Enqueue(Loss[0]);
                            if (LastAccuracies[icorpus].Count > SmoothN)
                                LastAccuracies[icorpus].Dequeue();
                            CheckpointAccuracies[icorpus].Enqueue(Loss[0]);
                        }
                        //if (!float.IsNaN(AccuracyParticles[1]))
                        //    LastBaselines[icorpus].Add(AccuracyParticles[1]);

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
                                Console.Write($"{NDone}/{NIterations}, loss = {MathHelper.Mean(LastAccuracies[0]):E2}, lr = {LearningRate:E2}, {TimeString} remaining");
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
                                for (int ib = 0; ib < BatchSize; ib++)
                                {
                                    int ExampleID = RGPrediction.Next(AllMicrographs[icorpus].Count);
                                    int2 Dims = AllDims[icorpus][ExampleID];

                                    int2 DimsValid = new int2(Math.Max(0, Dims.X - Border * 4), Math.Max(0, Dims.Y - Border * 4));
                                    int2 DimsBorder = (Dims - DimsValid) / 2;

                                    float2[] Translations = Helper.ArrayOfFunction(x => new float2(RGPrediction.Next(DimsValid.X) + DimsBorder.X,
                                                                                                   RGPrediction.Next(DimsValid.Y) + DimsBorder.Y), 1);

                                    float[] Rotations = Helper.ArrayOfFunction(i => 0f, 1);
                                    float3[] Scales = Helper.ArrayOfFunction(i => new float3(1, 1, 0), 1);
                                    float StdDev = 0;
                                    float OffsetMean = 0;
                                    float OffsetScale = 1;

                                    GPU.BoxNet2Augment(AllMicrographs[icorpus][ExampleID].GetDevice(Intent.Read),
                                                        AllLabels[icorpus][ExampleID].GetDevice(Intent.Read),
                                                        Dims,
                                                        d_AugmentedData[threadID].GetDeviceSlice(ib, Intent.Write),
                                                        d_AugmentedLabels[threadID].GetDeviceSlice(ib * 3, Intent.Write),
                                                        DimsAugmented,
                                                        Helper.ToInterleaved(Translations),
                                                        Rotations,
                                                        Helper.ToInterleaved(Scales),
                                                        OffsetMean,
                                                        OffsetScale,
                                                        StdDev,
                                                        RGPrediction.Next(99999),
                                                        false,
                                                        (uint)1);

                                    if (d_AugmentedData[threadID].GetHost(Intent.Read)[ib].All(v => v == 0))
                                    {
                                        AllMicrographs[icorpus][ExampleID].WriteMRC("d_allzeros.mrc", true);
                                        Console.WriteLine();
                                        Console.WriteLine(Dims);
                                        Console.WriteLine(DimsValid);
                                        Console.WriteLine(DimsBorder);
                                        throw new Exception("All zeros");
                                    }
                                }

                                Image Prediction, Probabilities;
                                NetworkTrain.Predict(d_AugmentedData[threadID], out Prediction, out Probabilities);
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

                                    Merged.GetHost(Intent.Write)[z * 2 + 0] = d_AugmentedData[threadID].GetHost(Intent.Read)[z];
                                    Merged.GetHost(Intent.Write)[z * 2 + 1] = Labels;
                                }

                                Merged.WriteMRC(CLI.ModelOut + $".{NCheckpoints:D3}" + ".checkpoint.mrc", true);
                                Merged.Dispose();
                            }

                            Console.WriteLine();
                            Console.WriteLine($"Average loss since previous checkpoint: {CheckpointAccuracies[0].Mean():E2}");
                            Console.WriteLine($"Saved checkpoint to {CheckpointName}");

                            CheckpointAccuracies[0].Clear();
                            CheckpointAccuracies[1].Clear();
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
}
