using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace Noise2Half
{
    class Noise2Half
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            string ExtRecStarPath = "";

            if (!Debugger.IsAttached)
            {
                if (args.Length != 1)
                    throw new Exception($"Expected exactly one argument with .star file path, got {args.Length} arguments instead.");
                else
                    ExtRecStarPath = args[0];
            }

            #region Load STAR tables and figure out which WorkSet you are

            Star TableGeneral = new StarParameters(ExtRecStarPath, "external_reconstruct_general");
            Star TableFSC = new Star(ExtRecStarPath, "external_reconstruct_tau2");

            string RecPath = TableGeneral.GetRowValue(0, "rlnExtReconsResult");
            string RecName = Helper.PathToNameWithExtension(RecPath);
            string RecDir = Helper.PathToFolder(RecPath);
            string RecPrefix = RecName.Substring(0, 9);

            int WorkSet = 0;
            if (RecName.ToLower().Contains("half1"))
                WorkSet = 1;
            else if (RecName.ToLower().Contains("half2"))
                WorkSet = 2;

            #endregion

            #region Make reconstruction by calling relion_external_reconstruct

            if (WorkSet > 0)
            {
                Console.WriteLine($"Half {WorkSet}: Starting relion_external_reconstruct with {ExtRecStarPath}");

                Process ExternalReconstructor = new Process
                {
                    StartInfo =
                {
                    FileName = Path.Combine("relion_external_reconstruct"),
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = ExtRecStarPath + " --no_map"
                }
                };
                ExternalReconstructor.Start();
                ExternalReconstructor.WaitForExit();

                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsDataReal"));
                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsDataImag"));
                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsWeight"));

                Console.WriteLine($"Half {WorkSet}: relion_external_reconstruct created {RecName}");
            }
            else
            {
                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsDataReal"));
                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsDataImag"));
                File.Delete(TableGeneral.GetRowValue(0, "rlnExtReconsWeight"));

                Console.WriteLine("Final iteration, won't reconstruct anything extra");
            }

            #endregion

            #region Modify FSC in STAR output

            if (TableFSC.HasColumn("rlnReferenceTau2"))
                TableFSC.RemoveColumn("rlnReferenceTau2");

            if (TableFSC.HasColumn("rlnGoldStandardFsc"))
            {
                float[] FSC = TableFSC.GetFloat("rlnGoldStandardFsc");
                //FSC = FSC.Select(v => ((float)Math.Sqrt(2 * (v + v * v)) + v) / (2 + v)).ToArray();
                FSC = FSC.Select(v => MathF.Sqrt(2 * v / (v + 1))).ToArray();
                TableFSC.SetColumn("rlnGoldStandardFsc", FSC.Select(v => v.ToString(CultureInfo.InvariantCulture)).ToArray());
            }

            Star.SaveMultitable(ExtRecStarPath, new Dictionary<string, Star>()
            {
                { "external_reconstruct_general", TableGeneral },
                { "external_reconstruct_tau2", TableFSC }
            });

            #endregion

            #region If you're WorkSet 1, synchronize with #2 to get the second half-map, if you're 2, wait for 1 to finish

            if (WorkSet == 2)
            {
                using (File.Create(RecPath + ".done")) { }
                Console.WriteLine("Half 2 has done its job and will wait for half 1 to finish now");

                while (!File.Exists(RecPath.Replace("half2", "half1") + ".done"))
                    Thread.Sleep(500);

                File.Delete(RecPath.Replace("half2", "half1") + ".done");

                Console.WriteLine("Half 2 will exit now that half 1 is finished");

                return;
            }
            else if (WorkSet == 1)
            {
                Console.WriteLine("Half 1 waiting for half 2 to finish");

                while (!File.Exists(RecPath.Replace("half1", "half2") + ".done"))
                    Thread.Sleep(500);

                File.Delete(RecPath.Replace("half1", "half2") + ".done");

                Console.WriteLine("Half 1 got data from half 2, proceeding");
            }
            else if (WorkSet == 0)
            {
                Console.WriteLine("Will train on final run_half*_class001_unfil.mrc maps");
                //return;
            }

            #endregion

            #region Figure out GPU IDs based on what's available

            List<(int, long)> MemoryAvailable = Helper.ArrayOfFunction(d => (d, GPU.GetFreeMemory(d)), GPU.GetDeviceCount()).ToList();
            MemoryAvailable.Sort((a, b) => -(a.Item2.CompareTo(b.Item2)));

            foreach (var item in MemoryAvailable)
                Console.WriteLine($"Device {item.Item1} has {item.Item2} MB");

            int GPUNetwork = MemoryAvailable[0].Item1;
            int GPUPreprocess = GPUNetwork;
            if (MemoryAvailable.Count > 1 && (MemoryAvailable[1].Item2) > 4000)
                GPUPreprocess = MemoryAvailable[1].Item1;

            Console.WriteLine($"Training on device {GPUNetwork} and preprocessing on {GPUPreprocess}");

            GPU.SetDevice(GPUPreprocess);

            #endregion

            #region Load and prepare data

            Console.Write("Preparing data... ");

            Image MapHalf1 = null;
            Image MapHalf2 = null;

            if (WorkSet == 1)
            {
                MapHalf1 = Image.FromFile(RecPath);
                MapHalf2 = Image.FromFile(RecPath.Replace("half1", "half2"));
            }
            else if (WorkSet == 0)
            {
                string Path1 = Path.Combine(RecDir, RecPrefix + "_half1_class001_unfil.mrc");
                string Path2 = Path.Combine(RecDir, RecPrefix + "_half2_class001_unfil.mrc");

                while (!File.Exists(Path1))
                {
                    Console.WriteLine($"Waiting for final half-map 1... ({Path1})");
                    Thread.Sleep(1000);
                }

                while (!File.Exists(Path2))
                {
                    Console.WriteLine($"Waiting for final half-map 2... ({Path2})");
                    Thread.Sleep(1000);
                }

                Thread.Sleep(1000);

                MapHalf1 = Image.FromFile(Path1);
                MapHalf2 = Image.FromFile(Path2);
            }
            else
                throw new Exception("Shouldn't be here");

            float PixelSize = MapHalf1.PixelSize;
            int3 BoundingBox = MapHalf1.Dims / 2;

            float2 MeanStdForDenoising;

            // Normalize to mean = 0, std = 1 taking the central 1/8 of the map as reference
            {
                Image Map1Center = MapHalf1.AsPadded(MapHalf1.Dims / 2);
                Image Map2Center = MapHalf2.AsPadded(MapHalf2.Dims / 2);

                MeanStdForDenoising = MathHelper.MeanAndStd(Helper.Combine(Map1Center.GetHostContinuousCopy(), Map2Center.GetHostContinuousCopy()));

                Map1Center.Dispose();
                Map2Center.Dispose();
            }

            MapHalf1.Add(-MeanStdForDenoising.X);
            MapHalf1.Multiply(1f / MeanStdForDenoising.Y);
            MapHalf2.Add(-MeanStdForDenoising.X);
            MapHalf2.Multiply(1f / MeanStdForDenoising.Y);

            Image MapHalf1Ori = MapHalf1.GetCopy();
            Image MapHalf2Ori = MapHalf2.GetCopy();

            GPU.PrefilterForCubic(MapHalf1.GetDevice(Intent.ReadWrite), MapHalf1.Dims);
            GPU.PrefilterForCubic(MapHalf2.GetDevice(Intent.ReadWrite), MapHalf2.Dims);

            ulong[] Texture1 = new ulong[1], TextureArray1 = new ulong[1];
            GPU.CreateTexture3D(MapHalf1.GetDevice(Intent.Read), MapHalf1.Dims, Texture1, TextureArray1, true);
            MapHalf1.FreeDevice();

            ulong[] Texture2 = new ulong[1], TextureArray2 = new ulong[1];
            GPU.CreateTexture3D(MapHalf2.GetDevice(Intent.Read), MapHalf2.Dims, Texture2, TextureArray2, true);
            MapHalf2.FreeDevice();

            Console.WriteLine("Done");

            #endregion

            #region Load model and load old weights if available

            string ModelPath = Path.Combine(RecDir, "Noise2Half.pt");
            bool TrainingFromScratch = !File.Exists(ModelPath);

            int3 Dim = new int3(64);
            int BatchSize = 4;
            int NIterations = TrainingFromScratch ? 2000 : (WorkSet == 0 ? 1000 : 300); // 2000 in first, 300 in intermediate, 1000 in final

            float LearningRateStart = TrainingFromScratch ? 1e-4f : 1e-5f;
            float LearningRateFinish = TrainingFromScratch ? 1e-5f : 1e-5f;

            Console.WriteLine($"Will train for {NIterations} iterations");

            // For half-map denoising, use a shallow model with depth = 1
            NoiseNet3DTorch TrainModel = new NoiseNet3DTorch(Dim, new[] { GPUNetwork }, BatchSize, depth: 1, progressiveDepth: false, maxWidth: 64);
            if (!TrainingFromScratch)
            {
                Console.WriteLine("Found model weights from a previous iteration, loading");
                TrainModel.Load(ModelPath);
            }

            #endregion

            #region Training

            GPU.SetDevice(GPUPreprocess);

            int RandSeed = Guid.NewGuid().GetHashCode();
            try
            {
                RandSeed = int.Parse(RecName.Substring(RecName.IndexOf("_it") + "_it".Length, 3));
            }
            catch { }
            Random Rand = new Random(RandSeed);

            Image ExtractedSource = new Image(new int3(Dim.X, Dim.Y, Dim.Z * BatchSize));
            Image ExtractedTarget = new Image(new int3(Dim.X, Dim.Y, Dim.Z * BatchSize));

            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            Queue<float> Losses = new Queue<float>();

            Image PredictedData = null;
            float[] Loss = null;

            for (int iter = 0; iter < NIterations; iter++)
            {
                {
                    int3 DimsMap = MapHalf1.Dims;
                    int3 Margin = (DimsMap - BoundingBox) / 2;

                    float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), BatchSize);

                    float3[] Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                            (float)Rand.NextDouble() * 360,
                                                                            (float)Rand.NextDouble() * 360) * Helper.ToRad, BatchSize);

                    GPU.Rotate3DExtractAt(Texture1[0],
                                            MapHalf1.Dims,
                                            ExtractedSource.GetDevice(Intent.Write),
                                            Dim,
                                            Helper.ToInterleaved(Angle),
                                            Helper.ToInterleaved(Position),
                                            (uint)BatchSize);
                    GPU.Rotate3DExtractAt(Texture2[0],
                                            MapHalf2.Dims,
                                            ExtractedTarget.GetDevice(Intent.Write),
                                            Dim,
                                            Helper.ToInterleaved(Angle),
                                            Helper.ToInterleaved(Position),
                                            (uint)BatchSize);
                }

                double CurrentLearningRate = MathHelper.Lerp(LearningRateStart,
                                                                LearningRateFinish,
                                                                iter / (float)NIterations);

                if (iter < 100)
                    CurrentLearningRate = MathHelper.Lerp(0, (float)CurrentLearningRate, iter / 99f);

                bool Twist = Rand.Next(2) == 0;

                TrainModel.Train(Twist ? ExtractedSource : ExtractedTarget,
                                    Twist ? ExtractedTarget : ExtractedSource,
                                    (float)CurrentLearningRate,
                                    out PredictedData,
                                    out Loss);

                Losses.Enqueue(Loss[0]);
                if (Losses.Count > 10)
                    Losses.Dequeue();

                TimeSpan TimeRemaining = Watch.Elapsed * (NIterations - 1 - iter);

                if (iter % 100 == 0 || iter == NIterations - 1)
                {
                    string ToWrite = $"{iter + 1}/{NIterations}, " +
                                        (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
                                        (iter > 10 ? $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " : "") +
                                        $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
                                        $"lr = {CurrentLearningRate:F6}" + (iter < 100 ? " (warm-up), " : ", ") +
                                        $"{GPU.GetFreeMemory(GPUNetwork)} MB free";

                    try
                    {
                        ClearCurrentConsoleLine();
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

            Watch.Stop();

            TrainModel.Save(ModelPath);

            Console.WriteLine("\nDone training!\n");

            #endregion

            #region Denoise

            GPU.SetDevice(GPUPreprocess);

            if (WorkSet == 1)
            {
                Console.WriteLine("Denoising combined map");

                Image MapCombined = MapHalf1Ori.GetCopy();
                MapCombined.Add(MapHalf2Ori);
                MapCombined.Multiply(0.5f);
                NoiseNet3DTorch.Denoise(MapCombined, new NoiseNet3DTorch[] { TrainModel });
                MapCombined.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapCombined.WriteMRC(RecPath.Replace("half1", "combined"), PixelSize, true);

                Console.WriteLine("Denoising half-map 1");
                NoiseNet3DTorch.Denoise(MapHalf1Ori, new NoiseNet3DTorch[] { TrainModel });
                MapHalf1Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapHalf1Ori.WriteMRC(RecPath, PixelSize, true);

                Console.WriteLine("Denoising half-map 2");
                NoiseNet3DTorch.Denoise(MapHalf2Ori, new NoiseNet3DTorch[] { TrainModel });
                MapHalf2Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapHalf2Ori.WriteMRC(RecPath.Replace("half1", "half2"), PixelSize, true);

                // Tell half 2 to exit
                using (File.Create(RecPath + ".done")) { }
            }
            else if (WorkSet == 0)
            {
                Console.WriteLine("Denoising combined map");

                Image MapCombined = MapHalf1Ori.GetCopy();
                MapCombined.Add(MapHalf2Ori);
                MapCombined.Multiply(0.5f);
                NoiseNet3DTorch.Denoise(MapCombined, new NoiseNet3DTorch[] { TrainModel });
                MapCombined.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapCombined.WriteMRC(RecPath, PixelSize, true);

                Console.WriteLine("Denoising half-map 1");
                NoiseNet3DTorch.Denoise(MapHalf1Ori, new NoiseNet3DTorch[] { TrainModel });
                MapHalf1Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapHalf1Ori.WriteMRC(Path.Combine(RecDir, "run_half1_class001_denoised.mrc"), PixelSize, true);

                Console.WriteLine("Denoising half-map 2");
                NoiseNet3DTorch.Denoise(MapHalf2Ori, new NoiseNet3DTorch[] { TrainModel });
                MapHalf2Ori.TransformValues(v => (v * MeanStdForDenoising.Y) + MeanStdForDenoising.X);
                MapHalf2Ori.WriteMRC(Path.Combine(RecDir, "run_half2_class001_denoised.mrc"), PixelSize, true);
            }

            Console.WriteLine("All done!");

            #endregion
        }

        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth - 2));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
