using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Handles model training
    /// </summary>
    public class ModelTrainer
    {
        private readonly ProcessingContext context;
        private readonly Options options;
        private NoiseNet3DTorch trainModel;

        public string TrainedModelName { get; private set; }

        public ModelTrainer(ProcessingContext context, Options options)
        {
            this.context = context;
            this.options = options;
        }

        /// <summary>
        /// Trains a new model or loads an existing one
        /// </summary>
        public void Train()
        {
            if (!string.IsNullOrEmpty(options.OldModelName))
            {
                TrainedModelName = options.OldModelName;
                return;
            }

            AdjustIterations();
            LoadOrCreateModel();
            RunTrainingLoop();
            SaveModel();

            Console.WriteLine("\nDone training!\n");
        }

        private void AdjustIterations()
        {
            if (options.BatchSize != 4 || context.Maps1.Count > 1)
            {
                options.NIterations = options.NIterations * 4 / options.BatchSize / Math.Min(8, context.Maps1.Count);
                Console.WriteLine($"Adjusting the number of iterations to {options.NIterations} to match batch size and number of maps.\n");
            }
        }

        private void LoadOrCreateModel()
        {
            string modelPath = GetModelPath();

            Console.WriteLine("Loading model, " + GPU.GetFreeMemory(options.GPUNetwork.First()) + " MB free.");
            trainModel = new NoiseNet3DTorch(context.TrainingDims,
                                             options.GPUNetwork.ToArray(),
                                             options.BatchSize,
                                             depth: options.MiniModel ? 1 : 2,
                                             progressiveDepth: !options.MiniModel,
                                             maxWidth: options.MiniModel ? 64 : 99999);

            if (!string.IsNullOrEmpty(modelPath))
                trainModel.Load(modelPath);

            Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(options.GPUNetwork.First()) + " MB remaining.\n");
        }

        private string GetModelPath()
        {
            if (string.IsNullOrEmpty(options.StartModelName))
                return null;

            string modelPath = options.StartModelName;

            if (File.Exists(Path.Combine(context.WorkingDirectory, options.StartModelName)))
                modelPath = Path.Combine(context.WorkingDirectory, options.StartModelName);
            else if (File.Exists(Path.Combine(context.ProgramFolder, options.StartModelName)))
                modelPath = Path.Combine(context.ProgramFolder, options.StartModelName);

            if (!File.Exists(modelPath))
                throw new Exception($"Could not find initial model '{options.StartModelName}'. Please make sure it can be found either here, or in the installation directory.");

            return modelPath;
        }

        private void RunTrainingLoop()
        {
            GPU.SetDevice(options.GPUPreprocess);

            Random rand = new Random(123);
            int nMaps = context.Maps1.Count;
            int nMapsPerBatch = Math.Min(8, nMaps);
            int mapSamples = options.BatchSize;

            int3 dim = context.TrainingDims;
            int3 dim2 = dim * 2;

            Image[] extractedSource = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedSourceRand = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedTargetRand = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedCTF = Helper.ArrayOfFunction(i => new Image(new int3(dim2.X, dim2.Y, dim2.Z * mapSamples), true), nMapsPerBatch);
            Image[] extractedCTFRand = Helper.ArrayOfFunction(i => new Image(new int3(dim2.X, dim2.Y, dim2.Z * mapSamples), true), nMapsPerBatch);

            foreach (var item in context.MapCTFs)
                item.GetDevice(Intent.Read);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            Queue<float> losses = new Queue<float>();
            Image predictedData = null;
            float[] loss = null;

            ulong[] texture1 = new ulong[1];
            ulong[] texture2 = new ulong[1];
            ulong[] textureArray1 = new ulong[1];
            ulong[] textureArray2 = new ulong[1];

            for (int iter = 0, iterFine = 0; iter < options.NIterations; iter++)
            {
                int[] shuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, nMaps, 1), nMapsPerBatch, rand.Next(9999999));

                ExtractTrainingData(shuffledMapIDs, extractedSource, extractedTarget, extractedCTF,
                                   rand, mapSamples, dim, dim2, texture1, texture2, textureArray1, textureArray2,
                                   ref iterFine, nMaps);

                ShuffleExamples(extractedSource, extractedTarget, extractedCTF, extractedSourceRand, extractedTargetRand, extractedCTFRand,
                               rand, mapSamples, dim, dim2, nMapsPerBatch);

                double currentLearningRate = CalculateLearningRate(iter, iterFine);

                TrainBatch(shuffledMapIDs, extractedSourceRand, extractedTargetRand, extractedCTFRand,
                          currentLearningRate, rand, mapSamples, losses, ref predictedData, ref loss, ref iterFine);

                PrintProgress(iter, watch, losses, currentLearningRate);

                if (float.IsNaN(loss[0]) || float.IsInfinity(loss[0]))
                    throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                GPU.CheckGPUExceptions();
                watch.Restart();
            }

            if (nMaps == 1)
            {
                GPU.DestroyTexture(texture1[0], textureArray1[0]);
                GPU.DestroyTexture(texture2[0], textureArray2[0]);
            }

            watch.Stop();
        }

        private void ExtractTrainingData(int[] shuffledMapIDs, Image[] extractedSource, Image[] extractedTarget, Image[] extractedCTF,
                                        Random rand, int mapSamples, int3 dim, int3 dim2,
                                        ulong[] texture1, ulong[] texture2, ulong[] textureArray1, ulong[] textureArray2,
                                        ref int iterFine, int nMaps)
        {
            for (int m = 0; m < shuffledMapIDs.Length; m++)
            {
                int mapID = shuffledMapIDs[m];
                Image map1 = context.Maps1[mapID];
                Image map2 = context.Maps2[mapID];
                int3 dimsMap = map1.Dims;

                int3 margin = dim / 2;
                float3[] position = GenerateRandomPositions(rand, dimsMap, margin, mapSamples);
                float3[] angle = GenerateRandomAngles(rand, mapSamples);

                // Extract from map1
                if (nMaps > 1 || iterFine == 0)
                    GPU.CreateTexture3D(map1.GetDevice(Intent.Read), map1.Dims, texture1, textureArray1, true);
                if (nMaps > 4)
                    map1.FreeDevice();

                GPU.Rotate3DExtractAt(texture1[0], map1.Dims, extractedSource[m].GetDevice(Intent.Write),
                                     dim, Helper.ToInterleaved(angle), Helper.ToInterleaved(position), (uint)mapSamples);

                if (nMaps > 1)
                    GPU.DestroyTexture(texture1[0], textureArray1[0]);

                // Extract from map2
                if (nMaps > 1 || iterFine == 0)
                    GPU.CreateTexture3D(map2.GetDevice(Intent.Read), map2.Dims, texture2, textureArray2, true);
                if (nMaps > 4)
                    map2.FreeDevice();

                GPU.Rotate3DExtractAt(texture2[0], map2.Dims, extractedTarget[m].GetDevice(Intent.Write),
                                     dim, Helper.ToInterleaved(angle), Helper.ToInterleaved(position), (uint)mapSamples);

                if (nMaps > 1)
                    GPU.DestroyTexture(texture2[0], textureArray2[0]);

                // Copy CTF
                for (int i = 0; i < mapSamples; i++)
                    GPU.CopyDeviceToDevice(context.MapCTFs[mapID].GetDevice(Intent.Read),
                                          extractedCTF[m].GetDeviceSlice(i * dim2.Z, Intent.Write),
                                          context.MapCTFs[mapID].ElementsReal);
            }
        }

        private float3[] GenerateRandomPositions(Random rand, int3 dimsMap, int3 margin, int mapSamples)
        {
            if (context.BoundsMin == new int3(0))
                return Helper.ArrayOfFunction(i => new float3((float)rand.NextDouble() * (dimsMap.X - margin.X * 2) + margin.X,
                                                              (float)rand.NextDouble() * (dimsMap.Y - margin.Y * 2) + margin.Y,
                                                              (float)rand.NextDouble() * (dimsMap.Z - margin.Z * 2) + margin.Z), mapSamples);
            else
                return Helper.ArrayOfFunction(i => new float3((float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).X + context.BoundsMin.X,
                                                              (float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).Y + context.BoundsMin.Y,
                                                              (float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).Z + context.BoundsMin.Z), mapSamples);
        }

        private float3[] GenerateRandomAngles(Random rand, int mapSamples)
        {
            if (options.DontAugment)
                return Helper.ArrayOfFunction(i => new float3((float)Math.Round(rand.NextDouble()) * 0,
                                                              (float)Math.Round(rand.NextDouble()) * 0,
                                                              (float)Math.Round(rand.NextDouble()) * 0) * Helper.ToRad, mapSamples);
            else
                return Helper.ArrayOfFunction(i => new float3((float)rand.NextDouble() * 360,
                                                              (float)rand.NextDouble() * 360,
                                                              (float)rand.NextDouble() * 360) * Helper.ToRad, mapSamples);
        }

        private void ShuffleExamples(Image[] extractedSource, Image[] extractedTarget, Image[] extractedCTF,
                                    Image[] extractedSourceRand, Image[] extractedTargetRand, Image[] extractedCTFRand,
                                    Random rand, int mapSamples, int3 dim, int3 dim2, int nMapsPerBatch)
        {
            for (int b = 0; b < mapSamples; b++)
            {
                int[] order = Helper.RandomSubset(Helper.ArrayOfSequence(0, nMapsPerBatch, 1), nMapsPerBatch, rand.Next(9999999));
                for (int i = 0; i < order.Length; i++)
                {
                    GPU.CopyDeviceToDevice(extractedSource[i].GetDeviceSlice(b * dim.Z, Intent.Read),
                                          extractedSourceRand[order[i]].GetDeviceSlice(b * dim.Z, Intent.Write),
                                          dim.Elements());
                    GPU.CopyDeviceToDevice(extractedTarget[i].GetDeviceSlice(b * dim.Z, Intent.Read),
                                          extractedTargetRand[order[i]].GetDeviceSlice(b * dim.Z, Intent.Write),
                                          dim.Elements());
                    GPU.CopyDeviceToDevice(extractedCTF[i].GetDeviceSlice(b * dim2.Z, Intent.Read),
                                          extractedCTFRand[order[i]].GetDeviceSlice(b * dim2.Z, Intent.Write),
                                          (dim2.X / 2 + 1) * dim2.Y * dim2.Z);
                }
            }
        }

        private double CalculateLearningRate(int iter, int iterFine)
        {
            double currentLearningRate = MathHelper.Lerp((float)options.LearningRateStart,
                                                         (float)options.LearningRateFinish,
                                                         iter / (float)options.NIterations);

            if (iterFine < 100)
                currentLearningRate = MathHelper.Lerp(0, (float)currentLearningRate, iterFine / 99f);

            return currentLearningRate;
        }

        private void TrainBatch(int[] shuffledMapIDs, Image[] extractedSourceRand, Image[] extractedTargetRand, Image[] extractedCTFRand,
                               double currentLearningRate, Random rand, int mapSamples, Queue<float> losses,
                               ref Image predictedData, ref float[] loss, ref int iterFine)
        {
            Image noiseMask = new Image(IntPtr.Zero, extractedSourceRand[0].Dims);

            for (int m = 0; m < shuffledMapIDs.Length; m++)
            {
                int mapID = m;
                bool twist = rand.Next(2) == 0;

                if (context.IsTomo)
                    trainModel.TrainDeconv((twist ? extractedSourceRand : extractedTargetRand)[mapID],
                                          (twist ? extractedTargetRand : extractedSourceRand)[mapID],
                                          extractedCTFRand[mapID],
                                          (float)currentLearningRate,
                                          false,
                                          null,
                                          null,
                                          out predictedData,
                                          out _,
                                          out _,
                                          out loss,
                                          out _);
                else
                    trainModel.Train((twist ? extractedSourceRand : extractedTargetRand)[mapID],
                                    (twist ? extractedTargetRand : extractedSourceRand)[mapID],
                                    (float)currentLearningRate,
                                    out predictedData,
                                    out loss);

                losses.Enqueue(loss[0]);
                if (losses.Count > 10)
                    losses.Dequeue();

                iterFine++;
            }

            noiseMask.Dispose();
        }

        private void PrintProgress(int iter, Stopwatch watch, Queue<float> losses, double currentLearningRate)
        {
            TimeSpan timeRemaining = watch.Elapsed * (options.NIterations - 1 - iter);

            string toWrite = $"{iter + 1}/{options.NIterations}, " +
                            (timeRemaining.Days > 0 ? (timeRemaining.Days + " days ") : "") +
                            $"{timeRemaining.Hours}:{timeRemaining.Minutes:D2}:{timeRemaining.Seconds:D2} remaining, " +
                            $"log(loss) = {Math.Log(MathHelper.Mean(losses)).ToString("F4")}, " +
                            $"lr = {currentLearningRate:F6}, " +
                            $"{GPU.GetFreeMemory(options.GPUNetwork.First())} MB free";

            try
            {
                VirtualConsole.ClearLastLine();
                Console.Write(toWrite);
            }
            catch
            {
                // When we're outputting to a text file when launched on HPC cluster
                Console.WriteLine(toWrite);
            }
        }

        private void SaveModel()
        {
            TrainedModelName = "NoiseNet3D_" + (!string.IsNullOrEmpty(options.StartModelName) ? (options.StartModelName + "_") : "") +
                              DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
            trainModel.Save(Path.Combine(context.WorkingDirectory, TrainedModelName));
            trainModel.Dispose();
        }

        public void Dispose()
        {
            trainModel?.Dispose();
        }
    }
}