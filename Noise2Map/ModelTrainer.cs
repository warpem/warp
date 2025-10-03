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
        private readonly ConcurrentTrainingQueue batchQueue;
        private NoiseNet3DTorch trainModel;

        public string TrainedModelName { get; private set; }

        public ModelTrainer(ProcessingContext context, Options options, ConcurrentTrainingQueue batchQueue = null)
        {
            this.context = context;
            this.options = options;
            this.batchQueue = batchQueue;
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
            int nMapsPerBatch = Math.Min(8, context.Maps1.Count);
            int mapSamples = options.BatchSize;

            foreach (var item in context.MapCTFs)
                item.GetDevice(Intent.Read);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            Queue<float> losses = new Queue<float>();
            Image predictedData = null;
            float[] loss = null;

            int iter = 0;
            int iterFine = 0;
            double smoothedIterTime = 0;

            while (iter < options.NIterations)
            {
                // Dequeue a prepared batch from the queue
                TrainingBatch batch = batchQueue.Dequeue();
                if (batch == null)
                    break;  // No more batches available

                try
                {
                    double currentLearningRate = CalculateLearningRate(iter, iterFine);

                    TrainBatch(batch.ShuffledMapIDs, batch.ExtractedSourceRand, batch.ExtractedTargetRand, batch.ExtractedCTFRand,
                              currentLearningRate, rand, mapSamples, losses, ref predictedData, ref loss, ref iterFine, nMapsPerBatch);

                    double iterTime = watch.Elapsed.TotalSeconds;
                    smoothedIterTime = UpdateSmoothedIterTime(smoothedIterTime, iterTime, iter);

                    PrintProgress(iter, smoothedIterTime, losses, currentLearningRate);

                    if (float.IsNaN(loss[0]) || float.IsInfinity(loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();
                    watch.Restart();

                    iter++;
                }
                finally
                {
                    // Dispose the batch after training
                    batch.Dispose();
                }
            }

            watch.Stop();
        }

        private double UpdateSmoothedIterTime(double previousSmoothed, double currentIterTime, int iter)
        {
            // Skip first 5 iterations (PyTorch warmup/optimization phase)
            const int warmupIters = 5;

            if (iter < warmupIters)
            {
                // During warmup, return current time but don't use for estimation
                return currentIterTime;
            }
            else if (iter < warmupIters + 10)
            {
                // After warmup: use running average for quick convergence
                int effectiveIter = iter - warmupIters;
                return (previousSmoothed * effectiveIter + currentIterTime) / (effectiveIter + 1);
            }
            else
            {
                // Exponential moving average: smoothed = alpha * current + (1 - alpha) * previous
                double alpha = 0.01;
                return alpha * currentIterTime + (1 - alpha) * previousSmoothed;
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
                               ref Image predictedData, ref float[] loss, ref int iterFine, int nMapsPerBatch)
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

        private void PrintProgress(int iter, double smoothedIterTime, Queue<float> losses, double currentLearningRate)
        {
            int remainingIters = options.NIterations - 1 - iter;
            double totalSecondsRemaining = smoothedIterTime * remainingIters;
            TimeSpan timeRemaining = TimeSpan.FromSeconds(totalSecondsRemaining);

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