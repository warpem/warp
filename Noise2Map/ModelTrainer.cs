using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Warp;
using Warp.Tools;
using Warp.Tools.Async;

namespace Noise2Map
{
    /// <summary>
    /// Handles model training
    /// </summary>
    public class ModelTrainer
    {
        private readonly ProcessingContext context;
        private readonly Options options;
        private readonly BoundedQueue<TrainingBatch> batchQueue;
        private NoiseNet3DTorch trainModel;
        private CancellationToken cancellationToken;

        public string TrainedModelName { get; private set; }

        public ModelTrainer(ProcessingContext context, Options options, BoundedQueue<TrainingBatch> batchQueue = null)
        {
            this.context = context;
            this.options = options;
            this.batchQueue = batchQueue;
        }

        /// <summary>
        /// Trains a new model or loads an existing one
        /// </summary>
        public void Train(CancellationToken cancellationToken = default)
        {
            this.cancellationToken = cancellationToken;

            if (!string.IsNullOrEmpty(options.OldModelName))
            {
                TrainedModelName = options.OldModelName;
                return;
            }

            LoadOrCreateModel();
            RunTrainingLoop();

            // Save model if not in online mode, or if cancelled in online mode
            if (!options.OnlineMode)
            {
                SaveModel();
            }
            else if (cancellationToken.IsCancellationRequested)
            {
                Console.WriteLine("\nShutdown requested. Saving final model...");
                SaveModelSafe();
            }

            Console.WriteLine("\nDone training!\n");
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

            // Progress tracking - start with finite progress for LR convergence
            ProgressTracker progressTracker = new ProgressTracker(
                totalItems: options.NIterations,
                warmupItems: 5,
                emaAlpha: 0.01,
                clearLine: true);

            Queue<float> losses = new Queue<float>();
            Image predictedData = null;
            float[] loss = null;

            int iter = 0;
            int iterFine = 0;
            DateTime lastSave = DateTime.Now;
            bool hasTransitionedToOnlineMode = false;

            // In online mode, run indefinitely; otherwise stop at NIterations
            while ((options.OnlineMode || iter < options.NIterations) && !cancellationToken.IsCancellationRequested)
            {
                // Dequeue a prepared batch from the queue
                Console.WriteLine($"[DEBUG] Training iter={iter}: Dequeuing batch...");
                TrainingBatch batch = batchQueue.Dequeue();
                if (batch == null || cancellationToken.IsCancellationRequested)
                    break;  // No more batches available or shutdown requested
                Console.WriteLine($"[DEBUG] Training iter={iter}: Got batch, training...");

                try
                {
                    double currentLearningRate = CalculateLearningRate(iter, iterFine);

                    TrainBatch(batch, currentLearningRate, rand, losses, ref predictedData, ref loss, ref iterFine);

                    if (float.IsNaN(loss[0]) || float.IsInfinity(loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();

                    iter++;
                    Console.WriteLine($"[DEBUG] Training iter={iter}: Completed iteration");

                    // Transition to indefinite online mode display after LR convergence
                    if (options.OnlineMode && iter >= options.NIterations && !hasTransitionedToOnlineMode)
                    {
                        progressTracker.Complete();
                        Console.WriteLine("\nLearning rate converged. Continuing online training indefinitely...\n");
                        hasTransitionedToOnlineMode = true;
                    }

                    // Update progress with training stats
                    int totalMaps = context.MapPool?.TotalMapCount ?? 0;
                    string statsMessage = $"log(loss) = {Math.Log(MathHelper.Mean(losses)):F4}, " +
                                         $"lr = {currentLearningRate:F6}, " +
                                         $"{GPU.GetFreeMemory(options.GPUNetwork.First())} MB free" +
                                         (options.OnlineMode ? $", {totalMaps} maps" : "");

                    if (!hasTransitionedToOnlineMode)
                    {
                        if ((iter < 100 && iter % 10 == 0) || (iter >= 100 && iter % 100 == 0))
                            progressTracker.Update(statsMessage);
                    }
                    else if ((iter < 100 && iter % 10 == 0) || (iter >= 100 && iter % 100 == 0))
                    {
                        Console.Write($"\rIter {iter}: {statsMessage}");
                    }

                    // Rotate maps in pool to prevent overfitting
                    context.MapPool?.RotateOldest();

                    // Periodic model saving in online mode
                    if (options.OnlineMode && (DateTime.Now - lastSave).TotalMinutes >= options.SaveIntervalMinutes)
                    {
                        SaveModelSafe();
                        lastSave = DateTime.Now;
                    }
                }
                finally
                {
                    // Dispose the batch after training
                    batch.Dispose();
                }
            }

            if (!hasTransitionedToOnlineMode)
                progressTracker.Complete();
        }

        private double CalculateLearningRate(int iter, int iterFine)
        {
            double currentLearningRate;

            // In online mode, freeze LR after convergence iterations
            if (options.OnlineMode && iter >= options.NIterations)
            {
                currentLearningRate = options.LearningRateFinish;
            }
            else
            {
                currentLearningRate = MathHelper.Lerp((float)options.LearningRateStart,
                                                     (float)options.LearningRateFinish,
                                                     Math.Min(1.0f, iter / (float)options.NIterations));
            }

            if (iterFine < 100)
                currentLearningRate = MathHelper.Lerp(0, (float)currentLearningRate, iterFine / 99f);

            return currentLearningRate;
        }

        private void TrainBatch(TrainingBatch batch, double currentLearningRate, Random rand, Queue<float> losses,
                               ref Image predictedData, ref float[] loss, ref int iterFine)
        {
            bool twist = rand.Next(2) == 0;

            if (context.IsTomo)
                trainModel.TrainDeconv(twist ? batch.ExtractedSource : batch.ExtractedTarget,
                                      twist ? batch.ExtractedTarget : batch.ExtractedSource,
                                      batch.ExtractedCTF,
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
                trainModel.Train(twist ? batch.ExtractedSource : batch.ExtractedTarget,
                                twist ? batch.ExtractedTarget : batch.ExtractedSource,
                                (float)currentLearningRate,
                                out predictedData,
                                out loss);

            losses.Enqueue(loss[0]);
            if (losses.Count > 10)
                losses.Dequeue();

            iterFine++;
        }

        private void SaveModel()
        {
            TrainedModelName = options.SaveModelName + ".pt";
            trainModel.Save(Path.Combine(context.WorkingDirectory, TrainedModelName));

            if (!options.OnlineMode)
                trainModel.Dispose();
        }

        /// <summary>
        /// Saves model safely for online mode (temp file + atomic rename)
        /// </summary>
        private void SaveModelSafe()
        {
            string modelName = options.SaveModelName + "_" +
                              DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
            string modelPath = Path.Combine(context.WorkingDirectory, modelName);
            string tempPath = modelPath + ".tmp";

            // Save to temp file first
            trainModel.Save(tempPath);

            // Atomic rename (safe for consumer to read)
            File.Move(tempPath, modelPath, overwrite: true);

            Console.WriteLine($"\nSaved model: {modelName}");
            TrainedModelName = modelName;
        }

        public void Dispose()
        {
            trainModel?.Dispose();
        }
    }
}