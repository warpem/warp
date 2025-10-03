using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
            int mapCount = context.MapPool?.CurrentPoolSize ?? 1;
            if (options.BatchSize != 4 || mapCount > 1)
            {
                options.NIterations = options.NIterations * 4 / options.BatchSize / Math.Min(8, mapCount);
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
            int mapSamples = options.BatchSize;

            // Progress tracking
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
                              currentLearningRate, rand, mapSamples, losses, ref predictedData, ref loss, ref iterFine);

                    if (float.IsNaN(loss[0]) || float.IsInfinity(loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();

                    // Update progress with training stats
                    string statsMessage = $"log(loss) = {Math.Log(MathHelper.Mean(losses)):F4}, " +
                                         $"lr = {currentLearningRate:F6}, " +
                                         $"{GPU.GetFreeMemory(options.GPUNetwork.First())} MB free";
                    progressTracker.Update(statsMessage);

                    iter++;

                    // Rotate maps in pool to prevent overfitting
                    context.MapPool?.RotateOldest();
                }
                finally
                {
                    // Dispose the batch after training
                    batch.Dispose();
                }
            }

            progressTracker.Complete();
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