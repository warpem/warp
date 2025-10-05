using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Warp.Tools.Async;

namespace Noise2Map
{
    /// <summary>
    /// Coordinates parallel batch preparation threads and training thread using producer-consumer pattern
    /// </summary>
    public class TrainingCoordinator
    {
        private readonly ProcessingContext context;
        private readonly Options options;

        public TrainingCoordinator(ProcessingContext context, Options options)
        {
            this.context = context;
            this.options = options;
        }

        /// <summary>
        /// Runs training with parallel batch preparation
        /// </summary>
        /// <param name="numPreparationThreads">Number of threads to use for preparing batches</param>
        /// <param name="queueCapacity">Maximum number of batches to buffer in queue</param>
        /// <param name="externalCancellationToken">Optional external cancellation token for graceful shutdown</param>
        /// <returns>The name of the trained model</returns>
        public string RunConcurrentTraining(int numPreparationThreads = 3, int queueCapacity = 6, CancellationToken externalCancellationToken = default)
        {
            using (var batchQueue = new BoundedQueue<TrainingBatch>(queueCapacity))
            {
                var cancellationSource = CancellationTokenSource.CreateLinkedTokenSource(externalCancellationToken);

                // Start preparation worker threads
                var preparationTasks = StartPreparationWorkers(batchQueue, numPreparationThreads, cancellationSource);

                // Signal completion when all preparation threads are done (only in non-online mode)
                if (!options.OnlineMode)
                {
                    Task.Run(async () =>
                    {
                        await Task.WhenAll(preparationTasks);
                        batchQueue.CompleteAdding();
                    });
                }

                // Train on prepared batches (runs on main thread)
                var trainer = new ModelTrainer(context, options, batchQueue);
                string trainedModelName;

                try
                {
                    trainer.Train(cancellationSource.Token);
                    trainedModelName = trainer.TrainedModelName;
                }
                finally
                {
                    cancellationSource.Cancel();
                    trainer.Dispose();
                }

                // Wait for all preparation threads to finish
                WaitForPreparationThreads(preparationTasks);

                return trainedModelName;
            }
        }

        private List<Task> StartPreparationWorkers(BoundedQueue<TrainingBatch> batchQueue, int numThreads, CancellationTokenSource cancellationSource)
        {
            var tasks = new List<Task>();

            for (int i = 0; i < numThreads; i++)
            {
                int threadId = i;
                var worker = new BatchPreparationWorker(context, options, batchQueue, seed: 123 + threadId);

                var task = Task.Run(() =>
                {
                    try
                    {
                        if (options.OnlineMode)
                        {
                            // In online mode, run indefinitely
                            int iter = 0;
                            while (!cancellationSource.Token.IsCancellationRequested)
                            {
                                worker.PrepareBatch(iter, cancellationSource.Token);
                                iter++;
                            }
                        }
                        else
                        {
                            // Standard mode: prepare fixed number of batches
                            int batchesPerThread = options.NIterations / numThreads;
                            int extraBatches = options.NIterations % numThreads;
                            int batchesToPrepare = batchesPerThread + (threadId < extraBatches ? 1 : 0);

                            for (int iter = 0; iter < batchesToPrepare; iter++)
                            {
                                cancellationSource.Token.ThrowIfCancellationRequested();
                                worker.PrepareBatch(iter, cancellationSource.Token);
                            }
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        // Expected during cancellation
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Preparation thread {threadId} error: {ex.Message}");
                        cancellationSource.Cancel();
                        throw;
                    }
                }, cancellationSource.Token);

                tasks.Add(task);
            }

            return tasks;
        }

        private void WaitForPreparationThreads(List<Task> tasks)
        {
            try
            {
                Task.WaitAll(tasks.ToArray());
            }
            catch (AggregateException)
            {
                // Ignore exceptions from cancelled tasks
            }
        }
    }
}
