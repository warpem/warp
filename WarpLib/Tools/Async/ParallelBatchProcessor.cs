using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Warp.Tools.Async
{
    /// <summary>
    /// Generic parallel batch processor with per-worker GPU assignment.
    /// Multiple workers process batches in parallel, results collected in order.
    ///
    /// IMPORTANT: ProcessFunc receives workerIndex. Use it to set GPU device:
    /// GPU.SetDevice(gpuDevices[workerIndex]) before any GPU operations.
    /// </summary>
    public class ParallelBatchProcessor<TInput, TOutput> : IDisposable
    {
        public delegate TOutput ProcessFunc(TInput input, int workerIndex, CancellationToken ct);

        private readonly ProcessFunc processor;
        private readonly int numWorkers;
        private readonly int[] gpuDevices;

        private readonly BoundedQueue<TInput> inputQueue;
        private readonly BoundedQueue<TOutput> outputQueue;
        private readonly Task[] workerTasks;
        private readonly CancellationTokenSource cancellationSource;

        /// <summary>
        /// Creates parallel batch processor
        /// </summary>
        /// <param name="processor">Function to process each input. Receives workerIndex for GPU assignment.</param>
        /// <param name="numWorkers">Number of parallel worker threads</param>
        /// <param name="queueCapacity">Capacity of input/output queues</param>
        /// <param name="gpuDevices">GPU device IDs for workers. If null, workers must handle GPU.SetDevice() themselves.</param>
        public ParallelBatchProcessor(
            ProcessFunc processor,
            int numWorkers,
            int queueCapacity,
            int[] gpuDevices = null)
        {
            this.processor = processor;
            this.numWorkers = numWorkers;
            this.gpuDevices = gpuDevices;

            this.inputQueue = new BoundedQueue<TInput>(queueCapacity);
            this.outputQueue = new BoundedQueue<TOutput>(queueCapacity);
            this.cancellationSource = new CancellationTokenSource();

            // Start worker threads
            workerTasks = new Task[numWorkers];
            for (int i = 0; i < numWorkers; i++)
            {
                int workerIndex = i;
                workerTasks[i] = Task.Run(() => WorkerThread(workerIndex));
            }
        }

        /// <summary>
        /// Enqueues batch for processing. Blocks if queue full.
        /// </summary>
        public void EnqueueBatch(TInput[] inputs, CancellationToken ct = default)
        {
            foreach (var input in inputs)
            {
                inputQueue.Enqueue(input, ct);
            }
        }

        /// <summary>
        /// Enqueues single input for processing. Blocks if queue full.
        /// </summary>
        public void Enqueue(TInput input, CancellationToken ct = default)
        {
            inputQueue.Enqueue(input, ct);
        }

        /// <summary>
        /// Dequeues processed output. Blocks if queue empty.
        /// Returns default(TOutput) when CompleteAdding() called and queue empty.
        /// </summary>
        public TOutput Dequeue(CancellationToken ct = default)
        {
            return outputQueue.Dequeue(ct);
        }

        /// <summary>
        /// Dequeues batch of processed outputs. Blocks until all available.
        /// </summary>
        public TOutput[] DequeueBatch(int count, CancellationToken ct = default)
        {
            TOutput[] results = new TOutput[count];
            for (int i = 0; i < count; i++)
            {
                results[i] = outputQueue.Dequeue(ct);
            }
            return results;
        }

        /// <summary>
        /// Returns enumerable that yields processed outputs as they become available
        /// </summary>
        public System.Collections.Generic.IEnumerable<TOutput> GetConsumingEnumerable()
        {
            return outputQueue.GetConsumingEnumerable();
        }

        /// <summary>
        /// Signals no more inputs will be added. Workers finish processing then exit.
        /// </summary>
        public void CompleteAdding()
        {
            inputQueue.CompleteAdding();
        }

        /// <summary>
        /// Waits for all workers to finish processing
        /// </summary>
        public void WaitForCompletion()
        {
            Task.WaitAll(workerTasks);
        }

        private void WorkerThread(int workerIndex)
        {
            try
            {
                // Set GPU device for this worker if specified
                if (gpuDevices != null && gpuDevices.Length > workerIndex)
                {
                    GPU.SetDevice(gpuDevices[workerIndex]);
                }

                foreach (var input in inputQueue.GetConsumingEnumerable())
                {
                    // Process input
                    TOutput output = processor(input, workerIndex, cancellationSource.Token);

                    // Enqueue result
                    outputQueue.Enqueue(output, cancellationSource.Token);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during cancellation
            }
            finally
            {
                // When all workers done, complete output queue
                if (workerTasks.All(t => t.IsCompleted))
                    outputQueue.CompleteAdding();
            }
        }

        public void Dispose()
        {
            cancellationSource?.Cancel();

            try
            {
                Task.WaitAll(workerTasks, 1000);
            }
            catch { }

            cancellationSource?.Dispose();
            inputQueue?.Dispose();
            outputQueue?.Dispose();
        }
    }
}
