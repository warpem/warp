using System;
using System.Collections.Concurrent;
using System.Threading;

namespace Noise2Map
{
    /// <summary>
    /// Thread-safe bounded queue for training batches with producer-consumer pattern support
    /// </summary>
    public class ConcurrentTrainingQueue : IDisposable
    {
        private readonly BlockingCollection<TrainingBatch> queue;
        private readonly int capacity;

        /// <summary>
        /// Creates a new bounded queue with the specified maximum capacity
        /// </summary>
        /// <param name="maxCapacity">Maximum number of batches to buffer (prevents memory overflow)</param>
        public ConcurrentTrainingQueue(int maxCapacity)
        {
            capacity = maxCapacity;
            queue = new BlockingCollection<TrainingBatch>(new ConcurrentQueue<TrainingBatch>(), maxCapacity);
        }

        /// <summary>
        /// Adds a batch to the queue. Blocks if the queue is at capacity until space is available.
        /// </summary>
        public void Enqueue(TrainingBatch batch, CancellationToken cancellationToken = default)
        {
            queue.Add(batch, cancellationToken);
        }

        /// <summary>
        /// Attempts to retrieve a batch from the queue. Blocks until a batch is available or production is complete.
        /// </summary>
        /// <returns>The next batch, or null if production is complete and queue is empty</returns>
        public TrainingBatch Dequeue(CancellationToken cancellationToken = default)
        {
            if (queue.TryTake(out TrainingBatch batch, Timeout.Infinite, cancellationToken))
                return batch;
            return null;
        }

        /// <summary>
        /// Signals that no more batches will be added to the queue
        /// </summary>
        public void CompleteAdding()
        {
            queue.CompleteAdding();
        }

        /// <summary>
        /// Returns true if production is complete and no more batches are available
        /// </summary>
        public bool IsCompleted => queue.IsCompleted;

        /// <summary>
        /// Current number of batches in the queue
        /// </summary>
        public int Count => queue.Count;

        public void Dispose()
        {
            queue?.Dispose();
        }
    }
}
