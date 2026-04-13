using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace Warp.Tools.Async
{
    /// <summary>
    /// Thread-safe bounded blocking collection with automatic backpressure.
    /// Producers block when queue is full, consumers block when empty.
    /// </summary>
    public class BoundedQueue<T> : IDisposable
    {
        private readonly BlockingCollection<T> queue;

        /// <summary>
        /// Creates a bounded queue with specified capacity
        /// </summary>
        /// <param name="capacity">Maximum number of items in queue. Producers block when full.</param>
        public BoundedQueue(int capacity)
        {
            queue = new BlockingCollection<T>(capacity);
        }

        /// <summary>
        /// Adds item to queue. Blocks if queue is full until space available.
        /// </summary>
        public void Enqueue(T item, CancellationToken cancellationToken = default)
        {
            // Use TryAdd with short timeout in a loop to ensure cancellation is checked frequently
            // Don't pass cancellation token to TryAdd to avoid it throwing during timeout
            while (!cancellationToken.IsCancellationRequested)
            {
                if (queue.TryAdd(item, 100))
                    return;

                // Check if queue was marked complete (shouldn't add anymore)
                if (queue.IsAddingCompleted)
                    throw new InvalidOperationException("Cannot add to completed queue");
            }
            cancellationToken.ThrowIfCancellationRequested();
        }

        /// <summary>
        /// Removes and returns item from queue. Blocks if queue is empty until item available.
        /// Returns default(T) if CompleteAdding() was called and queue is empty.
        /// </summary>
        public T Dequeue(CancellationToken cancellationToken = default)
        {
            // Use TryTake with short timeout in a loop to ensure cancellation is checked frequently
            // Don't pass cancellation token to TryTake to avoid it throwing during timeout
            while (!cancellationToken.IsCancellationRequested)
            {
                if (queue.TryTake(out T item, 100))
                    return item;

                // Check if adding is complete and queue is empty
                if (queue.IsCompleted)
                    return default(T);
            }
            cancellationToken.ThrowIfCancellationRequested();
            return default(T);
        }

        /// <summary>
        /// Attempts to add item without blocking
        /// </summary>
        public bool TryEnqueue(T item, int timeoutMs = 0)
        {
            return queue.TryAdd(item, timeoutMs);
        }

        /// <summary>
        /// Attempts to remove item without blocking
        /// </summary>
        public bool TryDequeue(out T item, int timeoutMs = 0)
        {
            return queue.TryTake(out item, timeoutMs);
        }

        /// <summary>
        /// Signals that no more items will be added.
        /// Consumers will finish processing remaining items then return.
        /// </summary>
        public void CompleteAdding()
        {
            queue.CompleteAdding();
        }

        /// <summary>
        /// Returns enumerable that blocks until items available.
        /// Exits when CompleteAdding() called and queue empty.
        /// </summary>
        public IEnumerable<T> GetConsumingEnumerable()
        {
            return queue.GetConsumingEnumerable();
        }

        /// <summary>
        /// Returns enumerable that blocks until items available.
        /// Exits when CompleteAdding() called and queue empty, or when cancellation requested.
        /// </summary>
        public IEnumerable<T> GetConsumingEnumerable(CancellationToken cancellationToken)
        {
            return queue.GetConsumingEnumerable(cancellationToken);
        }

        public bool IsAddingCompleted => queue.IsAddingCompleted;
        public bool IsCompleted => queue.IsCompleted;
        public int Count => queue.Count;

        public void Dispose()
        {
            queue?.Dispose();
        }
    }
}
