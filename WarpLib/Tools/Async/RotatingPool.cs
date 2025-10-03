using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using ZLinq;

namespace Warp.Tools.Async
{
    /// <summary>
    /// FIFO rotating pool for memory-constrained scenarios.
    /// Keeps limited number of items in memory, rotates on FIFO basis.
    /// Background preloader enables non-blocking rotation.
    ///
    /// IMPORTANT: LoadFunc must call GPU.SetDevice() if it performs GPU operations,
    /// as the preloader thread defaults to GPU 0.
    /// </summary>
    public class RotatingPool<TItem, TMetadata> : IDisposable
        where TItem : IDisposable
    {
        public delegate TItem LoadFunc(TMetadata metadata, CancellationToken ct);

        private class LoadedEntry
        {
            public int MetadataIndex { get; set; }
            public TItem Item { get; set; }
        }

        private class PreloadedEntry
        {
            public int MetadataIndex { get; set; }
            public TItem Item { get; set; }
        }

        private readonly List<TMetadata> allMetadata;
        private readonly int maxLoaded;
        private readonly LoadFunc loadFunction;
        private readonly int gpuDevice;

        private readonly LoadedEntry[] loadedPool;
        private readonly Queue<int> loadOrder; // FIFO queue of pool slot indices
        private readonly object poolLock = new object();
        private readonly Random rand = new Random(123);

        // Background preloader
        private readonly BoundedQueue<PreloadedEntry> preloadedQueue;
        private readonly CancellationTokenSource preloaderCancellation;
        private readonly System.Threading.Tasks.Task preloaderTask;

        /// <summary>
        /// Creates rotating pool with background preloading
        /// </summary>
        /// <param name="allMetadata">Complete list of all items' metadata</param>
        /// <param name="maxLoaded">Maximum items to keep in memory</param>
        /// <param name="loadFunction">Function to load item from metadata. Must handle GPU.SetDevice() if needed.</param>
        /// <param name="preloadCapacity">Number of items to keep preloaded (usually 2)</param>
        /// <param name="gpuDevice">GPU device ID for loader thread. -1 to not set device.</param>
        public RotatingPool(
            List<TMetadata> allMetadata,
            int maxLoaded,
            LoadFunc loadFunction,
            int preloadCapacity = 2,
            int gpuDevice = -1)
        {
            this.allMetadata = allMetadata;
            this.maxLoaded = Math.Min(maxLoaded, allMetadata.Count);
            this.loadFunction = loadFunction;
            this.gpuDevice = gpuDevice;

            this.loadedPool = new LoadedEntry[this.maxLoaded];
            this.loadOrder = new Queue<int>();

            // Load initial pool
            Console.Write($"Loading initial pool ({this.maxLoaded} items)");
            for (int i = 0; i < this.maxLoaded; i++)
            {
                loadedPool[i] = new LoadedEntry
                {
                    MetadataIndex = i,
                    Item = loadFunction(allMetadata[i], CancellationToken.None)
                };
                loadOrder.Enqueue(i);

                // Show progress
                if ((i + 1) % Math.Max(1, this.maxLoaded / 10) == 0 || i == this.maxLoaded - 1)
                    Console.Write($".");
            }
            Console.WriteLine($" Done.");

            // Start background preloader
            preloadedQueue = new BoundedQueue<PreloadedEntry>(preloadCapacity);
            preloaderCancellation = new CancellationTokenSource();
            preloaderTask = System.Threading.Tasks.Task.Run(() => PreloaderThread());
        }

        /// <summary>
        /// Thread-safe access to item by pool index (0 to maxLoaded-1)
        /// </summary>
        public void GetItem(int poolIndex, out TItem item)
        {
            lock (poolLock)
            {
                if (poolIndex < 0 || poolIndex >= maxLoaded)
                    throw new ArgumentException($"Pool index {poolIndex} out of range [0, {maxLoaded})");

                item = loadedPool[poolIndex].Item;
            }
        }

        /// <summary>
        /// Gets metadata index for a pool slot
        /// </summary>
        public int GetMetadataIndex(int poolIndex)
        {
            lock (poolLock)
            {
                return loadedPool[poolIndex].MetadataIndex;
            }
        }

        /// <summary>
        /// Rotates oldest item out, replacing with preloaded item (non-blocking).
        /// If no preloaded item ready, skips rotation to keep processing going.
        /// </summary>
        public void RotateOldest()
        {
            // If pool covers all items, no need to rotate
            if (allMetadata.Count <= maxLoaded)
                return;

            // Try non-blocking get of preloaded item
            if (!preloadedQueue.TryDequeue(out PreloadedEntry newEntry))
            {
                // No preloaded item ready, skip rotation to avoid blocking
                return;
            }

            lock (poolLock)
            {
                // Get oldest slot
                int oldestSlot = loadOrder.Dequeue();
                LoadedEntry oldEntry = loadedPool[oldestSlot];

                // Dispose old item
                oldEntry.Item?.Dispose();

                // Instant swap with preloaded
                loadedPool[oldestSlot] = new LoadedEntry
                {
                    MetadataIndex = newEntry.MetadataIndex,
                    Item = newEntry.Item
                };

                // Re-enqueue as newest
                loadOrder.Enqueue(oldestSlot);
            }
        }

        /// <summary>
        /// Gets current size of loaded pool
        /// </summary>
        public int LoadedCount => maxLoaded;

        /// <summary>
        /// Gets total number of items in dataset
        /// </summary>
        public int TotalCount => allMetadata.Count;

        private void PreloaderThread()
        {
            try
            {
                // Set GPU device for this thread if specified
                if (gpuDevice >= 0)
                    GPU.SetDevice(gpuDevice);

                while (!preloaderCancellation.Token.IsCancellationRequested)
                {
                    // Select random unloaded item
                    int metadataIndex = SelectRandomUnloadedItem();
                    if (metadataIndex < 0)
                    {
                        // All items are loaded, wait a bit
                        Thread.Sleep(100);
                        continue;
                    }

                    // Load item
                    TItem item = loadFunction(allMetadata[metadataIndex], preloaderCancellation.Token);

                    var entry = new PreloadedEntry
                    {
                        MetadataIndex = metadataIndex,
                        Item = item
                    };

                    // This blocks if preload queue is full
                    preloadedQueue.Enqueue(entry, preloaderCancellation.Token);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
            }
            finally
            {
                preloadedQueue.CompleteAdding();
            }
        }

        private int SelectRandomUnloadedItem()
        {
            lock (poolLock)
            {
                // Get set of currently loaded indices
                var loadedIndices = loadedPool.Select(e => e.MetadataIndex).ToHashSet();

                // Get set of preloaded indices
                var preloadedIndices = new HashSet<int>();
                PreloadedEntry temp;
                while (preloadedQueue.TryDequeue(out temp, timeoutMs: 0))
                {
                    preloadedIndices.Add(temp.MetadataIndex);
                    // Re-add for later consumption
                    preloadedQueue.TryEnqueue(temp, timeoutMs: 0);
                }

                // Find available indices
                var availableIndices = new List<int>();
                for (int i = 0; i < allMetadata.Count; i++)
                {
                    if (!loadedIndices.Contains(i) && !preloadedIndices.Contains(i))
                        availableIndices.Add(i);
                }

                if (availableIndices.Count == 0)
                    return -1;

                return availableIndices[rand.Next(availableIndices.Count)];
            }
        }

        public void Dispose()
        {
            preloaderCancellation?.Cancel();

            try
            {
                preloaderTask?.Wait(1000);
            }
            catch { }

            preloaderCancellation?.Dispose();
            preloadedQueue?.Dispose();

            // Dispose all loaded items
            if (loadedPool != null)
            {
                foreach (var entry in loadedPool)
                {
                    entry?.Item?.Dispose();
                }
            }
        }
    }
}
