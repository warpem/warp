using System;
using System.Threading;
using System.Threading.Tasks;

namespace Warp.Tools.Async
{
    /// <summary>
    /// Background preloader for non-blocking access to expensive-to-load items.
    /// Maintains a small buffer of preloaded items ready for instant retrieval.
    ///
    /// IMPORTANT: If LoadFunc performs GPU operations, it must call GPU.SetDevice()
    /// as threads default to GPU 0.
    /// </summary>
    public class AsyncPreloader<T> : IDisposable
    {
        public delegate T LoadFunc(CancellationToken ct);

        private readonly BoundedQueue<T> preloadedItems;
        private readonly Task preloaderTask;
        private readonly CancellationTokenSource cancellationSource;
        private readonly int gpuDevice;

        private LoadFunc currentLoader;
        private readonly object loaderLock = new object();

        /// <summary>
        /// Creates background preloader
        /// </summary>
        /// <param name="loader">Function to load items. Must handle GPU.SetDevice() if needed.</param>
        /// <param name="capacity">Number of items to keep preloaded (usually 2)</param>
        /// <param name="gpuDevice">GPU device ID for preloader thread. -1 to not set device.</param>
        public AsyncPreloader(LoadFunc loader, int capacity = 2, int gpuDevice = -1)
        {
            this.currentLoader = loader;
            this.gpuDevice = gpuDevice;
            this.preloadedItems = new BoundedQueue<T>(capacity);
            this.cancellationSource = new CancellationTokenSource();

            preloaderTask = Task.Run(() => PreloaderThread());
        }

        /// <summary>
        /// Non-blocking attempt to get preloaded item
        /// </summary>
        /// <returns>True if item was available, false if preloader still working</returns>
        public bool TryGet(out T item)
        {
            return preloadedItems.TryDequeue(out item);
        }

        /// <summary>
        /// Blocking get of preloaded item
        /// </summary>
        public T Get(CancellationToken cancellationToken = default)
        {
            return preloadedItems.Dequeue(cancellationToken);
        }

        /// <summary>
        /// Requests a specific item to be loaded next.
        /// Clears current preload queue and starts loading requested item.
        /// </summary>
        public void RequestLoad(LoadFunc specificLoader)
        {
            lock (loaderLock)
            {
                currentLoader = specificLoader;
            }
        }

        private void PreloaderThread()
        {
            try
            {
                // Set GPU device for this thread if specified
                if (gpuDevice >= 0)
                    GPU.SetDevice(gpuDevice);

                while (!cancellationSource.Token.IsCancellationRequested)
                {
                    LoadFunc loaderToUse;
                    lock (loaderLock)
                    {
                        loaderToUse = currentLoader;
                    }

                    if (loaderToUse != null)
                    {
                        T item = loaderToUse(cancellationSource.Token);

                        // This blocks if queue is full (capacity reached)
                        preloadedItems.Enqueue(item, cancellationSource.Token);
                    }
                    else
                    {
                        // No loader specified, wait a bit
                        Thread.Sleep(10);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
            }
            finally
            {
                preloadedItems.CompleteAdding();
            }
        }

        public void Dispose()
        {
            cancellationSource?.Cancel();

            try
            {
                preloaderTask?.Wait(1000);
            }
            catch { }

            cancellationSource?.Dispose();
            preloadedItems?.Dispose();
        }
    }
}
