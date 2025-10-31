using System;
using System.Diagnostics;

namespace Warp.Tools.Async
{
    /// <summary>
    /// Unified progress tracking with exponential moving average time estimation.
    /// Displays single updating line with completion estimate.
    /// </summary>
    public class ProgressTracker
    {
        private readonly int totalItems;
        private readonly int warmupItems;
        private readonly double emaAlpha;
        private readonly bool clearLine;

        private readonly Stopwatch watch = new Stopwatch();
        private int completedItems = 0;
        private double smoothedItemTime = 0;

        /// <summary>
        /// Creates progress tracker
        /// </summary>
        /// <param name="totalItems">Total number of items to process</param>
        /// <param name="warmupItems">Number of initial items to exclude from time estimation (e.g. GPU warmup)</param>
        /// <param name="emaAlpha">Exponential moving average smoothing factor (0-1, lower = smoother)</param>
        /// <param name="clearLine">Use VirtualConsole.ClearLastLine() for single-line updates</param>
        public ProgressTracker(
            int totalItems,
            int warmupItems = 5,
            double emaAlpha = 0.1,
            bool clearLine = true)
        {
            this.totalItems = totalItems;
            this.warmupItems = warmupItems;
            this.emaAlpha = emaAlpha;
            this.clearLine = clearLine;

            watch.Start();
        }

        /// <summary>
        /// Updates progress with current item name.
        /// Call this after completing each item.
        /// </summary>
        public void Update(string currentItemName)
        {
            completedItems++;
            double itemTime = watch.Elapsed.TotalSeconds;

            // Update smoothed time
            if (completedItems <= warmupItems)
            {
                // During warmup: just track current time, don't smooth
                smoothedItemTime = itemTime;
            }
            else if (completedItems <= warmupItems + 10)
            {
                // Post-warmup ramp: use simple average for stability
                int effectiveItem = completedItems - warmupItems;
                smoothedItemTime = (smoothedItemTime * (effectiveItem - 1) + itemTime) / effectiveItem;
            }
            else
            {
                // Steady state: exponential moving average
                smoothedItemTime = emaAlpha * itemTime + (1 - emaAlpha) * smoothedItemTime;
            }

            PrintProgress(currentItemName);
            watch.Restart();
        }

        /// <summary>
        /// Updates progress with explicit iteration number and current item name.
        /// Use this when updates are called sporadically (not every iteration).
        /// </summary>
        public void Update(int currentIteration, string currentItemName)
        {
            double itemTime = watch.Elapsed.TotalSeconds;
            int itemsSinceLastUpdate = currentIteration - completedItems;

            if (itemsSinceLastUpdate > 0)
            {
                double timePerItem = itemTime / itemsSinceLastUpdate;

                completedItems = currentIteration;

                // Update smoothed time
                if (completedItems <= warmupItems)
                {
                    // During warmup: just track current time, don't smooth
                    smoothedItemTime = timePerItem;
                }
                else if (completedItems <= warmupItems + 10)
                {
                    // Post-warmup ramp: use simple average for stability
                    int effectiveItem = completedItems - warmupItems;
                    smoothedItemTime = (smoothedItemTime * (effectiveItem - 1) + timePerItem) / effectiveItem;
                }
                else
                {
                    // Steady state: exponential moving average
                    smoothedItemTime = emaAlpha * timePerItem + (1 - emaAlpha) * smoothedItemTime;
                }

                PrintProgress(currentItemName);
            }

            watch.Restart();
        }

        /// <summary>
        /// Prints final completion message
        /// </summary>
        public void Complete()
        {
            watch.Stop();
            Console.WriteLine(); // New line after progress
        }

        private void PrintProgress(string currentItemName)
        {
            int remainingItems = totalItems - completedItems;
            double totalSecondsRemaining = smoothedItemTime * remainingItems;
            TimeSpan timeRemaining = TimeSpan.FromSeconds(totalSecondsRemaining);

            string toWrite = $"{completedItems}/{totalItems}, " +
                            (timeRemaining.Days > 0 ? (timeRemaining.Days + " days ") : "") +
                            $"{timeRemaining.Hours}:{timeRemaining.Minutes:D2}:{timeRemaining.Seconds:D2} remaining, " +
                            $"{currentItemName}";

            if (clearLine)
            {
                try
                {
                    VirtualConsole.ClearLastLine();
                    Console.Write(toWrite);
                }
                catch
                {
                    // Fallback for piped output or HPC clusters
                    Console.WriteLine(toWrite);
                }
            }
            else
            {
                Console.WriteLine(toWrite);
            }
        }
    }
}
