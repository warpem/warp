using System;
using System.Linq;
using Warp;

namespace Noise2Map
{
    /// <summary>
    /// Manages GPU device setup and validation
    /// </summary>
    public static class GpuManager
    {
        /// <summary>
        /// Initializes and validates GPU devices for training and preprocessing
        /// </summary>
        /// <param name="options">Configuration options containing GPU IDs</param>
        public static void Initialize(Options options)
        {
            int nDevices = GPU.GetDeviceCount();

            ValidateGpuIds(options, nDevices);
            AdjustGpuIds(options, nDevices);

            GPU.SetDevice(options.GPUPreprocess);
        }

        /// <summary>
        /// Validates that requested GPU IDs exist on the system
        /// </summary>
        private static void ValidateGpuIds(Options options, int nDevices)
        {
            if (options.GPUNetwork.Any(id => id >= nDevices))
            {
                Console.WriteLine($"Requested GPU ID ({options.GPUNetwork.First(id => id >= nDevices)}) that isn't present on this system.");
            }
        }

        /// <summary>
        /// Adjusts GPU IDs to valid values if needed
        /// </summary>
        private static void AdjustGpuIds(Options options, int nDevices)
        {
            if (options.GPUPreprocess >= nDevices)
            {
                options.GPUPreprocess = Math.Min(options.GPUPreprocess, nDevices - 1);
            }
        }
    }
}
