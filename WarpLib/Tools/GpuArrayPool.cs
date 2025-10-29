using System;
using System.Collections.Concurrent;
using Warp;

namespace Warp.Tools
{
    /// <summary>
    /// Provides a thread-safe pool for reusing GPU memory allocations to reduce allocation overhead.
    /// Tracks rented pointers to enable simplified return without size/device parameters.
    /// Maintains separate pools for each GPU device.
    /// </summary>
    public static class GpuArrayPool
    {
        /// <summary>
        /// Information about a rented GPU memory allocation.
        /// </summary>
        private readonly struct RentedInfo
        {
            public readonly long SizeInBytes;
            public readonly int DeviceId;

            public RentedInfo(long sizeInBytes, int deviceId)
            {
                SizeInBytes = sizeInBytes;
                DeviceId = deviceId;
            }
        }

        // Stores stacks of available GPU memory pointers, keyed by device ID, then by size in bytes.
        private static readonly ConcurrentDictionary<int, ConcurrentDictionary<long, ConcurrentStack<IntPtr>>> SDevicePools = new();
        
        // Tracks currently rented pointers with their size and device info.
        private static readonly ConcurrentDictionary<IntPtr, RentedInfo> SRentedPointers = new();

        /// <summary>
        /// Rents GPU memory of the specified size from the pool for the current device.
        /// If no suitable memory is available, a new allocation is made.
        /// </summary>
        /// <param name="sizeInBytes">The required size in bytes.</param>
        /// <returns>An IntPtr to the allocated GPU memory.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size is negative.</exception>
        public static IntPtr Rent(long sizeInBytes)
        {
            return Rent(sizeInBytes, GPU.GetDevice());
        }

        /// <summary>
        /// Rents GPU memory of the specified size from the pool for the specified device.
        /// If no suitable memory is available, a new allocation is made.
        /// </summary>
        /// <param name="sizeInBytes">The required size in bytes.</param>
        /// <param name="deviceId">The GPU device ID.</param>
        /// <returns>An IntPtr to the allocated GPU memory.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size is negative.</exception>
        public static IntPtr Rent(long sizeInBytes, int deviceId)
        {
            if (sizeInBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(sizeInBytes), "Size cannot be negative or zero.");

            var devicePool = SDevicePools.GetOrAdd(deviceId, _ => new ConcurrentDictionary<long, ConcurrentStack<IntPtr>>());
            var stack = devicePool.GetOrAdd(sizeInBytes, _ => new ConcurrentStack<IntPtr>());
            
            IntPtr ptr;
            if (stack.TryPop(out ptr))
            {
                // Remove from tracking if it was previously rented (reused pointer)
                SRentedPointers.TryRemove(ptr, out _);
            }
            else
            {
                // Need to allocate on the correct device
                int currentDevice = GPU.GetDevice();
                try
                {
                    if (currentDevice != deviceId)
                        GPU.SetDevice(deviceId);
                    ptr = GPU.MallocDevice(sizeInBytes);
                }
                finally
                {
                    if (currentDevice != deviceId)
                        GPU.SetDevice(currentDevice);
                }
            }

            // Track the rented pointer
            SRentedPointers[ptr] = new RentedInfo(sizeInBytes, deviceId);
            return ptr;
        }

        /// <summary>
        /// Returns GPU memory to the pool for the current device.
        /// Note: This method does not validate that the memory was rented from this pool.
        /// Double returns are ignored silently to prevent exceptions.
        /// </summary>
        /// <param name="ptr">The GPU memory pointer to return.</param>
        /// <param name="sizeInBytes">The size of the memory allocation in bytes.</param>
        public static void Return(IntPtr ptr, long sizeInBytes)
        {
            Return(ptr, sizeInBytes, GPU.GetDevice());
        }

        /// <summary>
        /// Returns GPU memory to the pool using tracked information.
        /// This is the preferred method as it automatically determines size and device.
        /// </summary>
        /// <param name="ptr">The GPU memory pointer to return.</param>
        /// <returns>True if the pointer was tracked and returned successfully, false otherwise.</returns>
        public static void Return(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                throw new ArgumentNullException(nameof(ptr), "Pointer cannot be null."); 

            if (!SRentedPointers.TryRemove(ptr, out RentedInfo info))
                throw new ArgumentException("Pointer was not tracked.", nameof(ptr));

            var devicePool = SDevicePools.GetOrAdd(info.DeviceId, _ => new ConcurrentDictionary<long, ConcurrentStack<IntPtr>>());
            var stack = devicePool.GetOrAdd(info.SizeInBytes, _ => new ConcurrentStack<IntPtr>());
            stack.Push(ptr);
        }

        /// <summary>
        /// Returns GPU memory to the pool for the specified device.
        /// Note: This method does not validate that the memory was rented from this pool.
        /// Double returns are ignored silently to prevent exceptions.
        /// </summary>
        /// <param name="ptr">The GPU memory pointer to return.</param>
        /// <param name="sizeInBytes">The size of the memory allocation in bytes.</param>
        /// <param name="deviceId">The GPU device ID.</param>
        public static void Return(IntPtr ptr, long sizeInBytes, int deviceId)
        {
            if (ptr == IntPtr.Zero || sizeInBytes <= 0)
                return;

            // Remove from tracking if present
            SRentedPointers.TryRemove(ptr, out _);

            var devicePool = SDevicePools.GetOrAdd(deviceId, _ => new ConcurrentDictionary<long, ConcurrentStack<IntPtr>>());
            var stack = devicePool.GetOrAdd(sizeInBytes, _ => new ConcurrentStack<IntPtr>());
            stack.Push(ptr);
        }

        /// <summary>
        /// Rents GPU memory wrapped in a disposable struct for easy cleanup with 'using'.
        /// </summary>
        /// <param name="sizeInBytes">The required size in bytes.</param>
        /// <returns>A DisposableGPUArray wrapping the rented GPU memory.</returns>
        public static DisposableGPUArray RentDisposable(long sizeInBytes)
        {
            int deviceId = GPU.GetDevice();
            return new DisposableGPUArray(Rent(sizeInBytes, deviceId), sizeInBytes, deviceId);
        }

        /// <summary>
        /// Rents GPU memory wrapped in a disposable struct for easy cleanup with 'using'.
        /// </summary>
        /// <param name="sizeInBytes">The required size in bytes.</param>
        /// <param name="deviceId">The GPU device ID.</param>
        /// <returns>A DisposableGPUArray wrapping the rented GPU memory.</returns>
        public static DisposableGPUArray RentDisposable(long sizeInBytes, int deviceId)
        {
            return new DisposableGPUArray(Rent(sizeInBytes, deviceId), sizeInBytes, deviceId);
        }

        /// <summary>
        /// Clears all GPU memory currently held in the pool for the specified size across all devices.
        /// WARNING: This will free all pooled GPU memory of this size!
        /// </summary>
        /// <param name="sizeInBytes">The size in bytes of memory to clear from the pool.</param>
        public static void Clear(long sizeInBytes)
        {
            if (sizeInBytes <= 0) 
                return;

            foreach (var devicePool in SDevicePools.Values)
                if (devicePool.TryRemove(sizeInBytes, out var stack))
                    // Free all GPU memory in the stack
                    while (stack.TryPop(out IntPtr ptr))
                        if (ptr != IntPtr.Zero)
                            GPU.FreeDevice(ptr);
        }

        /// <summary>
        /// Clears all GPU memory currently held in the pool for the specified size on a specific device.
        /// WARNING: This will free all pooled GPU memory of this size on the specified device!
        /// </summary>
        /// <param name="sizeInBytes">The size in bytes of memory to clear from the pool.</param>
        /// <param name="deviceId">The GPU device ID.</param>
        public static void Clear(long sizeInBytes, int deviceId)
        {
            if (sizeInBytes <= 0) 
                return;

            if (SDevicePools.TryGetValue(deviceId, out var devicePool) && 
                devicePool.TryRemove(sizeInBytes, out var stack))
                // Free all GPU memory in the stack
                while (stack.TryPop(out IntPtr ptr))
                    if (ptr != IntPtr.Zero)
                        GPU.FreeDevice(ptr);
        }

        /// <summary>
        /// Clears all GPU memory of all sizes currently held in the pool across all devices.
        /// WARNING: This will free all pooled GPU memory!
        /// </summary>
        public static void Clear()
        {
            foreach (var deviceKvp in SDevicePools)
                foreach (var sizeKvp in deviceKvp.Value)
                    while (sizeKvp.Value.TryPop(out IntPtr ptr))
                        if (ptr != IntPtr.Zero)
                            GPU.FreeDevice(ptr);
            
            SDevicePools.Clear();
        }

        /// <summary>
        /// Clears all GPU memory of all sizes currently held in the pool for a specific device.
        /// WARNING: This will free all pooled GPU memory on the specified device!
        /// </summary>
        /// <param name="deviceId">The GPU device ID.</param>
        public static void Clear(int deviceId)
        {
            if (SDevicePools.TryRemove(deviceId, out var devicePool))
                foreach (var sizeKvp in devicePool)
                    while (sizeKvp.Value.TryPop(out IntPtr ptr))
                        if (ptr != IntPtr.Zero)
                            GPU.FreeDevice(ptr);
        }

        /// <summary>
        /// A ref struct wrapper around rented GPU memory that ensures it is returned
        /// to the pool when disposed (typically via a 'using' statement).
        /// </summary>
        public ref struct DisposableGPUArray
        {
            private readonly IntPtr _ptr;
            private readonly long _sizeInBytes;
            private readonly int _deviceId;
            private bool _disposed;

            internal DisposableGPUArray(IntPtr ptr, long sizeInBytes, int deviceId)
            {
                _ptr = ptr;
                _sizeInBytes = sizeInBytes;
                _deviceId = deviceId;
                _disposed = false;
            }

            /// <summary>
            /// Gets the underlying GPU memory pointer.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public IntPtr Ptr
            {
                get
                {
                    if (_disposed) 
                        throw new ObjectDisposedException(nameof(DisposableGPUArray));
                    return _ptr;
                }
            }

            /// <summary>
            /// Gets the size in bytes of the allocated GPU memory.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public long SizeInBytes
            {
                get
                {
                    if (_disposed) 
                        throw new ObjectDisposedException(nameof(DisposableGPUArray));
                    return _sizeInBytes;
                }
            }

            /// <summary>
            /// Implicit conversion to IntPtr.
            /// </summary>
            public static implicit operator IntPtr(in DisposableGPUArray wrapper)
            {
                if (wrapper._disposed) 
                    throw new ObjectDisposedException(nameof(DisposableGPUArray));
                return wrapper._ptr;
            }

            /// <summary>
            /// Gets the GPU device ID for this memory allocation.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public int DeviceId
            {
                get
                {
                    if (_disposed) 
                        throw new ObjectDisposedException(nameof(DisposableGPUArray));
                    return _deviceId;
                }
            }

            /// <summary>
            /// Disposes the wrapper, returning the underlying GPU memory to the pool.
            /// </summary>
            public void Dispose()
            {
                if (_disposed)
                    return; // Allow multiple disposals silently

                if (_ptr != IntPtr.Zero)
                    GpuArrayPool.Return(_ptr, _sizeInBytes, _deviceId);

                _disposed = true;
            }
        }
    }
}