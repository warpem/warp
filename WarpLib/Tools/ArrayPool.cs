using System;
using System.Collections.Concurrent;
using System.Linq;
using ZLinq;

namespace Warp.Tools
{
    /// <summary>
    /// Provides a thread-safe pool for reusing arrays to reduce GC pressure.
    /// This implementation prioritizes simplicity and avoids memory leaks over validation.
    /// Arrays are not tracked after renting, preventing memory leaks if not returned.
    /// </summary>
    /// <typeparam name="T">The type of elements in the arrays.</typeparam>
    public static class ArrayPool<T>
    {
        // Stores stacks of available arrays, keyed by size.
        private static readonly ConcurrentDictionary<int, ConcurrentStack<T[]>> SAvailableArrays = new();

        /// <summary>
        /// Rents an array of the specified size from the pool.
        /// If no suitable array is available, a new one is allocated.
        /// </summary>
        /// <param name="size">The required size of the array.</param>
        /// <returns>An array of type T[] with the specified size.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size is negative.</exception>
        public static T[] Rent(int size)
        {
            if (size < 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative.");

            if (size == 0)
                return [];

            var stack = SAvailableArrays.GetOrAdd(size, _ => new ConcurrentStack<T[]>());
            return stack.TryPop(out T[] array) ? array : new T[size];
        }

        /// <summary>
        /// Rents multiple arrays of the specified size from the pool.
        /// </summary>
        /// <param name="size">The required size of each array.</param>
        /// <param name="count">The number of arrays to rent.</param>
        /// <returns>An array containing the rented T[] arrays.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size or count is negative.</exception>
        public static T[][] RentMultiple(int size, int count)
        {
            if (size < 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative.");
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");

            if (count == 0)
                return Array.Empty<T[]>();
            if (size == 0)
            {
                var emptyResults = new T[count][];
                for (int i = 0; i < count; i++)
                    emptyResults[i] = [];
                return emptyResults;
            }

            var results = new T[count][];
            var stack = SAvailableArrays.GetOrAdd(size, _ => new ConcurrentStack<T[]>());

            for (int i = 0; i < count; i++)
                results[i] = stack.TryPop(out T[] array) ? array : new T[size];

            return results;
        }

        /// <summary>
        /// Returns an array to the pool.
        /// Note: This method does not validate that the array was rented from this pool.
        /// Double returns are ignored silently to prevent exceptions.
        /// </summary>
        /// <param name="array">The array to return.</param>
        /// <param name="clearArray">If true, the array contents will be cleared before returning to pool.</param>
        public static void Return(T[] array, bool clearArray = true)
        {
            if (array == null || array.Length == 0)
                return;

            if (clearArray && typeof(T).IsPrimitive)
                Array.Clear(array, 0, array.Length);

            var stack = SAvailableArrays.GetOrAdd(array.Length, _ => new ConcurrentStack<T[]>());
            stack.Push(array);
        }

        /// <summary>
        /// Returns multiple arrays to the pool.
        /// </summary>
        /// <param name="arrays">The arrays to return.</param>
        /// <param name="clearArray">If true, the array contents will be cleared before returning to pool.</param>
        public static void ReturnMultiple(T[][] arrays, bool clearArray = true)
        {
            if (arrays == null)
                return;

            for (int i = 0; i < arrays.Length; i++)
                Return(arrays[i], clearArray);
        }

        /// <summary>
        /// Rents an array wrapped in a disposable struct for easy cleanup with 'using'.
        /// </summary>
        /// <param name="size">The required size of the array.</param>
        /// <param name="clearOnDispose">If true, the array will be cleared when disposed.</param>
        /// <returns>A DisposableArray wrapping the rented array.</returns>
        public static DisposableArray RentDisposable(int size, bool clearOnDispose = true)
        {
            return new DisposableArray(Rent(size), clearOnDispose);
        }

        /// <summary>
        /// Clears all arrays currently held in the pool for the specified size.
        /// </summary>
        /// <param name="size">The size of arrays to clear from the pool.</param>
        public static void Clear(int size)
        {
            if (size <= 0) 
                return;
            
            SAvailableArrays.TryRemove(size, out _);
        }

        /// <summary>
        /// Clears all arrays of all sizes currently held in the pool.
        /// </summary>
        public static void Clear()
        {
            SAvailableArrays.Clear();
        }

        /// <summary>
        /// Prints information about all currently available arrays in the pool.
        /// </summary>
        public static void PrintPoolInfo()
        {
            Console.WriteLine($"Array Pool Status for type {typeof(T).Name}:");

            if (SAvailableArrays.IsEmpty)
            {
                Console.WriteLine("  Pool is empty");
                return;
            }

            var sizes = SAvailableArrays.Keys.OrderBy(k => k).ToList();
            foreach (var size in sizes)
            {
                if (SAvailableArrays.TryGetValue(size, out var stack))
                {
                    Console.WriteLine($"  Size {size,8}: {stack.Count,6} arrays available");
                }
            }
        }

        /// <summary>
        /// A ref struct wrapper around a rented array that ensures it is returned
        /// to the pool when disposed (typically via a 'using' statement).
        /// </summary>
        public ref struct DisposableArray
        {
            private readonly T[] _array;
            private readonly bool _clearOnDispose;
            private bool _disposed;

            internal DisposableArray(T[] array, bool clearOnDispose)
            {
                _array = array;
                _clearOnDispose = clearOnDispose;
                _disposed = false;
            }

            /// <summary>
            /// Gets the underlying array.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public T[] Array
            {
                get
                {
                    if (_disposed) 
                        throw new ObjectDisposedException(nameof(DisposableArray));
                    return _array;
                }
            }

            /// <summary>
            /// Implicit conversion to T[].
            /// </summary>
            public static implicit operator T[](in DisposableArray wrapper)
            {
                if (wrapper._disposed) 
                    throw new ObjectDisposedException(nameof(DisposableArray));
                return wrapper._array;
            }

            /// <summary>
            /// Implicit conversion to ReadOnlySpan&lt;T&gt;.
            /// </summary>
            public static implicit operator ReadOnlySpan<T>(in DisposableArray wrapper)
            {
                if (wrapper._disposed) 
                    throw new ObjectDisposedException(nameof(DisposableArray));
                return wrapper._array;
            }

            /// <summary>
            /// Implicit conversion to Span&lt;T&gt;.
            /// </summary>
            public static implicit operator Span<T>(in DisposableArray wrapper)
            {
                if (wrapper._disposed) 
                    throw new ObjectDisposedException(nameof(DisposableArray));
                return wrapper._array;
            }

            /// <summary>
            /// Disposes the wrapper, returning the underlying array to the pool.
            /// </summary>
            public void Dispose()
            {
                if (_disposed)
                    return; // Allow multiple disposals silently

                if (_array != null)
                    ArrayPool<T>.Return(_array, _clearOnDispose);

                _disposed = true;
            }
        }
    }
}