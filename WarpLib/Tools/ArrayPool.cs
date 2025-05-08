using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace WarpLib.Tools
{
    /// <summary>
    /// Provides a thread-safe pool for reusing large arrays to reduce GC pressure.
    /// Arrays can be optionally cleared in the background upon return.
    /// Includes a disposable wrapper for easy deterministic cleanup using 'using'.
    /// </summary>
    /// <typeparam name="T">The type of elements in the arrays.</typeparam>
    public static class ArrayPool<T>
    {
        private enum ArrayState
        {
            Available,
            Clearing
        }

        /// <summary>
        /// Wrapper for arrays stored in the pool, tracking their state (available or clearing).
        /// </summary>
        private sealed class PooledArrayWrapper
        {
            internal readonly T[] Array;
            internal volatile ArrayState State;
            internal Task ClearingTask; // Task responsible for clearing the array

            internal PooledArrayWrapper(T[] array, ArrayState state)
            {
                Array = array;
                State = state;
            }
        }

        /// <summary>
        /// Tracks the set of arrays currently rented out for a specific size and provides locking.
        /// </summary>
        private sealed class RentedArrayTracking
        {
            // Using HashSet for efficient Add/Remove/Contains checks.
            // Storing the actual array reference allows us to verify the exact instance being returned.
            internal readonly HashSet<T[]> RentedSet = new HashSet<T[]>(ReferenceEqualityComparer.Instance);
            internal readonly object SyncLock = new object(); // Lock protects access to RentedSet
        }

        // Stores stacks of available arrays, keyed by size.
        // ConcurrentStack is thread-safe for Push/TryPop.
        private static readonly ConcurrentDictionary<int, ConcurrentStack<PooledArrayWrapper>> s_availableArrays = new();

        // Tracks arrays currently rented out, keyed by size.
        private static readonly ConcurrentDictionary<int, RentedArrayTracking> s_rentedArraysInfo = new();

        // Dedicated lock for the global Clear() operation to prevent deadlocks.
        private static readonly object s_clearAllLock = new object();

        /// <summary>
        /// Rents an array of the specified size from the pool.
        /// If an appropriate array is available but being cleared, this method will block
        /// synchronously until clearing is complete.
        /// If no suitable array is available, a new one is allocated.
        /// </summary>
        /// <param name="size">The required size of the array.</param>
        /// <returns>An array of type T[] with the specified size.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size is negative.</exception>
        public static T[] Rent(int size)
        {
            if (size < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative.");
            }

            if (size == 0)
            {
                // Return the singleton empty array for size 0. No need to pool/track.
                // Reference: https://docs.microsoft.com/en-us/dotnet/api/system.array.empty
                return Array.Empty<T>();
            }

            var stack = s_availableArrays.GetOrAdd(size, _ => new ConcurrentStack<PooledArrayWrapper>());
            T[] rentedArray = null;

            while (rentedArray == null)
            {
                if (stack.TryPop(out PooledArrayWrapper wrapper))
                {
                    // Check if the array is currently being cleared
                    if (wrapper.State == ArrayState.Clearing)
                    {
                        // Synchronously wait for the clearing task to complete.
                        // This might block the current thread.
                        try
                        {
                           wrapper.ClearingTask?.Wait();
                           // Task should set state to Available upon completion (success or failure handled internally)
                           Debug.Assert(wrapper.State == ArrayState.Available, "Array state should be Available after waiting for clear.");
                        }
                        catch (Exception ex) // Catch potential exceptions from the Wait() or the task itself
                        {
                           // Failed to clear or wait. What to do?
                           // Option 1: Treat as unavailable, let the loop try again or allocate new.
                           // Option 2: Log the error and potentially re-throw or return a new array.
                           // For now, let's log (if logging is available) and let it try to allocate new below.
                           Console.Error.WriteLine($"Error waiting for array clear (size {size}): {ex.Message}");
                           // Continue the loop - will likely result in allocation below if stack becomes empty.
                           continue;
                        }
                    }

                    // If state is Available (either initially or after waiting)
                    if(wrapper.State == ArrayState.Available)
                    {
                        rentedArray = wrapper.Array;
                    }
                    else
                    {
                        // Should not happen if wait logic is correct, but as fallback, continue loop
                        throw new Exception("this should not happen");
                        //continue;
                    }
                }
                else
                {
                    // Stack is empty, allocate a new array
                    rentedArray = new T[size];
                    break; // Exit loop after allocating
                }
            }

            // Track the rented array
            var tracking = s_rentedArraysInfo.GetOrAdd(size, _ => new RentedArrayTracking());
            lock (tracking.SyncLock)
            {
                bool added = tracking.RentedSet.Add(rentedArray);
                // If Add returns false, it means this specific array instance is somehow already tracked as rented.
                // This indicates a potential logic error or race condition elsewhere, potentially if Return failed to remove it?
                 Debug.Assert(added, $"Array instance (size {size}) was already in the rented set before renting.");
            }

            return rentedArray;
        }

        /// <summary>
        /// Rents multiple arrays of the specified size from the pool.
        /// This method aims to be more efficient than calling Rent multiple times by reducing lock contention.
        /// Arrays rented via this method must be returned individually using the Return method.
        /// </summary>
        /// <param name="size">The required size of each array.</param>
        /// <param name="count">The number of arrays to rent.</param>
        /// <returns>An array containing the rented T[] arrays.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if size or count is negative.</exception>
        public static T[][] RentMultiple(int size, int count)
        {
            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");
            }
            if (size < 0)
            {
                 throw new ArgumentOutOfRangeException(nameof(size), "Size cannot be negative.");
            }

            if (count == 0)
            {
                return Array.Empty<T[]>();
            }
            if (size == 0)
            {
                // Return an array containing 'count' references to the singleton Array.Empty<T>
                T[][] emptyResults = new T[count][]; // Declare the outer array
                // Fill the outer array with instances of the empty inner array
                for (int i = 0; i < count; i++)
                {
                   emptyResults[i] = Array.Empty<T>();
                }
                // No need to track empty arrays in the pool
                return emptyResults;
            }

            T[][] results = new T[count][];
            var stack = s_availableArrays.GetOrAdd(size, _ => new ConcurrentStack<PooledArrayWrapper>());

            for (int i = 0; i < count; i++)
            {
                if (stack.TryPop(out PooledArrayWrapper wrapper))
                {
                     // Check if the array is currently being cleared
                    if (wrapper.State == ArrayState.Clearing)
                    {
                        try
                        {
                           wrapper.ClearingTask?.Wait(); // Synchronously wait
                           Debug.Assert(wrapper.State == ArrayState.Available, "Array state should be Available after waiting for clear.");
                        }
                        catch (Exception ex)
                        {
                           Console.Error.WriteLine($"Error waiting for array clear (size {size}) during RentMultiple: {ex.Message}");
                           // If wait fails, allocate a new array instead of using this potentially broken one.
                           results[i] = new T[size];
                           continue; // Move to the next array
                        }
                    }

                    // Use the array if available
                    if(wrapper.State == ArrayState.Available)
                    {
                         results[i] = wrapper.Array;
                    }
                    else
                    {
                        // Fallback: Should not happen if wait logic is correct, but allocate new just in case.
                        results[i] = new T[size];
                    }
                }
                else
                {
                    // Stack is empty, allocate a new array
                    results[i] = new T[size];
                }
            }

            // Now track all rented arrays under a single lock
            var tracking = s_rentedArraysInfo.GetOrAdd(size, _ => new RentedArrayTracking());
            lock (tracking.SyncLock)
            {
                for (int i = 0; i < count; i++)
                {
                    // results[i] should never be null here
                    bool added = tracking.RentedSet.Add(results[i]);
                    Debug.Assert(added, $"Array instance (size {size}, index {i}) was already in the rented set during RentMultiple.");
                }
            }

            return results;
        }

        /// <summary>
        /// Returns an array to the pool.
        /// </summary>
        /// <param name="array">The array to return. Must have been previously obtained via Rent or RentDisposable.</param>
        /// <param name="clearArray">If true, the array contents will be cleared asynchronously in the background before being made available for rent again.</param>
        /// <exception cref="ArgumentNullException">Thrown if array is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the array was not rented from this pool or is returned multiple times.</exception>
        public static void Return(T[] array, bool clearArray = true)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            if (array.Length == 0)
            {
                // Do nothing for empty arrays, they aren't pooled.
                return;
            }

            int size = array.Length;
            if (s_rentedArraysInfo.TryGetValue(size, out RentedArrayTracking? tracking))
            {
                bool removed = false;
                lock (tracking.SyncLock)
                {
                    removed = tracking.RentedSet.Remove(array);
                }

                if (!removed)
                {
                    // Array wasn't found in the rented set for its size.
                    // Could be a double return, or returning an array not from this pool.
                    throw new InvalidOperationException($"Attempted to return an array (size {size}) that was not rented from this pool or was already returned.");
                }

                 // If tracking exists, the corresponding available stack should also exist or be creatable.
                var stack = s_availableArrays.GetOrAdd(size, _ => new ConcurrentStack<PooledArrayWrapper>());
                PooledArrayWrapper wrapper;

                if (clearArray && typeof(T).IsPrimitive /* Or other conditions where clearing is desired/safe */)
                {
                    // Create wrapper, mark as Clearing
                    wrapper = new PooledArrayWrapper(array, ArrayState.Clearing);
                    stack.Push(wrapper); // Push immediately so it's technically in the pool

                    // Start background task to clear it
                    wrapper.ClearingTask = Task.Run(() =>
                    {
                        try
                        {
                            // Perform the actual clear
                            Array.Clear(array, 0, size);
                        }
                        catch (Exception ex)
                        {
                            // Log error? What state to set on failure?
                            Console.Error.WriteLine($"Background array clear failed (size {size}): {ex.Message}");
                            // Potentially set a 'FailedClear' state if needed. For now, mark as Available.
                        }
                        finally
                        {
                           // CRITICAL: Ensure state is set back to Available regardless of success/failure
                           // so Rent doesn't wait forever or discard it unnecessarily.
                           wrapper.State = ArrayState.Available;
                        }
                    });
                }
                else
                {
                    // No clearing requested or needed, return as immediately available.
                    wrapper = new PooledArrayWrapper(array, ArrayState.Available);
                    stack.Push(wrapper);
                }
            }
            else
            {
                // No tracking info exists for this size, meaning Rent was likely never called for it,
                // or Clear(size) was called after Rent but before Return.
                throw new InvalidOperationException($"Attempted to return an array (size {size}) for which no tracking information exists (potentially cleared pool or not rented).");
            }
        }

        /// <summary>
        /// Rents an array wrapped in a disposable struct for easy cleanup with 'using'.
        /// </summary>
        /// <param name="size">The required size of the array.</param>
        /// <param name="clearOnDispose">If true, the array will be cleared when disposed (returned).</param>
        /// <returns>A DisposableArray<T> wrapping the rented array.</returns>
        public static DisposableArray RentDisposable(int size, bool clearOnDispose = true)
        {
            T[] rentedArray = Rent(size);
            // Pass the actual rented array instance to the disposable wrapper.
            return new DisposableArray(rentedArray, clearOnDispose);
        }

        /// <summary>
        /// Clears all arrays currently held in the pool for the specified size.
        /// Throws an exception if any arrays of this size are currently rented out.
        /// </summary>
        /// <param name="size">The size of arrays to clear from the pool.</param>
        /// <exception cref="InvalidOperationException">Thrown if arrays of the specified size are currently rented.</exception>
        public static void Clear(int size)
        {
            if (size < 0) return; // Or throw? Consistent with Rent? Let's be lenient here.
            if (size == 0) return; // Empty arrays aren't pooled.

            // Check rented status first
            if (s_rentedArraysInfo.TryGetValue(size, out RentedArrayTracking? tracking))
            {
                lock (tracking.SyncLock)
                {
                    if (tracking.RentedSet.Count > 0)
                    {
                        throw new InvalidOperationException($"Cannot clear pool for size {size}, {tracking.RentedSet.Count} array(s) are still rented.");
                    }
                }
                 // If count is 0, we can proceed to remove tracking and available arrays.
                 // We do this outside the lock on tracking.SyncLock to reduce lock duration,
                 // relying on the atomicity of ConcurrentDictionary removals.
                 s_rentedArraysInfo.TryRemove(size, out _);
            }

            // Remove the stack of available arrays for this size.
            // Existing wrappers/arrays in the removed stack will become eligible for GC
            // once any pending clearing tasks complete.
            s_availableArrays.TryRemove(size, out _);
        }


        /// <summary>
        /// Clears all arrays of all sizes currently held in the pool.
        /// Throws an exception if any arrays of any size are currently rented out.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if any arrays are currently rented.</exception>
        public static void Clear()
        {
            List<int> rentedSizes = null;

            // Use the dedicated global lock for this potentially sweeping operation
            lock (s_clearAllLock)
            {
                // Check all sizes for rented arrays *while holding the global lock*
                foreach (var kvp in s_rentedArraysInfo)
                {
                    int size = kvp.Key;
                    RentedArrayTracking tracking = kvp.Value;
                    // Lock individual tracking object to safely read count
                    lock (tracking.SyncLock)
                    {
                        if (tracking.RentedSet.Count > 0)
                        {
                           rentedSizes ??= new List<int>();
                           rentedSizes.Add(size);
                        }
                    }
                }

                // If any rented arrays were found, abort the clear operation
                if (rentedSizes != null)
                {
                    throw new InvalidOperationException($"Cannot clear pool, arrays are still rented for sizes: {string.Join(", ", rentedSizes)}.");
                }

                // If we get here, no arrays are rented. Clear everything.
                s_rentedArraysInfo.Clear();
                s_availableArrays.Clear(); // Clears all stacks
            }
            // Arrays held within the cleared stacks become eligible for GC.
        }


        // ---------------------------------------------------------------------
        // Disposable Wrapper Struct
        // ---------------------------------------------------------------------

        /// <summary>
        /// A ref struct wrapper around a rented array that ensures it is returned
        /// to the pool when the struct is disposed (typically via a 'using' statement).
        /// Provides an implicit conversion to the underlying T[].
        /// </summary>
        public ref struct DisposableArray // Must be public to be used with RentDisposable. NOT readonly.
        {
            private readonly T[] _array; // Store the specific array instance
            private readonly bool _clearOnDispose;
            private bool _disposed; // Mutable field to track disposal state

            public T[] A => _array;

            /// <summary>Initializes a new instance of the <see cref="DisposableArray"/> struct.</summary>
            /// <remarks>Internal constructor to be called only by the pool.</remarks>
            internal DisposableArray(T[] array, bool clearOnDispose)
            {
                _array = array ?? throw new ArgumentNullException(nameof(array)); // Should not happen if Rent is correct
                _clearOnDispose = clearOnDispose;
                _disposed = false; // Initialize disposed state
            }

            /// <summary>
            /// Gets the underlying array. Use this or the implicit conversion.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public T[] Array
            {
                get
                {
                    if (_disposed) throw new ObjectDisposedException(nameof(DisposableArray));
                    // _array should not be null if not disposed, but null-forgiving operator used for safety.
                    return _array!;
                }
            }

            /// <summary>
            /// Allows implicit conversion of the wrapper directly to a T[].
            /// </summary>
            /// <param name="wrapper">The DisposableArray instance.</param>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public static implicit operator T[](in DisposableArray wrapper)
            {
                 if (wrapper._disposed) throw new ObjectDisposedException(nameof(DisposableArray));
                 // _array should not be null if not disposed.
                 return wrapper._array!;
            }

             /// <summary>
            /// Allows implicit conversion of the wrapper directly to a ReadOnlySpan<T>.
            /// </summary>
            /// <param name="wrapper">The DisposableArray instance.</param>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
            public static implicit operator ReadOnlySpan<T>(in DisposableArray wrapper)
            {
                 if (wrapper._disposed) throw new ObjectDisposedException(nameof(DisposableArray));
                 // _array should not be null if not disposed.
                 return wrapper._array!; // Implicit conversion from T[] to ReadOnlySpan<T>
            }

            /// <summary>
            /// Allows implicit conversion of the wrapper directly to a Span<T>.
            /// </summary>
            /// <param name="wrapper">The DisposableArray instance.</param>
            /// <exception cref="ObjectDisposedException">Thrown if the wrapper has been disposed.</exception>
             public static implicit operator Span<T>(in DisposableArray wrapper)
            {
                 if (wrapper._disposed) throw new ObjectDisposedException(nameof(DisposableArray));
                 // _array should not be null if not disposed.
                 return wrapper._array!; // Implicit conversion from T[] to Span<T>
            }

            /// <summary>
            /// Disposes the wrapper, returning the underlying array to the ArrayPool<T>.
            /// </summary>
            /// <exception cref="ObjectDisposedException">Thrown if Dispose is called more than once.</exception>
            public void Dispose()
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(DisposableArray), "Cannot dispose more than once.");
                }

                if (_array != null)
                {
                    // Return the array to the pool
                    ArrayPool<T>.Return(_array, _clearOnDispose);
                }

                // Mark as disposed *after* returning successfully.
                _disposed = true;
            }
        }
    }


     /// <summary>
    /// Helper comparer for using reference equality in the HashSet.
    /// </summary>
    internal sealed class ReferenceEqualityComparer : EqualityComparer<object>
    {
        public static readonly ReferenceEqualityComparer Instance = new ReferenceEqualityComparer();

        private ReferenceEqualityComparer() { }

        public override bool Equals(object? x, object? y)
        {
            return ReferenceEquals(x, y);
        }

        public override int GetHashCode(object? obj)
        {
            // Use System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj) for reference hash code.
            return obj == null ? 0 : System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
        }
    }
}