using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Warp.Tools
{
    public class Spandex<T> : IDisposable where T : unmanaged
    {
        private readonly uint Alignment = 32;
        private int MaxLength;
        private Queue<IntPtr> Buffers = new Queue<IntPtr>();
        private List<RentedSpan> Rented = new List<RentedSpan>();
        private bool Disposed = false;

        private readonly object Sync = new object();

        public Spandex()
        {
        }

        public Span<T> Rent(int length)
        {
            lock (Sync)
            {
                unsafe
                {
                    if (length > MaxLength)
                    {
                        MaxLength = length;

                        // Dispose all existing buffers since they are smaller than the new maximum length
                        while (Buffers.Count > 0)
                            NativeMemory.AlignedFree((void*)Buffers.Dequeue());
                    }

                    // Try to reuse a buffer if possible
                    if (Buffers.Count > 0)
                    {
                        var existingBuffer = Buffers.Dequeue();
                        Rented.Add(new RentedSpan(existingBuffer, MaxLength));
                        return new Span<T>(existingBuffer.ToPointer(), length);
                    }

                    // Otherwise, allocate a new buffer
                    var newBuffer = NativeMemory.AlignedAlloc(new UIntPtr((ulong)MaxLength * (ulong)sizeof(T)), Alignment);
                    if (newBuffer == (void*)0)
                        throw new OutOfMemoryException();

                    Rented.Add(new RentedSpan(new IntPtr(newBuffer), MaxLength));
                    return new Span<T>(newBuffer, length);
                }
            }
        }

        public void Return(Span<T> buffer)
        {
            lock (Sync)
            {
                unsafe
                {
                    IntPtr BufferPtr = new IntPtr(Unsafe.AsPointer(ref buffer.GetPinnableReference()));
                    var RentedInfo = Rented.FirstOrDefault(r => r.Memory == BufferPtr, null);
                    if (RentedInfo == null)
                        throw new Exception("Tried to return a buffer that was not rented.");

                    Rented.Remove(RentedInfo);

                    // Only return the buffer to the pool if it matches the current max length
                    if (RentedInfo.Elements == MaxLength)
                    {
                        Buffers.Enqueue(BufferPtr);
                    }
                    else
                    {
                        // If it's smaller, dispose it directly
                        NativeMemory.AlignedFree((void*)BufferPtr);
                    }

                    //RentedInfo.Empty();
                }
            }
        }

        public bool HasUnreturned() => Rented.Any();

        public void Dispose()
        {
            lock (Sync)
            {
                if (Disposed) return;

                unsafe
                {
                    while (Buffers.Count > 0)
                        NativeMemory.AlignedFree((void*)Buffers.Dequeue());
                }

                Disposed = true;
            }
        }

        ~Spandex()
        {
            Dispose();
        }

        class RentedSpan //: IDisposable
        {
            public IntPtr Memory;
            public int Elements;
            private bool Disposed = false;

            public RentedSpan(IntPtr memory, int elements)
            {
                Memory = memory;
                Elements = elements;
            }

            //public void Empty()
            //{
            //    Memory = IntPtr.Zero;
            //    Elements = 0;
            //}

            //public void Dispose()
            //{
            //    if (!Disposed)
            //    {
            //        if (Memory != IntPtr.Zero)
            //            unsafe
            //            {
            //                NativeMemory.AlignedFree((void*)Memory);
            //            }

            //        Memory = IntPtr.Zero;
            //        Disposed = true;
            //    }
            //}

            //~RentedSpan()
            //{
            //    Dispose();
            //}
        }
    }
}