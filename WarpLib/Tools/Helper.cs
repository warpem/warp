using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Threading;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.Headers;

namespace Warp.Tools
{
    public static class Helper
    {
        public static IFormatProvider NativeFormat = CultureInfo.InvariantCulture.NumberFormat;
        public static IFormatProvider NativeDateTimeFormat = CultureInfo.InvariantCulture.DateTimeFormat;

        public static float PI = (float)Math.PI;
        public static float ToRad = (float)Math.PI / 180.0f;
        public static float ToDeg = 180.0f / (float)Math.PI;

        public static bool IsDebug => Environment.GetEnvironmentVariable("WARP_DEBUG") != null;

        public static void Swap<T>(ref T lhs, ref T rhs)
        {
            T temp;
            temp = lhs;
            lhs = rhs;
            rhs = temp;
        }

        public static float BSpline(float t)
        {
            t = Math.Abs(t);
            float a = 2.0f - t;

            if (t < 1.0f)
                return 2.0f / 3.0f - 0.5f * t * t * a;
            if (t < 2.0f)
                return a * a * a / 6.0f;

            return 0.0f;
        }

        public static float Sinc(float x)
        {
            if (Math.Abs(x) > 1e-8f)
                return (float) (Math.Sin(Math.PI * x) / (Math.PI * x));

            return 1f;
        }

        public static int3[] GetEqualGridSpacing(int2 dimsImage, int2 dimsRegion, float overlapFraction, out int2 dimsGrid)
        {
            int2 dimsoverlap = new int2((int)(dimsRegion.X * (1.0f - overlapFraction)), (int)(dimsRegion.Y * (1.0f - overlapFraction)));
            dimsGrid = new int2(MathHelper.NextMultipleOf(dimsImage.X - (dimsRegion.X - dimsoverlap.X), dimsoverlap.X) / dimsoverlap.X, 
                                MathHelper.NextMultipleOf(dimsImage.Y - (dimsRegion.Y - dimsoverlap.Y), dimsoverlap.Y) / dimsoverlap.Y);

            int2 shift;
            shift.X = dimsGrid.X > 1 ? (int)((dimsImage.X - dimsRegion.X) / (float)(dimsGrid.X - 1)) : (dimsImage.X - dimsRegion.X) / 2;
            shift.Y = dimsGrid.Y > 1 ? (int)((dimsImage.Y - dimsRegion.Y) / (float)(dimsGrid.Y - 1)) : (dimsImage.Y - dimsRegion.Y) / 2;
            int2 offset = new int2((dimsImage.X - shift.X * (dimsGrid.X - 1) - dimsRegion.X) / 2,
                                   (dimsImage.Y - shift.Y * (dimsGrid.Y - 1) - dimsRegion.Y) / 2);

            int3[] h_origins = new int3[dimsGrid.Elements()];

            for (int y = 0; y < dimsGrid.Y; y++)
                for (int x = 0; x < dimsGrid.X; x++)
                    h_origins[y * dimsGrid.X + x] = new int3(x * shift.X + offset.X, y * shift.Y + offset.Y, 0);

            return h_origins;
        }

        public static float[] ToInterleaved(float2[] array)
        {
            float[] Interleaved = new float[array.Length * 2];
            for (int i = 0; i < array.Length; i++)
            {
                Interleaved[i * 2] = array[i].X;
                Interleaved[i * 2 + 1] = array[i].Y;
            }

            return Interleaved;
        }

        public static float[] ToInterleaved(float3[] array)
        {
            float[] Interleaved = new float[array.Length * 3];
            for (int i = 0; i < array.Length; i++)
            {
                Interleaved[i * 3] = array[i].X;
                Interleaved[i * 3 + 1] = array[i].Y;
                Interleaved[i * 3 + 2] = array[i].Z;
            }

            return Interleaved;
        }

        public static float[] ToInterleaved(float4[] array)
        {
            float[] Interleaved = new float[array.Length * 4];
            for (int i = 0; i < array.Length; i++)
            {
                Interleaved[i * 4] = array[i].X;
                Interleaved[i * 4 + 1] = array[i].Y;
                Interleaved[i * 4 + 2] = array[i].Z;
                Interleaved[i * 4 + 3] = array[i].W;
            }

            return Interleaved;
        }

        public static int[] ToInterleaved(int2[] array)
        {
            int[] Interleaved = new int[array.Length * 2];
            for (int i = 0; i < array.Length; i++)
            {
                Interleaved[i * 2] = array[i].X;
                Interleaved[i * 2 + 1] = array[i].Y;
            }

            return Interleaved;
        }

        public static int[] ToInterleaved(int3[] array)
        {
            int[] Interleaved = new int[array.Length * 3];
            for (int i = 0; i < array.Length; i++)
            {
                Interleaved[i * 3] = array[i].X;
                Interleaved[i * 3 + 1] = array[i].Y;
                Interleaved[i * 3 + 2] = array[i].Z;
            }

            return Interleaved;
        }

        public static float2[] FromInterleaved2(float[] array)
        {
            float2[] Tuples = new float2[array.Length / 2];
            for (int i = 0; i < Tuples.Length; i++)
                Tuples[i] = new float2(array[i * 2], array[i * 2 + 1]);

            return Tuples;
        }

        public static float3[] FromInterleaved3(float[] array)
        {
            float3[] Tuples = new float3[array.Length / 3];
            for (int i = 0; i < Tuples.Length; i++)
                Tuples[i] = new float3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);

            return Tuples;
        }

        public static int2[] FromInterleaved2(int[] array)
        {
            int2[] Tuples = new int2[array.Length / 2];
            for (int i = 0; i < Tuples.Length; i++)
                Tuples[i] = new int2(array[i * 2], array[i * 2 + 1]);

            return Tuples;
        }

        public static int3[] FromInterleaved3(int[] array)
        {
            int3[] Tuples = new int3[array.Length / 3];
            for (int i = 0; i < Tuples.Length; i++)
                Tuples[i] = new int3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);

            return Tuples;
        }

        public static T[,] Reshape<T>(T[][] m)
        {
            T[,] Result = new T[m.Length, m[0].Length];

            for (int y = 0; y < m.Length; y++)
                for (int x = 0; x < m[y].Length; x++)
                    Result[y, x] = m[y][x];

            return Result;
        }

        public static T[][] Reshape<T>(T[,] m)
        {
            T[][] Result = new T[m.GetLength(0)][];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new T[m.GetLength(1)];

            for (int y = 0; y < Result.Length; y++)
                for (int x = 0; x < Result[y].Length; x++)
                    Result[y][x] = m[y, x];

            return Result;
        }

        public static T[,] ReshapeTransposed<T>(T[][] m)
        {
            T[,] Result = new T[m[0].Length, m.Length];

            for (int y = 0; y < m.Length; y++)
                for (int x = 0; x < m[y].Length; x++)
                    Result[y, x] = m[x][y];

            return Result;
        }

        public static T[][] ReshapeTransposed<T>(T[,] m)
        {
            T[][] Result = new T[m.GetLength(1)][];
            for (int i = 0; i < Result.Length; i++)
                Result[i] = new T[m.GetLength(0)];

            for (int y = 0; y < Result.Length; y++)
                for (int x = 0; x < Result[y].Length; x++)
                    Result[y][x] = m[x, y];

            return Result;
        }

        public static void Unzip(float2[] array, out float[] out1, out float[] out2)
        {
            out1 = new float[array.Length];
            out2 = new float[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                out1[i] = array[i].X;
                out2[i] = array[i].Y;
            }
        }

        public static void Unzip(float3[] array, out float[] out1, out float[] out2, out float[] out3)
        {
            out1 = new float[array.Length];
            out2 = new float[array.Length];
            out3 = new float[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                out1[i] = array[i].X;
                out2[i] = array[i].Y;
                out3[i] = array[i].Z;
            }
        }

        public static float2[] Zip(float[] in1, float[] in2)
        {
            float2[] Zipped = new float2[in1.Length];
            for (int i = 0; i < Zipped.Length; i++)
                Zipped[i] = new float2(in1[i], in2[i]);

            return Zipped;
        }

        public static float3[] Zip(float[] in1, float[] in2, float[] in3)
        {
            float3[] Zipped = new float3[in1.Length];
            for (int i = 0; i < Zipped.Length; i++)
                Zipped[i] = new float3(in1[i], in2[i], in3[i]);

            return Zipped;
        }

        public static float4[] Zip(float[] in1, float[] in2, float[] in3, float[] in4)
        {
            float4[] Zipped = new float4[in1.Length];
            for (int i = 0; i < Zipped.Length; i++)
                Zipped[i] = new float4(in1[i], in2[i], in3[i], in4[i]);

            return Zipped;
        }

        public static float5[] Zip(float[] in1, float[] in2, float[] in3, float[] in4, float[] in5)
        {
            float5[] Zipped = new float5[in1.Length];
            for (int i = 0; i < Zipped.Length; i++)
                Zipped[i] = new float5(in1[i], in2[i], in3[i], in4[i], in5[i]);

            return Zipped;
        }

        public static void Reorder<T>(IList<T> list, int[] indices)
        {
            List<T> OldOrder = new List<T>(list.Count);
            for (int i = 0; i < list.Count; i++)
                OldOrder.Add(list[i]);

            for (int i = 0; i < list.Count; i++)
                list[i] = OldOrder[indices[i]];
        }

        public static void Reorder<T>(T[] array, int[] indices)
        {
            List<T> OldOrder = new List<T>(array.Length);
            for (int i = 0; i < OldOrder.Count; i++)
                OldOrder[i] = array[i];

            for (int i = 0; i < array.Length; i++)
                array[i] = OldOrder[indices[i]];
        }

        public static T[] Combine<T>(IEnumerable<T[]> arrays)
        {
            int NElements = arrays.Select(a => a.Length).Sum();
            T[] Result = new T[NElements];

            int i = 0;
            foreach (var array in arrays)
                for (int j = 0; j < array.Length; j++, i++)
                    Result[i] = array[j];

            return Result;
        }

        public static T[] Combine<T>(params T[][] arrays)
        {
            int NElements = arrays.Select(a => a.Length).Sum();
            T[] Result = new T[NElements];

            int i = 0;
            foreach (var array in arrays)
                for (int j = 0; j < array.Length; j++, i++)
                    Result[i] = array[j];

            return Result;
        }

        public static void ForEachElement(int2 dims, Action<int, int> action)
        {
            for (int y = 0; y < dims.Y; y++)
                for (int x = 0; x < dims.X; x++)
                    action(x, y);
        }

        public static void ForEachElement(int2 dims, Action<int, int, int, int> action)
        {
            for (int y = 0; y < dims.Y; y++)
            {
                int yy = y - dims.Y / 2;

                for (int x = 0; x < dims.X; x++)
                {
                    int xx = x - dims.X / 2;

                    action(x, y, xx, yy);
                }
            }
        }

        public static void ForEachElement(int2 dims, Action<int, int, int, int, float, float> action)
        {
            for (int y = 0; y < dims.Y; y++)
            {
                int yy = y - dims.Y / 2;

                for (int x = 0; x < dims.X; x++)
                {
                    int xx = x - dims.X / 2;

                    action(x, y, xx, yy, (float)Math.Sqrt(xx * xx + yy * yy), (float)Math.Atan2(yy, xx));
                }
            }
        }

        public static void ForEachElementFT(int2 dims, Action<int, int> action)
        {
            for (int y = 0; y < dims.Y; y++)
                for (int x = 0; x < dims.X / 2 + 1; x++)
                    action(x, y);
        }

        public static void ForEachElementFT(int2 dims, Action<int, int, int, int> action)
        {
            for (int y = 0; y < dims.Y; y++)
            {
                int yy = y < dims.Y / 2 + 1 ? y : y - dims.Y;

                for (int x = 0; x < dims.X / 2 + 1; x++)
                {
                    int xx = x;

                    action(x, y, xx, yy);
                }
            }
        }

        public static void ForEachElementFTParallel(int2 dims, Action<int, int, int, int> action)
        {
            Parallel.For(0, dims.Y, y =>
            {
                int yy = y < dims.Y / 2 + 1 ? y : y - dims.Y;

                for (int x = 0; x < dims.X / 2 + 1; x++)
                {
                    int xx = x;

                    action(x, y, xx, yy);
                }
            });
        }

        public static void ForEachElementFT(int2 dims, Action<int, int, int, int, float, float> action)
        {
            for (int y = 0; y < dims.Y; y++)
            {
                int yy = y < dims.Y / 2 + 1 ? y : y - dims.Y;

                for (int x = 0; x < dims.X / 2 + 1; x++)
                {
                    int xx = x;

                    action(x, y, xx, yy, (float)Math.Sqrt(xx * xx + yy * yy), (float)Math.Atan2(yy, xx));
                }
            }
        }

        public static void ForEachElementFT(int3 dims, Action<int, int, int, int, int, int> action)
        {
            for (int z = 0; z < dims.Z; z++)
            {
                int zz = z < dims.Z / 2 + 1 ? z : z - dims.Z;

                for (int y = 0; y < dims.Y; y++)
                {
                    int yy = y < dims.Y / 2 + 1 ? y : y - dims.Y;

                    for (int x = 0; x < dims.X / 2 + 1; x++)
                    {
                        int xx = x;

                        action(x, y, z, xx, yy, zz);
                    }
                }
            }
        }

        public static void ForEachElementFT(int3 dims, Action<int, int, int, int, int, int, float> action)
        {
            for (int z = 0; z < dims.Z; z++)
            {
                int zz = z < dims.Z / 2 + 1 ? z : z - dims.Z;

                for (int y = 0; y < dims.Y; y++)
                {
                    int yy = y < dims.Y / 2 + 1 ? y : y - dims.Y;

                    for (int x = 0; x < dims.X / 2 + 1; x++)
                    {
                        int xx = x;

                        action(x, y, z, xx, yy, zz, (float)Math.Sqrt(xx * xx + yy * yy + zz * zz));
                    }
                }
            }
        }

        public static float[] Extract(float[] volume, int3 dimsvolume, int3 centerextract, int3 dimsextract)
        {
            int3 Origin = new int3(centerextract.X - dimsextract.X / 2,
                                   centerextract.Y - dimsextract.Y / 2,
                                   centerextract.Z - dimsextract.Z / 2);

            float[] Extracted = new float[dimsextract.Elements()];

            unsafe
            {
                fixed (float* volumePtr = volume)
                fixed (float* ExtractedPtr = Extracted)
                for (int z = 0; z < dimsextract.Z; z++)
                    for (int y = 0; y < dimsextract.Y; y++)
                        for (int x = 0; x < dimsextract.X; x++)
                        {
                            int3 Pos = new int3((Origin.X + x + dimsvolume.X) % dimsvolume.X,
                                                (Origin.Y + y + dimsvolume.Y) % dimsvolume.Y,
                                                (Origin.Z + z + dimsvolume.Z) % dimsvolume.Z);

                            float Val = volumePtr[(Pos.Z * dimsvolume.Y + Pos.Y) * dimsvolume.X + Pos.X];
                            ExtractedPtr[(z * dimsextract.Y + y) * dimsextract.X + x] = Val;
                        }
            }

            return Extracted;
        }

        public static float3[] GetHealpixAngles(int order, string symmetry = "C1", float limittilt = -91)
        {
            int N = CPU.GetAnglesCount(order, symmetry, limittilt);

            float[] Continuous = new float[N * 3];
            CPU.GetAngles(Continuous, order, symmetry, limittilt);

            return Helper.FromInterleaved3(Continuous);
        }

        public static float2[] GetHealpixRotTilt(int order, string symmetry = "C1", float limittilt = -91)
        {
            float3[] Angles = GetHealpixAngles(order, symmetry, limittilt);
            List<float2> Result = new List<float2>() { new float2(Angles[0]) };
            float2 LastRotTilt = new float2(Angles[0]);

            for (int i = 1; i < Angles.Length; i++)
            {
                float2 RotTilt = new float2(Angles[i]);
                if (RotTilt == LastRotTilt || (limittilt > 0 && RotTilt.Y > limittilt))
                    continue;

                LastRotTilt = RotTilt;
                Result.Add(RotTilt);
            }

            return Result.ToArray();
        }

        public static void ForEachGPU<T>(IEnumerable<T> items, GPUTaskIterator<T> iterator, int perDevice = 1, List<int> deviceList = null)
        {
            int NDevices = GPU.GetDeviceCount();

            if (deviceList == null)
                deviceList = Helper.ArrayOfSequence(0, Math.Min(NDevices, items.Count()), 1).ToList();

            Queue<DeviceToken> Devices = new Queue<DeviceToken>();
            for (int i = 0; i < perDevice; i++)
                for (int d = deviceList.Count - 1; d >= 0; d--)
                    Devices.Enqueue(new DeviceToken(deviceList[d]));

            int NTokens = Devices.Count;

            foreach (var item in items)
            {
                while (Devices.Count <= 0)
                    Thread.Sleep(5);

                DeviceToken CurrentDevice;
                lock (Devices)
                    CurrentDevice = Devices.Dequeue();

                Thread DeviceThread = new Thread(() =>
                {
                    GPU.SetDevice(CurrentDevice.ID % NDevices);

                    iterator(item, CurrentDevice.ID);

                    lock (Devices)
                        Devices.Enqueue(CurrentDevice);
                }) { Name = $"ForEachGPU Device {CurrentDevice.ID}" };

                DeviceThread.Start();
            }
            
            while (Devices.Count != NTokens)
                Thread.Sleep(5);
        }

        public static void ForEachGPU<T>(IEnumerable<T> items, GPUTaskCancelableIterator<T> iterator, int perDevice = 1, List<int> deviceList = null)
        {
            int NDevices = GPU.GetDeviceCount();

            if (deviceList == null)
                deviceList = Helper.ArrayOfSequence(0, Math.Min(NDevices, items.Count()), 1).ToList();

            Queue<DeviceToken> Devices = new Queue<DeviceToken>();
            for (int i = 0; i < perDevice; i++)
                for (int d = deviceList.Count - 1; d >= 0; d--)
                    Devices.Enqueue(new DeviceToken(deviceList[d]));

            int NTokens = Devices.Count;
            bool IsCanceled = false;

            foreach (var item in items)
            {
                if (IsCanceled)
                    break;

                while (Devices.Count <= 0)
                    Thread.Sleep(5);

                DeviceToken CurrentDevice;
                lock (Devices)
                    CurrentDevice = Devices.Dequeue();

                Thread DeviceThread = new Thread(() =>
                {
                    GPU.SetDevice(CurrentDevice.ID % NDevices);

                    IsCanceled = iterator(item, CurrentDevice.ID);

                    lock (Devices)
                        Devices.Enqueue(CurrentDevice);
                })
                { Name = $"ForEachGPU Device {CurrentDevice.ID}" };

                DeviceThread.Start();
            }

            while (Devices.Count != NTokens)
                Thread.Sleep(5);
        }

        public static void ForEachGPUOnce(Action<int> iterator, List<int> deviceList = null)
        {
            int NDevices = GPU.GetDeviceCount();

            if (deviceList == null)
                deviceList = Helper.ArrayOfSequence(0, NDevices, 1).ToList();

            Queue<DeviceToken> Devices = new Queue<DeviceToken>();
            for (int d = deviceList.Count - 1; d >= 0; d--)
                Devices.Enqueue(new DeviceToken(deviceList[d]));

            int NTokens = Devices.Count;

            for (int i = 0; i < deviceList.Count; i++)
            {
                DeviceToken CurrentDevice;
                lock (Devices)
                    CurrentDevice = Devices.Dequeue();

                Thread DeviceThread = new Thread(() =>
                {
                    GPU.SetDevice(CurrentDevice.ID % NDevices);

                    iterator(CurrentDevice.ID);

                    lock (Devices)
                        Devices.Enqueue(CurrentDevice);
                })
                { Name = $"ForEachGPU Device {CurrentDevice.ID}" };

                DeviceThread.Start();
            }

            while (Devices.Count != NTokens)
                Thread.Sleep(5);
        }

        public static void ForGPU(int fromInclusive, int toExclusive, GPUTaskIterator<int> iterator, int perDevice = 1, List<int> deviceList = null)
        {
            int[] Items = new int[toExclusive - fromInclusive];
            for (int i = 0; i < Items.Length; i++)
                Items[i] = i + fromInclusive;

            ForEachGPU(Items, iterator, perDevice, deviceList);
        }

        public static object[] ForCPU(int fromInclusive, int toExclusive, int nThreads, Func<int, object> funcSetup, Action<int, int, object> funcIterator, Func<int, object, object> funcTeardown)
        {
            object[] Results = new object[nThreads];
            object[] ThreadStuff = new object[nThreads];

            if (nThreads > 1)
            {
                Thread[] Pool = new Thread[nThreads];
                Barrier BarrierSetup = new Barrier(nThreads);
                Barrier BarrierIterator = new Barrier(nThreads);
                Barrier BarrierTeardown = new Barrier(nThreads + 1);

                for (int n = 0; n < nThreads; n++)
                {
                    Pool[n] = new Thread((id) =>
                    {
                        int ID = (int)id;
                        if (funcSetup != null)
                            ThreadStuff[ID] = funcSetup(ID);
                        BarrierSetup.SignalAndWait();

                        if (funcIterator != null)
                            for (int i = fromInclusive + ID; i < toExclusive; i += nThreads)
                                funcIterator(i, ID, ThreadStuff[ID]);
                        BarrierIterator.SignalAndWait();

                        if (funcTeardown != null)
                            Results[ID] = funcTeardown(ID, ThreadStuff[ID]);
                        BarrierTeardown.SignalAndWait();
                    })
                    { Name = "ForCPU" + n };
                    Pool[n].Start(n);
                }

                BarrierTeardown.SignalAndWait();
            }
            else
            {
                if (funcSetup != null)
                    ThreadStuff[0] = funcSetup(0);

                if (funcIterator != null)
                    for (int i = fromInclusive; i < toExclusive; i++)
                        funcIterator(i, 0, ThreadStuff[0]);

                if (funcTeardown != null)
                    Results[0] = funcTeardown(0, ThreadStuff[0]);
            }

            return Results;
        }

        public static void ForCPU(int fromInclusive, int toExclusive, int nThreads, Action<int> funcSetup, Action<int, int> funcIterator, Action<int> funcTeardown)
        {
            if (nThreads > 1)
            {
                Thread[] Pool = new Thread[nThreads];
                Barrier BarrierSetup = new Barrier(nThreads);
                Barrier BarrierIterator = new Barrier(nThreads);
                Barrier BarrierTeardown = new Barrier(nThreads + 1);

                for (int n = 0; n < nThreads; n++)
                {
                    Pool[n] = new Thread(id =>
                    {
                        int ID = (int)id;
                        funcSetup?.Invoke(ID);
                        BarrierSetup.SignalAndWait();

                        for (int i = fromInclusive + ID; i < toExclusive; i += nThreads)
                            funcIterator?.Invoke(i, ID);
                        BarrierIterator.SignalAndWait();

                        funcTeardown?.Invoke(ID);
                        BarrierTeardown.SignalAndWait();
                    })
                    { Name = "ForCPU" + n };
                    Pool[n].Start(n);
                }

                BarrierTeardown.SignalAndWait();
            }
            else
            {
                funcSetup?.Invoke(0);

                for (int i = fromInclusive; i < toExclusive; i++)
                    funcIterator?.Invoke(i, 0);

                funcTeardown?.Invoke(0);
            }
        }

        public static void ForCPUGreedy(int fromInclusive, int toExclusive, int nThreads, Action<int> funcSetup, Action<int, int> funcIterator, Action<int> funcTeardown)
        {
            if (nThreads > 1)
            {
                Thread[] Pool = new Thread[nThreads];
                Barrier BarrierSetup = new Barrier(nThreads);
                Barrier BarrierIterator = new Barrier(nThreads);
                Barrier BarrierTeardown = new Barrier(nThreads + 1);

                Queue<int> Items = new Queue<int>(ArrayOfSequence(fromInclusive, toExclusive, 1));

                for (int n = 0; n < nThreads; n++)
                {
                    Pool[n] = new Thread(id =>
                    {
                        int ID = (int)id;
                        funcSetup?.Invoke(ID);
                        BarrierSetup.SignalAndWait();

                        while (true)
                        {
                            int Item = 0;
                            lock (Items)
                                if (Items.Any())
                                    Item = Items.Dequeue();
                                else
                                    break;

                            funcIterator?.Invoke(Item, ID);
                        }
                        BarrierIterator.SignalAndWait();

                        funcTeardown?.Invoke(ID);
                        BarrierTeardown.SignalAndWait();
                    })
                    { Name = "ForCPU" + n };
                    Pool[n].Start(n);
                }

                BarrierTeardown.SignalAndWait();
            }
            else
            {
                funcSetup?.Invoke(0);

                for (int i = fromInclusive; i < toExclusive; i++)
                    funcIterator?.Invoke(i, 0);

                funcTeardown?.Invoke(0);
            }
        }

        public static T[] RandomSubset<T>(IEnumerable<T> values, int n, int seed = 123)
        {
            Random Rand = new Random(seed);
            List<T> Values = new List<T>(values);
            List<T> Subset = new List<T>(n);

            while (Subset.Count < n && Values.Count > 0)
            {
                int ID = Rand.Next(Values.Count);
                Subset.Add(Values[ID]);
                Values.RemoveAt(ID);
            }

            return Subset.ToArray();
        }

        public static T[] IndexedSubset<T>(T[] values, int[] indices)
        {
            T[] Result = new T[indices.Length];
            for (int i = 0; i < indices.Length; i++)
                Result[i] = values[indices[i]];

            return Result;
        }

        public static T[] Subset<T>(T[] values, int fromInclusive, int toExclusive)
        {
            T[] Result = new T[toExclusive - fromInclusive];

            for (int i = fromInclusive, j = 0; i < toExclusive; i++, j++)
                Result[j] = values[i];

            return Result;
        }

        public static T[] Subset<T>(List<T> values, int fromInclusive, int toExclusive)
        {
            T[] Result = new T[toExclusive - fromInclusive];

            for (int i = fromInclusive, j = 0; i < toExclusive; i++, j++)
                Result[j] = values[i];

            return Result;
        }

        public static int[] IndicesOf<T>(T[] items, Func<T, bool> predicate)
        {
            List<int> Indices = new List<int>(items.Length);
            for (int i = 0; i < items.Length; i++)
                if (predicate(items[i]))
                    Indices.Add(i);

            return Indices.ToArray();
        }

        public static T[] ArrayOfConstant<T>(T constant, int n)
        {
            T[] Copies = new T[n];
            for (int i = 0; i < n; i++)
                Copies[i] = constant;

            return Copies;
        }

        public static T[] ArrayOfFunction<T>(Func<T> generator, int n)
        {
            T[] Copies = new T[n];
            for (int i = 0; i < n; i++)
                Copies[i] = generator();

            return Copies;
        }

        public static T[] ArrayOfFunction<T>(Func<int, T> generator, int n)
        {
            T[] Copies = new T[n];
            for (int i = 0; i < n; i++)
                Copies[i] = generator(i);

            return Copies;
        }

        public static int[] ArrayOfSequence(int fromInclusive, int toExclusive, int step)
        {
            int N = (int)Math.Ceiling((toExclusive - fromInclusive) / (float)step);
            int[] Result = new int[N];

            for (int i = 0; i < N; i++)
                Result[i] = fromInclusive + i * step;

            return Result;
        }

        public static float[] ArrayOfSequence(float fromInclusive, float toExclusive, float step)
        {
            int N = (int)Math.Round((toExclusive - fromInclusive) / step);
            float[] Result = new float[N];

            for (int i = 0; i < N; i++)
                Result[i] = fromInclusive + i * step;

            return Result;
        }

        public static double[] ArrayOfSequence(double fromInclusive, double toExclusive, double step)
        {
            int N = (int)((toExclusive - fromInclusive) / step);
            double[] Result = new double[N];

            for (int i = 0; i < N; i++)
                Result[i] = fromInclusive + i * step;

            return Result;
        }

        public static void Fill<T>(T[] array, T value)
        {
            for (int i = 0; i < array.Length; i++)
                array[i] = value;
        }

        public static T[] CopyOf<T>(T[] original, int nElements, T defaultValue)
        {
            T[] Copy = new T[nElements];
            Array.Copy(original, 0, Copy, 0, Math.Min(nElements, original.Length));

            if (nElements > original.Length)
                for (int i = original.Length; i < nElements; i++)
                    Copy[i] = defaultValue;

            return Copy;
        }

        public static T[] AsSorted<T>(T[] values)
        {
            List<T> ToSort = new List<T>(values);
            ToSort.Sort();

            return ToSort.ToArray();
        }

        public static int[] AsSortedIndices<T>(T[] values, Comparison<T> comparison)
        {
            List<(T, int)> ToSort = new List<(T, int)>(Helper.ArrayOfFunction(i => (values[i], i), values.Length));

            ToSort.Sort((a, b) => comparison(a.Item1, b.Item1));

            return ToSort.Select(t => t.Item2).ToArray();
        }

        public static float[] AsSortedDescending(float[] values)
        {
            List<float> ToSort = new List<float>(values);
            ToSort.Sort((a, b) => -a.CompareTo(b));

            return ToSort.ToArray();
        }

        public static double[] AsSortedDescending(double[] values)
        {
            List<double> ToSort = new List<double>(values);
            ToSort.Sort((a, b) => -a.CompareTo(b));

            return ToSort.ToArray();
        }

        public static int[] AsSortedDescending(int[] values)
        {
            List<int> ToSort = new List<int>(values);
            ToSort.Sort((a, b) => -a.CompareTo(b));

            return ToSort.ToArray();
        }

        public static string PathToNameWithExtension(string path)
        {
            if (path.Contains("\\"))
                path = path.Substring(path.LastIndexOf("\\") + 1);
            if (path.Contains("/"))
                path = path.Substring(path.LastIndexOf("/") + 1);

            return path;
        }

        public static string PathToName(string path)
        {
            path = PathToNameWithExtension(path);
            if (path.Contains("."))
                path = path.Substring(0, path.LastIndexOf("."));

            return path;
        }

        public static string PathToExtension(string path)
        {
            return path.Substring(path.LastIndexOf("."));
        }

        public static string PathToFolder(string path)
        {
            string Folder = Path.GetDirectoryName(path);
            if (string.IsNullOrEmpty(Folder))
                Folder = ".";

            return Folder;
        }

        public static string MakePathRelativeTo(string path, string relativeTo)
        {
            if (string.IsNullOrEmpty(path))
                return path;
            if (string.IsNullOrEmpty(relativeTo))
                return path;

            return new Uri(relativeTo.Replace('\\', '/')).MakeRelativeUri(new Uri(path.Replace('\\', '/'))).ToString();
        }

        public static string PathCombine(params string[] paths)
        {
            if (paths == null || paths.Length == 0)
                return string.Empty;

            string CombinedPath = paths[0];

            for (int i = 1; i < paths.Length; i++)
            {
                string Path = paths[i];

                if (System.IO.Path.IsPathRooted(Path))
                    CombinedPath = Path;
                else
                    CombinedPath = System.IO.Path.Combine(CombinedPath, Path);
            }

            return Path.GetFullPath(CombinedPath);
        }

        private static object TimeSync = new object();
        public static void Time(string name, Action call)
        {
            Stopwatch Watch = new Stopwatch();
            Watch.Start();

            call();

            Watch.Stop();
            lock(TimeSync)
                Debug.WriteLine($"{name}: {Watch.ElapsedMilliseconds} ms");
        }

        public static T[][] SplitInParts<T>(T[] values, int nParts)
        {
            int PerPart = (int)Math.Ceiling((float)values.Length / nParts);
            return Helper.ArrayOfFunction(i => values.Skip(i * PerPart).Take(PerPart).ToArray(), nParts);
        }

        public static string[] StringSplitWhitespace(ReadOnlySpan<char> value, bool[] mask = null, string[] buffer = null)
        {
            int Cursor = 0;
            int Parts = 0;

            while (Cursor < value.Length)
            {
                while (Cursor < value.Length && (value[Cursor] == ' ' || value[Cursor] == '\t'))
                    Cursor++;

                int Start = Cursor;

                while (Cursor < value.Length && !(value[Cursor] == ' ' || value[Cursor] == '\t'))
                    Cursor++;

                int End = Cursor;

                if (End == Start)
                    continue;

                Parts++;

                if (buffer == null || buffer.Length < Parts)
                {
                    if (buffer == null)
                        buffer = new string[Parts];
                    else
                    {
                        string[] NewBuffer = new string[Parts];
                        for (int i = 0; i < Parts - 1; i++)
                            NewBuffer[i] = buffer[i];
                        buffer = NewBuffer;
                    }
                }

                if (mask != null && !mask[Parts - 1])
                    continue;

                buffer[Parts - 1] = new string(value.Slice(Start, End - Start));
            }

            if (buffer != null & Parts < buffer.Length)
            {
                string[] NewBuffer = new string[Parts];
                for (int i = 0; i < Parts; i++)
                    NewBuffer[i] = buffer[i];
                buffer = NewBuffer;
            }

            return buffer;
        }

        public static byte[] ToBytes(float[] data)
        {
            byte[] Bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static byte[] ToBytes(int[] data)
        {
            byte[] Bytes = new byte[data.Length * sizeof(int)];
            Buffer.BlockCopy(data, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static byte[] ToBytes(double[] data)
        {
            byte[] Bytes = new byte[data.Length * sizeof(double)];
            Buffer.BlockCopy(data, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static byte[] ToBytes(decimal[] data)
        {
            double[] DataDouble = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
                DataDouble[i] = (double)data[i];

            byte[] Bytes = new byte[DataDouble.Length * sizeof(double)];
            Buffer.BlockCopy(DataDouble, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static byte[] ToBytes(bool[] data)
        {
            byte[] Bytes = new byte[data.Length * sizeof(bool)];
            Buffer.BlockCopy(data, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static byte[] ToBytes(char[] data)
        {
            byte[] Bytes = new byte[data.Length * sizeof(char)];
            Buffer.BlockCopy(data, 0, Bytes, 0, Bytes.Length);

            return Bytes;
        }

        public static string RemoveInvalidChars(string path)
        {
            string Invalid = new string(Path.GetInvalidFileNameChars()) + new string(Path.GetInvalidPathChars()) + " ";
            foreach (char c in Invalid)
                path = path.Replace(c.ToString(), "");

            return path;
        }

        public static string ShortenString(string value, int maxLength, string ellipsis = "...")
        {
            if (value.Length <= maxLength)
                return value;

            int FirstLength = maxLength / 2 - ellipsis.Length;
            int SecondLength = maxLength - FirstLength - ellipsis.Length;

            return value.Substring(0, FirstLength) + ellipsis + value.Substring(value.Length - SecondLength, SecondLength);
        }

        public static HashSet<T> GetUniqueElements<T>(IEnumerable<T> elements)
        {
            HashSet<T> Unique = new HashSet<T>();
            foreach (var element in elements)
                if (!Unique.Contains(element))
                    Unique.Add(element);

            return Unique;
        }

        public static int[] GetIndicesOf<T>(T[] elements, Func<T, bool> qualifier)
        {
            List<int> Indices = new List<int>();
            for (int i = 0; i < elements.Length; i++)
                if (qualifier(elements[i]))
                    Indices.Add(i);

            return Indices.ToArray();
        }

        public static bool ExeutableIsOnPath(string executable)
        {
            string path = Environment.GetEnvironmentVariable("PATH");
            string[] directories = path.Split(Path.PathSeparator);
            return directories.Any(dir => File.Exists(Path.Combine(dir, executable)));
        }
    }

    public delegate void GPUTaskIterator<in T>(T item, int deviceID);
    public delegate bool GPUTaskCancelableIterator<in T>(T item, int deviceID);
}