using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Accord;
using BitMiracle.LibTiff.Classic;
using SkiaSharp;
using Warp.Headers;
using Warp.Tools;

namespace Warp
{
    public class Image : IDisposable
    {
        private readonly object Sync = new object();
        private static readonly object GlobalSync = new object();
        public static object FFT_CPU_Sync = new object();

        private bool IsDisposed = false;
        private readonly int ObjectID = -1;
        private readonly StackTrace ObjectCreationLocation = null;
        private static int LifetimeObjectCounter = 0;
        private static readonly List<int> LifetimeObjectIDs = new List<int>();
        private static readonly List<Image> LifetimeObjects = new List<Image>();
        private static readonly bool EnableObjectLogging = false;

        private static readonly HashSet<Image> OnDeviceObjects = new HashSet<Image>();
        public static void FreeDeviceAll()
        {
            Image[] Objects = OnDeviceObjects.ToArray();
            foreach (var item in Objects)
                item.FreeDevice();
        }

        private Image Parent;
        
        public int3 Dims;
        public int3 DimsFT => new int3(Dims.X / 2 + 1, Dims.Y, Dims.Z);
        public int2 DimsSlice => new int2(Dims.X, Dims.Y);
        public int2 DimsFTSlice => new int2(DimsFT.X, DimsFT.Y);
        public int3 DimsEffective => IsFT ? DimsFT : Dims;

        public float PixelSize = 1;

        public bool IsFT;
        public readonly bool IsComplex;
        public readonly bool IsHalf;

        public long ElementsComplex => IsFT ? DimsFT.Elements() : Dims.Elements();
        public long ElementsReal => IsComplex ? ElementsComplex * 2 : ElementsComplex;

        public long ElementsSliceComplex => IsFT ? DimsFTSlice.Elements() : DimsSlice.Elements();
        public long ElementsSliceReal => IsComplex ? ElementsSliceComplex * 2 : ElementsSliceComplex;

        public long ElementsLineComplex => IsFT ? DimsFTSlice.X : DimsSlice.X;
        public long ElementsLineReal => IsComplex ? ElementsLineComplex * 2 : ElementsLineComplex;

        private bool IsDeviceDirty = false;
        private IntPtr _DeviceData = IntPtr.Zero;

        private IntPtr DeviceData
        {
            get
            {
                if (_DeviceData == IntPtr.Zero)
                {
                    _DeviceData = !IsHalf ? GPU.MallocDevice(ElementsReal) : GPU.MallocDeviceHalf(ElementsReal);

                    lock (GlobalSync)
                        OnDeviceObjects.Add(this);

                    GPU.OnMemoryChanged();
                }

                return _DeviceData;
            }
        }

        private bool IsHostDirty = false;
        private float[][] _HostData = null;

        private float[][] HostData
        {
            get
            {
                if (_HostData == null)
                {
                    _HostData = new float[Dims.Z][];
                    for (int i = 0; i < Dims.Z; i++)
                        _HostData[i] = new float[ElementsSliceReal];
                }

                return _HostData;
            }
        }

        private bool IsHostPinnedDirty = false;
        private IntPtr _HostPinnedData = IntPtr.Zero;

        private IntPtr HostPinnedData
        {
            get
            {
                if (_HostPinnedData == IntPtr.Zero)
                {
                    _HostPinnedData = GPU.MallocHostPinned(ElementsReal);
                }

                return _HostPinnedData;
            }
        }

        public Image(float[][] data, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (data.Length != dims.Z || data[0].Length != ElementsSliceReal)
                throw new DimensionMismatchException();

            _HostData = data.ToArray();
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        public Image(float2[][] data, int3 dims, bool isft = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = true;
            IsHalf = ishalf;

            if (data.Length != dims.Z || data[0].Length != ElementsSliceComplex)
                throw new DimensionMismatchException();

            UpdateHostWithComplex(data);
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        public Image(float[] data, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (data.Length != ElementsReal)
                throw new DimensionMismatchException();

            float[][] Slices = new float[dims.Z][];

            for (int z = 0, i = 0; z < dims.Z; z++)
            {
                Slices[z] = new float[ElementsSliceReal];
                Array.Copy(data, z * dims.X * dims.Y, Slices[z], 0, dims.X * dims.Y);
                //for (int j = 0; j < Slices[z].Length; j++)
                //    Slices[z][j] = data[i++];
            }

            _HostData = Slices;
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        public Image(float2[] data, int3 dims, bool isft = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = true;
            IsHalf = ishalf;

            if (data.Length != ElementsComplex)
                throw new DimensionMismatchException();

            float[][] Slices = new float[dims.Z][];
            int i = 0;
            for (int z = 0; z < dims.Z; z++)
            {
                Slices[z] = new float[ElementsSliceReal];
                for (int j = 0; j < Slices[z].Length / 2; j++)
                {
                    Slices[z][j * 2] = data[i].X;
                    Slices[z][j * 2 + 1] = data[i].Y;
                    i++;
                }
            }

            _HostData = Slices;
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        public Image(float[] data, bool isft = false, bool iscomplex = false, bool ishalf = false) : 
            this(data, new int3(data.Length, 1, 1), isft, iscomplex, ishalf) { }

        public Image(float2[] data, bool isft = false, bool ishalf = false) : 
            this(data, new int3(data.Length, 1, 1), isft, ishalf) { }

        public Image(int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            _HostData = HostData; // Initializes new array since _HostData is null
            IsHostDirty = true;

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        public Image(IntPtr deviceData, int3 dims, bool isft = false, bool iscomplex = false, bool ishalf = false, bool fromPinned = false)
        {
            Dims = dims;
            IsFT = isft;
            IsComplex = iscomplex;
            IsHalf = ishalf;

            if (!fromPinned)
            {
                _DeviceData = !IsHalf ? GPU.MallocDevice(ElementsReal) : GPU.MallocDeviceHalf(ElementsReal);
                GPU.OnMemoryChanged();
                if (deviceData != IntPtr.Zero)
                {
                    if (!IsHalf)
                        GPU.CopyDeviceToDevice(deviceData, _DeviceData, ElementsReal);
                    else
                        GPU.CopyDeviceHalfToDeviceHalf(deviceData, _DeviceData, ElementsReal);
                }

                IsDeviceDirty = true;

                lock (GlobalSync)
                    OnDeviceObjects.Add(this);
            }
            else
            {
                _HostPinnedData = GPU.MallocHostPinned(ElementsReal);
                IsHostPinnedDirty = true;
            }

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    ObjectID = LifetimeObjectCounter++;
                    LifetimeObjectIDs.Add(ObjectID);
                    LifetimeObjects.Add(this);
                    ObjectCreationLocation = new StackTrace();
                }
        }

        ~Image()
        {
            Dispose();
        }

        public static Image FromFile(string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int[] layers, Stream stream = null)
        {
            MapHeader Header = MapHeader.ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType);
            float[][] Data = IOHelper.ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, layers, stream);
            if (layers is not null)
                Header.Dimensions.Z = layers.Length;

            return new Image(Data, Header.Dimensions) { PixelSize = Header.PixelSize.X };
        }

        public static Image FromFile(string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            MapHeader Header = MapHeader.ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType);
            float[][] Data = IOHelper.ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, layer < 0 ? null : new[] { layer }, stream);
            if (layer >= 0)
                Header.Dimensions.Z = 1;

            return new Image(Data, Header.Dimensions) { PixelSize = Header.PixelSize.X };
        }

        public static Image FromFile(string path, int layer = -1, Stream stream = null)
        {
            return FromFile(path, new int2(1, 1), 0, typeof(float), layer, stream);
        }

        public static Image FromFilePatient(int attempts, int mswait, string path, int2 headerlessSliceDims, int headerlessOffset, Type headerlessType, int layer = -1, Stream stream = null)
        {
            Image Result = null;
            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    Result = FromFile(path, headerlessSliceDims, headerlessOffset, headerlessType, layer, stream);
                    break;
                }
                catch (Exception exc)
                {
                    Thread.Sleep(mswait);
                }
            }

            if (Result == null)
                throw new Exception($"Could not successfully read {path} within the specified number of attempts.");

            return Result;
        }

        public static Image FromFilePatient(int attempts, int mswait, string path, int layer = -1, Stream stream = null)
        {
            return FromFilePatient(attempts, mswait, path, new int2(1, 1), 0, typeof(float), layer, stream);
        }

        public IntPtr GetDevice(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsHostDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        if (!IsHalf)
                            GPU.CopyHostToDevice(HostData[z], new IntPtr((long) DeviceData + ElementsSliceReal * z * sizeof (float)), ElementsSliceReal);
                        else
                            GPU.CopyHostToDeviceHalf(HostData[z], new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), ElementsSliceReal);

                    IsHostDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsHostPinnedDirty)
                {
                    GPU.CopyDeviceToHostPinned(HostPinnedData, DeviceData, ElementsReal);

                    IsHostPinnedDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsDeviceDirty = true;
                    IsHostDirty = false;
                    IsHostPinnedDirty = false;
                }

                return DeviceData;
            }
        }

        public void CopyToDevicePointer(IntPtr pointer)
        {
            if (IsDeviceDirty)
                GetHost(Intent.Read);

            lock (Sync)
            {
                for (int z = 0; z < Dims.Z; z++)
                    GPU.CopyHostToDevice(HostData[z], new IntPtr((long)pointer + ElementsSliceReal * z * sizeof(float)), ElementsSliceReal);
            }
        }

        public IntPtr GetDeviceSlice(int slice, Intent intent)
        {
            IntPtr Start = GetDevice(intent);
            Start = new IntPtr((long)Start + slice * ElementsSliceReal * (IsHalf ? sizeof(short) : sizeof (float)));

            return Start;
        }

        public float[][] GetHost(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsDeviceDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        if (!IsHalf)
                            GPU.CopyDeviceToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);
                        else
                            GPU.CopyDeviceHalfToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), HostData[z], ElementsSliceReal);

                    IsDeviceDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsHostPinnedDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        GPU.CopyHostToHost(new IntPtr((long)HostPinnedData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);

                    IsHostPinnedDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsHostDirty = true;
                    IsDeviceDirty = false;
                    IsHostPinnedDirty = false;
                }

                return HostData;
            }
        }

        public IntPtr GetHostPinned(Intent intent)
        {
            lock (Sync)
            {
                if ((intent & Intent.Read) > 0 && IsHostDirty)
                {
                    for (int z = 0; z < Dims.Z; z++)
                        GPU.CopyHostToHost(HostData[z], new IntPtr((long)HostPinnedData + ElementsSliceReal * z * sizeof(float)), ElementsSliceReal);

                    IsHostDirty = false;
                }
                else if ((intent & Intent.Read) > 0 && IsDeviceDirty)
                {
                    GPU.CopyDeviceToHostPinned(DeviceData, HostPinnedData, ElementsReal);

                    IsDeviceDirty = false;
                }

                if ((intent & Intent.Write) > 0)
                {
                    IsHostDirty = false;
                    IsDeviceDirty = false;
                    IsHostPinnedDirty = true;
                }

                return HostPinnedData;
            }
        }

        public IntPtr GetHostPinnedSlice(int slice, Intent intent)
        {
            IntPtr Start = GetHostPinned(intent);
            Start = new IntPtr((long)Start + slice * ElementsSliceReal * sizeof(float));

            return Start;
        }

        public float2[][] GetHostComplexCopy()
        {
            if (!IsComplex)
                throw new Exception("Data must be of complex type.");

            float[][] Data = GetHost(Intent.Read);
            float2[][] ComplexData = new float2[Dims.Z][];

            for (int z = 0; z < Dims.Z; z++)
            {
                float[] Slice = Data[z];
                float2[] ComplexSlice = new float2[DimsEffective.ElementsSlice()];
                for (int i = 0; i < ComplexSlice.Length; i++)
                    ComplexSlice[i] = new float2(Slice[i * 2], Slice[i * 2 + 1]);

                ComplexData[z] = ComplexSlice;
            }

            return ComplexData;
        }

        public void UpdateHostWithComplex(float2[][] complexData)
        {
            if (complexData.Length != Dims.Z ||
                complexData[0].Length != DimsEffective.ElementsSlice())
                throw new DimensionMismatchException();

            float[][] Data = GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                float[] Slice = Data[z];
                float2[] ComplexSlice = complexData[z];

                for (int i = 0; i < ComplexSlice.Length; i++)
                {
                    Slice[i * 2] = ComplexSlice[i].X;
                    Slice[i * 2 + 1] = ComplexSlice[i].Y;
                }
            }
        }

        public float[] GetHostContinuousCopy()
        {
            float[] Continuous = new float[ElementsReal];
            float[][] Data = GetHost(Intent.Read);
            unsafe
            {
                fixed (float* ContinuousPtr = Continuous)
                {
                    float* ContinuousP = ContinuousPtr;
                    for (int i = 0; i < Data.Length; i++)
                    {
                        fixed (float* DataPtr = Data[i])
                        {
                            float* DataP = DataPtr;
                            for (int j = 0; j < Data[i].Length; j++)
                                *ContinuousP++ = *DataP++;
                        }
                    }
                }
            }

            return Continuous;
        }

        public Image FreeDevice()
        {
            lock (Sync)
            {
                if (_DeviceData != IntPtr.Zero)
                {
                    if (IsDeviceDirty)
                        for (int z = 0; z < Dims.Z; z++)
                            if (!IsHalf)
                                GPU.CopyDeviceToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(float)), HostData[z], ElementsSliceReal);
                            else
                                GPU.CopyDeviceHalfToHost(new IntPtr((long)DeviceData + ElementsSliceReal * z * sizeof(short)), HostData[z], ElementsSliceReal);
                    GPU.FreeDevice(DeviceData);
                    GPU.OnMemoryChanged();
                    _DeviceData = IntPtr.Zero;
                    IsDeviceDirty = false;

                    lock (GlobalSync)
                        OnDeviceObjects.Remove(this);
                }

                IsHostDirty = true;
            }

            return this;
        }

        public Image AndFreeParent()
        {
            Parent?.FreeDevice();
            Parent = null;

            return this;
        }

        public Image AndDisposeParent()
        {
            if (Parent == null)
                throw new Exception("No parent to dispose");
            Parent?.Dispose();
            Parent = null;

            return this;
        }

        public void WriteMRC(string path, float pixelSize, bool doStatistics = false, HeaderMRC header = null)
        {
            if (header == null)
            {
                header = new HeaderMRC();
            }

            header.PixelSize = new float3(pixelSize);
            header.Dimensions = IsFT ? DimsFT : Dims;
            header.Dimensions.X *= IsComplex ? 2 : 1;

            //Stopwatch Watch = new Stopwatch();
            //Watch.Start();

            if (doStatistics)
            {
                float[][] Data = GetHost(Intent.Read);
                float Min = float.MaxValue, Max = float.MinValue;

                if (false)//Dims.Z > 4)
                    Parallel.For(0, Dims.Z, z =>
                    {
                        unsafe
                        {
                            fixed (float* DataPtr = Data[z])
                            {
                                float LocalMin = float.MaxValue;
                                float LocalMax = float.MinValue;
                                int Length = Data[z].Length;
                                float* DataP = DataPtr;

                                for (int i = 0; i < Length; i++)
                                {
                                    LocalMin = MathF.Min(LocalMin, *DataP);
                                    LocalMax = MathF.Max(LocalMax, *DataP);
                                    DataP++;
                                }

                                lock (Data)
                                {
                                    Min = Math.Min(LocalMin, Min);
                                    Max = Math.Max(LocalMax, Max);
                                }
                            }
                        }
                    });
                else
                    for (int z = 0; z < Dims.Z; z++)
                    {
                        unsafe
                        {
                            fixed (float* DataPtr = Data[z])
                            {
                                int Length = Data[z].Length;
                                float* DataP = DataPtr;

                                for (int i = 0; i < Length; i++)
                                {
                                    Min = MathF.Min(Min, *DataP);
                                    Max = MathF.Max(Max, *DataP);
                                    DataP++;
                                }
                            }
                        }
                    }
                header.MinValue = Min;
                header.MaxValue = Max;
                
                // check that min/max fall within range for fp16 and warn if not
                Type dataType = header.GetValueType();
                float typeMinValue = float.MinValue;
                float typeMaxValue = float.MaxValue;
                
                if (dataType == typeof(Half))
                {
                    typeMinValue = (float)Half.MinValue;
                    typeMaxValue = (float)Half.MaxValue;
                }

                if (header.MinValue < typeMinValue || header.MaxValue > typeMaxValue)
                {
                    Console.WriteLine($"WARNING: data being written to {path} contains values are outside range for {dataType}, switching to float32");
                    header.SetValueType(typeof(float));
                }
            }

            //Console.WriteLine("Stats: " + Watch.Elapsed.TotalMilliseconds);
            //Watch.Reset();
            //Watch.Start();

            IOHelper.WriteMapFloat(path, header, GetHost(Intent.Read));

            //Console.WriteLine("IO: " + Watch.Elapsed.TotalMilliseconds);
            //Watch.Stop();
        }

        public void WriteMRC(string path, bool doStatistics = false, HeaderMRC header = null)
        {
            WriteMRC(path, PixelSize, doStatistics, header);
        }

        public void WriteMRC16b(string path, float pixelSize, bool doStatistics = false, HeaderMRC header = null)
        {
            if (header == null)
                header = new HeaderMRC();
            if (Environment.GetEnvironmentVariable("WARP_FORCE_MRC_FLOAT32") != null)
                header.SetValueType(typeof(float));
            else
                header.SetValueType(typeof(Half));

            WriteMRC(path, pixelSize, doStatistics, header);
        }

        public void WriteMRC16b(string path, bool doStatistics = false, HeaderMRC header = null)
        {
            if (header == null)
                header = new HeaderMRC();
            if (Environment.GetEnvironmentVariable("WARP_FORCE_MRC_FLOAT32") != null)
                header.SetValueType(typeof(float));
            else
                header.SetValueType(typeof(Half));

            WriteMRC(path, doStatistics, header);
        }

        public void WriteTIFF(string path, float pixelSize, Type dataType)
        {
            string FlipYEnvVar = Environment.GetEnvironmentVariable("WARP_DONT_FLIPY");
            bool DoFlipY = string.IsNullOrEmpty(FlipYEnvVar);

            int Width = (int)ElementsLineReal;
            int Height = Dims.Y;
            int SamplesPerPixel = 1;
            int BitsPerSample = 8;

            if (dataType == typeof(byte))
                BitsPerSample = 8;
            else if (dataType == typeof(short))
                BitsPerSample = 16;
            else if (dataType == typeof(int))
                BitsPerSample = 32;
            else if (dataType == typeof(long))
                BitsPerSample = 64;
            else if (dataType == typeof(float))
                BitsPerSample = 32;
            else if (dataType == typeof(double))
                BitsPerSample = 64;
            else
                throw new Exception("Unsupported data type.");

            SampleFormat Format = SampleFormat.INT;
            if (dataType == typeof(byte))
                Format = SampleFormat.UINT;
            else if (dataType == typeof(float) || dataType == typeof(double))
                Format = SampleFormat.IEEEFP;

            int BytesPerSample = BitsPerSample / 8;

            float[][] Data = GetHost(Intent.Read);
            int PageLength = Data[0].Length;
            unsafe
            {
                using (Tiff output = Tiff.Open(path, "w"))
                {
                    float[] DataFlipped = DoFlipY ? new float[Data[0].Length] : null;
                    byte[] BytesData = new byte[ElementsSliceReal * BytesPerSample];

                    for (int z = 0; z < Dims.Z; z++)
                    {
                        if (DoFlipY)
                        {
                            // Annoyingly, flip Y axis to adhere to MRC convention

                            fixed (float* DataFlippedPtr = DataFlipped)
                            fixed (float* DataPtr = Data[z])
                            {
                                for (int y = 0; y < Height; y++)
                                {
                                    int YOffset = y * Width;
                                    int YOffsetFlipped = (Height - 1 - y) * Width;

                                    for (int x = 0; x < Width; x++)
                                        DataFlippedPtr[YOffset + x] = DataPtr[YOffsetFlipped + x];
                                }
                            }
                        }
                        else
                        {
                            DataFlipped = Data[z];
                        }

                        fixed (byte* BytesPtr = BytesData)
                        fixed (float* DataPtr = DataFlipped)
                        {
                            if (dataType == typeof(byte))
                            {
                                for (int i = 0; i < PageLength; i++)
                                    BytesPtr[i] = (byte)Math.Max(0, Math.Min(byte.MaxValue, (int)DataPtr[i]));
                            }
                            else if (dataType == typeof(short))
                            {
                                short* ConvPtr = (short*)BytesPtr;
                                for (int i = 0; i < PageLength; i++)
                                    ConvPtr[i] = (short)Math.Max(short.MinValue, Math.Min(short.MaxValue, (int)DataPtr[i]));
                            }
                            else if (dataType == typeof(int))
                            {
                                int* ConvPtr = (int*)BytesPtr;
                                for (int i = 0; i < PageLength; i++)
                                    ConvPtr[i] = (int)Math.Max(int.MinValue, Math.Min(int.MaxValue, (int)DataPtr[i]));
                            }
                            else if (dataType == typeof(long))
                            {
                                long* ConvPtr = (long*)BytesPtr;
                                for (int i = 0; i < PageLength; i++)
                                    ConvPtr[i] = (long)Math.Max(long.MinValue, Math.Min(long.MaxValue, (long)DataPtr[i]));
                            }
                            else if (dataType == typeof(float))
                            {
                                float* ConvPtr = (float*)BytesPtr;
                                for (int i = 0; i < PageLength; i++)
                                    ConvPtr[i] = DataPtr[i];
                            }
                            else if (dataType == typeof(double))
                            {
                                double* ConvPtr = (double*)BytesPtr;
                                for (int i = 0; i < PageLength; i++)
                                    ConvPtr[i] = DataPtr[i];
                            }
                        }

                        int page = z;
                        {
                            output.SetField(TiffTag.IMAGEWIDTH, Width / SamplesPerPixel);
                            output.SetField(TiffTag.IMAGELENGTH, Height);
                            output.SetField(TiffTag.SAMPLESPERPIXEL, SamplesPerPixel);
                            output.SetField(TiffTag.SAMPLEFORMAT, Format);
                            output.SetField(TiffTag.BITSPERSAMPLE, BitsPerSample);
                            output.SetField(TiffTag.ORIENTATION, Orientation.BOTLEFT);
                            output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

                            output.SetField(TiffTag.COMPRESSION, Compression.LZW);

                            output.SetField(TiffTag.ROWSPERSTRIP, output.DefaultStripSize(0));
                            output.SetField(TiffTag.XRESOLUTION, 100.0);
                            output.SetField(TiffTag.YRESOLUTION, 100.0);
                            output.SetField(TiffTag.RESOLUTIONUNIT, ResUnit.INCH);

                            // specify that it's a page within the multipage file
                            output.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                            // specify the page number
                            output.SetField(TiffTag.PAGENUMBER, page, Dims.Z);

                            for (int j = 0; j < Height; j++)
                                output.WriteScanline(Helper.Subset(BytesData, j * Width * BytesPerSample, (j + 1) * Width * BytesPerSample), j);

                            output.WriteDirectory();
                            output.FlushData();
                        }
                    }
                }
            }
        }

        public void WritePNG(string path)
        {
            if (Dims.Z > 1)
                throw new DimensionMismatchException("Image cannot have more than 1 layer for PNG.");

            using (SKBitmap Image = new SKBitmap(Dims.X, Dims.Y, SKColorType.Gray8, SKAlphaType.Opaque))
            {
                unsafe
                {
                    float[] Data = GetHost(Intent.Read)[0];

                    for (int y = 0; y < Dims.Y; y++)
                    {
                        for (int x = 0; x < Dims.X; x++)
                        {
                            int i = y * Dims.X + x;
                            byte PixelValue = (byte)Math.Max(0, Math.Min(255, (long)Data[(Dims.Y - 1 - y) * Dims.X + x]));
                            ((byte*)Image.GetAddress(x, y))[0] = PixelValue;
                        }
                    }
                }

                using (Stream s = File.Create(path))
                {
                    Image.Encode(s, SKEncodedImageFormat.Png, 100);
                }
            }
        }

        public void Dispose()
        {
            lock (Sync)
            {
                if (_DeviceData != IntPtr.Zero)
                {
                    GPU.FreeDevice(_DeviceData);
                    GPU.OnMemoryChanged();
                    _DeviceData = IntPtr.Zero;
                    IsDeviceDirty = false;

                    lock (GlobalSync)
                        OnDeviceObjects.Remove(this);
                }

                if (_HostPinnedData != IntPtr.Zero)
                {
                    GPU.FreeHostPinned(_HostPinnedData);
                    _HostPinnedData = IntPtr.Zero;
                    IsHostPinnedDirty = false;
                }

                _HostData = null;
                IsHostDirty = false;

                IsDisposed = true;
            }

            if (EnableObjectLogging)
                lock (GlobalSync)
                {
                    if (LifetimeObjectIDs.Contains(ObjectID))
                        LifetimeObjectIDs.Remove(ObjectID);
                    if (LifetimeObjects.Contains(this))
                        LifetimeObjects.Remove(this);
                }
        }

        public Image GetCopy()
        {
            Image Result = new Image(GetHostContinuousCopy(), Dims, IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };
            Result.Parent = this;

            return Result;
        }

        public Image GetCopyGPU()
        {
            Image Result = new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };
            Result.Parent = this;

            return Result;
        }

        public void TransformValues(Func<float, float> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            foreach (var slice in Data)
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = f(slice[i]);
        }

        public void TransformValues(Func<int, float, float> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            int gi = 0;
            foreach (var slice in Data)
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = f(gi++, slice[i]);
        }

        public void TransformValues(Func<int, int, int, float, float> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;
            if (IsComplex)
                Width *= 2;

            for (int z = 0; z < Dims.Z; z++)
                for (int y = 0; y < Dims.Y; y++)
                    for (int x = 0; x < Width; x++)
                        Data[z][y * Width + x] = f(x, y, z, Data[z][y * Width + x]);
        }

        public void TransformComplexValues(Func<float2, float2> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            foreach (var slice in Data)
                for (int i = 0; i < slice.Length / 2; i++)
                {
                    float2 Transformed = f(new float2(slice[i * 2 + 0], slice[i * 2 + 1]));
                    slice[i * 2 + 0] = Transformed.X;
                    slice[i * 2 + 1] = Transformed.Y;
                }
        }

        public void TransformComplexValues(Func<int, int, int, float2, float2> f)
        {
            float[][] Data = GetHost(Intent.ReadWrite);
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;

            Helper.ForEachElementFT(Dims, (x, y, z, xx, yy, zz) =>
            {
                float2 Transformed = f(xx, yy, zz, new float2(Data[z][(y * Width + x) * 2 + 0], Data[z][(y * Width + x) * 2 + 1]));
                Data[z][(y * Width + x) * 2 + 0] = Transformed.X;
                Data[z][(y * Width + x) * 2 + 1] = Transformed.Y;
            });
        }

        public void TransformRegionValues(int3 extent, int3 center, Func<int3, int3, float, float> f)
        {
            if (IsComplex)
                throw new Exception("Does not work on complex data");

            float[][] Data = GetHost(Intent.ReadWrite);
            int3 Start = center - extent / 2;
            Start = int3.Max(Start, 0);
            Start = int3.Min(Start, Dims - 1);
            int3 End = (center - extent / 2) + extent;
            End = int3.Max(End, 0);
            End = int3.Min(End, Dims);

            for (int z = Start.Z; z < End.Z; z++)
                for (int y = Start.Y; y < End.Y; y++)
                    for (int x = Start.X; x < End.X; x++)
                    {
                        int3 Coord = new int3(x, y, z);
                        int3 CoordCentered = Coord - center;

                        Data[z][y * Dims.X + x] = f(Coord, CoordCentered, Data[z][y * Dims.X + x]);
                    }
        }

        public float GetInterpolatedValue(float3 pos)
        {
            float3 Weights = new float3(pos.X - (float)Math.Floor(pos.X),
                                        pos.Y - (float)Math.Floor(pos.Y),
                                        pos.Z - (float)Math.Floor(pos.Z));

            float[][] Data = GetHost(Intent.Read);

            int3 Pos0 = new int3(Math.Max(0, Math.Min(Dims.X - 1, (int)pos.X)),
                                 Math.Max(0, Math.Min(Dims.Y - 1, (int)pos.Y)),
                                 Math.Max(0, Math.Min(Dims.Z - 1, (int)pos.Z)));
            int3 Pos1 = new int3(Math.Min(Dims.X - 1, Pos0.X + 1),
                                 Math.Min(Dims.Y - 1, Pos0.Y + 1),
                                 Math.Min(Dims.Z - 1, Pos0.Z + 1));

            if (Dims.Z == 1)
            {
                float v00 = Data[0][Pos0.Y * Dims.X + Pos0.X];
                float v01 = Data[0][Pos0.Y * Dims.X + Pos1.X];
                float v10 = Data[0][Pos1.Y * Dims.X + Pos0.X];
                float v11 = Data[0][Pos1.Y * Dims.X + Pos1.X];

                float v0 = MathHelper.Lerp(v00, v01, Weights.X);
                float v1 = MathHelper.Lerp(v10, v11, Weights.X);

                return MathHelper.Lerp(v0, v1, Weights.Y);
            }
            else
            {
                float v000 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos0.X];
                float v001 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos1.X];
                float v010 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos0.X];
                float v011 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos1.X];

                float v100 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos0.X];
                float v101 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos1.X];
                float v110 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos0.X];
                float v111 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos1.X];

                float v00 = MathHelper.Lerp(v000, v001, Weights.X);
                float v01 = MathHelper.Lerp(v010, v011, Weights.X);
                float v10 = MathHelper.Lerp(v100, v101, Weights.X);
                float v11 = MathHelper.Lerp(v110, v111, Weights.X);

                float v0 = MathHelper.Lerp(v00, v01, Weights.Y);
                float v1 = MathHelper.Lerp(v10, v11, Weights.Y);

                return MathHelper.Lerp(v0, v1, Weights.Z);
            }
        }

        public float GetInterpolatedValue(float2 pos, int slice)
        {
            float2 Weights = new float2(pos.X - (float)Math.Floor(pos.X),
                                        pos.Y - (float)Math.Floor(pos.Y));

            float[][] Data = GetHost(Intent.Read);

            int2 Pos0 = new int2(Math.Max(0, Math.Min(Dims.X - 1, (int)pos.X)),
                                 Math.Max(0, Math.Min(Dims.Y - 1, (int)pos.Y)));
            int2 Pos1 = new int2(Math.Min(Dims.X - 1, Pos0.X + 1),
                                 Math.Min(Dims.Y - 1, Pos0.Y + 1));

            float v00 = Data[slice][Pos0.Y * Dims.X + Pos0.X];
            float v01 = Data[slice][Pos0.Y * Dims.X + Pos1.X];
            float v10 = Data[slice][Pos1.Y * Dims.X + Pos0.X];
            float v11 = Data[slice][Pos1.Y * Dims.X + Pos1.X];

            float v0 = MathHelper.Lerp(v00, v01, Weights.X);
            float v1 = MathHelper.Lerp(v10, v11, Weights.X);

            return MathHelper.Lerp(v0, v1, Weights.Y);
        }

        public void AddInterpolatedValue(float value, float3 pos)
        {
            float3 Weights = new float3(pos.X - (float)Math.Floor(pos.X),
                                        pos.Y - (float)Math.Floor(pos.Y),
                                        pos.Z - (float)Math.Floor(pos.Z));

            float[][] Data = GetHost(Intent.Read);

            int3 Pos0 = new int3(Math.Max(0, Math.Min(Dims.X - 1, (int)pos.X)),
                                 Math.Max(0, Math.Min(Dims.Y - 1, (int)pos.Y)),
                                 Math.Max(0, Math.Min(Dims.Z - 1, (int)pos.Z)));
            int3 Pos1 = new int3(Math.Min(Dims.X - 1, Pos0.X + 1),
                                 Math.Min(Dims.Y - 1, Pos0.Y + 1),
                                 Math.Min(Dims.Z - 1, Pos0.Z + 1));

            if (Dims.Z == 1)
            {
                float w00 = (1 - Weights.X) * (1 - Weights.Y);
                float w01 = (Weights.X) * (1 - Weights.Y);
                float w10 = (1 - Weights.X) * (Weights.Y);
                float w11 = (Weights.X) * (Weights.Y);

                Data[0][Pos0.Y * Dims.X + Pos0.X] += value * w00;
                Data[0][Pos0.Y * Dims.X + Pos1.X] += value * w01;
                Data[0][Pos1.Y * Dims.X + Pos0.X] += value * w10;
                Data[0][Pos1.Y * Dims.X + Pos1.X] += value * w11;
            }
            else
            {
                float v000 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos0.X];
                float v001 = Data[Pos0.Z][Pos0.Y * Dims.X + Pos1.X];
                float v010 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos0.X];
                float v011 = Data[Pos0.Z][Pos1.Y * Dims.X + Pos1.X];

                float v100 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos0.X];
                float v101 = Data[Pos1.Z][Pos0.Y * Dims.X + Pos1.X];
                float v110 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos0.X];
                float v111 = Data[Pos1.Z][Pos1.Y * Dims.X + Pos1.X];

                float v00 = MathHelper.Lerp(v000, v001, Weights.X);
                float v01 = MathHelper.Lerp(v010, v011, Weights.X);
                float v10 = MathHelper.Lerp(v100, v101, Weights.X);
                float v11 = MathHelper.Lerp(v110, v111, Weights.X);

                float v0 = MathHelper.Lerp(v00, v01, Weights.Y);
                float v1 = MathHelper.Lerp(v10, v11, Weights.Y);

                float w000 = (1 - Weights.Z) * (1 - Weights.X) * (1 - Weights.Y);
                float w001 = (1 - Weights.Z) * (Weights.X) * (1 - Weights.Y);
                float w010 = (1 - Weights.Z) * (1 - Weights.X) * (Weights.Y);
                float w011 = (1 - Weights.Z) * (Weights.X) * (Weights.Y);

                float w100 = (Weights.Z) * (1 - Weights.X) * (1 - Weights.Y);
                float w101 = (Weights.Z) * (Weights.X) * (1 - Weights.Y);
                float w110 = (Weights.Z) * (1 - Weights.X) * (Weights.Y);
                float w111 = (Weights.Z) * (Weights.X) * (Weights.Y);

                Data[Pos0.Z][Pos0.Y * Dims.X + Pos0.X] += value * w000;
                Data[Pos0.Z][Pos0.Y * Dims.X + Pos1.X] += value * w001;
                Data[Pos0.Z][Pos1.Y * Dims.X + Pos0.X] += value * w010;
                Data[Pos0.Z][Pos1.Y * Dims.X + Pos1.X] += value * w011;

                Data[Pos1.Z][Pos0.Y * Dims.X + Pos0.X] += value * w100;
                Data[Pos1.Z][Pos0.Y * Dims.X + Pos1.X] += value * w101;
                Data[Pos1.Z][Pos1.Y * Dims.X + Pos0.X] += value * w110;
                Data[Pos1.Z][Pos1.Y * Dims.X + Pos1.X] += value * w111;
            }
        }

        #region As...

        public Image AsHalf()
        {
            Image Result;

            if (!IsHalf)
            {
                Result = new Image(IntPtr.Zero, Dims, IsFT, IsComplex, true);
                GPU.SingleToHalf(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), ElementsReal);
            }
            else
            {
                Result = new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, true);
            }

            Result.Parent = this;
            Result.PixelSize = this.PixelSize;

            return Result;
        }

        public Image AsSingle()
        {
            Image Result;

            if (IsHalf)
            {
                IntPtr Temp = GPU.MallocDevice(ElementsReal);
                GPU.OnMemoryChanged();
                GPU.HalfToSingle(GetDevice(Intent.Read), Temp, ElementsReal);

                Result = new Image(Temp, Dims, IsFT, IsComplex, false);
                GPU.FreeDevice(Temp);
                GPU.OnMemoryChanged();
            }
            else
            {
                Result = new Image(GetDevice(Intent.Read), Dims, IsFT, IsComplex, false);
            }

            Result.Parent = this;
            Result.PixelSize = this.PixelSize;

            return Result;
        }

        public Image AsSum3D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(1, 1, 1)) { PixelSize = PixelSize };
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsReal, 1);

            Result.Parent = this;

            return Result;
        }

        public Image AsSum2D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(Dims.Z, 1, 1)) { PixelSize = PixelSize };
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z);

            Result.Parent = this;

            return Result;
        }

        public Image AsSum1D()
        {
            if (IsComplex || IsHalf)
                throw new Exception("Data type not supported.");

            Image Result = new Image(IntPtr.Zero, new int3(Dims.Y * Dims.Z, 1, 1)) { PixelSize = PixelSize };
            GPU.Sum(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), (uint)ElementsLineReal, (uint)(Dims.Y * Dims.Z));

            Result.Parent = this;

            return Result;
        }

        public Image AsRegion(int3 origin, int3 dimensions)
        {
            if (origin.X + dimensions.X > Dims.X || 
                origin.Y + dimensions.Y > Dims.Y || 
                origin.Z + dimensions.Z > Dims.Z)
                throw new IndexOutOfRangeException();

            float[][] Source = GetHost(Intent.Read);
            float[][] Region = new float[dimensions.Z][];

            int3 RealSourceDimensions = DimsEffective;
            if (IsComplex)
                RealSourceDimensions.X *= 2;
            int3 RealDimensions = new int3((IsFT ? dimensions.X / 2 + 1 : dimensions.X) * (IsComplex ? 2 : 1),
                                           dimensions.Y,
                                           dimensions.Z);

            for (int z = 0; z < RealDimensions.Z; z++)
            {
                float[] SourceSlice = Source[z + origin.Z];
                float[] Slice = new float[RealDimensions.ElementsSlice()];

                unsafe
                {
                    fixed (float* SourceSlicePtr = SourceSlice)
                    fixed (float* SlicePtr = Slice)
                        for (int y = 0; y < RealDimensions.Y; y++)
                        {
                            int YOffset = y + origin.Y;
                            for (int x = 0; x < RealDimensions.X; x++)
                                SlicePtr[y * RealDimensions.X + x] = SourceSlicePtr[YOffset * RealSourceDimensions.X + x + origin.X];
                        }
                }

                Region[z] = Slice;
            }

            Image Result = new Image(Region, dimensions, IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };
            Result.Parent = this;

            return Result;
        }

        public Image AsPadded(int2 dimensions, bool isDecentered = false)
        {
            if (IsHalf)
                throw new Exception("Half precision not supported for padding.");

            if (IsComplex != IsFT)
                throw new Exception("FT format can only have complex data for padding purposes.");

            if (dimensions == new int2(Dims))
                return GetCopy();

            if (IsFT && (new int2(Dims) < dimensions) == (new int2(Dims) > dimensions))
                throw new Exception("For FT padding/cropping, both dimensions must be either smaller, or bigger.");

            Image Padded = null;

            if (!IsComplex && !IsFT)
            {
                Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), false, false, false) { PixelSize = PixelSize };
                if (isDecentered)
                {
                    if (dimensions.X > Dims.X && dimensions.Y > Dims.Y)
                        GPU.PadFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                    else if (dimensions.X < Dims.X && dimensions.Y < Dims.Y)
                        GPU.CropFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                    else
                        throw new Exception("All new dimensions must be either bigger or smaller than old ones.");
                }
                else
                {
                    GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                }
            }
            else if (IsComplex && IsFT)
            {
                Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), true, true, false) { PixelSize = PixelSize * (float)Dims.X / dimensions.X };
                if (dimensions > new int2(Dims))
                    GPU.PadFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
                else
                    GPU.CropFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);
            }

            Padded.Parent = this;

            return Padded;
        }

        public Image AsPaddedClamped(int2 dimensions)
        {
            if (IsHalf || IsComplex || IsFT)
                throw new Exception("Wrong data format, only real-valued non-FT supported.");

            Image Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, Dims.Z), false, false, false) { PixelSize = PixelSize };
            GPU.PadClamped(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims.Slice(), new int3(dimensions), (uint)Dims.Z);

            Padded.Parent = this;

            return Padded;
        }

        public Image AsPaddedClamped(int3 dimensions)
        {
            if (IsHalf || IsComplex || IsFT)
                throw new Exception("Wrong data format, only real-valued non-FT supported.");

            Image Padded = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, dimensions.Z), false, false, false) { PixelSize = PixelSize };
            GPU.PadClamped(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);

            Padded.Parent = this;

            return Padded;
        }

        public Image AsPadded(int3 dimensions, bool isDecentered = false)
        {
            if (IsHalf)
                throw new Exception("Half precision not supported for padding.");

            if (IsComplex != IsFT)
                throw new Exception("FT format can only have complex data for padding purposes.");

            if (IsFT && Dims < dimensions == Dims > dimensions)
                throw new Exception("For FT padding/cropping, both dimensions must be either smaller, or bigger.");

            Image Padded = null;

            if (!IsComplex && !IsFT)
            {
                //Padded = new Image(IntPtr.Zero, dimensions, false, false, false);
                //GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);

                Padded = new Image(IntPtr.Zero, dimensions, false, false, false) { PixelSize = PixelSize };
                if (isDecentered)
                {
                    if (dimensions.X > Dims.X && dimensions.Y > Dims.Y && dimensions.Z > Dims.Z)
                        GPU.PadFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                    else if (dimensions.X < Dims.X && dimensions.Y < Dims.Y && dimensions.Z < Dims.Z)
                        GPU.CropFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                    else
                        GPU.PadFTFull(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                    //throw new Exception("All new dimensions must be either bigger or smaller than old ones.");
                }
                else
                {
                    GPU.Pad(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                }
            }
            else if (IsComplex && IsFT)
            {
                Padded = new Image(IntPtr.Zero, dimensions, true, true, false) { PixelSize = PixelSize * (float)Dims.X / dimensions.X };
                if (dimensions > Dims)
                    GPU.PadFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
                else
                    GPU.CropFT(GetDevice(Intent.Read), Padded.GetDevice(Intent.Write), Dims, dimensions, 1);
            }

            Padded.Parent = this;

            return Padded;
        }

        public Image AsPadded_CPU(int2 dimensions)
        {
            Image Result = new Image(new int3(dimensions.X, dimensions.Y, Dims.Z), IsFT, IsComplex) { PixelSize = PixelSize * (IsFT ? (float)Dims.X / dimensions.X : 1) };

            if (!IsFT && IsComplex)
                throw new Exception("Format not supported");

            if (!IsFT)
            {
                int2 CenterOld = new int2(Dims) / 2;
                int2 CenterNew = dimensions / 2;

                for (int z = 0; z < Dims.Z; z++)
                {
                    float[] DataOld = GetHost(Intent.Read)[z];
                    float[] DataNew = Result.GetHost(Intent.Read)[z];

                    unsafe
                    {
                        fixed (float* DataOldP = DataOld)
                        fixed (float* DataNewP = DataNew)
                            for (int y = 0; y < dimensions.Y; y++)
                            {
                                int yy = y - CenterNew.Y + CenterOld.Y;

                                if (yy < 0 || yy >= Dims.Y)
                                    continue;

                                int OffsetOld = yy * Dims.X;
                                int OffsetNew = y * dimensions.X;

                                for (int x = 0; x < dimensions.X; x++)
                                {
                                    int xx = x - CenterNew.X + CenterOld.X;

                                    if (xx < 0 || xx >= Dims.X)
                                        continue;

                                    DataNewP[OffsetNew + x] = DataOldP[OffsetOld + xx];
                                }
                            }
                    }
                }
            }
            else
            {
                for (int z = 0; z < Dims.Z; z++)
                {
                    float[] DataOld = GetHost(Intent.Read)[z];
                    float[] DataNew = Result.GetHost(Intent.Read)[z];

                    unsafe
                    {
                        fixed(float* DataOldP = DataOld)
                        fixed(float* DataNewP = DataNew)
                            for (int y = 0; y < dimensions.Y; y++)
                            {
                                int yy = y < dimensions.Y / 2 + 1 ? y : y - dimensions.Y;

                                if (Dims.Y != dimensions.Y)
                                {
                                    if (yy >= Dims.Y / 2)
                                        continue;
                                    if (yy <= -Dims.Y / 2)
                                        continue;
                                }

                                if (yy < 0)
                                    yy += Dims.Y;

                                if (yy < 0 || yy >= Dims.Y)
                                    continue;

                                int OffsetNew = y * (dimensions.X / 2 + 1);
                                int OffsetOld = yy * (Dims.X / 2 + 1);
                                int Width = Math.Min(dimensions.X / 2 + 1, Dims.X / 2 + 1);

                                if (!IsComplex)
                                    for (int x = 0; x < Width; x++)
                                        DataNewP[OffsetNew + x] = DataOldP[OffsetOld + x];
                                else
                                    for (int x = 0; x < Width; x++)
                                    {
                                        DataNewP[(OffsetNew + x) * 2] = DataOldP[(OffsetOld + x) * 2];
                                        DataNewP[(OffsetNew + x) * 2 + 1] = DataOldP[(OffsetOld + x) * 2 + 1];
                                    }
                            }
                    }
                }
            }

            Result.Parent = this;
            return Result;
        }

        public Image AsFFT(bool isvolume = false, int plan = 0)
        {
            if (IsHalf || IsComplex || IsFT)
                throw new Exception("Data format not supported.");

            int Plan = plan;
            if (Plan == 0)
                Plan = FFTPlanCache.GetFFTPlan(isvolume ? Dims : Dims.Slice(), isvolume ? 1 : Dims.Z);

            Image FFT = new Image(IntPtr.Zero, Dims, true, true, false) { PixelSize = PixelSize };
            GPU.FFT(GetDevice(Intent.Read), FFT.GetDevice(Intent.Write), isvolume ? Dims : Dims.Slice(), isvolume ? 1 : (uint)Dims.Z, Plan);

            FFT.Parent = this;

            return FFT;
        }

        public Image AsFFT_CPU(bool isVolume)
        {
            Image Result = null;

            if (IsFT || IsComplex)
                throw new Exception("Data must be in non-FT, non-complex format");

            if (isVolume)
            {
                IntPtr Original;
                IntPtr Transformed;
                IntPtr Plan;

                // FFTW is not thread-safe (except the execute function)
                lock (Image.FFT_CPU_Sync)
                {
                    Original = FFTW.alloc_real(Dims.Elements());
                    Transformed = FFTW.alloc_complex(Dims.ElementsFFT());

                    Plan = FFTW.plan_dft_r2c(3, new int[] { Dims.Z, Dims.Y, Dims.X }, Original, Transformed, PlannerFlags.Measure);
                }

                for (int z = 0; z < Dims.Z; z++)
                    Marshal.Copy(GetHost(Intent.Read)[z], 0, new IntPtr((long)Original + Dims.ElementsSlice() * sizeof(float) * z), (int)Dims.ElementsSlice());

                FFTW.execute(Plan);

                Result = new Image(Dims, true, true) { PixelSize = PixelSize };

                for (int z = 0; z < Dims.Z; z++)
                    Marshal.Copy(new IntPtr((long)Transformed + Dims.ElementsFFTSlice() * sizeof(float) * 2 * z), Result.GetHost(Intent.Write)[z], 0, (int)Dims.ElementsFFTSlice() * 2);

                lock (Image.FFT_CPU_Sync)
                {
                    FFTW.destroy_plan(Plan);
                    FFTW.free(Transformed);
                    FFTW.free(Original);
                }

                float Norm = 1f / Dims.Elements();
                Result.TransformValues(v => v * Norm);
            }
            else
            {
                Result = new Image(Dims, true, true) { PixelSize = PixelSize };

                IntPtr Original;
                IntPtr Transformed;
                IntPtr Plan;

                // FFTW is not thread-safe (except the execute function)
                lock (Image.FFT_CPU_Sync)
                {
                    Original = FFTW.alloc_real(Dims.ElementsSlice());
                    Transformed = FFTW.alloc_complex(Dims.ElementsFFTSlice());

                    Plan = FFTW.plan_dft_r2c(2, new int[] { Dims.Y, Dims.X }, Original, Transformed, PlannerFlags.Measure);
                }

                for (int z = 0; z < Dims.Z; z++)
                {
                    Marshal.Copy(GetHost(Intent.Read)[z], 0, Original, (int)Dims.ElementsSlice());

                    FFTW.execute(Plan);

                    Marshal.Copy(Transformed, Result.GetHost(Intent.Write)[z], 0, (int)Dims.ElementsFFTSlice() * 2);
                }

                lock (Image.FFT_CPU_Sync)
                {
                    FFTW.destroy_plan(Plan);
                    FFTW.free(Transformed);
                    FFTW.free(Original);
                }

                float Norm = 1f / Dims.ElementsSlice();
                Result.TransformValues(v => v * Norm);
            }

            Result.Parent = this;

            return Result;
        }

        public Image AsIFFT(bool isvolume = false, int plan = 0, bool normalize = false, bool preserveSelf = false, Image preserveBuffer = null)
        {
            if (IsHalf || !IsComplex || !IsFT)
                throw new Exception("Data format not supported.");

            int Plan = plan;
            if (Plan == 0)
                Plan = FFTPlanCache.GetIFFTPlan(isvolume ? Dims : Dims.Slice(), isvolume ? 1 : Dims.Z);

            Image Buffer = null;
            if (preserveSelf)
            {
                if (preserveBuffer == null)
                {
                    Buffer = GetCopyGPU();
                }    
                else
                {
                    Buffer = preserveBuffer;
                    GPU.CopyDeviceToDevice(GetDevice(Intent.Read), Buffer.GetDevice(Intent.Write), ElementsReal);
                }
            }

            Image IFFT = new Image(IntPtr.Zero, Dims, false, false, false) { PixelSize = PixelSize };
            GPU.IFFT(preserveSelf ? Buffer.GetDevice(Intent.Read) : GetDevice(Intent.Read), IFFT.GetDevice(Intent.Write), isvolume ? Dims : Dims.Slice(), isvolume ? 1 : (uint)Dims.Z, Plan, normalize);

            if (preserveSelf && preserveBuffer == null)
                Buffer.Dispose();

            IFFT.Parent = this;

            return IFFT;
        }

        public Image AsIFFT_CPU(bool isVolume)
        {
            Image Result = null;

            if (!IsFT || !IsComplex)
                throw new Exception("Data must be in FT, complex format");

            if (isVolume)
            {
                IntPtr Original;
                IntPtr Transformed;
                IntPtr Plan;

                // FFTW is not thread-safe (except the execute function)
                lock(Image.FFT_CPU_Sync)
                {
                    Original = FFTW.alloc_complex(Dims.ElementsFFT());
                    Transformed = FFTW.alloc_real(Dims.Elements());

                    Plan = FFTW.plan_dft_c2r(3, new int[] { Dims.Z, Dims.Y, Dims.X }, Original, Transformed, PlannerFlags.Measure);
                }

                for (int z = 0; z < Dims.Z; z++)
                    Marshal.Copy(GetHost(Intent.Read)[z], 0, new IntPtr((long)Original + Dims.ElementsFFTSlice() * sizeof(float) * 2 * z), (int)Dims.ElementsFFTSlice() * 2);

                FFTW.execute(Plan);

                Result = new Image(Dims) { PixelSize = PixelSize };

                for (int z = 0; z < Dims.Z; z++)
                    Marshal.Copy(new IntPtr((long)Transformed + Dims.ElementsSlice() * sizeof(float) * z), Result.GetHost(Intent.Write)[z], 0, (int)Dims.ElementsSlice());

                lock (Image.FFT_CPU_Sync)
                {
                    FFTW.destroy_plan(Plan);
                    FFTW.free(Transformed);
                    FFTW.free(Original);
                }
            }
            else
            {
                Result = new Image(Dims) { PixelSize = PixelSize };

                IntPtr Original;
                IntPtr Transformed;
                IntPtr Plan;

                // FFTW is not thread-safe (except the execute function)
                lock (Image.FFT_CPU_Sync)
                {
                    Original = FFTW.alloc_complex(Dims.ElementsFFT());
                    Transformed = FFTW.alloc_real(Dims.Elements());

                    Plan = FFTW.plan_dft_c2r(2, new int[] { Dims.Y, Dims.X }, Original, Transformed, PlannerFlags.Measure);
                }

                for (int z = 0; z < Dims.Z; z++)
                {
                    Marshal.Copy(GetHost(Intent.Read)[z], 0, Original, (int)Dims.ElementsFFTSlice() * 2);

                    FFTW.execute(Plan);

                    Marshal.Copy(Transformed, Result.GetHost(Intent.Write)[z], 0, (int)Dims.ElementsSlice());
                }

                lock (Image.FFT_CPU_Sync)
                {
                    FFTW.destroy_plan(Plan);
                    FFTW.free(Transformed);
                    FFTW.free(Original);
                }
            }

            Result.Parent = this;

            return Result;
        }

        public Image AsMultipleRegions(int3[] origins, int2 dimensions, bool zeropad = false)
        {
            Image Extracted = new Image(IntPtr.Zero, new int3(dimensions.X, dimensions.Y, origins.Length), false, IsComplex, IsHalf) { PixelSize = PixelSize };

            if (IsHalf)
                GPU.ExtractHalf(GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                Dims, new int3(dimensions),
                                Helper.ToInterleaved(origins),
                                (uint) origins.Length);
            else
                GPU.Extract(GetDevice(Intent.Read),
                            Extracted.GetDevice(Intent.Write),
                            Dims, new int3(dimensions),
                            Helper.ToInterleaved(origins),
                            zeropad,
                            (uint) origins.Length);

            Extracted.Parent = this;

            return Extracted;
        }

        public Image AsReducedAlongZ()
        {
            Image Reduced = new Image(IntPtr.Zero, new int3(Dims.X, Dims.Y, 1), IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };

            if (IsHalf)
                GPU.ReduceMeanHalf(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z, 1);
            else
                GPU.ReduceMean(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)ElementsSliceReal, (uint)Dims.Z, 1);

            Reduced.Parent = this;

            return Reduced;
        }

        public Image AsReducedAlongY()
        {
            Image Reduced = new Image(IntPtr.Zero, new int3(Dims.X, 1, Dims.Z), IsFT, IsComplex, IsHalf) { PixelSize = PixelSize };

            if (IsHalf)
                GPU.ReduceMeanHalf(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)(DimsEffective.X * (IsComplex ? 2 : 1)), (uint)Dims.Y, (uint)Dims.Z);
            else
                GPU.ReduceMean(GetDevice(Intent.Read), Reduced.GetDevice(Intent.Write), (uint)(DimsEffective.X * (IsComplex ? 2 : 1)), (uint)Dims.Y, (uint)Dims.Z);

            Reduced.Parent = this;

            return Reduced;
        }

        public Image AsPolar(uint innerradius = 0, uint exclusiveouterradius = 0)
        {
            if (IsHalf || IsComplex)
                throw new Exception("Cannot transform fp16 or complex image.");

            if (exclusiveouterradius == 0)
                exclusiveouterradius = (uint)Dims.X / 2;
            exclusiveouterradius = (uint)Math.Min(Dims.X / 2, (int)exclusiveouterradius);
            uint R = exclusiveouterradius - innerradius;

            Image Result;

            if (IsFT)
            {
                Result = new Image(IntPtr.Zero, new int3((int)R, Dims.Y, Dims.Z));
                GPU.Cart2PolarFFT(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), DimsSlice, innerradius, exclusiveouterradius, (uint) Dims.Z);
            }
            else
            {
                Result = new Image(IntPtr.Zero, new int3((int)R, Dims.Y * 2, Dims.Z));
                GPU.Cart2Polar(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), DimsSlice, innerradius, exclusiveouterradius, (uint)Dims.Z);
            }

            Result.Parent = this;

            return Result;
        }

        public Image AsAmplitudes()
        {
            if (IsHalf || !IsComplex)
                throw new Exception("Data type not supported.");

            Image Amplitudes = new Image(IntPtr.Zero, Dims, IsFT, false, false) { PixelSize = PixelSize };
            GPU.Amplitudes(GetDevice(Intent.Read), Amplitudes.GetDevice(Intent.Write), ElementsComplex);

            Amplitudes.Parent = this;

            return Amplitudes;
        }

        public Image AsReal()
        {
            if (!IsComplex)
                throw new Exception("Data must be complex.");

            //float[][] Real = new float[Dims.Z][];
            //float[][] Complex = GetHost(Intent.Read);
            //for (int z = 0; z < Real.Length; z++)
            //{
            //    float[] ComplexSlice = Complex[z];
            //    float[] RealSlice = new float[ComplexSlice.Length / 2];
            //    for (int i = 0; i < RealSlice.Length; i++)
            //        RealSlice[i] = ComplexSlice[i * 2];

            //    Real[z] = RealSlice;
            //}

            Image Result = new Image(IntPtr.Zero, Dims, IsFT, false, IsHalf) { PixelSize = PixelSize };
            GPU.Real(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), ElementsComplex);

            Result.Parent = this;

            return Result;
        }

        public Image AsImaginary()
        {
            if (!IsComplex)
                throw new Exception("Data must be complex.");

            //float[][] Imaginary = new float[Dims.Z][];
            //float[][] Complex = GetHost(Intent.Read);
            //for (int z = 0; z < Imaginary.Length; z++)
            //{
            //    float[] ComplexSlice = Complex[z];
            //    float[] ImaginarySlice = new float[ComplexSlice.Length / 2];
            //    for (int i = 0; i < ImaginarySlice.Length; i++)
            //        ImaginarySlice[i] = ComplexSlice[i * 2 + 1];

            //    Imaginary[z] = ImaginarySlice;
            //}

            Image Result = new Image(IntPtr.Zero, Dims, IsFT, false, IsHalf) { PixelSize = PixelSize };
            GPU.Imag(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), ElementsComplex);

            Result.Parent = this;

            return Result;
        }

        public Image AsScaledMassive(int2 newSliceDims, int planForw = 0, int planBack = 0)
        {
            int3 Scaled = new int3(newSliceDims.X, newSliceDims.Y, Dims.Z);
            Image Output = new Image(Scaled) { PixelSize = PixelSize * (float)Dims.X / newSliceDims.X };
            
            for (int z = 0; z < Dims.Z; z++)
            {
                GPU.Scale(GetDeviceSlice(z, Intent.Read),
                          Output.GetDeviceSlice(z, Intent.Write),
                          Dims.Slice(),
                          new int3(newSliceDims),
                          1,
                          planForw,
                          planBack,
                          IntPtr.Zero,
                          IntPtr.Zero);
            }

            Output.Parent = this;

            return Output;
        }

        public Image AsScaled(int2 newSliceDims, int planForw = 0, int planBack = 0)
        {
            int3 Scaled = new int3(newSliceDims.X, newSliceDims.Y, Dims.Z);
            Image Output = new Image(IntPtr.Zero, Scaled) { PixelSize = PixelSize * (float)Dims.X / newSliceDims.X };

            GPU.Scale(GetDevice(Intent.Read),
                      Output.GetDevice(Intent.Write),
                      new int3(DimsSlice),
                      new int3(newSliceDims),
                      (uint)Dims.Z,
                      planForw,
                      planBack,
                      IntPtr.Zero,
                      IntPtr.Zero);

            Output.Parent = this;

            return Output;
        }

        public Image AsScaled(int3 newDims, int planForw = 0, int planBack = 0)
        {
            Image Output = new Image(IntPtr.Zero, newDims) { PixelSize = PixelSize * (float)Dims.X / newDims.X };

            GPU.Scale(GetDevice(Intent.Read),
                      Output.GetDevice(Intent.Write),
                      new int3(Dims),
                      new int3(newDims),
                      1,
                      planForw,
                      planBack,
                      IntPtr.Zero,
                      IntPtr.Zero);

            Output.Parent = this;

            return Output;
        }

        public Image AsScaled_CPU(int2 newSliceDims)
        {
            Image OldFT = AsFFT_CPU(false);
            Image NewFT = OldFT.AsPadded_CPU(newSliceDims).AndDisposeParent();
            Image New = NewFT.AsIFFT_CPU(false).AndDisposeParent();

            //float Norm = 1f / Dims.Elements();
            //New.TransformValues(v => v * Norm);

            New.Parent = this;
            return New;
        }

        public Image AsScaledCTF(int3 newDims)
        {
            Image CTFComplex = this.AsComplex();
            Image PSF = CTFComplex.AsIFFT(true).AndDisposeParent();
            PSF = PSF.AsPadded(newDims, true).AndDisposeParent();

            Image ScaledCTF = PSF.AsFFT(true).AndDisposeParent().AsReal().AndDisposeParent();
            ScaledCTF.Multiply(1f / newDims.Elements());

            ScaledCTF.Parent = this;
            return ScaledCTF;
        }

        public Image AsShiftedVolume(float3 shift)
        {
            Image Result;

            if (IsComplex)
            {
                if (IsHalf)
                    throw new Exception("Cannot shift complex fp16 volume.");
                if (!IsFT)
                    throw new Exception("Volume must be in FFTW format");

                Result = new Image(IntPtr.Zero, Dims, true, true);

                GPU.ShiftStackFT(GetDevice(Intent.Read),
                                 Result.GetDevice(Intent.Write),
                                 Dims,
                                 Helper.ToInterleaved(new[] { shift }),
                                 1);
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Cannot shift fp16 volume.");

                Result = new Image(IntPtr.Zero, Dims);

                GPU.ShiftStack(GetDevice(Intent.Read),
                               Result.GetDevice(Intent.Write),
                               DimsEffective,
                               Helper.ToInterleaved(new[] { shift }),
                               1);
            }
            
            Result.Parent = this;
            Result.PixelSize = this.PixelSize;

            return Result;
        }

        public Image AsProjections(float3[] angles, int2 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, angles.Length), true, true) { PixelSize = PixelSize };

            GPU.ProjectForward(GetDevice(Intent.Read),
                               Projections.GetDevice(Intent.Write),
                               Dims,
                               dimsprojection,
                               Helper.ToInterleaved(angles),
                               supersample,
                               (uint)angles.Length);

            Projections.Parent = this;

            return Projections;
        }

        public Image AsProjections(float3[] angles, float3[] shifts, float[] globalweights, int2 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, angles.Length), true, true) { PixelSize = PixelSize };

            GPU.ProjectForwardShifted(GetDevice(Intent.Read),
                                      Projections.GetDevice(Intent.Write),
                                      Dims,
                                      dimsprojection,
                                      Helper.ToInterleaved(angles),
                                      Helper.ToInterleaved(shifts),
                                      globalweights,
                                      supersample,
                                      (uint)angles.Length);

            Projections.Parent = this;

            return Projections;
        }

        public Image AsProjections3D(float3[] angles, int3 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, dimsprojection.Z * angles.Length), true, true) { PixelSize = PixelSize };

            GPU.ProjectForward3D(GetDevice(Intent.Read),
                                 Projections.GetDevice(Intent.Write),
                                 Dims,
                                 dimsprojection,
                                 Helper.ToInterleaved(angles),
                                 supersample,
                                 (uint)angles.Length);

            Projections.Parent = this;

            return Projections;
        }

        public Image AsProjections3D(float3[] angles, float3[] shifts, float[] globalweights, int3 dimsprojection, float supersample)
        {
            if (Dims.X != Dims.Y || Dims.Y != Dims.Z)
                throw new Exception("Volume must be a cube.");

            Image Projections = new Image(IntPtr.Zero, new int3(dimsprojection.X, dimsprojection.Y, dimsprojection.Z * angles.Length), true, true) { PixelSize = PixelSize };

            GPU.ProjectForward3DShifted(GetDevice(Intent.Read),
                                        Projections.GetDevice(Intent.Write),
                                        Dims,
                                        dimsprojection,
                                        Helper.ToInterleaved(angles),
                                        Helper.ToInterleaved(shifts),
                                        globalweights,
                                        supersample,
                                        (uint)angles.Length);

            Projections.Parent = this;

            return Projections;
        }

        public Image AsAnisotropyCorrected(int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample)
        {
            Image Corrected = new Image(IntPtr.Zero, new int3(dimsscaled.X, dimsscaled.Y, Dims.Z)) { PixelSize = PixelSize };

            GPU.CorrectMagAnisotropy(GetDevice(Intent.Read),
                                     DimsSlice,
                                     Corrected.GetDevice(Intent.Write),
                                     dimsscaled,
                                     majorpixel,
                                     minorpixel,
                                     majorangle,
                                     supersample,
                                     (uint)Dims.Z);

            Corrected.Parent = this;

            return Corrected;
        }

        public Image AsDistanceMap(int maxDistance = -1, bool isVolume = true)
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("No other formats than fp32 non-FT realspace supported.");

            Image Distance = new Image(IntPtr.Zero, Dims) { PixelSize = PixelSize };

            if (isVolume)
                GPU.DistanceMap(GetDevice(Intent.Read), Distance.GetDevice(Intent.Write), Dims, maxDistance <= 0 ? Dims.X : maxDistance);
            else
                for (int z = 0; z < Dims.Z; z++)
                    GPU.DistanceMap(GetDeviceSlice(z, Intent.Read), Distance.GetDeviceSlice(z, Intent.Write), Dims.Slice(), maxDistance <= 0 ? Dims.X : maxDistance);

            Distance.Parent = this;

            return Distance;
        }

        public Image AsDistanceMapExact(int maxDistance, bool isVolume = true)
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("No other formats than fp32 non-FT realspace supported.");

            Image Result = new Image(Dims) { PixelSize = PixelSize };

            if (isVolume)
                GPU.DistanceMapExact(GetDevice(Intent.Read), Result.GetDevice(Intent.Write), Dims, maxDistance);
            else
                for (int z = 0; z < Dims.Z; z++)
                    GPU.DistanceMapExact(GetDeviceSlice(z, Intent.Read), Result.GetDeviceSlice(z, Intent.Write), Dims.Slice(), maxDistance);

            Result.Parent = this;

            return Result;
        }

        public Image AsDilatedMask(float distance, bool isVolume = true)
        {
            Image Convolved = AsConvolvedSphere(distance, false);
            Convolved.Binarize(0.5f);

            Convolved.Parent = this;
            Convolved.PixelSize = this.PixelSize;

            return Convolved;
        }

        public float3 AsCenterOfMass()
        {
            double VX = 0, VY = 0, VZ = 0;
            double Samples = 0;
            float[][] Values = GetHost(Intent.Read);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0, i = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++, i++)
                    {
                        float Val = Values[z][i];
                        VX += x * Val;
                        VY += y * Val;
                        VZ += z * Val;
                        Samples += Val;
                    }
                }
            }

            return new float3((float)(VX / Samples), (float)(VY / Samples), (float)(VZ / Samples));
        }

        public Image AsSymmetrized(string symmetry, int paddingFactor = 2)
        {
            if (Dims.Z <= 1)
                throw new Exception("Must be a volume.");

            Image Padded = AsPadded(Dims * paddingFactor);
            Padded.RemapToFT(true);
            Image PaddedFT = Padded.AsFFT(true);
            Padded.Dispose();

            GPU.SymmetrizeFT(PaddedFT.GetDevice(Intent.ReadWrite), PaddedFT.Dims, symmetry);

            Padded = PaddedFT.AsIFFT(true, -1, true).AndDisposeParent();
            Padded.RemapFromFT(true);

            Image Unpadded = Padded.AsPadded(Dims);
            Padded.Dispose();

            Unpadded.Parent = this;
            Unpadded.PixelSize = this.PixelSize;

            return Unpadded;
        }

        public Image AsSpectrumFlattened(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1, bool isTiltSeries = false)
        {
            Image FT = AsFFT(isVolume);
            Image FTAmp = FT.AsAmplitudes();

            int SpectrumLength = Math.Min(Dims.X, Dims.Z > 1 ? Math.Min(Dims.Y, Dims.Z) : Dims.Y) / 2;
            if (spectrumLength > 0)
                SpectrumLength = Math.Min(spectrumLength, SpectrumLength);

            float[] Spectrum = new float[SpectrumLength];
            float[] Samples = new float[SpectrumLength];

            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        if (x == 0 && y == 0 && z == 0)
                            continue;

                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        //Val *= Val;

                        Spectrum[(int)r] += WeightLow * Val;
                        Samples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Spectrum[(int)r + 1] += WeightHigh * Val;
                            Samples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        if (x == 0 && y == 0 && z == 0)
                            continue;

                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        if (r > nyquistLowpass)
                        {
                            FTAmpData[z][y * (Dims.X / 2 + 1) + x] = 0;
                            continue;
                        }

                        r *= SpectrumLength;
                        r = Math.Min(SpectrumLength - 2, r);

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = Spectrum[(int)r] * WeightLow + Spectrum[(int)r + 1] * WeightHigh;

                        FTAmpData[z][y * (Dims.X / 2 + 1) + x] = Val > 1e-10f ? 1f / (float)(Val) : 0;
                    }
                }
            }

            FT.Multiply(FTAmp);
            FTAmp.Dispose();

            Image IFT = (isVolume ? FT.AsIFFT(true) : FT.AsIFFT(false)).AndDisposeParent();

            IFT.Parent = this;
            IFT.PixelSize = this.PixelSize;

            return IFT;
        }

        public float[] AsAmplitudes1D(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1)
        {
            if (IsHalf)
                throw new Exception("Not implemented for half data.");
            //if (IsFT)
            //    throw new DimensionMismatchException();

            Image FT = IsFT ? this : AsFFT(isVolume);
            Image FTAmp = (IsFT && !IsComplex) ? this : FT.AsAmplitudes();
            FTAmp.FreeDevice();
            if (FT != this)
                FT.Dispose();

            int SpectrumLength = Math.Min(Dims.X, Dims.Z > 1 ? Math.Min(Dims.Y, Dims.Z) : Dims.Y) / 2;
            if (spectrumLength > 0)
                SpectrumLength = Math.Min(spectrumLength, SpectrumLength);

            float[] Spectrum = new float[SpectrumLength];
            float[] Samples = new float[SpectrumLength];

            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x]; ;
                        if (Math.Abs(Val) < 1e-10f)
                            continue;

                        Spectrum[(int)r] += WeightLow * Val;
                        Samples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Spectrum[(int)r + 1] += WeightHigh * Val;
                            Samples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

            if (FTAmp != this)
                FTAmp.Dispose();

            return Spectrum;
        }

        public float[] AsAmplitudeVariance1D(bool isVolume = true, float nyquistLowpass = 1f, int spectrumLength = -1)
        {
            if (IsHalf)
                throw new Exception("Not implemented for half data.");
            //if (IsFT)
            //    throw new DimensionMismatchException();

            Image FT = IsFT ? this : (isVolume ? AsFFT_CPU(true) : AsFFT(false));
            Image FTAmp = (IsFT && !IsComplex) ? this : FT.AsAmplitudes();
            FTAmp.FreeDevice();
            if (!IsFT)
                FT.Dispose();

            int SpectrumLength = Math.Min(Dims.X, Dims.Z > 1 ? Math.Min(Dims.Y, Dims.Z) : Dims.Y) / 2;
            if (spectrumLength > 0)
                SpectrumLength = Math.Min(spectrumLength, SpectrumLength);

            float[] Spectrum = new float[SpectrumLength];
            float[] Samples = new float[SpectrumLength];

            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        if (Math.Abs(Val) < 1e-10f)
                            continue;
                        //Val *= Val;

                        Spectrum[(int)r] += WeightLow * Val;
                        Samples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Spectrum[(int)r + 1] += WeightHigh * Val;
                            Samples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Spectrum[i] = Spectrum[i] / Math.Max(1e-5f, Samples[i]);

            float[] Variance = new float[SpectrumLength];
            float[] VarianceSamples = new float[SpectrumLength];

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        //if (r > nyquistLowpass)
                        //    continue;

                        r *= SpectrumLength;
                        if (r > SpectrumLength - 1)
                            continue;

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = FTAmpData[z][y * (Dims.X / 2 + 1) + x];
                        if (Math.Abs(Val) < 1e-10f)
                            continue;

                        float Mean = Spectrum[(int)r] * WeightLow + Spectrum[Math.Min(Spectrum.Length - 1, (int)r + 1)] * WeightHigh;
                        float Diff = Val - Mean;
                        Diff *= Diff;

                        Variance[(int)r] += WeightLow * Diff;
                        VarianceSamples[(int)r] += WeightLow;

                        if ((int)r < SpectrumLength - 1)
                        {
                            Variance[(int)r + 1] += WeightHigh * Diff;
                            VarianceSamples[(int)r + 1] += WeightHigh;
                        }
                    }
                }
            }

            for (int i = 0; i < Spectrum.Length; i++)
                Variance[i] = Variance[i] / Math.Max(1e-5f, VarianceSamples[i]);

            return Variance;
        }

        public Image AsSpectrumMultiplied(bool isVolume, float[] spectrumMultiplicators)
        {
            Image FT = AsFFT(isVolume);
            Image FTAmp = FT.AsAmplitudes();
            float[][] FTAmpData = FTAmp.GetHost(Intent.ReadWrite);

            int SpectrumLength = spectrumMultiplicators.Length;

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z < Dims.Z / 2 ? z : z - Dims.Z;
                float fz = (float)zz / (Dims.Z / 2);
                fz *= fz;
                if (!isVolume)
                    fz = 0;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y < Dims.Y / 2 ? y : y - Dims.Y;
                    float fy = (float)yy / (Dims.Y / 2);
                    fy *= fy;

                    for (int x = 0; x < Dims.X / 2 + 1; x++)
                    {
                        float fx = (float)x / (Dims.X / 2);
                        fx *= fx;

                        float r = (float)Math.Sqrt(fx + fy + fz);
                        if (r > 1)
                        {
                            FTAmpData[z][y * (Dims.X / 2 + 1) + x] = 0;
                            continue;
                        }

                        r *= SpectrumLength;
                        r = Math.Min(SpectrumLength - 2, r);

                        float WeightLow = 1f - (r - (int)r);
                        float WeightHigh = 1f - WeightLow;
                        float Val = spectrumMultiplicators[(int)r] * WeightLow + spectrumMultiplicators[(int)r + 1] * WeightHigh;

                        FTAmpData[z][y * (Dims.X / 2 + 1) + x] = Val;
                    }
                }
            }

            FT.Multiply(FTAmp);
            FTAmp.Dispose();

            Image IFT = FT.AsIFFT(isVolume, 0, true).AndDisposeParent();

            IFT.Parent = this;
            IFT.PixelSize = this.PixelSize;

            return IFT;
        }

        public Image AsConvolvedSphere(float radius, bool normalize = true)
        {
            Image Sphere = new Image(Dims);
            float[][] SphereData = Sphere.GetHost(Intent.Write);
            double SphereSum = 0;
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z - Dims.Z / 2;
                zz *= zz;
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y - Dims.Y / 2;
                    yy *= yy;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = x - Dims.X / 2;
                        xx *= xx;

                        float r = MathF.Sqrt(xx + yy + zz);
                        float v = 1f - Math.Max(0, Math.Min(1, r - radius));

                        SphereSum += v;
                        SphereData[z][y * Dims.X + x] = v;
                    }
                }
            }

            Sphere.RemapToFT(true);
            Image SphereFT = Sphere.AsFFT(true).AndDisposeParent();

            if (normalize)
                SphereFT.Multiply(1f / (float)SphereSum);

            Image ThisFT = AsFFT(true);

            ThisFT.MultiplyConj(SphereFT);
            SphereFT.Dispose();

            Image Convolved = ThisFT.AsIFFT(true, 0, true).AndDisposeParent();

            Convolved.Parent = this;
            Convolved.PixelSize = this.PixelSize;

            return Convolved;
        }

        public Image AsConvolvedGaussian(float sigma, bool normalize = true)
        {
            sigma = -1f / (sigma * sigma * 2);

            Image Gaussian = new Image(Dims);
            float[][] GaussianData = Gaussian.GetHost(Intent.Write);
            double GaussianSum = 0;
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z - Dims.Z / 2;
                zz *= zz;
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y - Dims.Y / 2;
                    yy *= yy;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = x - Dims.X / 2;
                        xx *= xx;

                        float R2 = xx + yy + zz;
                        double G = Math.Exp(R2 * sigma);
                        if (G < 1e-4)
                            continue;

                        GaussianSum += G;
                        GaussianData[z][y * Dims.X + x] = (float)G;
                    }
                }
            }

            Gaussian.RemapToFT(true);
            Image GaussianFT = Gaussian.AsFFT(true);
            Gaussian.Dispose();

            if (normalize)
                GaussianFT.Multiply(1f / (float)GaussianSum);

            Image ThisFT = AsFFT(true);

            ThisFT.MultiplyConj(GaussianFT);
            GaussianFT.Dispose();

            Image Convolved = ThisFT.AsIFFT(true, 0, true).AndDisposeParent();

            Convolved.Parent = this;
            Convolved.PixelSize = this.PixelSize;

            return Convolved;
        }

        public Image AsConvolvedRaisedCosine(float innerRadius, float falloff, bool normalize = true)
        {
            Image Cosine = new Image(Dims);
            float[][] CosineData = Cosine.GetHost(Intent.Write);
            double CosineSum = 0;
            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z - Dims.Z / 2;
                zz *= zz;
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y - Dims.Y / 2;
                    yy *= yy;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = x - Dims.X / 2;
                        xx *= xx;

                        float R = (float)Math.Sqrt(xx + yy + zz);
                        double C = Math.Cos(Math.Max(0, Math.Min(falloff, R - innerRadius)) / falloff * Math.PI) * 0.5 + 0.5;
                        
                        CosineSum += C;
                        CosineData[z][y * Dims.X + x] = (float)C;
                    }
                }
            }

            Cosine.RemapToFT(true);
            Image CosineFT = Cosine.AsFFT(true);
            Cosine.Dispose();

            if (normalize)
                CosineFT.Multiply(1f / (float)CosineSum);

            Image ThisFT = AsFFT(true);

            ThisFT.MultiplyConj(CosineFT);
            CosineFT.Dispose();

            Image Convolved = ThisFT.AsIFFT(true, 0, true).AndDisposeParent();

            Convolved.Parent = this;
            Convolved.PixelSize = this.PixelSize;

            return Convolved;
        }

        public Image AsFlippedX()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims) { PixelSize = PixelSize };

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        int xx = Dims.X - 1 - x;

                        FlippedData[z][y * Dims.X + x] = Data[z][y * Dims.X + xx];
                    }
                }
            }

            Flipped.Parent = this;

            return Flipped;
        }

        public Image AsFlippedY()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims) { PixelSize = PixelSize };

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = Dims.Y - 1 - y;
                    for (int x = 0; x < Dims.X; x++)
                    {
                        FlippedData[z][y * Dims.X + x] = Data[z][yy * Dims.X + x];
                    }
                }
            }

            Flipped.Parent = this;

            return Flipped;
        }

        public Image AsFlippedZ()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Flipped = new Image(Dims) { PixelSize = PixelSize };

            float[][] Data = GetHost(Intent.Read);
            float[][] FlippedData = Flipped.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = Dims.Z - 1 - z;
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        FlippedData[z][y * Dims.X + x] = Data[zz][y * Dims.X + x];
                    }
                }
            }

            Flipped.Parent = this;

            return Flipped;
        }

        public Image AsTransposed()
        {
            if (IsComplex || IsFT || IsHalf)
                throw new Exception("Format not supported.");

            Image Transposed = new Image(new int3(Dims.Y, Dims.X, Dims.Z)) { PixelSize = PixelSize };

            float[][] Data = GetHost(Intent.Read);
            float[][] TransposedData = Transposed.GetHost(Intent.Write);

            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        TransposedData[z][x * Dims.Y + y] = Data[z][y * Dims.X + x];
                    }
                }
            }

            Transposed.Parent = this;

            return Transposed;
        }

        public Image AsRotated3D(float3 angles, int supersample = 1)
        {
            Image Result;

            Projector P = new Projector(this, supersample, true, 3);
            Image ResultFT = P.Project(Dims, new[] { angles });
            P.Dispose();

            Result = ResultFT.AsIFFT(true).AndDisposeParent();
            Result.RemapFromFT(true);

            Result.Parent = this;
            Result.PixelSize = this.PixelSize;

            return Result;
        }

        public Image AsRotated90(Matrix3 r, bool symmetricCenter)
        {
            Matrix3 RInv = r.Transposed();

            int3 DimsNew = new int3((r * new float3(Dims)).Round());
            DimsNew.X = Math.Abs(DimsNew.X);
            DimsNew.Y = Math.Abs(DimsNew.Y);
            DimsNew.Z = Math.Abs(DimsNew.Z);

            Image Rotated = new Image(DimsNew);

            float[][] OriData = GetHost(Intent.Read);
            float[][] RotatedData = Rotated.GetHost(Intent.Write);

            float3 OriCenter = new float3(Dims / 2);
            float3 NewCenter = new float3(DimsNew / 2);

            if (symmetricCenter)
            {
                OriCenter -= 0.5f;
                NewCenter -= 0.5f;
            }

            Parallel.For(0, DimsNew.Z, z =>
            {
                float fz = z - NewCenter.Z;

                for (int y = 0; y < DimsNew.Y; y++)
                {
                    float fy = y - NewCenter.Y;

                    for (int x = 0; x < DimsNew.X; x++)
                    {
                        float fx = x - NewCenter.X;
                        float3 Pos = new float3(fx, fy, fz);
                        float3 OriPos = RInv * Pos + OriCenter;

                        int3 OriPosInt = new int3(OriPos + 0.5f);
                        if (OriPosInt.X < 0 || OriPosInt.X >= Dims.X ||
                            OriPosInt.Y < 0 || OriPosInt.Y >= Dims.Y ||
                            OriPosInt.Z < 0 || OriPosInt.Z >= Dims.Z)
                            continue;

                        RotatedData[z][y * DimsNew.X + x] = OriData[OriPosInt.Z][OriPosInt.Y * Dims.X + OriPosInt.X];
                    }
                }
            });

            Rotated.Parent = this;
            Rotated.PixelSize = this.PixelSize;

            return Rotated;
        }

        public Image AsRotated90X(int nrotations, bool symmetricCenter = false) => AsRotated90(Matrix3.RotateX(nrotations * 90 * Helper.ToRad), symmetricCenter);

        public Image AsRotated90Y(int nrotations, bool symmetricCenter = false) => AsRotated90(Matrix3.RotateY(nrotations * 90 * Helper.ToRad), symmetricCenter);

        public Image AsRotated90Z(int nrotations, bool symmetricCenter = false) => AsRotated90(Matrix3.RotateZ(nrotations * 90 * Helper.ToRad), symmetricCenter);

        public Image AsExpandedBinary(int expandDistance)
        {
            Image BinaryExpanded = AsDistanceMapExact(expandDistance).AndFreeParent();
            BinaryExpanded.Multiply(-1);
            BinaryExpanded.Binarize(-expandDistance + 1e-6f);

            BinaryExpanded.Parent = this;
            BinaryExpanded.PixelSize = this.PixelSize;

            return BinaryExpanded;
        }

        public Image AsExpandedSmooth(int expandDistance)
        {
            Image ExpandedSmooth = AsDistanceMapExact(expandDistance).AndFreeParent();
            ExpandedSmooth.Multiply((float)Math.PI / expandDistance);
            ExpandedSmooth.Cos();
            ExpandedSmooth.Add(1);
            ExpandedSmooth.Multiply(0.5f);

            ExpandedSmooth.Parent = this;
            ExpandedSmooth.PixelSize = this.PixelSize;

            return ExpandedSmooth;
        }

        public Image AsComplex()
        {
            if (IsComplex)
                throw new FormatException("Data must be real-valued");

            Image Result = new Image(Dims, IsFT, true) { PixelSize = PixelSize };
            Result.Fill(new float2(1, 0));
            Result.Multiply(this);

            Result.Parent = this;

            return Result;
        }

        public Image AsSliceXY(int z)
        {
            Image Slice = new Image(GetHost(Intent.Read)[z], new int3(Dims.X, Dims.Y, 1), IsFT, IsComplex) { PixelSize = PixelSize };
            Slice.Parent = this;

            return Slice;
        }

        public Image AsSliceXZ(int y)
        {
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;

            Image Slice = new Image(new int3(Width, Dims.Z, 1)) { PixelSize = PixelSize };
            float[] SliceData = Slice.GetHost(Intent.Write)[0];
            float[][] Data = GetHost(Intent.Read);
            for (int z = 0; z < Dims.Z; z++)
                for (int x = 0; x < Width; x++)
                    SliceData[z * Width + x] = Data[z][y * Width + x];

            Slice.Parent = this;

            return Slice;
        }

        public Image AsSliceYZ(int x)
        {
            int Width = IsFT ? Dims.X / 2 + 1 : Dims.X;

            Image Slice = new Image(new int3(Dims.Y, Dims.Z, 1)) { PixelSize = PixelSize };
            float[] SliceData = Slice.GetHost(Intent.Write)[0];
            float[][] Data = GetHost(Intent.Read);
            for (int z = 0; z < Dims.Z; z++)
                for (int y = 0; y < Dims.Y; y++)
                    SliceData[z * Dims.Y + y] = Data[z][y * Width + x];

            Slice.Parent = this;

            return Slice;
        }

        public Image AsHelicalSymmetrized(float twist, float rise, float maxz, float maxr)
        {
            Image Result = new Image(IntPtr.Zero, Dims);

            Image Prefiltered = this.GetCopyGPU();
            GPU.PrefilterForCubic(Prefiltered.GetDevice(Intent.ReadWrite), Dims);

            ulong[] Texture = new ulong[1];
            ulong[] Array = new ulong[1];
            GPU.CreateTexture3D(Prefiltered.GetDevice(Intent.Read), Dims, Texture, Array, true);
            Prefiltered.Dispose();

            GPU.HelicalSymmetrize(Texture[0], Result.GetDevice(Intent.Write), Dims, twist, rise, maxz, maxr);

            GPU.DestroyTexture(Texture[0], Array[0]);

            Result.Parent = this;
            Result.PixelSize = PixelSize;
            return Result;
        }

        #endregion

        #region In-place

        public void RemapToFT(bool isvolume = false)
        {
            if (!IsFT && IsComplex)
                throw new Exception("Complex remap only supported for FT layout.");

            int3 WorkDims = isvolume ? Dims : Dims.Slice();
            uint WorkBatch = isvolume ? 1 : (uint)Dims.Z;

            if (IsComplex)
                GPU.RemapToFTComplex(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            else
            {
                if (IsFT)
                    GPU.RemapToFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
                else
                    GPU.RemapFullToFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            }
        }

        public void RemapFromFT(bool isvolume = false)
        {
            if (!IsFT && IsComplex)
                throw new Exception("Complex remap only supported for FT layout.");

            int3 WorkDims = isvolume ? Dims : Dims.Slice();
            uint WorkBatch = isvolume ? 1 : (uint)Dims.Z;

            if (IsComplex)
                GPU.RemapFromFTComplex(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            else
            {
                if (IsFT)
                    GPU.RemapFromFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
                else
                    GPU.RemapFullFromFTFloat(GetDevice(Intent.Read), GetDevice(Intent.Write), WorkDims, WorkBatch);
            }
        }

        public void Min(float value)
        {
            GPU.MinScalar(GetDevice(Intent.Read), GetDevice(Intent.Write), value, (uint)ElementsReal);
        }

        public void Max(float value)
        {
            GPU.MaxScalar(GetDevice(Intent.Read), GetDevice(Intent.Write), value, (uint)ElementsReal);
        }

        public void Xray(float ndevs)
        {
            if (IsComplex || IsHalf)
                throw new Exception("Complex and half are not supported.");

            for (int i = 0; i < Dims.Z; i++)
                GPU.Xray(new IntPtr((long)GetDevice(Intent.Read) + DimsEffective.ElementsSlice() * i * sizeof (float)),
                         new IntPtr((long)GetDevice(Intent.Write) + DimsEffective.ElementsSlice() * i * sizeof(float)),
                         ndevs,
                         new int2(DimsEffective),
                         1);
        }

        public void Fill(float val)
        {
            GPU.ValueFill(GetDevice(Intent.Write), ElementsReal, val);
        }

        public void Fill(float2 val)
        {
            GPU.ValueFillComplex(GetDevice(Intent.Write), ElementsComplex, val);
        }

        public void Sign()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");

            GPU.Sign(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Sqrt()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");
            if (IsComplex)
                throw new Exception("Does not work for complex data.");

            GPU.Sqrt(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Cos()
        {
            if (IsHalf || IsComplex)
                throw new Exception("Does not work for fp16 or complex.");

            GPU.Cos(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Sin()
        {
            if (IsHalf || IsComplex)
                throw new Exception("Does not work for fp16 or complex.");

            GPU.Sin(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void Abs()
        {
            if (IsHalf)
                throw new Exception("Does not work for fp16.");

            GPU.Abs(GetDevice(Intent.Read), GetDevice(Intent.Write), ElementsReal);
        }

        public void MaskSpherically(float diameter, float softEdge, bool isVolume, bool isDecentered = false)
        {
            GPU.SphereMask(GetDevice(Intent.Read),
                           GetDevice(Intent.Write),
                           isVolume ? Dims : Dims.Slice(),
                           diameter / 2,
                           softEdge,
                           isDecentered,
                           isVolume ? 1 : (uint)Dims.Z);
        }

        public void MaskRectangularly(int3 region, float softEdge, bool isVolume)
        {
            if (softEdge <= 0)
            {
                int3 CenterGlobal = Dims / 2;
                int3 CenterRegion = region / 2;

                TransformValues((x, y, z, v) =>
                {
                    int3 Pos = new int3(x, y, z) - CenterGlobal + CenterRegion;
                    bool Outside = Math.Min(Pos.X, Math.Min(Pos.Y, Pos.Z)) < 0 ||
                                   Pos.X >= region.X ||
                                   Pos.Y >= region.Y ||
                                   Pos.Z >= region.Z;

                    return v * (Outside ? 0 : 1);
                });
            }
            else
            {
                int3 BoxDim2 = Dims;
                int3 Margin = (Dims - region) / 2;
                float[][] Data = GetHost(Intent.ReadWrite);

                for (int z = 0; z < BoxDim2.Z; z++)
                {
                    float zz = Math.Max(Margin.Z - z, z - (Margin.Z + region.Z - 1)) / softEdge;
                    zz = Math.Max(0, Math.Min(1, zz));

                    for (int y = 0; y < BoxDim2.Y; y++)
                    {
                        float yy = Math.Max(Margin.Y - y, y - (Margin.Y + region.Y - 1)) / softEdge;
                        yy = Math.Max(0, Math.Min(1, yy));

                        for (int x = 0; x < BoxDim2.X; x++)
                        {
                            float xx = Math.Max(Margin.X - x, x - (Margin.X + region.X - 1)) / softEdge;
                            xx = Math.Max(0, Math.Min(1, xx));

                            float r = Math.Min(1, (float)Math.Sqrt(zz * zz + yy * yy + xx * xx));
                            float v = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                            Data[z][y * Dims.X + x] *= v;
                        }
                    }
                }
            }
        }

        private void Add(Image summands, uint elements, uint batch)
        {
            if (ElementsReal != elements * batch ||
                summands.ElementsReal != elements ||
                //IsFT != summands.IsFT ||
                IsComplex != summands.IsComplex)
                throw new DimensionMismatchException();

            if (IsHalf && summands.IsHalf)
            {
                GPU.AddToSlicesHalf(GetDevice(Intent.Read), summands.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else if (!IsHalf && !summands.IsHalf)
            {
                GPU.AddToSlices(GetDevice(Intent.Read), summands.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else
            {
                Image ThisSingle = AsSingle();
                Image SummandsSingle = summands.AsSingle();

                GPU.AddToSlices(ThisSingle.GetDevice(Intent.Read), SummandsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                if (IsHalf)
                    GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                else
                    GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                ThisSingle.Dispose();
                SummandsSingle.Dispose();
            }
        }

        public void Add(Image summands)
        {
            Add(summands, (uint) ElementsReal, 1);
        }

        public void AddToSlices(Image summands)
        {
            Add(summands, (uint) ElementsSliceReal, (uint) Dims.Z);
        }

        public void AddToLines(Image summands)
        {
            Add(summands, (uint) ElementsLineReal, (uint) (Dims.Y * Dims.Z));
        }

        public void Add(float scalar)
        {
            GPU.AddScalar(GetDevice(Intent.Read), scalar, GetDevice(Intent.Write), ElementsReal);
        }

        private void Subtract(Image subtrahends, uint elements, uint batch)
        {
            if (ElementsReal != elements * batch ||
                subtrahends.ElementsReal != elements ||
                IsFT != subtrahends.IsFT ||
                IsComplex != subtrahends.IsComplex)
                throw new DimensionMismatchException();

            if (IsHalf && subtrahends.IsHalf)
            {
                GPU.SubtractFromSlicesHalf(GetDevice(Intent.Read), subtrahends.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else if (!IsHalf && !subtrahends.IsHalf)
            {
                GPU.SubtractFromSlices(GetDevice(Intent.Read), subtrahends.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
            else
            {
                Image ThisSingle = AsSingle();
                Image SubtrahendsSingle = subtrahends.AsSingle();

                GPU.SubtractFromSlices(ThisSingle.GetDevice(Intent.Read), SubtrahendsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                if (IsHalf)
                    GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                else
                    GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                ThisSingle.Dispose();
                SubtrahendsSingle.Dispose();
            }
        }

        public void Subtract(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsReal, 1);
        }

        public void SubtractFromSlices(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsSliceReal, (uint) Dims.Z);
        }

        public void SubtractFromLines(Image subtrahends)
        {
            Subtract(subtrahends, (uint) ElementsLineReal, (uint) (Dims.Y * Dims.Z));
        }

        public void Multiply(float multiplicator)
        {
            GPU.MultiplyByScalar(GetDevice(Intent.Read),
                                 GetDevice(Intent.Write),
                                 multiplicator,
                                 ElementsReal);
        }

        public void Multiply(float[] scalarMultiplicators)
        {
            if (scalarMultiplicators.Length != Dims.Z)
                throw new DimensionMismatchException("Number of scalar multiplicators must equal number of slices.");

            IntPtr d_multiplicators = GPU.MallocDeviceFromHost(scalarMultiplicators, scalarMultiplicators.Length);

            GPU.MultiplyByScalars(GetDevice(Intent.Read),
                                  GetDevice(Intent.Write),
                                  d_multiplicators,
                                  ElementsSliceReal,
                                  (uint)Dims.Z);

            GPU.FreeDevice(d_multiplicators);
        }

        private void Multiply(Image multiplicators, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                multiplicators.ElementsComplex != elements ||
                //IsFT != multiplicators.IsFT ||
                (multiplicators.IsComplex && !IsComplex))
                throw new DimensionMismatchException();

            if (!IsComplex)
            {
                if (IsHalf && multiplicators.IsHalf)
                {
                    GPU.MultiplySlicesHalf(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else if (!IsHalf && !multiplicators.IsHalf)
                {
                    GPU.MultiplySlices(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else
                {
                    Image ThisSingle = AsSingle();
                    Image MultiplicatorsSingle = multiplicators.AsSingle();

                    GPU.MultiplySlices(ThisSingle.GetDevice(Intent.Read), MultiplicatorsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                    if (IsHalf)
                        GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                    else
                        GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                    ThisSingle.Dispose();
                    MultiplicatorsSingle.Dispose();
                }
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Complex multiplication not supported for fp16.");

                if (!multiplicators.IsComplex)
                    GPU.MultiplyComplexSlicesByScalar(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                else
                    GPU.MultiplyComplexSlicesByComplex(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
        }

        public void Multiply(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsComplex, 1);
        }

        public void MultiplySlices(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsSliceComplex, (uint) Dims.Z);
        }

        public void MultiplyLines(Image multiplicators)
        {
            Multiply(multiplicators, (uint) ElementsLineComplex, (uint) (Dims.Y * Dims.Z));
        }
        
        private void MultiplyConj(Image multiplicators, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                multiplicators.ElementsComplex != elements ||
                !multiplicators.IsComplex || 
                !IsComplex)
                throw new DimensionMismatchException();
            
            if (IsHalf)
                throw new Exception("Complex multiplication not supported for fp16.");

            GPU.MultiplyComplexSlicesByComplexConj(GetDevice(Intent.Read), multiplicators.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
        }

        public void MultiplyConj(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsComplex, 1);
        }

        public void MultiplyConjSlices(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsSliceComplex, (uint)Dims.Z);
        }

        public void MultiplyConjLines(Image multiplicators)
        {
            MultiplyConj(multiplicators, (uint)ElementsLineComplex, (uint)(Dims.Y * Dims.Z));
        }

        private void Divide(Image divisors, uint elements, uint batch)
        {
            if (ElementsComplex != elements * batch ||
                divisors.ElementsComplex != elements ||
                //IsFT != divisors.IsFT ||
                divisors.IsComplex)
                throw new DimensionMismatchException();

            if (!IsComplex)
            {
                if (!IsHalf && !divisors.IsHalf)
                {
                    GPU.DivideSlices(GetDevice(Intent.Read), divisors.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
                }
                else
                {
                    Image ThisSingle = AsSingle();
                    Image DivisorsSingle = divisors.AsSingle();

                    GPU.DivideSlices(ThisSingle.GetDevice(Intent.Read), DivisorsSingle.GetDevice(Intent.Read), ThisSingle.GetDevice(Intent.Write), elements, batch);

                    if (IsHalf)
                        GPU.HalfToSingle(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);
                    else
                        GPU.CopyDeviceToDevice(ThisSingle.GetDevice(Intent.Read), GetDevice(Intent.Write), elements * batch);

                    ThisSingle.Dispose();
                    DivisorsSingle.Dispose();
                }
            }
            else
            {
                if (IsHalf)
                    throw new Exception("Complex division not supported for fp16.");
                GPU.DivideComplexSlicesByScalar(GetDevice(Intent.Read), divisors.GetDevice(Intent.Read), GetDevice(Intent.Write), elements, batch);
            }
        }

        public void Divide(Image divisors)
        {
            Divide(divisors, (uint)ElementsComplex, 1);
        }

        public void DivideSlices(Image divisors)
        {
            Divide(divisors, (uint)ElementsSliceComplex, (uint)Dims.Z);
        }

        public void DivideLines(Image divisors)
        {
            Divide(divisors, (uint)ElementsLineComplex, (uint)(Dims.Y * Dims.Z));
        }

        public void ShiftSlices(float3[] shifts)
        {
            if (IsComplex)
            {
                if (IsHalf)
                    throw new Exception("Cannot shift complex half image.");
                if (!IsFT)
                    throw new Exception("Image must be in FFTW format");

                GPU.ShiftStackFT(GetDevice(Intent.Read),
                                 GetDevice(Intent.Write),
                                 Dims.Slice(),
                                 Helper.ToInterleaved(shifts),
                                 (uint)Dims.Z);
            }
            else
            {
                IntPtr Data;
                if (!IsHalf)
                    Data = GetDevice(Intent.ReadWrite);
                else
                {
                    Data = GPU.MallocDevice(ElementsReal);
                    GPU.OnMemoryChanged();
                    GPU.HalfToSingle(GetDevice(Intent.Read), Data, ElementsReal);
                }

                GPU.ShiftStack(Data,
                               Data,
                               DimsEffective.Slice(),
                               Helper.ToInterleaved(shifts),
                               (uint)Dims.Z);

                if (IsHalf)
                {
                    GPU.SingleToHalf(Data, GetDevice(Intent.Write), ElementsReal);
                    GPU.FreeDevice(Data);
                    GPU.OnMemoryChanged();
                }
            }
        }

        public void ShiftSlicesMassive(float3[] shifts)
        {
            if (IsComplex)
                throw new Exception("Cannot shift complex image.");

            IntPtr Data;
            if (!IsHalf)
                Data = GetDevice(Intent.ReadWrite);
            else
            {
                Data = GPU.MallocDevice(ElementsReal);
                GPU.OnMemoryChanged();
                GPU.HalfToSingle(GetDevice(Intent.Read), Data, ElementsReal);
            }

            GPU.ShiftStackMassive(Data,
                                  Data,
                                  DimsEffective.Slice(),
                                  Helper.ToInterleaved(shifts),
                                  (uint)Dims.Z);

            if (IsHalf)
            {
                GPU.SingleToHalf(Data, GetDevice(Intent.Write), ElementsReal);
                GPU.FreeDevice(Data);
                GPU.OnMemoryChanged();
            }
        }

        public void Bandpass(float nyquistLow, float nyquistHigh, bool isVolume, float nyquistsoftedge = 0, int batch = 1)
        {
            if (IsHalf)
                throw new Exception("Bandpass only works on fp32 data");
            if (!isVolume && batch != 1)
                throw new Exception("Batch can only be manually specified for volumetric data; for 2D it is the number of slices");

            if (IsComplex && IsFT)
                GPU.FourierBandpass(GetDevice(Intent.Read), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, nyquistsoftedge, isVolume ? (uint)batch : (uint)Dims.Z);
            else if (!IsComplex && !IsFT)
                GPU.Bandpass(GetDevice(Intent.Read), GetDevice(Intent.Write), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, nyquistsoftedge, isVolume ? 1 : (uint)Dims.Z);
            else
                throw new Exception("Bandpass needs either real-space data in non-FFTW format, or Fourier-space data in FFTW format");
        }

        public void Bandpass_CPU(float nyquistLow, float nyquistHigh, bool isVolume, float nyquistsoftedge = 0)
        {
            if (IsFT != IsComplex)
                throw new Exception("Data must be either FT & complex, or the opposite");

            int MaxR = Math.Max(Dims.X, Dims.Y) / 2;
            float[] Filter1D = Helper.ArrayOfFunction(r =>
            {
                float R = r / (float)(MaxR - 1);

                float Filter = 1;
                if (nyquistsoftedge > 0)
                {
                    float EdgeLow = (float)Math.Cos(Math.Min(1, Math.Max(0, nyquistLow - R) / nyquistsoftedge) * Math.PI) * 0.5f + 0.5f;
                    float EdgeHigh = (float)Math.Cos(Math.Min(1, Math.Max(0, (R - nyquistHigh) / nyquistsoftedge)) * Math.PI) * 0.5f + 0.5f;
                    Filter = EdgeLow * EdgeHigh;
                }
                else
                {
                    Filter = (R >= nyquistLow && R <= nyquistHigh) ? 1 : 0;
                }

                return Filter;
            }, MaxR);

            Image FT = (IsFT && IsComplex) ? this : AsFFT_CPU(isVolume);
            float[][] FTData = FT.GetHost(Intent.ReadWrite);

            if (isVolume)
                Helper.ForEachElementFT(Dims, (x, y, z, xx, yy, zz) =>
                {
                    float nx = xx / (float)(Dims.X / 2 - 1);
                    float ny = yy / (float)(Dims.Y / 2 - 1);
                    float nz = zz / (float)(Dims.Z / 2 - 1);

                    float r = (float)Math.Sqrt(nx * nx + ny * ny + nz * nz) * (Filter1D.Length - 1);
                    int ir0 = (int)Math.Floor(r);
                    int ir1 = ir0 + 1;

                    float Filter0 = ir0 >= Filter1D.Length ? 0f : Filter1D[ir0];
                    float Filter1 = ir1 >= Filter1D.Length ? 0f : Filter1D[ir1];

                    float Filter = MathHelper.Lerp(Filter0, Filter1, r - ir0);

                    int id = y * (Dims.X / 2 + 1) + x;
                    FTData[z][id * 2] *= Filter;
                    FTData[z][id * 2 + 1] *= Filter;
                });
            else
                Helper.ForEachElementFT(new int2(Dims), (x, y, xx, yy) =>
                {
                    float nx = xx / (float)(Dims.X / 2 - 1);
                    float ny = yy / (float)(Dims.Y / 2 - 1);

                    float r = (float)Math.Sqrt(nx * nx + ny * ny) * (Filter1D.Length - 1);
                    int ir0 = (int)Math.Floor(r);
                    int ir1 = ir0 + 1;

                    float Filter0 = ir0 >= Filter1D.Length ? 0f : Filter1D[ir0];
                    float Filter1 = ir1 >= Filter1D.Length ? 0f : Filter1D[ir1];

                    float Filter = MathHelper.Lerp(Filter0, Filter1, r - ir0);

                    int id = (y * (Dims.X / 2 + 1) + x) * 2;
                    for (int z = 0; z < Dims.Z; z++)
                    {
                        FTData[z][id] *= Filter;
                        FTData[z][id + 1] *= Filter;
                    }
                });

            if (!IsFT && !IsComplex)
            {
                Image Filtered = FT.AsIFFT_CPU(isVolume).AndDisposeParent();

                for (int z = 0; z < Dims.Z; z++)
                    Array.Copy(Filtered.GetHost(Intent.Read)[z], GetHost(Intent.Write)[z], ElementsSliceReal);

                Filtered.Dispose();
            }
        }

        public void BandpassGauss(float nyquistLow, float nyquistHigh, bool isVolume, float nyquistsigma, int batch = 1)
        {
            if (IsHalf)
                throw new Exception("Bandpass only works on fp32 data");
            if (!isVolume && batch != 1)
                throw new Exception("Batch can only be manually specified for volumetric data; for 2D it is the number of slices");

            if (IsComplex && IsFT)
                GPU.FourierBandpassGauss(GetDevice(Intent.Read), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, nyquistsigma, isVolume ? (uint)1 : (uint)Dims.Z);
            else if (!IsComplex && !IsFT)
                GPU.BandpassGauss(GetDevice(Intent.Read), GetDevice(Intent.Write), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, nyquistsigma, isVolume ? 1 : (uint)Dims.Z);
            else
                throw new Exception("Bandpass needs either real-space data in non-FFTW format, or Fourier-space data in FFTW format");
        }

        public void BandpassButter(float nyquistLow, float nyquistHigh, bool isVolume, int order = 8, int batch = 1)
        {
            if (IsHalf)
                throw new Exception("Bandpass only works on fp32 data");
            if (!isVolume && batch != 1)
                throw new Exception("Batch can only be manually specified for volumetric data; for 2D it is the number of slices");

            if (IsComplex && IsFT)
                GPU.FourierBandpassButter(GetDevice(Intent.Read), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, order, isVolume ? (uint)1 : (uint)Dims.Z);
            else if (!IsComplex && !IsFT)
                GPU.BandpassButter(GetDevice(Intent.Read), GetDevice(Intent.Write), isVolume ? Dims : Dims.Slice(), nyquistLow, nyquistHigh, order, isVolume ? 1 : (uint)Dims.Z);
            else
                throw new Exception("Bandpass needs either real-space data in non-FFTW format, or Fourier-space data in FFTW format");
        }

        public void Binarize(float threshold)
        {
            foreach (var slice in GetHost(Intent.ReadWrite))
                for (int i = 0; i < slice.Length; i++)
                    slice[i] = slice[i] >= threshold ? 1 : 0;
        }

        public void SubtractMeanGrid(int2 gridDims)
        {
            //if (Dims.Z > 1)
            //    throw new Exception("Does not work for volumes or stacks.");

            foreach (var MicData in GetHost(Intent.ReadWrite))
                if (gridDims.Elements() <= 1)
                    MathHelper.FitAndSubtractPlane(MicData, DimsSlice);
                else
                    MathHelper.FitAndSubtractGrid(MicData, DimsSlice, gridDims);
        }

        public void Taper(float distance, bool isVolume = false)
        {
            if (!isVolume)
            {
                Image MaskTaper = new Image(Dims.Slice());
                MaskTaper.TransformValues((x, y, z, v) =>
                {
                    float dx = 0, dy = 0;

                    if (x < distance)
                        dx = distance - x;
                    else if (x > Dims.X - 1 - distance)
                        dx = x - (Dims.X - 1 - distance);

                    if (y < distance)
                        dy = distance - y;
                    else if (y > Dims.Y - 1 - distance)
                        dy = y - (Dims.Y - 1 - distance);

                    float R = (float)Math.Sqrt(dx * dx + dy * dy) / distance;

                    return Math.Max(0.01f, (float)Math.Cos(Math.Max(0, Math.Min(1, R)) * Math.PI) * 0.5f + 0.5f);
                });

                MultiplySlices(MaskTaper);
                MaskTaper.Dispose();
            }
            else
            {
                Image MaskTaper = new Image(Dims);
                MaskTaper.TransformValues((x, y, z, v) =>
                {
                    float dx = 0, dy = 0, dz = 0;

                    if (x < distance)
                        dx = distance - x;
                    else if (x > Dims.X - 1 - distance)
                        dx = x - (Dims.X - 1 - distance);

                    if (y < distance)
                        dy = distance - y;
                    else if (y > Dims.Y - 1 - distance)
                        dy = y - (Dims.Y - 1 - distance);

                    if (z < distance)
                        dz = distance - z;
                    else if (z > Dims.Z - 1 - distance)
                        dz = z - (Dims.Z - 1 - distance);

                    float R = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz) / distance;

                    return Math.Max(0, (float)Math.Cos(Math.Max(0, Math.Min(1, R)) * Math.PI) * 0.5f + 0.5f);
                });

                Multiply(MaskTaper);
                MaskTaper.Dispose();
            }
        }
        
        public void Normalize(bool isVolume = false)
        {
            if (IsHalf || IsComplex)
                throw new Exception("Wrong format, only real-valued input supported.");

            GPU.Normalize(GetDevice(Intent.Read),
                          GetDevice(Intent.Write),
                          (uint)(isVolume ? ElementsReal : ElementsSliceReal),
                          (uint)(isVolume ? 1 : Dims.Z));
        }

        public void Normalize(Image mask, bool isVolume = false)
        {
            if (IsHalf || IsComplex)
                throw new Exception("Wrong format, only real-valued input supported.");

            GPU.NormalizeMasked(GetDevice(Intent.Read),
                                GetDevice(Intent.Write),
                                mask.GetDevice(Intent.Read),
                                (uint)(isVolume ? ElementsReal : ElementsSliceReal),
                                (uint)(isVolume ? 1 : Dims.Z));
        }

        public void NormalizeWithinMask(Image mask, bool subtractMean)
        {
            float[][] ThisData = GetHost(Intent.ReadWrite);
            float[][] MaskData = mask.GetHost(Intent.Read);

            List<float> UnderMask = new();
            for (int z = 0; z < ThisData.Length; z++)
                for (int i = 0; i < ThisData[z].Length; i++)
                    if (MaskData[z][i] > 0)
                        UnderMask.Add(ThisData[z][i]);

            float2 MeanStd = MathHelper.MeanAndStd(UnderMask);

            if (MeanStd.Y > 0)
                for (int z = 0; z < ThisData.Length; z++)
                    for (int i = 0; i < ThisData[z].Length; i++)
                        if (MaskData[z][i] > 0)
                        {
                            if (subtractMean)
                                ThisData[z][i] = (ThisData[z][i] - MeanStd.X) / MeanStd.Y;
                            else
                                ThisData[z][i] = (ThisData[z][i] - MeanStd.X) / MeanStd.Y + MeanStd.X;
                        }
        }

        public void Symmetrize(string sym)
        {
            Symmetry Sym = new Symmetry(sym);
            Matrix3[] Rotations = Sym.GetRotationMatrices();

            RemapToFT(true);

            Image FT = AsFFT(true);
            FT.FreeDevice();
            float[][] FTData = FT.GetHost(Intent.Read);
            Image FTSym = new Image(Dims, true, true);
            float[][] FTSymData = FTSym.GetHost(Intent.Write);

            int DimX = Dims.X / 2 + 1;
            float R2 = Dims.X * Dims.X / 4f;

            for (int z = 0; z < Dims.Z; z++)
            {
                int zz = z <= Dims.Z / 2 ? z : z - Dims.Z;

                for (int y = 0; y < Dims.Y; y++)
                {
                    int yy = y <= Dims.Y / 2 ? y : y - Dims.Y;

                    for (int x = 0; x < DimX; x++)
                    {
                        int xx = x;
                        float3 PosCentered = new float3(xx, yy, zz);
                        if (PosCentered.LengthSq() >= R2)
                            continue;

                        float2 VSum = new float2(0, 0);

                        foreach (var rotation in Rotations)
                        {
                            float3 PosRotated = rotation * PosCentered;
                            bool IsFlipped = false;
                            if (PosRotated.X < 0)
                            {
                                PosRotated *= -1;
                                IsFlipped = true;
                            }

                            int X0 = (int)Math.Floor(PosRotated.X);
                            int X1 = X0 + 1;
                            float XInterp = PosRotated.X - X0;
                            int Y0 = (int)Math.Floor(PosRotated.Y);
                            int Y1 = Y0 + 1;
                            float YInterp = PosRotated.Y - Y0;
                            int Z0 = (int)Math.Floor(PosRotated.Z);
                            int Z1 = Z0 + 1;
                            float ZInterp = PosRotated.Z - Z0;

                            X0 = Math.Max(0, Math.Min(DimX - 1, X0));
                            X1 = Math.Max(0, Math.Min(DimX - 1, X1));
                            Y0 = Math.Max(0, Math.Min(Dims.Y - 1, Y0 >= 0 ? Y0 : Y0 + Dims.Y));
                            Y1 = Math.Max(0, Math.Min(Dims.Y - 1, Y1 >= 0 ? Y1 : Y1 + Dims.Y));
                            Z0 = Math.Max(0, Math.Min(Dims.Z - 1, Z0 >= 0 ? Z0 : Z0 + Dims.Z));
                            Z1 = Math.Max(0, Math.Min(Dims.Z - 1, Z1 >= 0 ? Z1 : Z1 + Dims.Z));

                            {
                                float v000 = FTData[Z0][(Y0 * DimX + X0) * 2 + 0];
                                float v001 = FTData[Z0][(Y0 * DimX + X1) * 2 + 0];
                                float v010 = FTData[Z0][(Y1 * DimX + X0) * 2 + 0];
                                float v011 = FTData[Z0][(Y1 * DimX + X1) * 2 + 0];

                                float v100 = FTData[Z1][(Y0 * DimX + X0) * 2 + 0];
                                float v101 = FTData[Z1][(Y0 * DimX + X1) * 2 + 0];
                                float v110 = FTData[Z1][(Y1 * DimX + X0) * 2 + 0];
                                float v111 = FTData[Z1][(Y1 * DimX + X1) * 2 + 0];

                                float v00 = MathHelper.Lerp(v000, v001, XInterp);
                                float v01 = MathHelper.Lerp(v010, v011, XInterp);
                                float v10 = MathHelper.Lerp(v100, v101, XInterp);
                                float v11 = MathHelper.Lerp(v110, v111, XInterp);

                                float v0 = MathHelper.Lerp(v00, v01, YInterp);
                                float v1 = MathHelper.Lerp(v10, v11, YInterp);

                                float v = MathHelper.Lerp(v0, v1, ZInterp);

                                VSum.X += v;
                            }

                            {
                                float v000 = FTData[Z0][(Y0 * DimX + X0) * 2 + 1];
                                float v001 = FTData[Z0][(Y0 * DimX + X1) * 2 + 1];
                                float v010 = FTData[Z0][(Y1 * DimX + X0) * 2 + 1];
                                float v011 = FTData[Z0][(Y1 * DimX + X1) * 2 + 1];

                                float v100 = FTData[Z1][(Y0 * DimX + X0) * 2 + 1];
                                float v101 = FTData[Z1][(Y0 * DimX + X1) * 2 + 1];
                                float v110 = FTData[Z1][(Y1 * DimX + X0) * 2 + 1];
                                float v111 = FTData[Z1][(Y1 * DimX + X1) * 2 + 1];

                                float v00 = MathHelper.Lerp(v000, v001, XInterp);
                                float v01 = MathHelper.Lerp(v010, v011, XInterp);
                                float v10 = MathHelper.Lerp(v100, v101, XInterp);
                                float v11 = MathHelper.Lerp(v110, v111, XInterp);

                                float v0 = MathHelper.Lerp(v00, v01, YInterp);
                                float v1 = MathHelper.Lerp(v10, v11, YInterp);

                                float v = MathHelper.Lerp(v0, v1, ZInterp);

                                VSum.Y += IsFlipped ? -v : v;
                            }
                        }

                        FTSymData[z][(y * DimX + x) * 2 + 0] = VSum.X / Rotations.Length;
                        FTSymData[z][(y * DimX + x) * 2 + 1] = VSum.Y / Rotations.Length;
                    }
                }
            }

            //FT.IsSameAs(FTSym, 1e-5f);

            FT.Dispose();

            GPU.IFFT(FTSym.GetDevice(Intent.Read),
                     this.GetDevice(Intent.Write),
                     Dims,
                     1,
                     -1,
                     true);
            FTSym.Dispose();

            RemapFromFT(true);
            FreeDevice();
        }

        public void InsertValues(Image values, int3 position, bool replace)
        {
            int3 Start = new int3(Math.Max(Math.Min(Dims.X - 1, position.X), 0),
                                  Math.Max(Math.Min(Dims.X - 1, position.Y), 0),
                                  Math.Max(Math.Min(Dims.Z - 1, position.Z), 0));
            int3 End = new int3(Math.Max(Math.Min(Dims.X, position.X + values.Dims.X), 0),
                                Math.Max(Math.Min(Dims.X, position.Y + values.Dims.Y), 0),
                                Math.Max(Math.Min(Dims.Z, position.Z + values.Dims.Z), 0));

            float[][] ThisData = GetHost(Intent.ReadWrite);
            float[][] ValuesData = values.GetHost(Intent.Read);

            for (int z = Start.Z; z < End.Z; z++)
            {
                int vz = z - position.Z;

                for (int y = Start.Y; y < End.Y; y++)
                {
                    int vy = y - position.Y;

                    for (int x = Start.X; x < End.X; x++)
                    {
                        int vx = x - position.X;

                        if (replace)
                            ThisData[z][y * Dims.X + x] = ValuesData[vz][vy * values.Dims.X + vx];
                        else
                            ThisData[z][y * Dims.X + x] += ValuesData[vz][vy * values.Dims.X + vx];
                    }
                }
            }
        }

        #endregion

        public int3[] GetLocalPeaks(int localExtent, float threshold)
        {
            int[] NPeaks = new int[1];

            IntPtr PeaksPtr = GPU.LocalPeaks(GetDevice(Intent.Read), NPeaks, Dims, localExtent, threshold);

            if (NPeaks[0] > 0)
            {
                int[] Peaks = new int[NPeaks[0] * 3];
                Marshal.Copy(PeaksPtr, Peaks, 0, Peaks.Length);

                CPU.HostFree(PeaksPtr);

                return Helper.FromInterleaved3(Peaks);
            }
            else
            {
                return new int3[0];
            }
        }

        public void RealspaceProject(float3[] angles, Image result, int supersample)
        {
            GPU.RealspaceProjectForward(GetDevice(Intent.Read),
                                        Dims,
                                        result.GetDevice(Intent.ReadWrite),
                                        new int2(result.Dims),
                                        supersample,
                                        Helper.ToInterleaved(angles),
                                        angles.Length);
        }

        public void RealspaceBackproject(Image projections, float3[] angles, int supersample, bool normalizesamples = true)
        {
            GPU.RealspaceProjectBackward(GetDevice(Intent.Write),
                                         Dims,
                                         projections.GetDevice(Intent.Read),
                                         new int2(projections.Dims),
                                         supersample,
                                         Helper.ToInterleaved(angles),
                                         normalizesamples,
                                         angles.Length);
        }

        public (int[] ComponentIndices, int[] NeighborhoodIndices)[] GetConnectedComponents(int neighborhoodExtent = 0, int[] labelsBuffer = null)
        {
            if (Dims.Z > 1)
                throw new Exception("No volumetric data supported!");

            float[] PixelData = GetHost(Intent.Read)[0];

            List<List<int>> Components = new List<List<int>>();
            List<List<int>> Neighborhoods = new List<List<int>>();
            if (labelsBuffer == null)
                labelsBuffer = new int[PixelData.Length];

            for (int i = 0; i < labelsBuffer.Length; i++)
                labelsBuffer[i] = -1;

            List<int> Peaks = new List<int>();
            for (int i = 0; i < PixelData.Length; i++)
                if (PixelData[i] != 0)
                    Peaks.Add(i);

            Queue<int> Expansion = new Queue<int>(100);


            foreach (var peak in Peaks)
            {
                if (labelsBuffer[peak] >= 0)
                    continue;

                #region Connected component

                List<int> Component = new List<int>() { peak };
                int CN = Components.Count;

                labelsBuffer[peak] = CN;
                Expansion.Clear();
                Expansion.Enqueue(peak);

                while (Expansion.Count > 0)
                {
                    int PosElement = Expansion.Dequeue();
                    int2 pos = new int2(PosElement % Dims.X, PosElement / Dims.X);

                    int[] Neighbors = new int[]
                    {
                        PosElement - 1,
                        PosElement + 1,
                        PosElement - Dims.X,
                        PosElement + Dims.X,
                        PosElement + 1 - Dims.X,
                        PosElement + 1 + Dims.X,
                        PosElement - 1 - Dims.X,
                        PosElement - 1 + Dims.X
                    };

                    foreach (var neighbor in Neighbors)
                    {
                        if (neighbor < 0 || neighbor >= PixelData.Length)
                            continue;

                        if (PixelData[neighbor] > 0 && labelsBuffer[neighbor] < 0)
                        {
                            labelsBuffer[neighbor] = CN;
                            Component.Add(neighbor);
                            Expansion.Enqueue(neighbor);
                        }
                    }
                }

                Components.Add(Component);

                #endregion

                #region Optional neighborhood around component

                List<int> CurrentFrontier = new List<int>(Component);
                List<int> NextFrontier = new List<int>();
                List<int> Neighborhood = new List<int>();
                int NN = -(CN + 2);

                for (int iexpansion = 0; iexpansion < neighborhoodExtent; iexpansion++)
                {
                    foreach (int PosElement in CurrentFrontier)
                    {
                        int2 pos = new int2(PosElement % Dims.X, PosElement / Dims.X);

                        if (pos.X > 0 && PixelData[PosElement - 1] == 0 && labelsBuffer[PosElement - 1] != NN)
                        {
                            labelsBuffer[PosElement - 1] = NN;
                            Neighborhood.Add(PosElement + (-1));
                            NextFrontier.Add(PosElement + (-1));
                        }
                        if (pos.X < Dims.X - 1 && PixelData[PosElement + 1] == 0 && labelsBuffer[PosElement + 1] != NN)
                        {
                            labelsBuffer[PosElement + 1] = NN;
                            Neighborhood.Add(PosElement + (1));
                            NextFrontier.Add(PosElement + (1));
                        }

                        if (pos.Y > 0 && PixelData[PosElement - Dims.X] == 0 && labelsBuffer[PosElement - Dims.X] != NN)
                        {
                            labelsBuffer[PosElement - Dims.X] = NN;
                            Neighborhood.Add(PosElement + (-Dims.X));
                            NextFrontier.Add(PosElement + (-Dims.X));
                        }
                        if (pos.Y < Dims.Y - 1 && PixelData[PosElement + Dims.X] == 0 && labelsBuffer[PosElement + Dims.X] != NN)
                        {
                            labelsBuffer[PosElement + Dims.X] = NN;
                            Neighborhood.Add(PosElement + (Dims.X));
                            NextFrontier.Add(PosElement + (Dims.X));
                        }
                    }

                    CurrentFrontier = NextFrontier;
                    NextFrontier = new List<int>();
                }

                Neighborhoods.Add(Neighborhood);

                #endregion
            }

            return Helper.ArrayOfFunction(i => (Components[i].ToArray(), Neighborhoods[i].ToArray()), Components.Count);
        }

        public bool IsSameAs(Image other, float error = 0.001f)
        {
            float[] ThisMemory = GetHostContinuousCopy();
            float[] OtherMemory = other.GetHostContinuousCopy();

            if (ThisMemory.Length != OtherMemory.Length)
                return false;

            for (int i = 0; i < ThisMemory.Length; i++)
            {
                float ThisVal = ThisMemory[i];
                float OtherVal = OtherMemory[i];
                if (ThisVal != OtherVal)
                {
                    if (OtherVal == 0)
                        return false;

                    float Diff = Math.Abs((ThisVal - OtherVal) / OtherVal);
                    if (Diff > error)
                        return false;
                }
            }

            return true;
        }

        public override string ToString()
        {
            return Dims.ToString() + ", " + 
                   (IsFT ? "FT, " : "normal, ") + 
                   (IsComplex ? "complex, " : "real, ") + 
                   PixelSize + " A/px, " + 
                   "ID = " + ObjectID +
                   (_DeviceData == IntPtr.Zero ? "" : ", on device") +
                   (IsDisposed ? ", disposed" : "");
        }

        public static Image Stack(Image[] images)
        {
            int SumZ = images.Sum(i => i.Dims.Z);

            Image Stacked = new Image(new int3(images[0].Dims.X, images[0].Dims.Y, SumZ), images[0].IsFT, images[0].IsComplex);
            float[][] StackedData = Stacked.GetHost(Intent.Write);

            int OffsetZ = 0;
            foreach (var image in images)
            {
                float[][] ImageData = image.GetHost(Intent.Read);
                for (int i = 0; i < ImageData.Length; i++)
                    Array.Copy(ImageData[i], 0, StackedData[i + OffsetZ], 0, ImageData[i].Length);
                OffsetZ += ImageData.Length;
            }

            Stacked.PixelSize = images[0].PixelSize;

            return Stacked;
        }

        public static Image ReconstructSIRT(Image data, float3[] angles, int3 dimsrec, int supersample, int niterations, Image residuals = null)
        {
            int2 DimsProj = new int2(data.Dims);
            int2 DimsProjSuper = DimsProj * supersample;

            int PlanForwUp = GPU.CreateFFTPlan(new int3(DimsProj), (uint)angles.Length);
            int PlanBackUp = GPU.CreateIFFTPlan(new int3(DimsProjSuper), (uint)angles.Length);

            int PlanForwDown = GPU.CreateFFTPlan(new int3(DimsProjSuper), (uint)angles.Length);
            int PlanBackDown = GPU.CreateIFFTPlan(new int3(DimsProj), (uint)angles.Length);

            Image Projections = new Image(new int3(DimsProj.X, DimsProj.Y, angles.Length));
            Image ProjectionsSuper = new Image(new int3(DimsProjSuper.X, DimsProjSuper.Y, angles.Length));

            Image ProjectionSamples = new Image(Projections.Dims);
            
            Image VolReconstruction = new Image(dimsrec);
            Image VolCorrection = new Image(dimsrec);
            VolCorrection.Fill(1f);

            // Figure out number of samples per projection pixel
            {
                VolCorrection.RealspaceProject(angles, ProjectionsSuper, supersample);

                GPU.Scale(ProjectionsSuper.GetDevice(Intent.Read),
                            ProjectionSamples.GetDevice(Intent.Write),
                            new int3(DimsProjSuper),
                            new int3(DimsProj),
                            (uint)angles.Length,
                            PlanForwDown,
                            PlanBackDown,
                            IntPtr.Zero,
                            IntPtr.Zero);

                ProjectionSamples.Max(1f);
                //ProjectionSamples.WriteMRC("d_samples.mrc", true);
            }

            // Supersample data and backproject to initialize volume
            {
                GPU.Scale(data.GetDevice(Intent.Read),
                          ProjectionsSuper.GetDevice(Intent.Write),
                          new int3(DimsProj),
                          new int3(DimsProjSuper),
                          (uint)angles.Length,
                          PlanForwUp,
                          PlanBackUp,
                          IntPtr.Zero,
                          IntPtr.Zero);

                //ProjectionsSuper.WriteMRC("d_datasuper.mrc", true);

                VolReconstruction.RealspaceBackproject(ProjectionsSuper, angles, supersample);
            }

            for (int i = 0; i < niterations; i++)
            {
                VolReconstruction.RealspaceProject(angles, ProjectionsSuper, supersample);
                
                GPU.Scale(ProjectionsSuper.GetDevice(Intent.Read),
                            Projections.GetDevice(Intent.Write),
                            new int3(DimsProjSuper),
                            new int3(DimsProj),
                            (uint)angles.Length,
                            PlanForwDown,
                            PlanBackDown,
                            IntPtr.Zero,
                            IntPtr.Zero);
                //Projections.WriteMRC("d_projections.mrc", true);

                GPU.SubtractFromSlices(data.GetDevice(Intent.Read),
                                       Projections.GetDevice(Intent.Read),
                                       Projections.GetDevice(Intent.Write),
                                       Projections.ElementsReal,
                                       1);

                if (i == niterations - 1 && residuals != null)
                    GPU.CopyDeviceToDevice(Projections.GetDevice(Intent.Read),
                                           residuals.GetDevice(Intent.Write),
                                           Projections.ElementsReal);

                Projections.Divide(ProjectionSamples);
                //Projections.WriteMRC("d_correction2d.mrc", true);
                Projections.Taper(8);

                GPU.Scale(Projections.GetDevice(Intent.Read),
                          ProjectionsSuper.GetDevice(Intent.Write),
                          new int3(DimsProj),
                          new int3(DimsProjSuper),
                          (uint)angles.Length,
                          PlanForwUp,
                          PlanBackUp,
                          IntPtr.Zero,
                          IntPtr.Zero);

                VolCorrection.RealspaceBackproject(ProjectionsSuper, angles, supersample);
                //VolCorrection.WriteMRC("d_correction3d.mrc", true);

                VolReconstruction.Add(VolCorrection);
                //VolReconstruction.WriteMRC($"d_reconstruction_{i:D3}.mrc", true);
            }

            //VolReconstruction.WriteMRC($"d_reconstruction_final.mrc", true);

            VolCorrection.Dispose();
            ProjectionSamples.Dispose();
            ProjectionsSuper.Dispose();
            Projections.Dispose();

            GPU.DestroyFFTPlan(PlanForwUp);
            GPU.DestroyFFTPlan(PlanForwDown);
            GPU.DestroyFFTPlan(PlanBackUp);
            GPU.DestroyFFTPlan(PlanBackDown);

            return VolReconstruction;
        }

        public static void PrintObjectIDs()
        {
            lock (GlobalSync)
            {
                for (int i = 0; i < LifetimeObjects.Count; i++)
                {
                    Debug.WriteLine(LifetimeObjects[i].ToString());
                    Debug.WriteLine(LifetimeObjects[i].ObjectCreationLocation.ToString() + "\n");
                }
            }
        }
    }
    
    [Flags]
    public enum Intent
    {
        Read = 1 << 0,
        Write = 1 << 1,
        ReadWrite = Read | Write
    }
}
