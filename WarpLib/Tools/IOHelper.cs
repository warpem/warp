using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.AccessControl;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Warp.Headers;

namespace Warp.Tools
{
    public static class IOHelper
    {
        public static Stream OpenWithBigBuffer(string path)
        {
            return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 22, FileOptions.SequentialScan);
        }

        public static Stream CreateWithBigBuffer(string path)
        {
            return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write, 1 << 22);
        }

        public static int3 GetMapDimensions(string path)
        {
            int3 Dims = new int3(1, 1, 1);
            FileInfo Info = new FileInfo(path);

            using (BinaryReader Reader = new BinaryReader(OpenWithBigBuffer(path)))
            {
                if (Info.Extension.ToLower() == ".mrc" || Info.Extension.ToLower() == ".mrcs")
                {
                    HeaderMRC Header = new HeaderMRC(Reader);
                    Dims = Header.Dimensions;
                }
                else if (Info.Extension.ToLower() == ".em")
                {
                    HeaderEM Header = new HeaderEM(Reader);
                    Dims = Header.Dimensions;
                }
                else
                    throw new Exception("Format not supported.");
            }

            return Dims;
        }

        public static float[][] ReadMapFloatPatient(int attempts, int mswait, string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, int[] layers = null, Stream stream = null, float[][] reuseBuffer = null)
        {
            Exception LastException = null;

            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    return ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, layers, stream, reuseBuffer);
                }
                catch (Exception exc)
                {
                    Thread.Sleep(mswait);
                    LastException = exc;
                }
            }

            throw new Exception($"Could not successfully read {path} within the specified number of attempts:\n" +
                                $"{LastException}");
        }

        public static float[][] ReadMapFloat(string path, int[] layers = null, Stream stream = null, float[][] reuseBuffer = null) => ReadMapFloat(path, new int2(1), 0, typeof(float), layers, stream, reuseBuffer);

        public static float[][] ReadMapFloat(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, int[] layers = null, Stream stream = null, float[][] reuseBuffer = null)
        {
            try
            {
                return ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, false, layers, stream, reuseBuffer);
            }
            catch
            {
                return ReadMapFloat(path, headerlessSliceDims, headerlessOffset, headerlessType, true, layers, stream, reuseBuffer);
            }
        }

        public static float[][] ReadMapFloat(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, bool isBigEndian, int[] layers = null, Stream stream = null, float[][] reuseBuffer = null)
        {
            MapHeader Header = null;
            Type ValueType = null;
            float[][] Data;

            if (MapHeader.GetHeaderType(path) != typeof(HeaderTiff))
                using (BinaryReader Reader = isBigEndian ? new BinaryReaderBE(OpenWithBigBuffer(path)) : new BinaryReader(OpenWithBigBuffer(path)))
                {
                    Header = MapHeader.ReadFromFile(Reader, path, headerlessSliceDims, headerlessOffset, headerlessType);
                    ValueType = Header.GetValueType();
                    Data = reuseBuffer == null ? new float[layers == null ? Header.Dimensions.Z : layers.Length][] : reuseBuffer;

                    int ReadBatchSize = Math.Min((int)Header.Dimensions.ElementsSlice(), 1 << 20);
                    int ValueSize = (int)ImageFormatsHelper.SizeOf(ValueType);
                    byte[] Bytes = new byte[ReadBatchSize * ValueSize];

                    long ReaderDataStart = Reader.BaseStream.Position;
                    int CurrentLayer = 0;

                    for (int z = 0; z < (layers == null ? Data.Length : layers.Length); z++)
                    {
                        if (layers != null && CurrentLayer != layers[z])
                        {
                            Reader.BaseStream.Seek(Header.Dimensions.ElementsSlice() * ImageFormatsHelper.SizeOf(ValueType) * layers[z] + ReaderDataStart, SeekOrigin.Begin);
                            CurrentLayer = layers[z];
                        }

                        if (reuseBuffer == null)
                            Data[z] = new float[(int)Header.Dimensions.ElementsSlice()];

                        unsafe
                        {
                            fixed (byte* BytesPtr = Bytes)
                            fixed (float* DataPtr = Data[z])
                            {
                                for (int b = 0; b < (int)Header.Dimensions.ElementsSlice(); b += ReadBatchSize)
                                {
                                    int CurBatch = Math.Min(ReadBatchSize, (int)Header.Dimensions.ElementsSlice() - b);

                                    Reader.Read(Bytes, 0, CurBatch * ValueSize);

                                    if (isBigEndian)
                                    {
                                        if (ValueType == typeof(short) || ValueType == typeof(ushort) || ValueType == typeof(Half))
                                        {
                                            for (int i = 0; i < CurBatch * ValueSize / 2; i++)
                                                Array.Reverse(Bytes, i * 2, 2);
                                        }
                                        else if (ValueType == typeof(int) || ValueType == typeof(float))
                                        {
                                            for (int i = 0; i < CurBatch * ValueSize / 4; i++)
                                                Array.Reverse(Bytes, i * 4, 4);
                                        }
                                        else if (ValueType == typeof(double))
                                        {
                                            for (int i = 0; i < CurBatch * ValueSize / 8; i++)
                                                Array.Reverse(Bytes, i * 8, 8);
                                        }
                                    }

                                    float* DataP = DataPtr + b;

                                    if (ValueType == typeof(byte))
                                    {
                                        byte* BytesP = BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = (float)*BytesP++;
                                    }
                                    else if (ValueType == typeof(short))
                                    {
                                        short* BytesP = (short*)BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = (float)*BytesP++;
                                    }
                                    else if (ValueType == typeof(ushort))
                                    {
                                        ushort* BytesP = (ushort*)BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = (float)*BytesP++;
                                    }
                                    else if (ValueType == typeof(int))
                                    {
                                        int* BytesP = (int*)BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = (float)*BytesP++;
                                    }
                                    else if (ValueType == typeof(Half))
                                    {
                                        CPU.HalfToFloat((ushort*)BytesPtr, DataP, CurBatch);
                                    }
                                    else if (ValueType == typeof(float))
                                    {
                                        float* BytesP = (float*)BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = *BytesP++;
                                    }
                                    else if (ValueType == typeof(double))
                                    {
                                        double* BytesP = (double*)BytesPtr;
                                        for (int i = 0; i < CurBatch; i++)
                                            *DataP++ = (float)*BytesP++;
                                    }
                                }
                            }
                        }

                        CurrentLayer++;
                    }
                }
            else
            {
                Header = MapHeader.ReadFromFile(null, path, headerlessSliceDims, headerlessOffset, headerlessType, stream);
                if (Helper.PathToExtension(path).ToLower() == ".eer")
                {
                    Data = Helper.ArrayOfFunction(i => new float[Header.Dimensions.ElementsSlice()], layers == null ? Header.Dimensions.Z : layers.Length);
                    for (int i = 0; i < Data.Length; i++)
                    {
                        int z = layers == null ? i : layers[i];
                        EERNative.ReadEER(path, z * HeaderEER.GroupNFrames, (z + 1) * HeaderEER.GroupNFrames, HeaderEER.SuperResolution, Data[i]);
                    }
                }
                else
                {
                    Data = ((HeaderTiff)Header).ReadData(stream, layers);
                }
            }

            return Data;
        }

        public static void WriteMapFloat(string path, MapHeader header, float[] data)
        {
            Type ValueType = header.GetValueType();
            long Elements = header.Dimensions.Elements();

            using (BinaryWriter Writer = new BinaryWriter(CreateWithBigBuffer(path)))
            {
                header.Write(Writer);
                byte[] Bytes = null;
                
                Bytes = new byte[Elements * ImageFormatsHelper.SizeOf(ValueType)];

                unsafe
                {
                    fixed (float* DataPtr = data)
                    fixed (byte* BytesPtr = Bytes)
                    {
                        float* DataP = DataPtr;

                        if (ValueType == typeof(byte))
                        {
                            byte* BytesP = BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (byte)*DataP++;
                        }
                        else if (ValueType == typeof(short))
                        {
                            short* BytesP = (short*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (short)*DataP++;
                        }
                        else if (ValueType == typeof(ushort))
                        {
                            ushort* BytesP = (ushort*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (ushort)Math.Min(Math.Max(0f, *DataP++ * 16f), 65536f);
                        }
                        else if (ValueType == typeof(int))
                        {
                            int* BytesP = (int*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (int)*DataP++;
                        }
                        else if (ValueType == typeof(Half))
                        {
                            CPU.FloatToHalf(DataPtr, (ushort*)BytesPtr, Elements);
                        }
                        else if (ValueType == typeof(float))
                        {
                            float* BytesP = (float*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = *DataP++;
                        }
                        else if (ValueType == typeof(double))
                        {
                            double* BytesP = (double*)BytesPtr;
                            for (long i = 0; i < Elements; i++)
                                *BytesP++ = (double)*DataP++;
                        }
                    }
                }

                Writer.Write(Bytes);
            }
        }

        public static void WriteMapFloat(string path, MapHeader header, float[][] data)
        {
            Type ValueType = header.GetValueType();

            using (BinaryWriter Writer = new BinaryWriter(CreateWithBigBuffer(path)))
            {
                header.Write(Writer);
                long Elements = header.Dimensions.ElementsSlice();

                if (ValueType == typeof(float))
                {
                    for (int z = 0; z < header.Dimensions.Z; z++)
                    {
                        ReadOnlySpan<float> DataSpan = new ReadOnlySpan<float>(data[z]);
                        ReadOnlySpan<byte> Bytes = MemoryMarshal.AsBytes(DataSpan);

                        Writer.Write(Bytes);
                    }
                }
                else
                {
                    byte[] Bytes = new byte[Elements * ImageFormatsHelper.SizeOf(ValueType)];

                    for (int z = 0; z < header.Dimensions.Z; z++)
                    {
                        if (ValueType == typeof(float))
                        {
                            Buffer.BlockCopy(data[z], 0, Bytes, 0, Bytes.Length);
                        }
                        else
                        {
                            unsafe
                            {
                                fixed (float* DataPtr = data[z])
                                fixed (byte* BytesPtr = Bytes)
                                {
                                    float* DataP = DataPtr;

                                    if (ValueType == typeof(byte))
                                    {
                                        byte* BytesP = BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = (byte)*DataP++;
                                    }
                                    else if (ValueType == typeof(short))
                                    {
                                        short* BytesP = (short*)BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = (short)*DataP++;
                                    }
                                    else if (ValueType == typeof(ushort))
                                    {
                                        ushort* BytesP = (ushort*)BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = (ushort)Math.Min(Math.Max(0f, *DataP++ * 16f), 65536f);
                                    }
                                    else if (ValueType == typeof(int))
                                    {
                                        int* BytesP = (int*)BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = (int)*DataP++;
                                    }
                                    else if (ValueType == typeof(Half))
                                    {
                                        CPU.FloatToHalf(DataPtr, (ushort*)BytesPtr, Elements);
                                    }
                                    else if (ValueType == typeof(float))
                                    {
                                        float* BytesP = (float*)BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = *DataP++;
                                    }
                                    else if (ValueType == typeof(double))
                                    {
                                        double* BytesP = (double*)BytesPtr;
                                        for (long i = 0; i < Elements; i++)
                                            *BytesP++ = (double)*DataP++;
                                    }
                                }
                            }
                        }

                        Writer.Write(Bytes);
                    }
                }
            }
        }

        public static bool CheckFolderPermission(string path)
        {
            if (!Directory.Exists(path))
                return false;

            try
            {
                string FilePath = Guid.NewGuid().ToString() + ".test";
                using (TextWriter Writer = File.CreateText(Path.Combine(path, FilePath)))
                {
                    Writer.WriteLine("test");
                }
                File.Delete(Path.Combine(path, FilePath));

                return true;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
        }
    }
}
