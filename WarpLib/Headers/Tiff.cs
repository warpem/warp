using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BitMiracle.LibTiff.Classic;
using Warp.Tools;

/*
LICENSE NOTE FOR TIFF LIBRARY:

LibTiff.Net
Copyright (c) 2008-2011, Bit Miracle

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list 
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.

Neither the name of the Bit Miracle nor the names of its contributors may be used 
to endorse or promote products derived from this software without specific prior 
written permission.
*/

namespace Warp.Headers
{
    public enum TiffDataType
    {
        Byte = 0,
        Ushort = 1,
        Short = 2,
        Uint = 3,
        Int = 4,
        Ulong = 5,
        Long = 6,
        Half = 7,
        Float = 8,
        Double = 9
    }

    public class HeaderTiff : MapHeader
    {
        private string Path;
        public TiffDataType Mode = TiffDataType.Float;

        public HeaderTiff()
        {
        }

        public HeaderTiff(string path, Stream stream = null)
        {
            Tiff.SetErrorHandler(new DisableErrorHandler());

            Path = path;

            if (stream == null)
                stream = File.OpenRead(path);

            using (Tiff Image = Tiff.ClientOpen("inmemory", "r", stream, new TiffStream()))
            {
                {
                    FieldValue[] value = Image.GetField(TiffTag.IMAGEWIDTH);
                    Dimensions.X = value[0].ToInt();
                }
                {
                    FieldValue[] value = Image.GetField(TiffTag.IMAGELENGTH);
                    Dimensions.Y = value[0].ToInt();
                }
                {
                    Dimensions.Z = Image.NumberOfDirectories();
                }

                if (Helper.PathToExtension(path).ToLower() != ".eer")
                {
                    FieldValue[] value = Image.GetField(TiffTag.SAMPLEFORMAT);
                    SampleFormat Format = SampleFormat.UINT;
                    if (value != null)
                        Format = (SampleFormat)value[0].ToInt();

                    int BitsPerPixel = Image.GetField(TiffTag.BITSPERSAMPLE)[0].ToInt();

                    if (Format == SampleFormat.UINT)
                    {
                        if (BitsPerPixel == 8)
                            Mode = TiffDataType.Byte;
                        else if (BitsPerPixel == 16)
                            Mode = TiffDataType.Ushort;
                        else if (BitsPerPixel == 32)
                            Mode = TiffDataType.Uint;
                        else if (BitsPerPixel == 64)
                            Mode = TiffDataType.Ulong;
                        else
                            throw new FormatException("Unexpected bits per pixel.");
                    }
                    else if (Format == SampleFormat.INT)
                    {
                        if (BitsPerPixel == 8)
                            Mode = TiffDataType.Byte;
                        else if (BitsPerPixel == 16)
                            Mode = TiffDataType.Short;
                        else if (BitsPerPixel == 32)
                            Mode = TiffDataType.Int;
                        else if (BitsPerPixel == 64)
                            Mode = TiffDataType.Long;
                        else
                            throw new FormatException("Unexpected bits per pixel.");
                    }
                    else if (Format == SampleFormat.IEEEFP)
                    {
                        if (BitsPerPixel == 16)
                            Mode = TiffDataType.Half;
                        else if (BitsPerPixel == 32)
                            Mode = TiffDataType.Float;
                        else if (BitsPerPixel == 64)
                            Mode = TiffDataType.Double;
                        else
                            throw new FormatException("Unexpected bits per pixel.");
                    }
                    else
                        throw new FormatException("Unexpected format.");
                }
            }

            if (stream.GetType() != typeof(MemoryStream))
                stream.Close();
        }

        public override void Write(BinaryWriter writer)
        {
            throw new NotImplementedException();
        }

        public float[][] ReadData(int[] layers = null)
        {
            string FlipYEnvVar = Environment.GetEnvironmentVariable("WARP_DONT_FLIPY");
            bool DoFlipY = string.IsNullOrEmpty(FlipYEnvVar);

            float[][] Slices = new float[layers == null ? Dimensions.Z : layers.Length][];

            for (int slice = 0; slice < Slices.Length; slice++)
            {
                using (Tiff Image = Tiff.Open(Path, "r"))
                {
                    bool Encoded = false;
                    FieldValue[] compressionTagValue = Image.GetField(TiffTag.COMPRESSION);
                    if (compressionTagValue != null)
                        Encoded = (compressionTagValue[0].ToInt() != (int)Compression.NONE);

                    int NumberOfStrips = Image.NumberOfStrips();

                    float[] ConvertedData = new float[Dimensions.ElementsSlice()];
                    float[] ConvertedDataFlipY = DoFlipY ? new float[Dimensions.ElementsSlice()] : null;

                    Image.SetDirectory(layers == null ? (short)slice : (short)layers[slice]);

                    int Offset = 0;
                    byte[] StripsData = new byte[NumberOfStrips * Image.StripSize()];
                    for (int i = 0; i < NumberOfStrips; i++)
                    {
                        int bytesRead = readStrip(Image, i, StripsData, Offset, Encoded);
                        Offset += bytesRead;
                    }

                    unsafe
                    {
                        fixed (byte* StripsDataPtr = StripsData)
                        fixed (float* ConvertedDataPtr = ConvertedData)
                        {
                            int FullLength = ConvertedData.Length;
                            int SafeLength = FullLength / 4;

                            float* ConvertedDataP = ConvertedDataPtr;

                            if (Mode == TiffDataType.Byte)
                            {
                                byte* StripsDataP = (byte*)StripsDataPtr;
                                for (int i = 0; i < SafeLength; i++)
                                {
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                }
                                for (int i = SafeLength * 4; i < FullLength; i++)
                                {
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                }
                            }
                            else if (Mode == TiffDataType.Ushort)
                            {
                                ushort* StripsDataP = (ushort*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Short)
                            {
                                short* StripsDataP = (short*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Uint)
                            {
                                uint* StripsDataP = (uint*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Int)
                            {
                                int* StripsDataP = (int*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Ulong)
                            {
                                ulong* StripsDataP = (ulong*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Long)
                            {
                                long* StripsDataP = (long*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Half)
                            {
                                Half* StripsDataP = (Half*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Float)
                            {
                                float* StripsDataP = (float*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Double)
                            {
                                double* StripsDataP = (double*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else
                                throw new FormatException("Unexpected data type.");

                            // Annoyingly, flip Y axis to adhere to MRC convention
                            if (DoFlipY)
                            {
                                fixed (float* ConvertedDataFlipYPtr = ConvertedDataFlipY)
                                {
                                    int Width = Dimensions.X, Height = Dimensions.Y;
                                    int WidthSafe = (Width / 4) * 4;
                                    for (int y = 0; y < Height; y++)
                                    {
                                        int YOffset = y * Width;
                                        int YOffsetFlipped = (Height - 1 - y) * Width;

                                        for (int x = 0; x < WidthSafe; x += 4)
                                        {
                                            ConvertedDataFlipYPtr[YOffset + x] = ConvertedDataPtr[YOffsetFlipped + x];
                                            ConvertedDataFlipYPtr[YOffset + x + 1] = ConvertedDataPtr[YOffsetFlipped + x + 1];
                                            ConvertedDataFlipYPtr[YOffset + x + 2] = ConvertedDataPtr[YOffsetFlipped + x + 2];
                                            ConvertedDataFlipYPtr[YOffset + x + 3] = ConvertedDataPtr[YOffsetFlipped + x + 3];
                                        }

                                        for (int x = WidthSafe; x < Width; x++)
                                        {
                                            ConvertedDataFlipYPtr[YOffset + x] = ConvertedDataPtr[YOffsetFlipped + x];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Slices[slice] = DoFlipY ? ConvertedDataFlipY : ConvertedData;
                }
            }

            return Slices;
        }

        public float[][] ReadData(Stream stream, int[] layers = null)
        {
            float[][] Slices = new float[layers == null ? Dimensions.Z : layers.Length][];

            if (stream == null)
                stream = File.OpenRead(Path);

            string FlipYEnvVar = Environment.GetEnvironmentVariable("WARP_DONT_FLIPY");
            bool DoFlipY = string.IsNullOrEmpty(FlipYEnvVar);

            for (int slice = 0; slice < Slices.Length; slice++)
            {
                stream.Seek(0, SeekOrigin.Begin);

                Tiff Image = Tiff.ClientOpen("inmemory", "rm", stream, new TiffStream());
                {
                    bool Encoded = false;
                    FieldValue[] compressionTagValue = Image.GetField(TiffTag.COMPRESSION);
                    if (compressionTagValue != null)
                        Encoded = (compressionTagValue[0].ToInt() != (int)Compression.NONE);

                    int NumberOfStrips = Image.NumberOfStrips();

                    float[] ConvertedData = new float[Dimensions.ElementsSlice()];
                    float[] ConvertedDataFlipY = DoFlipY ? new float[Dimensions.ElementsSlice()] : null;

                    Image.SetDirectory(layers == null ? (short)slice : (short)layers[slice]);

                    int Offset = 0;
                    byte[] StripsData = new byte[NumberOfStrips * Image.StripSize()];
                    for (int i = 0; i < NumberOfStrips; i++)
                    {
                        int bytesRead = readStrip(Image, i, StripsData, Offset, Encoded);
                        Offset += bytesRead;
                    }

                    unsafe
                    {
                        fixed (byte* StripsDataPtr = StripsData)
                        fixed (float* ConvertedDataPtr = ConvertedData)
                        {
                            int FullLength = ConvertedData.Length;
                            int SafeLength = FullLength / 4;

                            float* ConvertedDataP = ConvertedDataPtr;

                            if (Mode == TiffDataType.Byte)
                            {
                                byte* StripsDataP = (byte*)StripsDataPtr;
                                for (int i = 0; i < SafeLength; i++)
                                {
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                }
                                for (int i = SafeLength * 4; i < FullLength; i++)
                                {
                                    *ConvertedDataP++ = (float)*StripsDataP++;
                                }
                            }
                            else if (Mode == TiffDataType.Ushort)
                            {
                                ushort* StripsDataP = (ushort*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Short)
                            {
                                short* StripsDataP = (short*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Uint)
                            {
                                uint* StripsDataP = (uint*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Int)
                            {
                                int* StripsDataP = (int*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Ulong)
                            {
                                ulong* StripsDataP = (ulong*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Long)
                            {
                                long* StripsDataP = (long*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Half)
                            {
                                Half* StripsDataP = (Half*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Float)
                            {
                                float* StripsDataP = (float*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = StripsDataP[i];
                            }
                            else if (Mode == TiffDataType.Double)
                            {
                                double* StripsDataP = (double*)StripsDataPtr;
                                for (int i = 0; i < ConvertedData.Length; i++)
                                    ConvertedDataPtr[i] = (float)StripsDataP[i];
                            }
                            else
                                throw new FormatException("Unexpected data type.");

                            // Annoyingly, flip Y axis to adhere to MRC convention
                            if (DoFlipY)
                            {
                                fixed (float* ConvertedDataFlipYPtr = ConvertedDataFlipY)
                                {
                                    int Width = Dimensions.X, Height = Dimensions.Y;
                                    int WidthSafe = (Width / 4) * 4;
                                    for (int y = 0; y < Height; y++)
                                    {
                                        int YOffset = y * Width;
                                        int YOffsetFlipped = (Height - 1 - y) * Width;

                                        for (int x = 0; x < WidthSafe; x += 4)
                                        {
                                            ConvertedDataFlipYPtr[YOffset + x] = ConvertedDataPtr[YOffsetFlipped + x];
                                            ConvertedDataFlipYPtr[YOffset + x + 1] = ConvertedDataPtr[YOffsetFlipped + x + 1];
                                            ConvertedDataFlipYPtr[YOffset + x + 2] = ConvertedDataPtr[YOffsetFlipped + x + 2];
                                            ConvertedDataFlipYPtr[YOffset + x + 3] = ConvertedDataPtr[YOffsetFlipped + x + 3];
                                        }

                                        for (int x = WidthSafe; x < Width; x++)
                                        {
                                            ConvertedDataFlipYPtr[YOffset + x] = ConvertedDataPtr[YOffsetFlipped + x];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Slices[slice] = DoFlipY ? ConvertedDataFlipY : ConvertedData;
                }
            }

            if (stream.GetType() != typeof(MemoryStream))
                stream.Close();

            return Slices;
        }

        private static int readStrip(Tiff image, int stripNumber, byte[] buffer, int offset, bool encoded)
        {
            //if (encoded)
                return image.ReadEncodedStrip(stripNumber, buffer, offset, buffer.Length - offset);
            //else
            //    return image.ReadRawStrip(stripNumber, buffer, offset, buffer.Length - offset);
        }

        public override Type GetValueType()
        {
            switch (Mode)
            {
                case TiffDataType.Byte:
                    return typeof(byte);
                case TiffDataType.Ushort:
                    return typeof(ushort);
                case TiffDataType.Short:
                    return typeof(short);
                case TiffDataType.Uint:
                    return typeof(uint);
                case TiffDataType.Int:
                    return typeof(int);
                case TiffDataType.Ulong:
                    return typeof(ulong);
                case TiffDataType.Long:
                    return typeof(long);
                case TiffDataType.Float:
                    return typeof(float);
                case TiffDataType.Double:
                    return typeof(double);
            }

            throw new Exception("Unknown data type.");
        }

        public override void SetValueType(Type t)
        {
            throw new NotImplementedException();
        }
    }
}
