using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using Warp.Tools;

namespace Warp.Headers
{
    public abstract class MapHeader
    {
        public int3 Dimensions = new int3(1, 1, 1);
        public float3 PixelSize = new float3(1);
        public abstract void Write(BinaryWriter writer);

        public abstract Type GetValueType();
        public abstract void SetValueType(Type t);

        public static MapHeader ReadFromFile(BinaryReader reader, string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, Stream stream = null)
        {
            MapHeader Header = null;
            string Extension = Helper.PathToExtension(path).ToLower();

            if (Extension == ".mrc" || Extension == ".mrcs" || Extension == ".rec" || Extension == ".st" || Extension == ".map")
                Header = new HeaderMRC(reader);
            else if (Extension == ".em")
                Header = new HeaderEM(reader);
            else if (Extension == ".dm4" || Extension == ".dm3")
                Header = new HeaderDM4(reader);
            else if (Extension == ".tif" || Extension == ".tiff" || Extension == ".gain")
                Header = new HeaderTiff(path, stream);
            else if (Extension == ".eer")
                Header = new HeaderEER(path, stream);
            else if (Extension == ".dat")
            {
                FileInfo info = new FileInfo(path);

                long SliceElements = headerlessSliceDims.Elements() * ImageFormatsHelper.SizeOf(headerlessType);
                long Slices = (info.Length - headerlessOffset) / SliceElements;
                int3 Dims3 = new int3(headerlessSliceDims.X, headerlessSliceDims.Y, (int)Slices);
                Header = new HeaderRaw(Dims3, headerlessOffset, headerlessType);
            }
            else
                throw new Exception("File type not supported.");

            return Header;
        }

        public static Type GetHeaderType(string path)
        {
            string Extension = Helper.PathToExtension(path).ToLower();

            if (Extension == ".mrc" || Extension == ".mrcs" || Extension == ".rec" || Extension == ".st" || Extension == ".map")
                return typeof(HeaderMRC);
            else if (Extension == ".em")
                return typeof(HeaderEM);
            else if (Extension == ".tif" || Extension == ".tiff" || Extension == ".gain")
                return typeof(HeaderTiff);
            else if (Extension == ".eer")
                return typeof(HeaderEER);
            else if (Extension == ".dat")
                return typeof(HeaderRaw);
            else if (Extension == ".dm4" || Extension == ".dm3")
                return typeof(HeaderDM4);
            else
                throw new Exception("File type not supported.");
        }

        public static MapHeader ReadFromFile(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, Stream stream = null)
        {
            try
            {
                MapHeader Result = ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType, false, stream);
                if (Result.Dimensions.X < 0 || 
                    Result.Dimensions.Y < 0 || 
                    Result.Dimensions.Z < 0 || 
                    (Result.Dimensions.ElementsSlice() > 1 && Result.Dimensions.Z > 9999999))
                    throw new Exception();

                return Result;
            }
            catch
            {
                return ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType, true, stream);
            }
        }

        public static MapHeader ReadFromFile(string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType, bool isBigEndian, Stream stream = null)
        {
            MapHeader Header = null;

            if (GetHeaderType(path) != typeof(HeaderTiff))
                using (BinaryReader Reader = isBigEndian ? new BinaryReaderBE(File.OpenRead(path)) : new BinaryReader(File.OpenRead(path)))
                {
                    Header = ReadFromFile(Reader, path, headerlessSliceDims, headerlessOffset, headerlessType);
                }
            else
                Header = ReadFromFile(null, path, headerlessSliceDims, headerlessOffset, headerlessType, stream);

            return Header;
        }

        public static MapHeader ReadFromFile(string path)
        {
            return ReadFromFile(path, new int2(1, 1), 0, typeof(byte));
        }

        public static MapHeader ReadFromFilePatient(int attempts, int mswait, string path, int2 headerlessSliceDims, long headerlessOffset, Type headerlessType)
        {
            MapHeader Result = null;
            for (int a = 0; a < attempts; a++)
            {
                try
                {
                    Result = ReadFromFile(path, headerlessSliceDims, headerlessOffset, headerlessType);
                    break;
                }
                catch
                {
                    Thread.Sleep(mswait);
                }
            }

            if (Result == null)
                throw new Exception($"Could not successfully read {path} within the specified number of attempts.");

            return Result;
        }
    }

    public enum ImageFormats
    { 
        MRC = 0,
        MRCS = 1,
        EM = 2,
        K2Raw = 3,
        FEIRaw = 4,
        TIFF = 5,
        TIFFF = 6,
        EER = 7,
        DM4 = 8
    }

    public static class ImageFormatsHelper
    {
        public static ImageFormats Parse(string format)
        {
            switch (format)
            {
                case "MRC":
                    return ImageFormats.MRC;
                case "MRCS":
                    return ImageFormats.MRCS;
                case "EM":
                    return ImageFormats.EM;
                case "K2Raw":
                    return ImageFormats.K2Raw;
                case "FEIRaw":
                    return ImageFormats.FEIRaw;
                case "TIFF":
                    return ImageFormats.TIFF;
                case "TIFFF":
                    return ImageFormats.TIFFF;
                case "EER":
                    return ImageFormats.EER;
                case "DM4":
                    return ImageFormats.DM4;
                default:
                    return ImageFormats.MRC;
            }
        }

        public static string ToString(ImageFormats format)
        { 
            switch (format)
            {
                case ImageFormats.MRC:
                    return "MRC";
                case ImageFormats.MRCS:
                    return "MRCS";
                case ImageFormats.EM:
                    return "EM";
                case ImageFormats.K2Raw:
                    return "K2Raw";
                case ImageFormats.FEIRaw:
                    return "FEIRaw";
                case ImageFormats.TIFF:
                    return "TIFF";
                case ImageFormats.TIFFF:
                    return "TIFFF";
                case ImageFormats.EER:
                    return "EER";
                case ImageFormats.DM4:
                    return "DM4";
                default:
                    return "";
            }
        }

        public static string GetExtension(ImageFormats format)
        {
            switch (format)
            {
                case ImageFormats.MRC:
                    return ".mrc";
                case ImageFormats.MRCS:
                    return ".mrcs";
                case ImageFormats.EM:
                    return ".em";
                case ImageFormats.K2Raw:
                    return ".dat";
                case ImageFormats.FEIRaw:
                    return ".raw";
                case ImageFormats.TIFF:
                    return ".tif";
                case ImageFormats.TIFFF:
                    return ".tiff";
                case ImageFormats.EER:
                    return ".eer";
                case ImageFormats.DM4:
                    return ".dm4";
                default:
                    return "";
            }
        }

        public static MapHeader CreateHeader(ImageFormats format)
        {
            switch (format)
            {
                case ImageFormats.MRC:
                    return new HeaderMRC();
                case ImageFormats.MRCS:
                    return new HeaderMRC();
                case ImageFormats.EM:
                    return new HeaderEM();
                case ImageFormats.K2Raw:
                    return new HeaderRaw(new int3(1, 1, 1), 0, typeof(byte));
                case ImageFormats.FEIRaw:
                    return new HeaderRaw(new int3(1, 1, 1), 49, typeof(int));
                case ImageFormats.TIFF:
                    return new HeaderTiff();
                case ImageFormats.TIFFF:
                    return new HeaderTiff();
                case ImageFormats.EER:
                    return new HeaderEER();
                case ImageFormats.DM4:
                    return new HeaderDM4();
                default:
                    return null;
            }
        }

        public static long SizeOf(Type type)
        {
            if (type == typeof(byte))
                return sizeof(byte);
            else if (type == typeof(ushort) || type == typeof(short))
                return sizeof(short);
            else if (type == typeof(uint) || type == typeof(int))
                return sizeof(int);
            else if (type == typeof(ulong) || type == typeof(long))
                return sizeof(long);
            else if (type == typeof(float))
                return sizeof(float);
            else if (type == typeof(double))
                return sizeof(double);
            else if (type == typeof(Half))
                return sizeof(ushort);
            else
                throw new Exception("Unknown type");
        }

        public static Type StringToType(string name)
        {
            if (name == "int8")
                return typeof (byte);
            else if (name == "int16")
                return typeof(short);
            else if (name == "int32")
                return typeof(int);
            else if (name == "int64")
                return typeof(long);
            else if (name == "float16")
                return typeof(Half);
            else if (name == "float32")
                return typeof(float);
            else if (name == "float64")
                return typeof (double);
            else
                return typeof (float);
        }
    }
}
