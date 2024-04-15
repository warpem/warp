using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp.Headers
{
    // Code adapted from BioFormats' DM3/4 reader written in Java
    // https://github.com/openmicroscopy/bioformats/blob/develop/components/formats-gpl/src/loci/formats/in/GatanReader.java

    public class HeaderDM4 : MapHeader
    {
        public const int DM3_MAGIC_BYTES = 3;
        public const int DM4_MAGIC_BYTES = 4;

        /** Tag types. */
        private const int GROUP = 20;
        private const int VALUE = 21;

        /** Data types. */
        private const int ARRAY = 15;
        private const int SHORT = 2;
        private const int USHORT = 4;
        private const int INT = 3;
        private const int UINT = 5;
        private const int FLOAT = 6;
        private const int DOUBLE = 7;
        private const int BYTE = 8;
        private const int UBYTE = 9;
        private const int CHAR = 10;
        private const int UNKNOWN = 11;
        private const int UNKNOWN2 = 12;

        /** Shape types */
        private const int LINE = 2;
        private const int RECTANGLE = 5;
        private const int ELLIPSE = 6;
        private const int TEXT = 13;

        // -- Fields --

        private BinaryReaderMixedEndian Reader;

        private Type ValueType;

        /** Offset to pixel data. */
        private long pixelOffset;

        /** List of pixel sizes. */
        private List<Double> pixelSizes;
        private List<String> units;

        private long numPixelBytes;

        private bool signed;
        private long timestamp;
        private double gamma, mag, voltage;
        private String info;

        private double posX, posY, posZ;
        private double sampleTime;

        private bool littleEndian = false;
        private bool adjustEndianness = true;
        private int version;

        public HeaderDM4()
        {

        }

        public HeaderDM4(BinaryReader reader)
        {
            Reader = new BinaryReaderMixedEndian(reader);

            pixelOffset = 0;

            //LOGGER.info("Verifying Gatan format");

            littleEndian = false;
            pixelSizes = new List<Double>();
            units = new List<String>();

            Reader.SetEndian(littleEndian);

            // only support version 3
            version = Reader.ReadInt32();
            if (version != 3 && version != 4)
            {
                throw new FormatException("invalid header");
            }

            //LOGGER.info("Reading tags");

            Reader.ReadBytes(4);
            skipPadding(Reader);
            littleEndian = Reader.ReadInt32() != 1;
            Reader.SetEndian(littleEndian);

            // TagGroup instance

            Reader.ReadBytes(2);
            skipPadding(Reader);
            int numTags = Reader.ReadInt32();
            if (numTags > Reader.BaseStream.Length)
            {
                littleEndian = !littleEndian;
                Reader.SetEndian(littleEndian);
                adjustEndianness = false;
            }
            //LOGGER.debug("tags ({}) {", numTags);
            try
            {
                parseTags(Reader, numTags, null, "  ");
            }
            catch (Exception e)
            {
                throw new FormatException("Unable to parse metadata tag", e);
            }
            //LOGGER.debug("}");

            //LOGGER.info("Populating metadata");

            littleEndian = true;

            if (Dimensions.X == 0 || Dimensions.Y == 0)
            {
                throw new FormatException("Dimensions information not found");
            }

            if (Dimensions.Z == 0)
            {
                Dimensions.Z = 1;
            }
            //m.sizeC = 1;
            //m.sizeT = 1;
            //m.dimensionOrder = "XYZTC";
            //m.imageCount = getSizeZ() * getSizeC() * getSizeT();

            int bytes = (int)(numPixelBytes / Dimensions.Elements());
            if (bytes != ImageFormatsHelper.SizeOf(ValueType))
            {
                throw new Exception("Bytes per pixel does not match the actual size of image data.");
            }

            reader.BaseStream.Seek(pixelOffset, SeekOrigin.Begin);
        }

        private void parseTags(BinaryReaderMixedEndian reader, int numTags, string parent, string indent)
        {
            for (int i = 0; i < numTags; i++)
            {
                if (reader.BaseStream.Position + 3 >= reader.BaseStream.Length) break;

                byte type = reader.ReadByte(); // can be 21 (data) or 20 (tag group)
                int length = reader.ReadInt16();

                // image data is in tag with type 21 and label 'Data'
                // image dimensions are in type 20 tag with 2 type 15 tags
                // bytes per pixel is in type 21 tag with label 'PixelDepth'

                String labelString = null;
                String value = null;

                if (type == VALUE)
                {
                    labelString = FromByteArray(reader, length);
                    skipPadding(reader);
                    skipPadding(reader);
                    int skip = reader.ReadInt32(); // equal to '%%%%' / 623191333
                    skipPadding(reader);
                    int n = reader.ReadInt32();
                    skipPadding(reader);
                    int dataType = reader.ReadInt32();
                    string sb = labelString;
                    if (sb.Length > 32)
                    {
                        sb = sb.Substring(0, 20) + "... (" + sb.Length + ")";
                    }

                    //LOGGER.debug("{}{}: n={}, dataType={}, label={}",
                    //             new Object[]
                    //             {
                    //                 indent,
                    //                 i,
                    //                 n,
                    //                 dataType,
                    //                 sb
                    //             });
                    if (skip != 623191333)
                        Debug.WriteLine("Skip mismatch: {}", skip);

                    if (n == 1)
                    {
                        if ("Dimensions".Equals(parent) && labelString.Length == 0)
                        {
                            if (adjustEndianness) Reader.SetEndian(!Reader.IsLittleEndian);
                            if (i == 0)
                            {
                                Dimensions.X = reader.ReadInt32();
                            }
                            else if (i == 1)
                            {
                                Dimensions.Y = reader.ReadInt32();
                            }
                            else if (i == 2)
                            {
                                Dimensions.Z = reader.ReadInt32();
                            }

                            if (adjustEndianness) Reader.SetEndian(!Reader.IsLittleEndian);
                        }
                        else
                        {
                            value = readValue(reader, dataType).ToString(CultureInfo.InvariantCulture);
                        }
                    }
                    else if (n == 2)
                    {
                        if (dataType == 18)
                        {
                            // this should always be true
                            length = reader.ReadInt32();
                        }
                        else
                        {
                            //LOGGER.warn("dataType mismatch: {}", dataType);
                        }

                        value = FromByteArray(reader, length);
                    }
                    else if (n == 3)
                    {
                        if (dataType == GROUP)
                        {
                            // this should always be true
                            skipPadding(reader);
                            dataType = reader.ReadInt32();
                            long dataLength = 0;
                            if (version == 4)
                            {
                                dataLength = reader.ReadInt64();
                            }
                            else
                            {
                                dataLength = reader.ReadInt32();
                            }

                            length = (int)(dataLength & 0xffffffff);
                            if (labelString.Equals("Data"))
                            {
                                if (dataLength > 0)
                                {
                                    pixelOffset = reader.BaseStream.Position;
                                    reader.BaseStream.Seek(reader.BaseStream.Position + getNumBytes(dataType) * dataLength, SeekOrigin.Begin);
                                    numPixelBytes = reader.BaseStream.Position - pixelOffset;
                                }
                            }
                            else
                            {
                                if (dataType == 10) reader.ReadBytes(length);
                                else value = FromByteArray(reader, length * 2);
                            }
                        }
                        else
                        {
                            //LOGGER.warn("dataType mismatch: {}", dataType);
                        }
                    }
                    else
                    {
                        // this is a normal struct of simple types
                        if (dataType == ARRAY)
                        {
                            reader.ReadBytes(4);
                            skipPadding(reader);
                            skipPadding(reader);
                            int numFields = reader.ReadInt32();
                            long startFP = reader.BaseStream.Position;
                            StringBuilder s = new StringBuilder();
                            reader.ReadBytes(4);
                            skipPadding(reader);
                            long baseFP = reader.BaseStream.Position;
                            if (version == 4)
                            {
                                baseFP += 4;
                            }

                            int width = version == 4 ? 16 : 8;
                            for (int j = 0; j < numFields; j++)
                            {
                                reader.BaseStream.Seek(baseFP + j * width, SeekOrigin.Begin);
                                dataType = reader.ReadInt32();
                                    reader.BaseStream.Seek(startFP + numFields * width + j * getNumBytes(dataType), SeekOrigin.Begin);
                                s.Append(readValue(reader, dataType));
                                if (j < numFields - 1) s.Append(", ");
                            }

                            value = s.ToString();
                        }
                        else if (dataType == GROUP)
                        {
                            // this is an array of structs
                            skipPadding(reader);
                            dataType = reader.ReadInt32();
                            if (dataType == ARRAY)
                            {
                                // should always be true
                                reader.ReadBytes(4);
                                skipPadding(reader);
                                skipPadding(reader);
                                int numFields = reader.ReadInt32();
                                int[] dataTypes = new int[numFields];
                                long baseFP = reader.BaseStream.Position + 12;
                                for (int j = 0; j < numFields; j++)
                                {
                                    reader.ReadBytes(4);
                                    if (version == 4)
                                    {
                                        reader.BaseStream.Seek(baseFP + j * 16, SeekOrigin.Begin);
                                    }

                                    dataTypes[j] = reader.ReadInt32();
                                }

                                skipPadding(reader);
                                int len = reader.ReadInt32();

                                double[][] values = Helper.ArrayOfFunction(a => new double[len], numFields);

                                for (int k = 0; k < len; k++)
                                {
                                    for (int q = 0; q < numFields; q++)
                                    {
                                        values[q][k] = readValue(reader, dataTypes[q]);
                                    }
                                }
                            }
                            else
                            {
                                //LOGGER.warn("dataType mismatch: {}", dataType);
                            }
                        }
                    }
                }
                else if (type == GROUP)
                {
                    labelString = FromByteArray(reader, length);
                    reader.ReadBytes(2);
                    skipPadding(reader);
                    skipPadding(reader);
                    skipPadding(reader);
                    int num = reader.ReadInt32();
                    //LOGGER.debug("{}{}: group({}) {} {", new Object[] { indent, i, num, labelString });
                    parseTags(reader, num, string.IsNullOrEmpty(labelString) ? parent : labelString, indent + "  ");
                    //LOGGER.debug("{}}", indent);
                }
                else
                {
                    //LOGGER.debug("{}{}: unknown type: {}", new Object[] { indent, i, type });
                }

                if (value != null)
                {
                    bool validPhysicalSize = parent != null && (parent.Equals("Dimension") ||
                                                                   ((pixelSizes.Count() == 4 || units.Count() == 4) && parent.Equals("2")));
                    if (labelString.Equals("Scale") && validPhysicalSize)
                    {
                        if (value.IndexOf(',') == -1)
                        {
                            pixelSizes.Add(double.Parse(value, CultureInfo.InvariantCulture));
                        }
                    }
                    else if (labelString.Equals("Units") && validPhysicalSize)
                    {
                        // make sure that we don't add more units than sizes
                        if (pixelSizes.Count() == units.Count() + 1)
                        {
                            units.Add(value);
                        }
                    }
                    else if (labelString.Equals("LowLimit"))
                    {
                        signed = double.Parse(value, CultureInfo.InvariantCulture) < 0;
                    }
                    else if (labelString.Equals("Acquisition Start Time (epoch)"))
                    {
                        timestamp = long.Parse(value, CultureInfo.InvariantCulture);
                    }
                    else if (labelString.Equals("Voltage"))
                    {
                        voltage = double.Parse(value, CultureInfo.InvariantCulture);
                    }
                    else if (labelString.Equals("Microscope Info")) info = value;
                    else if (labelString.Equals("Indicated Magnification"))
                    {
                        mag = double.Parse(value, CultureInfo.InvariantCulture);
                    }
                    else if (labelString.Equals("Gamma"))
                    {
                        gamma = double.Parse(value, CultureInfo.InvariantCulture);
                    }
                    else if (labelString.StartsWith("xPos"))
                    {
                        double number = double.Parse(value, CultureInfo.InvariantCulture);
                        posX = number;
                    }
                    else if (labelString.StartsWith("yPos"))
                    {
                        double number = double.Parse(value, CultureInfo.InvariantCulture);
                        posY = number;
                    }
                    else if (labelString.StartsWith("Specimen position"))
                    {
                        double number = double.Parse(value, CultureInfo.InvariantCulture);
                        posZ = number;
                    }
                    else if (labelString == "Sample Time")
                    {
                        sampleTime = double.Parse(value, CultureInfo.InvariantCulture);
                    }
                    else if (labelString == "DataType")
                    {
                        int pixelType = int.Parse(value);
                        switch (pixelType)
                        {
                            case 1:
                                ValueType = typeof(short);
                                break;
                            case 10:
                                ValueType = typeof(ushort);
                                break;
                            case 2:
                                ValueType = typeof(float);
                                break;
                            case 12:
                                ValueType = typeof(double);
                                break;
                            case 9:
                                ValueType = typeof(byte);
                                break;
                            case 6:
                                ValueType = typeof(byte);
                                break;
                            case 7:
                                ValueType = typeof(int);
                                break;
                            case 11:
                                ValueType = typeof(uint);
                                break;
                        }
                    }

                    value = null;
                }
            }
        }

        private string FromByteArray(BinaryReaderMixedEndian reader, int n)
        {
            byte[] Bytes = reader.ReadBytes(n);
            return System.Text.Encoding.UTF8.GetString(Bytes);
        }

        private double readValue(BinaryReaderMixedEndian reader, int type)
        {
            switch (type)
            {
                case SHORT:
                case USHORT:
                    return reader.ReadInt16();
                case INT:
                case UINT:
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    int i = reader.ReadInt32();
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    return i;
                case FLOAT:
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    float f = reader.ReadSingle();
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    return f;
                case DOUBLE:
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    double dbl = reader.ReadDouble();
                    if (adjustEndianness) reader.SetEndian(!reader.IsLittleEndian);
                    return dbl;
                case BYTE:
                case UBYTE:
                case CHAR:
                    return reader.ReadByte();
                case UNKNOWN:
                case UNKNOWN2:
                    return reader.ReadInt64();
            }

            return 0;
        }

        private int getNumBytes(int type)
        {
            switch (type)
            {
                case SHORT:
                case USHORT:
                    return 2;
                case INT:
                case UINT:
                case FLOAT:
                    return 4;
                case DOUBLE:
                    return 8;
                case BYTE:
                case UBYTE:
                case CHAR:
                    return 1;
            }
            return 0;
        }

        private void skipPadding(BinaryReaderMixedEndian reader)
        {
            if (version == 4)
                reader.ReadBytes(4);
        }

        public override void Write(BinaryWriter writer)
        {
            throw new NotImplementedException();
        }

        public override Type GetValueType()
        {
            return ValueType;
        }

        public override void SetValueType(Type t)
        {
            throw new NotImplementedException();
        }
    }

    public class BinaryReaderMixedEndian
    {
        private BinaryReader Reader;
        public bool IsLittleEndian = true;

        public Stream BaseStream => Reader.BaseStream;

        public BinaryReaderMixedEndian(BinaryReader reader)
        {
            Reader = reader;
        }

        public void SetEndian(bool isLittle)
        {
            IsLittleEndian = isLittle;
        }

        public Int16 ReadInt16()
        {
            if (IsLittleEndian)
                return Reader.ReadInt16();

            byte[] a16 = Reader.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToInt16(a16, 0);
        }

        public UInt16 ReadUInt16()
        {
            if (IsLittleEndian)
                return Reader.ReadUInt16();

            byte[] a16 = Reader.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToUInt16(a16, 0);
        }

        public int ReadInt32()
        {
            if (IsLittleEndian)
                return Reader.ReadInt32();

            byte[] a32 = Reader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToInt32(a32, 0);
        }

        public UInt32 ReadUInt32()
        {
            if (IsLittleEndian)
                return Reader.ReadUInt32();

            byte[] a32 = Reader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToUInt32(a32, 0);
        }

        public Int64 ReadInt64()
        {
            if (IsLittleEndian)
                return Reader.ReadInt64();

            byte[] a64 = Reader.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToInt64(a64, 0);
        }

        public UInt64 ReadUInt64()
        {
            if (IsLittleEndian)
                return Reader.ReadUInt64();

            byte[] a64 = Reader.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToUInt64(a64, 0);
        }

        public float ReadSingle()
        {
            if (IsLittleEndian)
                return Reader.ReadSingle();

            byte[] a32 = Reader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToSingle(a32, 0);
        }

        public double ReadDouble()
        {
            if (IsLittleEndian)
                return Reader.ReadDouble();

            byte[] a64 = Reader.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToDouble(a64, 0);
        }

        public byte ReadByte()
        {
            return Reader.ReadByte();
        }

        public byte[] ReadBytes(int n)
        {
            return Reader.ReadBytes(n);
        }
    }
}
