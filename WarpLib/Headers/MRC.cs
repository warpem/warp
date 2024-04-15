using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp.Headers
{
    public enum MRCDataType
    {
        Byte = 0,
        Short = 1,
        Float = 2,
        ShortComplex = 3,
        FloatComplex = 4,
        UnsignedShort = 6,
        Half = 12,
        RGB = 16
    }

    public class HeaderMRC : MapHeader
    {
        public MRCDataType Mode = MRCDataType.Float;
        public int3 StartSubImage = new int3(0, 0, 0);
        public int3 Griddimensions = new int3(1, 1, 1);
        public float3 Angles = new float3(90, 90, 90);
        public int3 MapOrder = new int3(1, 2, 3);

        public float MinValue;
        public float MaxValue;
        public float MeanValue;
        public int SpaceGroup;

        public int ExtendedBytes;
        public short CreatorID;

        public byte[] ExtraData10 = new byte[6];
        public char[] ExtID = new char[4];
        public byte[] ExtraData11 = new byte[20];

        public short NInt;
        public short NReal;

        public byte[] ExtraData2 = new byte[28];

        public short IDType;
        public short Lens;
        public short ND1;
        public short ND2;
        public short VD1;
        public short VD2;

        public float3 TiltOriginal;
        public float3 TiltCurrent;
        public float3 Origin;

        public char[] CMap = new char[] { 'M', 'A', 'P', ' ' };
        public byte[] Stamp = new byte[] { 68, 65, 0, 0 };

        public float StdDevValue;

        public int NumLabels;
        public char[][] Labels = new char[10][];

        public byte[] Extended;

        public bool ImodHasTilt;
        private bool ImodHasMontage;
        private bool ImodHasStagePos;
        public bool ImodHasMagnification;
        public bool ImodHasIntensity;
        public bool ImodHasExposure;

        public float ImodRotation;
        public float[] ImodTilt;
        public float[] ImodMagnification;
        public float[] ImodIntensity;
        public float[] ImodExposure;

        public HeaderMRC()
        {
            for (int i = 0; i < Labels.Length; i++)
                Labels[i] = new char[80];
        }

        public HeaderMRC(BinaryReader reader)
        {
            Dimensions = new int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
            Mode = (MRCDataType)reader.ReadInt32();
            StartSubImage = new int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
            Griddimensions = new int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
            PixelSize = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()) / new float3(Dimensions);
            Angles = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            MapOrder = new int3(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());

            MinValue = reader.ReadSingle();
            MaxValue = reader.ReadSingle();
            MeanValue = reader.ReadSingle();
            SpaceGroup = reader.ReadInt32();

            ExtendedBytes = reader.ReadInt32();
            CreatorID = reader.ReadInt16();

            ExtraData10 = reader.ReadBytes(6);
            ExtID = reader.ReadChars(4);
            ExtraData11 = reader.ReadBytes(20);

            NInt = reader.ReadInt16();
            NReal = reader.ReadInt16();

            ExtraData2 = reader.ReadBytes(28);

            IDType = reader.ReadInt16();
            Lens = reader.ReadInt16();
            ND1 = reader.ReadInt16();
            ND2 = reader.ReadInt16();
            VD1 = reader.ReadInt16();
            VD2 = reader.ReadInt16();

            TiltOriginal = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            TiltCurrent = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            Origin = new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

            CMap = reader.ReadChars(4);
            Stamp = reader.ReadBytes(4);

            StdDevValue = reader.ReadSingle();

            NumLabels = reader.ReadInt32();
            for (int i = 0; i < 10; i++)
            {
                byte[] Bytes = reader.ReadBytes(80);
                try
                {
                    Labels[i] = ASCIIEncoding.ASCII.GetString(Bytes).ToCharArray();
                }
                catch
                {
                    Labels[i] = new char[80];
                }
            }

            // In case this is from SerialEM, check how big ExtendedBytes should be at least to read per-frame data
            ImodHasTilt = (NReal & (1 << 0)) > 0;
            ImodHasMontage = (NReal & (1 << 1)) > 0;
            ImodHasStagePos = (NReal & (1 << 2)) > 0;
            ImodHasMagnification = (NReal & (1 << 3)) > 0;
            ImodHasIntensity = (NReal & (1 << 4)) > 0;
            ImodHasExposure = (NReal & (1 << 5)) > 0;

            int BytesPerSection = (ImodHasTilt ? 2 : 0) +
                                  (ImodHasMontage ? 6 : 0) +
                                  (ImodHasStagePos ? 4 : 0) +
                                  (ImodHasMagnification ? 2 : 0) +
                                  (ImodHasIntensity ? 2 : 0) +
                                  (ImodHasExposure ? 4 : 0);

            if (BytesPerSection * Dimensions.Z > ExtendedBytes) // Not from SerialEM, ignore extended header
            {
                Extended = reader.ReadBytes(ExtendedBytes);
            }
            else    // SerialEM extended header, read one section per frame
            {
                if (ImodHasTilt)
                    ImodTilt = new float[Dimensions.Z];
                if (ImodHasMagnification)
                    ImodMagnification = new float[Dimensions.Z];
                if (ImodHasIntensity)
                    ImodIntensity = new float[Dimensions.Z];
                if (ImodHasExposure)
                    ImodExposure = new float[Dimensions.Z];

                for (int i = 0; i < Dimensions.Z; i++)
                {
                    if (ImodHasTilt)
                        ImodTilt[i] = reader.ReadInt16() / 100f;
                    if (ImodHasMontage)
                        reader.ReadBytes(6);
                    if (ImodHasStagePos)
                        reader.ReadBytes(4);
                    if (ImodHasMagnification)
                        ImodMagnification[i] = reader.ReadInt16() / 100f;
                    if (ImodHasIntensity)
                        ImodIntensity[i] = reader.ReadInt16() / 2500f;
                    if (ImodHasExposure)
                    {
                        float val = reader.ReadSingle();
                        /*int s1 = reader.ReadInt16();
                        int s2 = reader.ReadInt16();

                        float val = Math.Sign(s1) * (Math.Abs(s1) * 256 + (Math.Abs(s2) % 256)) * (float)Math.Pow(2, Math.Sign(s2) * (Math.Abs(s2) / 256f));*/
                        ImodExposure[i] = val;
                    }
                }
            }

            reader.BaseStream.Seek(1024 + ExtendedBytes, SeekOrigin.Begin);
        }

        public override void Write(BinaryWriter writer)
        {
            writer.Write(Dimensions);
            writer.Write((int)Mode);
            writer.Write(StartSubImage);
            writer.Write(Dimensions);
            writer.Write(PixelSize * new float3(Dimensions));
            writer.Write(Angles);
            writer.Write(MapOrder);

            writer.Write(MinValue);
            writer.Write(MaxValue);
            writer.Write(MeanValue);
            writer.Write(SpaceGroup);

            if (Extended != null)
                writer.Write(Extended.Length);
            else if (ImodTilt != null)
                writer.Write(1024 * 128);
            else
                writer.Write(0);
            writer.Write(CreatorID);

            if (ImodTilt != null)
                ExtID = new[] { 'F', 'E', 'I', '1' };

            writer.Write(ExtraData10);
            writer.Write(ExtID);
            writer.Write(ExtraData11);

            writer.Write(NInt);

            // If there are tilt angles available, tell IMOD to expect them in the extended header
            //NReal = ImodTilt == null ? (short)0 : (short)1;
            writer.Write(NReal);

            writer.Write(ExtraData2);

            writer.Write(IDType);
            writer.Write(Lens);
            writer.Write(ND1);
            writer.Write(ND2);
            writer.Write(VD1);
            writer.Write(VD2);

            writer.Write(TiltOriginal);
            writer.Write(TiltCurrent);
            writer.Write(Origin);

            writer.Write(CMap);
            writer.Write(Stamp);

            writer.Write(StdDevValue);

            writer.Write(NumLabels);
            for (int i = 0; i < Labels.Length; i++)
                writer.Write(Labels[i]);

            if (Extended != null)
                writer.Write(Extended);

            if (ImodTilt != null)
            {
                for (int i = 0; i < 1024; i++)
                {
                    if (i < ImodTilt.Length)
                    {
                        writer.Write(ImodTilt[i]);
                        writer.Write(ImodRotation);

                        writer.Write(new byte[8 * 4]);

                        writer.Write(ImodRotation);
                        writer.Write(PixelSize.X);

                        writer.Write(new byte[80]);
                    }
                    else
                    {
                        writer.Write(new byte[128]);
                    }
                }
            }
        }

        public override Type GetValueType()
        {
            switch (Mode)
            {
                case MRCDataType.Byte:
                    return typeof(byte);
                case MRCDataType.Float:
                    return typeof(float);
                case MRCDataType.FloatComplex:
                    return typeof(float);
                case MRCDataType.RGB:
                    return typeof(byte);
                case MRCDataType.Short:
                    return typeof(short);
                case MRCDataType.ShortComplex:
                    return typeof(short);
                case MRCDataType.UnsignedShort:
                    return typeof(ushort);
                case MRCDataType.Half:
                    return typeof(Half);
            }

            throw new Exception("Unknown data type.");
        }

        public override void SetValueType(Type t)
        {
            if (t == typeof(byte))
                Mode = MRCDataType.Byte;
            else if (t == typeof(float))
                Mode = MRCDataType.Float;
            else if (t == typeof(short))
                Mode = MRCDataType.Short;
            else if (t == typeof(ushort))
                Mode = MRCDataType.UnsignedShort;
            else if (t == typeof(Half))
                Mode = MRCDataType.Half;
            else
                throw new Exception("Unknown data type.");
        }

        public static long GetHeaderSize(string path)
        {
            long Position;

            using (BinaryReader Reader = new BinaryReader(File.OpenRead(path)))
            {
                HeaderMRC Header = new HeaderMRC(Reader);
                Position = Reader.BaseStream.Position;
            }

            return Position;
        }
    }
}
