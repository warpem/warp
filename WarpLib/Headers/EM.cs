using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp.Headers
{
    public enum EMDataType
    {
        Byte = 1,
        Short = 2,
        ShortComplex = 3,
        Long = 4,
        Single = 5,
        SingleComplex = 8,
        Double = 9,
        DoubleComplex = 10
    }

    public class HeaderEM : MapHeader
    {
        public byte MachineCoding = (byte)6;
        public byte OS9;
        public byte Invalid;
        public EMDataType Mode;

        public byte[] Comment = new byte[80];

        public int Voltage = 300000;
        public float Cs = 2.2f;
        public int Aperture;
        public int Magnification = 50000;
        public float CCDMagnification = 1f;
        public float ExposureTime = 1f;
        public int EMCode;
        public float CCDPixelsize = 1f;
        public float CCDArea = 1f;
        public int Defocus;
        public int Astigmatism;
        public float AstigmatismAngle;
        public float FocusIncrement;
        public float DQE;
        public float C2Intensity;
        public int SlitWidth;
        public int EnergyOffset;
        public float TiltAngle;
        public float TiltAxis;
        public int NoName1;
        public int NoName2;
        public int NoName3;
        public int2 MarkerPosition;
        public int Resolution;
        public int Density;
        public int Contrast;
        public int NoName4;
        public int3 CenterOfMass;
        public int Height;
        public int NoName5;
        public int DreiStrahlBereich;
        public int AchromaticRing;
        public int Lambda;
        public int DeltaTheta;
        public int NoName6;
        public int NoName7;

        byte[] UserData = new byte[256];

        public HeaderEM()
        { }

        public HeaderEM(BinaryReader reader)
        {
            MachineCoding = reader.ReadByte();
            OS9 = reader.ReadByte();
            Invalid = reader.ReadByte();
            Mode = (EMDataType)reader.ReadByte();

            Dimensions = new int3(reader.ReadBytes(3 * sizeof(int)));

            Comment = reader.ReadBytes(80);

            Voltage = reader.ReadInt32();
            Cs = (float)reader.ReadInt32() / 1000f;
            Aperture = reader.ReadInt32();
            Magnification = reader.ReadInt32();
            CCDMagnification = (float)reader.ReadInt32() / 1000f;
            ExposureTime = (float)reader.ReadInt32() / 1000f;
            PixelSize = new float3(reader.ReadInt32() / 1000f);
            EMCode = reader.ReadInt32();
            CCDPixelsize = (float)reader.ReadInt32() / 1000f;
            CCDArea = (float)reader.ReadInt32() / 1000f;
            Defocus = reader.ReadInt32();
            Astigmatism = reader.ReadInt32();
            AstigmatismAngle = (float)reader.ReadInt32() / 1000f;
            FocusIncrement = (float)reader.ReadInt32() / 1000f;
            DQE = (float)reader.ReadInt32() / 1000f;
            C2Intensity = (float)reader.ReadInt32() / 1000f;
            SlitWidth = reader.ReadInt32();
            EnergyOffset = reader.ReadInt32();
            TiltAngle = (float)reader.ReadInt32() / 1000f;
            TiltAxis = (float)reader.ReadInt32() / 1000f;
            NoName1 = reader.ReadInt32();
            NoName2 = reader.ReadInt32();
            NoName3 = reader.ReadInt32();
            MarkerPosition = new int2(reader.ReadBytes(2 * sizeof(int)));
            Resolution = reader.ReadInt32();
            Density = reader.ReadInt32();
            Contrast = reader.ReadInt32();
            NoName4 = reader.ReadInt32();
            CenterOfMass = new int3(reader.ReadBytes(3 * sizeof(int)));
            Height = reader.ReadInt32();
            NoName5 = reader.ReadInt32();
            DreiStrahlBereich = reader.ReadInt32();
            AchromaticRing = reader.ReadInt32();
            Lambda = reader.ReadInt32();
            DeltaTheta = reader.ReadInt32();
            NoName6 = reader.ReadInt32();
            NoName7 = reader.ReadInt32();

            UserData = reader.ReadBytes(256);
        }

        public override void Write(BinaryWriter writer)
        {
            writer.Write(MachineCoding);
            writer.Write(OS9);
            writer.Write(Invalid);
            writer.Write((byte)Mode);

            writer.Write(Dimensions);

            writer.Write(Comment);

            writer.Write(Voltage);
            writer.Write((int)(Cs * 1000f));
            writer.Write(Aperture);
            writer.Write(Magnification);
            writer.Write((int)(CCDMagnification * 1000f));
            writer.Write((int)(ExposureTime * 1000f));
            writer.Write((int)(PixelSize.X * 1000f));
            writer.Write(EMCode);
            writer.Write((int)(CCDPixelsize * 1000f));
            writer.Write((int)(CCDArea * 1000f));
            writer.Write(Defocus);
            writer.Write(Astigmatism);
            writer.Write((int)(AstigmatismAngle * 1000f));
            writer.Write((int)(FocusIncrement * 1000f));
            writer.Write((int)(DQE * 1000f));
            writer.Write((int)(C2Intensity * 1000f));
            writer.Write(SlitWidth);
            writer.Write(EnergyOffset);
            writer.Write((int)(TiltAngle * 1000f));
            writer.Write((int)(TiltAxis * 1000f));
            writer.Write(NoName1);
            writer.Write(NoName2);
            writer.Write(NoName3);
            writer.Write(MarkerPosition);
            writer.Write(Resolution);
            writer.Write(Density);
            writer.Write(Contrast);
            writer.Write(NoName4);
            writer.Write(CenterOfMass);
            writer.Write(Height);
            writer.Write(NoName5);
            writer.Write(DreiStrahlBereich);
            writer.Write(AchromaticRing);
            writer.Write(Lambda);
            writer.Write(DeltaTheta);
            writer.Write(NoName6);
            writer.Write(NoName7);

            writer.Write(UserData);
        }

        public override Type GetValueType()
        {
            switch (Mode)
            {
                case EMDataType.Byte:
                    return typeof(byte);
                case EMDataType.Double:
                    return typeof(double);
                case EMDataType.DoubleComplex:
                    return typeof(double);
                case EMDataType.Long:
                    return typeof(int);
                case EMDataType.Short:
                    return typeof(short);
                case EMDataType.ShortComplex:
                    return typeof(short);
                case EMDataType.Single:
                    return typeof(float);
                case EMDataType.SingleComplex:
                    return typeof(float);
            }

            throw new Exception("Unknown data type.");
        }

        public override void SetValueType(Type t)
        {
            if (t == typeof(byte))
                Mode = EMDataType.Byte;
            else if (t == typeof(float))
                Mode = EMDataType.Single;
            else if (t == typeof(double))
                Mode = EMDataType.Double;
            else if (t == typeof(short))
                Mode = EMDataType.Short;
            else if (t == typeof(int))
                Mode = EMDataType.Long;
            else
                throw new Exception("Unknown data type.");
        }
    }
}
