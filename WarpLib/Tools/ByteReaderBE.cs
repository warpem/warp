using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class BinaryReaderBE : BinaryReader
    {
        public BinaryReaderBE(System.IO.Stream stream) : base(stream) { }

        public override int ReadInt32()
        {
            byte[] a32 = base.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToInt32(a32, 0);
        }

        public override Int16 ReadInt16()
        {
            byte[] a16 = base.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToInt16(a16, 0);
        }

        public override UInt16 ReadUInt16()
        {
            byte[] a16 = base.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToUInt16(a16, 0);
        }

        public override Int64 ReadInt64()
        {
            byte[] a64 = base.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToInt64(a64, 0);
        }

        public override UInt32 ReadUInt32()
        {
            byte[] a32 = base.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToUInt32(a32, 0);
        }

        public override float ReadSingle()
        {
            byte[] a32 = base.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToSingle(a32, 0);
        }

        public override double ReadDouble()
        {
            byte[] a64 = base.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToDouble(a64, 0);
        }

        // Static duplicates to use with a normal BinaryReader

        public static int ReadInt32(BinaryReader leReader)
        {
            byte[] a32 = leReader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToInt32(a32, 0);
        }

        public static Int16 ReadInt16(BinaryReader leReader)
        {
            byte[] a16 = leReader.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToInt16(a16, 0);
        }

        public static UInt16 ReadUInt16(BinaryReader leReader)
        {
            byte[] a16 = leReader.ReadBytes(2);
            Array.Reverse(a16);
            return BitConverter.ToUInt16(a16, 0);
        }

        public static Int64 ReadInt64(BinaryReader leReader)
        {
            byte[] a64 = leReader.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToInt64(a64, 0);
        }

        public static UInt32 ReadUInt32(BinaryReader leReader)
        {
            byte[] a32 = leReader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToUInt32(a32, 0);
        }

        public static float ReadSingle(BinaryReader leReader)
        {
            byte[] a32 = leReader.ReadBytes(4);
            Array.Reverse(a32);
            return BitConverter.ToSingle(a32, 0);
        }

        public static double ReadDouble(BinaryReader leReader)
        {
            byte[] a64 = leReader.ReadBytes(8);
            Array.Reverse(a64);
            return BitConverter.ToDouble(a64, 0);
        }
    }
}
