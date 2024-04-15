using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp.Headers
{
    public class HeaderRaw : MapHeader
    {
        long OffsetBytes;
        Type ValueType;

        public HeaderRaw(int3 dims, long offsetBytes, Type valueType)
        {
            Dimensions = dims;
            OffsetBytes = offsetBytes;
            ValueType = valueType;
        }

        public override void Write(BinaryWriter writer)
        {
            writer.Write(new byte[OffsetBytes]);
        }

        public override Type GetValueType()
        {
            return ValueType;
        }

        public override void SetValueType(Type t)
        {
            ValueType = t;
        }
    }
}
