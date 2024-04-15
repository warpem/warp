using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp
{
    public class LinearGrid4D
    {
        public readonly int4 Dimensions;
        public readonly DimensionSets DimensionSet;
        public float[] Values;

        private readonly float4 DimensionsFloat;

        public LinearGrid4D(int4 dimensions, float[] values)
        {
            Dimensions = dimensions;
            Values = values;

            DimensionsFloat = new float4(dimensions);
        }

        public LinearGrid4D(int4 dimensions) : this(dimensions, new float[dimensions.Elements()]) { }

        public float[] GetInterpolatedOld(float4[] coords)
        {
            float[] Result = new float[coords.Length];

            for (int i = 0; i < coords.Length; i++)
                Result[i] = GetInterpolatedOld(coords[i]);

            return Result;
        }

        public float[] GetInterpolated(float4[] coords)
        {
            float[] Result = new float[coords.Length];
            CPU.EvalLinear4Batch(Dimensions, Values, coords, coords.Length, Result);

            return Result;
        }

        public Span<float> GetInterpolated(Span<float4> coords, Span<float> result)
        {
            CPU.EvalLinear4Batch(Dimensions, Values, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));

            return result;
        }

        public float GetInterpolated(float4 coords)
        {
            return CPU.EvalLinear4(Dimensions, Values, coords);
        }

        public float GetInterpolatedOld(float4 coords)
        {
            coords *= DimensionsFloat - 1;

            int4 Pos0 = new int4(Math.Max(0, Math.Min((int)coords.X, Dimensions.X - 1)),
                                 Math.Max(0, Math.Min((int)coords.Y, Dimensions.Y - 1)),
                                 Math.Max(0, Math.Min((int)coords.Z, Dimensions.Z - 1)),
                                 Math.Max(0, Math.Min((int)coords.W, Dimensions.W - 1)));

            int4 Pos1 = new int4(Math.Min(Pos0.X + 1, Dimensions.X - 1),
                                 Math.Min(Pos0.Y + 1, Dimensions.Y - 1),
                                 Math.Min(Pos0.Z + 1, Dimensions.Z - 1),
                                 Math.Min(Pos0.W + 1, Dimensions.W - 1));

            coords -= new float4(Pos0);

            float C0000 = Values[(((Pos0.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos0.X)];
            float C0001 = Values[(((Pos0.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos1.X)];
            float C0010 = Values[(((Pos0.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos0.X)];
            float C0011 = Values[(((Pos0.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos1.X)];
            float C0100 = Values[(((Pos0.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos0.X)];
            float C0101 = Values[(((Pos0.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos1.X)];
            float C0110 = Values[(((Pos0.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos0.X)];
            float C0111 = Values[(((Pos0.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos1.X)];

            float C1000 = Values[(((Pos1.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos0.X)];
            float C1001 = Values[(((Pos1.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos1.X)];
            float C1010 = Values[(((Pos1.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos0.X)];
            float C1011 = Values[(((Pos1.W * Dimensions.Z + Pos0.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos1.X)];
            float C1100 = Values[(((Pos1.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos0.X)];
            float C1101 = Values[(((Pos1.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos0.Y) * Dimensions.X + Pos1.X)];
            float C1110 = Values[(((Pos1.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos0.X)];
            float C1111 = Values[(((Pos1.W * Dimensions.Z + Pos1.Z) * Dimensions.Y + Pos1.Y) * Dimensions.X + Pos1.X)];


            float C000 = MathHelper.Lerp(C0000, C0001, coords.X);
            float C001 = MathHelper.Lerp(C0010, C0011, coords.X);
            float C010 = MathHelper.Lerp(C0100, C0101, coords.X);
            float C011 = MathHelper.Lerp(C0110, C0111, coords.X);

            float C100 = MathHelper.Lerp(C1000, C1001, coords.X);
            float C101 = MathHelper.Lerp(C1010, C1011, coords.X);
            float C110 = MathHelper.Lerp(C1100, C1101, coords.X);
            float C111 = MathHelper.Lerp(C1110, C1111, coords.X);


            float C00 = MathHelper.Lerp(C000, C001, coords.Y);
            float C01 = MathHelper.Lerp(C010, C011, coords.Y);

            float C10 = MathHelper.Lerp(C100, C101, coords.Y);
            float C11 = MathHelper.Lerp(C110, C111, coords.Y);


            float C0 = MathHelper.Lerp(C00, C01, coords.Z);

            float C1 = MathHelper.Lerp(C10, C11, coords.Z);


            return MathHelper.Lerp(C0, C1, coords.W);
        }

        public LinearGrid4D Resize(int4 newSize)
        {
            float[] Result = new float[newSize.Elements()];

            float StepX = 1f / Math.Max(1, newSize.X - 1);
            float StepY = 1f / Math.Max(1, newSize.Y - 1);
            float StepZ = 1f / Math.Max(1, newSize.Z - 1);
            float StepW = 1f / Math.Max(1, newSize.W - 1);

            for (int w = 0, i = 0; w < newSize.W; w++)
                for (int z = 0; z < newSize.Z; z++)
                    for (int y = 0; y < newSize.Y; y++)
                        for (int x = 0; x < newSize.X; x++, i++)
                            Result[i] = GetInterpolated(new float4(x * StepX, y * StepY, z * StepZ, w * StepW));

            return new LinearGrid4D(newSize, Result);
        }

        public void Save(XmlTextWriter writer)
        {
            XMLHelper.WriteAttribute(writer, "Width", Dimensions.X);
            XMLHelper.WriteAttribute(writer, "Height", Dimensions.Y);
            XMLHelper.WriteAttribute(writer, "Depth", Dimensions.Z);
            XMLHelper.WriteAttribute(writer, "Duration", Dimensions.W);

            for (int w = 0; w < Dimensions.W; w++)
                for (int z = 0; z < Dimensions.Z; z++)
                    for (int y = 0; y < Dimensions.Y; y++)
                        for (int x = 0; x < Dimensions.X; x++)
                        {
                            writer.WriteStartElement("Node");
                            XMLHelper.WriteAttribute(writer, "X", x);
                            XMLHelper.WriteAttribute(writer, "Y", y);
                            XMLHelper.WriteAttribute(writer, "Z", z);
                            XMLHelper.WriteAttribute(writer, "W", w);
                            XMLHelper.WriteAttribute(writer, "Value", Values[((w * Dimensions.Z + z) * Dimensions.Y + y) * Dimensions.X + x]);
                            writer.WriteEndElement();
                        }
        }

        public static LinearGrid4D Load(XPathNavigator nav)
        {
            int4 Dimensions = new int4(XMLHelper.LoadAttribute(nav, "Width", 1),
                                       XMLHelper.LoadAttribute(nav, "Height", 1),
                                       XMLHelper.LoadAttribute(nav, "Depth", 1),
                                       XMLHelper.LoadAttribute(nav, "Duration", 1));

            float[] Values = new float[Dimensions.Elements()];

            foreach (XPathNavigator nodeNav in nav.SelectChildren("Node", ""))
            {
                //try
                {
                    int X = int.Parse(nodeNav.GetAttribute("X", ""), CultureInfo.InvariantCulture);
                    int Y = int.Parse(nodeNav.GetAttribute("Y", ""), CultureInfo.InvariantCulture);
                    int Z = int.Parse(nodeNav.GetAttribute("Z", ""), CultureInfo.InvariantCulture);
                    int W = int.Parse(nodeNav.GetAttribute("W", ""), CultureInfo.InvariantCulture);
                    float Value = float.Parse(nodeNav.GetAttribute("Value", ""), CultureInfo.InvariantCulture);

                    Values[((W * Dimensions.Z + Z) * Dimensions.Y + Y) * Dimensions.X + X] = Value;
                }
                //catch { }
            }

            return new LinearGrid4D(Dimensions, Values);
        }
    }
}
