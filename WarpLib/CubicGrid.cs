using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp
{
    public class CubicGrid : IDisposable
    {
        public readonly int3 Dimensions;
        public readonly DimensionSets DimensionSet;
        public readonly float3 Margins = new float3(0);
        public float[] Values;
        private IntPtr Einspline = IntPtr.Zero;
        
        private static readonly bool CanUseEinspline;

        public float[] FlatValues
        {
            get { return Values; }
        }
        
        static CubicGrid()
        {
            try
            {
                IntPtr Test = CPU.CreateEinspline3(new float[] { 1.0f, 2.0f, 3.0f }, new int3(1, 1, 1), new float3(0.1f, 0.1f, 0.1f));
                CPU.DestroyEinspline(Test);

                CanUseEinspline = true;
            }
            catch (DllNotFoundException ex)
            {
                //Console.WriteLine("Einspline will not be available: " + ex.Message);
                CanUseEinspline = false;
            }
            catch (EntryPointNotFoundException ex)
            {
                //Console.WriteLine("The required entry point in the native library is not found: " + ex.Message);
                CanUseEinspline = false;
            }
        }


        public CubicGrid(int3 dimensions, float valueMin, float valueMax, Dimension gradientDirection, bool centeredSpacing = false)
        {
            Dimensions = dimensions;
            DimensionSet = GetDimensions(Dimensions);

            if (centeredSpacing)
            {
                Margins = new float3(1f / Dimensions.X, 1f / Dimensions.Y, 1f / Dimensions.Z) / 2;

                if (Dimensions.X == 1)
                    Margins.X = 0;
                if (Dimensions.Y == 1)
                    Margins.Y = 0;
                if (Dimensions.Z == 1)
                    Margins.Z = 0;
            }

            float Step = valueMax - valueMin;
            if (gradientDirection == Dimension.X)
                Step /= Math.Max(1, dimensions.X - 1);
            else if (gradientDirection == Dimension.Y)
                Step /= Math.Max(1, dimensions.Y - 1);
            else if (gradientDirection == Dimension.Z)
                Step /= Math.Max(1, dimensions.Z - 1);

            Values = new float[dimensions.Elements()];
            for (int z = 0; z < dimensions.Z; z++)
                for (int y = 0; y < dimensions.Y; y++)
                    for (int x = 0; x < dimensions.X; x++)
                    {
                        float Value = valueMin;
                        if (gradientDirection == Dimension.X)
                            Value += x * Step;
                        if (gradientDirection == Dimension.Y)
                            Value += y * Step;
                        if (gradientDirection == Dimension.Z)
                            Value += z * Step;

                        Values[(z * Dimensions.Y + y) * Dimensions.X + x] = Value;
                    }

            if (CanUseEinspline)
                Einspline = MakeEinspline(Values, Dimensions, Margins);
        }

        public CubicGrid(int3 dimensions)
        {
            Dimensions = dimensions;
            DimensionSet = GetDimensions(Dimensions);

            Values = new float[dimensions.Elements()];
        }

        public CubicGrid(int3 dimensions, float[] values, bool centeredSpacing = false)
        {
            Dimensions = dimensions;
            DimensionSet = GetDimensions(Dimensions);

            if (centeredSpacing)
            {
                Margins = new float3(1f / Dimensions.X, 1f / Dimensions.Y, 1f / Dimensions.Z) / 2;

                if (Dimensions.X == 1)
                    Margins.X = 0;
                if (Dimensions.Y == 1)
                    Margins.Y = 0;
                if (Dimensions.Z == 1)
                    Margins.Z = 0;
            }

            Values = new float[dimensions.Elements()];
            Array.Copy(values, Values, (int)Dimensions.Elements());

            if (CanUseEinspline)
                Einspline = MakeEinspline(Values, Dimensions, Margins);
        }

        public CubicGrid(int3 dimensions, float[] values, float3 margins)
        {
            Dimensions = dimensions;
            DimensionSet = GetDimensions(Dimensions);

            Margins = margins;

            Values = new float[dimensions.Elements()];
            Array.Copy(values, Values, (int)Dimensions.Elements());

            if (CanUseEinspline)
                Einspline = MakeEinspline(Values, Dimensions, Margins);
        }

        ~CubicGrid()
        {
            Dispose();
        }

        public float[] GetInterpolated(float3[] coords)
        {
            if (Einspline == IntPtr.Zero || DimensionSet == DimensionSets.None)
                return Helper.ArrayOfConstant(Values[0], coords.Length);

            float[] Result = ArrayPool<float>.Rent(coords.Length);
            //float[] Coords = Helper.ToInterleaved(coords);

            switch (DimensionSet)
            {
                case DimensionSets.XYZ:
                    CPU.EvalEinspline3(Einspline, coords, coords.Length, Result);
                    break;

                case DimensionSets.XY:
                    CPU.EvalEinspline2XY(Einspline, coords, coords.Length, Result);
                    break;
                case DimensionSets.XZ:
                    CPU.EvalEinspline2XZ(Einspline, coords, coords.Length, Result);
                    break;
                case DimensionSets.YZ:
                    CPU.EvalEinspline2YZ(Einspline, coords, coords.Length, Result);
                    break;

                case DimensionSets.X:
                    CPU.EvalEinspline1X(Einspline, coords, coords.Length, Result);
                    break;
                case DimensionSets.Y:
                    CPU.EvalEinspline1Y(Einspline, coords, coords.Length, Result);
                    break;
                case DimensionSets.Z:
                    CPU.EvalEinspline1Z(Einspline, coords, coords.Length, Result);
                    break;
            }

            return Result;
        }

        public Span<float> GetInterpolated(Span<float3> coords, Span<float> result)
        {
            if (Einspline == IntPtr.Zero || DimensionSet == DimensionSets.None)
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] = Values[0];

                return result;
            }

            switch (DimensionSet)
            {
                case DimensionSets.XYZ:
                    CPU.EvalEinspline3(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;

                case DimensionSets.XY:
                    CPU.EvalEinspline2XY(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;
                case DimensionSets.XZ:
                    CPU.EvalEinspline2XZ(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;
                case DimensionSets.YZ:
                    CPU.EvalEinspline2YZ(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;

                case DimensionSets.X:
                    CPU.EvalEinspline1X(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;
                case DimensionSets.Y:
                    CPU.EvalEinspline1Y(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;
                case DimensionSets.Z:
                    CPU.EvalEinspline1Z(Einspline, ref MemoryMarshal.GetReference(coords), coords.Length, ref MemoryMarshal.GetReference(result));
                    break;
            }

            return result;
        }

        public float GetInterpolated(float3 coords)
        {
            return GetInterpolated(new[] { coords })[0];
        }

        public float[] GetInterpolated(int3 valueGrid, float3 border)
        {
            float StepX = (1f - border.X * 2) / Math.Max(1, valueGrid.X - 1);
            float OffsetX = border.X;
            
            float StepY = (1f - border.Y * 2) / Math.Max(1, valueGrid.Y - 1);
            float OffsetY = border.Y;

            float StepZ = (1f - border.Z * 2) / Math.Max(valueGrid.Z - 1, 1);
            float OffsetZ = valueGrid.Z == 1 ? 0.5f : border.Z;

            float3[] Coords = new float3[valueGrid.Elements()];
            
            for (int z = 0, i = 0; z < valueGrid.Z; z++)
                for (int y = 0; y < valueGrid.Y; y++)
                    for (int x = 0; x < valueGrid.X; x++, i++)
                        Coords[(z * valueGrid.Y + y) * valueGrid.X + x] = new float3(x * StepX + OffsetX, y * StepY + OffsetY, z * StepZ + OffsetZ);

            return GetInterpolated(Coords);
        }

        public float[] GetInterpolatedNative(int3 valueGrid, float3 border)
        {
            return GetInterpolated(valueGrid, border);
        }

        public float[] GetInterpolatedNative(float3[] positions)
        {
            return GetInterpolated(positions);
        }

        public CubicGrid Resize(int3 newSize)
        {
            float[] Result = new float[newSize.Elements()];

            float StepX = 1f / Math.Max(1, newSize.X - 1);
            float StepY = 1f / Math.Max(1, newSize.Y - 1);
            float StepZ = 1f / Math.Max(1, newSize.Z - 1);

            for (int z = 0, i = 0; z < newSize.Z; z++)
                for (int y = 0; y < newSize.Y; y++)
                    for (int x = 0; x < newSize.X; x++, i++)
                        Result[i] = GetInterpolated(new float3(x * StepX, y * StepY, z * StepZ));

            return new CubicGrid(newSize, Result, Margins.X > 0 || Margins.Y > 0 || Margins.Z > 0);
        }

        public CubicGrid CollapseXY()
        {
            float[] Collapsed = new float[Dimensions.Z];
            for (int z = 0; z < Collapsed.Length; z++)
            {
                float Mean = 0;
                for (int y = 0; y < Dimensions.Y; y++)
                    for (int x = 0; x < Dimensions.X; x++)
                        Mean += Values[(z * Dimensions.Y + y) * Dimensions.X + x];

                Mean /= Dimensions.ElementsSlice();
                Collapsed[z] = Mean;
            }

            return new CubicGrid(new int3(1, 1, Dimensions.Z), Collapsed, Margins.X > 0 || Margins.Y > 0 || Margins.Z > 0);
        }

        public CubicGrid CollapseZ()
        {
            float[] Collapsed = new float[Dimensions.ElementsSlice()];
            for (int y = 0; y < Dimensions.Y; y++)
            {
                for (int x = 0; x < Dimensions.X; x++)
                {
                    float Mean = 0;
                    for (int z = 0; z < Dimensions.Z; z++)
                        Mean += Values[(z * Dimensions.Y + y) * Dimensions.X + x];

                    Mean /= Dimensions.Z;
                    Collapsed[y * Dimensions.X + x] = Mean;
                }
            }

            return new CubicGrid(Dimensions.Slice(), Collapsed, Margins.X > 0 || Margins.Y > 0 || Margins.Z > 0);
        }

        public float[][] GetWiggleWeights(int3 valueGrid, float3 border, bool centeredSpacing = false)
        {
            float[][] Result = new float[Dimensions.Elements()][];

            for (int i = 0; i < Result.Length; i++)
            {
                float[] PlusValues = new float[Dimensions.Elements()];
                PlusValues[i] = 1f;
                CubicGrid PlusGrid = new CubicGrid(Dimensions, PlusValues, centeredSpacing);

                Result[i] = PlusGrid.GetInterpolatedNative(valueGrid, border);
            }

            return Result;
        }

        public (int index, float weight)[][] GetWiggleWeightsSparse(int3 valueGrid, float3 border, bool centeredSpacing = false)
        {
            var Result = new (int index, float weight)[Dimensions.Elements()][];

            for (int i = 0; i < Result.Length; i++)
            {
                List<(int index, float weight)> ResultList = new List<(int index, float weight)>();

                float[] PlusValues = new float[Dimensions.Elements()];
                PlusValues[i] = 1f;
                CubicGrid PlusGrid = new CubicGrid(Dimensions, PlusValues, centeredSpacing);

                var Sparse = PlusGrid.GetInterpolatedNative(valueGrid, border);
                for (int j = 0; j < Sparse.Length; j++)
                    if (Math.Abs(Sparse[j]) > 0.02f)
                        ResultList.Add((j, Sparse[j]));

                Result[i] = ResultList.ToArray();
            }

            return Result;
        }

        public float[] GetSliceXY(int z)
        {
            float[] Result = new float[Dimensions.X * Dimensions.Y];
            for (int y = 0; y < Dimensions.Y; y++)
                for (int x = 0; x < Dimensions.X; x++)
                    Result[y * Dimensions.X + x] = Values[(z * Dimensions.Y + y) * Dimensions.X + x];

            return Result;
        }

        public float[] GetSliceXZ(int y)
        {
            float[] Result = new float[Dimensions.X * Dimensions.Z];
            for (int z = 0; z < Dimensions.Z; z++)
                for (int x = 0; x < Dimensions.X; x++)
                    Result[z * Dimensions.X + x] = Values[(z * Dimensions.Y + y) * Dimensions.X + x];

            return Result;
        }

        public float[] GetSliceYZ(int x)
        {
            float[] Result = new float[Dimensions.Y * Dimensions.Z];
            for (int z = 0; z < Dimensions.Z; z++)
                for (int y = 0; y < Dimensions.Y; y++)
                    Result[z * Dimensions.Y + y] = Values[(z * Dimensions.Y + y) * Dimensions.X + x];

            return Result;
        }

        public void Dispose()
        {
            if (CanUseEinspline && Einspline != IntPtr.Zero)
            {
                CPU.DestroyEinspline(Einspline);
                Einspline = IntPtr.Zero;
            }
        }

        public void Save(XmlTextWriter writer)
        {
            XMLHelper.WriteAttribute(writer, "Width", Dimensions.X);
            XMLHelper.WriteAttribute(writer, "Height", Dimensions.Y);
            XMLHelper.WriteAttribute(writer, "Depth", Dimensions.Z);

            XMLHelper.WriteAttribute(writer, "MarginX", Margins.X);
            XMLHelper.WriteAttribute(writer, "MarginY", Margins.Y);
            XMLHelper.WriteAttribute(writer, "MarginZ", Margins.Z);

            for (int z = 0; z < Dimensions.Z; z++)
                for (int y = 0; y < Dimensions.Y; y++)
                    for (int x = 0; x < Dimensions.X; x++)
                    {
                        writer.WriteStartElement("Node");
                        XMLHelper.WriteAttribute(writer, "X", x);
                        XMLHelper.WriteAttribute(writer, "Y", y);
                        XMLHelper.WriteAttribute(writer, "Z", z);
                        XMLHelper.WriteAttribute(writer, "Value", Values[(z * Dimensions.Y + y) * Dimensions.X + x]);
                        writer.WriteEndElement();
                    }
        }

        public static CubicGrid Load(XPathNavigator nav)
        {
            int3 Dimensions = new int3(XMLHelper.LoadAttribute(nav, "Width", 1),
                                       XMLHelper.LoadAttribute(nav, "Height", 1),
                                       XMLHelper.LoadAttribute(nav, "Depth", 1));

            float3 Margins = new float3(XMLHelper.LoadAttribute(nav, "MarginX", 0f),
                                        XMLHelper.LoadAttribute(nav, "MarginY", 0f),
                                        XMLHelper.LoadAttribute(nav, "MarginZ", 0f));

            float[] Values = new float[Dimensions.Elements()];

            foreach (XPathNavigator nodeNav in nav.SelectChildren("Node", ""))
            {
                //try
                {
                    int X = int.Parse(nodeNav.GetAttribute("X", ""), CultureInfo.InvariantCulture);
                    int Y = int.Parse(nodeNav.GetAttribute("Y", ""), CultureInfo.InvariantCulture);
                    int Z = int.Parse(nodeNav.GetAttribute("Z", ""), CultureInfo.InvariantCulture);
                    float Value = float.Parse(nodeNav.GetAttribute("Value", ""), CultureInfo.InvariantCulture);

                    Values[(Z * Dimensions.Y + Y) * Dimensions.X + X] = Value;
                }
                //catch { }
            }

            return new CubicGrid(Dimensions, Values, Margins);
        }

        private static DimensionSets GetDimensions(int3 dims)
        {
            if (dims.X > 1 && dims.Y > 1 && dims.Z > 1)
                return DimensionSets.XYZ;
            if (dims.X > 1 && dims.Y > 1)
                return DimensionSets.XY;
            if (dims.X > 1 && dims.Z > 1)
                return DimensionSets.XZ;
            if (dims.Y > 1 && dims.Z > 1)
                return DimensionSets.YZ;
            if (dims.X > 1)
                return DimensionSets.X;
            if (dims.Y > 1)
                return DimensionSets.Y;
            if (dims.Z > 1)
                return DimensionSets.Z;

            return DimensionSets.None;
        }

        private static IntPtr MakeEinspline(float[] values, int3 dims, float3 margins)
        {
            DimensionSets DimSet = GetDimensions(dims);

            switch (DimSet)
            {
                case DimensionSets.XYZ:
                    return CPU.CreateEinspline3(values, dims, margins);

                case DimensionSets.XY:
                    return CPU.CreateEinspline2(values, new int2(dims.X, dims.Y), new float2(margins.X, margins.Y));
                case DimensionSets.XZ:
                    return CPU.CreateEinspline2(values, new int2(dims.X, dims.Z), new float2(margins.X, margins.Z));
                case DimensionSets.YZ:
                    return CPU.CreateEinspline2(values, new int2(dims.Y, dims.Z), new float2(margins.Y, margins.Z));

                case DimensionSets.X:
                    return CPU.CreateEinspline1(values, dims.X, margins.X);
                case DimensionSets.Y:
                    return CPU.CreateEinspline1(values, dims.Y, margins.Y);
                case DimensionSets.Z:
                    return CPU.CreateEinspline1(values, dims.Z, margins.Z);

                case DimensionSets.None:
                    return IntPtr.Zero;
            }

            return IntPtr.Zero;
        }
    }

    public enum Dimension
    {
        X,
        Y,
        Z
    }

    public enum DimensionSets
    {
        None = 0,
        X = 1 << 0,
        Y = 1 << 1,
        Z = 1 << 2,
        XY = 1 << 3,
        XZ = 1 << 4,
        YZ = 1 << 5,
        XYZ = 1 << 6
    }
}
