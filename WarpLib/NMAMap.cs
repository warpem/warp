using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using Warp.Headers;
using Warp.Tools;

namespace Warp
{
    public class NMAMap : IDisposable
    {
        private readonly object Sync = new object();

        public int NAtoms;
        public int NModes;
        public int3 DimsVolume;

        public readonly float Sigma;

        public int NSegments;
        public int[] Segmentation;
        public int[][] SegmentIndices;
        public float3[] SegmentCenters;
        public float3[][] SegmentCenteredAtoms;
        IntPtr[] d_SegmentCenteredAtoms;
        public float3[][][] SegmentModes;
        IntPtr[] d_SegmentModes;

        float3[] Atoms;
        IntPtr d_Atoms;

        float[] Intensities;
        IntPtr d_Intensities;

        float3[][] Modes;
        IntPtr d_Modes;

        KaiserTable BlobValues;

        public Projector[] BodyProjectors;
        ulong[] t_BodyProjectorsRe, t_BodyProjectorsIm;
        ulong[] a_BodyProjectorsRe, a_BodyProjectorsIm;

        public Projector[] BodyReconstructions;

        public NMAMap(float3[] atoms, float[] intensities, float sigma, float3[][] modes, int3 dimsVolume)
        {
            NAtoms = atoms.Length;
            NModes = modes.Length;
            Sigma = sigma;
            DimsVolume = dimsVolume;

            BlobValues = new KaiserTable(512, 1.0f * Sigma / 0.5f, 6f, 0);

            if (atoms.Length != modes[0].Length)
                throw new DimensionMismatchException();

            // Make independent copies of arrays
            Atoms = atoms.ToArray();
            Intensities = intensities.ToArray();
            Modes = modes.Select(v => v.ToArray()).ToArray();
        }

        public NMAMap(NMAMap coarseMap, float3[] atoms, float[] intensities, float sigma, int3 dimsVolume)
        {
            NAtoms = atoms.Length;
            NModes = coarseMap.NModes;
            Sigma = sigma;
            DimsVolume = dimsVolume;

            BlobValues = new KaiserTable(512, 1.0f * Sigma / 0.5f, 6f, 0);

            Atoms = atoms.ToArray();
            Intensities = intensities;
            Modes = new float3[NModes][];
            for (int m = 0; m < NModes; m++)
                Modes[m] = new float3[NAtoms];

            Parallel.For(0, atoms.Length, a1 =>
            {
                float3 ThisPos = atoms[a1];
                float[] Weights = new float[coarseMap.Atoms.Length];

                for (int a2 = 0; a2 < coarseMap.Atoms.Length; a2++)
                {
                    float Dist = (ThisPos - coarseMap.Atoms[a2]).LengthSq();
                    Weights[a2] = -Dist;
                }
                int[] HighestIndices;
                MathHelper.TakeNHighest(Weights, 100, out HighestIndices);

                float3[] Original = new float3[HighestIndices.Length];
                for (int i = 0; i < HighestIndices.Length; i++)
                    Original[i] = coarseMap.Atoms[HighestIndices[i]];

                float3 OriginalCenter = MathHelper.Mean(Original);
                for (int i = 0; i < HighestIndices.Length; i++)
                    Original[i] -= OriginalCenter;

                float3 ThisCentered = ThisPos - OriginalCenter;

                for (int m = 0; m < NModes; m++)
                {
                    float3[] Transformed = new float3[HighestIndices.Length];
                    for (int i = 0; i < HighestIndices.Length; i++)
                        Transformed[i] = Original[i] + coarseMap.Modes[m][HighestIndices[i]] * 0.01f;
                    float3 TransformedCenter = MathHelper.Mean(Transformed);
                    for (int i = 0; i < HighestIndices.Length; i++)
                        Transformed[i] -= TransformedCenter;

                    Matrix4 Transform = Matrix4.Translate(TransformedCenter) * new Matrix4(Matrix3.FromPointSets(Original, Transformed));
                    float3 ThisTransformed = Transform * ThisCentered;

                    Modes[m][a1] = (ThisTransformed - ThisCentered) * 100f;
                }
            });
        }

        public NMAMap(NMAMap copyMap, float[] intensities)
        {
            NAtoms = copyMap.Atoms.Length;
            NModes = copyMap.NModes;
            Sigma = copyMap.Sigma;
            DimsVolume = copyMap.DimsVolume;

            BlobValues = new KaiserTable(512, 1.0f * Sigma / 0.5f, 6f, 0);

            Atoms = copyMap.Atoms.ToArray();
            Intensities = intensities;
            Modes = copyMap.Modes.Select(a => a.ToArray()).ToArray();

            if (copyMap.Segmentation != null) Segmentation = copyMap.Segmentation.ToArray();
            if (copyMap.SegmentIndices != null) SegmentIndices = copyMap.SegmentIndices.Select(a => a.ToArray()).ToArray();
            if (copyMap.SegmentCenters != null) SegmentCenters = copyMap.SegmentCenters.ToArray();
            if (copyMap.SegmentCenteredAtoms != null) SegmentCenteredAtoms = copyMap.SegmentCenteredAtoms.Select(a => a.ToArray()).ToArray();
            if (copyMap.SegmentModes != null) SegmentModes = copyMap.SegmentModes.Select(a => a.Select(b => b.ToArray()).ToArray()).ToArray();
        }

        public NMAMap(string mrcPath)
        {
            HeaderMRC Header = (HeaderMRC)MapHeader.ReadFromFile(mrcPath);
            if (Header.Dimensions.Z > 1)
                throw new Exception("Map is not packed NMA because Z > 1");

            Image Packed = Image.FromFile(mrcPath, new int2(1, 1), 0, typeof(float));

            NAtoms = Packed.Dims.Y;
            NModes = (Packed.Dims.X - 4) / 3;
            Sigma = Header.PixelSize.X;
            DimsVolume = Header.Griddimensions;

            BlobValues = new KaiserTable(512, 1.0f * Sigma / 0.5f, 6f, 0);

            float[] PackedData = Packed.GetHostContinuousCopy();

            Atoms = new float3[NAtoms];
            Intensities = new float[NAtoms];
            Modes = new float3[NModes][];
            for (int m = 0; m < NModes; m++)
                Modes[m] = new float3[NAtoms];

            for (int a = 0; a < NAtoms; a++)
            {
                Atoms[a] = new float3(PackedData[a * Packed.Dims.X + 0], PackedData[a * Packed.Dims.X + 1], PackedData[a * Packed.Dims.X + 2]);
                Intensities[a] = PackedData[a * Packed.Dims.X + 3];

                for (int m = 0; m < NModes; m++)
                    Modes[m][a] = new float3(PackedData[a * Packed.Dims.X + 4 + m * 3 + 0], PackedData[a * Packed.Dims.X + 4 + m * 3 + 1], PackedData[a * Packed.Dims.X + 4 + m * 3 + 2]);
            }

            Packed.Dispose();
        }

        public NMAMap(string[] trajectoryFiles, float pixelSize, int3 dimsVolume, bool centerStructure = true)
        {
            List<float3[]> modes = new List<float3[]>();
            float3[] atoms = null;

            foreach (var trajectoryFile in trajectoryFiles)
            {
                List<Tuple<int, float3[]>> Models = new List<Tuple<int, float3[]>>();
                List<float3> CurrentModel = null;
                int CurrentModelID = 0;

                using (TextReader Reader = File.OpenText(trajectoryFile))
                {
                    string Line = null;
                    while ((Line = Reader.ReadLine()) != null)
                    {
                        if (Line.Substring(0, 5) == "MODEL")
                        {
                            CurrentModelID = int.Parse(Line.Substring(10, 4));
                            CurrentModel = new List<float3>();
                        }
                        else if (Line.Substring(0, 4) == "ATOM")
                        {
                            float3 AtomPos = new float3(float.Parse(Line.Substring(30, 8)),
                                                        float.Parse(Line.Substring(38, 8)),
                                                        float.Parse(Line.Substring(46, 8))) / pixelSize;
                            CurrentModel.Add(AtomPos);
                        }
                        else if (Line.Substring(0, 6) == "ENDMDL")
                        {
                            Models.Add(new Tuple<int, float3[]>(CurrentModelID, CurrentModel.ToArray()));
                        }
                    }
                }

                if (Models.Count < 2)
                    throw new Exception("Not a trajectory.");

                Models.Sort((a, b) => a.Item1.CompareTo(b.Item1));

                if (atoms == null)
                    atoms = Models[0].Item2;

                float3[] Mode = MathHelper.Subtract(Models[1].Item2, Models[0].Item2);
                float Variance = Mode.Select(v => v.Length()).Sum() / Mode.Length;
                Mode = Mode.Select(v => v / Variance).ToArray();

                modes.Add(Mode);
            }

            if (centerStructure)
            {
                float3 Center = new float3(dimsVolume) / 2 - MathHelper.Mean(atoms);
                atoms = atoms.Select(v => v + Center).ToArray();
            }

            NAtoms = atoms.Length;
            NModes = modes.Count;
            Sigma = 1;
            DimsVolume = dimsVolume;

            BlobValues = new KaiserTable(512, 1.0f * Sigma / 0.5f, 6f, 0);

            if (atoms.Length != modes[0].Length)
                throw new DimensionMismatchException();

            // Make independent copies of arrays
            Atoms = atoms;
            Intensities = Helper.ArrayOfConstant(1f, NAtoms);
            Modes = modes.ToArray();
        }

        void PutOnDevice()
        {
            Dispose();

            d_Atoms = GPU.MallocDeviceFromHost(Helper.ToInterleaved(Atoms), Atoms.Length * 3);
            d_Intensities = GPU.MallocDeviceFromHost(Intensities, Intensities.Length);
            float3[] AllModes = new float3[NAtoms * NModes];
            for (int m = 0; m < NModes; m++)
                for (int a = 0; a < NAtoms; a++)
                    AllModes[m * NAtoms + a] = Modes[m][a];
            d_Modes = GPU.MallocDeviceFromHost(Helper.ToInterleaved(AllModes), AllModes.Length * 3);
        }

        void PutBodyProjectorsOnDevice()
        {
            Dispose();

            if (BodyProjectors == null)
                throw new Exception("No body projectors initialized.");

            t_BodyProjectorsRe = new ulong[NSegments];
            t_BodyProjectorsIm = new ulong[NSegments];
            a_BodyProjectorsRe = new ulong[NSegments];
            a_BodyProjectorsIm = new ulong[NSegments];

            for (int s = 0; s < NSegments; s++)
            {
                ulong[] Textures = new ulong[2];
                ulong[] Arrays = new ulong[2];
                GPU.CreateTexture3DComplex(BodyProjectors[s].Data.GetDevice(Intent.Read), BodyProjectors[s].Data.DimsEffective, Textures, Arrays, false);

                t_BodyProjectorsRe[s] = Textures[0];
                t_BodyProjectorsIm[s] = Textures[1];
                a_BodyProjectorsRe[s] = Arrays[0];
                a_BodyProjectorsIm[s] = Arrays[1];

                BodyProjectors[s].FreeDevice();
            }
        }

        bool IsOnDevice()
        {
            return d_Atoms != IntPtr.Zero && d_Intensities != IntPtr.Zero && d_Modes != IntPtr.Zero;
        }

        public void FreeOnDevice()
        {
            Dispose();

            if (BodyProjectors != null)
                foreach (var b in BodyProjectors)
                    b?.FreeDevice();

            if (BodyReconstructions != null)
                foreach (var b in BodyReconstructions)
                    b?.FreeDevice();
        }

        public void FreeProjectors()
        {
            if (BodyProjectors != null)
                foreach (var p in BodyProjectors)
                    p?.FreeDevice();
        }

        public void FreeReconstructions()
        {
            if (BodyReconstructions != null)
                foreach (var p in BodyReconstructions)
                    p?.FreeDevice();
        }

        public void Dispose()
        {
            if (d_Atoms != IntPtr.Zero)
            {
                GPU.FreeDevice(d_Atoms);
                d_Atoms = IntPtr.Zero;
            }
            if (d_Intensities != IntPtr.Zero)
            {
                GPU.FreeDevice(d_Intensities);
                d_Intensities = IntPtr.Zero;
            }
            if (d_Modes != IntPtr.Zero)
            {
                GPU.FreeDevice(d_Modes);
                d_Modes = IntPtr.Zero;
            }
            if (t_BodyProjectorsRe != null)
            {
                for (int s = 0; s < NSegments; s++)
                {
                    GPU.DestroyTexture(t_BodyProjectorsRe[s], a_BodyProjectorsRe[s]);
                    GPU.DestroyTexture(t_BodyProjectorsIm[s], a_BodyProjectorsIm[s]);
                }

                t_BodyProjectorsRe = null;
                t_BodyProjectorsIm = null;
                a_BodyProjectorsRe = null;
                a_BodyProjectorsIm = null;
            }
            if (d_SegmentCenteredAtoms != null)
            {
                foreach (var ptr in d_SegmentCenteredAtoms)
                    GPU.FreeDevice(ptr);
                d_SegmentCenteredAtoms = null;
            }
            if (d_SegmentModes != null)
            {
                foreach (var ptr in d_SegmentModes)
                    GPU.FreeDevice(ptr);
                d_SegmentModes = null;
            }
        }

        public void AddSegmentation(Image segments)
        {
            if (segments.Dims != DimsVolume)
                throw new AccessViolationException();

            Segmentation = segments.GetHostContinuousCopy().Select(v => (int)Math.Round(v)).ToArray();
            if (MathHelper.Min(Segmentation) >= 0)
                throw new Exception("No background voxels.");
            NSegments = MathHelper.Max(Segmentation) + 1;

            List<int>[] IndexLists = new List<int>[NSegments];
            for (int i = 0; i < NSegments; i++)
                IndexLists[i] = new List<int>();

            for (int a = 0; a < NAtoms; a++)
            {
                int3 Coords = new int3(Atoms[a]);
                int Partition = Segmentation[segments.Dims.ElementFromPosition(Coords)];
                if (Partition >= 0)
                    IndexLists[Partition].Add(a);
            }

            SegmentIndices = IndexLists.Select(l => l.ToArray()).ToArray();

            float3 Center = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);
            SegmentCenters = SegmentIndices.Select(a => MathHelper.Mean(a.Select(i => Atoms[i])) - Center).ToArray();
            SegmentCenteredAtoms = Helper.ArrayOfFunction(s => SegmentIndices[s].Select(i => Atoms[i] - SegmentCenters[s] - Center).ToArray(), NSegments);
            SegmentModes = Helper.ArrayOfFunction(s => Modes.Select(m => SegmentIndices[s].Select(i => m[i]).ToArray()).ToArray(), NSegments);
        }

        public void ReduceToNSegments(int nSegments)
        {
            int Size = DimsVolume.X;

            List<List<int>> Partitions = Helper.ArrayOfSequence(0, NSegments, 1).Select(i => new List<int> { i }).ToList();

            #region Determine which partitions are spatially connected

            bool[][] Connections = Helper.ArrayOfFunction(() => new bool[NSegments], NSegments);

            for (int z = 1; z < Size - 1; z++)
                for (int y = 1; y < Size - 1; y++)
                    for (int x = 1; x < Size - 1; x++)
                    {
                        int Center = Segmentation[(z * Size + y) * Size + x];
                        if (Center < 0)
                            continue;

                        for (int dz = -1; dz <= 1; dz++)
                            for (int dy = -1; dy <= 1; dy++)
                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    int Neighbor = Segmentation[((z + dz) * Size + y + dy) * Size + x + dx];
                                    if (Neighbor < 0)
                                        continue;

                                    Connections[Center][Neighbor] = true;
                                    Connections[Neighbor][Center] = true;
                                }
                    }

            #endregion

            #region Helper methods

            Func<int, int[]> GetPartitionNeighbors = (id) =>
            {
                List<int> Neighbors = new List<int>(NSegments);
                for (int p = 0; p < Partitions.Count; p++)
                {
                    if (p == id)
                        continue;

                    bool AreConnected = false;
                    foreach (var p1 in Partitions[id])
                        foreach (var p2 in Partitions[p])
                            if (Connections[p1][p2])
                                AreConnected = true;

                    if (AreConnected)
                        Neighbors.Add(p);
                }

                return Neighbors.ToArray();
            };

            Func<int, int[]> GetPartitionIndices = (id) => Helper.Combine(Partitions[id].Select(v => SegmentIndices[v]));

            Func<int, float> GetPartitionError = (id) => MathHelper.Max(GetSegmentErrors(GetPartitionIndices(id)));

            Action<int, int> MergePartitions = (p1, p2) =>
            {
                List<int> MergedBodies = Helper.Combine(Partitions[p1].ToArray(), Partitions[p2].ToArray()).ToList();
                Partitions[p1] = MergedBodies;
                Partitions.RemoveAt(p2);
            };

            #endregion

            #region Merge partitions that are too small

            while (Helper.ArrayOfSequence(0, Partitions.Count, 1).Any(id => GetPartitionIndices(id).Length < 15))
            {
                int SmallestP = 0, SmallestIndices = GetPartitionIndices(0).Length;

                for (int p = 1; p < Partitions.Count; p++)
                    if (GetPartitionIndices(p).Length < SmallestIndices)
                    {
                        SmallestP = p;
                        SmallestIndices = GetPartitionIndices(p).Length;
                    }

                int[] PartitionNeighbors = GetPartitionNeighbors(SmallestP);
                if (PartitionNeighbors.Length == 0)
                {
                    Partitions.RemoveAt(SmallestP);
                    continue;
                }

                int SmallestNeighbor = PartitionNeighbors[0];
                int SmallestNeighborIndices = GetPartitionIndices(PartitionNeighbors[0]).Length;
                for (int n = 1; n < PartitionNeighbors.Length; n++)
                {
                    int NeighborIndices = GetPartitionIndices(PartitionNeighbors[n]).Length;
                    if (NeighborIndices < SmallestNeighborIndices)
                    {
                        SmallestNeighbor = PartitionNeighbors[n];
                        SmallestNeighborIndices = NeighborIndices;
                    }
                }

                MergePartitions(SmallestP, SmallestNeighbor);
            }

            #endregion

            #region Merge partition pairs that lead to the smallest increase in error

            while (Partitions.Count > nSegments)
            {
                List<Tuple<int, int, float>> Pairs = new List<Tuple<int, int, float>>();

                Parallel.For(0, Partitions.Count, p1 =>
                {
                    int[] PartitionNeighbors = GetPartitionNeighbors(p1);
                    List<Tuple<int, int, float>> ThisPairs = new List<Tuple<int, int, float>>();

                    foreach (int p2 in PartitionNeighbors)
                    {
                        int[] MergedIndices = Helper.Combine(GetPartitionIndices(p1), GetPartitionIndices(p2));
                        float PartitionError = MathHelper.Mean(GetSegmentErrors(MergedIndices));

                        float ErrorIncrease = PartitionError;

                        ThisPairs.Add(new Tuple<int, int, float>(p1, p2, ErrorIncrease));
                    }

                    lock (Partitions)
                    {
                        Pairs.AddRange(ThisPairs);
                    }
                });

                Pairs.Sort((a, b) =>
                {
                    if (a.Item3 != b.Item3)
                        return a.Item3.CompareTo(b.Item3);
                    else
                        return a.Item1.CompareTo(b.Item1);
                });

                Pairs = Pairs.Take(5).ToList();

                //Pairs.Sort((a, b) =>
                //{
                //    int SizeA = Math.Min(GetPartitionIndices(a.Item1).Length, GetPartitionIndices(a.Item2).Length);
                //    int SizeB = Math.Min(GetPartitionIndices(b.Item1).Length, GetPartitionIndices(b.Item2).Length);
                //    return SizeA.CompareTo(SizeB);
                //});

                MergePartitions(Pairs[0].Item1, Pairs[0].Item2);
            }
            
            #endregion

            int[] PartitionMapping = Helper.ArrayOfConstant(-1, NSegments);
            for (int p = 0; p < Partitions.Count; p++)
                foreach (int b in Partitions[p])
                    PartitionMapping[b] = p;

            float[] NewSegmentation = new float[Segmentation.Length];
            for (int i = 0; i < Segmentation.Length; i++)
            {
                if (Segmentation[i] < 0)
                    NewSegmentation[i] = -1f;
                else
                    NewSegmentation[i] = PartitionMapping[Segmentation[i]];
            }

            AddSegmentation(new Image(NewSegmentation, DimsVolume));
        }

        public Matrix4 GetSegmentTransform(int segment, float[] factors)
        {
            int[] Indices = SegmentIndices[segment];
            float3[] Original = new float3[Indices.Length];
            float3[] Transformed = new float3[Indices.Length];
            float3 CenterVolume = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);

            for (int i = 0; i < Indices.Length; i++)
            {
                Original[i] = Atoms[Indices[i]] - CenterVolume;

                Transformed[i] = Original[i];
                for (int m = 0; m < NModes; m++)
                    Transformed[i] += Modes[m][Indices[i]] * factors[m];
            }

            float3 CenterOriginal = MathHelper.Mean(Original);
            float3 CenterTransformed = MathHelper.Mean(Transformed);

            for (int i = 0; i < Original.Length; i++)
            {
                Original[i] -= CenterOriginal;
                Transformed[i] -= CenterTransformed;
            }

            Matrix3 R = Matrix3.FromPointSets(Original, Transformed);

            Matrix4 Transform = Matrix4.Translate(CenterTransformed.X, CenterTransformed.Y, CenterTransformed.Z) *
                                new Matrix4(R) *
                                Matrix4.Translate(-CenterOriginal.X, -CenterOriginal.Y, -CenterOriginal.Z);

            return Transform;
        }

        public Matrix4[] GetSegmentTransforms(int segment, float[][] factors)
        {
            if (SegmentCenteredAtoms == null || SegmentCenters == null || SegmentModes == null)
                throw new Exception("No segmentation data available.");

            lock (Sync)
            {
                if (d_SegmentCenteredAtoms == null)
                {
                    d_SegmentCenteredAtoms = new IntPtr[NSegments];
                    for (int s = 0; s < NSegments; s++)
                        d_SegmentCenteredAtoms[s] = GPU.MallocDeviceFromHost(Helper.ToInterleaved(SegmentCenteredAtoms[s]), SegmentCenteredAtoms[s].Length * 3);
                }

                if (d_SegmentModes == null)
                {
                    d_SegmentModes = new IntPtr[NSegments];
                    for (int s = 0; s < NSegments; s++)
                        d_SegmentModes[s] = GPU.MallocDeviceFromHost(Helper.ToInterleaved(Helper.Combine(SegmentModes[s])), SegmentCenteredAtoms[s].Length * NModes * 3);
                }
            }

            Matrix4[] Result = new Matrix4[factors.Length];

            float[] CentersTransFlat = new float[factors.Length * 3];
            float[] MatricesForSVD = new float[factors.Length * 9];

            GPU.ParticleNMAGetRigidTransform(d_SegmentCenteredAtoms[segment],
                                             d_SegmentModes[segment],
                                             (uint)SegmentIndices[segment].Length,
                                             Helper.Combine(factors),
                                             (uint)NModes,
                                             (uint)factors.Length,
                                             CentersTransFlat,
                                             MatricesForSVD);

            for (int i = 0; i < factors.Length; i++)
            {
                Matrix3 H = new Matrix3(Helper.Subset(MatricesForSVD, i * 9, (i + 1) * 9));

                Matrix3 U, V;
                float[] S;
                H.SVD(out U, out S, out V);

                Matrix3 R = V * U.Transposed();
                if (R.Determinant() < 0)
                {
                    V.M13 = -V.M13;
                    V.M23 = -V.M23;
                    V.M33 = -V.M33;

                    R = V * U.Transposed();
                }

                Result[i] = Matrix4.Translate(CentersTransFlat[i * 3 + 0] + SegmentCenters[segment].X,
                                              CentersTransFlat[i * 3 + 1] + SegmentCenters[segment].Y,
                                              CentersTransFlat[i * 3 + 2] + SegmentCenters[segment].Z) *
                            new Matrix4(R) *
                            Matrix4.Translate(-SegmentCenters[segment]);
            }

            return Result;
        }

        public Matrix4[] GetAllSegmentsTransform(float[] factors)
        {
            float3[] Original = new float3[NAtoms];
            float3[] Transformed = new float3[NAtoms];
            float3 CenterVolume = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);

            Matrix4[] Result = new Matrix4[NSegments];

            unsafe
            {
                fixed (float3* OriginalPtr = Original)
                fixed (float3* TransformedPtr = Transformed)
                {
                    fixed (float3* AtomsPtr = Atoms)
                    for (int i = 0; i < NAtoms; i++)
                    {
                        OriginalPtr[i] = AtomsPtr[i] - CenterVolume;
                        TransformedPtr[i] = OriginalPtr[i];
                    }

                    fixed (float* FactorsPtr = factors)
                    for (int m = 0; m < NModes; m++)
                    {
                        float3[] ModesPtr = Modes[m];
                        for (int i = 0; i < ModesPtr.Length; i++)
                            TransformedPtr[i] += ModesPtr[i] * FactorsPtr[m];
                    }

                    for (int s = 0; s < NSegments; s++)
                    {
                        int[] Indices = SegmentIndices[s];

                        float3[] SegmentOriginal = new float3[Indices.Length];
                        float3[] SegmentTransformed = new float3[Indices.Length];

                        float3 CenterOriginal = new float3();
                        float3 CenterTransformed = new float3();

                        fixed (float3* SegmentOriginalPtr = SegmentOriginal)
                        fixed (float3* SegmentTransformedPtr = SegmentTransformed)
                        {
                            fixed (int* IndicesPtr = Indices)
                            for (int i = 0; i < Indices.Length; i++)
                            {
                                float3 ValOri = OriginalPtr[IndicesPtr[i]];
                                SegmentOriginalPtr[i] = ValOri;
                                CenterOriginal += ValOri;

                                float3 ValTrans = TransformedPtr[IndicesPtr[i]];
                                SegmentTransformedPtr[i] = ValTrans;
                                CenterTransformed += ValTrans;
                            }

                            CenterOriginal /= Indices.Length;
                            CenterTransformed /= Indices.Length;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                SegmentOriginalPtr[i] -= CenterOriginal;
                                SegmentTransformedPtr[i] -= CenterTransformed;
                            }
                        }

                        Matrix3 R = Matrix3.FromPointSets(SegmentOriginal, SegmentTransformed);

                        Result[s] = Matrix4.Translate(CenterTransformed.X, CenterTransformed.Y, CenterTransformed.Z) *
                                    new Matrix4(R) *
                                    Matrix4.Translate(-CenterOriginal.X, -CenterOriginal.Y, -CenterOriginal.Z);
                    }
                }
            }

            return Result;
        }

        public Matrix4[][] GetAllSegmentsTransforms(float[][] factors)
        {
            if (SegmentCenteredAtoms == null || SegmentCenters == null || SegmentModes == null)
                throw new Exception("No segmentation data available.");

            Matrix4[][] Result = Helper.ArrayOfFunction(() => new Matrix4[NSegments], factors.Length);

            for (int s = 0; s < NSegments; s++)
            {
                Matrix4[] Transforms = GetSegmentTransforms(s, factors);
                for (int i = 0; i < factors.Length; i++)
                    Result[i][s] = Transforms[i];
            }

            return Result;
        }

        public Image[] GetSegmentMasks()
        {
            if (Segmentation == null)
                throw new Exception("No segmentation data available.");

            Image[] Result = new Image[NSegments];

            Parallel.For(0, NSegments, s =>
            {
                float[] SegmentData = new float[Segmentation.Length];
                for (int i = 0; i < SegmentData.Length; i++)
                    SegmentData[i] = Segmentation[i] == s ? 1f : 0f;

                Result[s] = new Image(SegmentData, DimsVolume);
            });

            return Result;
        }

        public Image[] GetSoftSegmentMasks(float segmentOverlap, float borderExtent = 0f, bool borderHard = false)
        {
            int Size = DimsVolume.X;
            Image[] HardMasks = GetSegmentMasks();

            float[][] Distances = new float[NSegments][];
            for (int b = 0; b < NSegments; b++)
            {
                Image DistanceMap = HardMasks[b].AsDistanceMap(Size / 2);
                Distances[b] = DistanceMap.GetHostContinuousCopy();
                //DistanceMap.WriteMRC($"d_distance{b + 1:D2}.mrc");

                DistanceMap.Dispose();
                HardMasks[b].Dispose();
            }

            float[][] SmoothMaskData = Helper.ArrayOfFunction(() => new float[Size * Size * Size], NSegments);
            float[] SmoothSamples = new float[Size * Size * Size];

            Parallel.For(0, SmoothMaskData[0].Length, i =>
            {
                double[] VoxelWeights = new double[NSegments];
                int IsWithin = 0, IsOverlap = 0;
                float MinDist = float.MaxValue;
                for (int b = 0; b < NSegments; b++)
                {
                    MinDist = Math.Min(MinDist, Distances[b][i]);
                    if (Distances[b][i] <= 1e-5f)
                        IsWithin++;
                    if (Distances[b][i] <= segmentOverlap + 1e-5f)
                        IsOverlap++;
                }

                if (IsWithin >= 1 && IsOverlap > 1)
                {
                    for (int b = 0; b < NSegments; b++)
                        VoxelWeights[b] = Math.Cos(Math.Min(1, Distances[b][i] / segmentOverlap) * Math.PI) * 0.5f + 0.5f;
                }
                else if (IsWithin == 1 && MinDist < 1)
                {
                    for (int b = 0; b < NSegments; b++)
                        VoxelWeights[b] = Distances[b][i] < 1 ? 1f : 0f;
                }
                else
                {
                    if (borderExtent <= 0f)
                        for (int b = 0; b < NSegments; b++)
                            VoxelWeights[b] = 1f / Math.Max(1, Distances[b][i] * Distances[b][i]);
                    else
                        for (int b = 0; b < NSegments; b++)
                        {
                            VoxelWeights[b] = (Math.Cos(Math.Min(1, Distances[b][i] / borderExtent) * Math.PI) * 0.5f + 0.5f); // / Math.Max(1, Distances[b][i] * Distances[b][i]);
                            if (!borderHard)
                                SmoothSamples[i] += Distances[b][i] <= borderExtent ? 1 : 0;
                            //else
                            //    VoxelWeights[b] *= VoxelWeights[b];
                        }
                }

                for (int b = 0; b < NSegments; b++)
                    SmoothMaskData[b][i] = (float)VoxelWeights[b];
            });

            Parallel.For(0, SmoothMaskData[0].Length, i =>
            {
                float WeightSum = 0;
                for (int b = 0; b < NSegments; b++)
                    WeightSum += SmoothMaskData[b][i];

                float BorderVal = SmoothSamples[i] > 0 ? 0f : 1f;
                if (BorderVal == 0f)
                {
                    for (int b = 0; b < NSegments; b++)
                        BorderVal = Math.Max(SmoothMaskData[b][i], BorderVal);
                }

                for (int b = 0; b < NSegments; b++)
                    SmoothMaskData[b][i] = SmoothMaskData[b][i] / Math.Max(1e-5f, WeightSum) * BorderVal;
            });

            float[][] ExtraSmoothMaskData = Helper.ArrayOfFunction(() => new float[Size * Size * Size], NSegments);

            Parallel.For(0, NSegments, b =>
            {
                // X pass
                for (int z = 3; z < Size - 3; z++)
                    for (int y = 3; y < Size - 3; y++)
                        for (int x = 3; x < Size - 3; x++)
                        {
                            float Sum = 0;
                            float Weights = 0;
                            
                            for (int dx = -3; dx <= 3; dx++)
                            {
                                float xx = dx * dx;
                                float Weight = (float)Math.Exp(-xx / 4);
                                Sum += SmoothMaskData[b][(z * Size + y) * Size + x + dx] * Weight;
                                Weights += Weight;
                            }

                            ExtraSmoothMaskData[b][(z * Size + y) * Size + x] = Sum / Weights;
                        }
                SmoothMaskData[b] = ExtraSmoothMaskData[b];
                ExtraSmoothMaskData[b] = new float[ExtraSmoothMaskData[b].Length];

                // Y pass
                for (int z = 3; z < Size - 3; z++)
                    for (int y = 3; y < Size - 3; y++)
                        for (int x = 3; x < Size - 3; x++)
                        {
                            float Sum = 0;
                            float Weights = 0;
                            
                                for (int dy = -3; dy <= 3; dy++)
                                {
                                    float yy = dy * dy;
                                    float Weight = (float)Math.Exp(-yy / 4);
                                    Sum += SmoothMaskData[b][(z * Size + y + dy) * Size + x] * Weight;
                                    Weights += Weight;
                                }

                            ExtraSmoothMaskData[b][(z * Size + y) * Size + x] = Sum / Weights;
                        }
                SmoothMaskData[b] = ExtraSmoothMaskData[b];
                ExtraSmoothMaskData[b] = new float[ExtraSmoothMaskData[b].Length];

                // Z pass
                for (int z = 3; z < Size - 3; z++)
                    for (int y = 3; y < Size - 3; y++)
                        for (int x = 3; x < Size - 3; x++)
                        {
                            float Sum = 0;
                            float Weights = 0;

                            for (int dz = -3; dz <= 3; dz++)
                            {
                                float zz = dz * dz;
                                    float Weight = (float)Math.Exp(-zz / 4);
                                    Sum += SmoothMaskData[b][((z + dz) * Size + y) * Size + x] * Weight;
                                    Weights += Weight;
                            }

                            ExtraSmoothMaskData[b][(z * Size + y) * Size + x] = Sum / Weights;
                        }
            });

            if (borderHard)
                Parallel.For(0, ExtraSmoothMaskData[0].Length, i =>
                {
                    float WeightSum = 0;
                    for (int b = 0; b < NSegments; b++)
                        WeightSum += ExtraSmoothMaskData[b][i];

                    for (int b = 0; b < NSegments; b++)
                        ExtraSmoothMaskData[b][i] = ExtraSmoothMaskData[b][i] / Math.Max(1e-10f, WeightSum);
                });

            return ExtraSmoothMaskData.Select(v => new Image(v, DimsVolume)).ToArray();
        }

        public void InitializeBodyProjectors(float segmentOverlap, float borderExtent, Image intensities, int maxSize, int oversample)
        {
            if (Segmentation == null)
                throw new Exception("No segmentation data avaialble.");

            if (BodyProjectors != null)
                foreach (var b in BodyProjectors)
                    b.Dispose();
            BodyProjectors = new Projector[NSegments];

            Image[] SoftMasks = GetSoftSegmentMasks(segmentOverlap, borderExtent);

            for (int s = 0; s < NSegments; s++)
            {
                SoftMasks[s].Multiply(intensities);

                Image Resized = SoftMasks[s].AsScaled(new int3(maxSize, maxSize, maxSize));
                SoftMasks[s].Dispose();

                BodyProjectors[s] = new Projector(Resized, oversample);
                BodyProjectors[s].FreeDevice();
                Resized.Dispose();
            }
        }

        public Image Project(int2 dims, float[] modeFactors, float3[] angles, float2[] shifts, float scale, int batch)
        {
            if (modeFactors.Length != NModes * batch || angles.Length != batch || shifts.Length != batch)
                throw new DimensionMismatchException();

            Image Proj = new Image(IntPtr.Zero, new int3(dims.X, dims.Y, batch));
            if (!IsOnDevice())
                PutOnDevice();

            IntPtr d_ModeFactors = GPU.MallocDeviceFromHost(modeFactors, modeFactors.Length);

            GPU.ProjectNMAPseudoAtoms(d_Atoms,
                                      d_Intensities,
                                      (uint)NAtoms,
                                      DimsVolume,
                                      Sigma,
                                      (uint)Math.Ceiling(BlobValues.Radius * scale),
                                      BlobValues.Values,
                                      BlobValues.Sampling * scale,
                                      (uint)BlobValues.Values.Length,
                                      d_Modes,
                                      d_ModeFactors,
                                      (uint)NModes,
                                      Helper.ToInterleaved(angles),
                                      Helper.ToInterleaved(shifts),
                                      scale,
                                      Proj.GetDevice(Intent.Write),
                                      dims,
                                      (uint)batch);

            GPU.FreeDevice(d_ModeFactors);

            return Proj;
        }

        public Image ProjectBodies(int2 dims, float[][] modeFactors, float3[] angles, float2[] shifts, float[][] bodyWeights, int batch)
        {
            if (BodyProjectors == null)
                throw new Exception("No projectors initialized.");

            if (t_BodyProjectorsRe == null)
                PutBodyProjectorsOnDevice();
            
            float ScaleFactor = (float)dims.X / DimsVolume.X;
            Image BodyProjections = new Image(new int3(dims.X, dims.Y, batch), true, true);
            
            Matrix4[] GlobalRotations = new Matrix4[batch];
            float2[] AllBodyShifts = new float2[NSegments * batch];
            float3[] AllBodyAngles = new float3[NSegments * batch];

            Matrix4[][] AllBodyTransforms = GetAllSegmentsTransforms(modeFactors);

            Parallel.For(0, batch, b =>
            {
                //Matrix4[] BodyTransforms = GetAllSegmentsTransform(modeFactors[b]);
                GlobalRotations[b] = Matrix4.Euler(angles[b].X, angles[b].Y, angles[b].Z);

                for (int s = 0; s < NSegments; s++)
                {
                    Matrix4 OverallTransform = GlobalRotations[b] * AllBodyTransforms[b][s];

                    AllBodyAngles[b * NSegments + s] = Matrix4.EulerFromMatrix(OverallTransform);

                    float3 RotationShift = OverallTransform * new float3(0, 0, 0);
                    AllBodyShifts[b * NSegments + s] = (shifts[b] + new float2(RotationShift)) * ScaleFactor; // THIS WOULD BE MINUS RotationShift IN BACK PROJECTION
                }
            });

            GPU.ParticleMultibodyProject(t_BodyProjectorsRe,
                                         t_BodyProjectorsIm,
                                         BodyProjectors[0].Data.Dims,
                                         BodyProjections.GetDevice(Intent.Write),
                                         dims,
                                         Helper.ToInterleaved(AllBodyAngles),
                                         Helper.ToInterleaved(AllBodyShifts),
                                         Helper.Combine(bodyWeights),
                                         BodyProjectors[0].Oversampling,
                                         (uint)NSegments,
                                         (uint)batch);

            return BodyProjections;
        }

        public Image ProjectBodiesRealspace(int2 dims, float[][] modeFactors, float3[] angles, float2[] shifts, float[][] bodyWeights, int batch)
        {
            Image ProjFT = ProjectBodies(dims, modeFactors, angles, shifts, bodyWeights, batch);
            Image Proj = ProjFT.AsIFFT().AndDisposeParent();
            Proj.RemapFromFT();

            return Proj;
        }

        public void InitializeBodyReconstructions(int[] bodyIndices, int oversample)
        {
            if (BodyReconstructions == null)
                BodyReconstructions = new Projector[NSegments];

            foreach (var s in bodyIndices)
            {
                BodyReconstructions[s] = new Projector(DimsVolume, oversample);
                BodyReconstructions[s].FreeDevice();
            }
        }

        public void BackProjectBodies(int[] bodyIndices, Image particles, Image weights, float[][] modeFactors, float3[] angles, float2[] shifts, float[][] bodyWeights, int batch)
        {
            if (BodyReconstructions == null)
                throw new Exception("No reconstructions initialized.");
            
            int si = 0;
            foreach (int s in bodyIndices)
            {
                if (BodyReconstructions[s] == null)
                    throw new Exception("Reconstruction not initialized.");

                float3[] GlobalShifts = new float3[batch];
                Matrix4[] GlobalRotations = new Matrix4[batch];
                float[] GlobalParticleWeights = Helper.ArrayOfConstant(1f, batch);

                for (int b = 0; b < batch; b++)
                {
                    GlobalShifts[b] = new float3(shifts[b]);
                    float3 EulerAngles = angles[b];
                    GlobalRotations[b] = Matrix4.Euler(EulerAngles.X, EulerAngles.Y, EulerAngles.Z);
                    GlobalParticleWeights[b] = bodyWeights[b][si];
                }

                float3[] BodyShifts = new float3[batch];
                float3[] BodyAngles = new float3[batch];

                Matrix4[] SegmentTransforms = GetSegmentTransforms(s, modeFactors);

                Parallel.For(0, batch, p =>
                {
                    Matrix4 NMATransform = SegmentTransforms[p];// GetSegmentTransform(s, modeFactors[p]);
                    Matrix4 OverallTransform = GlobalRotations[p] * NMATransform;

                    BodyAngles[p] = Matrix4.EulerFromMatrix(OverallTransform);

                    float3 RotationShift = OverallTransform * new float3(0, 0, 0);
                    BodyShifts[p] = GlobalShifts[p] - RotationShift;    // THIS WOULD BE PLUS RotationShift IN FORWARD PROJECTION
                });

                GPU.ProjectBackwardShifted(BodyReconstructions[s].Data.GetDevice(Intent.ReadWrite),
                                           BodyReconstructions[s].Weights.GetDevice(Intent.ReadWrite),
                                           BodyReconstructions[s].Data.Dims,
                                           particles.GetDevice(Intent.Read),
                                           weights.GetDevice(Intent.Read),
                                           new int2(particles.Dims),
                                           particles.Dims.X / 2,
                                           Helper.ToInterleaved(BodyAngles),
                                           Helper.ToInterleaved(BodyShifts),
                                           GlobalParticleWeights,
                                           BodyReconstructions[s].Oversampling,
                                           (uint)batch);

                si++;
            }
        }

        public Image[] ReconstructAllBodies()
        {
            Image[] Result = new Image[NSegments];

            for (int s = 0; s < NSegments; s++)
            {
                Image Rec = BodyReconstructions[s].Reconstruct(false);
                BodyReconstructions[s].FreeDevice();
                Result[s] = Rec;
                Rec.FreeDevice();
            }

            return Result;
        }

        public Image ReconstructAllBodies(float segmentOverlap, float borderExtent)
        {
            Image FrankenMap = new Image(DimsVolume);
            Image[] Masks = GetSoftSegmentMasks(segmentOverlap, borderExtent, true);

            for (int s = 0; s < NSegments; s++)
            {
                Image Rec = BodyReconstructions[s].Reconstruct(false);
                BodyReconstructions[s].FreeDevice();
                Rec.Multiply(Masks[s]);
                Masks[s].Dispose();

                FrankenMap.Add(Rec);
                Rec.Dispose();
            }

            return FrankenMap;
        }

        public Image RasterizeInVolume(int3 dims)
        {
            float3 OldCenter = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);
            float3 NewCenter = new float3(dims.X / 2, dims.Y / 2, dims.Z / 2);
            float3 Offset = NewCenter - OldCenter;

            float3[] OffsetAtoms = Atoms.Select(v => v + Offset).ToArray();

            return PhysicsHelper.GetVolumeFromAtoms(OffsetAtoms, dims, Sigma, Intensities);
        }

        public Image RasterizeDeformedInVolume(int3 dims, float[] nmaFactors)
        {
            if (nmaFactors.Length > NModes)
                throw new DimensionMismatchException();

            float3 OldCenter = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);
            float3 NewCenter = new float3(dims.X / 2, dims.Y / 2, dims.Z / 2);
            float3 Offset = NewCenter - OldCenter;

            float3[] OffsetAtoms = Atoms.Select(v => v + Offset).ToArray();

            for (int m = 0; m < nmaFactors.Length; m++)
                for (int a = 0; a < NAtoms; a++)
                    OffsetAtoms[a] += Modes[m][a] * nmaFactors[m];

            return PhysicsHelper.GetVolumeFromAtoms(OffsetAtoms, dims, Sigma, Intensities);
        }

        public void WriteToMRC(string mrcPath)
        {
            int3 DimsPacked = new int3((int)(4 + NModes * 3), (int)NAtoms, 1);
            float[] PackedData = new float[DimsPacked.Elements()];

            for (int a = 0; a < NAtoms; a++)
            {
                PackedData[a * DimsPacked.X + 0] = Atoms[a].X;
                PackedData[a * DimsPacked.X + 1] = Atoms[a].Y;
                PackedData[a * DimsPacked.X + 2] = Atoms[a].Z;

                PackedData[a * DimsPacked.X + 3] = Intensities[a];

                for (int m = 0; m < NModes; m++)
                {
                    PackedData[a * DimsPacked.X + 4 + m * 3 + 0] = Modes[m][a].X;
                    PackedData[a * DimsPacked.X + 4 + m * 3 + 1] = Modes[m][a].Y;
                    PackedData[a * DimsPacked.X + 4 + m * 3 + 2] = Modes[m][a].Z;
                }
            }

            Image Packed = new Image(PackedData, DimsPacked);

            HeaderMRC Header = new HeaderMRC();
            Header.PixelSize.X = Sigma;
            Header.Griddimensions = DimsVolume;

            Packed.WriteMRC(mrcPath, 1, false, Header);
        }

        public void SmoothModes(int groupSize)
        {
            int[][] AtomNeighbors = new int[NAtoms][];
            float DistCutoff = Sigma * Sigma * 16f;

            Parallel.For(0, NAtoms, i =>
            {
                float3 Pos = Atoms[i];
                List<int> Neighbors = new List<int>();

                for (int n = 0; n < NAtoms; n++)
                {
                    float Dist = (Atoms[n] - Pos).LengthSq();
                    if (Dist > 0 && Dist < DistCutoff)
                        Neighbors.Add(n);
                }

                AtomNeighbors[i] = Neighbors.ToArray();
            });

            float3[][] SmoothModes = new float3[NModes][];
            for (int m = 0; m < NModes; m++)
                SmoothModes[m] = new float3[NAtoms];

            Parallel.For(0, NAtoms, a =>
            {
                List<int> Neighborhood = new List<int> { a };

                while (Neighborhood.Count < groupSize)
                {
                    List<int> Extended = new List<int>(Neighborhood);
                    foreach (int n in Neighborhood)
                        foreach (int nn in AtomNeighbors[n])
                            if (!Extended.Contains(nn))
                                Extended.Add(nn);

                    if (Extended.Count == Neighborhood.Count)
                        break;

                    Neighborhood = Extended;
                }

                float3[] Original = new float3[Neighborhood.Count];
                for (int i = 0; i < Neighborhood.Count; i++)
                    Original[i] = Atoms[Neighborhood[i]];
                float3 CenterOriginal = MathHelper.Mean(Original);
                for (int i = 0; i < Original.Length; i++)
                    Original[i] -= CenterOriginal;

                for (int m = 0; m < NModes; m++)
                {
                    float3[] Transformed = new float3[Neighborhood.Count];

                    for (int i = 0; i < Neighborhood.Count; i++)
                        Transformed[i] = Original[i] + Modes[m][Neighborhood[i]] * 0.01f;

                    float3 CenterTransformed = MathHelper.Mean(Transformed);

                    for (int i = 0; i < Original.Length; i++)
                        Transformed[i] -= CenterTransformed;

                    Matrix3 R = Matrix3.FromPointSets(Original, Transformed);

                    Matrix4 Transform = Matrix4.Translate(CenterTransformed.X, CenterTransformed.Y, CenterTransformed.Z) *
                                        new Matrix4(R);

                    float3 SmoothDiff = (Transform * Original[0] - Original[0]) * 100f;
                    SmoothModes[m][a] = SmoothDiff;
                }
            });

            FreeOnDevice();
            Modes = SmoothModes;
        }

        public void LimitModesTo(int newLimit)
        {
            FreeOnDevice();
            NModes = newLimit;
            Modes = Modes.Take(newLimit).ToArray();
        }

        public float[] GetMeanDisplacement(float[][] nmaFactors)
        {
            float[] Result = new float[nmaFactors.Length];

            if (d_Modes == IntPtr.Zero)
                PutOnDevice();

            GPU.ParticleNMAGetMeanDisplacement(d_Modes,
                                               (uint)NAtoms,
                                               Helper.Combine(nmaFactors),
                                               (uint)NModes,
                                               (uint)nmaFactors.Length,
                                               Result);

            return Result;
        }

        public float GetMeanDisplacement(float[] nmaFactors)
        {
            if (nmaFactors.Length != NModes)
                throw new DimensionMismatchException();
            
            unsafe
            {
                float3[] Offsets = new float3[NAtoms];

                fixed (float3* OffsetsPtr = Offsets)
                fixed (float* NMAFactorsPtr = nmaFactors)
                {
                    for (int m = 0; m < nmaFactors.Length; m++)
                    {
                        float3[] ModesPtr = Modes[m];
                        for (int a = 0; a < ModesPtr.Length; a++)
                            OffsetsPtr[a] += ModesPtr[a] * NMAFactorsPtr[m];
                    }

                    float3 Mean = new float3();
                    for (int a = 0; a < NAtoms; a++)
                        Mean += OffsetsPtr[a];

                    Mean /= NAtoms;
                    float RMSD = 0;
                    for (int a = 0; a < NAtoms; a++)
                        RMSD += (OffsetsPtr[a] - Mean).LengthSq();
                    RMSD /= NAtoms;

                    return (float)Math.Sqrt(RMSD);
                }
            }
        }

        public static NMAMap FromScratch(Image intensity, Image mask, int natoms, int nmodes, float hessianRadius)
        {
            float R, Sigma, Corr;
            float3[] Atoms = PhysicsHelper.FillWithEquidistantPoints(mask, natoms, out R);
            float[] Intensities = PhysicsHelper.MatchVolumeIntensities(Atoms, intensity, mask, R, out Sigma, out Corr);
            Intensities = MathHelper.Max(Intensities, 0);

            double[][] Hessian = PhysicsHelper.GetHessian(Atoms, Intensities, hessianRadius);
            Matrix<double> HessianMat = Matrix<double>.Build.DenseOfColumnArrays(Hessian);
            Evd<double> Decomp = HessianMat.Evd(Symmetricity.Symmetric);
            Matrix<double> EigenvecsMat = Decomp.EigenVectors;

            // Skip first 6 vecs that store global motion and rotation, convert the next nmodes columns to float3[]
            float3[][] Modes = EigenvecsMat.ToColumnArrays().Skip(6).Take(nmodes).Select(a => Helper.FromInterleaved3(a.Select(v => (float)v * 100).ToArray())).ToArray();

            return new NMAMap(Atoms, Intensities, Sigma, Modes, intensity.Dims);
        }

        public float[] GetSegmentErrors(int[] indices)
        {
            float[] RMSDs = new float[NModes];

            for (int m = 0; m < NModes; m++)
            {
                float3[] Original = new float3[indices.Length];
                float3[] Transformed = new float3[indices.Length];
                float3 CenterVolume = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);

                for (int i = 0; i < indices.Length; i++)
                {
                    Original[i] = Atoms[indices[i]] - CenterVolume;
                    Transformed[i] = Original[i] + Modes[m][indices[i]];
                }

                float3 CenterOriginal = MathHelper.Mean(Original);
                float3 CenterTransformed = MathHelper.Mean(Transformed);

                for (int i = 0; i < Original.Length; i++)
                {
                    Original[i] -= CenterOriginal;
                    Transformed[i] -= CenterTransformed;
                }

                Matrix3 R = Matrix3.FromPointSets(Original, Transformed);
                float3 CenterDelta = CenterTransformed - CenterOriginal;

                Matrix4 Transform = Matrix4.Translate(CenterTransformed.X, CenterTransformed.Y, CenterTransformed.Z) *
                                    new Matrix4(R) *
                                    Matrix4.Translate(-CenterOriginal.X, -CenterOriginal.Y, -CenterOriginal.Z);

                float RMSD = 0;
                for (int i = 0; i < Original.Length; i++)
                    RMSD += (Transform * (Original[i] + CenterOriginal) - (Transformed[i] + CenterTransformed)).LengthSq();

                RMSDs[m] = (float)Math.Sqrt(RMSD / Original.Length);
            }

            return RMSDs;
        }

        public NMAMap GetCopy()
        {
            NMAMap Copy = new NMAMap(Atoms.ToArray(),
                                     Intensities.ToArray(),
                                     Sigma,
                                     Modes.Select(a => a.ToArray()).ToArray(),
                                     DimsVolume);

            Copy.NSegments = NSegments;
            if (Segmentation != null) Copy.Segmentation = Segmentation.ToArray();
            if (SegmentIndices != null) Copy.SegmentIndices = SegmentIndices.Select(a => a.ToArray()).ToArray();
            if (SegmentCenters != null) Copy.SegmentCenters = SegmentCenters.ToArray();
            if (SegmentCenteredAtoms != null) Copy.SegmentCenteredAtoms = SegmentCenteredAtoms.Select(a => a.ToArray()).ToArray();
            if (SegmentModes != null) Copy.SegmentModes = SegmentModes.Select(a => a.Select(b => b.ToArray()).ToArray()).ToArray();

            return Copy;
        }
    }
}
