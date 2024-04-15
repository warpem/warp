using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord;
using Warp.Tools;

namespace Warp
{
    public class SoftMap : IDisposable
    {
        public int3 DimsVolume;

        public int NCoarseNodes;
        public int NCoarseEdges;
        public float CoarseSigma;

        public float3[] CoarseNodes;
        private IntPtr d_CoarseNodes;

        public float[] CoarseIntensities;

        public int[] CoarseNeighborIDs;
        private IntPtr d_CoarseNeighborIDs;
        public int[] CoarseNeighborIDOffsets;
        private IntPtr d_CoarseNeighborIDOffsets;
        public float[] CoarseEdgeLengths;
        private IntPtr d_CoarseEdgeLengths;

        public int NActuators;
        public int[] ActuatorNodes;
        public float3[] ActuatorOffsets;
        public int[] ConnectedToActuator;
        private IntPtr d_ConnectedToActuator;

        private int NSimulatedInstances;
        private IntPtr d_LastSimulationResult;

        public int NFineNodes;
        public float FineSigma;
        public float3[] FineNodes;
        private IntPtr d_FineNodes;

        public float[] FineIntensities;
        private IntPtr d_FineIntensities;

        public int NFineNeighbors;
        public int[] FineNodeNeighbors;
        private IntPtr d_FineNodeNeighbors;

        public float[] FineNodeNeighborWeights;
        private IntPtr d_FineNodeNeighborWeights;

        public int NSegments;
        public int[] Segmentation;
        int[][] SegmentIndices;


        public SoftMap(float3[] coarseNodes, float[] coarseIntensities, float lengthCutoff, float3[] actuators, float coarseSigma, int3 dimsVolume)
        {
            DimsVolume = dimsVolume;

            NCoarseNodes = coarseNodes.Length;
            CoarseNodes = coarseNodes.ToList().ToArray();

            CoarseIntensities = coarseIntensities.ToList().ToArray();
            CoarseSigma = coarseSigma;

            lengthCutoff *= lengthCutoff;
            List<int>[] NeighborLists = new List<int>[NCoarseNodes];
            for (int n1 = 0; n1 < NCoarseNodes; n1++)
            {
                float3 Pos1 = CoarseNodes[n1];
                List<int> Neighbors = new List<int>();

                for (int n2 = 0; n2 < NCoarseNodes; n2++)
                {
                    if (n1 == n2)
                        continue;

                    float3 Diff = (Pos1 - CoarseNodes[n2]);
                    if (Diff.LengthSq() <= lengthCutoff)
                        Neighbors.Add(n2);
                }

                NeighborLists[n1] = Neighbors;
            }

            NCoarseEdges = NeighborLists.Select(l => l.Count).Sum();
            CoarseNeighborIDs = new int[NCoarseEdges];
            CoarseNeighborIDOffsets = new int[NCoarseNodes + 1];
            CoarseEdgeLengths = new float[NCoarseEdges];

            for (int n1 = 0, offset = 0; n1 < NCoarseNodes; n1++)
            {
                float3 Pos1 = CoarseNodes[n1];

                List<int> Neighbors = NeighborLists[n1];
                for (int i = 0; i < Neighbors.Count; i++)
                {
                    CoarseNeighborIDs[offset + i] = Neighbors[i];
                    CoarseEdgeLengths[offset + i] = (Pos1 - CoarseNodes[Neighbors[i]]).Length();
                }

                offset += Neighbors.Count;
                CoarseNeighborIDOffsets[n1 + 1] = offset;
            }

            NActuators = actuators.Length;
            ActuatorNodes = new int[NActuators];
            ActuatorOffsets = new float3[NActuators];
            ConnectedToActuator = new int[NCoarseNodes].Select(v => -1).ToArray();

            for (int i = 0; i < NActuators; i++)
            {
                float3 Pos = actuators[i];
                float ClosestDist = float.MaxValue;
                int ClosestNode = 0;

                for (int n = 0; n < NCoarseNodes; n++)
                {
                    float Dist = (Pos - CoarseNodes[n]).LengthSq();
                    if (Dist < ClosestDist)
                    {
                        ClosestDist = Dist;
                        ClosestNode = n;
                    }
                }

                ActuatorNodes[i] = ClosestNode;
                ConnectedToActuator[ClosestNode] = i;
            }
        }

        public float3[] GetDeformationDelta(float3[] actuatorDeltas, int iterations, int relaxationIterations, float stopDelta, out float force, out float strain, out float actuatorStrain)
        {
            float3[] OldDeformed = CoarseNodes.ToList().ToArray();

            Func<bool, Tuple<float, float>> SimulateStep = useActuators =>
            {
                float MeanDelta = 0;
                float MeanStrain = 0;
                float3[] NewDeformed = new float3[NCoarseNodes];
                float3[] Forces = new float3[NCoarseNodes];

                for (int n1 = 0; n1 < NCoarseNodes; n1++)
                {
                    float3 Force = new float3(0, 0, 0);
                    float3 Pos1 = OldDeformed[n1];

                    int NeighborsStart = CoarseNeighborIDOffsets[n1];
                    int NeighborsEnd = CoarseNeighborIDOffsets[n1 + 1];
                    for (int i = NeighborsStart; i < NeighborsEnd; i++)
                    {
                        float3 Pos2 = OldDeformed[CoarseNeighborIDs[i]];
                        float OriLength = CoarseEdgeLengths[i];
                        float3 Diff = Pos1 - Pos2;
                        float CurLength = Diff.Length();

                        float Delta = (OriLength - CurLength) * 0.1f;
                        float3 DiffNorm;
                        if (CurLength > 0)
                            DiffNorm = Diff / CurLength;
                        else
                            DiffNorm = new float3(1e-3f, 0, 0);
                        Force += DiffNorm * Delta;

                        MeanStrain += Math.Abs(OriLength - CurLength);
                    }

                    if (useActuators && ConnectedToActuator[n1] >= 0)
                    {
                        float3 ActuatorPos = CoarseNodes[n1] + actuatorDeltas[ConnectedToActuator[n1]];
                        float3 Diff = ActuatorPos - Pos1;
                        float CurLength = Diff.Length();

                        float Delta = CurLength * 0.2f;
                        float3 DiffNorm = new float3(0, 0, 0);
                        if (CurLength > 0)
                            DiffNorm = Diff / CurLength;
                        Force += DiffNorm * Delta;
                    }

                    NewDeformed[n1] = OldDeformed[n1] + Force;
                    MeanDelta += Force.Length();

                    Forces[n1] = Force;
                }

                OldDeformed = NewDeformed;
                MeanDelta /= NCoarseNodes;
                MeanStrain /= NCoarseEdges;

                return new Tuple<float, float>(MeanDelta, MeanStrain);
            };

            int Iteration = 0;
            while (true)
            {
                Tuple<float, float> Result = SimulateStep(true);

                force = Result.Item1;
                strain = Result.Item2;

                if (iterations <= 0 && stopDelta <= 0)
                    break; // Otherwise there is no end

                if (++Iteration >= iterations && iterations > 0)
                    break;
                if (Result.Item1 <= stopDelta)
                    break;
            }

            for (int i = 0; i < relaxationIterations; i++)
            {
                Tuple<float, float> Result = SimulateStep(false);
                force = Result.Item1;
                strain = Result.Item2;
            }

            actuatorStrain = 0;
            for (int a = 0; a < NActuators; a++)
                actuatorStrain += (OldDeformed[ActuatorNodes[a]] - (CoarseNodes[ActuatorNodes[a]] + actuatorDeltas[a])).Length();
            actuatorStrain /= NActuators;

            float3[] Deltas = new float3[NCoarseNodes];
            for (int n = 0; n < NCoarseNodes; n++)
                Deltas[n] = OldDeformed[n] - CoarseNodes[n];

            return Deltas;
        }

        public void DeformMultipleInstances(float3[] actuatorDeltas, int instances, int iterations, int relaxationIterations)
        {
            if (!IsOnDevice())
                PutOnDevice();

            if (d_LastSimulationResult != IntPtr.Zero)
                GPU.FreeDevice(d_LastSimulationResult);

            NSimulatedInstances = instances;
            d_LastSimulationResult = GPU.MallocDevice(NCoarseNodes * instances * 3);

            IntPtr d_ActuatorDeltas = GPU.MallocDeviceFromHost(Helper.ToInterleaved(actuatorDeltas), actuatorDeltas.Length * 3);

            GPU.ParticleSoftBodyDeform(d_CoarseNodes,
                                       d_LastSimulationResult,
                                       (uint)NCoarseNodes,
                                       d_CoarseNeighborIDs,
                                       d_CoarseNeighborIDOffsets,
                                       d_CoarseEdgeLengths,
                                       d_ConnectedToActuator,
                                       d_ActuatorDeltas,
                                       (uint)NActuators,
                                       (uint)iterations,
                                       (uint)relaxationIterations,
                                       (uint)instances);

            GPU.FreeDevice(d_ActuatorDeltas);
        }

        public float3[][] GetLastSimulationResults()
        {
            if (d_LastSimulationResult == IntPtr.Zero)
                return null;

            float[] RawData = new float[NCoarseNodes * 3 * NSimulatedInstances];
            GPU.CopyDeviceToHost(d_LastSimulationResult, RawData, RawData.Length);

            float3[][] Result = new float3[NSimulatedInstances][];
            for (int n = 0; n < NSimulatedInstances; n++)
                Result[n] = Helper.FromInterleaved3(RawData.Skip(n * NCoarseNodes * 3).Take(NCoarseNodes * 3).ToArray());

            return Result;
        }

        public void AttachFineNodes(float3[] fineNodes, float[] fineIntensities, float fineSigma, int nFineNeighbors)
        {
            if (IsFineOnDevice())
                DisposeFine();

            NFineNodes = fineNodes.Length;
            NFineNeighbors = nFineNeighbors;

            FineNodes = fineNodes.ToList().ToArray();
            FineIntensities = fineIntensities.ToList().ToArray();
            FineSigma = fineSigma;

            FineNodeNeighbors = new int[NFineNodes * NFineNeighbors];
            FineNodeNeighborWeights = new float[NFineNodes * NFineNeighbors];

            float CoarseSigma2 = CoarseSigma * CoarseSigma * 2;

            Parallel.For(0, NFineNodes, nf =>
            {
                float3 Pos1 = FineNodes[nf];

                float[] Distances = new float[NCoarseNodes];
                for (int nc = 0; nc < NCoarseNodes; nc++)
                    Distances[nc] = (Pos1 - CoarseNodes[nc]).LengthSq();
                int[] ClosestIDs;
                float[] ClosestDist = MathHelper.TakeNLowest(Distances, NFineNeighbors, out ClosestIDs);
                float MaxDist = MathHelper.Max(ClosestDist);

                float WeightSum = 0;
                for (int i = 0; i < NFineNeighbors; i++)
                {
                    float Weight = (float)Math.Exp(-ClosestDist[i] / CoarseSigma2);
                    //float Weight = Math.Max(0, 1 - ClosestDist[i]);
                    FineNodeNeighborWeights[nf * NFineNeighbors + i] = Weight;
                    WeightSum += Weight;

                    FineNodeNeighbors[nf * NFineNeighbors + i] = ClosestIDs[i];
                }

                for (int i = 0; i < NFineNeighbors; i++)
                    FineNodeNeighborWeights[nf * NFineNeighbors + i] /= WeightSum;
            });
        }

        public Image RasterizeDeformedInVolumeCoarse(int3 dims, float3[] deformationDeltas)
        {
            if (deformationDeltas.Length != NCoarseNodes)
                throw new DimensionMismatchException();

            float3 OldCenter = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);
            float3 NewCenter = new float3(dims.X / 2, dims.Y / 2, dims.Z / 2);
            float3 Offset = NewCenter - OldCenter;

            float3[] OffsetAtoms = new float3[NCoarseNodes];

            for (int n = 0; n < NCoarseNodes; n++)
                OffsetAtoms[n] = CoarseNodes[n] + Offset + deformationDeltas[n];

            return PhysicsHelper.GetVolumeFromAtoms(OffsetAtoms, dims, CoarseSigma, CoarseIntensities);
        }

        public Image RasterizeDeformedInVolumeFine(int3 dims, float3[] deformationDeltas)
        {
            if (deformationDeltas.Length != NCoarseNodes)
                throw new DimensionMismatchException();

            float3 OldCenter = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);
            float3 NewCenter = new float3(dims.X / 2, dims.Y / 2, dims.Z / 2);
            float3 Offset = NewCenter - OldCenter;

            float3[] OffsetAtoms = new float3[NFineNodes];

            for (int n = 0; n < NFineNodes; n++)
            {
                float3 WeightedOffset = new float3(0, 0, 0);
                for (int i = 0; i < NFineNeighbors; i++)
                    WeightedOffset += deformationDeltas[FineNodeNeighbors[n * NFineNeighbors + i]] * FineNodeNeighborWeights[n * NFineNeighbors + i];

                OffsetAtoms[n] = FineNodes[n] + Offset + WeightedOffset;
            }

            return PhysicsHelper.GetVolumeFromAtoms(OffsetAtoms, dims, FineSigma, FineIntensities);
        }

        public Image ProjectLastSimulation(int2 dims, float3[] angles, float2[] shifts, float scale)
        {
            if (d_LastSimulationResult == IntPtr.Zero)
                throw new Exception("No simulation data avaialble.");

            if (angles.Length != NSimulatedInstances || shifts.Length != NSimulatedInstances)
                throw new Exception("Number of simulated instances doesn't match parameters.");

            Image Proj = new Image(IntPtr.Zero, new int3(dims.X, dims.Y, NSimulatedInstances));

            if (!IsFineOnDevice())
                PutFineOnDevice();

            GPU.ProjectSoftPseudoAtoms(d_FineNodes,
                                       d_FineIntensities,
                                       (uint)NFineNodes,
                                       DimsVolume,
                                       FineSigma,
                                       (uint)Math.Ceiling(FineSigma * 3 * scale),
                                       d_LastSimulationResult,
                                       d_FineNodeNeighborWeights,
                                       d_FineNodeNeighbors,
                                       (uint)NFineNeighbors,
                                       (uint)NCoarseNodes,
                                       Helper.ToInterleaved(angles),
                                       Helper.ToInterleaved(shifts),
                                       scale,
                                       Proj.GetDevice(Intent.Write),
                                       dims,
                                       (uint)NSimulatedInstances);

            return Proj;
        }

        bool IsOnDevice()
        {
            return IsCoarseOnDevice() && (FineNodes == null || IsFineOnDevice());
        }

        bool IsCoarseOnDevice()
        {
            return d_CoarseNodes != IntPtr.Zero &&
                   d_CoarseNeighborIDs != IntPtr.Zero &&
                   d_CoarseNeighborIDOffsets != IntPtr.Zero &&
                   d_CoarseEdgeLengths != IntPtr.Zero &&
                   d_ConnectedToActuator != IntPtr.Zero;
        }

        bool IsFineOnDevice()
        {
            return d_FineIntensities != IntPtr.Zero &&
                   d_FineNodes != IntPtr.Zero &&
                   d_FineNodeNeighbors != IntPtr.Zero &&
                   d_FineNodeNeighborWeights != IntPtr.Zero;
        }

        void PutOnDevice()
        {
            PutCoarseOnDevice();

            if (FineNodes != null)
                PutFineOnDevice();
        }

        void PutCoarseOnDevice()
        {
            DisposeCoarse();

            d_CoarseNodes = GPU.MallocDeviceFromHost(Helper.ToInterleaved(CoarseNodes), NCoarseNodes * 3);
            d_CoarseNeighborIDs = GPU.MallocDeviceFromHostInt(CoarseNeighborIDs, CoarseNeighborIDs.Length);
            d_CoarseNeighborIDOffsets = GPU.MallocDeviceFromHostInt(CoarseNeighborIDOffsets, CoarseNeighborIDOffsets.Length);
            d_CoarseEdgeLengths = GPU.MallocDeviceFromHost(CoarseEdgeLengths, CoarseEdgeLengths.Length);
            d_ConnectedToActuator = GPU.MallocDeviceFromHostInt(ConnectedToActuator, ConnectedToActuator.Length);
        }

        void PutFineOnDevice()
        {
            if (FineNodes == null)
                throw new Exception("No fine data available.");

            d_FineNodes = GPU.MallocDeviceFromHost(Helper.ToInterleaved(FineNodes), NFineNodes * 3);
            d_FineIntensities = GPU.MallocDeviceFromHost(FineIntensities, NFineNodes);
            d_FineNodeNeighbors = GPU.MallocDeviceFromHostInt(FineNodeNeighbors, NFineNodes * NFineNeighbors);
            d_FineNodeNeighborWeights = GPU.MallocDeviceFromHost(FineNodeNeighborWeights, NFineNodes * NFineNeighbors);
        }

        public void Dispose()
        {
            DisposeCoarse();
            DisposeFine();
        }

        void DisposeCoarse()
        {
            if (d_CoarseNodes != IntPtr.Zero)
            {
                GPU.FreeDevice(d_CoarseNodes);
                d_CoarseNodes = IntPtr.Zero;
            }
            if (d_CoarseNeighborIDs != IntPtr.Zero)
            {
                GPU.FreeDevice(d_CoarseNeighborIDs);
                d_CoarseNeighborIDs = IntPtr.Zero;
            }
            if (d_CoarseNeighborIDOffsets != IntPtr.Zero)
            {
                GPU.FreeDevice(d_CoarseNeighborIDOffsets);
                d_CoarseNeighborIDOffsets = IntPtr.Zero;
            }
            if (d_CoarseEdgeLengths != IntPtr.Zero)
            {
                GPU.FreeDevice(d_CoarseEdgeLengths);
                d_CoarseEdgeLengths = IntPtr.Zero;
            }
            if (d_ConnectedToActuator != IntPtr.Zero)
            {
                GPU.FreeDevice(d_ConnectedToActuator);
                d_ConnectedToActuator = IntPtr.Zero;
            }
            if (d_LastSimulationResult != IntPtr.Zero)
            {
                GPU.FreeDevice(d_LastSimulationResult);
                d_LastSimulationResult = IntPtr.Zero;
            }
        }

        void DisposeFine()
        {
            if (d_FineNodes != IntPtr.Zero)
            {
                GPU.FreeDevice(d_FineNodes);
                d_FineNodes = IntPtr.Zero;
            }
            if (d_FineIntensities != IntPtr.Zero)
            {
                GPU.FreeDevice(d_FineIntensities);
                d_FineIntensities = IntPtr.Zero;
            }
            if (d_FineNodeNeighbors != IntPtr.Zero)
            {
                GPU.FreeDevice(d_FineNodeNeighbors);
                d_FineNodeNeighbors = IntPtr.Zero;
            }
            if (d_FineNodeNeighborWeights != IntPtr.Zero)
            {
                GPU.FreeDevice(d_FineNodeNeighborWeights);
                d_FineNodeNeighborWeights = IntPtr.Zero;
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

            for (int a = 0; a < NCoarseNodes; a++)
            {
                int3 Coords = new int3(CoarseNodes[a]);
                int Partition = Segmentation[segments.Dims.ElementFromPosition(Coords)];
                if (Partition >= 0)
                    IndexLists[Partition].Add(a);
            }

            SegmentIndices = IndexLists.Select(l => l.ToArray()).ToArray();
        }

        public Matrix4[] GetSegmentTransforms(int segment)
        {
            if (d_LastSimulationResult == IntPtr.Zero)
                throw new Exception("No simulation results available.");

            Matrix4[] Result = new Matrix4[NSimulatedInstances];
            float3[][] DeformationDeltas = GetLastSimulationResults();

            for (int instance = 0; instance < NSimulatedInstances; instance++)
            {
                int[] Indices = SegmentIndices[segment];
                float3[] Original = new float3[Indices.Length];
                float3[] Transformed = new float3[Indices.Length];
                float3 CenterVolume = new float3(DimsVolume.X / 2, DimsVolume.Y / 2, DimsVolume.Z / 2);

                for (int i = 0; i < Indices.Length; i++)
                {
                    Original[i] = CoarseNodes[Indices[i]] - CenterVolume;

                    Transformed[i] = Original[i] + DeformationDeltas[instance][i];
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

                Result[instance] = Transform;
            }

            return Result;
        }
    }
}
