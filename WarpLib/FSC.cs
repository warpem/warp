using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using BitMiracle.LibTiff.Classic;
using Warp.Tools;
using ZLinq;

namespace Warp
{
    public static class FSC
    {
        public static float[] GetFSC(Image volume1, Image volume2)
        {
            Image HalfFT1 = volume1.IsComplex ? volume1 : volume1.AsFFT(true);
            Image HalfFT2 = volume2.IsComplex ? volume2 : volume2.AsFFT(true);

            int NShells = volume1.Dims.X / 2;

            double[] Nums = new double[NShells];
            double[] Denoms1 = new double[NShells];
            double[] Denoms2 = new double[NShells];

            float[] Vol1 = HalfFT1.GetHostContinuousCopy();
            float[] Vol2 = HalfFT2.GetHostContinuousCopy();

            int3 DimsFT = HalfFT1.DimsFT;

            Parallel.For(0, 20, p =>
            {
                double[] ThreadNums = new double[NShells];
                double[] ThreadDenoms1 = new double[NShells];
                double[] ThreadDenoms2 = new double[NShells];

                for (int z = p; z < DimsFT.Z; z += 20)
                {
                    int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                    zz *= zz;

                    for (int y = 0; y < DimsFT.Y; y++)
                    {
                        int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                        yy *= yy;

                        for (int x = 0; x < DimsFT.X; x++)
                        {
                            int xx = x;
                            xx *= x;

                            float R = (float)Math.Sqrt(zz + yy + xx);
                            if (R >= NShells)
                                continue;

                            int ID = (int)R;
                            float W1 = R - ID;
                            float W0 = 1f - W1;

                            int i = (z * DimsFT.Y + y) * DimsFT.X + x;

                            float Nom = Vol1[i * 2] * Vol2[i * 2] + Vol1[i * 2 + 1] * Vol2[i * 2 + 1];
                            float Denom1 = Vol1[i * 2] * Vol1[i * 2] + Vol1[i * 2 + 1] * Vol1[i * 2 + 1];
                            float Denom2 = Vol2[i * 2] * Vol2[i * 2] + Vol2[i * 2 + 1] * Vol2[i * 2 + 1];

                            ThreadNums[ID] += W0 * Nom;
                            ThreadDenoms1[ID] += W0 * Denom1;
                            ThreadDenoms2[ID] += W0 * Denom2;

                            if (ID < NShells - 1)
                            {
                                ThreadNums[ID + 1] += W1 * Nom;
                                ThreadDenoms1[ID + 1] += W1 * Denom1;
                                ThreadDenoms2[ID + 1] += W1 * Denom2;
                            }
                        }
                    }
                }

                lock (Nums)
                    for (int i = 0; i < Nums.Length; i++)
                    {
                        Nums[i] += ThreadNums[i];
                        Denoms1[i] += ThreadDenoms1[i];
                        Denoms2[i] += ThreadDenoms2[i];
                    }
            });

            float[] Result = new float[NShells];

            for (int i = 0; i < Result.Length; i++)
                Result[i] = (float)(Nums[i] / Math.Max(1e-6f, Math.Sqrt(Denoms1[i] * Denoms2[i])));

            if (HalfFT1 != volume1)
                HalfFT1.Dispose();
            if (HalfFT2 != volume2)
                HalfFT2.Dispose();

            return Result;
        }

        public static float[] GetFSCNonCubic(Image volume1, Image volume2, Image fourierMask, int nshells)
        {
            Image HalfFT1 = volume1.IsComplex ? volume1 : volume1.AsFFT(true);
            Image HalfFT2 = volume2.IsComplex ? volume2 : volume2.AsFFT(true);

            int NShells = nshells;

            double[] Nums = new double[NShells];
            double[] Denoms1 = new double[NShells];
            double[] Denoms2 = new double[NShells];

            float[] Vol1 = HalfFT1.GetHostContinuousCopy();
            float[] Vol2 = HalfFT2.GetHostContinuousCopy();
            float[] Mask = fourierMask != null ? fourierMask.GetHostContinuousCopy() : null;

            int3 DimsFT = HalfFT1.DimsFT;

            Parallel.For(0, 20, p =>
            {
                double[] ThreadNums = new double[NShells];
                double[] ThreadDenoms1 = new double[NShells];
                double[] ThreadDenoms2 = new double[NShells];

                for (int z = p; z < DimsFT.Z; z += 20)
                {
                    float zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                    zz /= volume1.Dims.Z / 2;
                    zz *= zz;

                    for (int y = 0; y < DimsFT.Y; y++)
                    {
                        float yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                        yy /= volume1.Dims.Y / 2;
                        yy *= yy;

                        for (int x = 0; x < DimsFT.X; x++)
                        {
                            float xx = x;
                            xx /= volume1.Dims.X / 2;
                            xx *= xx;

                            float R = (float)Math.Sqrt(zz + yy + xx) * NShells;
                            if (R >= NShells)
                                continue;

                            int ID = (int)R;
                            float W1 = R - ID;
                            float W0 = 1f - W1;

                            int i = (z * DimsFT.Y + y) * DimsFT.X + x;

                            float Nom = Vol1[i * 2] * Vol2[i * 2] + Vol1[i * 2 + 1] * Vol2[i * 2 + 1];
                            float Denom1 = Vol1[i * 2] * Vol1[i * 2] + Vol1[i * 2 + 1] * Vol1[i * 2 + 1];
                            float Denom2 = Vol2[i * 2] * Vol2[i * 2] + Vol2[i * 2 + 1] * Vol2[i * 2 + 1];

                            if (Mask != null)
                            {
                                float m = Mask[i];
                                m *= m;
                                Nom *= m;
                                Denom1 *= m;
                                Denom2 *= m;
                            }

                            ThreadNums[ID] += W0 * Nom;
                            ThreadDenoms1[ID] += W0 * Denom1;
                            ThreadDenoms2[ID] += W0 * Denom2;

                            if (ID < NShells - 1)
                            {
                                ThreadNums[ID + 1] += W1 * Nom;
                                ThreadDenoms1[ID + 1] += W1 * Denom1;
                                ThreadDenoms2[ID + 1] += W1 * Denom2;
                            }
                        }
                    }
                }

                lock (Nums)
                    for (int i = 0; i < Nums.Length; i++)
                    {
                        Nums[i] += ThreadNums[i];
                        Denoms1[i] += ThreadDenoms1[i];
                        Denoms2[i] += ThreadDenoms2[i];
                    }
            });

            float[] Result = new float[NShells];

            for (int i = 0; i < Result.Length; i++)
                Result[i] = (float)(Nums[i] / Math.Max(1e-6f, Math.Sqrt(Denoms1[i] * Denoms2[i])));

            if (HalfFT1 != volume1)
                HalfFT1.Dispose();
            if (HalfFT2 != volume2)
                HalfFT2.Dispose();

            return Result;
        }

        public static float[] GetAmps1D(Image volume)
        {
            Image VolumeFT = volume.IsComplex ? volume : volume.AsFFT(true);
            Image VolumeAbs = VolumeFT.AsAmplitudes();
            if (VolumeFT != volume)
                VolumeFT.Dispose();

            int NShells = volume.Dims.X / 2;

            float[] PS1D = new float[NShells];
            float[] Samples = new float[NShells];

            float[] VolumeAbsData = VolumeAbs.GetHostContinuousCopy();

            int3 DimsFT = VolumeAbs.DimsFT;

            Parallel.For(0, 20, p =>
            {
                float[] ThreadPS1D = new float[NShells];
                float[] ThreadSamples = new float[NShells];

                for (int z = p; z < DimsFT.Z; z += 20)
                {
                    int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                    zz *= zz;

                    for (int y = 0; y < DimsFT.Y; y++)
                    {
                        int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                        yy *= yy;

                        for (int x = 0; x < DimsFT.X; x++)
                        {
                            int xx = x;
                            xx *= x;

                            float R = (float)Math.Sqrt(zz + yy + xx);
                            if (R >= NShells)
                                continue;

                            int ID = (int)R;
                            float W1 = R - ID;
                            float W0 = 1f - W1;

                            int i = (z * DimsFT.Y + y) * DimsFT.X + x;

                            float Amp = VolumeAbsData[i];

                            ThreadPS1D[ID] += W0 * Amp;
                            ThreadSamples[ID] += W0;

                            if (ID < NShells - 1)
                            {
                                ThreadPS1D[ID + 1] += W1 * Amp;
                                ThreadSamples[ID + 1] += W1;
                            }
                        }
                    }
                }

                lock (PS1D)
                    for (int i = 0; i < PS1D.Length; i++)
                    {
                        PS1D[i] += ThreadPS1D[i];
                        Samples[i] += ThreadSamples[i];
                    }
            });

            float[] Result = new float[NShells];

            for (int i = 0; i < Result.Length; i++)
                Result[i] = PS1D[i] / Math.Max(1e-6f, Samples[i]);

            return Result;
        }

        public static float GetCutoffShell(float[] fsc, float threshold)
        {
            if (fsc[0] <= threshold)
                return 0;

            int Shell = -1;

            for (int i = 0; i < fsc.Length; i++, Shell++)
                if (fsc[i] <= threshold)
                    break;

            if (Shell >= fsc.Length - 1)
                return Shell;

            float Frac = (fsc[Shell] - threshold) / Math.Max(1e-5f, fsc[Shell] - fsc[Shell + 1]);

            return Shell + Frac;
        }

        public static Image ApplyRamp(Image volume, float[] ramp)
        {
            Image VolumeFT = volume.IsComplex ? volume : volume.AsFFT(true);
            Image Filter = new Image(VolumeFT.Dims, true);
            float[][] FilterData = Filter.GetHost(Intent.Write);

            int3 DimsFT = Filter.DimsFT;
            for (int z = 0; z < DimsFT.Z; z++)
            {
                int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                zz *= zz;

                for (int y = 0, i = 0; y < DimsFT.Y; y++)
                {
                    int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                    yy *= yy;

                    for (int x = 0; x < DimsFT.X; x++, i++)
                    {
                        int xx = x;
                        xx *= x;

                        float R = (float)Math.Sqrt(zz + yy + xx);

                        int ID = (int)R;
                        float W1 = R - ID;
                        float W0 = 1f - W1;

                        FilterData[z][i] = W0 * ramp[Math.Min(ramp.Length - 1, ID)] + W1 * ramp[Math.Min(ramp.Length - 1, ID + 1)];
                    }
                }
            }

            VolumeFT.Multiply(Filter);
            Image Result = VolumeFT.AsIFFT(true, 0, false, volume == VolumeFT);

            Filter.Dispose();
            if (volume != VolumeFT)
                VolumeFT.Dispose();

            return Result;
        }

        public static void AverageLowFrequencies(Image volume1, Image volume2, int maxshell, out Image averaged1, out Image averaged2)
        {
            Image VolumeFT1 = volume1.AsFFT(true);
            volume1.FreeDevice();
            Image VolumeFT2 = volume2.AsFFT(true);
            volume2.FreeDevice();

            float[][] VolumeFT1Data = VolumeFT1.GetHost(Intent.ReadWrite);
            float[][] VolumeFT2Data = VolumeFT2.GetHost(Intent.ReadWrite);
            int3 DimsFT = VolumeFT1.DimsFT;
            Parallel.For(0, DimsFT.Z, z =>
            {
                int zz = z < DimsFT.Z / 2 + 1 ? z : z - DimsFT.Z;
                zz *= zz;

                for (int y = 0, i = 0; y < DimsFT.Y; y++)
                {
                    int yy = y < DimsFT.Y / 2 + 1 ? y : y - DimsFT.Y;
                    yy *= yy;

                    for (int x = 0; x < DimsFT.X; x++, i++)
                    {
                        int xx = x;
                        xx *= x;

                        if (xx + yy + zz < maxshell * maxshell)
                        {
                            float2 Average = new float2(VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 0] + VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 0],
                                                        VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 1] + VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 1]) * 0.5f;

                            VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 0] = Average.X;
                            VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 0] = Average.X;
                            VolumeFT1Data[z][(y * DimsFT.X + x) * 2 + 1] = Average.Y;
                            VolumeFT2Data[z][(y * DimsFT.X + x) * 2 + 1] = Average.Y;
                        }
                    }
                }
            });

            averaged1 = VolumeFT1.AsIFFT(true, -1, true);
            averaged2 = VolumeFT2.AsIFFT(true, -1, true);

            VolumeFT1.Dispose();
            VolumeFT2.Dispose();
        }

        public static Image MakeSoftMask(Image binaryMask, int expand, int smooth)
        {
            Image BinaryExpanded;
            if (expand > 0)
            {
                BinaryExpanded = binaryMask.AsDistanceMapExact(expand);
                BinaryExpanded.Multiply(-1);
                BinaryExpanded.Binarize(-expand + 1e-6f);
            }
            else
            {
                BinaryExpanded = binaryMask.GetCopyGPU();
            }
            binaryMask.FreeDevice();

            Image ExpandedSmooth;
            if (smooth > 0)
            {
                ExpandedSmooth = BinaryExpanded.AsDistanceMapExact(smooth);
                ExpandedSmooth.Multiply((float)Math.PI / smooth);
                ExpandedSmooth.Cos();
                ExpandedSmooth.Add(1);
                ExpandedSmooth.Multiply(0.5f);
            }
            else
            {
                ExpandedSmooth = BinaryExpanded.GetCopyGPU();
            }
            BinaryExpanded.Dispose();

            return ExpandedSmooth;
        }

        public static void GetCorrectedFSC(Image volume1,
                                           Image volume2,
                                           Image mask,
                                           float threshold,
                                           float thresholdRandomize,
                                           out float globalShell,
                                           out float[] fscUnmasked,
                                           out float[] fscMasked,
                                           out float[] fscRandomized,
                                           out float[] fscCorrected)
        {
            int RandomizeAfter = 1;

            // FSC unmasked
            {
                fscUnmasked = GetFSC(volume1, volume2);
                while (fscUnmasked[RandomizeAfter] > thresholdRandomize && RandomizeAfter < fscUnmasked.Length - 1)
                    RandomizeAfter++;
            }

            // FSC masked
            {
                Image Masked1 = volume1.GetCopyGPU();
                volume1.FreeDevice();
                Masked1.Multiply(mask);

                Image Masked2 = volume2.GetCopyGPU();
                volume2.FreeDevice();
                Masked2.Multiply(mask);

                fscMasked = GetFSC(Masked1, Masked2);

                Masked1.Dispose();
                Masked2.Dispose();
            }

            // FSC randomized masked
            {
                Random Rand = new Random(RandomizeAfter);

                Image Fuzz1 = new Image(volume1.Dims, true, true);
                Image Fuzz2 = new Image(volume2.Dims, true, true);
                float[][] Fuzz1Data = Fuzz1.GetHost(Intent.Write);
                float[][] Fuzz2Data = Fuzz2.GetHost(Intent.Write);
                int3 DimsFT = Fuzz1.DimsFT;
                Helper.ForEachElementFT(volume1.Dims, (x, y, z, xx, yy, zz, r) =>
                {
                    double Phase1 = Rand.NextDouble() * Math.PI * 2;
                    double Phase2 = Rand.NextDouble() * Math.PI * 2;
                    long i = y * DimsFT.X + x;

                    Fuzz1Data[z][i * 2 + 0] = r < RandomizeAfter ? 1f : (float)Math.Cos(Phase1);
                    Fuzz1Data[z][i * 2 + 1] = r < RandomizeAfter ? 0f : (float)Math.Sin(Phase1);

                    Fuzz2Data[z][i * 2 + 0] = r < RandomizeAfter ? 1f : (float)Math.Cos(Phase2);
                    Fuzz2Data[z][i * 2 + 1] = r < RandomizeAfter ? 0f : (float)Math.Sin(Phase2);
                });

                Image Randomized1FT = volume1.AsFFT(true).AndFreeParent();
                Randomized1FT.Multiply(Fuzz1);
                Fuzz1.Dispose();
                Image Randomized1 = Randomized1FT.AsIFFT(true).AndDisposeParent();
                Randomized1.Multiply(mask);

                Image Randomized2FT = volume2.AsFFT(true).AndFreeParent();
                Randomized2FT.Multiply(Fuzz2);
                Fuzz2.Dispose();
                Image Randomized2 = Randomized2FT.AsIFFT(true).AndDisposeParent();
                Randomized2.Multiply(mask);

                fscRandomized = GetFSC(Randomized1, Randomized2);
                Randomized1.Dispose();
                Randomized2.Dispose();
            }

            // FSC corrected
            {
                fscMasked[0] = 1;
                fscUnmasked[0] = 1;
                fscCorrected = new float[fscUnmasked.Length];

                for (int i = 0; i < fscCorrected.Length; i++)
                {
                    if (i < RandomizeAfter + 2)
                        fscCorrected[i] = fscMasked[i];
                    else
                        fscCorrected[i] = (fscMasked[i] - fscRandomized[i]) / (1 - fscRandomized[i]);
                }
            }

            globalShell = GetCutoffShell(fscCorrected, threshold);
        }

        public static void GetWeightedSharpened(Image volume,
                                                float[] fsc,
                                                float angpix,
                                                float fitWorstRes,
                                                float fitBestRes,
                                                int lowpassShell,
                                                out float globalBFactor,
                                                out float fitQuality,
                                                out Image volumeWeighted,
                                                out Image volumeSharpened)
        {
            Image VolumeFT = volume.AsFFT(true);
            VolumeFT.Multiply(1f / volume.Dims.Elements());
            volume.FreeDevice();

            float[] Amps1D = GetAmps1D(VolumeFT);
            int NShells = Amps1D.Length;

            // Apply weighting
            int ResMax = NShells - 1;
            float[] FSCWeighting = new float[NShells];
            for (int i = 0; i < NShells; i++)
            {
                if (fsc[i] < 1e-4f)
                    ResMax = Math.Min(i, ResMax);
                FSCWeighting[i] = i >= ResMax ? 0 : (float)Math.Sqrt(Math.Max(0, 2 * fsc[i] / (1 + fsc[i])));
                Amps1D[i] *= FSCWeighting[i];
            }

            // Make weighted Guinier plot
            float[] ResInv = Helper.ArrayOfFunction(i => i / ((float)volume.Dims.X * angpix), NShells);
            float3[] WeightedPoints = Helper.ArrayOfFunction(i => new float3(ResInv[i] * ResInv[i],
                                                                             (float)Math.Log(Math.Max(1e-20f, Amps1D[i])),
                                                                             1f / ResInv[i] < fitWorstRes &&
                                                                             1f / ResInv[i] >= fitBestRes &&
                                                                             i < ResMax
                                                                                 ? 1
                                                                                 : 0), NShells);

            // Fit line through plot
            {
                float ss_xy = 0;
                float ss_xx = 0;
                float ss_yy = 0;
                float ave_x = 0;
                float ave_y = 0;
                float sum_w = 0;
                for (int i = 1; i < ResMax; i++)
                {
                    ave_x += WeightedPoints[i].Z * WeightedPoints[i].X;
                    ave_y += WeightedPoints[i].Z * WeightedPoints[i].Y;
                    sum_w += WeightedPoints[i].Z;
                    ss_xx += WeightedPoints[i].Z * WeightedPoints[i].X * WeightedPoints[i].X;
                    ss_yy += WeightedPoints[i].Z * WeightedPoints[i].Y * WeightedPoints[i].Y;
                    ss_xy += WeightedPoints[i].Z * WeightedPoints[i].X * WeightedPoints[i].Y;
                }
                ave_x /= sum_w;
                ave_y /= sum_w;
                ss_xx -= sum_w * ave_x * ave_x;
                ss_yy -= sum_w * ave_y * ave_y;
                ss_xy -= sum_w * ave_x * ave_y;

                if (ss_xx > 0)
                {
                    globalBFactor = 4 * ss_xy / ss_xx;
                    fitQuality = ss_xy * ss_xy / (ss_xx * ss_yy);
                }
                else
                {
                    globalBFactor = fitQuality = 0;
                }
            }

            // Calculate ramp as FSCweight * -Bfac * lowpass
            float[] FilterWeight = new float[NShells];
            float[] FilterSharpen = new float[NShells];
            for (int i = 0; i < NShells; i++)
            {
                FilterWeight[i] = FSCWeighting[i];
                FilterWeight[i] *= 1f - Math.Max(0, Math.Min(1, (i - (lowpassShell - 1)) / 2f));

                FilterSharpen[i] = FilterWeight[i] * (float)Math.Exp(-globalBFactor / 4 * ResInv[i] * ResInv[i]);
            }

            volumeWeighted = ApplyRamp(VolumeFT, FilterWeight);
            volumeSharpened = ApplyRamp(VolumeFT, FilterSharpen);

            VolumeFT.Dispose();
        }

        public static Image GetAnisotropicFSC(Image volume1,
                                              Image volume2,
                                              Image mask,
                                              float pixelsize,
                                              float threshold,
                                              int healpixOrder,
                                              float scaleToGlobalShell)
        {
            float MaskFraction = mask.GetHostContinuousCopy().Sum() / mask.ElementsReal;

            Image Masked1 = volume1.GetCopyGPU();
            volume1.FreeDevice();
            Masked1.Multiply(mask);
            Image Volume1FT = Masked1.AsFFT(true);
            Masked1.Dispose();

            Image Masked2 = volume2.GetCopyGPU();
            Masked2.Multiply(mask);
            volume2.FreeDevice();
            Image Volume2FT = Masked2.AsFFT(true);
            Masked2.Dispose();

            mask.FreeDevice();

            float2[] Angles = Helper.GetHealpixRotTilt(healpixOrder, "C1", 91f);
            float3[] Directions = Angles.Select(a => Matrix3.Euler(a.X * Helper.ToRad, a.Y * Helper.ToRad, 0).Transposed() * float3.UnitZ).ToArray();

            float AngleSpacing = (float)(60.0 / Math.Pow(2, Math.Min(2, healpixOrder)));

            float[] ConeShells = new float[Directions.Length];

            CPU.ConicalFSC(Volume1FT.GetHostContinuousCopy(),
                           Volume2FT.GetHostContinuousCopy(),
                           volume1.Dims,
                           Helper.ToInterleaved(Directions),
                           Directions.Length,
                           AngleSpacing,
                           10,
                           threshold,
                           MaskFraction,
                           ConeShells);

            Volume1FT.Dispose();
            Volume2FT.Dispose();

            float MeanShell = MathHelper.MeanWeighted(ConeShells, MathHelper.Mult(ConeShells, ConeShells));
            ConeShells = ConeShells.Select(v => v / MeanShell * scaleToGlobalShell).ToArray();

            int NRot = (int)Math.Ceiling(360f / AngleSpacing * 2);
            float StepRot = 360f / NRot;
            int NTilt = (int)Math.Ceiling(90f / AngleSpacing * 2);
            float StepTilt = 90f / NTilt;
            float[] PolarValues = new float[NRot * NTilt];

            for (int t = 0; t < NTilt; t++)
            {
                float Tilt = t * StepTilt;
                for (int r = 0; r < NRot; r++)
                {
                    float Rot = r * StepRot;

                    float3 Direction = Matrix3.Euler(Rot * Helper.ToRad, Tilt * Helper.ToRad, 0).Transposed() * float3.UnitZ;

                    float WeightedShell = 0;
                    float Weights = 0;
                    for (int d = 0; d < Directions.Length; d++)
                    {
                        float AngleDiff = (float)Math.Acos(Math.Min(1, Math.Abs(float3.Dot(Directions[d], Direction)))) * Helper.ToDeg;
                        float Weight = 1 - Math.Min(1, AngleDiff / (AngleSpacing * 2));
                        WeightedShell += ConeShells[d] * Weight;
                        Weights += Weight;
                    }

                    PolarValues[t * NRot + r] = volume1.Dims.X * pixelsize / (WeightedShell / Weights);
                }
            }

            return new Image(PolarValues, new int3(NRot, NTilt, 1));
        }

        public static Image FilterToAnisotropicResolution(Image volume,
                                                          Image polarResolution,
                                                          float pixelsize)
        {
            Image VolumeFT = volume.AsFFT(true);
            float2[][] VolumeFTData = VolumeFT.GetHostComplexCopy();
            VolumeFT.Dispose();

            float[] ShellData = polarResolution.GetHostContinuousCopy();
            ShellData = ShellData.Select(v => pixelsize / v * volume.Dims.X).ToArray();

            Helper.ForEachElementFT(volume.Dims, (x, y, z, xx, yy, zz, r) =>
            {
                if (r == 0)
                    return;

                if (zz < 0)
                {
                    xx = -xx;
                    yy = -yy;
                    zz = -zz;
                }

                float fx = xx / r;
                float fy = yy / r;
                float fz = zz / r;

                float Tilt = (float)(1 - (Math.Asin(fz) * Helper.ToDeg / 90)) * (polarResolution.Dims.Y - 1);
                float Rot = (float)(((Math.Atan2(fy, fx) + Math.PI * 2) % (Math.PI * 2)) / (Math.PI * 2)) * (polarResolution.Dims.X - 1);

                int x0 = (int)Rot;
                int x1 = Math.Min(x0 + 1, polarResolution.Dims.X - 1);
                int y0 = (int)Tilt;
                int y1 = Math.Min(y0 + 1, polarResolution.Dims.Y - 1);

                float v00 = ShellData[y0 * polarResolution.Dims.X + x0];
                float v01 = ShellData[y0 * polarResolution.Dims.X + x1];
                float v10 = ShellData[y1 * polarResolution.Dims.X + x0];
                float v11 = ShellData[y1 * polarResolution.Dims.X + x1];

                float v0 = MathHelper.Lerp(v00, v01, Rot - x0);
                float v1 = MathHelper.Lerp(v10, v11, Rot - x0);
                float v = MathHelper.Lerp(v0, v1, Tilt - y0);

                float Shell = v;
                float Filter = Math.Min(1, Math.Max(0, 1 - (r - Shell)));

                VolumeFTData[z][y * (volume.Dims.X / 2 + 1) + x] *= Filter;
            });

            VolumeFT = new Image(VolumeFTData, volume.Dims, true);
            Image Filtered = VolumeFT.AsIFFT(true, 0, true).AndDisposeParent();

            return Filtered;
        }

        public static void LocalFSCFilter(Image halfMap1,
                                          Image halfMap2,
                                          Image maskSoft,
                                          float pixelSize,
                                          int windowSize,
                                          float fscThreshold,
                                          float globalResolution,
                                          float globalBFactor,
                                          float smoothingSigma,
                                          bool doSharpen,
                                          bool doNormalize,
                                          bool outputHalfMaps,
                                          out Image filteredMap1,
                                          out Image filteredMap2,
                                          out Image localResolution,
                                          out Image localBFactor)
        {
            localResolution = new Image(halfMap1.Dims);
            localBFactor = new Image(halfMap1.Dims);

            int LocalResolutionWindow = Math.Max(30, (int)Math.Round(globalResolution * 5 / pixelSize / 2) * 2);

            localResolution.Fill(LocalResolutionWindow * pixelSize / 2);
            localBFactor.Fill(globalBFactor);

            Image MapSum = halfMap1.GetCopyGPU();
            MapSum.Add(halfMap2);
            MapSum.Multiply(0.5f);

            #region Mask maps and calculate local resolution as well as average FSC and amplitude curves for each resolution

            Image Half1Masked = halfMap1.GetCopyGPU();
            halfMap1.FreeDevice();
            Half1Masked.Multiply(maskSoft);
            Image Half2Masked = halfMap2.GetCopyGPU();
            halfMap2.FreeDevice();
            Half2Masked.Multiply(maskSoft);

            int SpectrumLength = LocalResolutionWindow / 2;
            int SpectrumOversampling = 2;
            int NSpectra = SpectrumLength * SpectrumOversampling;

            Image AverageFSC = new Image(new int3(SpectrumLength, 1, NSpectra));
            Image AverageAmps = new Image(AverageFSC.Dims);
            Image AverageSamples = new Image(new int3(NSpectra, 1, 1));
            float[] GlobalLocalFSC = new float[LocalResolutionWindow / 2];

            GPU.LocalFSC(Half1Masked.GetDevice(Intent.Read),
                         Half2Masked.GetDevice(Intent.Read),
                         maskSoft.GetDevice(Intent.Read),
                         halfMap1.Dims,
                         1,
                         pixelSize,
                         localResolution.GetDevice(Intent.Write),
                         LocalResolutionWindow,
                         fscThreshold,
                         AverageFSC.GetDevice(Intent.ReadWrite),
                         AverageAmps.GetDevice(Intent.ReadWrite),
                         AverageSamples.GetDevice(Intent.ReadWrite),
                         SpectrumOversampling,
                         GlobalLocalFSC);

            Half1Masked.Dispose();
            Half2Masked.Dispose();
            AverageFSC.FreeDevice();
            AverageAmps.FreeDevice();
            AverageSamples.FreeDevice();

            #endregion

            #region Figure out scaling factor to bring mean local resolution to global value

            float LocalResScale;
            {
                Image MapSumAbs = MapSum.GetCopyGPU();
                MapSumAbs.Abs();
                MapSumAbs.Multiply(maskSoft);
                Image MapSumAbsConvolved = MapSumAbs.AsConvolvedRaisedCosine(0, LocalResolutionWindow / 2);
                MapSumAbsConvolved.Multiply(maskSoft);
                MapSumAbs.Dispose();

                float[] LocalResData = localResolution.GetHostContinuousCopy();
                float[] MaskConvolvedData = MapSumAbsConvolved.GetHostContinuousCopy();
                MapSumAbsConvolved.Dispose();

                double WeightedSum = 0;
                double Weights = 0;
                for (int i = 0; i < LocalResData.Length; i++)
                {
                    float Freq = 1 / LocalResData[i];

                    // No idea why local power * freq^2 produces good results, but it does!
                    float Weight = MaskConvolvedData[i] * Freq;
                    Weight *= Weight;

                    WeightedSum += Freq * Weight;
                    Weights += Weight;
                }
                WeightedSum /= Weights;
                WeightedSum = 1 / WeightedSum;
                LocalResScale = globalResolution / (float)WeightedSum;
            }

            #endregion

            #region Build resolution-dependent B-factor model

            #region Get average 1D amplitude spectra for each local resolution, weight by corresponding average local FSC curve, fit B-factors

            List<float3> ResolutionBFacs = new List<float3>();
            float[] AverageSamplesData = AverageSamples.GetHostContinuousCopy();

            float[][] AverageFSCData = AverageFSC.GetHost(Intent.Read).Select((a, i) => a.Select(v => v / Math.Max(1e-10f, AverageSamplesData[i])).ToArray()).ToArray();
            float[][] AverageAmpsData = AverageAmps.GetHost(Intent.Read);

            // Weights are FSC if we're filtering individual half-maps, or sqrt(2 * FSC / (1 + FSC)) if we're filtering their average
            float[][] FSCWeights = AverageFSCData.Select(a => a.Select(v => outputHalfMaps ? Math.Max(0, v) : (float)Math.Sqrt(Math.Max(0, 2 * v / (1 + v)))).ToArray()).ToArray();
            AverageAmpsData = AverageAmpsData.Select((a, i) => MathHelper.Mult(a, FSCWeights[i])).ToArray();

            float[] ResInv = Helper.ArrayOfFunction(i => i / (LocalResolutionWindow * pixelSize), SpectrumLength);

            if (doSharpen)
                for (int i = 0; i < NSpectra; i++)
                {
                    if (AverageSamplesData[i] < 100)
                        continue;

                    float ShellFirst = LocalResolutionWindow * pixelSize / 10f;
                    float ShellLast = i / (float)SpectrumOversampling * LocalResScale;
                    if (ShellLast - ShellFirst + 1 < 2.5f)
                        continue;

                    float[] ShellWeights = Helper.ArrayOfFunction(s =>
                    {
                        float WeightFirst = Math.Max(0, Math.Min(1, 1 - (ShellFirst - s)));
                        float WeightLast = Math.Max(0, Math.Min(1, 1 - (s - ShellLast)));
                        return Math.Min(WeightFirst, WeightLast);
                    }, SpectrumLength);

                    float3[] Points = Helper.ArrayOfFunction(s => new float3(ResInv[s] * ResInv[s],
                                                                             (float)Math.Log(Math.Max(1e-20f, AverageAmpsData[i][s])),
                                                                             ShellWeights[s]), SpectrumLength);

                    float3 Fit = MathHelper.FitLineWeighted(Points.Skip(1).ToArray());
                    ResolutionBFacs.Add(new float3(ShellLast, -Fit.X * 4, AverageSamplesData[i]));
                }

            #endregion

            #region Re-scale per-resolution B-factors to match global average, fit a*freq^b params to fit frequency vs. B-factor plot

            float2 BFactorModel = new float2(0, 0);
            if (doSharpen)
            {
                float WeightedMeanBFac = MathHelper.MeanWeighted(ResolutionBFacs.Select(v => v.Y).ToArray(), ResolutionBFacs.Select(v => v.Z * v.Z * v.X).ToArray());
                ResolutionBFacs = ResolutionBFacs.Select(v => new float3(v.X, v.Y * -globalBFactor / WeightedMeanBFac, v.Z)).ToList();

                float3[] BFacsLogLog = ResolutionBFacs.Select(v => new float3((float)Math.Log10(v.X), (float)Math.Log10(v.Y), v.Z)).ToArray();
                float3 LineFit = MathHelper.FitLineWeighted(BFacsLogLog);

                BFactorModel = new float2((float)Math.Pow(10, LineFit.Y), LineFit.X);
            }

            #endregion

            #region Calculate filter ramps for each local resolution value

            // Filter ramp consists of low-pass filter * FSC-based weighting (Rosenthal 2003) * B-factor correction
            // Normalized by average ramp value within [0; low-pass shell] to prevent low-resolution regions having lower intensity

            float[][] FilterRampsData = new float[NSpectra][];
            for (int i = 0; i < NSpectra; i++)
            {
                float ShellLast = i / (float)SpectrumOversampling;
                float[] ShellWeights = Helper.ArrayOfFunction(s => Math.Max(0, Math.Min(1, 1 - (s - ShellLast))), SpectrumLength);

                if (doSharpen)
                {
                    float BFac = i == 0 ? 0 : BFactorModel.X * (float)Math.Pow(ShellLast, BFactorModel.Y);
                    float[] ShellSharps = ResInv.Select(r => (float)Math.Exp(Math.Min(50, r * r * BFac * 0.25))).ToArray();

                    FilterRampsData[i] = MathHelper.Mult(ShellWeights, ShellSharps);
                }
                else
                {
                    FilterRampsData[i] = ShellWeights;
                }

                float FilterSum = FilterRampsData[i].Sum() / Math.Max(1, ShellLast);

                if (AverageSamplesData[i] > 10)
                    FilterRampsData[i] = MathHelper.Mult(FilterRampsData[i], FSCWeights[i]);

                if (doNormalize)
                    FilterRampsData[i] = FilterRampsData[i].Select(v => v / FilterSum).ToArray();
            }
            Image FilterRamps = new Image(FilterRampsData, new int3(SpectrumLength, 1, NSpectra));

            #endregion

            #endregion

            #region Convolve local res with small Gaussian to get rid of local outliers
            {
                Image LocalResInv = new Image(IntPtr.Zero, localResolution.Dims);
                LocalResInv.Fill(pixelSize * LocalResolutionWindow); // Scale mean to global resolution in this step as well

                //localResolution.Multiply(LocalResScale);

                LocalResInv.Divide(localResolution);
                localResolution.Dispose();

                Image LocalResInvSmooth = LocalResInv.AsConvolvedGaussian(smoothingSigma, true);
                LocalResInv.Dispose();

                localResolution = new Image(IntPtr.Zero, LocalResInv.Dims);
                localResolution.Fill(pixelSize * LocalResolutionWindow);
                localResolution.Divide(LocalResInvSmooth);
                LocalResInvSmooth.Dispose();
            }
            #endregion

            #region Finally, apply filter ramps (low-pass + B-factor) locally

            if (!outputHalfMaps)
            {
                Image MapLocallyFiltered = new Image(MapSum.Dims);
                GPU.LocalFilter(MapSum.GetDevice(Intent.Read),
                                MapLocallyFiltered.GetDevice(Intent.Write),
                                MapSum.Dims,
                                localResolution.GetDevice(Intent.Read),
                                LocalResolutionWindow,
                                pixelSize,
                                FilterRamps.GetDevice(Intent.Read),
                                SpectrumOversampling);

                filteredMap1 = MapLocallyFiltered;
                MapLocallyFiltered.FreeDevice();

                filteredMap2 = null;
            }
            else
            {
                filteredMap1 = halfMap1.GetCopyGPU();
                GPU.LocalFilter(halfMap1.GetDevice(Intent.Read),
                                filteredMap1.GetDevice(Intent.Write),
                                MapSum.Dims,
                                localResolution.GetDevice(Intent.Read),
                                LocalResolutionWindow,
                                pixelSize,
                                FilterRamps.GetDevice(Intent.Read),
                                SpectrumOversampling);

                filteredMap2 = halfMap2.GetCopyGPU();
                GPU.LocalFilter(halfMap2.GetDevice(Intent.Read),
                                filteredMap2.GetDevice(Intent.Write),
                                MapSum.Dims,
                                localResolution.GetDevice(Intent.Read),
                                LocalResolutionWindow,
                                pixelSize,
                                FilterRamps.GetDevice(Intent.Read),
                                SpectrumOversampling);
            }

            #endregion
        }

        public static float2[] FitBFactors(float3[][] corr, float[][] ctfWeights, float pixelSize, float lowResLimitAngstrom)
        {
            int NFrames = corr.Length;
            int FSCLength = corr[0].Length;

            float2[] ResultFrames = Helper.ArrayOfConstant(new float2(1, 0), NFrames);

            CTF CTF = new CTF()
            {
                PixelSize = (decimal)pixelSize,
                Defocus = 0,
                Amplitude = 1,
                Cs = 0
            };

            float[] Envelope;

            #region Set up low and high res limits

            int K0 = (int)Math.Ceiling(pixelSize * 2 / lowResLimitAngstrom * FSCLength);
            int K1 = K0;
            while (K1 < FSCLength - 1 && corr[0][K1].X != 0)
                K1++;

            float[] Weights = Helper.ArrayOfFunction(i => i >= K0 && i < K1 ? 1f : 0f, FSCLength);

            #endregion

            #region Methods for recalculating frame and items FSCs

            Func<float[][]> CalculateFrameFSCs = () =>
            {
                float[][] Result = new float[NFrames][];

                for (int f = 0; f < NFrames; f++)
                {
                    float3[] Sum = new float3[FSCLength];

                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.Scale = (decimal)(ResultFrames[f].X);
                    CTFCopy.Bfactor = (decimal)(ResultFrames[f].Y);

                    float[] Sim = CTFCopy.Get1D(FSCLength, false);

                    for (int j = 0; j < FSCLength; j++)
                        Sum[j] += corr[f][j] * Sim[j];

                    Result[f] = Sum.Select(v => v.X / (float)Math.Max(1e-16, Math.Sqrt(v.Y * v.Z))).ToArray();
                }

                return Result;
            };

            #endregion

            float[][] FrameFSCs = CalculateFrameFSCs();

            new Image(Helper.Combine(FrameFSCs), new int3(FSCLength, NFrames, 1)).WriteMRC("d_framefsc.mrc", true);

            #region Take a first guess at the envelope

            {
                int BestFSC = 0;
                float BestFSCSum = -float.MaxValue;
                float[] FSCSums = FrameFSCs.Select(f => f.Select((v, i) => v * Weights[i]).Sum()).ToArray();

                for (int i = 0; i < NFrames; i++)
                {
                    if (FSCSums[i] > BestFSCSum)
                    {
                        BestFSCSum = FSCSums[i];
                        BestFSC = i;
                    }
                }

                Envelope = FrameFSCs[BestFSC].ToList().ToArray();
            }

            #endregion

            float[][] RobustWeights = Helper.ArrayOfFunction(f => Helper.ArrayOfConstant(1f, FSCLength), NFrames);

            Func<float[][]> CalculateRobustWeights = () =>
            {
                float[][] Result = new float[NFrames][];

                float[][] AllDiffs = Helper.ArrayOfFunction(i => new float[NFrames], FSCLength);

                for (int f = 0; f < NFrames; f++)
                {
                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.Scale = (decimal)(ResultFrames[f].X);
                    CTFCopy.Bfactor = (decimal)(ResultFrames[f].Y);

                    float[] Sim = CTFCopy.Get1D(FSCLength, false);

                    for (int i = 0; i < FSCLength; i++)
                    {
                        if (Weights[i] == 0)
                            continue;

                        float d = (Sim[i] * Envelope[i] - FrameFSCs[f][i]);
                        AllDiffs[i][f] = d;
                    }
                }

                float[] StdDevs = AllDiffs.Select((a, i) => Weights[i] == 0 ? 1f : MathHelper.StdDev(a)).ToArray();

                for (int f = 0; f < NFrames; f++)
                {
                    Result[f] = new float[FSCLength];

                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.Scale = (decimal)(ResultFrames[f].X);
                    CTFCopy.Bfactor = (decimal)(ResultFrames[f].Y);

                    float[] Sim = CTFCopy.Get1D(FSCLength, false);

                    for (int i = 0; i < FSCLength; i++)
                    {
                        if (Weights[i] == 0)
                        {
                            Result[f][i] = 1;
                            continue;
                        }

                        float d = (Sim[i] * Envelope[i] - FrameFSCs[f][i]);

                        Result[f][i] = (float)Math.Exp(-Math.Abs(d) / StdDevs[i]);
                    }

                    Result[f] = MathHelper.MovingWindowMedian(Result[f], 7);
                }

                return Result;
            };

            #region Method for recalculating the envelope

            Func<float[]> CalculateEnvelopeFrames = () =>
            {
                float[] Result = new float[FSCLength];

                float[] Sum = new float[FSCLength];
                float[] WeightSum = new float[FSCLength];

                for (int f = 0; f < NFrames; f++)
                {
                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.Scale = (decimal)(ResultFrames[f].X);
                    CTFCopy.Bfactor = (decimal)(ResultFrames[f].Y);

                    float[] Sim = CTFCopy.Get1D(FSCLength, false);

                    for (int j = 0; j < FSCLength; j++)
                    {
                        Sum[j] += FrameFSCs[f][j] * Sim[j] * ctfWeights[f][j] * RobustWeights[f][j];
                        WeightSum[j] += Sim[j] * Sim[j] * ctfWeights[f][j] * RobustWeights[f][j];
                    }
                }

                for (int i = 0; i < FSCLength; i++)
                    Result[i] = Math.Max(0, Sum[i] / (float)Math.Max(1e-16, WeightSum[i]));

                return Result;
            };

            #endregion

            Func<float[][]> VisualizeCurrentModel = () =>
            {
                float[][] Result = new float[NFrames][];

                for (int f = 0; f < NFrames; f++)
                {
                    Result[f] = new float[FSCLength];

                    CTF CTFCopy = CTF.GetCopy();
                    CTFCopy.Scale = (decimal)(ResultFrames[f].X);
                    CTFCopy.Bfactor = (decimal)(ResultFrames[f].Y);

                    float[] Sim = CTFCopy.Get1D(FSCLength, false);

                    for (int i = 0; i < FSCLength; i++)
                    {
                        if (FrameFSCs[f][i] == 0)
                            continue;

                        Result[f][i] = Sim[i] * Envelope[i] * ctfWeights[f][i];
                    }
                }

                return Result;
            };

            #region Method for comparing weighted FSC to current envelope

            Func<float[], float2, float[], float[], float> EvalOne = (fsc, b, ctfWeight, robustWeight) =>
            {
                float Diff = 0;
                float Samples = 0;

                CTF CTFCopy = CTF.GetCopy();
                CTFCopy.Scale = (decimal)Math.Exp(b.X * 0.01);
                CTFCopy.Bfactor = (decimal)b.Y;

                float[] Sim = CTFCopy.Get1D(fsc.Length, false);

                for (int i = 0; i < fsc.Length; i++)
                {
                    float d = (Sim[i] * Envelope[i] - fsc[i]);
                    d *= d;
                    d *= Weights[i] * ctfWeight[i] * robustWeight[i];
                    Diff += d;
                    Samples += Weights[i] * ctfWeight[i] * robustWeight[i];
                }

                return Diff / Samples * 1000;
            };

            #endregion

            double[] StartParamsFrames = new double[NFrames * 2];

            for (int iopt = 0; iopt < 5; iopt++)
            {
                // Per frame
                if (iopt > 0)
                    Envelope = CalculateEnvelopeFrames();
                for (int isubopt = 0; isubopt < 5; isubopt++)
                {
                    Func<double[], double> Eval = (input) =>
                    {
                        double Diff = 0;

                        for (int f = 0; f < NFrames; f++)
                            Diff += EvalOne(FrameFSCs[f], new float2((float)input[f * 2 + 0], (float)input[f * 2 + 1]), ctfWeights[f], RobustWeights[f]);

                        return Diff;
                    };

                    Func<double[], double[]> Grad = (input) =>
                    {
                        double[] Grads = new double[input.Length];

                        float Delta = 0.001f;
                        float Delta2 = Delta * 2;

                        for (int f = 0; f < NFrames; f++)
                        {
                            float ScoreWeightPlus = EvalOne(FrameFSCs[f], new float2((float)input[f * 2 + 0] + Delta, (float)input[f * 2 + 1]), ctfWeights[f], RobustWeights[f]);
                            float ScoreWeightMinus = EvalOne(FrameFSCs[f], new float2((float)input[f * 2 + 0] - Delta, (float)input[f * 2 + 1]), ctfWeights[f], RobustWeights[f]);
                            Grads[f * 2 + 0] = (ScoreWeightPlus - ScoreWeightMinus) / Delta2;

                            float ScoreBPlus = EvalOne(FrameFSCs[f], new float2((float)input[f * 2 + 0], (float)input[f * 2 + 1] + Delta), ctfWeights[f], RobustWeights[f]);
                            float ScoreBMinus = EvalOne(FrameFSCs[f], new float2((float)input[f * 2 + 0], (float)input[f * 2 + 1] - Delta), ctfWeights[f], RobustWeights[f]);
                            Grads[f * 2 + 1] = (ScoreBPlus - ScoreBMinus) / Delta2;
                        }

                        return Grads;
                    };

                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParamsFrames.Length, Eval, Grad);
                    Optimizer.Minimize(StartParamsFrames);

                    for (int f = 0; f < NFrames; f++)
                        ResultFrames[f] = new float2((float)Math.Exp(StartParamsFrames[f * 2 + 0] * 0.01),
                                                         (float)StartParamsFrames[f * 2 + 1]);

                    //float MaxWeight = MathHelper.Max(ResultFrames.Select(v => v.X));
                    //float MaxB = MathHelper.Max(ResultFrames.Select(v => v.Y));
                    //ResultFrames = ResultFrames.Select(v => new float2(v.X / MaxWeight, v.Y - MaxB)).ToArray();

                    Envelope = CalculateEnvelopeFrames();
                    RobustWeights = CalculateRobustWeights();
                    Envelope = CalculateEnvelopeFrames();
                    RobustWeights = CalculateRobustWeights();
                    Envelope = CalculateEnvelopeFrames();
                    RobustWeights = CalculateRobustWeights();

                    new Image(Helper.Combine(VisualizeCurrentModel()), new int3(FSCLength, NFrames, 1)).WriteMRC($"d_frameviz_{iopt}_{isubopt}.mrc", true);
                    new Image(Helper.Combine(RobustWeights), new int3(FSCLength, NFrames, 1)).WriteMRC($"d_framerobustweights_{iopt}_{isubopt}.mrc", true);
                }

                FrameFSCs = CalculateFrameFSCs();

                new Image(Helper.Combine(FrameFSCs), new int3(FSCLength, NFrames, 1)).WriteMRC($"d_framefsc_{iopt}.mrc", true);
            }

            return ResultFrames;
        }


        public static float2[] FitGridBFactors(float3[][] corr, float2[] positions, float[] averageFSC, int2 dims, float pixelSize, float lowResLimitAngstrom)
        {
            int NParticles = corr.Length;
            int FSCLength = corr[0].Length;

            float2[] ResultGrid = Helper.ArrayOfConstant(new float2(1, 0), (int)dims.Elements());

            CTF CTF = new CTF()
            {
                PixelSize = (decimal)pixelSize,
                Defocus = 0,
                Amplitude = 1,
                Cs = 0
            };

            #region Set up low and high res limits

            int K0 = (int)Math.Ceiling(pixelSize * 2 / lowResLimitAngstrom * FSCLength);
            int K1 = K0;
            while (K1 < FSCLength - 1 && corr[0][K1].X != 0)
                K1++;

            float[] Weights = Helper.ArrayOfFunction(i => i >= K0 && i < K1 ? 1f : 0f, FSCLength);

            #endregion

            float[] Envelope = new float[FSCLength];
            for (int i = K0; i < K1; i++)
                Envelope[i] = averageFSC[i];

            float[][] ParticleFSCs = new float[NParticles][];
            for (int p = 0; p < NParticles; p++)
            {
                float[] ParticleFSC = new float[FSCLength];
                for (int i = K0; i < K1; i++)
                    ParticleFSC[i] = corr[p][i].X / (float)Math.Max(1e-16, Math.Sqrt(corr[p][i].Y * corr[p][i].Z));

                ParticleFSCs[p] = ParticleFSC;
            }

            float3[] Positions = positions.Select(v => new float3(v.X, v.Y, 0.5f)).ToArray();

            #region Method for comparing FSC to current envelope

            Func<float[], float2, float> EvalOne = (fsc, b) =>
            {
                float Diff = 0;

                CTF CTFCopy = CTF.GetCopy();
                CTFCopy.Scale = (decimal)b.X;
                CTFCopy.Bfactor = (decimal)b.Y;

                float[] Sim = CTFCopy.Get1D(fsc.Length, false);

                for (int i = 0; i < FSCLength; i++)
                {
                    float d = (Sim[i] * Envelope[i] - fsc[i]);
                    d *= d;
                    Diff += d;
                }

                return (float)Math.Sqrt(Diff / FSCLength) * 1000;
            };

            #endregion

            double[] StartParamsGrid = new double[dims.Elements() * 2];

            Func<double[], double> Eval = (input) =>
            {
                CubicGrid GridScale = new CubicGrid(new int3(dims), input.Take((int)dims.Elements()).Select(v => (float)Math.Exp(v * 0.01)).ToArray());
                CubicGrid GridBFac = new CubicGrid(new int3(dims), input.Skip((int)dims.Elements()).Take((int)dims.Elements()).Select(v => (float)v).ToArray());

                float[] ParticleScales = GridScale.GetInterpolated(Positions);
                float[] ParticleBFacs = GridBFac.GetInterpolated(Positions);

                double Diff = 0;

                for (int p = 0; p < NParticles; p++)
                    Diff += EvalOne(ParticleFSCs[p], new float2(ParticleScales[p], ParticleBFacs[p]));

                return Diff;
            };

            int NItersDone = 0;
            Func<double[], double[]> Grad = (input) =>
            {
                double[] Grads = new double[input.Length];

                if (NItersDone++ > 30)
                    return Grads;

                float Delta = 0.001f;
                float Delta2 = Delta * 2;

                //Parallel.For(0, input.Length, new ParallelOptions() { MaxDegreeOfParallelism = 2 }, i =>
                for (int i = 0; i < input.Length; i++)
                {
                    double[] InputPlus = input.ToArray();
                    InputPlus[i] += Delta;
                    double ScorePlus = Eval(InputPlus);

                    double[] InputMinus = input.ToArray();
                    InputMinus[i] -= Delta;
                    double ScoreMinus = Eval(InputMinus);

                    Grads[i] = (ScorePlus - ScoreMinus) / Delta2;
                }//);

                //Console.WriteLine(Eval(input));

                return Grads;
            };

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParamsGrid.Length, Eval, Grad);
            Optimizer.Minimize(StartParamsGrid);

            for (int i = 0; i < ResultGrid.Length; i++)
                ResultGrid[i] = new float2((float)Math.Exp(StartParamsGrid[i] * 0.01), (float)StartParamsGrid[dims.Elements() + i]);

            return ResultGrid;
        }

        public static (float[] Scales, float3[] Bfactors, Image SumCorr) FitBFactors2D(Image corrAB, Image corrA2, Image corrB2, Image ctfWeights, float pixelSize, float lowResLimitAngstrom, bool doAnisotropy, int batchSize, Action<float> progressCallback)
        {
            int2 Dims = new int2(corrAB.Dims);

            int NItems = corrAB.Dims.Z;
            int FSCLength = corrAB.Dims.X / 2 + 1;

            int BatchSize = Math.Min(NItems, batchSize);
            int3 DimsBatch = new int3(Dims.X, Dims.Y, BatchSize);
            int BatchStart = 0;
            int BatchEnd = BatchSize;

            float[] ResultScales = Helper.ArrayOfConstant(1f, NItems);
            float3[] ResultBfactors = new float3[NItems];

            CTF CTFProto = new CTF()
            {
                PixelSize = (decimal)pixelSize,
                Defocus = 0,
                Amplitude = 1,
                Cs = 0
            };
            CTFStruct[] CTFStructs = new CTFStruct[BatchSize];
            Image CTFSimBatch = new Image(IntPtr.Zero, DimsBatch, true);

            Image CTFCoords = CTF.GetCTFCoords(Dims.X, Dims.X);

            Image CorrABBatch = new Image(IntPtr.Zero, DimsBatch, true);
            Image CorrA2Batch = new Image(IntPtr.Zero, DimsBatch, true);
            Image CorrB2Batch = new Image(IntPtr.Zero, DimsBatch, true);

            Image CorrPremultBatch = new Image(IntPtr.Zero, DimsBatch, true);

            Image Diff = new Image(IntPtr.Zero, new int3(Dims), true);
            Image CorrABReduced = new Image(IntPtr.Zero, new int3(Dims), true);
            Image CorrA2Reduced = new Image(IntPtr.Zero, new int3(Dims), true);
            Image CorrB2Reduced = new Image(IntPtr.Zero, new int3(Dims), true);

            Image FSCsBatch = new Image(IntPtr.Zero, DimsBatch, true);
            Image FSCOverall = new Image(IntPtr.Zero, new int3(Dims), true);

            Image FinalAB = new Image(IntPtr.Zero, new int3(Dims), true);
            Image FinalA2 = new Image(IntPtr.Zero, new int3(Dims), true);
            Image FinalB2 = new Image(IntPtr.Zero, new int3(Dims), true);

            float[] DiffPerParticle = new float[BatchSize];

            #region Set up low and high res limits

            int K0 = (int)Math.Ceiling(pixelSize * 2 / lowResLimitAngstrom * FSCLength);
            int K1 = K0;
            {
                float[] CorrABData = corrAB.GetHost(Intent.Read)[0];
                while (K1 < FSCLength - 1 && CorrABData[K1] != 0)
                    K1++;
            }

            Image Mask = new Image(corrAB.Dims.Slice(), true);
            {
                float[] MaskData = Mask.GetHost(Intent.ReadWrite)[0];
                Helper.ForEachElementFT(new int2(Mask.Dims), (x, y, xx, yy, r, angle) =>
                {
                    if (r >= K0 && r < K1)
                        MaskData[y * (Dims.X / 2 + 1) + x] = 1 / Math.Max(1, r);    // Divide by r to give higher shells less weight in 2D
                });
            }

            #endregion

            Action UpdateCTFSim = () =>
            {
                int CurBatch = BatchEnd - BatchStart;

                for (int i = 0; i < CurBatch; i++)
                {
                    CTFProto.Scale = (decimal)ResultScales[BatchStart + i];
                    CTFProto.Bfactor = (decimal)ResultBfactors[BatchStart + i].X;
                    CTFProto.BfactorDelta = (decimal)ResultBfactors[BatchStart + i].Y;
                    CTFProto.BfactorAngle = (decimal)ResultBfactors[BatchStart + i].Z;

                    CTFStructs[i] = CTFProto.ToStruct();
                }

                GPU.CreateCTF(CTFSimBatch.GetDevice(Intent.Write),
                              CTFCoords.GetDevice(Intent.Read),
                              IntPtr.Zero,
                              (uint)CTFCoords.ElementsSliceComplex,
                              CTFStructs,
                              false,
                              (uint)CurBatch);
            };

            Action UpdateFSCOverall = () =>
            {
                Image CorrA2Overall = new Image(CorrABReduced.Dims, true);
                Image CorrB2Overall = new Image(CorrABReduced.Dims, true);
                FSCOverall.Fill(0);

                for (int b = 0; b < NItems; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NItems - b);
                    BatchStart = b;
                    BatchEnd = b + CurBatch;

                    UpdateCTFSim();

                    for (int i = 0; i < CurBatch; i++)
                    {
                        Array.Copy(corrAB.GetHost(Intent.Read)[b + i], CorrABBatch.GetHost(Intent.Write)[i], corrAB.ElementsSliceReal);
                        Array.Copy(corrA2.GetHost(Intent.Read)[b + i], CorrA2Batch.GetHost(Intent.Write)[i], corrA2.ElementsSliceReal);
                        Array.Copy(corrB2.GetHost(Intent.Read)[b + i], CorrB2Batch.GetHost(Intent.Write)[i], corrB2.ElementsSliceReal);
                        
                        //CorrABBatch.GetHost(Intent.Write)[i] = corrAB.GetHost(Intent.Read)[b + i];
                        //CorrA2Batch.GetHost(Intent.Write)[i] = corrA2.GetHost(Intent.Read)[b + i];
                        //CorrB2Batch.GetHost(Intent.Write)[i] = corrB2.GetHost(Intent.Read)[b + i];
                    }

                    GPU.CopyDeviceToDevice(CorrABBatch.GetDevice(Intent.Read), CorrPremultBatch.GetDevice(Intent.Write), CorrABBatch.ElementsReal);
                    CorrPremultBatch.Multiply(CTFSimBatch);
                    //CorrPremultBatch.Multiply(ctfWeights);
                    GPU.ReduceAdd(CorrPremultBatch.GetDevice(Intent.Read),
                                  CorrABReduced.GetDevice(Intent.Write),
                                  (uint)CorrPremultBatch.ElementsSliceReal,
                                  (uint)CurBatch,
                                  1);
                    FSCOverall.Add(CorrABReduced);

                    GPU.CopyDeviceToDevice(CorrA2Batch.GetDevice(Intent.Read), CorrPremultBatch.GetDevice(Intent.Write), CorrA2Batch.ElementsReal);
                    CorrPremultBatch.Multiply(CTFSimBatch);
                    //CorrPremultBatch.Multiply(ctfWeights);
                    GPU.ReduceAdd(CorrPremultBatch.GetDevice(Intent.Read),
                                  CorrA2Reduced.GetDevice(Intent.Write),
                                  (uint)CorrPremultBatch.ElementsSliceReal,
                                  (uint)CurBatch,
                                  1);
                    CorrA2Overall.Add(CorrA2Reduced);

                    GPU.CopyDeviceToDevice(CorrB2Batch.GetDevice(Intent.Read), CorrPremultBatch.GetDevice(Intent.Write), CorrB2Batch.ElementsReal);
                    CorrPremultBatch.Multiply(CTFSimBatch);
                    //CorrPremultBatch.Multiply(ctfWeights);
                    GPU.ReduceAdd(CorrPremultBatch.GetDevice(Intent.Read),
                                  CorrB2Reduced.GetDevice(Intent.Write),
                                  (uint)CorrPremultBatch.ElementsSliceReal,
                                  (uint)CurBatch,
                                  1);
                    CorrB2Overall.Add(CorrB2Reduced);
                }

                GPU.CopyDeviceToDevice(FSCOverall.GetDevice(Intent.Read), FinalAB.GetDevice(Intent.Write), FinalAB.ElementsReal);
                GPU.CopyDeviceToDevice(CorrA2Overall.GetDevice(Intent.Read), FinalA2.GetDevice(Intent.Write), FinalAB.ElementsReal);
                GPU.CopyDeviceToDevice(CorrB2Overall.GetDevice(Intent.Read), FinalB2.GetDevice(Intent.Write), FinalAB.ElementsReal);

                CorrA2Overall.Multiply(CorrB2Overall);
                CorrA2Overall.Sqrt();
                CorrA2Overall.Max(1e-20f);

                FSCOverall.Divide(CorrA2Overall);

                CorrA2Overall.Dispose();
                CorrB2Overall.Dispose();
            };

            Action UpdateFSCs = () =>
            {
                GPU.MultiplySlices(CorrA2Batch.GetDevice(Intent.Read),
                                   CorrB2Batch.GetDevice(Intent.Read),
                                   CorrPremultBatch.GetDevice(Intent.Write),
                                   CorrA2Batch.ElementsReal,
                                   1);
                CorrPremultBatch.Sqrt();
                CorrPremultBatch.Max(1e-20f);

                GPU.DivideSlices(CorrABBatch.GetDevice(Intent.Read),
                                 CorrPremultBatch.GetDevice(Intent.Read),
                                 FSCsBatch.GetDevice(Intent.Write),
                                 CorrABBatch.ElementsReal,
                                 1);
            };

            Func<double[], float[]> EvalIndividually = (input) =>
            {
                int CurBatch = BatchEnd - BatchStart;

                for (int i = 0; i < CurBatch; i++)
                {
                    ResultScales[BatchStart + i] = (float)Math.Exp(input[i * 4 + 0] * 0.1);
                    ResultBfactors[BatchStart + i] = new float3((float)input[i * 4 + 1],
                                                                doAnisotropy ? (float)input[i * 4 + 2] : 0,
                                                                doAnisotropy ? (float)(input[i * 4 + 3] * 1) : 0);
                }

                UpdateCTFSim();

                CTFSimBatch.MultiplySlices(FSCOverall);
                CTFSimBatch.Subtract(FSCsBatch);
                CTFSimBatch.Abs();
                CTFSimBatch.MultiplySlices(Mask);

                GPU.ReduceAdd(CTFSimBatch.GetDevice(Intent.Read),
                              CorrPremultBatch.GetDevice(Intent.Write),
                              1,
                              (uint)CTFSimBatch.ElementsSliceReal,
                              (uint)CurBatch);

                GPU.CopyDeviceToHost(CorrPremultBatch.GetDevice(Intent.Read),
                                     DiffPerParticle,
                                     CurBatch);

                return DiffPerParticle.Take(CurBatch).ToArray();
            };

            Func<double[], double> Eval = (input) =>
            {
                float[] DiffData = EvalIndividually(input);
                double Result = 0;
                foreach (var item in DiffData)
                    Result += item;

                return Result * 5;
            };

            int NIterations = 0;
            Func<double[], double[]> Grad = (input) =>
            {
                double Delta = 0.025;

                double[] Result = new double[input.Length];

                //float[] Scores0 = EvalIndividually(input);
                //Console.WriteLine($"{NIterations++}: {Scores0.Sum()}");

                double[] InputAltered = input.ToList().ToArray();

                int CurBatch = BatchEnd - BatchStart;

                foreach (int ii in (doAnisotropy ? new[] { 0, 1, 2, 3 } : new[] { 0, 1 }))
                {
                    for (int i = 0; i < CurBatch; i++)
                        InputAltered[i * 4 + ii] = input[i * 4 + ii] + Delta;

                    float[] ScoresPlus = EvalIndividually(InputAltered);

                    for (int i = 0; i < CurBatch; i++)
                        InputAltered[i * 4 + ii] = input[i * 4 + ii] - Delta;

                    float[] ScoresMinus = EvalIndividually(InputAltered);

                    for (int i = 0; i < CurBatch; i++)
                    {
                        Result[i * 4 + ii] = (ScoresPlus[i] - ScoresMinus[i]) / (Delta * 2) * 5;

                        InputAltered[i * 4 + ii] = input[i * 4 + ii];
                    }
                }

                return Result;
            };

            UpdateCTFSim();
            UpdateFSCOverall();

            FSCOverall.WriteMRC("d_fscoverall.mrc", true);

            int NEpochs = 3;
            int NBatchesOverall = (NItems + BatchSize - 1) / BatchSize * NEpochs;
            int BatchesDone = 0;

            for (int ioptim = 0; ioptim < NEpochs; ioptim++)
            {
                for (int b = 0; b < NItems; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NItems - b);
                    BatchStart = b;
                    BatchEnd = b + CurBatch;

                    for (int i = 0; i < CurBatch; i++)
                    {
                        Array.Copy(corrAB.GetHost(Intent.Read)[b + i], CorrABBatch.GetHost(Intent.Write)[i], corrAB.ElementsSliceReal);
                        Array.Copy(corrA2.GetHost(Intent.Read)[b + i], CorrA2Batch.GetHost(Intent.Write)[i], corrA2.ElementsSliceReal);
                        Array.Copy(corrB2.GetHost(Intent.Read)[b + i], CorrB2Batch.GetHost(Intent.Write)[i], corrB2.ElementsSliceReal);
                        
                        //CorrABBatch.GetHost(Intent.Write)[i] = corrAB.GetHost(Intent.Read)[b + i];
                        //CorrA2Batch.GetHost(Intent.Write)[i] = corrA2.GetHost(Intent.Read)[b + i];
                        //CorrB2Batch.GetHost(Intent.Write)[i] = corrB2.GetHost(Intent.Read)[b + i];
                    }

                    UpdateFSCs();
                    FSCsBatch.WriteMRC("d_fscs.mrc", true);

                    double[] Params = new double[CurBatch * 4];
                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(Params.Length, Eval, Grad);

                    Optimizer.Minimize(Params);
                    NIterations = 0;

                    if (progressCallback != null)
                        progressCallback((++BatchesDone) / (float)NBatchesOverall);
                }

                UpdateFSCOverall();
                FSCOverall.WriteMRC($"d_fscoverall_{ioptim + 1}.mrc", true);
            }

            CTFSimBatch.Dispose();
            CTFCoords.Dispose();
            Diff.Dispose();
            CorrPremultBatch.Dispose();
            CorrA2Reduced.Dispose();
            CorrB2Reduced.Dispose();
            FSCsBatch.Dispose();
            FSCOverall.Dispose();
            Mask.Dispose();

            Image ResultCorr = new Image(new float[][] { FinalAB.GetHost(Intent.Read)[0].ToArray(),
                                                         FinalA2.GetHost(Intent.Read)[0].ToArray(),
                                                         FinalB2.GetHost(Intent.Read)[0].ToArray() },
                                         new int3(Dims.X, Dims.Y, 3), true);

            FinalAB.Dispose();
            FinalA2.Dispose();
            FinalB2.Dispose();

            return (ResultScales, ResultBfactors, ResultCorr);
        }
    }
}
