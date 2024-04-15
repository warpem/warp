using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Math.Optimization;

namespace Warp.Tools
{
    public static class ImageHelper
    {
        public static Image AlignLocallyToTarget(Image map, Image target, int alignmentSize, int oversampling, out float3 shift, out float3 rotation)
        {
            float ScaleFactor = (float)alignmentSize / map.Dims.X;
            int3 DimsScaled = new int3(alignmentSize, alignmentSize, alignmentSize);

            #region Prepare Scaled & FFTed maps

            Image MapScaled = map.AsScaled(DimsScaled);
            map.FreeDevice();

            Image TargetScaled = target.AsScaled(DimsScaled);
            target.FreeDevice();
            TargetScaled.RemapToFT(true);
            TargetScaled.WriteMRC("d_targetscaled.mrc", true);
            Image TargetScaledFT = TargetScaled.AsFFT(true);
            TargetScaled.Dispose();

            Projector ProjMapScaled = new Projector(MapScaled, oversampling);
            Image TestFT = ProjMapScaled.Project(DimsScaled, new float3[1]);
            Image Test = TestFT.AsIFFT(true).AndDisposeParent();
            Test.RemapFromFT(true);
            Test.WriteMRC("d_projected.mrc", true);

            #endregion

            float3 CurShift = new float3(), CurRotation = new float3();

            Func<double[], double> GetDiff = input =>
            {
                CurShift = new float3((float)input[0], (float)input[1], (float)input[2]);
                CurRotation = new float3((float)input[3], (float)input[4], (float)input[5]) * Helper.ToRad;
                CurRotation = Matrix3.EulerFromMatrix(Matrix3.RotateX(CurRotation.X) * Matrix3.RotateY(CurRotation.Y) * Matrix3.RotateZ(CurRotation.Z)) * Helper.ToDeg;

                Image TargetFTShifted = TargetScaledFT.AsShiftedVolume(-CurShift * ScaleFactor);
                Image MapRotatedFT = ProjMapScaled.Project(DimsScaled, new[] { CurRotation * Helper.ToRad });

                GPU.MultiplySlices(TargetFTShifted.GetDevice(Intent.Read),
                                    MapRotatedFT.GetDevice(Intent.Read),
                                    TargetFTShifted.GetDevice(Intent.Write),
                                    TargetFTShifted.ElementsReal,
                                    1);
                MapRotatedFT.Dispose();

                IntPtr d_Sum = GPU.MallocDevice(1);
                GPU.Sum(TargetFTShifted.GetDevice(Intent.Read), d_Sum, (uint)TargetFTShifted.ElementsReal, 1);
                TargetFTShifted.Dispose();

                float[] h_Sum = new float[1];
                GPU.CopyDeviceToHost(d_Sum, h_Sum, 1);

                GPU.FreeDevice(d_Sum);
                
                return h_Sum[0];
            };

            Func<double[], double[]> GetGradient = input =>
            {
                float Delta = 0.05f;
                double[] Result = new double[input.Length];

                Console.WriteLine(GetDiff(input));

                for (int i = 0; i < input.Length; i++)
                {
                    double[] InputPlus = input.ToList().ToArray();
                    InputPlus[i] += Delta;
                    double ScorePlus = GetDiff(InputPlus);

                    double[] InputMinus = input.ToList().ToArray();
                    InputMinus[i] -= Delta;
                    double ScoreMinus = GetDiff(InputMinus);

                    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                }

                return Result;
            };

            double[] StartParams = new double[6];
            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartParams.Length, GetDiff, GetGradient);
            Optimizer.MaxIterations = 20;
            Optimizer.Maximize(StartParams);

            GetDiff(StartParams);
            shift = CurShift;
            rotation = CurRotation;

            #region Shift and rotate input map by determined values for final output

            ProjMapScaled.Dispose();
            Image MapResult = map.AsShiftedVolume(shift);
            map.FreeDevice();

            Projector ProjMapResult = new Projector(MapResult, oversampling);
            MapResult.Dispose();

            Image MapResultFT = ProjMapResult.Project(map.Dims, new[] { rotation * Helper.ToRad });
            ProjMapResult.Dispose();

            MapResult = MapResultFT.AsIFFT(true).AndDisposeParent();
            MapResult.RemapFromFT(true);

            #endregion

            return MapResult;
        }

        public static Image MakeCircularMask(int size, float radius, float falloff)
        {
            float[] ParticleMaskData = new float[size * size];
            float Radius2 = radius;

            for (int y = 0; y < size; y++)
            {
                int yy = y - size / 2;
                yy *= yy;

                for (int x = 0; x < size; x++)
                {
                    int xx = x - size / 2;
                    xx *= xx;
                    float R = (float)Math.Sqrt(xx + yy);

                    float V = R <= Radius2 ? 1f : (float)(Math.Cos(Math.Min(1, (R - Radius2) / falloff) * Math.PI) * 0.5 + 0.5);
                    ParticleMaskData[y * size + x] = V;
                }
            }

            return new Image(ParticleMaskData, new int3(size, size, 1));
        }
    }
}
