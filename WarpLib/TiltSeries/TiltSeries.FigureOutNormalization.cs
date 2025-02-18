using System;
using System.Linq;
using Accord.Math.Optimization;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public void FigureOutNormalization(ProcessingOptionsTomoSubReconstruction options)
    {
        options.BinTimes = (decimal)Math.Log(30 / (double)options.PixelSizeMean, 2.0);

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        //Movie[] TiltMovies;
        Image[] TiltData;
        Image[] TiltMasks;
        LoadMovieData(options, out _, out TiltData, false, out _, out _);
        LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            EraseDirt(TiltData[z], TiltMasks[z]);
            TiltMasks[z]?.FreeDevice();

            if (options.NormalizeInput)
            {
                TiltData[z].SubtractMeanGrid(new int2(1));
                TiltData[z].Bandpass(1f / 8f, 1f, false, 1f / 8f);

                Image Cropped = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2);
                float2 MeanStd = MathHelper.MeanAndStd(Cropped.GetHost(Intent.Read)[0]);
                Cropped.Dispose();

                TiltData[z].Add(-MeanStd.X);
                //if (MeanStd.Y > 0)
                //    TiltData[z].Multiply(1 / MeanStd.Y);

                Console.WriteLine(MeanStd.Y);

                //GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                //              TiltData[z].GetDevice(Intent.Write),
                //              (uint)TiltData[z].ElementsReal,
                //              1);
            }

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            TiltData[z].FreeDevice();

            //TiltData[z].Multiply(TiltMasks[z]);
        }

        Console.WriteLine("\n\n\n");

        int DimFull = Math.Min(TiltData[0].Dims.X, TiltData[0].Dims.Y);
        int DimCompare = DimFull / 4 * 2;

        Image CTFCoords = CTF.GetCTFCoords(DimFull, DimFull);
        Image CTFOri = new Image(new int3(DimFull, DimFull, NTilts), true, false);

        float3[] ParticlePositions = Helper.ArrayOfConstant(new float3(0.5f) * VolumeDimensionsPhysical, NTilts);
        float3[] ParticleAngles = GetAngleInAllTilts(ParticlePositions);

        Image ImagesOriFT = GetImagesForOneParticle(options, TiltData, DimFull, ParticlePositions, -1, -1, 8, true);
        GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFOri);

        Image ImagesFlippedFT = ImagesOriFT.GetCopyGPU();
        Image ImagesWeightedFT = ImagesOriFT.GetCopyGPU();
        Image CTFWeighted = CTFOri.GetCopyGPU();
        Image CTFSign = CTFOri.GetCopyGPU();

        CTFSign.Sign();
        ImagesFlippedFT.Multiply(CTFSign);

        ImagesWeightedFT.Multiply(CTFSign);
        //CTFWeighted.Multiply(CTFOri);
        CTFWeighted.Abs();

        CTFSign.Dispose();

        int[] TiltIndicesSorted = IndicesSortedAbsoluteAngle;
        float[] Scales = new float[NTilts];

        for (int seriesLength = 0; seriesLength < NTilts; seriesLength++)
        {
            int[] CurrentIndices = TiltIndicesSorted.Take(Math.Max(1, seriesLength)).ToArray();
            float3[] CurrentAngles = Helper.IndexedSubset(ParticleAngles, CurrentIndices);

            Image StackWeightedFT = new Image(Helper.IndexedSubset(ImagesWeightedFT.GetHost(Intent.Read), CurrentIndices), new int3(DimFull, DimFull, CurrentIndices.Length), true, true);
            Image StackCTFWeighted = new Image(Helper.IndexedSubset(CTFWeighted.GetHost(Intent.Read), CurrentIndices), new int3(DimFull, DimFull, CurrentIndices.Length), true, false);

            Projector Reconstructor = new Projector(new int3(DimFull), 2);
            Reconstructor.BackProject(StackWeightedFT, StackCTFWeighted, CurrentAngles, Matrix2.Identity());
            Reconstructor.Weights.Max(1);
            Image VolFull = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0);
            //VolFull.MaskSpherically(VolFull.Dims.X - 32, 16, true);
            VolFull.WriteMRC("d_volfull.mrc", true);

            StackWeightedFT.Fill(new float2(1, 0));
            StackCTFWeighted.Fill(1);
            Reconstructor.Data.Fill(0);
            Reconstructor.Weights.Fill(0);

            Reconstructor.BackProject(StackWeightedFT, StackCTFWeighted, CurrentAngles, Matrix2.Identity());
            Reconstructor.Weights.Max(1);
            Image PSF = Reconstructor.Reconstruct(false, "C1", null, -1, -1, -1, 0);
            PSF.WriteMRC("d_psf.mrc", true);

            Reconstructor.Dispose();

            Projector ProjectorCTF = new Projector(PSF, 2);
            Projector ProjectorVol = new Projector(VolFull, 2);

            Image NextFlippedFT = new Image(ImagesFlippedFT.GetHost(Intent.Read)[TiltIndicesSorted[seriesLength]], new int3(DimFull, DimFull, 1), true, true);
            Image WeightComplex = ProjectorCTF.Project(new int2(DimFull), new float3[] { ParticleAngles[TiltIndicesSorted[seriesLength]] });
            Image Weight = WeightComplex.AsAmplitudes().AndDisposeParent();
            Weight.WriteMRC("d_weight.mrc", true);

            NextFlippedFT.Multiply(Weight);
            Image NextFlipped = NextFlippedFT.AsIFFT(false, 0, true).AndDisposeParent();
            NextFlipped.MaskSpherically(NextFlipped.Dims.X / 2, 16, false, true);
            NextFlipped.WriteMRC("d_nextflipped.mrc", true);
            NextFlippedFT = NextFlipped.AsFFT().AndDisposeParent();

            Image RefFT = ProjectorVol.Project(new int2(DimFull), new float3[] { ParticleAngles[TiltIndicesSorted[seriesLength]] });
            Image Ref = RefFT.AsIFFT(false, 0, true).AndDisposeParent();
            Ref.MaskSpherically(Ref.Dims.X / 2, 16, false, true);
            Ref.WriteMRC("d_ref.mrc", true);
            RefFT = Ref.AsFFT().AndDisposeParent();

            ProjectorCTF.Dispose();
            ProjectorVol.Dispose();

            Func<double[], double> Eval = (input) =>
            {
                Image NextCopy = NextFlippedFT.GetCopyGPU();
                NextCopy.Multiply((float)input[0]);
                NextCopy.Subtract(RefFT);
                float Diff = NextCopy.GetHost(Intent.Read)[0].Skip(1).Select(v => v * v).Sum();
                NextCopy.Dispose();

                return Diff * 1000;
            };

            Func<double[], double[]> Grad = (input) =>
            {
                double Score0 = Eval(input);
                double Score1 = Eval(new double[] { input[0] + 1e-4 });
                //Console.WriteLine(Score0);

                return new double[] { (Score1 - Score0) / 1e-4 };
            };

            double[] StartParams = { 1.0 };

            BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(1, Eval, Grad);
            Optimizer.Minimize(StartParams);

            Scales[TiltIndicesSorted[seriesLength]] = (float)StartParams[0];
            Console.WriteLine(StartParams[0]);

            RefFT.Dispose();
            WeightComplex.Dispose();
            NextFlippedFT.Dispose();
            PSF.Dispose();
            VolFull.Dispose();
            StackCTFWeighted.Dispose();
            StackWeightedFT.Dispose();
        }
    }
}