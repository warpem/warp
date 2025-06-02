using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Math.Optimization;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public void AlignToMaxL2(ProcessingOptionsTomoFullReconstruction options)
    {
        VolumeDimensionsPhysical = options.DimensionsPhysical;
        int SizeRegion = options.SubVolumeSize;

        #region Load and preprocess data

        Image[] TiltData;
        Image[] TiltMasks;
        LoadMovieData(options, out _, out TiltData, false, out _, out _, false);
        LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            EraseDirt(TiltData[z], TiltMasks[z]);
            TiltMasks[z]?.FreeDevice();

            TiltData[z].SubtractMeanGrid(new int2(1));
            TiltData[z] = TiltData[z].AsPaddedClamped(new int2(TiltData[z].Dims) * 2).AndDisposeParent();
            TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
            //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
            //TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
            TiltData[z].Bandpass(2f / (300f / (float)options.BinnedPixelSizeMean), 1f, false, 2f / (300f / (float)options.BinnedPixelSizeMean));
            TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
            //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

            GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                TiltData[z].GetDevice(Intent.Write),
                (uint)TiltData[z].ElementsReal,
                1);

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            TiltData[z].FreeDevice();
        }

        int2 DimsImage = new int2(TiltData[0].Dims);
        int SizeReconstruction = SizeRegion;
        int SizeReconstructionPadded = SizeReconstruction * 2;

        #endregion

        float3[] TransformedAngles = GetAngleInAllTilts(VolumeDimensionsPhysical / 2).Select(a => Matrix3.EulerFromMatrix(Matrix3.Euler(a) * Matrix3.RotateX(0 * Helper.ToRad))).ToArray();

        #region Make CTF

        Image BlobCTF;
        Image Multiplicity;
        Image RecCTFs, RecCTFsAbs;
        {
            Image CTFCoords = CTF.GetCTFCoords(SizeReconstructionPadded, SizeReconstructionPadded);
            Image CTFs = GetCTFsForOneParticle(options, VolumeDimensionsPhysical * 0.5f, CTFCoords, null, true, false, false);
            Image CTFsAbs = CTFs.GetCopyGPU();
            CTFsAbs.Abs();
            RecCTFs = CTFs.GetCopyGPU();
            RecCTFsAbs = CTFsAbs.GetCopyGPU();

            //RecCTFs.Fill(1);
            //RecCTFsAbs.Fill(1);

            // CTF has to be converted to complex numbers with imag = 0, and weighted by itself
            Image CTFsComplex = new Image(CTFs.Dims, true, true);
            CTFsComplex.Fill(new float2(1f / (SizeReconstructionPadded * SizeReconstructionPadded), 0));
            CTFsComplex.Multiply(CTFs);
            CTFsComplex.Multiply(CTFs);

            //RandomNormal Random = new RandomNormal(123);
            //CTFsComplex.ShiftSlices(Helper.ArrayOfFunction(i => new float3(0, MathF.Abs(MathF.Sin(Angles[i] * Helper.ToRad)), 0) * 5f, NTilts));

            // Back-project and reconstruct
            Projector ProjCTF = new Projector(new int3(SizeReconstructionPadded), 1);
            Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

            //ProjCTF.Weights.Fill(0.01f);

            ProjCTF.BackProject(CTFsComplex, CTFsAbs, TransformedAngles, MagnificationCorrection);

            CTFsAbs.Fill(1);
            ProjCTFWeights.BackProject(CTFsComplex, CTFsAbs, TransformedAngles, MagnificationCorrection);
            ProjCTFWeights.Weights.Min(1);
            ProjCTF.Data.Multiply(ProjCTFWeights.Weights);

            Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();

            CTFsComplex.Dispose();
            ProjCTFWeights.Dispose();

            Image PSF = ProjCTF.Reconstruct(false, "C1", null, -1, -1, -1, 0).AsPadded(new int3(SizeReconstruction)).AndDisposeParent();
            PSF.WriteMRC("d_psf.mrc", true);
            //Console.WriteLine(PSF.GetHost(Intent.Read).SelectMany(v => v).Select(v => MathF.Min(0, v)).Sum());
            PSF.RemapToFT(true);
            ProjCTF.Dispose();

            BlobCTF = PSF.AsFFT(true).AndDisposeParent().AsAmplitudes().AndDisposeParent();
            BlobCTF.Multiply(1f / (SizeRegion * SizeRegion));

            CTFs.Dispose();
            CTFsAbs.Dispose();
        }
        BlobCTF.WriteMRC("d_ctf.mrc", true);

        #endregion

        #region Make blobs

        int[] BlobSizes = { 40 };
        Image[] BlobVolumes = new Image[BlobSizes.Length];

        for (int i = 0; i < BlobSizes.Length; i++)
        {
            Image BlobVolume = new Image(new int3(SizeRegion));
            BlobVolume.Fill(1);
            BlobVolume.MaskSpherically(BlobSizes[i] / (float)options.BinnedPixelSizeMean, 3, true, true);

            float BlobSum = BlobVolume.GetHost(Intent.Read).SelectMany(v => v).Sum();
            //BlobVolume.Multiply(1f / BlobSum);

            BlobVolume.WriteMRC($"d_blob_{BlobSizes[i]}.mrc", true);

            BlobVolume = BlobVolume.AsFFT(true).AndDisposeParent();
            //BlobVolume.Multiply(BlobCTF);

            BlobVolumes[i] = BlobVolume;
        }

        #endregion

        int2 DimsPositionGrid;
        int3[] PositionGrid = Helper.GetEqualGridSpacing(new int2(new float2(VolumeDimensionsPhysical.X, VolumeDimensionsPhysical.Y) / (float)options.BinnedPixelSizeMean) - 64,
            new int2(SizeRegion),
            0.25f,
            out DimsPositionGrid).Select(v => new int3(v.X + 32 + SizeRegion / 2,
            v.Y + 32 + SizeRegion / 2,
            0)).ToArray();
        float3[] PositionGridPhysical = PositionGrid.Select(v => new float3(v.X * (float)options.BinnedPixelSizeMean,
            v.Y * (float)options.BinnedPixelSizeMean,
            VolumeDimensionsPhysical.Z / 2)).ToArray();

        Image RegionMask = new Image(new int3(SizeRegion));
        RegionMask.Fill(1);
        //RegionMask.MaskRectangularly(new int3(SizeRegion / 2), SizeRegion / 4, true);
        RegionMask.MaskSpherically(SizeRegion / 2, SizeRegion / 4, true, false);
        RegionMask.WriteMRC("d_mask.mrc", true);

        Projector Reconstructor = new Projector(new int3(SizeReconstructionPadded), 1);
        Image Reconstruction = new Image(new int3(SizeReconstructionPadded));
        Image RecCropped = new Image(new int3(SizeReconstruction));

        int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeReconstructionPadded, SizeReconstructionPadded, 1), (uint)NTilts);
        int PlanForwRec, PlanBackRec, PlanCTFBack;
        Projector.GetPlans(Reconstructor.Dims, Reconstructor.Oversampling, out PlanForwRec, out PlanBackRec, out PlanCTFBack);

        Image ParticlesExtracted = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts));
        Image ParticlesExtractedFT = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);

        Image RecSum = new Image(new int3(1));
        float[] h_RecSum = new float[1];

        GridMovementX = GridMovementX.Resize(new int3(1, 1, NTilts));
        GridMovementY = GridMovementY.Resize(GridMovementX.Dimensions);

        float2[] VecZ = new float2[NTilts];

        #region Parameter setting

        float[] OriX = GridMovementX.Values.ToArray();
        float[] OriY = GridMovementY.Values.ToArray();

        int OptimizedTilts = 1;

        Action<TiltSeries, double[]> SetParams = (series, input) =>
        {
            var ParamsX = input.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
            var ParamsY = input.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
            float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
            //for (int i = 1; i <= NTilts / 2; i++)
            //{
            //    var PairXY = new List<float2>(2);
            //    var PairZ = new List<float2>(2);

            //    if (NTilts / 2 - i >= 0)
            //    {
            //        PairXY.Add(ParamsXY[NTilts / 2 - i]);
            //        PairZ.Add(VecZ[NTilts / 2 - i]);
            //    }
            //    if (NTilts / 2 + i < NTilts)
            //    {
            //        PairXY.Add(ParamsXY[NTilts / 2 + i]);
            //        PairZ.Add(VecZ[NTilts / 2 + i]);
            //    }
            //    float Length = MathF.Sqrt(PairZ.Select(v => v.LengthSq()).Sum());
            //    if (Length > 0)
            //    {
            //        float Dot = PairXY.Zip(PairZ, (v, z) => float2.Dot(v, z / Length)).Sum();

            //        if (NTilts / 2 - i >= 0)
            //            ParamsXY[NTilts / 2 - i] -= VecZ[NTilts / 2 - i] / Length * Dot;
            //        if (NTilts / 2 + i < NTilts)
            //            ParamsXY[NTilts / 2 + i] -= VecZ[NTilts / 2 + i] / Length * Dot;
            //    }
            //}
            if (OptimizedTilts <= 0)
            {
                float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
            }

            series.GridMovementX = new CubicGrid(GridMovementX.Dimensions, ParamsXY.Zip(OriX, (v, o) => v.X + o).ToArray());
            series.GridMovementY = new CubicGrid(GridMovementY.Dimensions, ParamsXY.Zip(OriY, (v, o) => v.Y + o).ToArray());
        };

        #endregion

        #region Wiggle weights

        int NWiggleDifferentiable = GridMovementX.Values.Length +
                                    GridMovementY.Values.Length;
        (int[] indices, float2[] weights)[] AllWiggleWeights = new (int[] indices, float2[] weights)[NWiggleDifferentiable];

        {
            TiltSeries[] ParallelSeriesCopies = Helper.ArrayOfFunction(i => new TiltSeries(this.Path), 1);

            int NPositions = PositionGrid.Length;

            var ParticlePositionsOri = new float2[NPositions * NTilts];
            for (int p = 0; p < NPositions; p++)
            {
                float3[] Positions = GetPositionInAllTilts(PositionGridPhysical[p]);
                for (int t = 0; t < NTilts; t++)
                    ParticlePositionsOri[p * NTilts + t] = new float2(Positions[t].X, Positions[t].Y);
            }

            Helper.ForCPU(0, NWiggleDifferentiable, ParallelSeriesCopies.Length,
                (threadID) =>
                {
                    ParallelSeriesCopies[threadID].VolumeDimensionsPhysical = VolumeDimensionsPhysical;
                    ParallelSeriesCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                    ParallelSeriesCopies[threadID].SizeRoundingFactors = SizeRoundingFactors;
                },
                (iwiggle, threadID) =>
                {
                    double[] WiggleParams = new double[NWiggleDifferentiable];
                    WiggleParams[iwiggle] = 1;
                    SetParams(ParallelSeriesCopies[threadID], WiggleParams);

                    float2[] RawShifts = new float2[NPositions * NTilts];
                    for (int p = 0; p < NPositions; p++)
                    {
                        float3[] ParticlePositionsProjected = ParallelSeriesCopies[threadID].GetPositionInAllTilts(PositionGridPhysical[p]);

                        for (int t = 0; t < NTilts; t++)
                            RawShifts[p * NTilts + t] = new float2(ParticlePositionsProjected[t]) - ParticlePositionsOri[p * NTilts + t];
                    }

                    List<int> Indices = new List<int>();
                    List<float2> Weights = new List<float2>();
                    for (int i = 0; i < RawShifts.Length; i++)
                    {
                        if (RawShifts[i].LengthSq() > 1e-6f)
                        {
                            Indices.Add(i / NTilts);
                            Weights.Add(RawShifts[i]);

                            if (Math.Abs(RawShifts[i].X) > 1.5f)
                            {
                                throw new Exception("");
                            }
                        }
                    }

                    AllWiggleWeights[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                }, null);
        }

        #endregion

        Func<double[], float2, int, double[]> EvalIndividual = (input, shiftXY, shiftedTilt) =>
        {
            if (shiftXY != new float2(0))
            {
                input = input.ToArray();
                input[shiftedTilt] += shiftXY.X;
                input[NTilts + shiftedTilt] += shiftXY.Y;
            }

            SetParams(this, input);

            double[] Result = new double[PositionGrid.Length];

            for (int ipos = 0; ipos < PositionGrid.Length; ipos++)
            {
                var Pos = PositionGridPhysical[ipos];

                Reconstructor.Data.Fill(0);
                Reconstructor.Weights.Fill(0);

                GetImagesForOneParticle(options, TiltData, SizeReconstructionPadded, Pos, PlanForwParticles, SizeReconstructionPadded - 36, 16, ParticlesExtracted, ParticlesExtractedFT);
                //ParticlesExtractedFT.Fill(new float2(1f / (SizeReconstruction * SizeReconstruction), 0));
                //ParticlesExtractedFT.ShiftSlices(Helper.ArrayOfFunction(t => new float3((float)(GridMovementX.Values[t] / (float)options.BinnedPixelSizeMean), 
                //                                                                        (float)(GridMovementY.Values[t] / (float)options.BinnedPixelSizeMean), 0), NTilts));
                //if (shiftXY != new float2(0))
                //    ParticlesExtractedFT.ShiftSlices(Helper.ArrayOfFunction(t => t == shiftedTilt ? new float3(shiftXY) / (float)options.BinnedPixelSizeMean : new float3(0), NTilts));
                ParticlesExtractedFT.Multiply(RecCTFs);
                if (OptimizedTilts > 0)
                    ParticlesExtractedFT.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                Reconstructor.BackProject(ParticlesExtractedFT, RecCTFsAbs, GetAngleInAllTilts(Pos), MagnificationCorrection);
                Reconstructor.Data.Multiply(Multiplicity);
                //Reconstructor.Weights.Fill(1);

                Reconstructor.Reconstruct(Reconstruction.GetDevice(Intent.Write), false, "C1", null, PlanForwRec, PlanBackRec, PlanCTFBack, 0);
                //Reconstruction.WriteMRC("d_reconstruction.mrc", true);

                GPU.Pad(Reconstruction.GetDevice(Intent.Read),
                    RecCropped.GetDevice(Intent.Write),
                    new int3(SizeReconstructionPadded),
                    new int3(SizeReconstruction),
                    1);
                //RecCropped.Max(0.1f);

                RecCropped.Multiply(SizeReconstruction);
                RecCropped.Abs();
                RecCropped.Multiply(RecCropped);
                RecCropped.Multiply(RecCropped);
                RecCropped.Multiply(RegionMask);

                GPU.Sum(RecCropped.GetDevice(Intent.Read), RecSum.GetDevice(Intent.Write), (uint)RecCropped.ElementsReal, 1);
                GPU.CopyDeviceToHost(RecSum.GetDevice(Intent.Read), h_RecSum, 1);

                Result[ipos] = h_RecSum[0] / RecCropped.ElementsReal * 100f / PositionGrid.Length;
            }

            return Result;
        };

        Func<double[], double> Eval = (input) => { return EvalIndividual(input, new float2(0), 0).Sum(); };

        Func<double[], double[]> Grad = (input) =>
        {
            double Delta = 0.5;
            double[] Result = new double[input.Length];

            //for (int i = 0; i < input.Length; i++)
            //{
            //    int itilt = (i / (int)GridMovementX.Dimensions.ElementsSlice()) % NTilts;
            //    if (itilt == NTilts / 2)
            //        continue;

            //    if (itilt != NTilts / 2 - OptimizedTilts && itilt != NTilts / 2 + OptimizedTilts)
            //        continue;

            //    double[] InputPlus = input.ToArray();
            //    InputPlus[i] += Delta;
            //    double ScorePlus = Eval(InputPlus);

            //    double[] InputMinus = input.ToArray();
            //    InputMinus[i] -= Delta;
            //    double ScoreMinus = Eval(InputMinus);

            //    Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
            //}

            int NSlice = (int)GridMovementX.Dimensions.ElementsSlice();

            for (int i = 0; i < GridMovementX.Dimensions.Z; i++)
            {
                int itilt = i;
                if (itilt == NTilts / 2)
                    continue;

                if (OptimizedTilts > 0)
                    if (itilt != NTilts / 2 - OptimizedTilts && itilt != NTilts / 2 + OptimizedTilts)
                        continue;

                // Flipping the sign here because EvalIndividual image shift goes in the opposite direction of particle position change
                double[] ScorePlusX = EvalIndividual(input, new float2(-(float)Delta, 0), itilt);
                double[] ScoreMinusX = EvalIndividual(input, new float2((float)Delta, 0), itilt);
                float[] GradsX = ScorePlusX.Zip(ScoreMinusX, (p, m) => (float)((p - m) / (Delta * 2))).ToArray();

                double[] ScorePlusY = EvalIndividual(input, new float2(0, -(float)Delta), itilt);
                double[] ScoreMinusY = EvalIndividual(input, new float2(0, (float)Delta), itilt);
                float[] GradsY = ScorePlusY.Zip(ScoreMinusY, (p, m) => (float)((p - m) / (Delta * 2))).ToArray();

                float2[] Grads = GradsX.Zip(GradsY, (x, y) => new float2(x, y)).ToArray();

                for (int ielement = 0; ielement < NSlice; ielement++)
                {
                    {
                        (int[] Elements, float2[] Weights) = AllWiggleWeights[i * NSlice + ielement];
                        float WeightedSum = 0;
                        for (int j = 0; j < Elements.Length; j++)
                            WeightedSum += float2.Dot(Grads[Elements[j]], Weights[j]);

                        Result[i * NSlice + ielement] = WeightedSum;
                    }

                    {
                        (int[] Elements, float2[] Weights) = AllWiggleWeights[GridMovementX.Values.Length + i * NSlice + ielement];
                        float WeightedSum = 0;
                        for (int j = 0; j < Elements.Length; j++)
                            WeightedSum += float2.Dot(Grads[Elements[j]], Weights[j]);

                        Result[GridMovementX.Values.Length + i * NSlice + ielement] = WeightedSum;
                    }
                }
            }

            Console.WriteLine(Eval(input));

            return Result;
        };

        double[] StartValues = new double[GridMovementX.Dimensions.Elements() * 2];

        BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(StartValues.Length, Eval, Grad);
        Optimizer.MaxLineSearch = 10;
        Optimizer.MaxIterations = 10;
        if (true)
            for (int i = 1; i <= NTilts / 2; i++)
            {
                OptimizedTilts = i;

                RecCTFsAbs?.Dispose();
                RecCTFsAbs = RecCTFs.GetCopyGPU();
                RecCTFsAbs.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                {
                    Image OnesComplex = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);
                    OnesComplex.Fill(new float2(1, 0));
                    Image Ones = OnesComplex.AsReal();
                    Ones.Multiply(Helper.ArrayOfFunction(t => (t >= NTilts / 2 - OptimizedTilts && t <= NTilts / 2 + OptimizedTilts) ? 1f : 0, NTilts));

                    Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

                    ProjCTFWeights.BackProject(OnesComplex, Ones, TransformedAngles, MagnificationCorrection);
                    //ProjCTFWeights.Weights.Min(1);

                    Multiplicity?.Dispose();
                    Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();
                    Multiplicity.WriteMRC($"d_multiplicity_{i:D2}.mrc", true);

                    ProjCTFWeights.Dispose();
                    Ones.Dispose();
                    OnesComplex.Dispose();
                }

                {
                    float2[] Pos0 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f).Select(v => new float2(v)).ToArray();
                    float2[] Pos1 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f + float3.UnitZ).Select(v => new float2(v)).ToArray();
                    float2[] Diff = Pos1.Zip(Pos0, (p1, p0) => p1 - p0).ToArray();
                    VecZ = new float2[NTilts];
                    for (int t = 0; t < NTilts; t++)
                        if (t == NTilts / 2 - OptimizedTilts || t == NTilts / 2 + OptimizedTilts)
                            VecZ[t] = Diff[t];
                    //VecZ = Diff;
                    float Length = MathF.Sqrt(VecZ.Select(v => v.LengthSq()).Sum());
                    VecZ = VecZ.Select(v => v / Length).ToArray();
                }

                Eval(StartValues);
                Reconstruction.WriteMRC($"d_reconstruction_ori_{i:D2}.mrc", true);
                Optimizer.Maximize(StartValues);
                Reconstruction.WriteMRC($"d_reconstruction_ali_{i:D2}.mrc", true);

                if (false)
                {
                    var ParamsX = StartValues.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
                    var ParamsY = StartValues.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
                    float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
                    float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                    float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                    ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
                    StartValues = Helper.Combine(ParamsXY.Select(v => (double)v.X).ToArray(), ParamsXY.Select(v => (double)v.Y).ToArray());
                }

                Console.WriteLine(Eval(StartValues));

                int NSlice = (int)GridMovementX.Dimensions.ElementsSlice();

                if (NTilts / 2 - OptimizedTilts - 1 >= 0)
                    for (int e = 0; e < NSlice; e++)
                    {
                        StartValues[(NTilts / 2 - OptimizedTilts - 1) * NSlice + e] = StartValues[(NTilts / 2 - OptimizedTilts) * NSlice + e];
                        StartValues[GridMovementX.Values.Length + (NTilts / 2 - OptimizedTilts - 1) * NSlice + e] = StartValues[GridMovementX.Values.Length + (NTilts / 2 - OptimizedTilts) * NSlice + e];
                    }

                if (NTilts / 2 + OptimizedTilts + 1 < NTilts)
                {
                    for (int e = 0; e < NSlice; e++)
                    {
                        StartValues[(NTilts / 2 + OptimizedTilts + 1) * NSlice + e] = StartValues[(NTilts / 2 + OptimizedTilts) * NSlice + e];
                        StartValues[GridMovementX.Values.Length + (NTilts / 2 + OptimizedTilts + 1) * NSlice + e] = StartValues[GridMovementX.Values.Length + (NTilts / 2 + OptimizedTilts) * NSlice + e];
                    }
                }
            }

        if (true)
        {
            OptimizedTilts = -1;
            RecCTFsAbs?.Dispose();
            RecCTFsAbs = RecCTFs.GetCopyGPU();

            {
                Image OnesComplex = new Image(new int3(SizeReconstructionPadded, SizeReconstructionPadded, NTilts), true, true);
                OnesComplex.Fill(new float2(1, 0));
                Image Ones = OnesComplex.AsReal();

                Projector ProjCTFWeights = new Projector(new int3(SizeReconstructionPadded), 1);

                ProjCTFWeights.BackProject(OnesComplex, Ones, TransformedAngles, MagnificationCorrection);
                //ProjCTFWeights.Weights.Min(1);

                Multiplicity?.Dispose();
                Multiplicity = ProjCTFWeights.Weights.GetCopyGPU();
                Multiplicity.WriteMRC($"d_multiplicity_global.mrc", true);

                ProjCTFWeights.Dispose();
                Ones.Dispose();
                OnesComplex.Dispose();
            }

            {
                float2[] Pos0 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f).Select(v => new float2(v)).ToArray();
                float2[] Pos1 = GetPositionInAllTilts(VolumeDimensionsPhysical * 0.5f + float3.UnitZ).Select(v => new float2(v)).ToArray();
                float2[] Diff = Pos1.Zip(Pos0, (p1, p0) => p1 - p0).ToArray();
                VecZ = Diff;
                VecZ[NTilts / 2] = new float2(0);
                float Length = MathF.Sqrt(VecZ.Select(v => v.LengthSq()).Sum());
                VecZ = VecZ.Select(v => v / Length).ToArray();
            }

            Optimizer.MaxLineSearch = 30;
            Optimizer.MaxIterations = 30;

            Eval(StartValues);
            Reconstruction.WriteMRC($"d_reconstruction_ori_global.mrc", true);
            Optimizer.Maximize(StartValues);
            Reconstruction.WriteMRC($"d_reconstruction_ali_global.mrc", true);

            {
                var ParamsX = StartValues.Take((int)GridMovementX.Dimensions.Elements()).Select(v => (float)v).ToArray();
                var ParamsY = StartValues.Skip((int)GridMovementX.Dimensions.Elements()).Take((int)GridMovementY.Dimensions.Elements()).Select(v => (float)v).ToArray();
                float2[] ParamsXY = Helper.Zip(ParamsX, ParamsY);
                float Dot = ParamsXY.Zip(VecZ, (v, z) => float2.Dot(v, z)).Sum();
                float Scale = MathHelper.FitScaleLeastSq(VecZ.SelectMany(v => new[] { v.X, v.Y }).ToArray(), ParamsXY.SelectMany(v => new[] { v.X, v.Y }).ToArray());
                ParamsXY = ParamsXY.Zip(VecZ, (v, z) => v - z * Scale).ToArray();
                StartValues = Helper.Combine(ParamsXY.Select(v => (double)v.X).ToArray(), ParamsXY.Select(v => (double)v.Y).ToArray());
            }
        }
    }
}