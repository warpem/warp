using System;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public string[] ReconstructSubtomos(ProcessingOptionsTomoSubReconstruction options, float3[] positions, float3[] angles)
    {
        int GPUID = GPU.GetDevice();

        bool IsCanceled = false;
        if (options.UseCPU)
            Console.WriteLine("Using CPU");

        if (!Directory.Exists(SubtomoDir))
            Directory.CreateDirectory(SubtomoDir);

        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        CTF MaxDefocusCTF = GetTiltCTF(IndicesSortedDose[0]);
        MaxDefocusCTF.PixelSize = options.BinnedPixelSizeMean;
        int MinimumBoxSize = (int)Math.Round(MaxDefocusCTF.GetAliasingFreeSize((float)options.BinnedPixelSizeMean * 2, (float)(options.ParticleDiameter / options.BinnedPixelSizeMean)) / 2f) * 2;

        int SizeSub = options.BoxSize;
        int SizeSubSuper = Math.Min(1024, Math.Max(SizeSub * 2, MinimumBoxSize));

        #endregion

        #region Load and preprocess tilt series

        Movie[] TiltMovies;
        Image[] TiltData;
        Image[] TiltMasks;
        LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
        LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            EraseDirt(TiltData[z], TiltMasks[z]);
            TiltMasks[z]?.FreeDevice();

            if (options.NormalizeInput)
            {
                TiltData[z].SubtractMeanGrid(new int2(1));
                TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);

                GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                    TiltData[z].GetDevice(Intent.Write),
                    (uint)TiltData[z].ElementsReal,
                    1);
            }
            else
            {
                TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);
            }

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            TiltData[z].FreeDevice();

            //TiltData[z].Multiply(TiltMasks[z]);
        }

        #endregion

        #region Memory and FFT plan allocation

        Image CTFCoords = CTF.GetCTFCoords(SizeSubSuper, SizeSubSuper);

        int NThreads = 1;

        int[] PlanForwRec = new int[NThreads], PlanBackRec = new int[NThreads];
        if (!options.UseCPU)
            for (int i = 0; i < NThreads; i++)
            {
                //Projector.GetPlans(new int3(SizeSubSuper), 1, out PlanForwRec[i], out PlanBackRec[i], out PlanForwCTF[i]);
                PlanForwRec[i] = GPU.CreateFFTPlan(new int3(SizeSubSuper), 1);
                PlanBackRec[i] = GPU.CreateIFFTPlan(new int3(SizeSubSuper), 1);
            }

        int[] PlanForwRecCropped = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSub), 1), NThreads);
        int[] PlanForwParticle = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSubSuper, SizeSubSuper, 1), (uint)NTilts), NThreads);

        Projector[] Projectors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubSuper), 1), NThreads);
        Projector[] ProjectorsMultiplicity = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubSuper), 1), NThreads);

        Image[] VolumeCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);
        Image[] VolumeCTFCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub), true), NThreads);

        Image[] Subtomo = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper)), NThreads);
        float[][] CPUBuffer = null;
        if (options.UseCPU)
            CPUBuffer = Helper.ArrayOfFunction(i => new float[new int3(SizeSubSuper).Elements()], NThreads);
        //Image[] SubtomoCTF = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper), true), NThreads);
        //Image[] SubtomoCTFComplex = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper), true, true), NThreads);
        Image[] SubtomoSparsityMask = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub), true), NThreads);
        Image[] Images = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts)), NThreads);
        Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true, true), NThreads);
        Image[] CTFs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
        Image[] CTFsAbs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
        Image[] CTFsUnweighted = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true), NThreads);
        Image[] CTFsComplex = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubSuper, SizeSubSuper, NTilts), true, true), NThreads);

        Image[] SumAllParticles = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);

        GPU.CheckGPUExceptions();

        #endregion

        float[] TiltWeights = new float[NTilts];
        if (options.DoLimitDose)
            for (int t = 0; t < Math.Min(NTilts, options.NTilts); t++)
            {
                int TiltIndex = IndicesSortedDose[t];
                TiltWeights[TiltIndex] = UseTilt[TiltIndex] ? 1f : 0f;
            }
        else
            TiltWeights = UseTilt.Select(v => v ? 1f : 0f).ToArray();

        int NParticles = positions.Length / NTilts;
        string[] ParticleVisibleFrames = new string[NParticles];
        int NReconstructed = 0;

        // Match the 2D selection and frame ordering.
        int[] UsedTilts = options.DoLimitDose
            ? IndicesSortedDose.Take(options.NTilts).ToArray()
            : IndicesSortedDose.ToArray();
        Array.Sort(UsedTilts);

        Helper.ForCPU(0, NParticles, NThreads,
            threadID => GPU.SetDevice(GPUID),
            (p, threadID) =>
            {
                if (IsCanceled)
                    return;

                float3[] ParticlePositions = positions.Skip(p * NTilts).Take(NTilts).ToArray();
                float3[] ParticleAngles = options.PrerotateParticles ? angles.Skip(p * NTilts).Take(NTilts).ToArray() : null;

                // Match the particle-footprint visibility rule used by 2D export.
                float HalfDiameter = options.ParticleDiameter / 2f;
                float3[] ImagePositions = GetPositionInAllTilts(ParticlePositions);
                float[] ParticleTiltWeights = new float[NTilts];
                for (int t = 0; t < NTilts; t++)
                {
                    float3 ImagePosition = ImagePositions[t];
                    bool Visible = ImagePosition.X > HalfDiameter &&
                                   ImagePosition.X < ImageDimensionsPhysical.X - HalfDiameter &&
                                   ImagePosition.Y > HalfDiameter &&
                                   ImagePosition.Y < ImageDimensionsPhysical.Y - HalfDiameter;
                    ParticleTiltWeights[t] = Visible ? TiltWeights[t] : 0f;
                }

                ParticleVisibleFrames[p] =
                    $"[{string.Join(",", UsedTilts.Select(t =>
                        ParticleTiltWeights[t] > 0f ? "1" : "0").ToArray())}]";

                // The manager removes this row before writing the final STAR.
                if (!ParticleTiltWeights.Any(w => w > 0f))
                    return;

                #region Multiplicity

                ProjectorsMultiplicity[threadID].Data.Fill(0);
                ProjectorsMultiplicity[threadID].Weights.Fill(0);
                CTFsComplex[threadID].Fill(new float2(1, 0));
                CTFs[threadID].Fill(1);
                CTFs[threadID].Multiply(ParticleTiltWeights);
                CTFsComplex[threadID].Multiply(ParticleTiltWeights);

                ProjectorsMultiplicity[threadID].BackProject(CTFsComplex[threadID], CTFs[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);
                ProjectorsMultiplicity[threadID].Weights.Min(1);

                #endregion

                Timing.Start("ExtractImageData");
                GetImagesForOneParticle(options, TiltData, SizeSubSuper, ParticlePositions, PlanForwParticle[threadID], -1, 8, true, Images[threadID], ImagesFT[threadID]);
                Timing.Finish("ExtractImageData");

                Timing.Start("CreateRawCTF");
                GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, true, false, false, CTFs[threadID]);
                GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFsUnweighted[threadID]);
                Timing.Finish("CreateRawCTF");

                // Exclude invalid tilts from both the signal and denominator.
                CTFs[threadID].Multiply(ParticleTiltWeights);
                CTFsUnweighted[threadID].Multiply(ParticleTiltWeights);

                // Subtomo is (Image * CTFweighted) / abs(CTFunweighted)
                // 3D CTF is (CTFweighted * CTFweighted) / abs(CTFweighted)

                ImagesFT[threadID].Multiply(CTFs[threadID]);
                //GPU.Abs(CTFs[threadID].GetDevice(Intent.Read),
                //        CTFsAbs[threadID].GetDevice(Intent.Write),
                //        CTFs[threadID].ElementsReal);

                CTFsComplex[threadID].Fill(new float2(1, 0));
                CTFsComplex[threadID].Multiply(CTFsUnweighted[threadID]); // What the raw image is like: unweighted, unflipped
                CTFsComplex[threadID].Multiply(CTFs[threadID]); // Weight by the same CTF as raw image: weighted, unflipped

                CTFsUnweighted[threadID].Abs();

                #region Sub-tomo

                Projectors[threadID].Data.Fill(0);
                Projectors[threadID].Weights.Fill(0);

                Timing.Start("ProjectImageData");
                Projectors[threadID].BackProject(ImagesFT[threadID], CTFsUnweighted[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);
                Timing.Finish("ProjectImageData");

                //Projectors[threadID].Weights.Fill(1);
                Projectors[threadID].Data.Multiply(ProjectorsMultiplicity[threadID].Weights);

                Timing.Start("ReconstructSubtomo");
                if (options.UseCPU)
                    Projectors[threadID].ReconstructCPU(Subtomo[threadID], CPUBuffer[threadID], false, "C1");
                else
                    Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForwRec[threadID], PlanBackRec[threadID], PlanForwRec[threadID], 0);
                Timing.Finish("ReconstructSubtomo");

                GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                    VolumeCropped[threadID].GetDevice(Intent.Write),
                    new int3(SizeSubSuper),
                    new int3(SizeSub),
                    1);

                if (options.NormalizeOutput)
                    GPU.NormParticles(VolumeCropped[threadID].GetDevice(Intent.Read),
                        VolumeCropped[threadID].GetDevice(Intent.Write),
                        new int3(SizeSub),
                        (uint)Math.Round(options.ParticleDiameter / options.BinnedPixelSizeMean / 2),
                        false,
                        1);

                SumAllParticles[threadID].Add(VolumeCropped[threadID]);

                VolumeCropped[threadID].WriteMRC16b(System.IO.Path.Combine(SubtomoDir, $"{RootName}{options.Suffix}_{p:D7}_{options.BinnedPixelSizeMean:F2}A.mrc"), (float)options.BinnedPixelSizeMean, true);

                #endregion

                #region CTF

                // Back-project and reconstruct
                Projectors[threadID].Data.Fill(0);
                Projectors[threadID].Weights.Fill(0);

                Projectors[threadID].BackProject(CTFsComplex[threadID], CTFsUnweighted[threadID], !options.PrerotateParticles ? GetAngleInAllTilts(ParticlePositions) : GetParticleAngleInAllTilts(ParticlePositions, ParticleAngles), MagnificationCorrection);

                //Projectors[threadID].Weights.Fill(1);
                Projectors[threadID].Data.Multiply(ProjectorsMultiplicity[threadID].Weights);

                Timing.Start("ReconstructCTF");
                if (options.UseCPU)
                    Projectors[threadID].ReconstructCPU(Subtomo[threadID], CPUBuffer[threadID], false, "C1");
                else
                    Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForwRec[threadID], PlanBackRec[threadID], PlanForwRec[threadID], 0);
                Timing.Finish("ReconstructCTF");

                Timing.Start("3DCTFCrop");
                //SubtomoCTFComplex[threadID].Fill(new float2(1, 0));
                //SubtomoCTFComplex[threadID].Multiply(SubtomoCTF[threadID]);
                //GPU.IFFT(SubtomoCTFComplex[threadID].GetDevice(Intent.Read),
                //         Subtomo[threadID].GetDevice(Intent.Write),
                //         new int3(SizeSubSuper),
                //         1,
                //         PlanBackRec[threadID],
                //         false);

                GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                    VolumeCropped[threadID].GetDevice(Intent.Write),
                    new int3(SizeSubSuper),
                    new int3(SizeSub),
                    1);

                GPU.FFT(VolumeCropped[threadID].GetDevice(Intent.Read),
                    Subtomo[threadID].GetDevice(Intent.Write),
                    new int3(SizeSub),
                    1,
                    PlanForwRecCropped[threadID]);

                GPU.ShiftStackFT(Subtomo[threadID].GetDevice(Intent.Read),
                    Subtomo[threadID].GetDevice(Intent.Write),
                    new int3(SizeSub),
                    new[] { SizeSub / 2f, SizeSub / 2f, SizeSub / 2f },
                    1);

                GPU.Real(Subtomo[threadID].GetDevice(Intent.Read),
                    VolumeCTFCropped[threadID].GetDevice(Intent.Write),
                    VolumeCTFCropped[threadID].ElementsReal);

                VolumeCTFCropped[threadID].Multiply(1f / (SizeSubSuper * SizeSubSuper));
                Timing.Finish("3DCTFCrop");

                if (options.MakeSparse)
                {
                    GPU.Abs(VolumeCTFCropped[threadID].GetDevice(Intent.Read),
                        SubtomoSparsityMask[threadID].GetDevice(Intent.Write),
                        VolumeCTFCropped[threadID].ElementsReal);
                    SubtomoSparsityMask[threadID].Binarize(0.01f);

                    VolumeCTFCropped[threadID].Multiply(SubtomoSparsityMask[threadID]);
                }

                VolumeCTFCropped[threadID].WriteMRC16b(System.IO.Path.Combine(SubtomoDir, $"{RootName}{options.Suffix}_{p:D7}_ctf_{options.BinnedPixelSizeMean:F2}A.mrc"), (float)options.BinnedPixelSizeMean, true);

                #endregion

                System.Threading.Interlocked.Increment(ref NReconstructed);

                //Console.WriteLine(SizeSubSuper);
                //Timing.PrintMeasurements();
            }, null);

        // Write the sum of all particles
        if (NReconstructed > 0)
        {
            for (int i = 1; i < NThreads; i++)
                SumAllParticles[0].Add(SumAllParticles[i]);

            SumAllParticles[0].Multiply(1f / NReconstructed);

            SumAllParticles[0].WriteMRC16b(
                System.IO.Path.Combine(
                    SubtomoDir,
                    $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_average.mrc"),
                (float)options.BinnedPixelSizeMean,
                true);
        }

        #region Teardown

        for (int i = 0; i < NThreads; i++)
        {
            if (!options.UseCPU)
            {
                GPU.DestroyFFTPlan(PlanForwRec[i]);
                GPU.DestroyFFTPlan(PlanBackRec[i]);
            }

            //GPU.DestroyFFTPlan(PlanForwCTF[i]);
            GPU.DestroyFFTPlan(PlanForwParticle[i]);
            Projectors[i].Dispose();
            ProjectorsMultiplicity[i].Dispose();
            Subtomo[i].Dispose();
            //SubtomoCTF[i].Dispose();
            SubtomoSparsityMask[i].Dispose();
            Images[i].Dispose();
            ImagesFT[i].Dispose();
            CTFs[i].Dispose();
            CTFsAbs[i].Dispose();
            CTFsUnweighted[i].Dispose();
            CTFsComplex[i].Dispose();

            SumAllParticles[i].Dispose();

            GPU.DestroyFFTPlan(PlanForwRecCropped[i]);
            VolumeCropped[i].Dispose();
            VolumeCTFCropped[i].Dispose();
            //SubtomoCTFComplex[i].Dispose();
        }

        CTFCoords.Dispose();
        //CTFCoordsPadded.Dispose();
        foreach (var image in TiltData)
            image.FreeDevice();
        foreach (var tiltMask in TiltMasks)
            tiltMask?.FreeDevice();

        #endregion

        if (ParticleVisibleFrames.Any(v => v == null))
            throw new InvalidOperationException(
                $"Visibility was not recorded for every particle in {RootName}.");

        return ParticleVisibleFrames;
    }
}

[Serializable]
public class ProcessingOptionsTomoSubReconstruction : TomoProcessingOptionsBase
{
    [WarpSerializable] public string Suffix { get; set; }
    [WarpSerializable] public int BoxSize { get; set; }
    [WarpSerializable] public int ParticleDiameter { get; set; }
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool NormalizeInput { get; set; }
    [WarpSerializable] public bool NormalizeOutput { get; set; }
    [WarpSerializable] public bool PrerotateParticles { get; set; }
    [WarpSerializable] public bool DoLimitDose { get; set; }
    [WarpSerializable] public int NTilts { get; set; }
    [WarpSerializable] public bool MakeSparse { get; set; }
    [WarpSerializable] public bool UseCPU { get; set; }
    [WarpSerializable] public bool DontPremultiply { get; set; } = false;
}
