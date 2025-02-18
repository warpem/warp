using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp;

public partial class Movie
{
    public void ExportParticles(Image originalStack, float2[] positions, ProcessingOptionsParticleExport options)
    {
        IsProcessing = true;

        options.PreflipPhases &= OptionsCTF != null;

        #region Make sure all directories are there

        if (options.DoAverage && !Directory.Exists(ParticlesDir))
            Directory.CreateDirectory(ParticlesDir);

        if (options.DoDenoisingPairs)
        {
            Directory.CreateDirectory(ParticlesDenoisingOddDir);
            Directory.CreateDirectory(ParticlesDenoisingEvenDir);
        }

        #endregion

        #region Helper variables

        int NParticles = positions.Length;

        int3 DimsMovie = originalStack.Dims;
        int FirstFrame = Math.Max(0, Math.Min(DimsMovie.Z - 1, options.SkipFirstN));
        int LastFrameExclusive = Math.Min(DimsMovie.Z, DimsMovie.Z - options.SkipLastN);
        DimsMovie.Z = LastFrameExclusive - FirstFrame;

        int3 DimsParticle = new int3(options.BoxSize, options.BoxSize, NParticles);
        int3 DimsPreflip = DimsParticle.MultXY(2);
        int3 DimsExtraction = options.PreflipPhases ? DimsPreflip : DimsParticle;
        int3 DimsResampled = options.BoxSizeResample > 0 ? new int3(options.BoxSizeResample, options.BoxSizeResample, NParticles).MultXY(options.PreflipPhases ? 2 : 1) : DimsExtraction;

        Task WriteHalfAverageAsync = null;

        float PixelSize = (float)options.BinnedPixelSizeMean;

        #endregion

        #region Figure out where to extract, and how much to shift afterwards

        float3[] ParticleCenters = positions.Select(p => new float3(p) / (float)options.BinnedPixelSizeMean).ToArray(); // From Angstrom to binned pixels

        float3[][] ParticleOrigins = Helper.ArrayOfFunction(z =>
        {
            float Z = (z + FirstFrame) / (float)Math.Max(1, originalStack.Dims.Z - 1);
            return ParticleCenters.Select(p =>
            {
                float2 LocalShift = GetShiftFromPyramid(new float3(p.X * PixelSize / options.Dimensions.X, // ParticleCenters are in binned pixels, Dimensions are in Angstrom
                    p.Y * PixelSize / options.Dimensions.Y, Z)) / PixelSize; // Shifts are in physical Angstrom, convert to binned pixels
                return new float3(p.X - LocalShift.X - DimsExtraction.X / 2, p.Y - LocalShift.Y - DimsExtraction.Y / 2, 0);
            }).ToArray();
        }, DimsMovie.Z);

        int3[][] ParticleIntegerOrigins = ParticleOrigins.Select(a => a.Select(p => new int3(p.Floor())).ToArray()).ToArray();
        float3[][] ParticleResidualShifts = Helper.ArrayOfFunction(z =>
                Helper.ArrayOfFunction(i =>
                        new float3(ParticleIntegerOrigins[z][i]) - ParticleOrigins[z][i],
                    NParticles),
            ParticleOrigins.Length);

        #endregion

        #region Pre-calc phase flipping and dose weighting

        Image CTFSign = null;
        Image CTFCoords = CTF.GetCTFCoords(DimsExtraction.X, DimsExtraction.X);
        Image GammaCorrection = CTF.GetGammaCorrection((float)options.BinnedPixelSizeMean, DimsExtraction.X);
        Image[] DoseWeights = null;

        if (options.PreflipPhases || options.DoDenoisingPairs)
        {
            CTFStruct[] Params = positions.Select(p =>
            {
                CTF Local = CTF.GetCopy();
                Local.Defocus = (decimal)GridCTFDefocus.GetInterpolated(new float3(p.X / options.Dimensions.X, p.Y / options.Dimensions.Y, 0));
                return Local.ToStruct();
            }).ToArray();

            CTFSign = new Image(DimsExtraction, true);
            GPU.CreateCTF(CTFSign.GetDevice(Intent.Write),
                CTFCoords.GetDevice(Intent.Read),
                GammaCorrection.GetDevice(Intent.Read),
                (uint)CTFCoords.ElementsSliceComplex,
                Params,
                false,
                (uint)CTFSign.Dims.Z);

            GPU.Sign(CTFSign.GetDevice(Intent.Read),
                CTFSign.GetDevice(Intent.Write),
                CTFSign.ElementsReal);
        }

        if (options.DosePerAngstromFrame != 0)
        {
            float DosePerFrame = (float)options.DosePerAngstromFrame;
            if (options.DosePerAngstromFrame < 0)
                DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

            DoseWeights = Helper.ArrayOfFunction(z =>
            {
                Image Weights = new Image(IntPtr.Zero, DimsExtraction.Slice(), true);

                CTF CTFBfac = new CTF
                {
                    PixelSize = (decimal)PixelSize,
                    Defocus = 0,
                    Amplitude = 1,
                    Cs = 0,
                    Cc = 0,
                    Bfactor = GridDoseBfacs.Values.Length == 1 ? -(decimal)(DosePerFrame * (z + FirstFrame + 0.5f) * 4) : (decimal)GridDoseBfacs.Values[z + FirstFrame],
                    Scale = GridDoseWeights.Values.Length == 1 ? 1 : (decimal)GridDoseWeights.Values[z + FirstFrame]
                };
                GPU.CreateCTF(Weights.GetDevice(Intent.Write),
                    CTFCoords.GetDevice(Intent.Read),
                    IntPtr.Zero,
                    (uint)CTFCoords.ElementsSliceComplex,
                    new[] { CTFBfac.ToStruct() },
                    false,
                    1);

                //if (z % 2 == 1)
                //    Weights.Multiply(0);

                return Weights;
            }, DimsMovie.Z);

            Image DoseWeightsSum = new Image(DimsExtraction.Slice(), true);
            foreach (var weights in DoseWeights)
                DoseWeightsSum.Add(weights);
            DoseWeightsSum.Max(1e-6f);

            foreach (var weights in DoseWeights)
                weights.Divide(DoseWeightsSum);

            DoseWeightsSum.Dispose();
        }

        GammaCorrection.Dispose();
        CTFCoords.Dispose();

        #endregion

        #region Make FFT plans and memory

        Image Extracted = new Image(IntPtr.Zero, DimsExtraction);
        Image ExtractedFT = new Image(IntPtr.Zero, DimsExtraction, true, true);
        Image ExtractedFTResampled = options.BoxSizeResample > 0 ? new Image(IntPtr.Zero, DimsResampled, true, true) : ExtractedFT;

        Image AverageFT = options.DoAverage ? new Image(DimsResampled, true, true) : null;
        Image[] AverageOddEvenFT = options.DoDenoisingPairs ? Helper.ArrayOfFunction(i => new Image(DimsResampled, true, true), 2) : null;

        int BatchSize = Math.Min(NParticles, 128);
        int PlanForw = GPU.CreateFFTPlan(DimsExtraction.Slice(), (uint)BatchSize);

        int3 DimsBuffer = new int3(Math.Max(DimsExtraction.X, DimsResampled.X), Math.Max(DimsExtraction.Y, DimsResampled.Y), BatchSize);
        Image BufferFFTSource = new Image(IntPtr.Zero, DimsBuffer);
        Image BufferFFTTarget = new Image(IntPtr.Zero, DimsBuffer, true, true);

        #endregion

        #region Extract and process everything

        for (int nframe = 0; nframe < DimsMovie.Z; nframe++)
        {
            GPU.Extract(originalStack.GetDeviceSlice(nframe + FirstFrame, Intent.Read),
                Extracted.GetDevice(Intent.Write),
                DimsMovie.Slice(),
                DimsExtraction.Slice(),
                Helper.ToInterleaved(ParticleIntegerOrigins[nframe]),
                false,
                (uint)NParticles);

            for (int b = 0; b < NParticles; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, NParticles - b);

                GPU.CopyDeviceToDevice(Extracted.GetDeviceSlice(b, Intent.Read),
                    BufferFFTSource.GetDevice(Intent.Write),
                    CurBatch * DimsExtraction.ElementsSlice());

                GPU.FFT(BufferFFTSource.GetDevice(Intent.Read),
                    BufferFFTTarget.GetDevice(Intent.Write),
                    DimsExtraction.Slice(),
                    (uint)BatchSize,
                    PlanForw);

                GPU.CopyDeviceToDevice(BufferFFTTarget.GetDevice(Intent.Read),
                    ExtractedFT.GetDeviceSlice(b, Intent.Write),
                    CurBatch * DimsExtraction.ElementsFFTSlice() * 2);
            }

            if (CTFSign != null)
                ExtractedFT.Multiply(CTFSign);

            if (options.DosePerAngstromFrame != 0)
                ExtractedFT.MultiplySlices(DoseWeights[nframe]);

            ExtractedFT.ShiftSlices(ParticleResidualShifts[nframe]);

            if (options.BoxSizeResample > 0)
                GPU.CropFT(ExtractedFT.GetDevice(Intent.Read),
                    ExtractedFTResampled.GetDevice(Intent.Write),
                    DimsExtraction.Slice(),
                    DimsResampled.Slice(),
                    (uint)NParticles);

            if (options.DoAverage || options.DoDenoisingPairs)
            {
                if (options.DoAverage)
                    AverageFT.Add(ExtractedFTResampled);

                if (options.DoDenoisingPairs)
                    AverageOddEvenFT[nframe % 2].Add(ExtractedFTResampled);
            }
        }

        originalStack.FreeDevice();

        #region Clean up #1

        Extracted.Dispose();
        ExtractedFT.Dispose();
        if (options.BoxSizeResample > 0)
            ExtractedFTResampled.Dispose();
        GPU.DestroyFFTPlan(PlanForw);

        #endregion

        int PlanBack = GPU.CreateIFFTPlan(DimsResampled.Slice(), (uint)BatchSize);

        if (options.DoAverage)
        {
            Image Average = new Image(IntPtr.Zero, DimsResampled);

            for (int b = 0; b < NParticles; b += BatchSize)
            {
                int CurBatch = Math.Min(BatchSize, NParticles - b);

                GPU.CopyDeviceToDevice(AverageFT.GetDeviceSlice(b, Intent.Read),
                    BufferFFTTarget.GetDevice(Intent.Write),
                    CurBatch * DimsResampled.ElementsFFTSlice() * 2);

                GPU.IFFT(BufferFFTTarget.GetDevice(Intent.Read),
                    BufferFFTSource.GetDevice(Intent.Write),
                    DimsResampled.Slice(),
                    (uint)BatchSize,
                    PlanBack,
                    false);

                GPU.CopyDeviceToDevice(BufferFFTSource.GetDevice(Intent.Read),
                    Average.GetDeviceSlice(b, Intent.Write),
                    CurBatch * DimsResampled.ElementsSlice());
            }

            AverageFT.Dispose();

            Image AverageCropped = Average.AsPadded(new int2(DimsParticle));
            Average.Dispose();

            #region Subtract background plane

            AverageCropped.FreeDevice();
            float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
            for (int p = 0; p < NParticles; p++)
            {
                float[] ParticleData = AverageData[p];
                float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                for (int i = 0; i < ParticleData.Length; i++)
                    ParticleData[i] -= Background[i];
            }

            #endregion

            if (options.Normalize)
                GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                    AverageCropped.GetDevice(Intent.Write),
                    DimsParticle.Slice(),
                    (uint)(options.Diameter / PixelSize / 2),
                    options.Invert,
                    (uint)NParticles);
            else if (options.Invert)
                AverageCropped.Multiply(-1f);

            AverageCropped.FreeDevice();

            WriteHalfAverageAsync = new Task(() =>
            {
                float PixelSizeResampled = (options.BoxSizeResample > 0 ? ((float)options.BoxSize / options.BoxSizeResample) : 1) * (float)options.BinnedPixelSizeMean;
                AverageCropped.WriteMRC16b(System.IO.Path.Combine(ParticlesDir, RootName + options.Suffix + ".mrcs"), PixelSizeResampled, true);
                AverageCropped.Dispose();
            });
            WriteHalfAverageAsync.Start();
            GlobalTasks.Add(WriteHalfAverageAsync);
        }

        if (options.DoDenoisingPairs)
        {
            string[] OddEvenDir = { ParticlesDenoisingOddDir, ParticlesDenoisingEvenDir };

            for (int idenoise = 0; idenoise < 2; idenoise++)
            {
                Image Average = new Image(IntPtr.Zero, AverageOddEvenFT[idenoise].Dims); // AverageOddEvenFT[idenoise].AsIFFT(false, PlanBack, true);

                for (int b = 0; b < NParticles; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NParticles - b);

                    GPU.CopyDeviceToDevice(AverageOddEvenFT[idenoise].GetDeviceSlice(b, Intent.Read),
                        BufferFFTTarget.GetDevice(Intent.Write),
                        CurBatch * DimsResampled.ElementsFFTSlice() * 2);

                    GPU.IFFT(BufferFFTTarget.GetDevice(Intent.Read),
                        BufferFFTSource.GetDevice(Intent.Write),
                        DimsResampled.Slice(),
                        (uint)BatchSize,
                        PlanBack,
                        false);

                    GPU.CopyDeviceToDevice(BufferFFTSource.GetDevice(Intent.Read),
                        Average.GetDeviceSlice(b, Intent.Write),
                        CurBatch * DimsResampled.ElementsSlice());
                }

                AverageOddEvenFT[idenoise].Dispose();

                Image AverageCropped = Average.AsPadded(new int2(DimsParticle));
                Average.Dispose();

                #region Subtract background plane

                AverageCropped.FreeDevice();
                float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
                for (int p = 0; p < NParticles; p++)
                {
                    float[] ParticleData = AverageData[p];
                    float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                    for (int i = 0; i < ParticleData.Length; i++)
                        ParticleData[i] -= Background[i];
                }

                #endregion

                if (options.Normalize)
                    GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                        AverageCropped.GetDevice(Intent.Write),
                        DimsParticle.Slice(),
                        (uint)(options.Diameter / PixelSize / 2),
                        options.Invert,
                        (uint)NParticles);
                else if (options.Invert)
                    AverageCropped.Multiply(-1f);

                AverageCropped.FreeDevice();

                WriteHalfAverageAsync = new Task(() =>
                {
                    float PixelSizeResampled = (options.BoxSizeResample > 0 ? ((float)options.BoxSize / options.BoxSizeResample) : 1) * (float)options.BinnedPixelSizeMean;
                    AverageCropped.WriteMRC16b(System.IO.Path.Combine(OddEvenDir[idenoise], RootName + options.Suffix + ".mrcs"), PixelSizeResampled, true);
                    AverageCropped.Dispose();
                });
                WriteHalfAverageAsync.Start();
                GlobalTasks.Add(WriteHalfAverageAsync);
            }
        }

        #endregion

        #region Clean up #2

        GPU.DestroyFFTPlan(PlanBack);
        BufferFFTSource.Dispose();
        BufferFFTTarget.Dispose();

        CTFSign?.Dispose();
        if (DoseWeights != null)
            foreach (var doseWeight in DoseWeights)
                doseWeight.Dispose();

        #endregion

        // Wait for all async IO to finish
        //WriteAverageAsync?.Wait();

        OptionsParticlesExport = options;
        SaveMeta();

        IsProcessing = false;
    }
}

[Serializable]
public class ProcessingOptionsParticleExport : ProcessingOptionsBase
{
    [WarpSerializable] public string Suffix { get; set; }
    [WarpSerializable] public int BoxSize { get; set; }
    [WarpSerializable] public int BoxSizeResample { get; set; }
    [WarpSerializable] public int Diameter { get; set; }
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool Normalize { get; set; }
    [WarpSerializable] public bool DoAverage { get; set; }
    [WarpSerializable] public bool DoDenoisingPairs { get; set; }
    [WarpSerializable] public int StackGroupSize { get; set; }
    [WarpSerializable] public int SkipFirstN { get; set; }
    [WarpSerializable] public int SkipLastN { get; set; }
    [WarpSerializable] public decimal DosePerAngstromFrame { get; set; }
    [WarpSerializable] public int Voltage { get; set; }
    [WarpSerializable] public bool CorrectAnisotropy { get; set; }
    [WarpSerializable] public bool PreflipPhases { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsParticleExport)obj);
    }

    protected bool Equals(ProcessingOptionsParticleExport other)
    {
        return base.Equals(other) &&
               Suffix == other.Suffix &&
               BoxSize == other.BoxSize &&
               BoxSizeResample == other.BoxSizeResample &&
               Diameter == other.Diameter &&
               Invert == other.Invert &&
               Normalize == other.Normalize &&
               DoAverage == other.DoAverage &&
               DoDenoisingPairs == other.DoDenoisingPairs &&
               StackGroupSize == other.StackGroupSize &&
               SkipFirstN == other.SkipFirstN &&
               SkipLastN == other.SkipLastN &&
               DosePerAngstromFrame == other.DosePerAngstromFrame &&
               Voltage == other.Voltage &&
               CorrectAnisotropy == other.CorrectAnisotropy &&
               PreflipPhases == other.PreflipPhases;
    }

    public static bool operator ==(ProcessingOptionsParticleExport left, ProcessingOptionsParticleExport right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsParticleExport left, ProcessingOptionsParticleExport right)
    {
        return !Equals(left, right);
    }
}