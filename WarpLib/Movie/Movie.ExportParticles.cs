using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class Movie
{
    /// <param name="releaseStack">Free the stack's device copy afterwards (see <see cref="ProcessShift"/>).</param>
    public void ExportParticles(Image originalStack, float2[] positions, ProcessingOptionsParticleExport options, bool releaseStack = true)
    {
        // A stack on the GPU is read in place; a host stack goes up a frame at a time instead of all at once, which
        // needs no device copy of the whole movie.
        bool OnDevice = originalStack.IsDeviceCurrent;
        float[][] HostFrames = OnDevice ? null : originalStack.GetHost(Intent.Read);
        IntPtr Staging = OnDevice ? IntPtr.Zero : GPU.MallocDevice(originalStack.ElementsSliceReal);
        try
        {
            using var Export = new MovieParticleExport(this, options, positions, originalStack.Dims, 1);
            for (int z = Export.FirstFrame; z < Export.LastFrameExclusive; z++)
            {
                IntPtr Frame = Staging;
                if (OnDevice)
                    Frame = originalStack.GetDeviceSlice(z, Intent.Read);
                else
                    GPU.CopyHostToDevice(HostFrames[z], Staging, originalStack.ElementsSliceReal);
                Export.AddFrame(z, Frame, 0);
            }

            if (releaseStack)
                originalStack.FreeDevice();

            Export.Finish();
        }
        finally
        {
            if (Staging != IntPtr.Zero)
                GPU.FreeDevice(Staging);
        }
    }
}

/// <summary>
/// Extracts particles from a movie's frames as they come, one frame at a time, so a movie can be streamed from its
/// file without holding the whole stack (see Movie.ExportParticles for a loaded stack). Each frame's boxes follow the
/// particles' local motion; they are weighted by exposure (and pre-flipped by the CTF's sign if asked), shifted by
/// their sub-pixel remainder in Fourier space and summed. <see cref="Finish"/> writes the stacks and the metadata.
/// Frames may arrive in any order and from several threads, each with its own lane (buffers and FFT plan); all GPU
/// work goes to the legacy default stream, which orders the sums.
/// </summary>
public sealed class MovieParticleExport : IDisposable
{
    readonly Movie Owner;
    readonly ProcessingOptionsParticleExport Options;
    readonly int NParticles;
    readonly int Chunk;
    readonly int3 DimsFrame, DimsExtraction, DimsResampled, DimsOutput;
    readonly float PixelSizeOutput;

    /// <summary>The stack's frames that are extracted (the rest are skipped), in stack indices.</summary>
    public readonly int FirstFrame, LastFrameExclusive;

    readonly int[][] FrameOrigins;      // Per extracted frame, each particle's box corner (interleaved int3)
    readonly float[][] FrameShifts;     // Per extracted frame, each particle's sub-pixel remainder (interleaved float3)

    Image CTFSign;
    Image[] DoseWeights;
    Image AverageFT;
    Image[] AverageOddEvenFT;

    sealed class Lane
    {
        public Image Extracted, ExtractedFT, ExtractedFTResampled;
        public int Plan;
    }
    Lane[] Lanes;

    /// <param name="stackDims">The movie's frame size and number of frames as loaded, at the options' binned pixel size.</param>
    /// <param name="lanes">How many threads may add frames at the same time.</param>
    public MovieParticleExport(Movie movie, ProcessingOptionsParticleExport options, float2[] positions, int3 stackDims, int lanes)
    {
        Owner = movie;
        Options = options;
        Owner.IsProcessing = true;

        options.PreflipPhases &= movie.OptionsCTF != null;

        if (options.DoAverage)
            Directory.CreateDirectory(movie.ParticlesDir);
        if (options.DoDenoisingPairs)
        {
            Directory.CreateDirectory(movie.ParticlesDenoisingOddDir);
            Directory.CreateDirectory(movie.ParticlesDenoisingEvenDir);
        }

        NParticles = positions.Length;
        int NFramesStack = stackDims.Z;
        FirstFrame = Math.Max(0, Math.Min(NFramesStack - 1, options.SkipFirstN));
        LastFrameExclusive = Math.Min(NFramesStack, NFramesStack - options.SkipLastN);
        int NFrames = Math.Max(0, LastFrameExclusive - FirstFrame);

        if (options.BoxSizeResample > options.BoxSize)
            throw new ArgumentOutOfRangeException(nameof(options.BoxSizeResample), "Particles can only be resampled to a smaller box.");

        DimsFrame = stackDims.Slice();
        DimsExtraction = new int3(options.BoxSize, options.BoxSize, 1).MultXY(options.PreflipPhases ? 2 : 1);
        // Resampling Fourier-crops each particle: the written box is BoxSizeResample, at a proportionally coarser pixel size.
        DimsResampled = options.BoxSizeResample > 0 ? new int3(options.BoxSizeResample, options.BoxSizeResample, 1).MultXY(options.PreflipPhases ? 2 : 1) : DimsExtraction;
        DimsOutput = options.BoxSizeResample > 0 ? new int3(options.BoxSizeResample, options.BoxSizeResample, 1) : new int3(options.BoxSize, options.BoxSize, 1);

        float PixelSize = (float)options.BinnedPixelSizeMean;
        PixelSizeOutput = PixelSize * options.BoxSize / DimsOutput.X;

        // Boxes in chunks, so the per-frame buffers stay small however many particles a movie has.
        Chunk = Math.Max(1, Math.Min(NParticles, 256));

        #region Where to extract in each frame, and how much to shift afterwards

        float2[] Centers = positions.Select(p => p / PixelSize).ToArray(); // From Angstrom to binned pixels
        FrameOrigins = new int[NFrames][];
        FrameShifts = new float[NFrames][];
        Parallel.For(0, NFrames, f =>
        {
            float Z = (f + FirstFrame) / (float)Math.Max(1, NFramesStack - 1);
            int[] Origins = new int[NParticles * 3];
            float[] Shifts = new float[NParticles * 3];
            for (int p = 0; p < NParticles; p++)
            {
                float2 c = Centers[p];
                // Centers are in binned pixels, Dimensions in Angstrom; shifts are in Angstrom, converted to binned pixels.
                float2 LocalShift = movie.GetShiftFromPyramid(new float3(c.X * PixelSize / options.Dimensions.X, c.Y * PixelSize / options.Dimensions.Y, Z)) / PixelSize;
                float OriginX = c.X - LocalShift.X - DimsExtraction.X / 2, OriginY = c.Y - LocalShift.Y - DimsExtraction.Y / 2;
                int IntX = (int)MathF.Floor(OriginX), IntY = (int)MathF.Floor(OriginY);
                Origins[p * 3 + 0] = IntX;
                Origins[p * 3 + 1] = IntY;
                Shifts[p * 3 + 0] = IntX - OriginX;
                Shifts[p * 3 + 1] = IntY - OriginY;
            }
            FrameOrigins[f] = Origins;
            FrameShifts[f] = Shifts;
        });

        #endregion

        #region Phase flipping and exposure weighting

        Image CTFCoords = CTF.GetCTFCoords(DimsExtraction.X, DimsExtraction.X);
        Image GammaCorrection = movie.CTF.GetGammaCorrection(PixelSize, DimsExtraction.X);

        if (NParticles > 0 && (options.PreflipPhases || options.DoDenoisingPairs))
        {
            CTFStruct[] Params = positions.Select(p =>
            {
                CTF Local = movie.CTF.GetCopy();
                Local.Defocus = (decimal)movie.GridCTFDefocus.GetInterpolated(new float3(p.X / options.Dimensions.X, p.Y / options.Dimensions.Y, 0));
                return Local.ToStruct();
            }).ToArray();

            CTFSign = new Image(new int3(DimsExtraction.X, DimsExtraction.Y, NParticles), true);
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

        if (options.DosePerAngstromFrame != 0 && NFrames > 0)
        {
            float DosePerFrame = (float)options.DosePerAngstromFrame;
            if (options.DosePerAngstromFrame < 0)
                DosePerFrame = -(float)options.DosePerAngstromFrame / NFramesStack;

            DoseWeights = Helper.ArrayOfFunction(f =>
            {
                int z = f + FirstFrame;
                Image Weights = new Image(IntPtr.Zero, DimsExtraction, true);

                CTF CTFBfac = new CTF
                {
                    PixelSize = (decimal)PixelSize,
                    Defocus = 0,
                    Amplitude = 1,
                    Cs = 0,
                    Cc = 0,
                    Bfactor = movie.GridDoseBfacs.Values.Length == 1 ? -(decimal)(DosePerFrame * (z + 0.5f) * 4) : (decimal)movie.GridDoseBfacs.Values[z],
                    Scale = movie.GridDoseWeights.Values.Length == 1 ? 1 : (decimal)movie.GridDoseWeights.Values[z]
                };
                GPU.CreateCTF(Weights.GetDevice(Intent.Write),
                              CTFCoords.GetDevice(Intent.Read),
                              IntPtr.Zero,
                              (uint)CTFCoords.ElementsSliceComplex,
                              new[] { CTFBfac.ToStruct() },
                              false,
                              1);

                return Weights;
            }, NFrames);

            Image DoseWeightsSum = new Image(DimsExtraction, true);
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

        #region Sums and per-lane buffers

        int3 DimsSums = new int3(DimsResampled.X, DimsResampled.Y, Math.Max(1, NParticles));
        Image NewSums()
        {
            // On the GPU only: a host copy of the sums would never be read.
            var Sums = new Image(IntPtr.Zero, DimsSums, true, true);
            Sums.Fill(0);
            return Sums;
        }
        AverageFT = options.DoAverage ? NewSums() : null;
        AverageOddEvenFT = options.DoDenoisingPairs ? Helper.ArrayOfFunction(i => NewSums(), 2) : null;

        Lanes = Helper.ArrayOfFunction(i => new Lane
        {
            Extracted = new Image(IntPtr.Zero, new int3(DimsExtraction.X, DimsExtraction.Y, Chunk)),
            ExtractedFT = new Image(IntPtr.Zero, new int3(DimsExtraction.X, DimsExtraction.Y, Chunk), true, true),
            ExtractedFTResampled = options.BoxSizeResample > 0 ? new Image(IntPtr.Zero, new int3(DimsResampled.X, DimsResampled.Y, Chunk), true, true) : null,
            Plan = GPU.CreateFFTPlan(DimsExtraction, (uint)Chunk)
        }, Math.Max(1, lanes));

        #endregion
    }

    /// <summary>Adds the boxes from one frame (stack index z, on the GPU, of the stack's frame size); lane identifies the calling thread.</summary>
    public void AddFrame(int z, IntPtr frame, int lane)
    {
        if (z < FirstFrame || z >= LastFrameExclusive || NParticles == 0)
            return;

        int f = z - FirstFrame;
        Lane L = Lanes[lane];
        long ElementsFTExtraction = DimsExtraction.ElementsFFTSlice();    // complex
        long ElementsFTResampled = DimsResampled.ElementsFFTSlice();      // complex
        IntPtr Extracted = L.Extracted.GetDevice(Intent.Write);
        IntPtr ExtractedFT = L.ExtractedFT.GetDevice(Intent.Write);
        IntPtr Resampled = L.ExtractedFTResampled != null ? L.ExtractedFTResampled.GetDevice(Intent.Write) : ExtractedFT;

        for (int b = 0; b < NParticles; b += Chunk)
        {
            int CurChunk = Math.Min(Chunk, NParticles - b);

            GPU.Extract(frame,
                        Extracted,
                        DimsFrame,
                        DimsExtraction,
                        FrameOrigins[f].AsSpan(b * 3, CurChunk * 3).ToArray(),
                        false,
                        (uint)CurChunk);

            // The plan's batch is a whole chunk; boxes past the last particle are left over from before, and ignored.
            GPU.FFT(Extracted, ExtractedFT, DimsExtraction, (uint)Chunk, L.Plan);

            if (CTFSign != null)
                GPU.MultiplyComplexSlicesByScalar(ExtractedFT,
                                                  CTFSign.GetDeviceSlice(b, Intent.Read),
                                                  ExtractedFT,
                                                  ElementsFTExtraction * CurChunk,
                                                  1);

            if (DoseWeights != null)
                GPU.MultiplyComplexSlicesByScalar(ExtractedFT,
                                                  DoseWeights[f].GetDevice(Intent.Read),
                                                  ExtractedFT,
                                                  ElementsFTExtraction,
                                                  (uint)CurChunk);

            GPU.ShiftStackFT(ExtractedFT, ExtractedFT, DimsExtraction, FrameShifts[f].AsSpan(b * 3, CurChunk * 3).ToArray(), (uint)CurChunk);

            if (L.ExtractedFTResampled != null)
                GPU.CropFT(ExtractedFT, Resampled, DimsExtraction, DimsResampled, (uint)CurChunk);

            if (AverageFT != null)
            {
                IntPtr Sum = AverageFT.GetDeviceSlice(b, Intent.ReadWrite);
                GPU.AddToSlices(Sum, Resampled, Sum, ElementsFTResampled * 2 * CurChunk, 1);
            }

            if (AverageOddEvenFT != null)
            {
                IntPtr Sum = AverageOddEvenFT[f % 2].GetDeviceSlice(b, Intent.ReadWrite);
                GPU.AddToSlices(Sum, Resampled, Sum, ElementsFTResampled * 2 * CurChunk, 1);
            }
        }
    }

    /// <summary>
    /// Writes the particle stacks (in the background, see GlobalTasks) and saves the movie's metadata. Particles
    /// are inverted and normalized as the options say, after their background plane is subtracted.
    /// </summary>
    public void Finish()
    {
        FreeLanes();

        if (NParticles > 0)
        {
            int BatchSize = Math.Min(NParticles, 128);
            int PlanBack = GPU.CreateIFFTPlan(DimsResampled, (uint)BatchSize);
            Image BufferFFTSource = new Image(IntPtr.Zero, new int3(DimsResampled.X, DimsResampled.Y, BatchSize));
            Image BufferFFTTarget = new Image(IntPtr.Zero, new int3(DimsResampled.X, DimsResampled.Y, BatchSize), true, true);

            void WriteSums(Image sumsFT, string path)
            {
                Image Average = new Image(IntPtr.Zero, new int3(DimsResampled.X, DimsResampled.Y, NParticles));

                for (int b = 0; b < NParticles; b += BatchSize)
                {
                    int CurBatch = Math.Min(BatchSize, NParticles - b);

                    GPU.CopyDeviceToDevice(sumsFT.GetDeviceSlice(b, Intent.Read),
                                           BufferFFTTarget.GetDevice(Intent.Write),
                                           CurBatch * DimsResampled.ElementsFFTSlice() * 2);

                    GPU.IFFT(BufferFFTTarget.GetDevice(Intent.Read),
                             BufferFFTSource.GetDevice(Intent.Write),
                             DimsResampled,
                             (uint)BatchSize,
                             PlanBack,
                             false);

                    GPU.CopyDeviceToDevice(BufferFFTSource.GetDevice(Intent.Read),
                                           Average.GetDeviceSlice(b, Intent.Write),
                                           CurBatch * DimsResampled.ElementsSlice());
                }

                Image AverageCropped = Average.AsPadded(new int2(DimsOutput));
                Average.Dispose();

                // Subtract each particle's background plane
                AverageCropped.FreeDevice();
                float[][] AverageData = AverageCropped.GetHost(Intent.ReadWrite);
                Parallel.For(0, NParticles, p =>
                {
                    float[] ParticleData = AverageData[p];
                    float[] Background = MathHelper.FitAndGeneratePlane(ParticleData, new int2(AverageCropped.Dims));
                    for (int i = 0; i < ParticleData.Length; i++)
                        ParticleData[i] -= Background[i];
                });

                if (Options.Normalize)
                    GPU.NormParticles(AverageCropped.GetDevice(Intent.Read),
                                      AverageCropped.GetDevice(Intent.Write),
                                      DimsOutput,
                                      (uint)(Options.Diameter / PixelSizeOutput / 2),
                                      Options.Invert,
                                      (uint)NParticles);
                else if (Options.Invert)
                    AverageCropped.Multiply(-1f);

                AverageCropped.FreeDevice();

                Task WriteAsync = new Task(() =>
                {
                    AverageCropped.WriteMRC16b(path, PixelSizeOutput, true);
                    AverageCropped.Dispose();
                });
                WriteAsync.Start();
                GlobalTasks.Add(WriteAsync);
            }

            string FileName = Owner.RootName + Options.Suffix + ".mrcs";
            if (AverageFT != null)
                WriteSums(AverageFT, Path.Combine(Owner.ParticlesDir, FileName));
            if (AverageOddEvenFT != null)
            {
                WriteSums(AverageOddEvenFT[0], Path.Combine(Owner.ParticlesDenoisingOddDir, FileName));
                WriteSums(AverageOddEvenFT[1], Path.Combine(Owner.ParticlesDenoisingEvenDir, FileName));
            }

            GPU.DestroyFFTPlan(PlanBack);
            BufferFFTSource.Dispose();
            BufferFFTTarget.Dispose();
        }

        Owner.OptionsParticlesExport = Options;
        Owner.SaveMeta();

        Owner.IsProcessing = false;
    }

    void FreeLanes()
    {
        if (Lanes == null)
            return;
        foreach (var lane in Lanes)
        {
            lane.Extracted.Dispose();
            lane.ExtractedFT.Dispose();
            lane.ExtractedFTResampled?.Dispose();
            GPU.DestroyFFTPlan(lane.Plan);
        }
        Lanes = null;
    }

    public void Dispose()
    {
        FreeLanes();
        CTFSign?.Dispose();
        CTFSign = null;
        if (DoseWeights != null)
            foreach (var weights in DoseWeights)
                weights.Dispose();
        DoseWeights = null;
        AverageFT?.Dispose();
        AverageFT = null;
        if (AverageOddEvenFT != null)
            foreach (var sums in AverageOddEvenFT)
                sums.Dispose();
        AverageOddEvenFT = null;
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

    public override int GetHashCode()
    {
        return base.GetHashCode();
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