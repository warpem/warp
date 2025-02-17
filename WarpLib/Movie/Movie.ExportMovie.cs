using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp;

public partial class Movie
{
    public void ExportMovie(Image originalStack, ProcessingOptionsMovieExport options)
    {
        Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

        IsProcessing = true;

        #region Make sure all directories are there

        Directory.CreateDirectory(AverageDir);

        if (options.DoDenoise)
        {
            Directory.CreateDirectory(AverageOddDir);
            Directory.CreateDirectory(AverageEvenDir);
        }

        if (options.DoDenoiseDeconv)
        {
            Directory.CreateDirectory(DenoiseTrainingDirOdd);
            Directory.CreateDirectory(DenoiseTrainingDirEven);
            //Directory.CreateDirectory(DenoiseTrainingDirCTF);
        }

        if (options.DoStack)
            Directory.CreateDirectory(ShiftedStackDir);
        if (options.DoDeconv)
            Directory.CreateDirectory(DeconvolvedDir);

        #endregion

        #region Helper variables

        int3 Dims = originalStack.Dims;
        int FirstFrame = Math.Max(0, Math.Min(Dims.Z - 1, options.SkipFirstN));
        int LastFrameExclusive = Math.Min(Dims.Z, Dims.Z - options.SkipLastN);
        Dims.Z = LastFrameExclusive - FirstFrame;
        bool CanDenoise = Dims.Z > 1 && options.DoDenoise;
        bool CanDenoiseDeconv = Dims.Z > 1 && options.DoDenoiseDeconv;
        float DenoisingAngPix = Math.Max(NoiseNet2DTorch.PixelSize, (float)options.BinnedPixelSizeMean); // Denoising always done at least at 5 A/px
        int2 DimsDenoise = new int2(new float2(Dims.X, Dims.Y) * (float)options.BinnedPixelSizeMean / DenoisingAngPix + 1) / 2 * 2;

        Task WriteDeconvAsync = null;
        Task WriteStackAsync = null;

        #endregion

        var Timer1 = OutputTimers[1].Start();

        #region Prepare spectral coordinates

        float PixelSize = (float)options.BinnedPixelSizeMean;
        if (ExportMovieCTFCoords == null || ExportMovieCTFCoords.Dims.Slice() != Dims.Slice())
        {
            if (ExportMovieCTFCoords != null)
                ExportMovieCTFCoords.Dispose();
            ExportMovieCTFCoords = CTF.GetCTFCoordsParallel(new int2(Dims), new int2(Dims));
        }

        {
            if (options.DoDeconv)
            {
                if (ExportMovieWiener == null || ExportMovieWiener.Dims.Slice() != Dims.Slice())
                {
                    if (ExportMovieWiener != null)
                        ExportMovieWiener.Dispose();

                    float2[] CTFCoordsData = new float2[Dims.Slice().ElementsFFT()];
                    Helper.ForEachElementFTParallel(new int2(Dims), (x, y, xx, yy) =>
                    {
                        float xs = xx / (float)Dims.X;
                        float ys = yy / (float)Dims.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);
                        float CurrentPixelSize = PixelSize;

                        CTFCoordsData[y * (Dims.X / 2 + 1) + x] = new float2(r / CurrentPixelSize, angle);
                    });

                    float[] CTF2D = CTF.Get2DFromScaledCoords(CTFCoordsData, false);
                    float HighPassNyquist = PixelSize * 2 / 100;
                    float Strength = (float)Math.Pow(10, 3 * (double)options.DeconvolutionStrength);
                    float Falloff = (float)options.DeconvolutionFalloff * 100 / PixelSize;

                    Helper.ForEachElementFT(new int2(Dims), (x, y, xx, yy) =>
                    {
                        float xs = xx / (float)Dims.X * 2;
                        float ys = yy / (float)Dims.Y * 2;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);

                        float HighPass = 1 - (float)Math.Cos(Math.Min(1, r / HighPassNyquist) * Math.PI);
                        float SNR = (float)Math.Exp(-r * Falloff) * Strength * HighPass;
                        float CTFVal = CTF2D[y * (Dims.X / 2 + 1) + x];
                        CTF2D[y * (Dims.X / 2 + 1) + x] = CTFVal / (CTFVal * CTFVal + 1 / SNR);
                    });

                    ExportMovieWiener = new Image(CTF2D, Dims.Slice(), true);
                }
            }
        }

        #endregion

        OutputTimers[1].Finish(Timer1);

        Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

        #region Timer 0

        var Timer0 = OutputTimers[0].Start();

        Image AverageFT = new Image(Dims.Slice(), true, true);
        Image AverageOddFT = (CanDenoise || CanDenoiseDeconv) ? new Image(Dims.Slice(), true, true) : null;
        Image AverageEvenFT = (CanDenoise || CanDenoiseDeconv) ? new Image(Dims.Slice(), true, true) : null;

        #region Warp, get FTs of all relevant frames, apply spectral filter, and add to average

        var Timer7 = OutputTimers[7].Start();

        int PlanForw = FFTPlanCache.GetFFTPlan(Dims.Slice(), 1); // GPU.CreateFFTPlan(Dims.Slice(), 1);

        IntPtr[] TempArray = { GPU.MallocArray(new int2(Dims * 1)) };
        Image[] Frame = { new Image(IntPtr.Zero, Dims.Slice()) };
        Image[] FrameFT = { new Image(IntPtr.Zero, Dims.Slice(), true, true) };
        Image[] FramePrefiltered = { new Image(IntPtr.Zero, Dims.Slice()) };
        Image[] DoseImage = { new Image(IntPtr.Zero, Dims.Slice(), true) };
        Image[] PS = { new Image(Dims.Slice(), true) };

        float StepZ = 1f / Math.Max(originalStack.Dims.Z - 1, 1);

        int DeviceID = GPU.GetDevice();

        OutputTimers[7].Finish(Timer7);

        Helper.ForCPU(0, Dims.Z, 1, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
        {
            #region Timer 6

            var Timer6 = OutputTimers[6].Start();

            int2 DimsWarp = new int2(16);
            float3[] InterpPoints = new float3[DimsWarp.Elements()];
            for (int y = 0; y < DimsWarp.Y; y++)
            for (int x = 0; x < DimsWarp.X; x++)
                InterpPoints[y * DimsWarp.X + x] = new float3((float)x / (DimsWarp.X - 1), (float)y / (DimsWarp.Y - 1), (z + FirstFrame) * StepZ);

            float2[] WarpXY = GetShiftFromPyramid(InterpPoints);
            float[] WarpX = WarpXY.Select(v => v.X / (float)options.BinnedPixelSizeMean).ToArray();
            float[] WarpY = WarpXY.Select(v => v.Y / (float)options.BinnedPixelSizeMean).ToArray();

            OutputTimers[6].Finish(Timer6);

            #endregion

            #region Timer 2

            var Timer2 = OutputTimers[2].Start();

            GPU.CopyDeviceToDevice(originalStack.GetDeviceSlice(z + FirstFrame, Intent.Read),
                FramePrefiltered[threadID].GetDevice(Intent.Write),
                Dims.ElementsSlice());

            GPU.PrefilterForCubic(FramePrefiltered[threadID].GetDevice(Intent.ReadWrite), Dims.Slice());

            GPU.WarpImage(FramePrefiltered[threadID].GetDevice(Intent.Read),
                Frame[threadID].GetDevice(Intent.Write),
                new int2(Dims),
                WarpX,
                WarpY,
                DimsWarp,
                TempArray[threadID]);

            PS[threadID].Fill(1f);
            int nframe = z + FirstFrame;

            OutputTimers[2].Finish(Timer2);

            #endregion

            #region Timer 5

            var Timer5 = OutputTimers[5].Start();

            // Apply dose weighting.
            {
                float DosePerFrame = (float)options.DosePerAngstromFrame;
                if (options.DosePerAngstromFrame < 0)
                    DosePerFrame = -(float)options.DosePerAngstromFrame / originalStack.Dims.Z;

                CTF CTFBfac = new CTF()
                {
                    PixelSize = (decimal)PixelSize,
                    Defocus = 0,
                    Amplitude = 1,
                    Cs = 0,
                    Cc = 0,
                    Bfactor = GridDoseBfacs.Values.Length <= 1 ? -(decimal)(DosePerFrame * (nframe + 0.5f) * 3) : (decimal)GridDoseBfacs.GetInterpolated(new float3(0.5f, 0.5f, (float)nframe / Math.Max(NFrames - 1, 1))),
                    Scale = GridDoseWeights.Values.Length <= 1 ? 1 : (decimal)GridDoseWeights.GetInterpolated(new float3(0.5f, 0.5f, (float)nframe / Math.Max(NFrames - 1, 1)))
                };
                GPU.CreateCTF(DoseImage[threadID].GetDevice(Intent.Write),
                    ExportMovieCTFCoords.GetDevice(Intent.Read),
                    IntPtr.Zero,
                    (uint)ExportMovieCTFCoords.ElementsSliceComplex,
                    new[] { CTFBfac.ToStruct() },
                    false,
                    1);

                PS[threadID].Multiply(DoseImage[threadID]);
                //DoseImage.WriteMRC("dose.mrc");
            }
            //PS.WriteMRC("ps.mrc");

            OutputTimers[5].Finish(Timer5);

            #endregion

            #region Timer 3

            lock(Frame)
            {
                var Timer3 = OutputTimers[3].Start();

                GPU.FFT(Frame[threadID].GetDevice(Intent.Read),
                    FrameFT[threadID].GetDevice(Intent.Write),
                    Dims.Slice(),
                    1,
                    PlanForw);

                GPU.MultiplyComplexSlicesByScalar(FrameFT[threadID].GetDevice(Intent.Read),
                    PS[threadID].GetDevice(Intent.Read),
                    FrameFT[threadID].GetDevice(Intent.Write),
                    PS[threadID].ElementsSliceReal,
                    1);

                AverageFT.Add(FrameFT[threadID]);

                if (CanDenoise || CanDenoiseDeconv)
                    (z % 2 != 0 ? AverageOddFT : AverageEvenFT).Add(FrameFT[threadID]); // Odd/even frame averages for denoising training data

                OutputTimers[3].Finish(Timer3);
            }

            #endregion
        }, null);

        originalStack.FreeDevice();

        //GPU.DestroyFFTPlan(PlanForw);

        //Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

        #endregion

        OutputTimers[0].Finish(Timer0);

        #endregion

        foreach (var array in TempArray)
            GPU.FreeArray(array);
        foreach (var image in Frame)
            image.Dispose();
        foreach (var image in FrameFT)
            image.Dispose();
        foreach (var image in FramePrefiltered)
            image.Dispose();
        foreach (var image in DoseImage)
            image.Dispose();
        foreach (var image in PS)
            image.Dispose();

        //AverageFT.Divide(Weights);
        //AverageFT.WriteMRC("averageft.mrc");
        //Weights.WriteMRC("weights.mrc");
        //Weights.Dispose();

        #region Timer 4

        var Timer4 = OutputTimers[4].Start();

        Image Average = null, AverageOdd = null, AverageEven = null;
        if (options.DoAverage)
        {
            Average = AverageFT.AsIFFT(false, 0, true, true);

            // Previous division by weight sum brought values to stack average, multiply by number of frame to go back to sum
            Average.Multiply(Dims.Z);
            Average.FreeDevice();

            Image AverageDeconvOdd = null, AverageDeconvEven = null;

            if (CanDenoiseDeconv)
            {
                Image AverageOddFTPadded;
                Image AverageEvenFTPadded;
                if (DimsDenoise != new int2(AverageOddFT.Dims))
                {
                    AverageOddFTPadded = AverageOddFT.AsPadded(DimsDenoise);
                    AverageEvenFTPadded = AverageEvenFT.AsPadded(DimsDenoise);
                }
                else
                {
                    AverageOddFTPadded = AverageOddFT.GetCopyGPU();
                    AverageEvenFTPadded = AverageEvenFT.GetCopyGPU();
                }

                AverageDeconvOdd = AverageOddFTPadded.AsIFFT(false, 0, true).AndDisposeParent();
                AverageDeconvEven = AverageEvenFTPadded.AsIFFT(false, 0, true).AndDisposeParent();

                if (OptionsCTF != null)
                {
                    Image SoftMask = new Image(new int3(DimsDenoise.X * 2, DimsDenoise.Y * 2, 1));
                    SoftMask.TransformValues((x, y, z, v) =>
                    {
                        float xx = (float)Math.Max(0, Math.Max(DimsDenoise.X / 2 - x, x - DimsDenoise.X * 3 / 2)) / (DimsDenoise.X / 2);
                        float yy = (float)Math.Max(0, Math.Max(DimsDenoise.Y / 2 - y, y - DimsDenoise.Y * 3 / 2)) / (DimsDenoise.Y / 2);

                        float r = Math.Min(1, (float)Math.Sqrt(xx * xx + yy * yy));
                        float w = ((float)Math.Cos(r * Math.PI) + 1) * 0.5f;

                        return w;
                    });

                    AverageDeconvOdd = AverageDeconvOdd.AsPadded(DimsDenoise - 2).AndDisposeParent();
                    AverageDeconvEven = AverageDeconvEven.AsPadded(DimsDenoise - 2).AndDisposeParent();

                    Image OddPadded = AverageDeconvOdd.AsPaddedClamped(DimsDenoise * 2).AndDisposeParent();
                    Image EvenPadded = AverageDeconvEven.AsPaddedClamped(DimsDenoise * 2).AndDisposeParent();

                    OddPadded.MultiplySlices(SoftMask);
                    EvenPadded.MultiplySlices(SoftMask);

                    Image OddPaddedFT = OddPadded.AsFFT().AndDisposeParent();
                    Image EvenPaddedFT = EvenPadded.AsFFT().AndDisposeParent();

                    SoftMask.Dispose();

                    CTF DeconvCTF = CTF.GetCopy();
                    DeconvCTF.PixelSize = (decimal)DenoisingAngPix;

                    float HighpassNyquist = DenoisingAngPix * 2 / 500f;

                    GPU.DeconvolveCTF(OddPaddedFT.GetDevice(Intent.Read),
                        OddPaddedFT.GetDevice(Intent.Write),
                        OddPaddedFT.Dims,
                        DeconvCTF.ToStruct(),
                        0.85f,
                        0.25f,
                        HighpassNyquist);
                    GPU.DeconvolveCTF(EvenPaddedFT.GetDevice(Intent.Read),
                        EvenPaddedFT.GetDevice(Intent.Write),
                        EvenPaddedFT.Dims,
                        DeconvCTF.ToStruct(),
                        0.85f,
                        0.25f,
                        HighpassNyquist);

                    OddPadded = OddPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();
                    EvenPadded = EvenPaddedFT.AsIFFT(false, 0, true).AndDisposeParent();

                    AverageDeconvOdd = OddPadded.AsPadded(DimsDenoise).AndDisposeParent();
                    AverageDeconvEven = EvenPadded.AsPadded(DimsDenoise).AndDisposeParent();

                    //AverageDeconvOdd.Multiply(1f / (3f / 4 * NFrames));
                    //AverageDeconvEven.Multiply(1f / (1f / 4 * NFrames));

                    //Image CTFCoords = CTF.GetCTFCoords(256, 256);
                    //CTF DeconvCTF = CTF.GetCopy();
                    //DeconvCTF.PixelSize = (decimal)DenoisingAngPix;

                    //Image CTFSimulated = new Image(new int3(256, 256, 1), true);
                    //GPU.CreateCTF(CTFSimulated.GetDevice(Intent.Write),
                    //              CTFCoords.GetDevice(Intent.Read),
                    //              IntPtr.Zero,
                    //              (uint)CTFCoords.ElementsComplex,
                    //              new[] { DeconvCTF.ToStruct() },
                    //              false,
                    //              1);

                    //CTFSimulated.WriteMRC(DenoiseTrainingCTFPath, true);
                    //CTFSimulated.Dispose();
                    //CTFCoords.Dispose();
                }
            }

            if (CanDenoise)
            {
                AverageOdd = AverageOddFT.AsIFFT(false, 0, true);
                AverageEven = AverageEvenFT.AsIFFT(false, 0, true);
            }

            AverageOddFT?.Dispose();
            AverageEvenFT?.Dispose();

            // Write average async to disk
            WriteAverageAsync = new Task(() =>
            {
                Average.WriteMRC16b(AveragePath, (float)options.BinnedPixelSizeMean, true);
                Average.Dispose();

                AverageOdd?.WriteMRC16b(AverageOddPath, (float)options.BinnedPixelSizeMean, true);
                AverageOdd?.Dispose();
                AverageEven?.WriteMRC16b(AverageEvenPath, (float)options.BinnedPixelSizeMean, true);
                AverageEven?.Dispose();

                AverageDeconvOdd?.WriteMRC16b(DenoiseTrainingOddPath, DenoisingAngPix, true);
                AverageDeconvOdd?.Dispose();
                AverageDeconvEven?.WriteMRC16b(DenoiseTrainingEvenPath, DenoisingAngPix, true);
                AverageDeconvEven?.Dispose();

                OnAverageChanged();
            });
            WriteAverageAsync.Start();
        }

        Image Deconvolved = null;
        if (options.DoDeconv)
        {
            AverageFT.Multiply(ExportMovieWiener);
            //ExportMovieWiener.Dispose();
            Deconvolved = AverageFT.AsIFFT(false, 0, true);
            Deconvolved.Multiply(Dims.Z); // It's only the average at this point, needs to be sum

            // Write deconv async to disk
            WriteDeconvAsync = new Task(() =>
            {
                Deconvolved.WriteMRC16b(DeconvolvedPath, (float)options.BinnedPixelSizeMean, true);
                Deconvolved.Dispose();
            });
            WriteDeconvAsync.Start();
        }

        AverageFT.Dispose();
        //ExportMovieCTFCoords.Dispose();

        //Debug.WriteLine(GPU.GetFreeMemory(GPU.GetDevice()));

        // Wait for all async IO to finish
        WriteStackAsync?.Wait();
        //WriteDeconvAsync?.Wait();
        WriteAverageAsync?.Wait();

        OutputTimers[4].Finish(Timer4);

        #endregion

        OptionsMovieExport = options;
        SaveMeta();

        IsProcessing = false;

        lock(OutputTimers)
        {
            if (OutputTimers[0].NItems > 1)
                foreach (var timer in OutputTimers)
                    Console.WriteLine(timer.Name + ": " + timer.GetAverageMilliseconds(100).ToString("F0"));
        }
    }

    [Serializable]
    public class ProcessingOptionsMovieExport : ProcessingOptionsBase
    {
        [WarpSerializable]
        public bool DoAverage { get; set; }
        [WarpSerializable]
        public bool DoStack { get; set; }
        [WarpSerializable]
        public bool DoDeconv { get; set; }
        [WarpSerializable]
        public bool DoDenoise { get; set; }
        [WarpSerializable]
        public bool DoDenoiseDeconv { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionStrength { get; set; }
        [WarpSerializable]
        public decimal DeconvolutionFalloff { get; set; }
        [WarpSerializable]
        public int StackGroupSize { get; set; }
        [WarpSerializable]
        public int SkipFirstN { get; set; }
        [WarpSerializable]
        public int SkipLastN { get; set; }
        [WarpSerializable]
        public decimal DosePerAngstromFrame { get; set; }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((ProcessingOptionsMovieExport)obj);
        }

        protected bool Equals(ProcessingOptionsMovieExport other)
        {
            return base.Equals(other) &&
                   DoAverage == other.DoAverage &&
                   DoStack == other.DoStack &&
                   DoDeconv == other.DoDeconv &&
                   DoDenoise == other.DoDenoise &&
                   DeconvolutionStrength == other.DeconvolutionStrength &&
                   DeconvolutionFalloff == other.DeconvolutionFalloff &&
                   StackGroupSize == other.StackGroupSize &&
                   SkipFirstN == other.SkipFirstN &&
                   SkipLastN == other.SkipLastN &&
                   DosePerAngstromFrame == other.DosePerAngstromFrame;
        }

        public static bool operator ==(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return Equals(left, right);
        }

        public static bool operator !=(ProcessingOptionsMovieExport left, ProcessingOptionsMovieExport right)
        {
            return !Equals(left, right);
        }
    }
}