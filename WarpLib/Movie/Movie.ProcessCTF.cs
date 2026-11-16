using System;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class Movie
{
    /// <param name="releaseStack">Free the stack's device copy afterwards (see <see cref="ProcessShift"/>).</param>
    public virtual void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options, bool releaseStack = true)
    {
        IsProcessing = true;
        Image grouped = null;
        var totalTimer = System.Diagnostics.Stopwatch.StartNew();
        var timer = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            Directory.CreateDirectory(PowerSpectrumDir);
            Image input = originalStack;
            var groups = ExposureGroups(originalStack.Dims.Z, options);
            if (groups.Length < originalStack.Dims.Z)
            {
                input = grouped = CtfExposureGroups.Average(originalStack, FrameWarps(originalStack.Dims.Z, (float)options.BinnedPixelSizeMean), groups);
                // Spectra come from the groups alone: the frames' device copy can go before the extractors allocate.
                if (releaseStack)
                    originalStack.FreeDevice();
            }
            int window = options.Window;
            int2 dims = new int2(input.Dims);
            if (options.GridDims.Elements() == 0)
                options.GridDims = AutoCtfGrid(dims, originalStack.Dims.Z, options.DosePerAngstromFrame, options.BinnedPixelSizeMean);
            double inputSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            // The fitting extractor stays alive so the full-spectrum diagnostics can reuse its frame sums.
            using var extractor = new CtfPowerSpectrum.Extractor(dims, options);
            var extraction = extractor.Extract(input);
            if (extraction.AperturePixels != window)
                Console.WriteLine($"CTF: enlarged extraction aperture from {window} to {extraction.AperturePixels} pixels to retain physical support at the requested defocus range.");
            double extractionSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            if (releaseStack && grouped == null)
                originalStack.FreeDevice();
            // The full-spectrum diagnostics resample the fitting extractor's frame sums, which nothing below changes. That
            // runs on its own thread and GPU stream while the fit is refined, whose kernels keep the device mostly idle
            // (one warp per patch); the spectra are the same as reading them afterwards. Only resampling may run there:
            // when the sums can't be shared, the frames would be read again, and host frames are uploaded on the default
            // stream, unordered with the side stream's kernels. That read waits for the refinement, as it always did.
            int device = GPU.GetDevice();
            CtfPowerSpectrum.Extractor diagnosticExtractor = null;
            var diagnosticRead = System.Threading.Tasks.Task.Run(() =>
            {
                GPU.SetDevice(device);
                diagnosticExtractor = new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
                if (!diagnosticExtractor.CanShareFrameSums(extractor))
                    return null;
                diagnosticExtractor.UseOwnStream();
                return diagnosticExtractor.Extract(input, framesSummedBy: extractor);
            });
            try
            {
                var records = extraction.Observations.ToArray();
                int3 defocusDims = new int3(Math.Clamp(options.GridDims.X, 1, extraction.PositionGrid.X),
                    Math.Clamp(options.GridDims.Y, 1, extraction.PositionGrid.Y), 1);
                int3 phaseDims = new int3(1);
                float3[] positions = records.Select(r => r.Position).ToArray();
                double[][] dw = CtfFitGeometry.GridWeights(defocusDims, positions), pw = CtfFitGeometry.GridWeights(phaseDims, positions);
                var apertureWeights = CtfFitGeometry.GridApertureWeights(defocusDims, positions,
                    new float2((float)extraction.AperturePixels / dims.X, (float)extraction.AperturePixels / dims.Y));
                var geometry = records.Select((r, i) => new CtfFitGeometry(dw[i], pw[i],
                    defocusWidthXWeights: apertureWeights.X[i], defocusWidthYWeights: apertureWeights.Y[i])).ToArray();
                var seed = CtfFitEngine.InitializeMovie(records, options);
                double searchSeconds = timer.Elapsed.TotalSeconds;
                timer.Restart();
                int nd = (int)defocusDims.Elements(), np = (int)phaseDims.Elements();
                double[] initial = new double[nd + 3 + np];
                Array.Fill(initial, seed.Defocus, 0, nd);
                Array.Fill(initial, seed.Phase, nd + 2, np);
                // First fit a shared CTF; only then release the spatial grid.
                var sharedGeometry = records.Select(_ => new CtfFitGeometry(new[] { 1.0 }, new[] { 1.0 })).ToArray();
                var sharedFit = CtfFitEngine.Refine(records, sharedGeometry, new[] { seed.Defocus, 0.0, 0.0, seed.Phase, 0.0 }, options);
                var shared = sharedFit.Parameters;
                Array.Fill(initial, shared[0], 0, nd); initial[nd] = shared[1]; initial[nd + 1] = shared[2];
                Array.Fill(initial, shared[3], nd + 2, np);
                initial[^1] = shared[^1];
                var fit = CtfFitEngine.Refine(records, geometry, initial, options);
                double refinementSeconds = timer.Elapsed.TotalSeconds;
                timer.Restart();
                var p = fit.Parameters;
                CTFSpecimenThicknessAngstrom = (decimal)(Math.Sqrt(p[^1])*1e4);
                GridCTFDefocus = new CubicGrid(defocusDims, p.Take(nd).Select(v => (float)v).ToArray());
                GridCTFPhase = new CubicGrid(phaseDims, p.Skip(nd + 2).Take(np).Select(v => (float)(v / Math.PI)).ToArray());
                CTF = CtfFitEngine.MakeCtf(options, p.Take(nd).Average(), p[nd], p[nd + 1], p.Skip(nd + 2).Take(np).Average());
                CtfPowerSpectrum.Extraction earlyRead = null;
                try { earlyRead = diagnosticRead.GetAwaiter().GetResult(); }
                catch (Exception ex)
                {
                    // For example out of device memory next to the fit batches: read the spectra the serial way instead.
                    Console.WriteLine($"CTF: the background full-spectrum read failed ({ex.Message}); reading the spectra after the fit instead.");
                    diagnosticExtractor?.Dispose();
                    diagnosticExtractor = null;
                }
                diagnosticExtractor ??= new CtfPowerSpectrum.Extractor(dims, options, fullSpectrum: true);
                var diagnosticInputs = new[] { new CtfFitDiagnostics.Input(input, geometry, 0, extractor, earlyRead) };
                var diagnostic = CtfFitDiagnostics.CreateFullSpectrum(diagnosticExtractor, diagnosticInputs, fit,
                    new[] { CTF }, CTF, window, new[] { extraction.Display }).Global;
                PS1D = diagnostic.Spectrum; SimulatedBackground = diagnostic.Background; SimulatedScale = diagnostic.Envelope;
                CTFResolutionEstimate = diagnostic.Resolution;
                CTFQuality = diagnostic.Quality;
                CTFSignal = CtfSignal.Score(records, CTF, (double)options.ZMin, (double)options.ZMax,
                    CtfSignal.SlabModulation(records, geometry, p, options.Voltage));
                double diagnosticSeconds = timer.Elapsed.TotalSeconds;
                timer.Restart();
                using (var display = new Image(new[] { extraction.Display }, new int3(window, window / 2, 1))) display.WriteMRC(PowerSpectrumPath, true);
                Simulated1D = GetSimulated1D();
                OptionsCTF = options;
                SaveMeta();
                Console.WriteLine($"CTF: input {inputSeconds:F2}s, spectra {extractionSeconds:F2}s, global search {searchSeconds:F2}s, refinement {refinementSeconds:F2}s ({sharedFit.Evaluations}+{fit.Evaluations} evaluations), diagnostics {diagnosticSeconds:F2}s, output {timer.Elapsed.TotalSeconds:F2}s, total {totalTimer.Elapsed.TotalSeconds:F2}s");
            }
            finally
            {
                // The diagnostic read uses the fitting extractor and the input, which are disposed next. A failure of
                // its own surfaces where it is awaited above.
                try { diagnosticRead.Wait(); } catch (AggregateException) { }
                diagnosticExtractor?.Dispose();
            }
        }
        finally { grouped?.Dispose(); IsProcessing = false; }
    }

    /// <summary>
    /// The aligned exposure groups CTF spectra come from (<see cref="CtfExposureGroups"/>). Aligning needs this movie's
    /// motion; without it, frames stay separate and the spectrum is the per-frame one.
    /// </summary>
    public (int Start, int End)[] ExposureGroups(int frames, ProcessingOptionsMovieCTF options)
    {
        double exposure = (double)(options.DosePerAngstromFrame < 0 ? -options.DosePerAngstromFrame : options.DosePerAngstromFrame * frames);
        if (OptionsMovement == null && frames > 1)
        {
            Console.WriteLine("CTF: no motion model to align frames with, so spectra come from individual frames.");
            exposure = 0;
        }
        return CtfExposureGroups.Partition(frames, exposure, (double)options.GroupExposure);
    }

    /// <summary>
    /// The automatic spatial defocus grid: about 5 nodes per 4000 Å at 30 e⁻/Å² or more, fewer at lower exposure.
    /// <paramref name="frames"/> is the movie's frame count, not the number of exposure groups the spectra come from.
    /// </summary>
    public static int2 AutoCtfGrid(int2 dims, int frames, decimal dosePerAngstromFrame, decimal pixelSize)
    {
        float dose = (float)(dosePerAngstromFrame < 0 ? -dosePerAngstromFrame : dosePerAngstromFrame * frames);
        float extent = Math.Min(dims.X, dims.Y) * (float)pixelSize;
        int side = Math.Max(2, (int)MathF.Round(5 * Math.Min(1, dose / 30) * extent / 4000));
        return new int2(Math.Max(1, side * dims.X / Math.Min(dims.X, dims.Y)), Math.Max(1, side * dims.Y / Math.Min(dims.X, dims.Y)));
    }

    /// <summary>Each frame's whole-image shift at the centre, in pixels, as the warp <see cref="ExportMovie"/> applies.</summary>
    public float2[] FrameWarps(int frames, float pixelSize)
    {
        float step = 1f / Math.Max(frames - 1, 1);
        var centres = new float3[frames];
        for (int z = 0; z < frames; z++) centres[z] = new float3(0.5f, 0.5f, z * step);
        float2[] shifts = GetShiftFromPyramid(centres);
        for (int z = 0; z < frames; z++) shifts[z] /= pixelSize;
        return shifts;
    }
}

[Serializable]
public class ProcessingOptionsMovieCTF : ProcessingOptionsBase
{
    [WarpSerializable] public int Window { get; set; }
    [WarpSerializable] public decimal RangeMin { get; set; }
    [WarpSerializable] public decimal RangeMax { get; set; }
    [WarpSerializable] public int Voltage { get; set; }
    [WarpSerializable] public decimal Cs { get; set; }
    [WarpSerializable] public decimal Cc { get; set; }
    [WarpSerializable] public decimal Amplitude { get; set; }
    [WarpSerializable] public bool DoPhase { get; set; }
    [WarpSerializable] public bool FitThickness { get; set; }
    /// <summary>Exposure per aligned frame group the spectra come from, in e⁻/Å² (<see cref="CtfExposureGroups"/>).</summary>
    [WarpSerializable] public decimal GroupExposure { get; set; } = CtfExposureGroups.DefaultExposure;
    [WarpSerializable] public decimal ZMin { get; set; }
    [WarpSerializable] public decimal ZMax { get; set; }
    [WarpSerializable] public int2 GridDims { get; set; }
    [WarpSerializable] public decimal DosePerAngstromFrame { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsMovieCTF)obj);
    }

    protected bool Equals(ProcessingOptionsMovieCTF other)
    {
        return base.Equals(other) &&
               Window == other.Window &&
               RangeMin == other.RangeMin &&
               RangeMax == other.RangeMax &&
               Voltage == other.Voltage &&
               Cs == other.Cs &&
               Cc == other.Cc &&
               Amplitude == other.Amplitude &&
               DoPhase == other.DoPhase &&
               FitThickness == other.FitThickness &&
               GroupExposure == other.GroupExposure &&
               ZMin == other.ZMin &&
               ZMax == other.ZMax &&
               GridDims == other.GridDims &&
               DosePerAngstromFrame == other.DosePerAngstromFrame;
    }

    public override int GetHashCode()
    {
        return base.GetHashCode();
    }

    public static bool operator ==(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsMovieCTF left, ProcessingOptionsMovieCTF right)
    {
        return !Equals(left, right);
    }
}
