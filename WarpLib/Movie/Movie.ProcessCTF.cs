using System;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class Movie
{
    public virtual void ProcessCTF(Image originalStack, ProcessingOptionsMovieCTF options)
    {
        IsProcessing = true;
        Image average = null;
        var totalTimer = System.Diagnostics.Stopwatch.StartNew();
        var timer = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            Directory.CreateDirectory(PowerSpectrumDir);
            Image input = originalStack;
            if (options.UseMovieSum)
                input = average = File.Exists(AveragePath) ? Image.FromFile(AveragePath) : originalStack.AsReducedAlongZ();
            int window = options.Window;
            int2 dims = new int2(input.Dims);
            if (options.GridDims.Elements() == 0)
            {
                float dose = (float)(options.DosePerAngstromFrame < 0 ? -options.DosePerAngstromFrame : options.DosePerAngstromFrame * input.Dims.Z);
                float extent = Math.Min(dims.X, dims.Y) * (float)options.BinnedPixelSizeMean;
                int side = Math.Max(2, (int)MathF.Round(5 * Math.Min(1, dose / 30) * extent / 4000));
                options.GridDims = new int3(Math.Max(1, side * dims.X / Math.Min(dims.X, dims.Y)),
                    Math.Max(1, side * dims.Y / Math.Min(dims.X, dims.Y)), Math.Max(1, (int)Math.Ceiling(dose)));
            }
            int groups = Math.Clamp(options.GridDims.Z, 1, input.Dims.Z);
            double inputSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            var extraction = CtfPowerSpectrum.Extract(input, options, groups);
            double extractionSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            originalStack.FreeDevice();
            var records = extraction.Observations.ToArray();
            int3 defocusDims = new int3(Math.Clamp(options.GridDims.X, 1, extraction.PositionGrid.X),
                Math.Clamp(options.GridDims.Y, 1, extraction.PositionGrid.Y), groups);
            int3 phaseDims = new int3(1, 1, options.DoPhase ? groups : 1);
            float3[] positions = records.Select(r => r.Position).ToArray();
            double[][] dw = CtfFitGeometry.GridWeights(defocusDims, positions), pw = CtfFitGeometry.GridWeights(phaseDims, positions);
            var geometry = records.Select((r, i) => new CtfFitGeometry(dw[i], pw[i])).ToArray();
            var seed = CtfFitEngine.Initialize(records, new double[records.Length], options);
            double searchSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            int nd = (int)defocusDims.Elements(), np = (int)phaseDims.Elements();
            double[] initial = new double[nd + 3 + np];
            Array.Fill(initial, seed.Defocus, 0, nd);
            Array.Fill(initial, seed.Phase, nd + 2, np);
            // First fit a shared CTF; only then release the spatial/temporal grid.
            var sharedGeometry = records.Select(_ => new CtfFitGeometry(new[] { 1.0 }, new[] { 1.0 })).ToArray();
            var shared = CtfFitEngine.Refine(records, sharedGeometry, new[] { seed.Defocus, 0.0, 0.0, seed.Phase, 0.0 }, options).Parameters;
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
            var diagnostic = CtfFitDiagnostics.Create(records, geometry, fit, new int[records.Length],
                new[] { CTF }, CTF, extraction.FourierSize, window, new[] { extraction.Display }).Global;
            PS1D = diagnostic.Spectrum; SimulatedBackground = diagnostic.Background; SimulatedScale = diagnostic.Envelope;
            CTFResolutionEstimate = diagnostic.Resolution;
            double diagnosticSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            using (var display = new Image(new[] { extraction.Display }, new int3(window, window / 2, 1))) display.WriteMRC(PowerSpectrumPath, true);
            Simulated1D = GetSimulated1D();
            OptionsCTF = options;
            SaveMeta();
            Console.WriteLine($"CTF: input {inputSeconds:F2}s, spectra {extractionSeconds:F2}s, global search {searchSeconds:F2}s, refinement {refinementSeconds:F2}s, diagnostics {diagnosticSeconds:F2}s, output {timer.Elapsed.TotalSeconds:F2}s, total {totalTimer.Elapsed.TotalSeconds:F2}s");
        }
        finally { average?.Dispose(); IsProcessing = false; }
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
    [WarpSerializable] public bool UseMovieSum { get; set; }
    [WarpSerializable] public decimal ZMin { get; set; }
    [WarpSerializable] public decimal ZMax { get; set; }
    [WarpSerializable] public int3 GridDims { get; set; }
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
               UseMovieSum == other.UseMovieSum &&
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