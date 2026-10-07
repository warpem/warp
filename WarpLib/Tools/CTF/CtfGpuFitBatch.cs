using System;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

/// <summary>Resident GPU spectra, constrained nuisance solves, and analytic local CTF derivatives.
/// Frequency-sized GPU storage and bulk arithmetic are FP32; only small spline solves
/// and selectively retried ill-conditioned matrices use FP64.
/// The CPU optimizer transfers four local parameters and receives six scalars per patch.</summary>
public sealed class CtfGpuFitBatch : IDisposable
{
    IntPtr context;
    readonly CtfSpectrumFit[] spectra;
    readonly double[] output;
    public CtfGpuFitBatch(CtfSpectrumFit[] spectra)
    {
        if (spectra.Length == 0) throw new ArgumentException("No CTF spectra.");
        this.spectra = spectra;
        var first = spectra[0]; int samples = first.Samples.Length;
        foreach (var s in spectra)
            if (!first.HasSameGpuLayout(s)) throw new ArgumentException("A CTF GPU batch requires common frequency sampling and microscope parameters.");
        int count = checked(samples * spectra.Length);
        var data = new double[count]; var counts = new double[count]; var currentWeights = new double[count];
        for (int i = 0; i < spectra.Length; i++) spectra[i].CopyGpuData(data, counts, currentWeights, i * samples);
        CtfNative.Check(CtfNative.FitCreate(spectra.Length, samples, first.KnotCount, first.GpuMoments(), first.GpuBasis(), data, counts, currentWeights, out context), "create fitting batch");
        output = new double[spectra.Length * 6];
    }
    // Trial rows contain defocus and phase; output rows contain one score per patch.
    public double[] Search(double[] trials, double[] offsets)
    {
        ObjectDisposedException.ThrowIf(context == IntPtr.Zero, this);
        if (trials.Length == 0 || trials.Length % 2 != 0 || offsets.Length != spectra.Length)
            throw new ArgumentException("Invalid coarse CTF search dimensions.");
        var scores = new double[checked(trials.Length / 2 * spectra.Length)];
        CtfNative.Check(CtfNative.FitSearch(context, trials, offsets, trials.Length / 2, scores), "search defocus and phase");
        return scores;
    }
    /// <summary>Poses are interleaved defocus, astigmatism X/Y (µm), phase (radians).
    /// Returns one row of loss, four derivatives, and weight change per patch; reused on the next call.</summary>
    public double[] Evaluate(double[] poses, bool reweight = false)
    {
        ObjectDisposedException.ThrowIf(context == IntPtr.Zero, this);
        if (poses.Length != spectra.Length * 4) throw new ArgumentException("Invalid local CTF pose count.");
        CtfNative.Check(CtfNative.FitEvaluate(context, poses, reweight ? 1 : 0, output), "evaluate fitting batch");
        return output;
    }
    // Call after evaluating the accepted final pose: a failed line search can leave
    // the device holding a rejected trial. Export only the compact spline coefficients.
    public float[] ReadCoefficients()
    {
        ObjectDisposedException.ThrowIf(context == IntPtr.Zero, this);
        var coefficients = new float[checked(spectra.Length * spectra[0].KnotCount * 2)];
        CtfNative.Check(CtfNative.FitReadCoefficients(context, coefficients), "read nuisance coefficients");
        return coefficients;
    }
    // One final download makes subsequent fitting stages use the same robust weights.
    public void SynchronizeWeights()
    {
        ObjectDisposedException.ThrowIf(context == IntPtr.Zero, this);
        int samples = spectra[0].Samples.Length;
        var weights = new double[checked(samples * spectra.Length)];
        CtfNative.Check(CtfNative.FitReadWeights(context, weights), "read robust weights");
        Parallel.For(0, spectra.Length, i => spectra[i].SetGpuWeights(weights, i * samples));
    }
    public void Dispose()
    {
        if (context != IntPtr.Zero) { CtfNative.FitDestroy(context); context = IntPtr.Zero; }
        GC.SuppressFinalize(this);
    }
    ~CtfGpuFitBatch() { Dispose(); }
}
