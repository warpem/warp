using System;
using System.Collections.Generic;
using System.Linq;
using ZLinq;

using Warp.Tools;
using Sample = Warp.Tools.CtfSpectrumFit.Sample;

namespace Tests;

/// <summary>Profiled power-spectrum fit. Frequencies are in Å⁻¹, defocus/astigmatism in µm,
/// phase in radians. The data and model use the same Fourier-bin moments.</summary>
public sealed class CtfCpuSpectrum
{
    public readonly record struct Evaluation(double Loss, double[] Gradient, double[] Background, double[] Envelope, double[] Model);
    public readonly Sample[] Samples;
    public readonly double PowerScale;
    readonly double[,] basis;
    readonly int[] firstBasis, lastBasis;
    readonly double[] data, weight, baseWeight, backgroundResidual;
    readonly int knots;
    readonly double kDefocus, kCs, amplitudePhase;
    readonly double[,] backgroundGram;
    readonly double[] backgroundRhs;
    readonly object backgroundSync = new();
    volatile bool backgroundDirty = true;
    const double Ridge = 1e-7;

    public static double Wavelength(double voltageKV) => 12.2643247 / Math.Sqrt(voltageKV * 1000 * (1 + voltageKV * 1000 * .978466e-6));

    public CtfCpuSpectrum(Sample[] samples, double voltageKV, double csMM, double amplitude, CtfCpuSpectrum basisSource = null)
    {
        if (samples == null || samples.Length < 16 || !double.IsFinite(voltageKV) || !(voltageKV > 0) || !double.IsFinite(csMM) || csMM < 0 || !double.IsFinite(amplitude) || amplitude < 0 || amplitude >= 1)
            throw new ArgumentException("Insufficient spectrum samples or invalid microscope parameters.");
        if (samples.Any(s => !double.IsFinite(s.Power) || s.Power < 0 || !(s.Count > 0) || !double.IsFinite(s.Count) || !double.IsFinite(s.Q2) || !(s.Q2 > 0) || !double.IsFinite(s.Q4) || !double.IsFinite(s.AstigX) || !double.IsFinite(s.AstigY)))
            throw new ArgumentException("Invalid power-spectrum samples.");
        Samples = samples;
        double lambda = Wavelength(voltageKV);
        kDefocus = Math.PI * lambda * 1e4;
        kCs = -.5 * Math.PI * csMM * 1e7 * lambda * lambda * lambda;
        amplitudePhase = Math.Asin(amplitude);
        double lo = Math.Sqrt(samples.Min(s => s.Q2)), hi = Math.Sqrt(samples.Max(s => s.Q2));
        if (!(hi > lo) || !samples.Any(s => s.Power > 0)) throw new ArgumentException("The spectrum has no frequency range or no power.");
        // A smooth nuisance basis, shared across angular sectors, describes broadband power.
        int intervals = Math.Max(1, (int)Math.Round((hi - lo) / .025));
        knots = intervals + 3;
        bool reuse = basisSource != null && basisSource.knots == knots && basisSource.Samples.Length == samples.Length;
        if (reuse) for (int i = 0; i < samples.Length; i++) reuse &= samples[i].Q2 == basisSource.Samples[i].Q2;
        if (reuse)
        {
            basis = basisSource.basis; firstBasis = basisSource.firstBasis; lastBasis = basisSource.lastBasis;
        }
        else
        {
            basis = new double[samples.Length, knots];
            double[] edges = new double[knots + 4];
            for (int i = 0; i < edges.Length; i++) edges[i] = lo + (hi - lo) * Math.Clamp(i - 3, 0, intervals) / intervals;
            for (int i = 0; i < samples.Length; i++)
            {
                double q = Math.Sqrt(samples[i].Q2);
                if (q >= hi) { basis[i, knots - 1] = 1; continue; }
                double[] b = new double[knots + 3];
                for (int j = 0; j < b.Length; j++) b[j] = q >= edges[j] && q < edges[j + 1] ? 1 : 0;
                for (int degree = 1; degree <= 3; degree++)
                    for (int j = 0; j < b.Length - degree; j++)
                        b[j] = (edges[j + degree] > edges[j] ? (q - edges[j]) / (edges[j + degree] - edges[j]) * b[j] : 0)
                             + (edges[j + degree + 1] > edges[j + 1] ? (edges[j + degree + 1] - q) / (edges[j + degree + 1] - edges[j + 1]) * b[j + 1] : 0);
                for (int j = 0; j < knots; j++) basis[i, j] = b[j];
            }
            firstBasis = new int[samples.Length]; lastBasis = new int[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                int first = 0; while (first < knots - 1 && basis[i, first] == 0) first++;
                firstBasis[i] = first; lastBasis[i] = Math.Min(knots, first + 4);
            }
        }
        double scale = PowerScale = Math.Max(1e-30, samples.Sum(s => s.Power * s.Count) / samples.Sum(s => s.Count));
        data = samples.Select(s => s.Power / scale).ToArray();
        // Smooth total power, rather than the noisy periodogram, supplies fixed variance weights.
        var gram = new double[knots, knots]; var rhs = new double[knots];
        for (int i = 0; i < samples.Length; i++)
            for (int j = firstBasis[i]; j < lastBasis[i]; j++)
            {
                rhs[j] += basis[i, j] * samples[i].Count * data[i];
                for (int k = firstBasis[i]; k <= j; k++) gram[j, k] += basis[i, j] * basis[i, k] * samples[i].Count;
            }
        CompleteGram(gram);
        double[] smooth = Solve(gram, rhs);
        weight = new double[samples.Length]; baseWeight = new double[samples.Length]; backgroundResidual = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            double p = 0; for (int j = firstBasis[i]; j < lastBasis[i]; j++) p += basis[i, j] * smooth[j];
            weight[i] = baseWeight[i] = samples[i].Count / Math.Pow(Math.Max(.02, p), 2);
        }
        backgroundGram = new double[knots, knots]; backgroundRhs = new double[knots];
        backgroundDirty = true;
    }

    internal void SetGpuWeights(double[] source, int offset)
    {
        Array.Copy(source, offset, weight, 0, weight.Length);
        backgroundDirty = true;
    }

    // Cache the background-only projection once per weight update.
    void EnsureBackgroundProjection()
    {
        if (!backgroundDirty) return;
        lock (backgroundSync)
        {
            if (!backgroundDirty) return;
            UpdateBackgroundProjection();
            backgroundDirty = false;
        }
    }

    void UpdateBackgroundProjection()
    {
        Array.Clear(backgroundGram); Array.Clear(backgroundRhs);
        for (int i = 0; i < Samples.Length; i++)
            for (int j = firstBasis[i]; j < lastBasis[i]; j++)
            {
                backgroundRhs[j] += weight[i] * basis[i, j] * data[i];
                for (int k = firstBasis[i]; k <= j; k++) backgroundGram[j, k] += weight[i] * basis[i, j] * basis[i, k];
            }
        CompleteGram(backgroundGram);
        double[] bg = Solve(backgroundGram, backgroundRhs);
        for (int i = 0; i < Samples.Length; i++)
        {
            backgroundResidual[i] = data[i];
            for (int j = firstBasis[i]; j < lastBasis[i]; j++) backgroundResidual[i] -= basis[i, j] * bg[j];
        }
    }

    /// <summary>Student-t IRLS update (8 degrees of freedom) between optimization passes. Weights remain fixed within a pass,
    /// preserving the analytic variable-projection derivative, including at active constraints.</summary>
    public double Reweight(double defocus, double astigX, double astigY, double phase)
    {
        var e = Evaluate(defocus, astigX, astigY, phase, true);
        double change = 0;
        for (int i = 0; i < Samples.Length; i++)
        {
            double residual = e.Background[i] + e.Envelope[i] * e.Model[i] - data[i];
            double factor = Math.Min(1, 9 / (8 + residual * residual * baseWeight[i]));
            change = Math.Max(change, Math.Abs(factor - weight[i] / baseWeight[i]));
            weight[i] = baseWeight[i] * factor;
        }
        backgroundDirty = true;
        return change;
    }

    public double QuickScore(double defocus, double phase = 0)
    {
        EnsureBackgroundProjection();
        double c = 0, p = 0;
        for (int i = 0; i < Samples.Length; i++)
        {
            Sample s = Samples[i];
            double m = -.5 * Math.Cos(2 * (kDefocus * s.Q2 * defocus + kCs * s.Q4 + amplitudePhase + phase));
            c += weight[i] * backgroundResidual[i] * m;
            p += weight[i] * m * m;
        }
        return c / Math.Sqrt(Math.Max(1e-30, p));
    }

    public Evaluation Evaluate(double defocus, double astigX, double astigY, double phase, bool details = false)
    {
        EnsureBackgroundProjection();
        int n = Samples.Length, size = 2 * knots;
        double[] model = new double[n], derivative = new double[n];
        var gram = new double[size, size]; var rhs = new double[size];
        for (int j = 0; j < knots; j++)
        {
            rhs[j] = backgroundRhs[j];
            for (int k = 0; k <= j; k++) gram[j, k] = backgroundGram[j, k] - (j == k ? Ridge : 0);
        }
        for (int i = 0; i < n; i++)
        {
            Sample s = Samples[i];
            double gamma = kDefocus * (s.Q2 * defocus + s.AstigX * astigX + s.AstigY * astigY) + kCs * s.Q4 + amplitudePhase + phase;
            model[i] = .5 - .5 * Math.Cos(2 * gamma); derivative[i] = Math.Sin(2 * gamma);
            for (int j = firstBasis[i]; j < lastBasis[i]; j++)
            {
                if (basis[i, j] == 0) continue;
                double v = weight[i] * basis[i, j] * model[i];
                rhs[j + knots] += v * data[i];
                for (int k = firstBasis[i]; k < lastBasis[i]; k++) gram[j + knots, k] += v * basis[i, k];
                for (int k = firstBasis[i]; k <= j; k++) gram[j + knots, k + knots] += v * basis[i, k] * model[i];
            }
        }
        CompleteGram(gram);
        double[] coefficients = NonnegativeEnvelope(gram, rhs, knots);
        double loss = 0; double[] gradient = new double[4];
        double[] bg = details ? new double[n] : null, env = details ? new double[n] : null;
        for (int i = 0; i < n; i++)
        {
            double a = 0, e = 0;
            for (int j = firstBasis[i]; j < lastBasis[i]; j++) { a += basis[i, j] * coefficients[j]; e += basis[i, j] * coefficients[j + knots]; }
            double residual = a + e * model[i] - data[i];
            loss += .5 * weight[i] * residual * residual;
            // Variable projection: derivatives of the optimal nuisance coefficients cancel.
            double d = weight[i] * residual * e * derivative[i];
            Sample s = Samples[i];
            gradient[0] += d * kDefocus * s.Q2;
            gradient[1] += d * kDefocus * s.AstigX;
            gradient[2] += d * kDefocus * s.AstigY;
            gradient[3] += d;
            if (details) { bg[i] = a; env[i] = e; }
        }
        loss += .5 * Ridge * coefficients.Sum(c => c * c);
        return new Evaluation(loss, gradient, bg, env, details ? model : null);
    }

    static void CompleteGram(double[,] a)
    {
        for (int j = 0; j < a.GetLength(0); j++)
        {
            a[j, j] += Ridge;
            for (int k = 0; k < j; k++) a[k, j] = a[j, k];
        }
    }

    // Background coefficients are free; the power envelope is nonnegative.
    // Active-set bounded least squares prevents an anticorrelated CTF from fitting by flipping the envelope.
    static double[] NonnegativeEnvelope(double[,] a, double[] b, int firstBounded)
    {
        int n = b.Length;
        double[] full = Solve(a, b);
        bool feasible = true;
        for (int i = firstBounded; i < n; i++) feasible &= full[i] >= 0;
        if (feasible) return full;
        bool[] active = Enumerable.Range(0, n).Select(i => i < firstBounded).ToArray();
        double[] x = SolveSubset(a, b, active);
        for (int iteration = 0; iteration < 8 * n * n; iteration++)
        {
            int enter = -1; double best = 1e-9;
            for (int i = firstBounded; i < n; i++) if (!active[i])
                {
                    double w = b[i]; for (int j = 0; j < n; j++) w -= a[i, j] * x[j];
                    if (w > best) { best = w; enter = i; }
                }
            if (enter < 0) return x;
            active[enter] = true;
            for (int inner = 0; inner <= n; inner++)
            {
                double[] z = SolveSubset(a, b, active);
                double alpha = 1;
                for (int i = firstBounded; i < n; i++) if (active[i] && z[i] <= 0)
                        alpha = Math.Min(alpha, x[i] / Math.Max(1e-30, x[i] - z[i]));
                if (alpha == 1) { x = z; break; }
                for (int i = 0; i < n; i++) x[i] += alpha * (z[i] - x[i]);
                for (int i = firstBounded; i < n; i++) if (active[i] && x[i] <= 1e-12) { active[i] = false; x[i] = 0; }
            }
        }
        throw new InvalidOperationException("CTF nuisance fit did not converge.");
    }

    static double[] SolveSubset(double[,] a, double[] b, bool[] active)
    {
        int[] ids = Enumerable.Range(0, b.Length).Where(i => active[i]).ToArray();
        var sub = new double[ids.Length, ids.Length]; var rhs = new double[ids.Length];
        for (int i = 0; i < ids.Length; i++) { rhs[i] = b[ids[i]]; for (int j = 0; j < ids.Length; j++) sub[i, j] = a[ids[i], ids[j]]; }
        double[] v = Solve(sub, rhs), result = new double[b.Length];
        for (int i = 0; i < ids.Length; i++) result[ids[i]] = v[i];
        return result;
    }

    static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length; var l = new double[n, n];
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++)
            {
                double sum = a[i, j]; for (int k = 0; k < j; k++) sum -= l[i, k] * l[j, k];
                l[i, j] = i == j ? Math.Sqrt(Math.Max(1e-20, sum)) : sum / l[j, j];
            }
        var x = (double[])b.Clone();
        for (int i = 0; i < n; i++) { for (int j = 0; j < i; j++) x[i] -= l[i, j] * x[j]; x[i] /= l[i, i]; }
        for (int i = n - 1; i >= 0; i--) { for (int j = i + 1; j < n; j++) x[i] -= l[j, i] * x[j]; x[i] /= l[i, i]; }
        return x;
    }
}
