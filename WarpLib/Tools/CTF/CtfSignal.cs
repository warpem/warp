using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

/// <summary>
/// How clearly a movie's spectrum shows its fitted CTF, as a robust z-score: the best correlation with the spectrum
/// near the fit, against the best correlations near decoy defoci. The fit always finds some defocus, even in a
/// spectrum without Thon rings (an empty hole, say), and its resolution estimate then describes noise. Such fits score
/// about 0–3; spectra with rings score far higher.
/// </summary>
/// <remarks>
/// The patches' spectra are pooled, and their log power is flattened by a quartic in frequency per angular sector,
/// independently of the CTF (the fit's own background is fitted together with the CTF, so it can bend noise towards
/// it). Fit and decoys are treated alike, as local maxima: each starts from its defocus (decoys more than
/// <see cref="Exclusion"/> µm from the fit's, with the best of a coarse grid of astigmatisms) and climbs in defocus and
/// astigmatism. Fits to noise are the best of many such maxima, so they score like the best decoys rather than
/// standing out; scoring decoys without the same freedom lets such fits reach 5 or more. When the fit has a specimen
/// thickness, fit and decoys alike carry its slab modulation, which damps the rings and flips their contrast past its
/// first zero (inside the fitting band for ice thicker than about 1000 Å).
/// </remarks>
public static class CtfSignal
{
    /// <summary>Changes whenever scores would change, so stored scores can be recomputed.</summary>
    public const int Revision = 1;

    /// <summary>
    /// The score below which a movie has no usable CTF signal by default. On 352 movies of one session, the 264 with
    /// Thon rings scored 6.8 or more (median 16.5), and the 88 from holes without any scored at most 3.7 (median 1.1).
    /// </summary>
    public const double DefaultThreshold = 5;

    const int Degree = 4, Sectors = 24, MinimumDecoys = 30;
    const double Exclusion = .25, DecoyStep = .03, AstigmatismBound = .5;
    static readonly double[] AstigmatismSteps = { -.3, -.15, 0, .15, .3 };
    static readonly (double Defocus, double Astigmatism)[] ClimbSteps = { (.004, .04), (.002, .02), (.001, .01) };

    /// <summary>Scores the fitting band's patch spectra, as <see cref="CtfPowerSpectrum.Extractor"/> returns them.</summary>
    /// <returns>The score, at least 0; NaN if the spectrum or the defocus search range is too small to tell.</returns>
    /// <param name="modulation">Per sample, the transfer that multiplies sin²γ (<see cref="SlabModulation"/>), or null for none.</param>
    public static double Score(IReadOnlyList<CtfPowerSpectrum.Observation> patches, CTF fit, double zMin, double zMax, double[] modulation = null)
    {
        if (patches == null || patches.Count == 0) return double.NaN;
        var layout = patches[0].Spectrum.Samples;
        var power = new double[layout.Length];
        foreach (var patch in patches)
        {
            var samples = patch.Spectrum.Samples;
            if (samples.Length != layout.Length) throw new ArgumentException("Patch spectra must share one sample layout.");
            for (int j = 0; j < samples.Length; j++) power[j] += samples[j].Power;
        }
        var pooled = new CtfSpectrumFit.Sample[layout.Length];
        for (int j = 0; j < pooled.Length; j++) pooled[j] = layout[j] with { Power = power[j] / patches.Count };
        return Score(pooled, fit, zMin, zMax, modulation);
    }

    /// <summary>
    /// The fit's slab modulation per sample, averaged over the patches the spectrum pools; null when the fit has no
    /// thickness. It is the specimen-thickness and aperture transfer the CTF fit itself multiplies sin²γ with.
    /// </summary>
    public static double[] SlabModulation(IReadOnlyList<CtfPowerSpectrum.Observation> patches, CtfFitGeometry[] geometry, double[] parameters, double voltageKV)
    {
        if (patches == null || patches.Count == 0 || geometry.Length != patches.Count) return null;
        if (!(parameters[geometry[0].ThicknessIndex] > 0)) return null;
        double kd = Math.PI * CtfSpectrumFit.Wavelength(voltageKV) * 1e4;
        var keys = geometry.Select(g => { var slab = g.SlabGeometry(parameters); return (Squared: parameters[g.ThicknessIndex] * slab.Factor, slab.WidthX, slab.WidthY); }).ToArray();
        var samples = patches[0].Spectrum.Samples;
        var modulation = new double[samples.Length];
        Parallel.For(0, samples.Length, j =>
        {
            double sum = 0, kq2 = kd * samples[j].Q2;
            foreach (var key in keys) sum += CtfSlabModel.ModulationValue(kq2, key.Squared, key.WidthX, key.WidthY);
            modulation[j] = sum / keys.Length;
        });
        return modulation;
    }

    /// <summary>Scores one spectrum; <paramref name="zMin"/> and <paramref name="zMax"/> bound the defocus search, in µm.</summary>
    public static double Score(CtfSpectrumFit.Sample[] samples, CTF fit, double zMin, double zMax, double[] modulation = null)
    {
        var spectrum = Spectrum.Create(samples, fit, modulation);
        if (spectrum == null || !(zMax > zMin)) return double.NaN;
        double defocus = (double)fit.Defocus;
        var (ax, ay) = Astigmatism(fit);
        int steps = Math.Max(2 * MinimumDecoys, (int)Math.Ceiling((zMax - zMin) / DecoyStep));
        var decoys = Enumerable.Range(0, steps + 1).Select(i => zMin + (zMax - zMin) * i / steps)
            .Where(d => Math.Abs(d - defocus) > Exclusion).ToArray();
        if (decoys.Length < MinimumDecoys) return double.NaN;
        var best = new double[decoys.Length];
        Parallel.For(0, decoys.Length, i =>
        {
            var start = (decoys[i], 0.0, 0.0); double value = double.NegativeInfinity;
            foreach (double x in AstigmatismSteps)
                foreach (double y in AstigmatismSteps)
                {
                    double v = spectrum.Correlation(decoys[i], x, y);
                    if (v > value) { value = v; start = (decoys[i], x, y); }
                }
            best[i] = spectrum.Climb(start, value, zMin, zMax);
        });
        double median = Median(best), spread = 1.4826 * Median(best.Select(b => Math.Abs(b - median)).ToArray());
        if (!(spread > 1e-12)) return double.NaN;
        double own = spectrum.Climb((defocus, ax, ay), spectrum.Correlation(defocus, ax, ay), zMin, zMax);
        return Math.Max(0, (own - median) / spread);
    }

    /// <summary>The fit's astigmatism components, in µm, as the CTF fit parametrizes them.</summary>
    public static (double X, double Y) Astigmatism(CTF fit)
    {
        double half = (double)fit.DefocusDelta / 2, angle = 2 * (double)fit.DefocusAngle * Math.PI / 180;
        return (half * Math.Cos(angle), half * Math.Sin(angle));
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone(); Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : .5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    }

    /// <summary>A spectrum flattened for scoring, against which any CTF can be correlated.</summary>
    public sealed class Spectrum
    {
        readonly double[] weight, flat, basis, modulation;
        // Per sample, in single precision for speed: γ = defocusFactor·defocus + astigmatism factors·astigmatism + constant.
        readonly float[] defocusFactor, astigmatismXFactor, astigmatismYFactor, constant;
        readonly int[] sectorStart;
        readonly double[][] inverseGram;
        readonly double flatEnergy;

        Spectrum(double[] q2, double[] q4, double[] astigX, double[] astigY, double[] weight, double[] logPower, double[] modulation, int[] sectorStart, CTF fit)
        {
            this.weight = weight; this.modulation = modulation; this.sectorStart = sectorStart;
            double lambda = CtfSpectrumFit.Wavelength((double)fit.Voltage);
            double kd = Math.PI * lambda * 1e4;
            double kc = -.5 * Math.PI * (double)fit.Cs * 1e7 * lambda * lambda * lambda;
            double phase = Math.Asin((double)fit.Amplitude) + (double)fit.PhaseShift * Math.PI;
            int n = q2.Length, terms = Degree + 1;
            defocusFactor = q2.Select(v => (float)(kd * v)).ToArray();
            astigmatismXFactor = astigX.Select(v => (float)(kd * v)).ToArray();
            astigmatismYFactor = astigY.Select(v => (float)(kd * v)).ToArray();
            constant = q4.Select(v => (float)(kc * v + phase)).ToArray();
            double lo = Math.Sqrt(q2.Min()), hi = Math.Sqrt(q2.Max());
            basis = new double[n * terms];
            for (int j = 0; j < n; j++)
            {
                // Legendre polynomials over the band keep the per-sector least squares well conditioned.
                double x = 2 * (Math.Sqrt(q2[j]) - lo) / (hi - lo) - 1, p0 = 1, p1 = x;
                basis[j * terms] = p0; basis[j * terms + 1] = p1;
                for (int k = 2; k < terms; k++) { double p = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; basis[j * terms + k] = p; p0 = p1; p1 = p; }
            }
            int sectors = sectorStart.Length - 1;
            inverseGram = new double[sectors][];
            flat = new double[n];
            double energy = 0;
            for (int s = 0; s < sectors; s++)
            {
                var gram = new double[terms * terms]; var rhs = new double[terms];
                for (int j = sectorStart[s]; j < sectorStart[s + 1]; j++)
                    for (int a = 0; a < terms; a++)
                    {
                        rhs[a] += weight[j] * basis[j * terms + a] * logPower[j];
                        for (int b = 0; b < terms; b++) gram[a * terms + b] += weight[j] * basis[j * terms + a] * basis[j * terms + b];
                    }
                inverseGram[s] = Invert(gram, terms);
                var c = Multiply(inverseGram[s], rhs, terms);
                for (int j = sectorStart[s]; j < sectorStart[s + 1]; j++)
                {
                    double trend = 0;
                    for (int a = 0; a < terms; a++) trend += basis[j * terms + a] * c[a];
                    flat[j] = logPower[j] - trend;
                    energy += weight[j] * flat[j] * flat[j];
                }
            }
            flatEnergy = energy;
        }

        /// <summary>The flattened spectrum, or null if it has too few samples for a background per sector.</summary>
        /// <param name="modulation">Per sample, the transfer that multiplies sin²γ, or null for none.</param>
        public static Spectrum Create(CtfSpectrumFit.Sample[] samples, CTF fit, double[] modulation = null)
        {
            if (samples == null || modulation != null && modulation.Length != samples.Length) return null;
            var kept = samples.Select((s, i) => (Sample: s, Sector: Sector(s), Modulation: modulation?[i] ?? 1))
                .Where(s => s.Sample.Power > 0 && double.IsFinite(s.Sample.Power) && s.Sample.Count > 0 && s.Sample.Q2 > 0)
                .OrderBy(s => s.Sector).ToArray();
            var sectorStart = new int[Sectors + 1];
            foreach (var s in kept) sectorStart[s.Sector + 1]++;
            for (int s = 0; s < Sectors; s++) sectorStart[s + 1] += sectorStart[s];
            // Each sector's background needs comfortably more samples than coefficients.
            for (int s = 0; s < Sectors; s++)
                if (sectorStart[s + 1] - sectorStart[s] is > 0 and < 4 * (Degree + 1)) return null;
            if (kept.Length < 4 * (Degree + 1) || kept.Min(s => s.Sample.Q2) >= kept.Max(s => s.Sample.Q2)) return null;
            double meanCount = kept.Average(s => s.Sample.Count);
            Spectrum spectrum;
            try
            {
                spectrum = new Spectrum(kept.Select(s => s.Sample.Q2).ToArray(), kept.Select(s => s.Sample.Q4).ToArray(),
                    kept.Select(s => s.Sample.AstigX).ToArray(), kept.Select(s => s.Sample.AstigY).ToArray(),
                    kept.Select(s => s.Sample.Count / meanCount).ToArray(), kept.Select(s => Math.Log(s.Sample.Power)).ToArray(),
                    kept.Select(s => s.Modulation).ToArray(), sectorStart, fit);
            }
            catch (ArithmeticException) { return null; }   // a sector's samples span too little frequency for a background
            return spectrum.flatEnergy > 0 ? spectrum : null;
        }

        static int Sector(CtfSpectrumFit.Sample s)
        {
            double angle = .5 * Math.Atan2(s.AstigY, s.AstigX);
            if (angle < 0) angle += Math.PI;
            return Math.Min(Sectors - 1, (int)(angle * Sectors / Math.PI));
        }

        /// <summary>
        /// Weighted correlation between the flattened spectrum and sin²γ for a defocus and astigmatism (µm), after
        /// removing from the model the same per-sector background the spectrum lost.
        /// </summary>
        public double Correlation(double defocus, double astigmatismX, double astigmatismY)
        {
            int terms = Degree + 1;
            float d = (float)defocus, x = (float)astigmatismX, y = (float)astigmatismY;
            double cross = 0, energy = 0, explained = 0;
            Span<double> projection = stackalloc double[terms];
            for (int s = 0; s < inverseGram.Length; s++)
            {
                projection.Clear();
                for (int j = sectorStart[s]; j < sectorStart[s + 1]; j++)
                {
                    float sine = MathF.Sin(defocusFactor[j] * d + astigmatismXFactor[j] * x + astigmatismYFactor[j] * y + constant[j]);
                    double model = modulation[j] * sine * sine, w = weight[j] * model;
                    // The flattened spectrum is orthogonal to each sector's background, so only the model's energy
                    // needs that background removed.
                    cross += w * flat[j]; energy += w * model;
                    for (int a = 0; a < terms; a++) projection[a] += w * basis[j * terms + a];
                }
                var inverse = inverseGram[s];
                for (int a = 0; a < terms; a++)
                {
                    double sum = 0;
                    for (int b = 0; b < terms; b++) sum += inverse[a * terms + b] * projection[b];
                    explained += projection[a] * sum;
                }
            }
            double residual = energy - explained;
            return residual > 0 ? cross / Math.Sqrt(residual * flatEnergy) : 0;
        }

        /// <summary>
        /// Coordinate ascent in defocus and astigmatism from <paramref name="start"/>, whose correlation is
        /// <paramref name="value"/>, within the defocus search range and the fit's astigmatism bounds.
        /// </summary>
        /// <returns>The correlation at the local maximum.</returns>
        public double Climb((double Defocus, double X, double Y) start, double value, double zMin, double zMax)
        {
            var at = start;
            foreach (var (defocusStep, astigmatismStep) in ClimbSteps)
                for (int round = 0; round < 4; round++)
                {
                    var next = at; bool moved = false;
                    for (int direction = 0; direction < 6; direction++)
                    {
                        double sign = direction % 2 == 0 ? 1 : -1;
                        var trial = direction switch
                        {
                            < 2 => (Math.Clamp(at.Defocus + sign * defocusStep, zMin, zMax), at.X, at.Y),
                            < 4 => (at.Defocus, Math.Clamp(at.X + sign * astigmatismStep, -AstigmatismBound, AstigmatismBound), at.Y),
                            _ => (at.Defocus, at.X, Math.Clamp(at.Y + sign * astigmatismStep, -AstigmatismBound, AstigmatismBound))
                        };
                        double v = Correlation(trial.Item1, trial.Item2, trial.Item3);
                        if (v > value) { value = v; next = trial; moved = true; }
                    }
                    if (!moved) break;
                    at = next;
                }
            return value;
        }

        static double[] Multiply(double[] matrix, double[] vector, int n)
        {
            var result = new double[n];
            for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) result[a] += matrix[a * n + b] * vector[b];
            return result;
        }

        // Gauss–Jordan with partial pivoting; the matrices are 5×5 Gram matrices of a Legendre basis.
        static double[] Invert(double[] matrix, int n)
        {
            var a = (double[])matrix.Clone(); var inverse = new double[n * n];
            for (int i = 0; i < n; i++) inverse[i * n + i] = 1;
            for (int c = 0; c < n; c++)
            {
                int pivot = c;
                for (int r = c + 1; r < n; r++) if (Math.Abs(a[r * n + c]) > Math.Abs(a[pivot * n + c])) pivot = r;
                if (!(Math.Abs(a[pivot * n + c]) > 1e-12 * Math.Abs(matrix[0]))) throw new ArithmeticException("Singular CTF signal background.");
                if (pivot != c)
                    for (int k = 0; k < n; k++)
                    {
                        (a[c * n + k], a[pivot * n + k]) = (a[pivot * n + k], a[c * n + k]);
                        (inverse[c * n + k], inverse[pivot * n + k]) = (inverse[pivot * n + k], inverse[c * n + k]);
                    }
                double scale = 1 / a[c * n + c];
                for (int k = 0; k < n; k++) { a[c * n + k] *= scale; inverse[c * n + k] *= scale; }
                for (int r = 0; r < n; r++)
                {
                    if (r == c) continue;
                    double f = a[r * n + c];
                    if (f == 0) continue;
                    for (int k = 0; k < n; k++) { a[r * n + k] -= f * a[c * n + k]; inverse[r * n + k] -= f * inverse[c * n + k]; }
                }
            }
            return inverse;
        }
    }
}
