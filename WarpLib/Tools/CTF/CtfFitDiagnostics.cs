using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

public static class CtfFitDiagnostics
{
    const float EnvelopeFractionFloor = 1e-3f;
    public sealed record Diagnostic(float2[] Spectrum, Cubic1D Background, Cubic1D Envelope, decimal Resolution);
    public sealed record Result(Diagnostic[] Groups, Diagnostic Global);

    sealed class Accumulator
    {
        public readonly float[] Sum, Weight, GlobalSum, GlobalWeight, Background, Envelope, Count;
        public Accumulator(int bins, int displayBins, bool sameReference)
        {
            Sum = new float[bins]; Weight = new float[bins];
            GlobalSum = sameReference ? Sum : new float[bins];
            GlobalWeight = sameReference ? Weight : new float[bins];
            Background = new float[displayBins]; Envelope = new float[displayBins]; Count = new float[displayBins];
        }
    }

    // A single pass over each group's patches produces its curve, the combined curve,
    // and the display normalization. Memory scales with histogram size, not samples × patches.
    // All frequency-sized arithmetic is FP32; fixed patch chunks run in parallel, including movies with one output group.
    public static Result Create(CtfPowerSpectrum.Observation[] records, CtfFitGeometry[] geometry,
        CtfFitEngine.Fit fit, int[] groups, CTF[] references, CTF globalReference,
        int fourierSize, int window, float[][] displays)
    {
        if (records.Length == 0 || records.Length != geometry.Length || records.Length != groups.Length ||
            references.Length == 0 || references.Length != displays.Length ||
            groups.Any(g => g < 0 || g >= references.Length)) throw new ArgumentException("Invalid CTF diagnostic groups.");
        int stride = records[0].Spectrum.KnotCount * 2;
        if (fit.Coefficients == null || fit.Coefficients.Length != records.Length * stride)
            throw new ArgumentException("Diagnostics require the accepted GPU nuisance coefficients.");
        int bins = fourierSize / 2, displayBins = window / 2, nd = geometry[0].DefocusWeights.Length;
        var members = Enumerable.Range(0, records.Length).GroupBy(i => groups[i]).ToDictionary(g => g.Key, g => g.ToArray());
        // Bound the number of partial histograms independently of movie length.
        // Fixed partitioning also makes reductions independent of CPU thread count.
        int chunkSize = Math.Max(1, (records.Length + 63) / 64);
        var chunks = new List<(int Group, int[] Ids)>();
        for (int g = 0; g < references.Length; g++)
            if (members.TryGetValue(g, out var ids))
                foreach (var chunk in ids.Chunk(chunkSize)) chunks.Add((g, chunk));
        var partials = new Accumulator[chunks.Count];
        var accumulators = new Accumulator[references.Length];
        float kd = (float)(Math.PI * CtfSpectrumFit.Wavelength((double)globalReference.Voltage) * 1e4);
        float kc = (float)(-.5 * Math.PI * (double)globalReference.Cs * 1e7 * Math.Pow(CtfSpectrumFit.Wavelength((double)globalReference.Voltage), 3));
        float pixel = (float)globalReference.PixelSize, ax = (float)fit.Parameters[nd], ay = (float)fit.Parameters[nd + 1];
        float globalDf = kd * (float)globalReference.Defocus, globalPhase = (float)globalReference.PhaseShift * MathF.PI;
        // Bin boundaries use the original frequency moments once, avoiding FP32
        // rounding moving a sample that lies exactly on a display-pixel boundary.
        int[] displayBin = records[0].Spectrum.Samples.Select(s => (int)(Math.Sqrt(s.Q2) * (double)globalReference.PixelSize * window)).ToArray();
        // All patches with the same plane/window share their depth/aperture modulation.
        // Cache it once instead of evaluating trigonometry in every diagnostic sample loop.
        var modulationKeys = geometry.Select(g =>
        {
            var slab = g.SlabGeometry(fit.Parameters);
            return (Squared:fit.Parameters[g.ThicknessIndex]*slab.Factor, slab.WidthX, slab.WidthY);
        }).ToArray();
        var modulationCache = new Dictionary<(double Squared,double WidthX,double WidthY),float[]>();
        foreach (var key in modulationKeys.Distinct())
        {
            var values = new float[records[0].Spectrum.Samples.Length];
            for (int j = 0; j < values.Length; j++)
                values[j] = (float)CtfSlabModel.Modulation(kd*records[0].Spectrum.Samples[j].Q2,key.Squared,key.WidthX,key.WidthY).Value;
            modulationCache.Add(key,values);
        }
        var diagnostics = new Diagnostic[references.Length];
        Parallel.For(0, chunks.Count, chunk =>
        {
            int group = chunks[chunk].Group;
            if (displays[group].Length != window * displayBins) throw new ArgumentException("Invalid CTF display size.");
            bool same = ReferenceEquals(references[group], globalReference);
            var a = partials[chunk] = new Accumulator(bins, displayBins, same);
            float referenceDf = kd * (float)references[group].Defocus, referencePhase = (float)references[group].PhaseShift * MathF.PI;
            foreach (int i in chunks[chunk].Ids)
            {
                var spectrum = records[i].Spectrum;
                var local = geometry[i].Evaluate(fit.Parameters);
                var modulations = modulationCache[modulationKeys[i]];
                float df = (float)local.Defocus, phase = (float)local.Phase, powerScale = (float)spectrum.PowerScale;
                float envelopeMaximum = 0;
                for (int k = stride / 2; k < stride; k++) envelopeMaximum = MathF.Max(envelopeMaximum, fit.Coefficients[i * stride + k]);
                // Positive B-spline coefficients bound the envelope. Dividing by a vanishing
                // envelope creates huge curve spikes even though those bins carry no signal.
                float envelopeFloor = MathF.Max(1e-8f, envelopeMaximum * EnvelopeFractionFloor);
                for (int j = 0; j < spectrum.Samples.Length; j++)
                {
                    var s = spectrum.Samples[j];
                    var (background, envelope) = spectrum.EvaluateNuisance(j, fit.Coefficients, i * stride);
                    // Express the depth/patch-averaged model as effective background + signed
                    // envelope * sin²(gamma). This aligns curves without treating sinc reversals
                    // as anticorrelated CTFs; near-zero transfer carries negligible diagnostic weight.
                    float modulation = modulations[j];
                    background += .5f*envelope*(1-modulation);
                    envelope *= modulation;
                    float q2 = (float)s.Q2, count = (float)s.Count;
                    int b = displayBin[j];
                    if (b < displayBins)
                    {
                        a.Background[b] += background * powerScale * count;
                        a.Envelope[b] += envelope * powerScale * count; a.Count[b] += count;
                    }
                    if (MathF.Abs(envelope) <= envelopeFloor) continue;
                    float target = kd * (q2 * df + (float)s.AstigX * ax + (float)s.AstigY * ay) + kc * (float)s.Q4 + phase;
                    float weight = count * envelope * envelope;
                    float value = ((float)(s.Power / spectrum.PowerScale) - background) / envelope;
                    Accumulate(a.Sum, a.Weight, target - referencePhase, referenceDf, kc, q2, pixel * fourierSize, weight, value);
                    if (!same) Accumulate(a.GlobalSum, a.GlobalWeight, target - globalPhase, globalDf, kc, q2, pixel * fourierSize, weight, value);
                }
            }
        });
        for (int g = 0; g < references.Length; g++) accumulators[g] = new Accumulator(bins, displayBins, ReferenceEquals(references[g], globalReference));
        for (int c = 0; c < chunks.Count; c++)
        {
            var a = accumulators[chunks[c].Group]; var part = partials[c];
            Add(a.Sum, part.Sum); Add(a.Weight, part.Weight);
            if (!ReferenceEquals(a.Sum, a.GlobalSum)) { Add(a.GlobalSum, part.GlobalSum); Add(a.GlobalWeight, part.GlobalWeight); }
            Add(a.Background, part.Background); Add(a.Envelope, part.Envelope); Add(a.Count, part.Count);
        }
        Parallel.For(0, references.Length, group =>
        {
            var a = accumulators[group];
            NormalizeDisplay(displays[group], a, window);
            diagnostics[group] = Finish(a.Sum, a.Weight, references[group], fourierSize);
        });
        if (references.Length == 1 && ReferenceEquals(references[0], globalReference)) return new(diagnostics, diagnostics[0]);
        // Fixed reduction order makes diagnostic output independent of worker scheduling.
        var sum = new float[bins]; var weights = new float[bins];
        foreach (var a in accumulators) for (int b = 0; b < bins; b++) { sum[b] += a.GlobalSum[b]; weights[b] += a.GlobalWeight[b]; }
        return new(diagnostics, Finish(sum, weights, globalReference, fourierSize));
    }

    static void Add(float[] target, float[] source)
    {
        for (int i = 0; i < target.Length; i++) target[i] += source[i];
    }

    // Solve kc*u² + kdDf*u = target without cancellation in the low-frequency root.
    // If both roots are positive, choose the branch nearest the source frequency.
    public static float AlignFrequencySquared(float target, float kdDf, float kc, float sourceQ2)
    {
        if (kc == 0) return kdDf != 0 ? target / kdDf : float.NaN;
        float discriminant = kdDf * kdDf + 4 * kc * target;
        if (discriminant < 0) return float.NaN;
        float q = -.5f * (kdDf + MathF.CopySign(MathF.Sqrt(discriminant), kdDf));
        float first = q / kc, second = q != 0 ? -target / q : float.NaN;
        if (!(first > 0)) return second;
        if (!(second > 0)) return first;
        return MathF.Abs(first - sourceQ2) < MathF.Abs(second - sourceQ2) ? first : second;
    }

    static void Accumulate(float[] sum, float[] weights, float target, float kdDf, float kc,
        float sourceQ2, float scale, float weight, float value)
    {
        float q2 = AlignFrequencySquared(target, kdDf, kc, sourceQ2);
        if (!(q2 > 0) || !float.IsFinite(q2)) return;
        float r = MathF.Sqrt(q2) * scale;
        if (r >= sum.Length - 1) return;
        int b = (int)r; float f = r - b;
        sum[b] += weight * value * (1 - f); weights[b] += weight * (1 - f);
        sum[b + 1] += weight * value * f; weights[b + 1] += weight * f;
    }

    static void NormalizeDisplay(float[] display, Accumulator a, int window)
    {
        int bins = a.Count.Length;
        for (int b = 0; b < bins; b++) if (a.Count[b] > 0) { a.Background[b] /= a.Count[b]; a.Envelope[b] /= a.Count[b]; }
        float floor = a.Envelope.Max(v => MathF.Abs(v)) * EnvelopeFractionFloor;
        for (int y = 0; y < window / 2; y++) for (int x = 0; x < window; x++)
        {
            int xx = x - window / 2, yy = window / 2 - 1 - y, b = (int)MathF.Sqrt(xx * xx + yy * yy), i = y * window + x;
            display[i] = b < bins && a.Count[b] > 0 && MathF.Abs(a.Envelope[b]) > floor ? (display[i] - a.Background[b]) / a.Envelope[b] : 0;
        }
    }

    static Diagnostic Finish(float[] sum, float[] weight, CTF reference, int window)
    {
        int bins = sum.Length;
        float2[] ps = new float2[bins];
        for (int b = 0; b < bins; b++) ps[b] = new float2((float)b / window, weight[b] > 0 ? sum[b] / weight[b] : 0);
        double pixel = (double)reference.PixelSize;
        double[] model = Enumerable.Range(0, bins).Select(b => Math.Pow(reference.Get1DDouble((double)b / window / pixel, false, true, true), 2)).ToArray();
        int last = 0, good = 0, bad = 0;
        // These small scalar correlation sums use FP64 to avoid cancellation in variance.
        for (int b = 8; b < bins - 8; b++)
        {
            double n = 0, x = 0, y = 0, xx = 0, yy = 0, xy = 0;
            for (int k = b - 8; k <= b + 8; k++) if (weight[k] > 0) { double a = ps[k].Y, c = model[k]; n++; x += a; y += c; xx += a * a; yy += c * c; xy += a * c; }
            double corr = n >= 12 ? (xy - x * y / n) / Math.Sqrt(Math.Max(1e-30, (xx - x * x / n) * (yy - y * y / n))) : 0;
            if (corr > .3)
            {
                good++; bad = 0;
                if (good >= 16) last = b;
            }
            else
            {
                bad++;
                if (bad >= 16 && last > 0) break;
                good = 0;
            }
        }
        var zero = new Cubic1D(new[] { new float2(0, 0), new float2(.5f, 0) });
        var one = new Cubic1D(new[] { new float2(0, 1), new float2(.5f, 1) });
        return new(ps, zero, one, last > 0 ? (decimal)Math.Round(pixel * window / last, 1) : 0);
    }
}
