using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

public static class CtfFitDiagnostics
{
    public sealed record Diagnostic(float2[] Spectrum, Cubic1D Background, Cubic1D Envelope, decimal Resolution, float2[] Quality)
    {
        internal float[] Weights { get; init; }
        internal float[] ModelAmplitude { get; init; }
    }
    public sealed record Result(Diagnostic[] Groups, Diagnostic Global);

    public sealed record Input(Image Image, CtfFitGeometry[] Geometry, int Group, int FirstFrame = 0, int FrameCount = -1);

    /// <summary>Full-Nyquist diagnostics after fitting. Only nuisance curves are refitted;
    /// CTF parameters remain frozen. Stream one tilt/frame group at a time to bound memory.</summary>
    public static Result CreateFullSpectrum(CtfPowerSpectrum.Extractor extractor, IEnumerable<Input> inputs,
        CtfFitEngine.Fit fit, CTF[] references, CTF globalReference, int window, float[][] displays)
    {
        float[][] sums = null, weights = null, models = null; float[] globalSum = null, globalWeight = null, globalModel = null;
        var displayWeights = new double[references.Length]; int fourierSize = 0;
        foreach (var display in displays) Array.Clear(display);
        foreach (var input in inputs)
        {
            var extraction = extractor.Extract(input.Image, firstFrame: input.FirstFrame, frameCount: input.FrameCount);
            var records = extraction.Observations.ToArray();
            if (records.Length != input.Geometry.Length) throw new ArgumentException("Diagnostic patch geometry differs from fitting geometry.");
            using var batch = new CtfGpuFitBatch(records.Select(r => r.Spectrum).ToArray());
            var poses = new double[records.Length * 7];
            for (int i = 0; i < records.Length; i++) input.Geometry[i].WritePose(fit.Parameters, poses, i * 7);
            // Robustly profile broadband background/envelope without updating any CTF parameter.
            for (int pass = 0; pass < 6; pass++)
            {
                var output = batch.Evaluate(poses, true);
                double change = 0;
                for (int i = 0; i < records.Length; i++) change = Math.Max(change, output[i * 9 + 8]);
                if (change < .01) break;
            }
            batch.Evaluate(poses);
            var profiled = fit with { Coefficients = batch.ReadCoefficients() };
            var diagnostic = Create(records, input.Geometry, profiled, new int[records.Length],
                new[] { references[input.Group] }, globalReference, extraction.FourierSize, window, new[] { extraction.Display });
            if (sums == null)
            {
                fourierSize = extraction.FourierSize; int bins = fourierSize / 2;
                sums = references.Select(_ => new float[bins]).ToArray(); weights = references.Select(_ => new float[bins]).ToArray();
                globalSum = new float[bins]; globalWeight = new float[bins]; globalModel = new float[bins];
                models = references.Select(_ => new float[bins]).ToArray();
            }
            void AddDiagnostic(Diagnostic d, float[] sum, float[] weight, float[] model)
            {
                for (int i = 0; i < sum.Length; i++) { sum[i] += d.Spectrum[i].Y * d.Weights[i]; weight[i] += d.Weights[i]; model[i] += d.ModelAmplitude[i] * d.Weights[i]; }
            }
            AddDiagnostic(diagnostic.Groups[0], sums[input.Group], weights[input.Group], models[input.Group]);
            AddDiagnostic(diagnostic.Global, globalSum, globalWeight, globalModel);
            int frames = input.FrameCount < 0 ? input.Image.Dims.Z - input.FirstFrame : input.FrameCount;
            displayWeights[input.Group] += frames;
            for (int i = 0; i < extraction.Display.Length; i++) displays[input.Group][i] += extraction.Display[i] * frames;
        }
        if (sums == null) throw new ArgumentException("No diagnostic inputs.");
        for (int g = 0; g < displays.Length; g++) if (displayWeights[g] > 0)
            for (int i = 0; i < displays[g].Length; i++) displays[g][i] /= (float)displayWeights[g];
        var groups = references.Select((r, i) => Finish(sums[i], weights[i], r, fourierSize, window, models[i])).ToArray();
        return new(groups, Finish(globalSum, globalWeight, globalReference, fourierSize, window, globalModel));
    }

    sealed class Accumulator
    {
        public readonly float[] Sum, Weight, Model, GlobalSum, GlobalWeight, GlobalModel, Background, Count;
        public Accumulator(int bins, int displayBins, bool sameReference)
        {
            Sum = new float[bins]; Weight = new float[bins]; Model = new float[bins];
            GlobalSum = sameReference ? Sum : new float[bins];
            GlobalWeight = sameReference ? Weight : new float[bins];
            GlobalModel = sameReference ? Model : new float[bins];
            Background = new float[displayBins]; Count = new float[displayBins];
        }
    }

    // A single pass over each group's patches produces its curve, the combined curve,
    // and the background-subtracted display. Memory scales with histogram size, not samples × patches.
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
                for (int j = 0; j < spectrum.Samples.Length; j++)
                {
                    var s = spectrum.Samples[j];
                    var (background, envelope) = spectrum.EvaluateNuisance(j, fit.Coefficients, i * stride);
                    // Express the depth/patch-averaged model as effective background + signed
                    // envelope * sin²(gamma). Keep that signed envelope in the prediction:
                    // sinc reversals remain visible and no inverse-transfer filtering is applied.
                    float modulation = modulations[j];
                    background += .5f*envelope*(1-modulation);
                    envelope *= modulation;
                    float q2 = (float)s.Q2, count = (float)s.Count;
                    int b = displayBin[j];
                    if (b < displayBins)
                    {
                        a.Background[b] += background * powerScale * count;
                        a.Count[b] += count;
                    }
                    // Preserve measured power: subtract the effective smooth background,
                    // but never divide by the envelope or the slab/aperture transfer.
                    float modelAmplitude = envelope * powerScale;
                    float target = kd * (q2 * df + (float)s.AstigX * ax + (float)s.AstigY * ay) + kc * (float)s.Q4 + phase;
                    float weight = count;
                    float value = (float)s.Power - background * powerScale;
                    Accumulate(a.Sum, a.Weight, a.Model, modelAmplitude, target - referencePhase, referenceDf, kc, q2, pixel * fourierSize, weight, value);
                    if (!same) Accumulate(a.GlobalSum, a.GlobalWeight, a.GlobalModel, modelAmplitude, target - globalPhase, globalDf, kc, q2, pixel * fourierSize, weight, value);
                }
            }
        });
        for (int g = 0; g < references.Length; g++) accumulators[g] = new Accumulator(bins, displayBins, ReferenceEquals(references[g], globalReference));
        for (int c = 0; c < chunks.Count; c++)
        {
            var a = accumulators[chunks[c].Group]; var part = partials[c];
            Add(a.Sum, part.Sum); Add(a.Weight, part.Weight); Add(a.Model, part.Model);
            if (!ReferenceEquals(a.Sum, a.GlobalSum)) { Add(a.GlobalSum, part.GlobalSum); Add(a.GlobalWeight, part.GlobalWeight); Add(a.GlobalModel, part.GlobalModel); }
            Add(a.Background, part.Background); Add(a.Count, part.Count);
        }
        Parallel.For(0, references.Length, group =>
        {
            var a = accumulators[group];
            SubtractDisplayBackground(displays[group], a, window);
            diagnostics[group] = Finish(a.Sum, a.Weight, references[group], fourierSize, window, a.Model);
        });
        if (references.Length == 1 && ReferenceEquals(references[0], globalReference)) return new(diagnostics, diagnostics[0]);
        // Fixed reduction order makes diagnostic output independent of worker scheduling.
        var sum = new float[bins]; var weights = new float[bins]; var model = new float[bins];
        foreach (var a in accumulators) for (int b = 0; b < bins; b++) { sum[b] += a.GlobalSum[b]; weights[b] += a.GlobalWeight[b]; model[b] += a.GlobalModel[b]; }
        return new(diagnostics, Finish(sum, weights, globalReference, fourierSize, window, model));
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

    static void Accumulate(float[] sum, float[] weights, float[] model, float amplitude, float target, float kdDf, float kc,
        float sourceQ2, float scale, float weight, float value)
    {
        float q2 = AlignFrequencySquared(target, kdDf, kc, sourceQ2);
        if (!(q2 > 0) || !float.IsFinite(q2)) return;
        float r = MathF.Sqrt(q2) * scale;
        if (r >= sum.Length - 1) return;
        int b = (int)r; float f = r - b;
        sum[b] += weight * value * (1 - f); weights[b] += weight * (1 - f);
        sum[b + 1] += weight * value * f; weights[b + 1] += weight * f;
        model[b] += weight * amplitude * (1 - f); model[b + 1] += weight * amplitude * f;
    }

    static void SubtractDisplayBackground(float[] display, Accumulator a, int window)
    {
        int bins = a.Count.Length;
        for (int b = 0; b < bins; b++) if (a.Count[b] > 0) a.Background[b] /= a.Count[b];
        for (int y = 0; y < window / 2; y++) for (int x = 0; x < window; x++)
        {
            int xx = x - window / 2, yy = window / 2 - 1 - y, b = (int)MathF.Sqrt(xx * xx + yy * yy), i = y * window + x;
            display[i] = b < bins && a.Count[b] > 0 ? display[i] - a.Background[b] : 0;
        }
    }

    // X is frequency in cycles/pixel, matching PS1D. NaN marks unsupported frequencies.
    // A two-oscillation window follows CTF phase, with a minimum width set by the
    // real-space aperture rather than FFT padding. Correlation is descriptive, not a p-value.
    public static float2[] CalculateQuality(float2[] spectrum, float[] support, CTF reference, int spatialWindow, float[] modelAmplitude = null)
    {
        int n = spectrum.Length;
        if (n < 2 || support.Length != n || spatialWindow <= 0)
            throw new ArgumentException("Invalid CTF quality inputs.");
        var arc = PhaseArc(spectrum, reference);
        var model = spectrum.Select((p, i) => Math.Pow(reference.Get1DDouble(p.X / (double)reference.PixelSize, false, true, true), 2) * (modelAmplitude == null ? 1 : modelAmplitude[i])).ToArray();
        var result = spectrum.Select(p => new float2(p.X, float.NaN)).ToArray();
        double dataScale = spectrum.Where((p, i) => support[i] > 0 && float.IsFinite(p.Y)).Select(p => Math.Abs((double)p.Y)).DefaultIfEmpty(0).Max();
        double modelScale = model.Select(Math.Abs).DefaultIfEmpty(0).Max();
        if (!(dataScale > 0) || !(modelScale > 0)) return result;
        // Prefix moments make the adaptive windows linear-time. FP64 only for these
        // small scalar sums, where subtraction of nearly equal moments loses precision.
        var moments = new double[6][];
        for (int j = 0; j < moments.Length; j++) moments[j] = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < moments.Length; j++) moments[j][i + 1] = moments[j][i];
            if (!(support[i] > 0) || !float.IsFinite(spectrum[i].Y)) continue;
            double x = spectrum[i].Y / dataScale, y = model[i] / modelScale;
            double[] values = { 1, x, y, x*x, y*y, x*y };
            for (int j = 0; j < moments.Length; j++) moments[j][i + 1] += values[j];
        }
        int left = 0, right = 0;
        double halfWidth = 2.0 / spatialWindow;
        for (int i = 0; i < n; i++)
        {
            while (left < i && arc[i] - arc[left + 1] >= Math.PI && spectrum[i].X - spectrum[left + 1].X >= halfWidth) left++;
            right = Math.Max(right, i);
            while (right < n - 1 && (arc[right] - arc[i] < Math.PI || spectrum[right].X - spectrum[i].X < halfWidth)) right++;
            if (!(support[i] > 0) || arc[i] - arc[left] < Math.PI || arc[right] - arc[i] < Math.PI ||
                spectrum[i].X - spectrum[left].X < halfWidth || spectrum[right].X - spectrum[i].X < halfWidth) continue;
            double count = moments[0][right + 1] - moments[0][left];
            if (count < 6 || count < .8 * (right - left + 1)) continue;
            double x = moments[1][right + 1] - moments[1][left], y = moments[2][right + 1] - moments[2][left];
            double energyX = moments[3][right + 1] - moments[3][left], energyY = moments[4][right + 1] - moments[4][left];
            double xx = energyX - x*x/count, yy = energyY - y*y/count;
            double xy = moments[5][right + 1] - moments[5][left] - x*y/count;
            if (xx <= Math.Max(1e-30, 1e-12 * energyX) || yy <= Math.Max(1e-30, 1e-12 * energyY)) continue;
            result[i].Y = (float)Math.Clamp(xy / Math.Sqrt(xx*yy), -1, 1);
        }
        return result;
    }

    static double[] PhaseArc(float2[] spectrum, CTF reference)
    {
        double lambda = CtfSpectrumFit.Wavelength((double)reference.Voltage);
        double kd = Math.PI * lambda * 1e4 * (double)reference.Defocus;
        double kc = -.5 * Math.PI * (double)reference.Cs * 1e7 * lambda*lambda*lambda;
        var arc = new double[spectrum.Length]; double previous = 0;
        for (int i = 0; i < arc.Length; i++)
        {
            double q = spectrum[i].X / (double)reference.PixelSize, phase = kd*q*q + kc*q*q*q*q;
            if (i > 0) arc[i] = arc[i-1] + Math.Abs(phase-previous);
            previous = phase;
        }
        return arc;
    }

    public static decimal EstimateResolution(float2[] quality, CTF reference)
    {
        var arc = PhaseArc(quality, reference);
        int goodStart = -1, badStart = -1, last = -1;
        for (int i = 0; i < quality.Length; i++)
        {
            if (float.IsFinite(quality[i].Y) && quality[i].Y > .3f)
            {
                badStart = -1;
                if (goodStart < 0) goodStart = i;
                // Require support over a full CTF-power oscillation, independent of padding.
                if (arc[i] - arc[goodStart] >= Math.PI) last = i;
            }
            else
            {
                goodStart = -1;
                if (badStart < 0) badStart = i;
                if (last >= 0 && arc[i] - arc[badStart] >= Math.PI) break;
            }
        }
        return last >= 0 && quality[last].X > 0 ?
            (decimal)Math.Round((double)reference.PixelSize / quality[last].X, 1) : 0;
    }

    static Diagnostic Finish(float[] sum, float[] weight, CTF reference, int fourierSize, int spatialWindow, float[] modelSum)
    {
        var ps = new float2[sum.Length];
        for (int b = 0; b < ps.Length; b++) ps[b] = new float2((float)b / fourierSize, weight[b] > 0 ? sum[b] / weight[b] : 0);
        var amplitude = weight.Select((w, i) => w > 0 ? modelSum[i] / w : 0).ToArray();
        var quality = CalculateQuality(ps, weight, reference, spatialWindow, amplitude);
        var zero = new Cubic1D(new[] { new float2(0, 0), new float2(.5f, 0) });
        var envelope = new Cubic1D(ps.Select((p, i) => new float2(p.X, amplitude[i])).ToArray());
        return new(ps, zero, envelope, EstimateResolution(quality, reference), quality) { Weights = weight, ModelAmplitude = amplitude };
    }
}
