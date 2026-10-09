using System;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>
/// Robust defocus trend learned from low-tilt search profiles. Profile contrast selects
/// coherent seeds; a soft Gaussian prior stabilizes uninformative tilts during joint fitting.
/// Strong spectral evidence can still support a real focus change.
/// </summary>
public sealed class CtfDefocusPrior
{
    readonly double[] angles;
    readonly int[] nodes, anchors;
    readonly double[] anchorWeights;
    readonly double zmin, step;
    readonly double[][] evidence;
    double intercept, slope, scatter, meanX, sumWeight, spreadX;

    // Defocus and its uncertainty are in micrometers. Prediction uncertainty grows
    // outside the angular coverage of the anchors; this is not a fixed defocus window.
    public double[] Centers => angles.Select(a => intercept + slope * Math.Sin(a)).ToArray();
    public double[] Scales => angles.Select(a => scatter * Math.Sqrt(1 + 1 / sumWeight + Math.Pow(Math.Sin(a) - meanX, 2) / spreadX)).ToArray();

    CtfDefocusPrior(double[][] profiles, double[] angles, int[] nodes, double zmin, double step)
    {
        this.angles = angles;
        this.nodes = nodes;
        this.zmin = zmin;
        this.step = step;
        // The coarse search improvement has a spectrum-dependent scale. Robust
        // profile contrast is a seed-selection heuristic, not a calibrated likelihood.
        // Continuous refinement uses the actual weighted spectral residual instead.
        evidence = profiles.Select(p =>
        {
            double median = Median(p);
            double noise = Math.Max(1e-8, 1.4826 * Median(p.Select(v => Math.Abs(v - median)).ToArray()));
            return p.Select(v => .5 * Math.Pow(Math.Max(0, (v - median) / noise), 2)).ToArray();
        }).ToArray();
        double lowLimit = Math.Max(Math.PI / 6, angles.Min(Math.Abs) + Math.PI / 18);
        var low = Enumerable.Range(0, angles.Length).Where(i => Math.Abs(angles[i]) <= lowLimit).ToArray();

        // Each tilt gets at most one vote. A spurious, exceptionally strong peak in
        // one low tilt must not define the trend for the entire series.
        var votes = new double[profiles[0].Length];
        foreach (int i in low)
        {
            double peak = evidence[i].Max();
            if (peak < 2) continue;
            for (int z = 0; z < votes.Length; z++)
            {
                double best = 0;
                for (int k = Math.Max(0, z - (int)Math.Ceiling(.3 / step)); k <= Math.Min(votes.Length - 1, z + (int)Math.Ceiling(.3 / step)); k++)
                    best = Math.Max(best, evidence[i][k] / peak * Math.Exp(-.5 * Math.Pow((z - k) * step / .15, 2)));
                votes[z] += Math.Min(1, peak / 8) * best;
            }
        }
        double center = zmin + Array.IndexOf(votes, votes.Max()) * step;
        int[] selected = low.Select(i => Best(evidence[i], center, .2)).ToArray();
        anchors = low.Where((i, j) => Math.Abs(zmin + selected[j] * step - center) <= .5 && evidence[i][selected[j]] >= 2).ToArray();
        anchorWeights = anchors.Select(i => Math.Min(1, evidence[i][Best(evidence[i], center, .2)] / 8)).ToArray();
        var initial = new double[nodes.Max() + 1];
        foreach (int i in anchors) initial[nodes[i]] = zmin + Best(evidence[i], center, .2) * step;
        Update(initial);
    }

    public static CtfDefocusPrior FromProfiles(double[][] profiles, double[] angles, int[] nodes, double zmin, double step)
    {
        // Shared defocus grids already constrain the tilts; this prior is for the
        // independent one-node-per-tilt geometry only.
        if (profiles.Length < 3 || angles.Length != profiles.Length || nodes.Length != profiles.Length || nodes.Distinct().Count() != nodes.Length)
            return null;
        var prior = new CtfDefocusPrior(profiles, angles, nodes, zmin, step);
        // Without three coherent low-tilt anchors, do not invent a known defocus.
        return prior.anchors.Length >= 3 ? prior : null;
    }

    static double Median(double[] values)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) * .5 : sorted[n / 2];
    }

    int Best(double[] scores, double center, double sigma)
    {
        int best = 0;
        double maximum = double.NegativeInfinity;
        for (int z = 0; z < scores.Length; z++)
        {
            double score = scores[z] - .5 * Math.Pow((zmin + z * step - center) / sigma, 2);
            if (score > maximum) { maximum = score; best = z; }
        }
        return best;
    }

    public int Select(int group) => Best(evidence[group], Centers[group], Scales[group]);

    public void Update(double[] parameters)
    {
        if (anchors.Length < 3) return;
        double[] x = anchors.Select(i => Math.Sin(angles[i])).ToArray();
        double[] y = anchors.Select(i => parameters[nodes[i]]).ToArray();
        intercept = Median(y);
        slope = 0;
        var w = (double[])anchorWeights.Clone();
        for (int pass = 0; pass < 6; pass++)
        {
            sumWeight = w.Sum();
            meanX = x.Select((v, i) => v * w[i]).Sum() / sumWeight;
            double meanY = y.Select((v, i) => v * w[i]).Sum() / sumWeight;
            spreadX = x.Select((v, i) => w[i] * Math.Pow(v - meanX, 2)).Sum();
            slope = x.Select((v, i) => w[i] * (v - meanX) * (y[i] - meanY)).Sum() / Math.Max(.01, spreadX);
            intercept = meanY - slope * meanX;
            double[] residual = y.Select((v, i) => v - intercept - slope * x[i]).ToArray();
            // A 0.1 um floor avoids treating an unusually stable series as precisely
            // known. Larger measured scatter automatically weakens the restraint.
            scatter = Math.Max(.1, 1.4826 * Median(residual.Select(Math.Abs).ToArray()));
            for (int i = 0; i < w.Length; i++)
                w[i] = anchorWeights[i] * Math.Min(1, 1.5 * scatter / Math.Max(1e-12, Math.Abs(residual[i])));
        }
        spreadX = Math.Max(.01, spreadX);
    }

    public double Evaluate(double[] parameters, double[] gradient)
    {
        var centers = Centers;
        var scales = Scales;
        double loss = 0;
        for (int i = 0; i < nodes.Length; i++)
        {
            double residual = parameters[nodes[i]] - centers[i], precision = 1 / (scales[i] * scales[i]);
            loss += .5 * residual * residual * precision;
            gradient[nodes[i]] += residual * precision;
        }
        return loss;
    }
}
