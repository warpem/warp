using System;
using System.Linq;
using Warp.Tools;
using Xunit;

namespace Tests;

public class CtfDefocusPriorTests
{
    static readonly double[] Angles = { -60, -25, -15, 0, 15, 25, 60 };
    static double[][] Profiles(bool strongJump = false)
    {
        return Angles.Select((angle, i) => Enumerable.Range(0, 301).Select(j =>
        {
            double z = .5 + j * .02;
            double height = Math.Abs(angle) < 30 ? 12 : 1;
            double noise = Math.Sin(j * 1.71 + i) + .7 * Math.Cos(j * .61 - i);
            double truePeak = height * Math.Exp(-.5 * Math.Pow((z - 1.8) / .04, 2));
            double falsePeak = (i == 0 ? 3 : i == 6 ? (strongJump ? 35 : 4) : 0) * Math.Exp(-.5 * Math.Pow((z - 3.8) / .04, 2));
            return noise + truePeak + falsePeak;
        }).ToArray()).ToArray();
    }
    static CtfDefocusPrior Prior(bool strongJump = false) => CtfDefocusPrior.FromProfiles(Profiles(strongJump), Angles.Select(a => a * Math.PI / 180).ToArray(), Enumerable.Range(0, 7).ToArray(), .5, .02);

    [Fact]
    public void WeakTiltsUseConsensusButStrongEvidenceCanSupportRealFocusChanges()
    {
        var weak = Prior();
        Assert.NotNull(weak);
        Assert.InRange(.5 + weak.Select(0) * .02, 1.6, 2);
        Assert.InRange(.5 + weak.Select(6) * .02, 1.6, 2);
        var strong = Prior(true);
        Assert.InRange(.5 + strong.Select(6) * .02, 3.7, 3.9);
    }

    [Fact]
    public void PriorGradientMatchesFiniteDifferences()
    {
        var prior = Prior();
        double[] p = { 2.1, 1.7, 1.8, 1.9, 1.75, 1.85, 2.2 };
        var gradient = new double[p.Length];
        prior.Evaluate(p, gradient);
        for (int i = 0; i < p.Length; i++)
        {
            const double h = 1e-5;
            p[i] += h; double plus = prior.Evaluate(p, new double[p.Length]);
            p[i] -= 2*h; double minus = prior.Evaluate(p, new double[p.Length]);
            p[i] += h;
            Assert.InRange(Math.Abs((plus-minus)/(2*h)-gradient[i]), 0, 1e-7);
        }
    }

    [Fact]
    public void FlatProfilesDoNotInventAnAnchor()
    {
        Assert.Null(CtfDefocusPrior.FromProfiles(Angles.Select(_ => new double[301]).ToArray(), Angles, Enumerable.Range(0, 7).ToArray(), .5, .02));
    }

    [Fact]
    public void AngleSignAndNodeOrderingDoNotChangeTheConstraint()
    {
        var prior = Prior();
        var reversed = CtfDefocusPrior.FromProfiles(Profiles(), Angles.Select(a => -a * Math.PI/180).ToArray(), Enumerable.Range(0,7).Reverse().ToArray(), .5, .02);
        double[] p = { 2.1, 1.7, 1.8, 1.9, 1.75, 1.85, 2.2 };
        Assert.Equal(prior.Evaluate(p, new double[7]), reversed.Evaluate(p.Reverse().ToArray(), new double[7]), 8);
    }
    [Fact]
    public void OneCorruptedLowTiltCannotSetTheSeriesTrend()
    {
        var profiles = Profiles();
        for (int j = 0; j < profiles[2].Length; j++)
            profiles[2][j] += 200 * Math.Exp(-.5 * Math.Pow((.5+j*.02-4.8)/.04, 2));
        var prior = CtfDefocusPrior.FromProfiles(profiles, Angles.Select(a => a*Math.PI/180).ToArray(), Enumerable.Range(0,7).ToArray(), .5, .02);
        Assert.NotNull(prior);
        foreach (double center in prior.Centers) Assert.InRange(center, 1.6, 2);
    }

}
