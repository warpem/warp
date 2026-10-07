using System;
using System.Collections.Generic;
using System.Linq;
using ZLinq;

namespace Warp.Tools;

/// <summary>Bounded limited-memory BFGS with analytic gradients and an Armijo line search.</summary>
public static class CtfFitOptimizer
{
    public readonly record struct Result(double[] Parameters, double Loss, int Iterations, int Evaluations, bool Converged);
    // Each optimizer retains its own line search and history; only objective evaluations
    // are batched. This avoids thousands of tiny GPU round trips during tilt initialization.
    sealed class Request
    {
        public double[] Parameters;
        public double Loss;
        public double[] Gradient;
    }
    sealed class State { public Result Result; }

    public static Result Minimize(Func<double[], (double Loss, double[] Gradient)> evaluate,
        double[] initial, double[] scale, double[] lower, double[] upper, int maximumIterations = 80)
    {
        var state = new State();
        foreach (var request in Iterate(state, initial, scale, lower, upper, maximumIterations))
            (request.Loss, request.Gradient) = evaluate(request.Parameters);
        return state.Result;
    }

    internal static Result[] MinimizeMany(Func<double[][], (double Loss, double[] Gradient)[]> evaluate,
        double[][] initial, double[] scale, double[] lower, double[] upper, int maximumIterations)
    {
        var states = initial.Select(_ => new State()).ToArray();
        var iterators = initial.Select((p, i) => Iterate(states[i], p, scale, lower, upper, maximumIterations).GetEnumerator()).ToArray();
        var active = Enumerable.Repeat(true, initial.Length).ToArray();
        var poses = initial.Select(p => (double[])p.Clone()).ToArray();
        try
        {
            while (true)
            {
                bool any = false;
                for (int i = 0; i < iterators.Length; i++)
                    if (active[i])
                    {
                        active[i] = iterators[i].MoveNext();
                        if (active[i]) { poses[i] = iterators[i].Current.Parameters; any = true; }
                    }
                if (!any) break;
                var output = evaluate(poses);
                for (int i = 0; i < iterators.Length; i++) if (active[i])
                    (iterators[i].Current.Loss, iterators[i].Current.Gradient) = output[i];
            }
        }
        finally { foreach (var iterator in iterators) iterator.Dispose(); }
        return states.Select(s => s.Result).ToArray();
    }

    static IEnumerable<Request> Iterate(State state, double[] initial, double[] scale, double[] lower, double[] upper, int maximumIterations)
    {
        int n = initial.Length, evaluations = 0;
        if (scale.Length != n || lower.Length != n || upper.Length != n || scale.Any(s => !(s > 0))) throw new ArgumentException("Invalid optimizer dimensions/scales.");
        double[] x = initial.Select((v, i) => Math.Clamp(v, lower[i], upper[i]) / scale[i]).ToArray();
        Request MakeRequest(double[] q)
        {
            evaluations++;
            return new Request { Parameters = q.Select((v, i) => v * scale[i]).ToArray() };
        }
        (double Loss, double[] Gradient) Read(Request request) =>
            (request.Loss, request.Gradient.Select((v, i) => v * scale[i]).ToArray());
        var first = MakeRequest(x); yield return first;
        var current = Read(first);
        if (!double.IsFinite(current.Loss) || current.Gradient.Any(v => !double.IsFinite(v))) throw new InvalidOperationException("Invalid initial CTF objective.");
        var history = new List<(double[] S, double[] Y, double Rho)>();
        Result Finish(int iter, bool converged) => new(x.Select((v, i) => v * scale[i]).ToArray(), current.Loss, iter, evaluations, converged);
        for (int iteration = 0; iteration < maximumIterations; iteration++)
        {
            double[] g = (double[])current.Gradient.Clone();
            for (int i = 0; i < n; i++) if ((x[i] * scale[i] <= lower[i] && g[i] > 0) || (x[i] * scale[i] >= upper[i] && g[i] < 0)) g[i] = 0;
            double norm = Math.Sqrt(g.Sum(v => v * v));
            if (norm < 1e-7 * Math.Max(1, Math.Sqrt(Math.Abs(current.Loss)))) { state.Result = Finish(iteration, true); yield break; }
            double[] direction = (double[])g.Clone(), alpha = new double[history.Count];
            for (int k = history.Count - 1; k >= 0; k--)
            {
                alpha[k] = history[k].Rho * Dot(history[k].S, direction);
                for (int i = 0; i < n; i++) direction[i] -= alpha[k] * history[k].Y[i];
            }
            double gamma = history.Count > 0 ? Dot(history[^1].S, history[^1].Y) / Dot(history[^1].Y, history[^1].Y) : 1 / Math.Max(1, norm);
            for (int i = 0; i < n; i++) direction[i] *= gamma;
            for (int k = 0; k < history.Count; k++)
            {
                double beta = history[k].Rho * Dot(history[k].Y, direction);
                for (int i = 0; i < n; i++) direction[i] += history[k].S[i] * (alpha[k] - beta);
            }
            for (int i = 0; i < n; i++) direction[i] = -direction[i];
            if (Dot(direction, g) >= 0) { history.Clear(); for (int i = 0; i < n; i++) direction[i] = -g[i] / Math.Max(1, norm); }
            double cap = Math.Max(1, direction.Max(Math.Abs) / 4);
            for (int i = 0; i < n; i++) direction[i] /= cap;
            double[] trial = null; (double Loss, double[] Gradient) next = default; bool accepted = false;
            for (double step = 1; step >= 1.0 / 65536; step *= .5)
            {
                trial = x.Select((v, i) => Math.Clamp((v + step * direction[i]) * scale[i], lower[i], upper[i]) / scale[i]).ToArray();
                double predicted = 0; for (int i = 0; i < n; i++) predicted += current.Gradient[i] * (trial[i] - x[i]);
                if (!(predicted < 0)) continue;
                var request = MakeRequest(trial); yield return request;
                next = Read(request);
                if (double.IsFinite(next.Loss) && next.Gradient.All(double.IsFinite) && next.Loss <= current.Loss + 1e-4 * predicted) { accepted = true; break; }
            }
            if (!accepted) { state.Result = Finish(iteration, false); yield break; }
            double[] s = trial.Select((v, i) => v - x[i]).ToArray(), y = next.Gradient.Select((v, i) => v - current.Gradient[i]).ToArray();
            double sy = Dot(s, y);
            if (sy > 1e-10 * Math.Sqrt(Dot(s, s) * Dot(y, y)))
            {
                if (history.Count == 8) history.RemoveAt(0);
                history.Add((s, y, 1 / sy));
            }
            x = trial; current = next;
            if (s.Max(Math.Abs) < 1e-5) { state.Result = Finish(iteration + 1, true); yield break; }
        }
        state.Result = Finish(maximumIterations, false);
    }
    static double Dot(double[] a, double[] b) { double sum = 0; for (int i = 0; i < a.Length; i++) sum += a[i] * b[i]; return sum; }
}
