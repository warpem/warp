using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public void ProcessCTFSimultaneous(ProcessingOptionsMovieCTF options)
    {
        IsProcessing = true;
        Image[] images = null;
        var totalTimer = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            Directory.CreateDirectory(PowerSpectrumDir);
            LoadMovieData(options, out _, out images, false, out _, out _);
            double loadSeconds = totalTimer.Elapsed.TotalSeconds;
            var records = new List<CtfPowerSpectrum.Observation>();
            var geometry = new List<CtfFitGeometry>();
            var display = new float[NTilts][];
            int nd = NTilts, np = options.DoPhase ? Math.Max(1, NTilts / 3) : 1;
            var phaseDims = new int3(1, 1, np);
            float minDose = Dose.Min(), maxDose = Dose.Max();
            // Phase-plate evolution follows acquisition dose, not the angle-sorted file order.
            float3[] tiltPositions = Enumerable.Range(0, NTilts).Select(t => new float3(.5f, .5f,
                maxDose > minDose ? (Dose[t] - minDose) / (maxDose - minDose) : .5f)).ToArray();
            double[][] phaseWeights = CtfFitGeometry.GridWeights(phaseDims, tiltPositions);
            double[] initial = new double[nd + 2 + np + 3];
            Array.Fill(initial,(double)(options.ZMin+options.ZMax)*.5,0,nd);
            CtfSpectrumFit basisSource = null;
            double extractionSeconds = 0, searchSeconds = 0;
            var timer = System.Diagnostics.Stopwatch.StartNew();
            using var extractor = new CtfPowerSpectrum.Extractor(new int2(images[0].Dims), options);
            double setupSeconds = timer.Elapsed.TotalSeconds;
            for (int t = 0; t < NTilts; t++)
            {
                timer.Restart();
                var extraction = extractor.Extract(images[t], t, basisSource);
                basisSource ??= extraction.Observations[0].Spectrum;
                extractionSeconds += timer.Elapsed.TotalSeconds;
                images[t].FreeDevice();
                display[t] = extraction.Display;
                Matrix3 rotation = Matrix3.Euler(0, 0, -TiltAxisAngles[t] * Helper.ToRad) *
                    Matrix3.Euler(0, Angles[t] * (AreAnglesInverted ? -1 : 1) * Helper.ToRad, 0);
                var local = extraction.Observations.ToArray();
                for (int i = 0; i < local.Length; i++)
                {
                    double[] dw = new double[NTilts]; dw[t] = 1;
                    var g = new CtfFitGeometry(dw, phaseWeights[t],
                        (local[i].Position.X - .5) * images[t].Dims.X * (double)options.BinnedPixelSizeMean * 1e-4,
                        (local[i].Position.Y - .5) * images[t].Dims.Y * (double)options.BinnedPixelSizeMean * 1e-4, rotation, options.Window * (double)options.BinnedPixelSizeMean * 1e-4);
                    geometry.Add(g);
                }
                records.AddRange(local);
            }
            var allRecords = records.ToArray(); var allGeometry = geometry.ToArray();
            timer.Restart();
            var fit = CtfFitEngine.Refine(allRecords, allGeometry, initial, options);
            CTFFitReliability=fit.Reliability??Array.Empty<CtfFitReliability.Curve>();
            var limits=CTFFitReliability.Select(c=>c.HalfWeightResolution).Where(r=>r>0).ToArray();
            if(limits.Length>0)Console.WriteLine($"CTF reliability: {limits.Length}/{NTilts} tilts with sustained spatial support; half-weight resolution {limits.Min():F1}–{limits.Max():F1} Å.");
            searchSeconds=fit.PlaneInitializationSeconds;
            double refinementSeconds = timer.Elapsed.TotalSeconds-searchSeconds;
            if(fit.PlaneAtBoundary) Console.WriteLine("CTF specimen inclination reached the grazing-incidence model boundary; interpret the fitted plane with caution.");
            timer.Restart();
            var p = fit.Parameters;
            CTFSpecimenThicknessAngstrom = (decimal)(Math.Sqrt(p[^1])*1e4);
            GridCTFDefocus = new CubicGrid(new int3(1, 1, NTilts), p.Take(nd).Select(v => (float)v).ToArray());
            float delta = (float)(2 * Math.Sqrt(p[nd] * p[nd] + p[nd + 1] * p[nd + 1]));
            float angle = (float)(.5 * Math.Atan2(p[nd + 1], p[nd]) * 180 / Math.PI);
            GridCTFDefocusDelta = new CubicGrid(new int3(1, 1, NTilts), Enumerable.Repeat(delta, NTilts).ToArray());
            GridCTFDefocusAngle = new CubicGrid(new int3(1, 1, NTilts), Enumerable.Repeat(angle, NTilts).ToArray());
            // Downstream tomography code indexes these grids directly by tilt, so expand the fitted phase spline.
            float[] tiltPhase = phaseWeights.Select(w => (float)(w.Select((v, j) => v * p[nd + 2 + j]).Sum() / Math.PI)).ToArray();
            GridCTFPhase = new CubicGrid(new int3(1, 1, NTilts), tiltPhase);
            PlaneNormal = new float3((float)p[nd + 2 + np], (float)p[nd + 3 + np], 1);
            PlaneNormal /= PlaneNormal.Length();
            CTF = CtfFitEngine.MakeCtf(options, p.Take(nd).Average(), p[nd], p[nd + 1], tiltPhase.Average() * Math.PI);
            TiltPS1D = new ObservableCollection<float2[]>();
            TiltCTFQuality = new ObservableCollection<float2[]>();
            TiltSimulatedBackground = new ObservableCollection<Cubic1D>();
            TiltSimulatedScale = new ObservableCollection<Cubic1D>();
            var references = Enumerable.Range(0, NTilts).Select(t =>
            {
                double phase = 0; for (int j = 0; j < np; j++) phase += p[nd + 2 + j] * phaseWeights[t][j];
                return CtfFitEngine.MakeCtf(options, p[t], p[nd], p[nd + 1], phase);
            }).ToArray();
            using var diagnosticExtractor = new CtfPowerSpectrum.Extractor(new int2(images[0].Dims), options, fullSpectrum: true);
            var diagnosticInputs = Enumerable.Range(0, NTilts).Select(t =>
                new CtfFitDiagnostics.Input(images[t], allGeometry.Where((_, i) => allRecords[i].Group == t).ToArray(), t));
            var diagnostics = CtfFitDiagnostics.CreateFullSpectrum(diagnosticExtractor, diagnosticInputs.ToArray(), fit,
                references, CTF, options.Window, display);
            foreach (var diagnostic in diagnostics.Groups)
            {
                TiltCTFQuality.Add(diagnostic.Quality);
                TiltPS1D.Add(diagnostic.Spectrum); TiltSimulatedBackground.Add(diagnostic.Background); TiltSimulatedScale.Add(diagnostic.Envelope);
            }
            var global = diagnostics.Global;
            PS1D = global.Spectrum; SimulatedBackground = global.Background; SimulatedScale = global.Envelope;
            CTFResolutionEstimate = global.Resolution;
            CTFQuality = global.Quality;
            double diagnosticSeconds = timer.Elapsed.TotalSeconds;
            timer.Restart();
            using (var image = new Image(display, new int3(options.Window, options.Window / 2, NTilts))) image.WriteMRC(PowerSpectrumPath, true);
            Simulated1D = GetSimulated1D();
            OptionsCTF = options;
            SaveMeta();
            Console.WriteLine($"CTF: load {loadSeconds:F2}s, setup {setupSeconds:F2}s, spectra {extractionSeconds:F2}s, global search {searchSeconds:F2}s, refinement {refinementSeconds:F2}s ({fit.Evaluations} evaluations), diagnostics {diagnosticSeconds:F2}s, output {timer.Elapsed.TotalSeconds:F2}s, total {totalTimer.Elapsed.TotalSeconds:F2}s");
        }
        finally
        {
            // LoadMovieData lends its per-device cache; keep host buffers alive for the next item.
            if (images != null) foreach (var image in images) image?.FreeDevice();
            IsProcessing = false;
        }
        TiltCTFProcessed?.Invoke();
    }
}
