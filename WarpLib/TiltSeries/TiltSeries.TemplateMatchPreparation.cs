using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ZLinq;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    private double[] matchDetectorVariance;
    private double? matchDoseSlope;
    private double[] matchTiltScales;

    // Estimate the independent white level before binning; a binned Nyquist annulus may contain protein signal.
    private void PrepareMatchDetectorNoise(ProcessingOptionsTomoFullMatch options)
    {
        matchDoseSlope = null;
        matchTiltScales = null;
        decimal bin = options.BinTimes;
        try
        {
            options.BinTimes = 0;
            LoadMovieData(options, out _, out Image[] tilts, false, out _, out _);
            LoadMovieMasks(options, out Image[] masks);
            matchDetectorVariance = new double[NTilts];
            int box = Math.Min(256, tilts.Min(t => Math.Min(t.Dims.X, t.Dims.Y)) / 2 * 2);
            for (int t = 0; t < NTilts; t++)
            {
                if (!UseTilt[t]) { matchDetectorVariance[t] = 1; continue; }
                List<double> powers = new();
                for (int attempt = 0, kept = 0; attempt < 256 && kept < 16; attempt++)
                {
                    int x = (int)(((attempt + .5) * .6180339887498949 % 1) * (tilts[t].Dims.X - box + 1));
                    int y = (int)(((attempt + .5) * .4142135623730951 % 1) * (tilts[t].Dims.Y - box + 1));
                    if (!MatchPatchIsUsable(tilts[t], masks[t], x, y, box)) continue;
                    using Image patch = new(new int3(box, box, 1));
                    float[] data = patch.GetHost(Intent.Write)[0], source = tilts[t].GetHost(Intent.Read)[0];
                    for (int j = 0; j < box; j++) Array.Copy(source, (y + j) * tilts[t].Dims.X + x, data, j * box, box);
                    using Image ft = patch.AsFFT();
                    float[] f = ft.GetHost(Intent.Read)[0];
                    for (int j = 0; j < box; j++) for (int i = 1; i < box / 2; i++)
                        {
                            int ky = Math.Min(j, box - j); double r2 = (i * i + ky * ky) / (double)(box * box);
                            if (r2 < .16 || r2 > .25) continue;
                            int q = 2 * (j * (box / 2 + 1) + i);
                            powers.Add(((double)f[q] * f[q] + (double)f[q + 1] * f[q + 1]) / (box * box));
                        }
                    kept++;
                }
                if (powers.Count == 0) throw new InvalidDataException($"No unmasked patches for detector-noise estimation in tilt {t}.");
                matchDetectorVariance[t] = TemplateMatchStatistics.Quantile(powers, .5) / Math.Log(2);
                if (!(matchDetectorVariance[t] > 0)) throw new InvalidDataException($"Zero detector noise in tilt {t}.");
            }
            foreach (Image tilt in tilts) tilt.FreeDevice();
            foreach (Image mask in masks) mask?.FreeDevice();
        }
        finally { options.BinTimes = bin; }
    }

    // Defocus-binned real-space backprojection. The position grid uses Warp's complete local geometry;
    // interpolation avoids evaluating specimen/deformation splines at every voxel for every defocus bin.
    private Image ReconstructMatchVolume(ProcessingOptionsTomoFullMatch options, int3 dims, Func<float, string, bool> progress)
    {
        PrepareMatchDetectorNoise(options);
        LoadMovieData(options, out _, out Image[] tilts, false, out _, out _);
        LoadMovieMasks(options, out Image[] masks);
        float pixel = (float)options.BinnedPixelSizeMean;
        Image volume = new(dims);
        volume.Fill(0);
        const int spacing = 16;
        int3 grid = (dims + spacing - 1) / spacing + 1;
        using Image geometry = new(new int3(grid.X * 3, grid.Y, grid.Z));
        float3[] positions = new float3[grid.Elements()];
        for (int z = 0; z < grid.Z; z++) for (int y = 0; y < grid.Y; y++) for (int x = 0; x < grid.X; x++)
                    positions[(z * grid.Y + y) * grid.X + x] = new float3(x, y, z) * (spacing * pixel);
        try
        {
            for (int t = 0; t < NTilts; t++)
            {
                if (!UseTilt[t]) continue;
                MatchProgress(progress, (float)t / NTilts, $"Matched-filter reconstruction: tilt {t + 1}/{NTilts}");
                // Borrowed movie cache: edits are refreshed by the next LoadMovieData call.
                EraseDirt(tilts[t], masks[t]);
                if (!options.DontInvert) tilts[t].Multiply(-1);
                tilts[t].SubtractMeanGrid(new int2(1));
                using Image ft = tilts[t].AsFFT();
                int2 im = new(tilts[t].Dims);
                using Image coords = CTF.GetCTFCoords(im, im);
                using Image transfer = new(new int3(im), true);
                using Image filtered = new(new int3(im), true, true);
                float3[] projected = GetPositionsInOneTilt(positions, t);
                float[][] g = geometry.GetHost(Intent.Write);
                for (int i = 0; i < projected.Length; i++)
                {
                    int z = i / (grid.X * grid.Y), j = 3 * (i % (grid.X * grid.Y));
                    g[z][j] = projected[i].X / pixel; g[z][j + 1] = projected[i].Y / pixel; g[z][j + 2] = projected[i].Z;
                }
                // <=200 A spacing, with linear interpolation in defocus between filtered images.
                const float step = .02f; // micrometers
                int lo = (int)Math.Floor(projected.Min(p => p.Z) / step), hi = (int)Math.Ceiling(projected.Max(p => p.Z) / step);
                for (int k = lo; k <= hi; k++)
                {
                    CTF parameters = GetCTFParamsForOneTilt(pixel, [k * step], [VolumeDimensionsPhysical / 2], t, weighted: true)[0];
                    parameters.Scale /= (decimal)(matchDetectorVariance[t] * Math.Pow((double)options.PixelSizeMean / pixel, 2));
                    GPU.CreateCTF(transfer.GetDevice(Intent.Write), coords.GetDevice(Intent.Read), IntPtr.Zero,
                        (uint)transfer.ElementsReal, [parameters.ToStruct()], false, 1);
                    GPU.CopyDeviceToDevice(ft.GetDevice(Intent.Read), filtered.GetDevice(Intent.Write), ft.ElementsReal);
                    filtered.Multiply(transfer);
                    using Image image = filtered.AsIFFT(normalize: true, preserveSelf: true);
                    GPU.MatchBackproject(image.GetDevice(Intent.Read), im, volume.GetDevice(Intent.ReadWrite), dims,
                        geometry.GetDevice(Intent.Read), grid, spacing, k * step, step);
                }
                tilts[t].FreeDevice(); masks[t]?.FreeDevice();
            }
            return volume;
        }
        catch { volume.Dispose(); throw; }
    }

    // Blackman–Tukey PSD: taper the autocorrelation, retaining directional multiplicity and clutter power.
    private static Image WhitenMatchVolume(ref Image volume, float pixel)
    {
        using Image ft = volume.AsFFT(true);
        using Image power = ft.AsAmplitudes();
        power.Multiply(power);
        using Image complexPower = new(volume.Dims, true, true);
        complexPower.Fill(new float2(1, 0)); complexPower.Multiply(power);
        using Image acf = complexPower.AsIFFT(true, normalize: true);
        GPU.MatchAutocorrelationWindow(acf.GetDevice(Intent.ReadWrite), acf.Dims, pixel, 130);
        using Image smoothFT = acf.AsFFT(true);
        Image rootPower = smoothFT.AsReal();
        float maximum = rootPower.GetHost(Intent.Read).Max(s => s.Max());
        rootPower.Max(Math.Max(1e-30f, maximum * 1e-8f)); rootPower.Sqrt();
        ft.Divide(rootPower);
        Image whitened = ft.AsIFFT(true, normalize: true);
        volume.Dispose(); volume = whitened;
        return rootPower;
    }
}
