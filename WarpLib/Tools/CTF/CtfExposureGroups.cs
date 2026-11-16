using System;

namespace Warp.Tools;

/// <summary>
/// Frames for CTF power spectra, averaged in short aligned exposure groups. Summing every frame's own spectrum adds
/// power incoherently, so Thon rings barely rise above each frame's shot noise; the spectrum of the whole movie's
/// average is coherent but blurred by all the motion the exposure accumulates, which costs the high-resolution rings
/// and the ice ring. Averaging aligned frames over about 3 e⁻/Å², then summing the groups' spectra, keeps both: the
/// signal adds coherently within a group, and a group is too short to move much. Frames need aligning because a
/// group averages complex images, not shift-invariant amplitudes.
/// </summary>
public static class CtfExposureGroups
{
    /// <summary>Exposure per group the partition aims for, in e⁻/Å².</summary>
    public const decimal DefaultExposure = 3;

    /// <summary>
    /// Consecutive frame ranges [Start, End) splitting <paramref name="frames"/> into the integer number of groups whose
    /// exposure is closest to <paramref name="target"/> (never more groups than frames). Without a known exposure,
    /// every frame is its own group, which is the per-frame spectrum.
    /// </summary>
    public static (int Start, int End)[] Partition(int frames, double totalExposure, double target)
    {
        if (frames < 1) throw new ArgumentOutOfRangeException(nameof(frames));
        int groups = frames;
        if (totalExposure > 0 && target > 0)
        {
            double ideal = totalExposure / target;
            int fewer = Math.Max(1, (int)Math.Floor(ideal)), more = Math.Max(1, (int)Math.Ceiling(ideal));
            groups = Math.Abs(totalExposure / fewer - target) <= Math.Abs(totalExposure / more - target) ? fewer : more;
            groups = Math.Min(groups, frames);
        }
        var ranges = new (int Start, int End)[groups];
        for (int g = 0; g < groups; g++)
            ranges[g] = ((int)((long)g * frames / groups), (int)((long)(g + 1) * frames / groups));
        return ranges;
    }

    /// <summary>
    /// The mean of each group's frames after shifting them into register. <paramref name="warps"/> are per-frame shifts in
    /// pixels, in the convention of the warp <see cref="Movie.ExportMovie"/> applies (the motion model's shift divided by
    /// the pixel size), so groups line up exactly as the exported average does. Fourier shifts, so no interpolation blur.
    /// </summary>
    /// <remarks>
    /// Identical to <see cref="AverageFrameByFrame"/>: every frame is shifted by the same transforms and averaged with the
    /// same arithmetic, but read in place on the device and accumulated as it goes, with the transform plans and
    /// buffers made once instead of per group (and no copy of the group's frames).
    /// </remarks>
    public static Image Average(Image stack, float2[] warps, (int Start, int End)[] groups)
    {
        if (warps.Length < stack.Dims.Z) throw new ArgumentException("One shift per frame is needed.", nameof(warps));
        foreach (var group in groups)
            if (group.End - group.Start < 1 || group.Start < 0 || group.End > stack.Dims.Z) throw new ArgumentException("Group outside the stack.", nameof(groups));
        int3 slice = stack.Dims.Slice();
        var result = new Image(IntPtr.Zero, new int3(slice.X, slice.Y, groups.Length)) { PixelSize = stack.PixelSize };
        // Same sign as the warp: pinned against WarpImage by GroupsAlignFramesLikeTheExportedAverage.
        var shifts = new float[stack.Dims.Z * 2];
        for (int z = 0; z < stack.Dims.Z; z++) { shifts[2 * z] = warps[z].X; shifts[2 * z + 1] = warps[z].Y; }
        int[] starts = new int[groups.Length], ends = new int[groups.Length];
        for (int g = 0; g < groups.Length; g++) (starts[g], ends[g]) = groups[g];
        CtfNative.Check(CtfNative.ExposureGroupAverage(stack.GetDevice(Intent.Read), slice.X, slice.Y, stack.Dims.Z, groups.Length, starts, ends, shifts,
            result.GetDevice(Intent.Write)), "average exposure groups");
        return result;
    }

    /// <summary>
    /// The reference for <see cref="Average"/>: each group's frames copied out, shifted with ShiftSlicesMassive and
    /// averaged with ReduceMean.
    /// </summary>
    public static Image AverageFrameByFrame(Image stack, float2[] warps, (int Start, int End)[] groups)
    {
        if (warps.Length < stack.Dims.Z) throw new ArgumentException("One shift per frame is needed.", nameof(warps));
        int3 slice = stack.Dims.Slice();
        long elements = slice.Elements();
        var result = new Image(IntPtr.Zero, new int3(slice.X, slice.Y, groups.Length)) { PixelSize = stack.PixelSize };
        for (int index = 0; index < groups.Length; index++)
        {
            var group = groups[index];
            int count = group.End - group.Start;
            if (count < 1 || group.Start < 0 || group.End > stack.Dims.Z) throw new ArgumentException("Group outside the stack.", nameof(groups));
            using var part = new Image(IntPtr.Zero, new int3(slice.X, slice.Y, count));
            var shifts = new float3[count];
            for (int i = 0; i < count; i++)
            {
                GPU.CopyDeviceToDevice(stack.GetDeviceSlice(group.Start + i, Intent.Read), part.GetDeviceSlice(i, Intent.Write), elements);
                shifts[i] = new float3(warps[group.Start + i].X, warps[group.Start + i].Y, 0);
            }
            part.ShiftSlicesMassive(shifts);
            GPU.ReduceMean(part.GetDevice(Intent.Read), result.GetDeviceSlice(index, Intent.Write), (uint)elements, (uint)count, 1);
        }
        return result;
    }
}
