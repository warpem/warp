using System;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

static partial class WorkerProcess
{
    [Command(nameof(WorkerWrapper.TomoStack))]
    static void TomoStack(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoStack Options = (ProcessingOptionsTomoStack)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.StackTilts(Options);
        T.SaveMeta();

        Console.WriteLine($"Created tilt stack for {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoReconstruct))]
    static void TomoReconstruct(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoFullReconstruction)Command.Content[1];

        string LastMessage = "";
        TiltSeries T = new TiltSeries(Path);
        T.ReconstructFull(Options, (grid, gridElements, message) =>
        {
            if (message != LastMessage)
            {
                LastMessage = message;
                Console.WriteLine(message);
            }
            return false;
        });

        Console.WriteLine($"Reconstructed full tomogram for {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoAutoLevel))]
    static void TomoAutoLevel(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoAutoLevel Options = (ProcessingOptionsTomoAutoLevel)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.AutoLevel(Options);
        T.SaveMeta();

        Console.WriteLine($"Processed auto-leveling for {Path}");
    }

    [Command(nameof(WorkerWrapper.LoadTomoDenoiser))]
    static void LoadTomoDenoiser(NamedSerializableObject Command)
    {
        DenoiserModel?.Dispose();

        string Path = (string)Command.Content[0];
        int3 WindowSize = (int3)Command.Content[1];
        int BatchSize = (int)Command.Content[2];

        DenoiserModel = new NoiseNet3DTorch(WindowSize, new[] { DeviceID }, BatchSize);
        DenoiserModel.Load(Path);

        Console.WriteLine($"Denoiser model with window size = {WindowSize}, batch size = {BatchSize} loaded from {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoDenoise))]
    static void TomoDenoise(NamedSerializableObject Command)
    {
        if (DenoiserModel == null)
            throw new Exception("No denoiser model loaded");

        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoDenoise Options = (ProcessingOptionsTomoDenoise)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.Denoise(Options, DenoiserModel);

        Console.WriteLine($"Denoised {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoPeakAlign))]
    static void TomoPeakAlign(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoPeakAlign Options = (ProcessingOptionsTomoPeakAlign)Command.Content[1];
        var TemplatePath = (string)Command.Content[2];
        float3[] Positions = (float3[])Command.Content[3];
        float3[] Angles = (float3[])Command.Content[4];

        Image Template = Image.FromFile(TemplatePath);

        TiltSeries T = new TiltSeries(Path);
        T.PeakAlign(Options, Template, Positions, Angles);
        T.SaveMeta();

        Template.Dispose();

        Console.WriteLine($"Performed alignment to peaks for {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoExportParticleSubtomos))]
    static void TomoExportParticleSubtomos(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
        float3[] Coordinates = (float3[])Command.Content[2];
        float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;

        TiltSeries T = new TiltSeries(Path);
        T.ReconstructSubtomos(Options, Coordinates, Angles);
        T.SaveMeta();

        Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
    }

    [Command(nameof(WorkerWrapper.TomoExportParticleSeries))]
    static void TomoExportParticleSeries(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
        float3[] Coordinates = (float3[])Command.Content[2];
        float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;
        string PathsRelativeTo = (string)Command.Content[4];
        string PathTableOut = (string)Command.Content[5];

        TiltSeries T = new TiltSeries(Path);
        T.ReconstructParticleSeries(Options, Coordinates, Angles, PathsRelativeTo, out Star TableOut);
        T.SaveMeta();

        if (!string.IsNullOrEmpty(PathTableOut))
            TableOut.Save(PathTableOut);

        Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
    }
}
