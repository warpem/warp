using System;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

static partial class WorkerProcess
{
    // Mock TomoStack: the real StackTilts does heavy GPU/I-O to write a .st stack.
    // In mock mode just touch the metadata so downstream collection has something to
    // load, and skip the actual stacking — no GPU required.
    [MockCommand(nameof(WorkerWrapper.TomoStack))]
    static void MockTomoStack(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];

        TiltSeries T = new TiltSeries(Path);
        T.SaveMeta();

        Console.WriteLine($"[MOCK] Skipped tilt stack for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.TomoReconstruct))]
    static void MockTomoReconstruct(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped full reconstruction for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.TomoAutoLevel))]
    static void MockTomoAutoLevel(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped auto-leveling for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.LoadTomoDenoiser))]
    static void MockLoadTomoDenoiser(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped denoiser load from {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.TomoDenoise))]
    static void MockTomoDenoise(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped denoising for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.TomoPeakAlign))]
    static void MockTomoPeakAlign(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped peak alignment for {Path}");
    }

    // Touch the metadata so the orchestrator's onSuccess hook (which builds the 3D
    // output STAR table from the loaded tilt-series meta) has something to load.
    [MockCommand(nameof(WorkerWrapper.TomoExportParticleSubtomos))]
    static void MockTomoExportParticleSubtomos(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];

        TiltSeries T = new TiltSeries(Path);
        T.SaveMeta();

        Console.WriteLine($"[MOCK] Skipped subtomo export for {Path}");
    }

    // 2D export is not exercised in mock mode: the real worker writes the per-series
    // temp STAR file that onSuccess reads back, which the mock does not produce.
    [MockCommand(nameof(WorkerWrapper.TomoExportParticleSeries))]
    static void MockTomoExportParticleSeries(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped particle-series export for {Path}");
    }
}
