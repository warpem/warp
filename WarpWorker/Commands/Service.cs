using System;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
{
    [Command(nameof(WorkerWrapper.WaitAsyncTasks))]
    static void WaitAsyncTasks(NamedSerializableObject command)
    {
        Console.Write("Waiting for all async tasks to finish...");

        GlobalTasks.WaitAll();

        Console.WriteLine($" Done");
    }

    [Command(nameof(WorkerWrapper.GcCollect))]
    static void GcCollect(NamedSerializableObject command)
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);

        Console.WriteLine("Garbage collection performed");
    }

    [Command(nameof(WorkerWrapper.SetHeaderlessParams))]
    static void SetHeaderlessParams(NamedSerializableObject command)
    {
        HeaderlessDims = (int2)command.Content[0];
        HeaderlessOffset = (long)command.Content[1];
        HeaderlessType = (string)command.Content[2];

        Console.WriteLine($"Set headerless parameters to {HeaderlessDims}, {HeaderlessOffset}, {HeaderlessType}");
    }

    [Command(nameof(WorkerWrapper.LoadGainRef))]
    static void LoadGainRef(NamedSerializableObject command)
    {
        GainRef?.Dispose();
        DefectMap?.Dispose();

        string GainPath = (string)command.Content[0];
        bool FlipX = (bool)command.Content[1];
        bool FlipY = (bool)command.Content[2];
        bool Transpose = (bool)command.Content[3];
        string DefectsPath = (string)command.Content[4];

        if (!string.IsNullOrEmpty(GainPath))
        {
            GainRef = LoadAndPrepareGainReference(GainPath, FlipX, FlipY, Transpose);
        }
        if (!string.IsNullOrEmpty(DefectsPath))
        {
            DefectMap = LoadAndPrepareDefectMap(DefectsPath, FlipX, FlipY, Transpose);
        }

        Console.WriteLine($"Loaded gain reference and defect map: {GainRef}, {FlipX}, {FlipY}, {Transpose}, {DefectsPath}");
    }

    [Command(nameof(WorkerWrapper.LoadStack))]
    static void LoadStack(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        decimal ScaleFactor = (decimal)command.Content[1];
        int EERGroupFrames = (int)command.Content[2];
        bool CorrectGain = (bool)command.Content[3];

        HeaderEER.GroupNFrames = EERGroupFrames;

        OriginalStack = LoadAndPrepareStack(Path, ScaleFactor, CorrectGain);
        OriginalStackOwner = Helper.PathToNameWithExtension(Path);

        Console.WriteLine($"Loaded stack: {OriginalStack}, {ScaleFactor}");
    }

    [Command(nameof(WorkerWrapper.LoadBoxNet))]
    static void LoadBoxNet(NamedSerializableObject command)
    {
        BoxNetModel?.Dispose();

        string Path = (string)command.Content[0];
        int BoxSize = (int)command.Content[1];
        int BatchSize = (int)command.Content[2];

        BoxNetModel = new BoxNetTorch(new int2(BoxSize), new float[3], new[] { DeviceID }, BatchSize);
        BoxNetModel.Load(Path);

        Console.WriteLine($"Model with box size = {BoxSize}, batch size = {BatchSize} loaded from {Path}");
    }

    [Command(nameof(WorkerWrapper.DropBoxNet))]
    static void DropBoxNet(NamedSerializableObject command)
    {
        BoxNetModel?.Dispose();

        Console.WriteLine("Model dropped");
    }

    [Command("Exit")]
    static void ExitCommand(NamedSerializableObject command)
    {
        Exit();
    }
}