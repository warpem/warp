using System;
using System.Threading;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
{
    [MockCommand(nameof(WorkerWrapper.WaitAsyncTasks))]
    static void MockWaitAsyncTasks(NamedSerializableObject command)
    {
        Console.Write("Waiting for all async tasks to finish...");

        GlobalTasks.WaitAll();

        Console.WriteLine($" Done");
    }

    [MockCommand(nameof(WorkerWrapper.GcCollect))]
    static void MockGcCollect(NamedSerializableObject command)
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);

        Console.WriteLine("Garbage collection performed");
    }

    [MockCommand(nameof(WorkerWrapper.SetHeaderlessParams))]
    static void MockSetHeaderlessParams(NamedSerializableObject command)
    {
        HeaderlessDims = (int2)command.Content[0];
        HeaderlessOffset = (long)command.Content[1];
        HeaderlessType = (string)command.Content[2];

        Console.WriteLine($"Set headerless parameters to {HeaderlessDims}, {HeaderlessOffset}, {HeaderlessType}");
    }

    [MockCommand(nameof(WorkerWrapper.LoadGainRef))]
    static void MockLoadGainRef(NamedSerializableObject command)
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
            GainRef = new Image(new int3(4096, 4096, 1));
        }
        if (!string.IsNullOrEmpty(DefectsPath))
        {
            //DefectMap = LoadAndPrepareDefectMap(DefectsPath, FlipX, FlipY, Transpose);
        }
        
        Thread.Sleep(1000 + Random.Shared.Next(500));

        Console.WriteLine($"Loaded gain reference and defect map: {GainRef}, {FlipX}, {FlipY}, {Transpose}, {DefectsPath}");
    }

    [MockCommand(nameof(WorkerWrapper.LoadStack))]
    static void MockLoadStack(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        decimal ScaleFactor = (decimal)command.Content[1];
        int EERGroupFrames = (int)command.Content[2];
        bool CorrectGain = (bool)command.Content[3];

        HeaderEER.GroupNFrames = EERGroupFrames;

        if (OriginalStack == null)
            OriginalStack = new Image(new int3(4096, 4096, 40));

        Thread.Sleep(5000 + Random.Shared.Next(1000));
        
        OriginalStackOwner = Helper.PathToNameWithExtension(Path);

        Console.WriteLine($"Loaded stack: {OriginalStack}, {ScaleFactor}");
    }

    [MockCommand(nameof(WorkerWrapper.LoadBoxNet))]
    static void MockLoadBoxNet(NamedSerializableObject command)
    {
        BoxNetModel?.Dispose();

        string Path = (string)command.Content[0];
        int BoxSize = (int)command.Content[1];
        int BatchSize = (int)command.Content[2];

        Thread.Sleep(2000 + Random.Shared.Next(1000));

        Console.WriteLine($"Model with box size = {BoxSize}, batch size = {BatchSize} loaded from {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.DropBoxNet))]
    static void MockDropBoxNet(NamedSerializableObject command)
    {
        BoxNetModel?.Dispose();

        Console.WriteLine("Model dropped");
    }
}