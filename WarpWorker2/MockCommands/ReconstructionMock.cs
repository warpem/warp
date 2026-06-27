using System;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

// Averaged reconstruction is not exercisable in mock mode: the map step does real
// GPU back-projection and the reduce step reads the Projector partials it would have
// produced. The mocks are log-only so a mock run doesn't fault on the missing GPU.
static partial class WorkerProcess
{
    [MockCommand(nameof(WorkerWrapper.InitReconstructions))]
    static void MockInitReconstructions(NamedSerializableObject Command)
    {
        Console.WriteLine("[MOCK] Skipped reconstruction init");
    }

    [MockCommand(nameof(WorkerWrapper.TomoAddToReconstructionAndSave))]
    static void MockTomoAddToReconstructionAndSave(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped reconstruction back-projection for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.TomoFinishReconstruction))]
    static void MockTomoFinishReconstruction(NamedSerializableObject Command)
    {
        Console.WriteLine("[MOCK] Skipped reconstruction reduce");
    }
}
