using System;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

// Multi-particle refinement is not exercisable in mock mode: every phase does real GPU
// work (denoising, refinement, reconstruction) and reads/writes resident state and
// on-disk partials. The mocks are log-only so a mock run doesn't fault on the missing GPU.
static partial class WorkerProcess
{
    [MockCommand(WorkerCommandNames.MPAPrepareSpecies)]
    static void MockMPAPrepareSpecies(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        Console.WriteLine($"[MOCK] Skipped pre-flight for {Path}");
    }

    [MockCommand(WorkerCommandNames.MPAPreparePopulation)]
    static void MockMPAPreparePopulation(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        Console.WriteLine($"[MOCK] Skipped population preparation for {Path}");
    }

    [MockCommand(WorkerCommandNames.MPARefineAndSave)]
    static void MockMPARefineAndSave(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        Console.WriteLine($"[MOCK] Skipped refinement for {Path}");
    }

    [MockCommand(WorkerCommandNames.MPAFinishSpecies)]
    static void MockMPAFinishSpecies(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        Console.WriteLine($"[MOCK] Skipped post-flight for {Path}");
    }
}
