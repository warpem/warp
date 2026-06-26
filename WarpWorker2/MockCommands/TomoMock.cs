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
}
