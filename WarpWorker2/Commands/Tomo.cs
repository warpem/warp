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
}
