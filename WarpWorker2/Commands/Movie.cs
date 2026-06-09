using System;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

static partial class WorkerProcess
{
    [Command(nameof(WorkerWrapper.MovieProcessCTF))]
    static void MovieProcessCTF(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.ProcessCTF(OriginalStack, Options);
        M.SaveMeta();

        Console.WriteLine($"Processed CTF for {Path}");
    }
}
