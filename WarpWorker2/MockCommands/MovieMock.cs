using System;
using System.Threading;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

static partial class WorkerProcess
{
    // Mock LoadStack: the real LoadStack does heavy GPU I/O, so in mock mode it would
    // be a no-op and leave OriginalStack null. Allocate a tiny host-only placeholder
    // instead so downstream mock handlers (e.g. MockMovieProcessCTF, which reads
    // OriginalStack.Dims) have a valid stack to work with — no GPU required.
    [MockCommand(nameof(WorkerWrapper.LoadStack))]
    static void MockLoadStack(NamedSerializableObject Command)
    {
        OriginalStack?.Dispose();
        OriginalStack = new Image(new[] { new float[64 * 64] }, new int3(64, 64, 1));

        Console.WriteLine($"[MOCK] Loaded placeholder stack {OriginalStack.Dims}");
    }

    [MockCommand(nameof(WorkerWrapper.MovieProcessCTF))]
    static void MockMovieProcessCTF(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];
        // Tolerate a missing stack (e.g. a task without a LoadStack init/main step):
        // fall back to a nominal size rather than NPE.
        int3 StackDims = OriginalStack?.Dims ?? new int3(64, 64, 1);
        Options.Dimensions = StackDims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.OptionsCTF = Options;
        M.CTF = new CTF
        {
            Amplitude = Options.Amplitude,
            Defocus = (decimal)(Random.Shared.NextDouble() * 3 + 0.5),
            DefocusDelta = (decimal)(Random.Shared.NextDouble() * 0.1),
            DefocusAngle = (decimal)(Random.Shared.NextDouble() * Math.PI),
            Cs = Options.Cs,
            Voltage = Options.Voltage,
        };
        Thread.Sleep(3000 + Random.Shared.Next(1000));
        M.SaveMeta();

        Console.WriteLine($"Processed CTF for {Path}");
    }
}
