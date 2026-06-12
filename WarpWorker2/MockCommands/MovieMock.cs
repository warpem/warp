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

        Console.WriteLine($"[MOCK] Processed CTF for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.MovieProcessMovement))]
    static void MockMovieProcessMovement(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieMovement Options = (ProcessingOptionsMovieMovement)Command.Content[1];
        int3 StackDims = OriginalStack?.Dims ?? new int3(64, 64, 1);
        Options.Dimensions = StackDims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.OptionsMovement = Options;
        // Fabricate trivial motion (zero shifts) so SaveMeta has something to write.
        M.GridMovementX = new CubicGrid(new int3(1), new float[] { 0 });
        M.GridMovementY = new CubicGrid(new int3(1), new float[] { 0 });
        Thread.Sleep(100 + Random.Shared.Next(50));
        M.SaveMeta();

        Console.WriteLine($"[MOCK] Processed movement for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.MovieExportMovie))]
    static void MockMovieExportMovie(NamedSerializableObject Command)
    {
        // No-op in mock mode: we have no real stack to write.
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped export for {Path}");
    }

    [MockCommand(nameof(WorkerWrapper.MovieCreateThumbnail))]
    static void MockMovieCreateThumbnail(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        Console.WriteLine($"[MOCK] Skipped thumbnail for {Path}");
    }
}
