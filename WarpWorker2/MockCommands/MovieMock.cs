using System;
using System.Threading;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2;

static partial class WorkerProcess
{
    [MockCommand(nameof(WorkerWrapper.MovieProcessCTF))]
    static void MockMovieProcessCTF(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

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
