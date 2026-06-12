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

    [Command(nameof(WorkerWrapper.MovieProcessMovement))]
    static void MovieProcessMovement(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieMovement Options = (ProcessingOptionsMovieMovement)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.ProcessShift(OriginalStack, Options);
        M.SaveMeta();

        Console.WriteLine($"Processed movement for {Path}");
    }

    [Command(nameof(WorkerWrapper.MovieExportMovie))]
    static void MovieExportMovie(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieExport Options = (ProcessingOptionsMovieExport)Command.Content[1];

        Movie M = new Movie(Path);
        M.ExportMovie(OriginalStack, Options);
        M.SaveMeta();

        Console.WriteLine($"Exported movie for {Path}");
    }

    [Command(nameof(WorkerWrapper.MovieCreateThumbnail))]
    static void MovieCreateThumbnail(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        int Size = (int)Command.Content[1];
        float Range = (float)Command.Content[2];

        Movie M = new Movie(Path);
        M.CreateThumbnail(Size, Range);

        Console.WriteLine($"Created thumbnail for {Path}");
    }
}
