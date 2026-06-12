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

    [Command(nameof(WorkerWrapper.LoadBoxNet))]
    static void LoadBoxNet(NamedSerializableObject Command)
    {
        BoxNetModel?.Dispose();

        string Path = (string)Command.Content[0];
        int BoxSize = (int)Command.Content[1];
        int BatchSize = (int)Command.Content[2];

        BoxNetModel = new BoxNetTorch(new int2(BoxSize), new float[3], new[] { DeviceID }, BatchSize);
        BoxNetModel.Load(Path);

        Console.WriteLine($"BoxNet loaded: box={BoxSize}, batch={BatchSize}, path={Path}");
    }

    [Command(nameof(WorkerWrapper.DropBoxNet))]
    static void DropBoxNet(NamedSerializableObject Command)
    {
        BoxNetModel?.Dispose();
        BoxNetModel = null;
        Console.WriteLine("BoxNet dropped");
    }

    [Command(nameof(WorkerWrapper.MoviePickBoxNet))]
    static void MoviePickBoxNet(NamedSerializableObject Command)
    {
        if (BoxNetModel == null)
            throw new Exception("No BoxNet model loaded");

        string Path = (string)Command.Content[0];
        ProcessingOptionsBoxNet Options = (ProcessingOptionsBoxNet)Command.Content[1];

        Movie M = new Movie(Path);
        M.MatchBoxNet2(new[] { BoxNetModel }, Options, null);
        M.SaveMeta();

        Console.WriteLine($"Picked particles for {Path}");
    }

    [Command(nameof(WorkerWrapper.MovieExportParticles))]
    static void MovieExportParticles(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
        float2[] Coordinates = (float2[])Command.Content[2];

        Movie M = new Movie(Path);
        M.ExportParticles(OriginalStack, Coordinates, Options);

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }
}
