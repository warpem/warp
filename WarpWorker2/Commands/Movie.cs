using System;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

static partial class WorkerProcess
{
    [Command(WorkerCommandNames.MovieProcessCTF)]
    static void MovieProcessCTF(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.ProcessCTF(OriginalStack, Options, releaseStack: !StackResident);
        M.SaveMeta();

        Console.WriteLine($"Processed CTF for {Path}");
    }

    [Command(WorkerCommandNames.MovieProcessMovement)]
    static void MovieProcessMovement(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieMovement Options = (ProcessingOptionsMovieMovement)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.ProcessShift(OriginalStack, Options, releaseStack: !StackResident);
        M.SaveMeta();

        Console.WriteLine($"Processed movement for {Path}");
    }

    [Command(WorkerCommandNames.MovieExportMovie)]
    static void MovieExportMovie(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieExport Options = (ProcessingOptionsMovieExport)Command.Content[1];

        Movie M = new Movie(Path);
        int first = Math.Max(0, Math.Min(OriginalStack.Dims.Z - 1, Options.SkipFirstN));
        int last = Math.Min(OriginalStack.Dims.Z, OriginalStack.Dims.Z - Options.SkipLastN);
        if (Options.DoDenoise && last - first < 2)
        {
            // A reprocessed one-frame input must not inherit old training halves.
            System.IO.File.Delete(M.AverageOddPath);
            System.IO.File.Delete(M.AverageEvenPath);
            System.IO.File.Delete(M.DenoiseTrainingOddPath);
            System.IO.File.Delete(M.DenoiseTrainingEvenPath);
        }
        M.ExportMovie(OriginalStack, Options, releaseStack: !StackResident);
        M.SaveMeta();

        Console.WriteLine($"Exported movie for {Path}");
    }

    [Command(WorkerCommandNames.MovieCreateThumbnail)]
    static void MovieCreateThumbnail(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        int Size = (int)Command.Content[1];
        float Range = (float)Command.Content[2];

        Movie M = new Movie(Path);
        M.CreateThumbnail(Size, Range);

        Console.WriteLine($"Created thumbnail for {Path}");
    }

    [Command(WorkerCommandNames.MicrographCreatePreview)]
    static void MicrographCreatePreview(NamedSerializableObject Command)
    {
        string source = (string)Command.Content[0], destination = (string)Command.Content[1];
        int maxSize = (int)Command.Content[2];
        using Image micrograph = Image.FromFile(source);
        var (gray, width, height) = MicrographPreview.Render(micrograph.GetHost(Intent.Read)[0], micrograph.Dims.X, micrograph.Dims.Y, maxSize);
        System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(destination)!);
        using Image preview = new Image(gray, new int3(width, height, 1));
        preview.WritePNG(destination);
        Console.WriteLine($"Created preview for {source}");
    }

    [Command(WorkerCommandNames.LoadBoxNet)]
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

    [Command(WorkerCommandNames.DropBoxNet)]
    static void DropBoxNet(NamedSerializableObject Command)
    {
        BoxNetModel?.Dispose();
        BoxNetModel = null;
        Console.WriteLine("BoxNet dropped");
    }

    [Command(WorkerCommandNames.MoviePickBoxNet)]
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

    [Command(WorkerCommandNames.MicrographPickBlobs)]
    static void MicrographPickBlobs(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        float Radius = (float)Command.Content[1];
        float InnerRadius = Command.Content.Length > 2 ? (float)Command.Content[2] : 0;
        float? Separation = Command.Content.Length > 3 ? (float)Command.Content[3] : null;
        string Output = BlobPicker.Match(Path, Radius, InnerRadius, Separation);
        Console.WriteLine($"Picked blobs for {Path}: {Output}");
    }

    [Command(WorkerCommandNames.MovieExportParticles)]
    static void MovieExportParticles(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
        float2[] Coordinates = (float2[])Command.Content[2];

        Movie M = new Movie(Path);
        M.ExportParticles(OriginalStack, Coordinates, Options, releaseStack: !StackResident);

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }
}
