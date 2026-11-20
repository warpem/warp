using System;
using Warp;
using Warp.Headers;
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
        string junkMask = (string)Command.Content[3];
        var (gray, width, height) = MicrographPreview.RenderFile(source, maxSize, junkMask);
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
        float InnerRadius = (float)Command.Content[2];
        float Separation = (float)Command.Content[3];
        string JunkMask = (string)Command.Content[4];
        float JunkDistance = (float)Command.Content[5];
        string Output = BlobPicker.Match(Path, Radius, InnerRadius, Separation, JunkMask, JunkDistance);
        Console.WriteLine($"Picked blobs for {Path}: {Output}");
    }

    [Command(WorkerCommandNames.MovieExportParticles)]
    static void MovieExportParticles(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
        float2[] Coordinates = (float2[])Command.Content[2];
        // The frames' physical size, as the stack was loaded (EER frames are rendered at their own size).
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.ExportParticles(OriginalStack, Coordinates, Options, releaseStack: !StackResident);

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }

    /// <summary>
    /// Particles straight from the movie file: each frame is extracted from as soon as it is decoded and corrected, so
    /// no stack is held or copied back and forth (on the host, a 4K movie of 150 frames is 10 GB). The worker's stack
    /// is dropped first, as this command's frames don't go there. If an earlier command of this task already loaded
    /// the same file with the same settings (e.g. for motion correction), the particles come from that stack instead,
    /// as MovieExportParticles takes them, and the movie is not decoded again.
    /// </summary>
    [Command(WorkerCommandNames.MovieExportParticlesFromFile)]
    static void MovieExportParticlesFromFile(NamedSerializableObject Command)
    {
        string DataPath = (string)Command.Content[0];
        decimal ScaleFactor = (decimal)Command.Content[1];
        int EERGroupFrames = (int)Command.Content[2];
        bool CorrectGain = (bool)Command.Content[3];
        string Path = (string)Command.Content[4];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[5];
        float2[] Coordinates = (float2[])Command.Content[6];

        HeaderEER.GroupNFrames = EERGroupFrames;

        var Source = new StackSource(System.IO.Path.GetFullPath(DataPath), ScaleFactor, EERGroupFrames, CorrectGain, _currentTask?.TaskId);
        if (OriginalStack != null && Source.TaskId != null && Source == OriginalStackSource)
        {
            Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);
            new Movie(Path).ExportParticles(OriginalStack, Coordinates, Options, releaseStack: !StackResident);
            Console.WriteLine($"Exported {Coordinates.Length} particles for {Path} from the stack already loaded");
            return;
        }

        OriginalStack?.Dispose(); OriginalStack = null; OriginalStackOwner = ""; OriginalStackSource = null; StackResident = false;

        Movie M = new Movie(Path);
        MovieParticleExport Export = null;
        try
        {
            LoadAndPrepareStack(DataPath, ScaleFactor, CorrectGain,
                                frameSink: dims =>
                                {
                                    // The frames' physical size, as they are loaded (EER frames are rendered at their own size).
                                    Options.Dimensions = dims.MultXY((float)Options.BinnedPixelSizeMean);
                                    Export = new MovieParticleExport(M, Options, Coordinates, dims, StreamLanes);
                                    return Export.AddFrame;
                                },
                                // Storage is idle while this movie is decoded and extracted: read the next one meanwhile.
                                afterRead: () => MoviePrefetch.StartNext(PrefetchQueue, PrefetchStages, DataPath));

            Export.Finish();
        }
        finally
        {
            Export?.Dispose();
        }

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }
}
