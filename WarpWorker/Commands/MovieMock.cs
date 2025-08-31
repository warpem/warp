using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
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
        M.SaveMeta();

        Console.WriteLine($"Processed CTF for {Path}");
    }
                
    [MockCommand(nameof(WorkerWrapper.MovieProcessMovement))]
    static void MockMovieProcessMovement(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieMovement Options = (ProcessingOptionsMovieMovement)Command.Content[1];
        Options.Dimensions = OriginalStack.Dims.MultXY((float)Options.BinnedPixelSizeMean);

        Movie M = new Movie(Path);
        M.OptionsMovement = Options;
        M.GridMovementX = new CubicGrid(new(1, 1, 40), 
                                        Enumerable.Range(0, 40)
                                                  .Select(i => (float)(Random.Shared.NextDouble() * 6 - 3))
                                                  .ToArray());
        M.GridMovementY = new CubicGrid(new(1, 1, 40), 
                                        Enumerable.Range(0, 40)
                                                  .Select(i => (float)(Random.Shared.NextDouble() * 6 - 3))
                                                  .ToArray());
        M.GridLocalX = new CubicGrid(new(5, 5, 3),
                                     Enumerable.Range(0, 5 * 5 * 3)
                                               .Select(i => (float)(Random.Shared.NextDouble() * 2 - 1))
                                               .ToArray());
        M.GridLocalY = new CubicGrid(new(5, 5, 3),
                                     Enumerable.Range(0, 5 * 5 * 3)
                                               .Select(i => (float)(Random.Shared.NextDouble() * 2 - 1))
                                               .ToArray());
        M.SaveMeta();

        Console.WriteLine($"Processed movement for {Path}");
    }
                
    [MockCommand(nameof(WorkerWrapper.MoviePickBoxNet))]
    static void MockMoviePickBoxNet(NamedSerializableObject Command)
    {
        if (BoxNetModel == null)
            throw new Exception("No BoxNet model loaded");

        string Path = (string)Command.Content[0];
        ProcessingOptionsBoxNet Options = (ProcessingOptionsBoxNet)Command.Content[1];

        Movie M = new Movie(Path);
        M.OptionsBoxNet = Options;
        
        Star TableOut = new Star(new string[]
        {
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnMicrographName",
            "rlnAutopickFigureOfMerit"
        });
        for (int i = 0; i < 100; i++)
        {
            var x = (Random.Shared.NextDouble() * 4096).ToString(CultureInfo.InvariantCulture);
            var y = (Random.Shared.NextDouble() * 4096).ToString(CultureInfo.InvariantCulture);
            var score = Random.Shared.NextDouble().ToString(CultureInfo.InvariantCulture);
            TableOut.AddRow([x, y, M.RootName, score]);
        }
        Directory.CreateDirectory(M.MatchingDir);
        TableOut.Save(System.IO.Path.Combine(M.MatchingDir, M.RootName + "_boxnet.star"));
        
        M.SaveMeta();

        Console.WriteLine($"Picked particles for {Path}");
    }
                
    [MockCommand(nameof(WorkerWrapper.MovieExportMovie))]
    static void MockMovieExportMovie(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieExport Options = (ProcessingOptionsMovieExport)Command.Content[1];

        Movie M = new Movie(Path);
        M.OptionsMovieExport = Options;

        Directory.CreateDirectory(M.AverageDir);
        var Rng = new Random(Random.Shared.Next());
        Image Average = new Image([Enumerable.Range(0, 4096 * 4096).Select(i => Rng.NextSingle() * 40).ToArray()],
                                  new int3(4096, 4096, 1));
        Average.WriteMRC16b(M.AveragePath, (float)Options.PixelSize, true);
        
        M.SaveMeta();

        Console.WriteLine($"Exported movie for {Path}");
    }
                
    [MockCommand(nameof(WorkerWrapper.MovieCreateThumbnail))]
    static void MockMovieCreateThumbnail(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        int Size = (int)Command.Content[1];
        float Range = (float)Command.Content[2];

        Movie M = new Movie(Path);
        Directory.CreateDirectory(M.ThumbnailsDir);
        var Rng = new Random(Random.Shared.Next());
        Image Thumb = new Image([Enumerable.Range(0, 128 * 128).Select(i => Rng.NextSingle() * 255).ToArray()],
                                  new int3(128, 128, 1));
        Thumb.WritePNG(M.ThumbnailsPath);

        Console.WriteLine($"Exported movie for {Path}");
    }
    
    [MockCommand(nameof(WorkerWrapper.MovieExportParticles))]
    static void MockMovieExportParticles(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
        float2[] Coordinates = (float2[])Command.Content[2];

        Movie M = new Movie(Path);
        M.OptionsParticlesExport = Options;

        Directory.CreateDirectory(M.ParticlesDir);
        var Rng = new RandomNormal(Random.Shared.Next());
        Image Stack = new Image([Enumerable.Range(0, 100 * 64 * 64).Select(i => Rng.NextSingle(0, 1)).ToArray()],
                                  new int3(64, 64, 100));
        Stack.WriteMRC16b(System.IO.Path.Combine(M.ParticlesDir, M.RootName + Options.Suffix + ".mrcs"), (float)Options.PixelSize, true);
        
        M.SaveMeta();

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }
}