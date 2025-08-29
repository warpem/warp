using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
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

        Console.WriteLine($"Exported movie for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.MovieExportParticles))]
    static void MovieExportParticles(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsParticleExport Options = (ProcessingOptionsParticleExport)Command.Content[1];
        float2[] Coordinates = (float2[])Command.Content[2];

        Movie M = new Movie(Path);
        M.ExportParticles(OriginalStack, Coordinates, Options);
        M.SaveMeta();

        Console.WriteLine($"Exported {Coordinates.Length} particles for {Path}");
    }
                
    [Command(nameof(WorkerWrapper.TardisSegmentMembranes2D))]
    static void TardisSegmentMembranes2D(NamedSerializableObject Command)
    {
        string[] paths = Command.Content[0].ToString().Split(';');
        Console.WriteLine(string.Join(';', paths));
        ProcessingOptionsTardisSegmentMembranes2D options = (ProcessingOptionsTardisSegmentMembranes2D)Command.Content[1];

        Movie[] movies = paths.Select(p => new Movie(p)).ToArray();
        
        // Create a temporary directory
        string randomId = Path.GetRandomFileName().Replace(".", "");
        string tempDir = Path.Combine(movies.First().MembraneSegmentationDir, $"temp_{randomId}");
        Directory.CreateDirectory(tempDir);
        
        // downsample images to 15Apx
        string[] downsampledImagePaths = movies.Select(
            m => Path.Combine(tempDir, m.RootName + "_15.00Apx.mrc")
        ).ToArray();
        foreach (var (movie, outputPath) in movies.Zip(downsampledImagePaths))
        {
            // load average
            Image average = Image.FromFile(movie.AveragePath);
            
            // downsample to 15Apx
            float averagePixelSize = average.PixelSize;
            float targetPixelSize = 15;
            int2 dimsOut = (new int2(average.Dims * averagePixelSize / targetPixelSize) + 1) / 2 * 2;
            Image scaled = average.AsScaled(dimsOut);
            
            // write out downsampled image, force header pixel size to 15.00
            scaled.PixelSize = (float)15.00;
            scaled.WriteMRC(outputPath);
        }
        
        // run tardis in tempdir
        string Arguments = $"--path {tempDir} --output_format mrc_None --device {DeviceID} --patch_size 64";
        Console.WriteLine($"Executing tardis_mem2d in {tempDir} with arguments: {Arguments}");
        File.WriteAllText(Path.Combine(tempDir, "command.txt"), $"tardis_mem2d {Arguments}");
        
        Process Tardis = new Process
        {
            StartInfo =
            {
                FileName = "tardis_mem2d",
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Minimized,
                WorkingDirectory = tempDir,
                Arguments = Arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }
        };
        
        using (var stdout = File.CreateText(Path.Combine(tempDir, "run.out")))
        using (var stderr = File.CreateText(Path.Combine(tempDir, "run.err")))
        {
            Tardis.OutputDataReceived += (sender, args) => 
            {
                if (args.Data != null) stdout.WriteLine(args.Data);
            };

            Tardis.ErrorDataReceived += (sender, args) => 
            {
                if (args.Data != null) stderr.WriteLine(args.Data);
            };

            DataReceivedEventHandler toConsole = (sender, args) =>
            {
                if (args.Data != null) Console.WriteLine(args.Data);
            };
            Tardis.OutputDataReceived += toConsole;
            Tardis.ErrorDataReceived += toConsole;

            Tardis.Start();
            Tardis.BeginOutputReadLine();
            Tardis.BeginErrorReadLine();
            Tardis.WaitForExit();
        }
        
        // copy files to correct directory
        string[] membraneImageFiles = downsampledImagePaths.Select(
            p =>
            {
                var dir = Path.Combine(tempDir, "Predictions");
                var filename = Path.GetFileName(p).Replace(".mrc", "_semantic.mrc");
                return Path.Combine(dir, filename);
            }).ToArray();

        Directory.CreateDirectory(movies.First().MembraneSegmentationDir);
        foreach (var (movie, membraneImageFile) in movies.Zip(membraneImageFiles))
        {
            try
            {
                var destFile = Path.Combine(movie.MembraneSegmentationDir,
                    Path.GetFileName(membraneImageFile).Replace("_15.00Apx_semantic", ""));
                File.WriteAllText(Path.Combine(tempDir, Path.GetFileName(destFile)), $"{membraneImageFile}");
                File.Copy(membraneImageFile, destFile, overwrite: true);
            }
            catch (IOException ex)
            {
                Console.WriteLine($"Error occurred copying file {membraneImageFile}: {ex.Message}");
            }
        }
        
        // remove all files recursively from temp dir
        Directory.Delete(tempDir, recursive: true);
        Console.WriteLine($"Segmented membranes using TARDIS");
    }
}