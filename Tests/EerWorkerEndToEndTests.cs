using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Xml;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Xunit;
using Xunit.Abstractions;

namespace Tests;

/// <summary>
/// The real worker on real EER movies, not part of the regular suite: WARP_EER_E2E names a folder with movies/ and the
/// historical Warp XMLs in warp_metadata/ (EMPIAR-11366's test subset). Each movie is loaded, aligned, CTF-fitted and
/// averaged as Relay asks for it, once as the worker renders EER and once on the 16K grid itself (the reference the
/// renderer approximates), and the fits must agree. The historical fits are reported, not compared: they come out
/// twice our defocus for every movie, the 16K reference included, so their processing differed in some convention.
/// </summary>
public class EerWorkerEndToEndTests : IDisposable
{
    readonly ITestOutputHelper output;
    readonly string root = Path.Combine(Path.GetTempPath(), "warp-eer-e2e-" + Guid.NewGuid().ToString("N"));
    public EerWorkerEndToEndTests(ITestOutputHelper output) { this.output = output; Directory.CreateDirectory(root); }
    public void Dispose() { try { Directory.Delete(root, true); } catch (IOException) { } }

    public sealed class EerEndToEndFactAttribute : FactAttribute
    {
        public EerEndToEndFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WARP_EER_E2E")))
                Skip = "Set WARP_EER_E2E to a folder with movies/ and warp_metadata/ on a CUDA host.";
        }
    }

    /// <summary>The worker executable with its assembly beside it: the build output, which Release builds put in Release/.</summary>
    static string Worker()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            foreach (string folder in new[] { "", "Release", "bin" })
                if (File.Exists(Path.Combine(dir.FullName, folder, "WarpWorker2.dll")) && File.Exists(Path.Combine(dir.FullName, folder, "WarpWorker2")))
                    return Path.Combine(dir.FullName, folder, "WarpWorker2");
        throw new FileNotFoundException("The WarpWorker2 executable was not found above the test folder.");
    }

    static double HistoricalDefocus(string xml)
    {
        var document = new XmlDocument();
        document.Load(xml);
        var node = document.SelectSingleNode("//CTF/Param[@Name='Defocus']") ?? throw new Exception($"{xml} has no defocus.");
        return double.Parse(node.Attributes["Value"].Value, System.Globalization.CultureInfo.InvariantCulture);
    }

    [EerEndToEndFact]
    public void MotionCtfAndAverageOfEerMovies()
    {
        string data = Environment.GetEnvironmentVariable("WARP_EER_E2E");
        string[] movies = Directory.GetFiles(Path.Combine(data, "movies"), "*.eer").OrderBy(p => p).Take(2).ToArray();

        // As Relay's WorkerOptions: 0.9 Å on the 4K grid (the historical 0.45 Å was the 8K grid's), to 1.8 Å.
        var options = new OptionsWarp();
        options.Import.PixelSize = 0.9M;
        options.Import.BinTimes = 1;
        options.Import.BinnedPixelSize = 1.8M;
        options.Import.EERGroupFrames = -40;
        options.Import.DosePerAngstromFrame = -30;
        options.Import.CorrectGain = false;
        options.Movement.RangeMin = 500; options.Movement.RangeMax = 8; options.Movement.Bfactor = -500;
        options.CTF.Window = 512; options.CTF.RangeMin = 30; options.CTF.RangeMax = 4; options.CTF.ZMin = 0.3M; options.CTF.ZMax = 8;
        options.CTF.Voltage = 300; options.CTF.Cs = 2.7M; options.CTF.Amplitude = 0.07M;
        options.Export.DoAverage = true; options.Export.DoStack = false;

        var approximate = Process(movies, options, "default", null);
        var reference = Process(movies, options, "16K", "1");
        for (int m = 0; m < movies.Length; m++)
        {
            double historical = HistoricalDefocus(Path.Combine(data, "warp_metadata", Path.GetFileNameWithoutExtension(movies[m]) + ".xml"));
            output.WriteLine($"{Path.GetFileName(movies[m])[..16]}…: defocus {approximate[m].Defocus:F3} µm, on the 16K grid {reference[m].Defocus:F3} (historical {historical:F3}); " +
                             $"resolution estimate {approximate[m].Resolution:F1} Å, on the 16K grid {reference[m].Resolution:F1}; average {approximate[m].Dims}");
        }
        for (int m = 0; m < movies.Length; m++)
        {
            Assert.Equal(new int3(2048, 2048, 1), approximate[m].Dims);
            Assert.InRange(approximate[m].Defocus, reference[m].Defocus * 0.995, reference[m].Defocus * 1.005);
            Assert.InRange(approximate[m].Resolution, reference[m].Resolution - 0.3, reference[m].Resolution + 0.3);
        }
    }

    /// <summary>Runs one worker over the movies (WARP_EER_KERNEL_WIDTH set to kernelWidth if given) and reads the fits.</summary>
    (double Defocus, double Resolution, int3 Dims)[] Process(string[] movies, OptionsWarp options, string name, string kernelWidth)
    {
        string folder = Path.Combine(root, name);
        Directory.CreateDirectory(folder);
        var layout = new QueueLayout(Path.Combine(folder, "queue"));
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var links = movies.Select(movie =>
        {
            string link = Path.Combine(folder, Path.GetFileName(movie));
            File.CreateSymbolicLink(link, movie);
            var export = options.GetProcessingMovieExport();
            export.DoAverage = true; export.DoDenoise = false; export.DoDenoiseDeconv = false; export.DoDeconv = false;
            var task = new TaskItem
            {
                TaskId = Path.GetFileNameWithoutExtension(movie)[..16], RequiresGpu = true,
                Init = [WorkerCommands.LoadGainRef("", false, false, false, "")],
                Main = [WorkerCommands.LoadStack(link, 0.5M, -40), WorkerCommands.MovieProcessMovement(link, options.GetProcessingMovieMovement()),
                        WorkerCommands.MovieProcessCTF(link, options.GetProcessingMovieCTF()), WorkerCommands.MovieExportMovie(link, export),
                        WorkerCommands.WaitAsyncTasks()]
            };
            task.ComputeInitFingerprint();
            queue.Enqueue(task);
            return (Link: link, Task: task.TaskId);
        }).ToArray();

        var info = new ProcessStartInfo { FileName = Worker(), UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (string arg in new[] { "--device", "0", "--queue-dir", layout.Root }) // exits once the queue is empty
            info.ArgumentList.Add(arg);
        if (kernelWidth != null)
            info.Environment["WARP_EER_KERNEL_WIDTH"] = kernelWidth;
        var timer = Stopwatch.StartNew();
        using (var process = System.Diagnostics.Process.Start(info)!)
        {
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            bool exited = process.WaitForExit(600_000);
            if (!exited) process.Kill(true);
            string logs = Path.Combine(layout.Root, "logs");
            foreach (string line in (Directory.Exists(logs) ? Directory.GetFiles(logs, "*.log").SelectMany(File.ReadLines) : Enumerable.Empty<string>())
                                    .Concat(stdout.Result.Split('\n')).Where(l => l.Contains("Rendered") || l.Contains("[time]")).Distinct())
                output.WriteLine($"{name}: {line.Trim()}");
            Assert.True(exited && process.ExitCode == 0, $"worker exit {(exited ? process.ExitCode : -1)}\n{stdout.Result[^Math.Min(4000, stdout.Result.Length)..]}\n{stderr.Result}");
        }
        output.WriteLine($"{name}: {links.Length} movies in {timer.Elapsed.TotalSeconds:F1} s, worker start included");

        return links.Select(l =>
        {
            string failed = Path.Combine(layout.Failed, l.Task + ".json");
            Assert.False(File.Exists(failed), File.Exists(failed) ? File.ReadAllText(failed) : "");
            Assert.True(File.Exists(Path.Combine(layout.Done, l.Task + ".json")));
            var processed = new Movie(l.Link);
            processed.LoadMeta();
            return ((double)processed.CTF.Defocus, (double)processed.CTFResolutionEstimate, MapHeader.ReadFromFile(processed.AveragePath).Dimensions);
        }).ToArray();
    }
}
