using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    abstract class BaseCommand
    {
        public virtual async Task Run(object options)
        {
            {
                var Attributes = options.GetType()
                    .GetCustomAttributes(typeof(CommandLine.VerbAttribute), false);
                if (Attributes.Length > 0)
                {
                    var Option = (CommandLine.VerbAttribute)Attributes[0];
                    Console.WriteLine($"Running command {Option.Name} with:");
                }
            }

            var Type = options.GetType();

            foreach (var field in Type.GetProperties())
            {
                var Attributes =
                    field.GetCustomAttributes(typeof(CommandLine.OptionAttribute),
                        false);
                if (Attributes.Length > 0)
                {
                    var Option = (CommandLine.OptionAttribute)Attributes[0];
                    var NameString = Option.LongName;

                    var Value = field.GetValue(options);
                    string ValueString;
                    if (Value is Array arr)
                    {
                        var FormattedItems = arr.Cast<object>()
                            .Select(item => item?.ToString() ?? "null");
                        ValueString = $"{{ {string.Join(", ", FormattedItems)} }}";
                    }
                    else
                    {
                        ValueString = Value?.ToString() ?? "null";
                    }

                    Console.WriteLine($"{NameString} = {ValueString}");
                }
            }

            Console.WriteLine("");
        }

        internal void IterateOverItems<T>(
            WorkerWrapper[] workers,
            BaseOptions cli,
            Action<WorkerWrapper, T> body,
            Func<int, int, T> getBatch = null,
            Func<T, int> getBatchSize = null,
            int oversubscribe = 1
        ) where T : class
        {
            // Default implementations for single item processing
            getBatch ??= (start, _) => (T)(object)cli.InputSeries[start];
            getBatchSize ??= _ => 1;

            string logDirectory = Path.Combine(cli.OutputProcessing, "logs");
            Directory.CreateDirectory(logDirectory);

            var jsonFilePath = Path.Combine(cli.OutputProcessing, "processed_items.json");
            List<Task> jsonTasks = new();
            List<Movie> processedItems = new List<Movie>();

            foreach (var item in cli.InputSeries)
                item.ProcessingStatus = ProcessingStatus.Unprocessed;

            Console.Write($"0/{cli.InputSeries.Length}");

            int nDone = 0;
            int nFailed = 0;
            Queue<long> processingTimes = new Queue<long>();
            Stopwatch timerOverall = Stopwatch.StartNew();

            if (typeof(T) == typeof(Movie))
            {
                // Single item processing using ForCPUGreedy
                Helper.ForCPUGreedy(0, cli.InputSeries.Length, workers.Length * oversubscribe, null, (iitem, threadID) =>
                {
                    Stopwatch timer = Stopwatch.StartNew();

                    WorkerWrapper processor = workers[threadID % workers.Length];
                    T item = getBatch(iitem, iitem + 1);
                    var movie = item as Movie;

                    if (Path.GetFullPath(cli.OutputProcessing) != Path.GetFullPath(Path.GetDirectoryName(movie.DataPath)))
                    {
                        if (string.IsNullOrEmpty(movie.DataDirectoryName))
                            movie.DataDirectoryName = Path.GetDirectoryName(movie.Path);

                        movie.Path = Path.Combine(cli.OutputProcessing, Path.GetFileName(movie.Path));
                        movie.SaveMeta();
                    }

                    processor.Console.Clear();
                    processor.Console.SetFileOutput(Path.Combine(logDirectory, $"{movie.RootName}.log"));

                    try
                    {
                        body(processor, item);
                        movie.ProcessingStatus = ProcessingStatus.Processed;
                    }
                    catch
                    {
                        movie.UnselectManual = true;
                        movie.ProcessingStatus = ProcessingStatus.LeaveOut;
                        movie.SaveMeta();

                        lock(workers)
                        {
                            VirtualConsole.ClearLastLine();
                            Console.Error.WriteLine($"Failed to process {movie.Path}, marked as unselected");
                            Console.Error.WriteLine($"Check logs in {logDirectory} for more info.");
                            Console.Error.WriteLine("Use the change_selection WarpTool to reactivate this item if required.");
                            nFailed++;
                        }
                    }
                    finally
                    {
                        processor.Console.SetFileOutput("");

                        jsonTasks.Add(Task.Run(() =>
                        {
                            List<Movie> immutableProcessed;
                            lock(workers)
                            {
                                processedItems.Add(movie);
                                immutableProcessed = processedItems.ToList();
                            }

                            JsonArray itemsJson = new JsonArray(immutableProcessed
                                .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                                .ToArray());
                            File.WriteAllText(jsonFilePath + $".{iitem}",
                                itemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

                            bool success = false;
                            Stopwatch watch = Stopwatch.StartNew();
                            while(!success && watch.ElapsedMilliseconds < 10_000)
                            {
                                try
                                {
                                    lock(workers)
                                        File.Move(jsonFilePath + $".{iitem}", jsonFilePath, true);
                                    success = true;
                                }
                                catch
                                {
                                }
                            }
                        }));

                        timer.Stop();

                        lock(workers)
                        {
                            nDone++;
                            processingTimes.Enqueue(timer.ElapsedMilliseconds);
                            if (processingTimes.Count > 20)
                                processingTimes.Dequeue();

                            long averageTime = (long)Math.Max(1, processingTimes.Average() / (workers.Length * oversubscribe));
                            long remainingTime = (cli.InputSeries.Length - nDone) * averageTime;
                            TimeSpan remainingTimeSpan = TimeSpan.FromMilliseconds(remainingTime);

                            string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                            string timeString = remainingTimeSpan.ToString(
                                (int)remainingTimeSpan.TotalDays > 0
                                    ? @"dd\.hh\:mm\:ss"
                                    : ((int)remainingTimeSpan.TotalHours > 0
                                        ? @"hh\:mm\:ss"
                                        : @"mm\:ss"));

                            VirtualConsole.ClearLastLine();
                            Console.Write($"{nDone}/{cli.InputSeries.Length}{failedString}, {timeString} remaining");
                        }
                    }
                }, null);
            }
            else
            {
                // Batch processing logic
                int nBatches = workers.Length * oversubscribe;
                int itemsPerBatch = (int)Math.Ceiling(cli.InputSeries.Length / (double)nBatches);
                List<Task> batchTasks = new List<Task>();

                for (int batchIndex = 0; batchIndex < nBatches; batchIndex++)
                {
                    int startIndex = batchIndex * itemsPerBatch;
                    if (startIndex >= cli.InputSeries.Length)
                        break;

                    int endIndex = Math.Min(startIndex + itemsPerBatch, cli.InputSeries.Length);
                    T batchOrItem = getBatch(startIndex, endIndex);
                    WorkerWrapper worker = workers[batchIndex % workers.Length];

                    Console.WriteLine($"Submitted {getBatchSize(batchOrItem)} items in batch {batchIndex} to worker process {batchIndex % workers.Length}");

                    batchTasks.Add(Task.Run(() =>
                    {
                        Stopwatch timer = Stopwatch.StartNew();

                        try
                        {
                            foreach (var movie in (Movie[])((object)batchOrItem))
                            {
                                if (Path.GetFullPath(cli.OutputProcessing) != Path.GetFullPath(Path.GetDirectoryName(movie.DataPath)))
                                {
                                    if (string.IsNullOrEmpty(movie.DataDirectoryName))
                                        movie.DataDirectoryName = Path.GetDirectoryName(movie.Path);

                                    movie.Path = Path.Combine(cli.OutputProcessing, Path.GetFileName(movie.Path));
                                    movie.SaveMeta();
                                }
                            }

                            var logFile = Path.Combine(logDirectory, $"batch{batchIndex}.log");
                            worker.Console.Clear();
                            worker.Console.SetFileOutput(logFile);

                            body(worker, batchOrItem);

                            foreach (var movie in (Movie[])((object)batchOrItem))
                                movie.ProcessingStatus = ProcessingStatus.Processed;
                        }
                        catch
                        {
                            foreach (var movie in (Movie[])((object)batchOrItem))
                            {
                                movie.UnselectManual = true;
                                movie.ProcessingStatus = ProcessingStatus.LeaveOut;
                                movie.SaveMeta();
                            }

                            lock(workers)
                            {
                                VirtualConsole.ClearLastLine();
                                Console.Error.WriteLine($"Failed to process batch {batchIndex}, marked as unselected");
                                Console.Error.WriteLine($"Check logs in {logDirectory} for more info.");
                                Console.Error.WriteLine("Use the change_selection WarpTool to reactivate these items if required.");
                                nFailed += getBatchSize(batchOrItem);
                            }
                        }
                        finally
                        {
                            worker.Console.SetFileOutput("");

                            jsonTasks.Add(Task.Run(() =>
                            {
                                List<Movie> immutableProcessed;
                                lock(workers)
                                {
                                    processedItems.AddRange((Movie[])((object)batchOrItem));
                                    immutableProcessed = processedItems.ToList();
                                }

                                JsonArray itemsJson = new JsonArray(immutableProcessed
                                    .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                                    .ToArray());
                                File.WriteAllText(jsonFilePath + $".{batchIndex}",
                                    itemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

                                bool success = false;
                                Stopwatch watch = Stopwatch.StartNew();
                                while(!success && watch.ElapsedMilliseconds < 10_000)
                                {
                                    try
                                    {
                                        lock(workers)
                                            File.Move(jsonFilePath + $".{batchIndex}", jsonFilePath, true);
                                        success = true;
                                    }
                                    catch
                                    {
                                    }
                                }
                            }));

                            timer.Stop();

                            lock(workers)
                            {
                                nDone += getBatchSize(batchOrItem);
                                processingTimes.Enqueue(timer.ElapsedMilliseconds);
                                if (processingTimes.Count > 20)
                                    processingTimes.Dequeue();

                                long averageTime = (long)Math.Max(1, processingTimes.Average() / nBatches);
                                long remainingTime = (cli.InputSeries.Length - nDone) * averageTime;
                                TimeSpan remainingTimeSpan = TimeSpan.FromMilliseconds(remainingTime);

                                string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                                string timeString = remainingTimeSpan.ToString(
                                    (int)remainingTimeSpan.TotalDays > 0
                                        ? @"dd\.hh\:mm\:ss"
                                        : ((int)remainingTimeSpan.TotalHours > 0
                                            ? @"hh\:mm\:ss"
                                            : @"mm\:ss"));

                                VirtualConsole.ClearLastLine();
                                Console.Write($"{nDone}/{cli.InputSeries.Length}{failedString}, {timeString} remaining");
                            }
                        }
                    }));
                }

                Task.WaitAll(batchTasks.ToArray());
            }

            timerOverall.Stop();

            Task.WaitAll(jsonTasks.ToArray());
            JsonArray finalItemsJson = new JsonArray(processedItems
                .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                .ToArray());
            File.WriteAllText(jsonFilePath,
                finalItemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

            Console.WriteLine(
                $"\nFinished processing in {TimeSpan.FromMilliseconds(timerOverall.ElapsedMilliseconds):hh\\:mm\\:ss}");
        }

        private void HandleMovieFailure(Movie movie)
        {
            movie.UnselectManual = true;
            movie.ProcessingStatus = ProcessingStatus.LeaveOut;
            movie.SaveMeta();
        }

        private void EnsureCorrectPaths(Movie movie, BaseOptions cli)
        {
            if (Path.GetFullPath(cli.OutputProcessing) !=
                Path.GetFullPath(Path.GetDirectoryName(movie.DataPath)))
            {
                if (string.IsNullOrEmpty(movie.DataDirectoryName))
                    movie.DataDirectoryName = Path.GetDirectoryName(movie.Path);

                movie.Path = Path.Combine(cli.OutputProcessing,
                    Path.GetFileName(movie.Path));
                movie.SaveMeta();
            }
        }
    }

    class CommandRunner : Attribute
    {
        public Type Type { get; set; }

        public CommandRunner(Type type)
        {
            Type = type;
        }
    }
}
