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
            // default implementations of getBatch and getBatchSize
            // for single-item processing
            getBatch ??= (start, _) => (T)(object)cli.InputSeries[start];
            getBatchSize ??= _ => 1;

            // setup log dir
            string logDirectory = Path.Combine(cli.OutputProcessing, "logs");
            Directory.CreateDirectory(logDirectory);

            // setup json file for external progress tracking
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

            int nItems = cli.InputSeries.Length;
            List<Task> processingTasks = new List<Task>();

            bool isBatchProcessing = typeof(T).IsArray;

            if (isBatchProcessing)
            {
                int nBatches = workers.Length * oversubscribe;
                int itemsPerBatch = (int)Math.Ceiling(nItems / (double)nBatches);

                for (int batchIndex = 0; batchIndex < nBatches; batchIndex++)
                {
                    int startIndex = batchIndex * itemsPerBatch;
                    if (startIndex >= nItems)
                        break;

                    int endIndex = Math.Min(startIndex + itemsPerBatch, nItems);
                    T batchOrItem = getBatch(startIndex, endIndex);
                    WorkerWrapper worker = workers[batchIndex % workers.Length];

                    Console.WriteLine($"Submitted {getBatchSize(batchOrItem)} items in batch {batchIndex} to worker process {batchIndex % workers.Length}");

                    AddProcessingTask(batchIndex, startIndex, worker, batchOrItem);
                }
            }
            else
            {
                for (int itemIndex = 0; itemIndex < nItems; itemIndex++)
                {
                    T item = getBatch(itemIndex, itemIndex + 1);
                    WorkerWrapper worker = workers[itemIndex % workers.Length];
                    AddProcessingTask(itemIndex, itemIndex, worker, item);
                }
            }

            Task.WaitAll(processingTasks.ToArray());
            timerOverall.Stop();

            // Write out full Json one last time
            Task.WaitAll(jsonTasks.ToArray());
            JsonArray finalItemsJson = new JsonArray(processedItems
                .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                .ToArray());
            File.WriteAllText(jsonFilePath,
                finalItemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

            Console.WriteLine(
                $"\nFinished processing in {TimeSpan.FromMilliseconds(timerOverall.ElapsedMilliseconds):hh\\:mm\\:ss}");

            void AddProcessingTask(int taskIndex, int itemIndex, WorkerWrapper worker, T batchOrItem)
            {
                processingTasks.Add(Task.Run(() =>
                {
                    Stopwatch timer = Stopwatch.StartNew();

                    try
                    {
                        // Ensure correct paths for all movies in batch/single item
                        if (batchOrItem is Movie[] movies)
                        {
                            foreach (var movie in movies)
                                EnsureCorrectPaths(movie, cli);
                            var logFile = Path.Combine(logDirectory, $"batch{taskIndex}.log");
                            worker.Console.Clear();
                            worker.Console.SetFileOutput(logFile);
                        }
                        else if (batchOrItem is Movie movie)
                        {
                            EnsureCorrectPaths(movie, cli);
                            var logFile = Path.Combine(logDirectory, $"{movie.RootName}.log");
                            worker.Console.Clear();
                            worker.Console.SetFileOutput(logFile);
                        }

                        // Process the batch or single item
                        body(worker, batchOrItem);

                        // Mark as processed
                        if (batchOrItem is Movie[] processedMovies)
                        {
                            foreach (var movie in processedMovies)
                                movie.ProcessingStatus = ProcessingStatus.Processed;
                        }
                        else if (batchOrItem is Movie processedMovie)
                        {
                            processedMovie.ProcessingStatus = ProcessingStatus.Processed;
                        }
                    }
                    catch(Exception ex)
                    {
                        if (batchOrItem is Movie[] failedMovies)
                        {
                            foreach (var movie in failedMovies)
                                HandleMovieFailure(movie);
                        }
                        else if (batchOrItem is Movie failedMovie)
                        {
                            HandleMovieFailure(failedMovie);
                        }

                        lock(workers)
                        {
                            VirtualConsole.ClearLastLine();
                            string itemDescription = isBatchProcessing ? $"batch {taskIndex}" : $"item {itemIndex}";
                            Console.Error.WriteLine($"Failed to process {itemDescription}, marked as unselected");
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
                                if (batchOrItem is Movie[] movieBatch)
                                    processedItems.AddRange(movieBatch);
                                else if (batchOrItem is Movie movie)
                                    processedItems.Add(movie);

                                immutableProcessed = processedItems.ToList();
                            }

                            // Write processed_items.json
                            JsonArray itemsJson = new JsonArray(immutableProcessed
                                .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                                .ToArray());
                            File.WriteAllText(jsonFilePath + $".{taskIndex}",
                                itemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

                            bool success = false;
                            Stopwatch watch = Stopwatch.StartNew();
                            while(!success && watch.ElapsedMilliseconds < 10_000)
                            {
                                try
                                {
                                    lock(workers)
                                        File.Move(jsonFilePath + $".{taskIndex}", jsonFilePath, overwrite: true);
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

                            long averageTime = (long)Math.Max(1, processingTimes.Average());
                            if (isBatchProcessing)
                                averageTime /= workers.Length * oversubscribe;

                            long remainingTime = (nItems - nDone) * averageTime;
                            TimeSpan remainingTimeSpan = TimeSpan.FromMilliseconds(remainingTime);

                            string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                            string timeString = remainingTimeSpan.ToString(
                                (int)remainingTimeSpan.TotalDays > 0
                                    ? @"dd\.hh\:mm\:ss"
                                    : ((int)remainingTimeSpan.TotalHours > 0
                                        ? @"hh\:mm\:ss"
                                        : @"mm\:ss"));

                            VirtualConsole.ClearLastLine();
                            Console.Write($"{nDone}/{nItems}{failedString}, {timeString} remaining");
                        }
                    }
                }));
            }
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
