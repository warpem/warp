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
                        var FormattedItems = arr.Cast<object>().Select(item => item?.ToString() ?? "null");
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
            int oversubscribe = 1
        ) where T : class
        {
            // Default implementation for single item processing
            getBatch ??= (start, _) => (T)(object)cli.InputSeries[start];

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

            bool isBatch = typeof(T) == typeof(Movie[]);
            int nBatches = workers.Length * oversubscribe;
            int itemsPerBatch = (int)Math.Ceiling(cli.InputSeries.Length / (double)nBatches);

            if (!isBatch)
            {
                // single-item processing
                Helper.ForCPUGreedy(0, cli.InputSeries.Length, workers.Length * oversubscribe, null, (iitem, threadID) => { ProcessItem(iitem, threadID, iitem); }, null);
            }
            else
            {
                // batch processing
                List<Task> batchTasks = new List<Task>();

                for (int batchIndex = 0; batchIndex < nBatches; batchIndex++)
                {
                    int startIndex = batchIndex * itemsPerBatch;
                    if (startIndex >= cli.InputSeries.Length)
                        break;

                    int endIndex = Math.Min(startIndex + itemsPerBatch, cli.InputSeries.Length);

                    Console.WriteLine($"Submitted {endIndex - startIndex} items in batch {batchIndex} to worker process {batchIndex % workers.Length}");

                    int currentBatch = batchIndex;
                    batchTasks.Add(Task.Run(() => ProcessItem(currentBatch, currentBatch, currentBatch)));
                }

                Task.WaitAll(batchTasks.ToArray());
            }

            timerOverall.Stop();

            // write out full Json one last time
            Task.WaitAll(jsonTasks.ToArray());
            JsonArray finalItemsJson = new JsonArray(processedItems
                .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                .ToArray());
            File.WriteAllText(jsonFilePath,
                finalItemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

            Console.WriteLine(
                $"\nFinished processing in {TimeSpan.FromMilliseconds(timerOverall.ElapsedMilliseconds):hh\\:mm\\:ss}");

            // close for processing individual item
            void ProcessItem(int index, int threadID, int jsonIndex)
            {
                Stopwatch timer = Stopwatch.StartNew();
                WorkerWrapper processor = workers[threadID % workers.Length];

                T item = getBatch(index, isBatch ? Math.Min(index + itemsPerBatch, cli.InputSeries.Length) : index + 1);
                Movie[] moviesToProcess = isBatch ? (Movie[])((object)item) : new[] { (Movie)((object)item) };

                // Path correction (identical to original)
                foreach (var movie in moviesToProcess)
                {
                    if (Path.GetFullPath(cli.OutputProcessing) != Path.GetFullPath(Path.GetDirectoryName(movie.DataPath)))
                    {
                        if (string.IsNullOrEmpty(movie.DataDirectoryName))
                            movie.DataDirectoryName = Path.GetDirectoryName(movie.Path);

                        movie.Path = Path.Combine(cli.OutputProcessing, Path.GetFileName(movie.Path));
                        movie.SaveMeta();
                    }
                }

                // Log file setup (differs slightly for batches)
                processor.Console.Clear();
                string logFile = isBatch ? Path.Combine(logDirectory, $"batch{index}.log") : Path.Combine(logDirectory, $"{moviesToProcess[0].RootName}.log");
                processor.Console.SetFileOutput(logFile);

                try
                {
                    body(processor, item);

                    foreach (var movie in moviesToProcess)
                    {
                        movie.LoadMeta();
                        movie.ProcessingStatus = ProcessingStatus.Processed;
                        movie.SaveMeta();
                    }
                }
                catch
                {
                    foreach (var movie in moviesToProcess)
                    {
                        movie.LoadMeta();
                        movie.UnselectManual = true;
                        movie.ProcessingStatus = ProcessingStatus.LeaveOut;
                        movie.SaveMeta();
                    }

                    lock(workers)
                    {
                        VirtualConsole.ClearLastLine();
                        // Error message differs slightly for batches
                        if (isBatch)
                            Console.Error.WriteLine($"Failed to process batch {index}, marked as unselected");
                        else
                            Console.Error.WriteLine($"Failed to process {moviesToProcess[0].Path}, marked as unselected");

                        Console.Error.WriteLine($"Check logs in {logDirectory} for more info.");
                        Console.Error.WriteLine("Use the change_selection WarpTool to reactivate this item if required.");
                        nFailed += moviesToProcess.Length;
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
                            processedItems.AddRange(moviesToProcess);
                            immutableProcessed = processedItems.ToList();
                        }

                        JsonArray itemsJson = new JsonArray(immutableProcessed
                            .Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix))
                            .ToArray());
                        File.WriteAllText(jsonFilePath + $".{jsonIndex}",
                            itemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

                        bool success = false;
                        Stopwatch watch = Stopwatch.StartNew();
                        while(!success && watch.ElapsedMilliseconds < 10_000)
                        {
                            try
                            {
                                lock(workers)
                                    File.Move(jsonFilePath + $".{jsonIndex}", jsonFilePath, true);
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
                        nDone += moviesToProcess.Length;
                        processingTimes.Enqueue(timer.ElapsedMilliseconds);
                        if (processingTimes.Count > 20)
                            processingTimes.Dequeue();

                        // Time calculation differs slightly for batches
                        long averageTime = (long)Math.Max(1, processingTimes.Average() /
                                                             (isBatch ? workers.Length * oversubscribe : (workers.Length * oversubscribe)));
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
