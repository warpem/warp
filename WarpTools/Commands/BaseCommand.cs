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

        internal void IterateOverItems<T>(WorkerWrapper[] workers,
                                          BaseOptions cli,
                                          Action<WorkerWrapper, T> body,
                                          Func<int, int, T[]> getBatch = null,
                                          int oversubscribe = 1,
                                          bool crashOnFail = false) where T : Movie
        {
            Action<WorkerWrapper, T[]> wrappedBody = (worker, items) => body(worker, items.First());
            IterateOverItems<T>(workers, 
                                cli, 
                                wrappedBody, 
                                getBatch, 
                                oversubscribe,
                                crashOnFail);
        }

        internal void IterateOverItems<T>(WorkerWrapper[] workers,
                                          BaseOptions cli,
                                          Action<WorkerWrapper, T[]> body,
                                          Func<int, int, T[]> getBatch = null,
                                          int oversubscribe = 1,
                                          bool crashOnFail = false) where T : Movie
        {
            // Default implementation for single item processing
            Func<int, int, T[]> internalGetBatch = getBatch ??
                                                   ((start, _) => [(T)cli.InputSeries[start]]);

            object sync = new();

            string logDirectory = Path.Combine(cli.OutputProcessing, "logs");
            Directory.CreateDirectory(logDirectory);

            var jsonSuccessFilePath = Path.Combine(cli.OutputProcessing, "processed_items.json");
            var jsonFailFilePath = Path.Combine(cli.OutputProcessing, "failed_items.json");
            List<Task> jsonTasks = new();
            List<T> processedItems = new();
            List<T> failedItems = new();

            foreach (var item in cli.InputSeries)
                item.ProcessingStatus = ProcessingStatus.Unprocessed;

            Console.Write($"0/{cli.InputSeries.Length}");

            int nDone = 0;
            int nFailed = 0;
            Queue<long> processingTimes = new Queue<long>();
            Stopwatch timerOverall = Stopwatch.StartNew();

            bool isBatch = getBatch != null;
            int nBatches = (workers?.Length ?? 1) * oversubscribe;
            int itemsPerBatch = (int)Math.Ceiling(cli.InputSeries.Length / (double)nBatches);

            if (!isBatch)
            {
                // single-item processing
                Helper.ForCPUGreedy(0, cli.InputSeries.Length, (workers?.Length ?? 1) * oversubscribe,
                                    null,
                                    (iitem, threadID) => { ProcessItem(iitem, threadID, iitem); },
                                    null);
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

                    Console.WriteLine($"Submitted {endIndex - startIndex} items in batch {batchIndex} to worker process {batchIndex % (workers?.Length ?? 1)}");

                    int currentBatch = batchIndex;
                    batchTasks.Add(Task.Run(() => ProcessItem(currentBatch, currentBatch, currentBatch)));
                }

                Task.WaitAll(batchTasks.ToArray());
            }

            timerOverall.Stop();

            // write out full Json one last time
            Task.WaitAll(jsonTasks.ToArray());

            WriteMiniJson(jsonSuccessFilePath, processedItems);

            if (failedItems.Any())
                WriteMiniJson(jsonFailFilePath, failedItems);

            Console.WriteLine();
            Console.WriteLine($"Finished processing in {TimeSpan.FromMilliseconds(timerOverall.ElapsedMilliseconds):hh\\:mm\\:ss}");

            // close for processing individual item
            void ProcessItem(int index, int threadID, int jsonIndex)
            {
                Stopwatch timer = Stopwatch.StartNew();
                WorkerWrapper processor = workers?[threadID % workers.Length];

                T[] itemsToProcess = internalGetBatch(index, isBatch ?
                                                                 Math.Min(index + itemsPerBatch,
                                                                          cli.InputSeries.Length) :
                                                                 index + 1);

                // Path correction (identical to original)
                foreach (var item in itemsToProcess)
                {
                    if (Path.GetFullPath(cli.OutputProcessing) !=
                        Path.GetFullPath(Path.GetDirectoryName(item.DataPath)))
                    {
                        if (string.IsNullOrEmpty(item.DataDirectoryName))
                            item.DataDirectoryName = Path.GetDirectoryName(item.Path);

                        item.Path = Path.Combine(cli.OutputProcessing, Path.GetFileName(item.Path));
                        item.SaveMeta();
                    }
                }

                // Log file setup (differs slightly for batches)
                processor?.Console.Clear();
                string logFile = isBatch ?
                                     Path.Combine(logDirectory, $"batch{index}.log") :
                                     Path.Combine(logDirectory, $"{itemsToProcess[0].RootName}.log");
                processor?.Console.SetFileOutput(logFile);

                try
                {
                    body(processor, itemsToProcess);

                    foreach (var item in itemsToProcess)
                    {
                        item.LoadMeta();
                        item.ProcessingStatus = ProcessingStatus.Processed;
                        item.SaveMeta();
                    }
                }
                catch (Exception ex)
                {
                    foreach (var item in itemsToProcess)
                    {
                        item.LoadMeta();
                        item.UnselectManual = true;
                        item.ProcessingStatus = ProcessingStatus.LeaveOut;
                        item.SaveMeta();
                    }

                    lock (sync)
                    {
                        // Clearing the last line in an actual terminal will mess up strict formatting
                        if (!cli.StrictFormatting)
                            VirtualConsole.ClearLastLine();

                        // Error message differs slightly for batches
                        if (isBatch)
                            Console.Error.WriteLine($"Failed to process batch {index}, marked as unselected");
                        else
                            Console.Error.WriteLine($"Failed to process {itemsToProcess[0].Path}, marked as unselected");

                        Console.Error.WriteLine($"Check logs in {logDirectory} for more info.");
                        Console.Error.WriteLine("Use the change_selection WarpTool to reactivate this item if required.");

                        Console.Error.WriteLine("Exception details: " + ex);

                        nFailed += itemsToProcess.Length;
                        failedItems.AddRange(itemsToProcess);
                    }

                    if (crashOnFail)
                        throw;
                }
                finally
                {
                    processor?.Console.SetFileOutput("");

                    jsonTasks.Add(Task.Run(() =>
                    {
                        List<T> immutableProcessed;
                        List<T> immutableFailed;
                        lock (sync)
                        {
                            processedItems.AddRange(itemsToProcess);
                            immutableProcessed = processedItems.ToList();
                            immutableFailed = failedItems.ToList();
                        }

                        var tempSuccess = jsonSuccessFilePath + $".{jsonIndex}";
                        var tempFail = jsonFailFilePath + $".{jsonIndex}";

                        WriteMiniJson(tempSuccess, immutableProcessed);

                        if (immutableFailed.Any())
                            WriteMiniJson(tempFail, immutableFailed);

                        bool success = false;
                        Stopwatch watch = Stopwatch.StartNew();
                        while (!success && watch.ElapsedMilliseconds < 10_000)
                            try
                            {
                                lock (sync)
                                {
                                    File.Move(tempSuccess, jsonSuccessFilePath, true);
                                    File.Move(tempFail, jsonFailFilePath, true);
                                }

                                success = true;
                            }
                            catch { }
                    }));

                    timer.Stop();

                    lock (sync)
                    {
                        nDone += itemsToProcess.Length;
                        processingTimes.Enqueue(timer.ElapsedMilliseconds);
                        if (processingTimes.Count > 20)
                            processingTimes.Dequeue();

                        // Time calculation differs slightly for batches
                        long averageTime = (long)Math.Max(1, processingTimes.Average() /
                                                             (isBatch ?
                                                                  (workers?.Length ?? 1) * oversubscribe :
                                                                  ((workers?.Length ?? 1) * oversubscribe)));
                        long remainingTime = (cli.InputSeries.Length - nDone) * averageTime;
                        TimeSpan remainingTimeSpan = TimeSpan.FromMilliseconds(remainingTime);

                        string failedString = nFailed > 0 ? $", {nFailed} failed" : "";
                        string timeString = remainingTimeSpan.ToString((int)remainingTimeSpan.TotalDays > 0 ?
                                                                           @"dd\.hh\:mm\:ss" :
                                                                           ((int)remainingTimeSpan.TotalHours > 0 ?
                                                                                @"hh\:mm\:ss" :
                                                                                @"mm\:ss"));

                        VirtualConsole.ClearLastLine();
                        Console.Write($"{nDone}/{cli.InputSeries.Length}{failedString}, {timeString} remaining");
                    }
                }
            }
        }

        protected void WriteMiniJson<T>(string path, IEnumerable<T> items) where T : Movie
        {
            JsonArray itemsJson = new JsonArray(items.Select(series => series.ToMiniJson(""))
                                                     .ToArray());
            File.WriteAllText(path,
                              itemsJson.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
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