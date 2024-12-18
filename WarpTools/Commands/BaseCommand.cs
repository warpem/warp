﻿using MathNet.Numerics.Distributions;
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
                var Attributes = options.GetType().GetCustomAttributes(typeof(CommandLine.VerbAttribute), false);
                if (Attributes.Length > 0)
                {
                    var Option = (CommandLine.VerbAttribute)Attributes[0];
                    Console.WriteLine($"Running command {Option.Name} with:");
                }
            }

            var Type = options.GetType();

            foreach (var field in Type.GetProperties())
            {
                var Attributes = field.GetCustomAttributes(typeof(CommandLine.OptionAttribute), false);
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

        internal void IterateOverItems(WorkerWrapper[] workers, BaseOptions cli, Action<WorkerWrapper, Movie> body, int oversubscribe = 1)
        {
            string LogDirectory = Path.Combine(cli.OutputProcessing, "logs");
            Directory.CreateDirectory(LogDirectory);

            var JsonFilePath = Path.Combine(cli.OutputProcessing, "processed_items.json");
            List<Task> JsonTasks = new();
            List<Movie> ProcessedItems = new List<Movie>();

            foreach (var item in cli.InputSeries)
                item.ProcessingStatus = ProcessingStatus.Unprocessed;

            Console.Write($"0/{cli.InputSeries.Length}");

            int NDone = 0;
            int NFailed = 0;
            Queue<long> ProcessingTimes = new Queue<long>();
            Stopwatch TimerOverall = Stopwatch.StartNew();
            Helper.ForCPUGreedy(0, cli.InputSeries.Length, workers.Length * oversubscribe, null, (iitem, threadID) =>
            {
                Stopwatch Timer = Stopwatch.StartNew();

                WorkerWrapper Processor = workers[threadID % workers.Length];
                Movie M = cli.InputSeries[iitem];

                if (Path.GetFullPath(cli.OutputProcessing) != Path.GetFullPath(Path.GetDirectoryName(M.DataPath)))
                {
                    if (string.IsNullOrEmpty(M.DataDirectoryName))
                        M.DataDirectoryName = Path.GetDirectoryName(M.Path);

                    M.Path = Path.Combine(cli.OutputProcessing, Path.GetFileName(M.Path));
                    M.SaveMeta();
                }

                Processor.Console.Clear();
                Processor.Console.SetFileOutput(Path.Combine(LogDirectory, $"{M.RootName}.log"));

                try
                {
                    // process the movie
                    body(Processor, M);

                    M.ProcessingStatus = ProcessingStatus.Processed;
                }
                catch
                {
                    M.UnselectManual = true;
                    M.ProcessingStatus = ProcessingStatus.LeaveOut;
                    M.SaveMeta();

                    lock (workers)
                    {
                        VirtualConsole.ClearLastLine();
                        Console.Error.WriteLine($"Failed to process {M.Path}, marked as unselected");
                        Console.Error.WriteLine($"Check logs in {LogDirectory} for more info.");
                        Console.Error.WriteLine("Use the change_selection WarpTool to reactivate this item if required.");
                        NFailed++;
                    }
                }
                finally
                {
                    JsonTasks.Add(Task.Run(() =>
                    {
                        List<Movie> ImmutableProcessed;
                        lock (workers)
                        {
                            ProcessedItems.Add(M);
                            ImmutableProcessed = ProcessedItems.ToList();
                        }

                        // write processed_items.json
                        JsonArray ItemsJson = new JsonArray(ImmutableProcessed.Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix)).ToArray());
                        File.WriteAllText(JsonFilePath + $".{iitem}", ItemsJson.ToJsonString(new JsonSerializerOptions() { WriteIndented = true }));

                        bool Success = false;
                        Stopwatch Watch = Stopwatch.StartNew();
                        while (!Success && Watch.ElapsedMilliseconds < 10_000)
                        {
                            try
                            {
                                lock (workers)
                                    File.Move(JsonFilePath + $".{iitem}", JsonFilePath, true);
                                Success = true;
                            }
                            catch { }
                        }
                    }));
                }

                Processor.Console.SetFileOutput("");

                Timer.Stop();

                lock (workers)
                {
                    NDone++;
                    ProcessingTimes.Enqueue(Timer.ElapsedMilliseconds);
                    if (ProcessingTimes.Count > 20)
                        ProcessingTimes.Dequeue();

                    long AverageTime = (long)Math.Max(1, ProcessingTimes.Average() / (workers.Length * oversubscribe));
                    long RemainingTime = (cli.InputSeries.Length - NDone) * AverageTime;
                    TimeSpan RemainingTimeSpan = TimeSpan.FromMilliseconds(RemainingTime);

                    string FailedString = NFailed > 0 ? $", {NFailed} failed" : "";


                    string TimeString = RemainingTimeSpan.ToString((int)RemainingTimeSpan.TotalDays > 0 ? 
                                        @"dd\.hh\:mm\:ss" : 
                                        ((int)RemainingTimeSpan.TotalHours > 0 ? 
                                            @"hh\:mm\:ss" : 
                                            @"mm\:ss"));

                    VirtualConsole.ClearLastLine();
                    Console.Write($"{NDone}/{cli.InputSeries.Length}{FailedString}, {TimeString} remaining");
                }
            }, null);
            TimerOverall.Stop();

            // Write out full Json one last time in case the last thread to write it out wasn't the thread processing the last item
            Task.WaitAll(JsonTasks.ToArray());
            JsonArray ItemsJson = new JsonArray(ProcessedItems.Select(series => series.ToMiniJson(cli.Options.Filter.ParticlesSuffix)).ToArray());
            File.WriteAllText(JsonFilePath, ItemsJson.ToJsonString(new JsonSerializerOptions() { WriteIndented = true }));

            Console.WriteLine($"\nFinished processing in {TimeSpan.FromMilliseconds(TimerOverall.ElapsedMilliseconds):hh\\:mm\\:ss}");
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
