using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

/// <summary>
/// Covers the reconstruction-only task graph that MTools' `reconstruct` verb emits, without
/// needing a GPU or real data. What is actually verified here is the plumbing that GPU tests
/// can't reach cheaply: that the graph drains through TaskRunner + WorkerPoolSettings, and
/// that the optional trailing command arguments survive the JSON transport in both
/// directions (new payload read by the new handler, legacy payload read by the new handler).
/// </summary>
public class ReconstructOnlyMockTests : IDisposable
{
    private readonly string _root;
    public ReconstructOnlyMockTests() { _root = Path.Combine(Path.GetTempPath(), "recon-" + Guid.NewGuid().ToString("N")); }
    public void Dispose() { try { Directory.Delete(_root, true); } catch { } }

    private static TaskItem[] BuildReconstructOnlyGraph(string tempDir, string populationPath, string speciesPath)
    {
        var Source = new DataSource { Name = "mocksource", PixelSize = 1M };
        var Options = new ProcessingOptionsMPARefine { NIterations = 0, BatchSize = 16 };

        // Exactly the init triple the verb builds: headerless params, gain/defects, and the
        // resident population prepared WITHOUT references.
        var Init = new[]
        {
            WorkerCommands.SetHeaderlessParams(new int2(0), 0, "float"),
            WorkerCommands.LoadGainRef(Source.GainPath, Source.GainFlipX, Source.GainFlipY,
                                       Source.GainTranspose, Source.DefectsPath),
            WorkerCommands.MPAPreparePopulation(populationPath, null, makeRefs: false),
        };

        var Tasks = Enumerable.Range(0, 3).Select(i =>
        {
            var t = new TaskItem
            {
                TaskId = $"{i:D6}-reconstruct-series{i}",
                Stage = "preprocess",
                Init = Init,
                Main = new[]
                {
                    WorkerCommands.MPARefineAndSave($"series{i}.tomostar", Options, Source, tempDir,
                                                    saveItemMeta: false),
                    WorkerCommands.GcCollect(),
                },
            };
            t.ComputeInitFingerprint();
            return t;
        }).ToList();

        var Reduce = new TaskItem
        {
            TaskId = "0000-reconstruct-mockspecies",
            Stage = "preprocess",
            Main = new[]
            {
                WorkerCommands.MPAReconstructAverage(speciesPath, new[] { Path.Combine(tempDir, "worker_0") }, true),
                WorkerCommands.GcCollect(),
            },
        };
        Reduce.ComputeInitFingerprint();
        Tasks.Add(Reduce);

        return Tasks.ToArray();
    }

    [Fact]
    public void ReconstructOnlyGraphDrainsInMockMode()
    {
        var layout = new QueueLayout(_root);
        layout.EnsureDirectories();
        var queue = new TaskQueue(layout);
        var pool = new WorkPool(layout, queue);

        var tasks = BuildReconstructOnlyGraph(Path.Combine(_root, "temp"),
                                              Path.Combine(_root, "mock.population"),
                                              Path.Combine(_root, "mock.species"));

        var provisioner = new LocalProvisioner(_root, new[] { 0 }, perDevice: 2, mock: true);
        var scheduler = new Scheduler(layout, queue, provisioner, target: 2,
            workerStallTimeoutMs: 30_000, workerStartupGraceMs: 60_000);

        pool.Enqueue(tasks);

        var schedCts = new System.Threading.CancellationTokenSource();
        var schedThread = new System.Threading.Thread(
            () => scheduler.RunToDrain(pollMs: 500, cancel: schedCts.Token)) { IsBackground = true };
        schedThread.Start();

        var results = pool.Distribute(tasks, pollMs: 200);
        schedCts.Cancel();
        schedThread.Join();
        provisioner.Shutdown();

        Assert.Equal(tasks.Length, results.Count);
        Assert.All(results.Values, r => Assert.Equal(WorkOutcome.Done, r.Outcome));

        // Decisive: a real worker process actually moved each task to done/. Without this,
        // the assertions above could pass on results that never touched a worker.
        Assert.All(tasks, t => Assert.True(File.Exists(Path.Combine(layout.Done, t.TaskId + ".json")),
                                           $"{t.TaskId} never reached done/"));
    }

    // The verb drives its two phases through TaskRunner rather than the raw primitives, so
    // cover that path too. Regression test for TaskRunner starting the scheduler before
    // enqueueing: workers then spawned against an empty queue, polled twice, exited, and the
    // whole run stalled with tasks stuck in pending/.
    [Fact]
    public void TaskRunnerDrainsTheGraph()
    {
        string queueDir = Path.Combine(_root, "tasks");
        string logDir = Path.Combine(_root, "logs");

        var tasks = BuildReconstructOnlyGraph(Path.Combine(_root, "temp"),
                                              Path.Combine(_root, "mock.population"),
                                              Path.Combine(_root, "mock.species"));

        // DeviceList is explicit: leaving it empty makes CreateProvisioner call
        // GPU.GetDeviceCount(), which mock mode exists to avoid needing.
        var settings = new WorkerPoolSettings
        {
            DeviceList = new[] { 0 },
            ProcessesPerDevice = 2,
            Mock = true,
        };

        int completed = 0;
        TaskRunner.Run(queueDir, logDir, tasks, settings,
                       onItemDone: (task, result) => { if (result.Outcome == WorkOutcome.Done) completed++; },
                       pollMs: 200);

        // TaskRunner only throws when EVERY task fails, so assert per-task success as well.
        Assert.Equal(tasks.Length, completed);
        Assert.All(tasks, t => Assert.True(File.Exists(Path.Combine(new QueueLayout(queueDir).Done, t.TaskId + ".json")),
                                           $"{t.TaskId} never reached done/"));
    }

    [Fact]
    public void OptionalTrailingArgsSurviveTheJsonTransport()
    {
        var tasks = BuildReconstructOnlyGraph("/tmp/x", "/tmp/x.population", "/tmp/x.species");

        // Round-trip through the same serializer the filesystem queue uses.
        var RoundTripped = tasks.Select(t => TaskItem.FromJson(t.ToJson())).ToArray();

        var Prepare = RoundTripped[0].Init.Single(c => c.Name == WorkerCommandNames.MPAPreparePopulation);
        Assert.Equal(3, Prepare.Content.Length);
        Assert.False((bool)Prepare.Content[2]);
        // stagingLoad must be "" rather than null: the converter calls GetType() on every element.
        Assert.Equal("", (string)Prepare.Content[1]);

        var Refine = RoundTripped[0].Main.Single(c => c.Name == WorkerCommandNames.MPARefineAndSave);
        Assert.Equal(5, Refine.Content.Length);
        Assert.False((bool)Refine.Content[4]);

        var Reduce = RoundTripped[^1].Main.Single(c => c.Name == WorkerCommandNames.MPAReconstructAverage);
        Assert.Equal(3, Reduce.Content.Length);
        Assert.True((bool)Reduce.Content[2]);
    }

    [Fact]
    public void LegacyShortPayloadsStillCarryTheDefaults()
    {
        // WorkerWrapper (used by the M GUI) still emits the pre-existing 2- and 4-element
        // forms of these commands by hand. The new handlers read the extra elements
        // length-guarded, so those payloads must survive the transport at their old length
        // and let the handler fall back to the previous behaviour (refs made, meta saved).
        var Legacy = new TaskItem
        {
            TaskId = "0000-legacy",
            Init = new[] { new NamedSerializableObject(WorkerCommandNames.MPAPreparePopulation, "p.population", "staging") },
            Main = new[]
            {
                new NamedSerializableObject(WorkerCommandNames.MPARefineAndSave,
                                            "series.tomostar",
                                            new ProcessingOptionsMPARefine(),
                                            new DataSource { Name = "s" },
                                            "tmp"),
            },
        };

        var RoundTripped = TaskItem.FromJson(Legacy.ToJson());

        Assert.Equal(2, RoundTripped.Init[0].Content.Length);
        Assert.Equal(4, RoundTripped.Main[0].Content.Length);
    }
}
