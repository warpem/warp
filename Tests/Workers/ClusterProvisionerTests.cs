using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class ClusterProvisionerTests
{
    private static ClusterQueueDefinition FakeQueue() => new ClusterQueueDefinition
    {
        Submit = "SUBMIT {{script_path}}",
        SubmitJobIdRegex = @"Submitted batch job (\d+)",
        Cancel = "CANCEL {{job_id}}",
    };

    // Fake shell: SUBMIT lines return an incrementing job id; CANCEL lines are recorded.
    private sealed class FakeShell
    {
        private readonly object _sync = new();
        public int NextId = 100;
        public readonly List<string> Submits = new();
        public readonly List<string> Cancels = new();
        public int StatusCalls;
        public string StatusOutput = "";
        public int StatusExitCode;
        public int SubmitFailureOnCall;
        public bool FailedSubmitIncludesJobId;
        public int CancelFailuresRemaining;

        public ShellResult Run(string cmd)
        {
            lock (_sync)
            {
                if (cmd.StartsWith("SUBMIT"))
                {
                    Submits.Add(cmd);
                    int id = NextId++;
                    if (SubmitFailureOnCall == Submits.Count)
                        return new ShellResult(17,
                            FailedSubmitIncludesJobId ? $"Submitted batch job {id}" : "",
                            "scheduler unavailable");
                    return new ShellResult(0, $"Submitted batch job {id}", "");
                }
                if (cmd.StartsWith("STATUS"))
                {
                    StatusCalls++;
                    return new ShellResult(StatusExitCode, StatusOutput,
                        StatusExitCode == 0 ? "" : "status unavailable");
                }
                if (cmd.StartsWith("CANCEL"))
                {
                    var m = Regex.Match(cmd, @"CANCEL '?([^']+)'?");
                    Cancels.Add(m.Groups[1].Value);
                    if (CancelFailuresRemaining > 0)
                    {
                        CancelFailuresRemaining--;
                        return new ShellResult(1, "", "cancel failed");
                    }
                    return new ShellResult(0, "", "");
                }
                return new ShellResult(1, "", "unexpected command");
            }
        }
    }

    private sealed class FakeClock
    {
        public DateTime Now = new(2026, 7, 16, 12, 0, 0, DateTimeKind.Utc);
        public DateTime UtcNow() => Now;
        public void AdvanceSeconds(int seconds) => Now = Now.AddSeconds(seconds);
    }

    private static ClusterQueueDefinition StatusQueue(int graceSeconds = 5) => new ClusterQueueDefinition
    {
        Submit = "SUBMIT {{script_path}}",
        SubmitJobIdRegex = @"Submitted batch job (\d+)",
        Cancel = "CANCEL {{job_id}}",
        StatusList = "STATUS {{user}}",
        StatusPollSeconds = 1,
        RegistrationGraceSeconds = graceSeconds,
        MaxSubmissionsPerTick = 100,
        MaxSubmissionMultiplier = 4,
        CancelParallelism = 4,
    };

    private static string CreateQueueDir(int pendingTasks)
    {
        string root = Path.Combine(Path.GetTempPath(), "cluster_queue_" + Guid.NewGuid().ToString("N"));
        string pending = Path.Combine(root, "pending");
        Directory.CreateDirectory(pending);
        for (int i = 0; i < pendingTasks; i++)
            File.WriteAllText(Path.Combine(pending, $"task-{i}.json"), "{}");
        return root;
    }

    private static string WriteScript()
    {
        string p = Path.Combine(Path.GetTempPath(), "worker_" + Guid.NewGuid().ToString("N") + ".sh");
        File.WriteAllText(p, "#!/bin/bash\necho hi\n");
        return p;
    }

    [Fact]
    public void EnsureWorkers_SubmitsTargetCount_AndParsesIds()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(3);

        Assert.Equal(3, shell.Submits.Count);
        Assert.Equal(3, prov.LiveWorkerCount());
    }

    [Fact]
    public void EnsureWorkers_IsIdempotent_DoesNotOversubmit()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(2);
        prov.EnsureWorkers(2);

        Assert.Equal(2, shell.Submits.Count);
    }

    [Fact]
    public void Shutdown_CancelsEverySubmittedJobOnce()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(3);
        prov.Shutdown();
        prov.Shutdown();   // idempotent: must not cancel again

        Assert.Equal(3, shell.Cancels.Count);
        Assert.Contains("100", shell.Cancels);
        Assert.Contains("101", shell.Cancels);
        Assert.Contains("102", shell.Cancels);
    }

    [Fact]
    public void EnsureWorkers_UnparseableSubmitOutput_Throws()
    {
        ShellRunner garbage = _ => new ShellResult(0, "no id here", "");
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), garbage, registerSignalHandlers: false);

        Assert.ThrowsAny<Exception>(() => prov.EnsureWorkers(1));
    }

    [Fact]
    public void StatusPolling_TracksCountersAndReplenishesOnlyAfterGrace()
    {
        string queueDir = CreateQueueDir(3);
        var shell = new FakeShell();
        var clock = new FakeClock();
        try
        {
            var prov = new ClusterProvisioner(StatusQueue(), WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, utcNow: clock.UtcNow);

            prov.EnsureWorkers(3);
            Assert.Equal(3, shell.Submits.Count);

            clock.AdvanceSeconds(1);
            shell.StatusOutput = "100,RUNNING\n101,PENDING\n";
            prov.EnsureWorkers(3);
            Assert.Equal(1, prov.RunningJobCount);
            Assert.Equal(1, prov.PendingJobCount);
            Assert.Equal(1, prov.RegisteringJobCount);

            clock.AdvanceSeconds(4);
            prov.EnsureWorkers(3);
            Assert.Equal(3, shell.Submits.Count); // 102 is still inside its missing grace

            clock.AdvanceSeconds(2);
            prov.EnsureWorkers(3);
            Assert.Equal(4, shell.Submits.Count); // 103 replaces 102
            Assert.Equal(3, prov.LiveWorkerCount());

            string log = File.ReadAllText(Path.Combine(queueDir, "cluster", "provisioner.log"));
            Assert.Contains("running=1 pending=1 registering=1", log);
            Assert.Contains("marking it gone and allowing replenishment", log);
            Assert.Contains("submitting 1 job(s)", log);
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void StatusFailure_FailsClosedWithoutReplenishing()
    {
        string queueDir = CreateQueueDir(2);
        var shell = new FakeShell();
        var clock = new FakeClock();
        try
        {
            var prov = new ClusterProvisioner(StatusQueue(), WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, utcNow: clock.UtcNow);
            prov.EnsureWorkers(2);

            clock.AdvanceSeconds(1);
            shell.StatusExitCode = 1;
            prov.EnsureWorkers(2);

            Assert.Equal(2, shell.Submits.Count);
            Assert.Equal(2, prov.LiveWorkerCount());
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void MalformedStatusOutput_FailsClosedAcrossGracePeriod()
    {
        string queueDir = CreateQueueDir(1);
        var shell = new FakeShell();
        var clock = new FakeClock();
        try
        {
            var prov = new ClusterProvisioner(StatusQueue(), WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, utcNow: clock.UtcNow);
            prov.EnsureWorkers(1);
            shell.StatusOutput = "not a job status record";

            clock.AdvanceSeconds(1);
            prov.EnsureWorkers(1);
            clock.AdvanceSeconds(20);
            prov.EnsureWorkers(1);

            Assert.Single(shell.Submits);
            Assert.Equal(1, prov.LiveWorkerCount());
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void UnknownSchedulerStatus_IsTreatedAsAliveAndPending()
    {
        string queueDir = CreateQueueDir(1);
        var shell = new FakeShell();
        var clock = new FakeClock();
        try
        {
            var prov = new ClusterProvisioner(StatusQueue(), WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, utcNow: clock.UtcNow);
            prov.EnsureWorkers(1);
            shell.StatusOutput = "100,FUTURE_SCHEDULER_STATE";

            clock.AdvanceSeconds(1);
            prov.EnsureWorkers(1);

            Assert.Equal(1, prov.PendingJobCount);
            Assert.Equal(0, prov.RegisteringJobCount);
            Assert.Single(shell.Submits);
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void SubmissionBurstAndLifetimeCaps_BoundReplenishment()
    {
        string queueDir = CreateQueueDir(5);
        var shell = new FakeShell();
        var clock = new FakeClock();
        var queue = StatusQueue(graceSeconds: 1);
        queue.MaxSubmissionsPerTick = 2;
        queue.MaxSubmissionMultiplier = 1;
        try
        {
            var prov = new ClusterProvisioner(queue, WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, utcNow: clock.UtcNow);

            prov.EnsureWorkers(5);
            Assert.Equal(2, shell.Submits.Count);
            prov.EnsureWorkers(5);
            Assert.Equal(4, shell.Submits.Count);
            prov.EnsureWorkers(5);
            Assert.Equal(5, shell.Submits.Count);

            clock.AdvanceSeconds(1);
            shell.StatusOutput = "";
            prov.EnsureWorkers(5);
            clock.AdvanceSeconds(2);
            prov.EnsureWorkers(5);

            Assert.Equal(5, shell.Submits.Count); // lifetime cap blocks replacements
            Assert.Equal(0, prov.LiveWorkerCount());
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void PendingTaskCount_LimitsJobsUsingWorkersPerJob()
    {
        string queueDir = CreateQueueDir(3);
        var shell = new FakeShell();
        try
        {
            var prov = new ClusterProvisioner(StatusQueue(), WriteScript(), shell.Run,
                registerSignalHandlers: false, queueDir: queueDir, workersPerJob: 2);

            prov.EnsureWorkers(10);

            Assert.Equal(2, shell.Submits.Count);
            Assert.Equal(2, prov.LiveWorkerCount());
        }
        finally { try { Directory.Delete(queueDir, true); } catch { } }
    }

    [Fact]
    public void InitialSubmitFailure_RollsBackEveryParsableJobId()
    {
        var shell = new FakeShell
        {
            SubmitFailureOnCall = 2,
            FailedSubmitIncludesJobId = true,
        };
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run,
            registerSignalHandlers: false);

        Assert.ThrowsAny<Exception>(() => prov.EnsureWorkers(2));

        Assert.Equal(2, shell.Submits.Count);
        Assert.Equal(2, shell.Cancels.Count);
        Assert.Equal(0, prov.LiveWorkerCount());
    }

    [Fact]
    public void FailedCancellation_IsRetriedOnLaterShutdown()
    {
        var shell = new FakeShell { CancelFailuresRemaining = 1 };
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run,
            registerSignalHandlers: false);
        prov.EnsureWorkers(1);

        prov.Shutdown();
        Assert.Equal(1, prov.LiveWorkerCount());
        prov.Shutdown();

        Assert.Equal(2, shell.Cancels.Count);
        Assert.Equal(0, prov.LiveWorkerCount());
    }

    [Fact]
    public void Create_ExternalProvisionerConflict_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: true,
            poolSize: 1, perDevice: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_PoolSizeZero_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: false,
            poolSize: 0, perDevice: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_MissingConfigPath_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "", externalProvisioner: false,
            poolSize: 1, perDevice: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_RendersScriptWithCommandAndClusterVars_Submits()
    {
        // Real config file + template file; fake shell. Proves Create wires the whole chain.
        string cfgPath = Path.Combine(Path.GetTempPath(), "cfg_" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(cfgPath, @"{
            ""submit"": ""SUBMIT {{script_path}}"",
            ""submit_job_id_regex"": ""Submitted batch job (\\d+)"",
            ""cancel"": ""CANCEL {{job_id}}""
        }");
        string tmplPath = Path.Combine(Path.GetTempPath(), "tmpl_" + Guid.NewGuid().ToString("N") + ".sh");
        File.WriteAllText(tmplPath, "#!/bin/bash\n#SBATCH -p {{partition}}\n{{command}}\n");
        string queueDir = Path.Combine(Path.GetTempPath(), "q_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(queueDir, "pending"));
        File.WriteAllText(Path.Combine(queueDir, "pending", "one.json"), "{}");
        File.WriteAllText(Path.Combine(queueDir, "pending", "two.json"), "{}");

        var shell = new FakeShell();
        try
        {
            var prov = ClusterProvisioner.Create(
                clusterScriptPath: tmplPath, clusterConfigPath: cfgPath, externalProvisioner: false,
                poolSize: 2, perDevice: 1, clusterVars: new[] { "partition=gpu" },
                workerExePath: "/opt/warp/WarpWorker2", queueDir: queueDir, logDir: Path.Combine(queueDir, "logs"),
                runner: shell.Run, registerSignalHandlers: false);

            string written = File.ReadAllText(Path.Combine(queueDir, "cluster", "worker.sh"));
            Assert.Contains("#SBATCH -p gpu", written);
            Assert.Contains("--device 0", written);
            Assert.Contains("$(hostname)-$$", written);
            Assert.Contains("--persistent", written);

            prov.EnsureWorkers(2);
            Assert.Equal(2, shell.Submits.Count);
        }
        finally
        {
            File.Delete(cfgPath); File.Delete(tmplPath);
            try { Directory.Delete(queueDir, true); } catch { }
        }
    }

    [Fact]
    public void Create_PerDeviceGreaterThanOne_LaunchesMultipleWorkersAndWaits()
    {
        string cfgPath = Path.Combine(Path.GetTempPath(), "cfg_" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(cfgPath, @"{
            ""submit"": ""SUBMIT {{script_path}}"",
            ""submit_job_id_regex"": ""Submitted batch job (\\d+)"",
            ""cancel"": ""CANCEL {{job_id}}""
        }");
        string tmplPath = Path.Combine(Path.GetTempPath(), "tmpl_" + Guid.NewGuid().ToString("N") + ".sh");
        File.WriteAllText(tmplPath, "#!/bin/bash\n{{command}}\n");
        string queueDir = Path.Combine(Path.GetTempPath(), "q_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(queueDir);

        try
        {
            ClusterProvisioner.Create(
                clusterScriptPath: tmplPath, clusterConfigPath: cfgPath, externalProvisioner: false,
                poolSize: 1, perDevice: 3, clusterVars: null,
                workerExePath: "/opt/warp/WarpWorker2", queueDir: queueDir, logDir: Path.Combine(queueDir, "logs"),
                runner: _ => new ShellResult(0, "Submitted batch job 1", ""), registerSignalHandlers: false);

            string written = File.ReadAllText(Path.Combine(queueDir, "cluster", "worker.sh"));
            // Three backgrounded worker processes, each with a distinct per-process index.
            Assert.Contains("$(hostname)-$$-0", written);
            Assert.Contains("$(hostname)-$$-1", written);
            Assert.Contains("$(hostname)-$$-2", written);
            Assert.Equal(3, Regex.Matches(written, @"--persistent").Count);
            // A single trailing `wait` so the job holds its allocation until workers exit.
            Assert.EndsWith("wait", written.TrimEnd());
        }
        finally
        {
            File.Delete(cfgPath); File.Delete(tmplPath);
            try { Directory.Delete(queueDir, true); } catch { }
        }
    }

    [Fact]
    public void DefaultShellRunner_RealShell_SubmitAndCancel()
    {
        // Exercises the real /bin/sh path (not the fake runner). Unix-only.
        if (OperatingSystem.IsWindows()) return;

        string marker = Path.Combine(Path.GetTempPath(), "cancelled_" + Guid.NewGuid().ToString("N") + ".txt");
        var queue = new ClusterQueueDefinition
        {
            Submit = "echo Submitted batch job 4242",
            SubmitJobIdRegex = @"Submitted batch job (\d+)",
            Cancel = $"echo {{{{job_id}}}} >> \"{marker}\"",
        };
        var prov = new ClusterProvisioner(queue, WriteScript(),
            runner: null /* default real shell */, registerSignalHandlers: false);
        try
        {
            prov.EnsureWorkers(1);
            Assert.Equal(1, prov.LiveWorkerCount());

            prov.Shutdown();
            string[] lines = File.ReadAllLines(marker);
            Assert.Single(lines);
            Assert.All(lines, l => Assert.Equal("4242", l.Trim()));
        }
        finally { try { File.Delete(marker); } catch { } }
    }
}
