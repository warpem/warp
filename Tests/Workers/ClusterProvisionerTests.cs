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
        public int NextId = 100;
        public readonly List<string> Submits = new();
        public readonly List<string> Cancels = new();

        public ShellResult Run(string cmd)
        {
            if (cmd.StartsWith("SUBMIT"))
            {
                Submits.Add(cmd);
                return new ShellResult(0, $"Submitted batch job {NextId++}", "");
            }
            if (cmd.StartsWith("CANCEL"))
            {
                var m = Regex.Match(cmd, @"CANCEL (\d+)");
                Cancels.Add(m.Groups[1].Value);
                return new ShellResult(0, "", "");
            }
            return new ShellResult(1, "", "unexpected command");
        }
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
    public void Create_ExternalProvisionerConflict_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: true,
            poolSize: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_PoolSizeZero_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: false,
            poolSize: 0, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_MissingConfigPath_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "", externalProvisioner: false,
            poolSize: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
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
        Directory.CreateDirectory(queueDir);

        var shell = new FakeShell();
        try
        {
            var prov = ClusterProvisioner.Create(
                clusterScriptPath: tmplPath, clusterConfigPath: cfgPath, externalProvisioner: false,
                poolSize: 2, clusterVars: new[] { "partition=gpu" },
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
            prov.EnsureWorkers(2);
            Assert.Equal(2, prov.LiveWorkerCount());

            prov.Shutdown();
            string[] lines = File.ReadAllLines(marker);
            Assert.Equal(2, lines.Length);
            Assert.All(lines, l => Assert.Equal("4242", l.Trim()));
        }
        finally { try { File.Delete(marker); } catch { } }
    }
}
