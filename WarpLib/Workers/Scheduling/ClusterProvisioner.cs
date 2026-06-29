using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace Warp.Workers.Scheduling
{
    /// <summary>Result of running one shell command line.</summary>
    public readonly record struct ShellResult(int ExitCode, string StdOut, string StdErr);

    /// <summary>Runs one shell command line and returns its result. Injectable for tests.</summary>
    public delegate ShellResult ShellRunner(string commandLine);

    /// <summary>
    /// Minimal cluster provisioning: submit a fixed pool of identical --persistent
    /// WarpWorker2 jobs to a batch scheduler, then cancel them all on shutdown. No status
    /// polling or resubmission — the filesystem queue's stall-sweep re-pends a dead
    /// worker's task for a surviving worker. See the design spec for the full rationale.
    /// </summary>
    public class ClusterProvisioner : IWorkerProvisioner
    {
        private readonly ClusterQueueDefinition _queue;
        private readonly string _scriptPath;
        private readonly ShellRunner _runner;

        private readonly List<string> _jobIds = new();
        private readonly object _sync = new();
        private bool _cancelled;
        private readonly List<IDisposable> _signalRegs = new();

        public ClusterProvisioner(ClusterQueueDefinition queue, string scriptPath,
                                  ShellRunner runner = null, bool registerSignalHandlers = true)
        {
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
            _runner = runner ?? RunShell;

            if (registerSignalHandlers)
            {
                // --persistent workers won't self-stop, so make sure Ctrl-C / kill on the
                // manager cancels the pool before we die. Shutdown() is idempotent.
                TryRegister(PosixSignal.SIGINT);
                TryRegister(PosixSignal.SIGTERM);
            }
        }

        private void TryRegister(PosixSignal signal)
        {
            try { _signalRegs.Add(PosixSignalRegistration.Create(signal, _ => Shutdown())); }
            catch { /* signal not supported on this platform; ignore */ }
        }

        /// <summary>
        /// Build a ClusterProvisioner from raw CLI option values: validate the option
        /// combination, load the config, parse --cluster_var, build the built-in
        /// {{command}}, render the submission script once to &lt;queueDir&gt;/cluster/worker.sh.
        /// </summary>
        public static ClusterProvisioner Create(
            string clusterScriptPath, string clusterConfigPath, bool externalProvisioner,
            int poolSize, IEnumerable<string> clusterVars,
            string workerExePath, string queueDir, string logDir,
            ShellRunner runner = null, bool registerSignalHandlers = true)
        {
            // Validation first (no file IO), so misconfiguration fails fast and cheap.
            if (externalProvisioner)
                throw new Exception("--cluster_script cannot be combined with --external_provisioner; pick one provisioning mode.");
            if (poolSize <= 0)
                throw new Exception("--pool_size must be greater than 0 in cluster mode.");
            if (string.IsNullOrEmpty(clusterConfigPath))
                throw new Exception("--cluster_script requires --cluster_config (the queue-definition JSON).");
            if (string.IsNullOrEmpty(clusterScriptPath) || !File.Exists(clusterScriptPath))
                throw new Exception($"Cluster submission-script template not found: {clusterScriptPath}");

            ClusterQueueDefinition def = ClusterQueueDefinition.Load(clusterConfigPath);

            Dictionary<string, string> vars = ClusterVarParser.Parse(clusterVars);

            // Built-in command: one GPU per job (device 0), self-naming by hostname+pid so
            // ids never collide across nodes; --persistent so a transient empty queue does
            // not make the worker exit and hang a re-pended task. The compute node's shell
            // expands $(hostname) and $$ at runtime.
            string command =
                $"\"{workerExePath}\" --device 0 --queue-dir \"{queueDir}\" --log-dir \"{logDir}\" " +
                $"--persistent --worker-id \"$(hostname)-$$\"";
            vars["command"] = command;   // built-in wins over any user-supplied command var

            string template = File.ReadAllText(clusterScriptPath);
            string script = TemplateRenderer.Render(template, vars);

            string scriptDir = Path.Combine(queueDir, "cluster");
            Directory.CreateDirectory(scriptDir);
            string scriptPath = Path.Combine(scriptDir, "worker.sh");
            File.WriteAllText(scriptPath, script);

            return new ClusterProvisioner(def, scriptPath, runner, registerSignalHandlers);
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                if (_cancelled) return;
                while (_jobIds.Count < target)
                {
                    string cmd = TemplateRenderer.Render(
                        _queue.Submit, new Dictionary<string, string> { ["script_path"] = _scriptPath });
                    ShellResult r = _runner(cmd);

                    Match m = Regex.Match(r.StdOut ?? "", _queue.SubmitJobIdRegex);
                    if (!m.Success || m.Groups.Count < 2)
                        throw new Exception(
                            $"Could not parse a job id from the scheduler's submit output using " +
                            $"submit_job_id_regex '{_queue.SubmitJobIdRegex}'.\nstdout: {r.StdOut}\nstderr: {r.StdErr}");

                    _jobIds.Add(m.Groups[1].Value);
                }
            }
        }

        public int LiveWorkerCount()
        {
            lock (_sync) return _jobIds.Count;
        }

        public void Shutdown()
        {
            lock (_sync)
            {
                if (_cancelled) return;
                _cancelled = true;

                foreach (string id in _jobIds)
                {
                    try
                    {
                        string cmd = TemplateRenderer.Render(
                            _queue.Cancel, new Dictionary<string, string> { ["job_id"] = id });
                        _runner(cmd);
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"Failed to cancel cluster job {id}: {ex.Message}");
                    }
                }

                foreach (IDisposable reg in _signalRegs)
                    try { reg.Dispose(); } catch { }
                _signalRegs.Clear();
            }
        }

        /// <summary>Default ShellRunner: run the command line through the platform shell.</summary>
        public static ShellResult RunShell(string commandLine)
        {
            bool win = OperatingSystem.IsWindows();
            var psi = new ProcessStartInfo
            {
                FileName = win ? "cmd.exe" : "/bin/sh",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add(win ? "/c" : "-c");
            psi.ArgumentList.Add(commandLine);

            using var p = Process.Start(psi);
            string o = p.StandardOutput.ReadToEnd();
            string e = p.StandardError.ReadToEnd();
            p.WaitForExit();
            return new ShellResult(p.ExitCode, o, e);
        }
    }
}
