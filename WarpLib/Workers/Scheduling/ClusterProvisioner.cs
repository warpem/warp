using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Warp.Workers.Scheduling
{
    /// <summary>Result of running one shell command line.</summary>
    public readonly record struct ShellResult(int ExitCode, string StdOut, string StdErr);

    /// <summary>Runs one shell command line and returns its result. Injectable for tests.</summary>
    public delegate ShellResult ShellRunner(string commandLine);

    /// <summary>
    /// Provisions WarpWorker2 jobs through a user-configured batch scheduler. With only
    /// submit/cancel configured it preserves the original submit-once behaviour. When
    /// status_list is configured it also tracks running/pending jobs and replenishes jobs
    /// that remain absent from the scheduler beyond a registration/missing grace period.
    /// </summary>
    public class ClusterProvisioner : IWorkerProvisioner
    {
        private enum JobState { Registering, Pending, Running }
        private enum StatusRefreshResult { NotConfigured, NotDue, Success, Failed }

        private sealed class TrackedJob
        {
            public DateTime SubmittedUtc { get; init; }
            public DateTime? MissingSinceUtc { get; set; }
            public JobState State { get; set; } = JobState.Registering;
        }

        // Cross-scheduler defaults. A config can replace these with exact token lists.
        private static readonly HashSet<string> DefaultRunningStatuses = new(StringComparer.Ordinal)
        {
            "RUNNING", "R", "COMPLETING", "CG", "RESIZING", // SLURM
            "RUN",                                             // LSF
            "E",                                               // PBS
            "r", "t", "Rr",                                  // SGE
        };

        private static readonly HashSet<string> DefaultPendingStatuses = new(StringComparer.Ordinal)
        {
            "PENDING", "PD", "SUSPENDED", "S",              // SLURM / PBS
            "PEND", "SSUSP", "USUSP", "PSUSP",              // LSF
            "Q", "H", "W", "T",                              // PBS
            "qw", "Rq", "hqw", "hRwq",                      // SGE
        };

        private readonly ClusterQueueDefinition _queue;
        private readonly string _scriptPath;
        private readonly string _queueDir;
        private readonly int _workersPerJob;
        private readonly ShellRunner _runner;
        private readonly Regex _submitJobIdRegex;
        private readonly Func<DateTime> _utcNow;
        private readonly string _provisionerLogPath;
        private readonly HashSet<string> _runningStatuses;
        private readonly HashSet<string> _pendingStatuses;

        // Active jobs participate in deficit calculation. Every successfully parsed job id
        // stays in _uncancelledJobIds even after it is considered gone, so a false-negative
        // status query cannot make shutdown forget a possibly-live allocation.
        private readonly Dictionary<string, TrackedJob> _activeJobs = new();
        private readonly HashSet<string> _uncancelledJobIds = new();
        private readonly HashSet<string> _warnedUnknownStatuses = new();
        private readonly object _sync = new();
        private readonly object _logSync = new();

        private readonly List<IDisposable> _signalRegs = new();
        private ConsoleCancelEventHandler _consoleCancelHandler;
        private bool _signalHandlersDisposed;
        private bool _cancelled;
        private bool _submitOnceLogged;
        private bool _submissionCapLogged;
        private bool _fileLogFailureReported;
        private int _totalSubmissions;
        private DateTime _nextStatusPollUtc = DateTime.MinValue;
        private DateTime _submissionBackoffUntilUtc = DateTime.MinValue;

        private int _runningCount;
        private int _pendingCount;
        private int _registeringCount;
        private (int Running, int Pending, int Registering, int Tracked)? _lastLoggedCounts;

        public ClusterProvisioner(ClusterQueueDefinition queue, string scriptPath,
                                  ShellRunner runner = null, bool registerSignalHandlers = true,
                                  string queueDir = null, int workersPerJob = 1,
                                  Func<DateTime> utcNow = null)
        {
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
            _queueDir = queueDir;
            if (workersPerJob <= 0) throw new ArgumentOutOfRangeException(nameof(workersPerJob));
            _workersPerJob = workersPerJob;
            _runner = runner ?? RunShell;
            _utcNow = utcNow ?? (() => DateTime.UtcNow);

            if (string.IsNullOrWhiteSpace(_queue.SubmitJobIdRegex))
                throw new ArgumentException("submit_job_id_regex is empty.", nameof(queue));
            try { _submitJobIdRegex = new Regex(_queue.SubmitJobIdRegex, RegexOptions.Compiled); }
            catch (Exception ex)
            {
                throw new ArgumentException($"Invalid submit_job_id_regex: {ex.Message}", nameof(queue), ex);
            }

            _runningStatuses = new HashSet<string>(
                _queue.RunningStatuses?.Length > 0 ? _queue.RunningStatuses : DefaultRunningStatuses,
                StringComparer.Ordinal);
            _pendingStatuses = new HashSet<string>(
                _queue.PendingStatuses?.Length > 0 ? _queue.PendingStatuses : DefaultPendingStatuses,
                StringComparer.Ordinal);

            if (!string.IsNullOrEmpty(queueDir))
            {
                string clusterDir = Path.Combine(queueDir, "cluster");
                Directory.CreateDirectory(clusterDir);
                _provisionerLogPath = Path.Combine(clusterDir, "provisioner.log");
            }

            if (registerSignalHandlers)
            {
                // Posix registrations cover normal cluster operation. CancelKeyPress also
                // covers console-hosted/platform cases where SIGINT registration is absent.
                TryRegister(PosixSignal.SIGINT);
                TryRegister(PosixSignal.SIGTERM);
                try
                {
                    _consoleCancelHandler = (_, _) => Shutdown();
                    Console.CancelKeyPress += _consoleCancelHandler;
                }
                catch { _consoleCancelHandler = null; }
            }

            Log(_queue.HasStatusPolling
                ? $"provisioner ready; status polling every {_queue.StatusPollSeconds}s, " +
                  $"missing grace {_queue.RegistrationGraceSeconds}s, {_workersPerJob} worker(s)/job"
                : $"provisioner ready in submit-once mode; {_workersPerJob} worker(s)/job");
        }

        private void TryRegister(PosixSignal signal)
        {
            try { _signalRegs.Add(PosixSignalRegistration.Create(signal, _ => Shutdown())); }
            catch { /* signal not supported on this platform; Console.CancelKeyPress remains */ }
        }

        /// <summary>
        /// Build a ClusterProvisioner from raw CLI option values: validate the option
        /// combination, load the config, parse --cluster_var, build the built-in command,
        /// and render the submission script once to &lt;queueDir&gt;/cluster/worker.sh.
        /// </summary>
        public static ClusterProvisioner Create(
            string clusterScriptPath, string clusterConfigPath, bool externalProvisioner,
            int poolSize, int perDevice, IEnumerable<string> clusterVars,
            string workerExePath, string queueDir, string logDir,
            ShellRunner runner = null, bool registerSignalHandlers = true)
        {
            if (externalProvisioner)
                throw new Exception("--cluster_script cannot be combined with --external_provisioner; pick one provisioning mode.");
            if (poolSize <= 0)
                throw new Exception("--pool_size must be greater than 0 in cluster mode.");
            if (perDevice <= 0)
                throw new Exception("--perdevice must be greater than 0 in cluster mode.");
            if (string.IsNullOrEmpty(clusterConfigPath))
                throw new Exception("--cluster_script requires --cluster_config (the queue-definition JSON).");
            if (string.IsNullOrEmpty(clusterScriptPath) || !File.Exists(clusterScriptPath))
                throw new Exception($"Cluster submission-script template not found: {clusterScriptPath}");

            ClusterQueueDefinition def = ClusterQueueDefinition.Load(clusterConfigPath);
            Dictionary<string, string> vars = ClusterVarParser.Parse(clusterVars);

            // Each cluster job owns one GPU (device 0). Multiple --perdevice processes
            // share it. The shell pid and per-process index make worker ids unique.
            var sb = new StringBuilder();
            for (int i = 0; i < perDevice; i++)
                sb.AppendLine(
                    $"{ShellQuote(workerExePath)} --device 0 --queue-dir {ShellQuote(queueDir)} " +
                    $"--log-dir {ShellQuote(logDir)} --persistent --worker-id \"$(hostname)-$$-{i}\" &");
            sb.Append("wait");

            // Built-ins win over user-supplied variables. tasks_dir/logs_dir are raw paths
            // for use in scheduler directives; static command arguments are shell-quoted.
            vars["command"] = sb.ToString();
            vars["tasks_dir"] = queueDir;
            vars["logs_dir"] = logDir;

            string template = File.ReadAllText(clusterScriptPath);
            if (!Regex.IsMatch(template, @"\{\{\s*command\s*\}\}"))
                throw new Exception("Cluster submission-script template must contain a {{command}} placeholder.");
            string script = TemplateRenderer.Render(template, vars);

            string scriptDir = Path.Combine(queueDir, "cluster");
            Directory.CreateDirectory(scriptDir);
            string scriptPath = Path.Combine(scriptDir, "worker.sh");
            string tmpPath = scriptPath + ".tmp." + Environment.ProcessId + "." + Guid.NewGuid().ToString("N");
            File.WriteAllText(tmpPath, script);
            File.Move(tmpPath, scriptPath, overwrite: true);

            return new ClusterProvisioner(def, scriptPath, runner, registerSignalHandlers,
                queueDir: queueDir, workersPerJob: perDevice);
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                if (_cancelled || target <= 0) return;

                DateTime now = _utcNow();
                StatusRefreshResult refresh = RefreshJobStatesIfDue(now);
                if (refresh == StatusRefreshResult.Failed)
                {
                    Log("status query failed; retaining all tracked jobs and skipping replenishment this tick");
                    return;
                }

                int desired = target;
                int pendingScanLimit = target > int.MaxValue / _workersPerJob
                    ? int.MaxValue
                    : target * _workersPerJob;
                int pendingTasks = CountPendingTasksAtMost(pendingScanLimit);
                if (pendingTasks == -2) return; // queue inspection failed; fail closed this tick
                if (pendingTasks >= 0)
                {
                    if (pendingTasks == 0) return; // no useful work: never replenish an idle pool
                    int jobsForPending = (pendingTasks + _workersPerJob - 1) / _workersPerJob;
                    desired = Math.Min(target, jobsForPending);
                }

                int deficit = desired - _activeJobs.Count;
                if (deficit <= 0) return;

                if (now < _submissionBackoffUntilUtc) return;

                long totalCapLong = Math.Max((long)target,
                    (long)target * Math.Max(1, _queue.MaxSubmissionMultiplier));
                int totalCap = (int)Math.Min(int.MaxValue, totalCapLong);
                int remainingBudget = totalCap - _totalSubmissions;
                if (remainingBudget <= 0)
                {
                    if (!_submissionCapLogged)
                    {
                        _submissionCapLogged = true;
                        Log($"submission safety cap reached ({_totalSubmissions}/{totalCap}); " +
                            "not submitting more jobs. Increase max_submission_multiplier only after checking scheduler state.", error: true);
                    }
                    return;
                }

                int toSubmit = Math.Min(deficit, Math.Max(1, _queue.MaxSubmissionsPerTick));
                toSubmit = Math.Min(toSubmit, remainingBudget);
                bool hadJobsBefore = _activeJobs.Count > 0;
                var jobIdsBeforeSubmit = new HashSet<string>(_activeJobs.Keys);

                Log($"submitting {toSubmit} job(s): desired={desired}, tracked={_activeJobs.Count}, " +
                    $"pending_tasks={(pendingTasks < 0 ? "unknown" : pendingTasks.ToString())}, total_submitted={_totalSubmissions}");

                try
                {
                    for (int i = 0; i < toSubmit; i++)
                        SubmitOne(now);
                    RecalculateAndLogCounts();
                }
                catch (Exception ex)
                {
                    _submissionBackoffUntilUtc = now.AddSeconds(30);
                    Log($"job submission failed: {ex.Message}; backing off for 30s", error: true);

                    // Roll back only jobs created in this batch. Existing useful workers keep
                    // running if a later replenishment attempt fails.
                    var submittedThisCall = new List<string>();
                    foreach (string id in _activeJobs.Keys)
                        if (!jobIdsBeforeSubmit.Contains(id)) submittedThisCall.Add(id);
                    HashSet<string> cancelled = CancelJobIds(submittedThisCall, "submission rollback");
                    foreach (string id in cancelled)
                    {
                        _activeJobs.Remove(id);
                        _uncancelledJobIds.Remove(id);
                    }
                    RecalculateAndLogCounts();

                    // Initial provisioning has no useful pool to preserve, so surface the
                    // actionable failure after cleaning up every trackable partial submit.
                    if (!hadJobsBefore) throw;
                }
            }
        }

        private string SubmitOne(DateTime now)
        {
            string cmd = TemplateRenderer.Render(
                _queue.Submit,
                new Dictionary<string, string> { ["script_path"] = ShellQuote(_scriptPath) });
            ShellResult result = _runner(cmd);
            Match match = _submitJobIdRegex.Match(result.StdOut ?? "");
            bool hasJobId = match.Success && match.Groups.Count >= 2 &&
                            !string.IsNullOrWhiteSpace(match.Groups[1].Value);

            // Some scheduler clients return a non-zero exit code after the server has
            // accepted a job. Preserve a parsable id so the batch rollback can cancel it.
            string id = hasJobId ? match.Groups[1].Value.Trim() : null;
            if (hasJobId) TrackSubmittedJob(id, now);

            if (result.ExitCode != 0)
                throw new Exception(
                    $"submit command exited with code {result.ExitCode}; stdout: {Compact(result.StdOut)}; stderr: {Compact(result.StdErr)}");

            if (!hasJobId)
                throw new Exception(
                    $"Could not parse a job id from submit output using submit_job_id_regex " +
                    $"'{_queue.SubmitJobIdRegex}'. stdout: {Compact(result.StdOut)}; stderr: {Compact(result.StdErr)}");

            return id;
        }

        private void TrackSubmittedJob(string id, DateTime now)
        {
            if (_uncancelledJobIds.Contains(id))
                throw new Exception($"Scheduler returned duplicate job id '{id}'; refusing to lose track of an allocation.");

            _activeJobs[id] = new TrackedJob { SubmittedUtc = now, State = JobState.Registering };
            _uncancelledJobIds.Add(id);
            _totalSubmissions++;
            Log($"submitted job {id} ({_activeJobs.Count} tracked, {_totalSubmissions} total submission(s))");
        }

        private StatusRefreshResult RefreshJobStatesIfDue(DateTime now)
        {
            if (!_queue.HasStatusPolling)
            {
                if (!_submitOnceLogged)
                {
                    _submitOnceLogged = true;
                    Log("status_list is not configured; replenishment and running/pending counters are disabled");
                }
                return StatusRefreshResult.NotConfigured;
            }
            if (_activeJobs.Count == 0 || now < _nextStatusPollUtc)
                return StatusRefreshResult.NotDue;

            _nextStatusPollUtc = now.AddSeconds(Math.Max(1, _queue.StatusPollSeconds));
            string user = Environment.GetEnvironmentVariable("USER") ??
                          Environment.GetEnvironmentVariable("USERNAME") ?? "";
            string cmd;
            try
            {
                cmd = TemplateRenderer.Render(_queue.StatusList,
                    new Dictionary<string, string> { ["user"] = ShellQuote(user) });
            }
            catch (Exception ex)
            {
                Log($"could not render status_list: {ex.Message}", error: true);
                return StatusRefreshResult.Failed;
            }

            ShellResult result;
            try { result = _runner(cmd); }
            catch (Exception ex)
            {
                Log($"status command failed: {ex.Message}", error: true);
                return StatusRefreshResult.Failed;
            }
            if (result.ExitCode != 0)
            {
                Log($"status command exited with code {result.ExitCode}; stderr: {Compact(result.StdErr)}", error: true);
                return StatusRefreshResult.Failed;
            }

            var visible = new Dictionary<string, JobState>();
            int formattedLines = 0;
            string[] lines = (result.StdOut ?? "").Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (string raw in lines)
            {
                string line = raw.Trim();
                if (line.Length == 0) continue;

                int comma = line.IndexOf(',');
                string id;
                string status;
                if (comma >= 0)
                {
                    id = line.Substring(0, comma).Trim();
                    status = line.Substring(comma + 1).Trim();
                    if (id.Length > 0) formattedLines++;
                }
                else
                {
                    id = line;
                    status = "";
                    if (_activeJobs.ContainsKey(id)) formattedLines++;
                }

                if (!_activeJobs.ContainsKey(id)) continue;
                if (_runningStatuses.Contains(status))
                    visible[id] = JobState.Running;
                else
                {
                    // Unknown statuses are deliberately treated as alive/pending. A new
                    // scheduler token must never trigger a resubmission storm.
                    visible[id] = JobState.Pending;
                    if (status.Length > 0 && !_pendingStatuses.Contains(status) && _warnedUnknownStatuses.Add(status))
                        Log($"unrecognized scheduler status '{status}'; treating matching jobs as pending", error: true);
                }
            }

            if (lines.Length > 0 && formattedLines == 0)
            {
                Log($"status output did not contain any 'job_id,status' records; " +
                    $"retaining {_activeJobs.Count} tracked job(s). stdout: {Compact(result.StdOut)}", error: true);
                return StatusRefreshResult.Failed;
            }

            TimeSpan grace = TimeSpan.FromSeconds(Math.Max(1, _queue.RegistrationGraceSeconds));
            foreach (var pair in _activeJobs.ToArray())
            {
                string id = pair.Key;
                TrackedJob job = pair.Value;
                if (visible.TryGetValue(id, out JobState state))
                {
                    job.State = state;
                    job.MissingSinceUtc = null;
                    continue;
                }

                job.State = JobState.Registering;
                job.MissingSinceUtc ??= now;
                if (now - job.MissingSinceUtc.Value >= grace)
                {
                    _activeJobs.Remove(id);
                    Log($"job {id} has been absent from scheduler status for " +
                        $"{(int)(now - job.MissingSinceUtc.Value).TotalSeconds}s; marking it gone and allowing replenishment");
                }
            }

            RecalculateAndLogCounts();
            return StatusRefreshResult.Success;
        }

        private int CountPendingTasksAtMost(int limit)
        {
            if (string.IsNullOrEmpty(_queueDir)) return -1; // direct unit-test/back-compat constructor
            string pendingDir = Path.Combine(_queueDir, "pending");
            if (!Directory.Exists(pendingDir)) return 0;

            int count = 0;
            try
            {
                foreach (string _ in Directory.EnumerateFiles(pendingDir, "*.json"))
                {
                    count++;
                    if (count >= Math.Max(1, limit)) break;
                }
            }
            catch (DirectoryNotFoundException) { return 0; }
            catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException)
            {
                Log($"could not count pending task files in '{pendingDir}': {ex.Message}; " +
                    "skipping submissions this tick", error: true);
                return -2;
            }
            return count;
        }

        private void RecalculateAndLogCounts()
        {
            _runningCount = _activeJobs.Values.Count(j => j.State == JobState.Running);
            _pendingCount = _activeJobs.Values.Count(j => j.State == JobState.Pending);
            _registeringCount = _activeJobs.Values.Count(j => j.State == JobState.Registering);
            var current = (_runningCount, _pendingCount, _registeringCount, _activeJobs.Count);
            if (_lastLoggedCounts != current)
            {
                _lastLoggedCounts = current;
                Log($"jobs: running={_runningCount} pending={_pendingCount} " +
                    $"registering={_registeringCount} tracked={_activeJobs.Count} total_submitted={_totalSubmissions}");
            }
        }

        public int LiveWorkerCount()
        {
            lock (_sync) return _activeJobs.Count;
        }

        public int RunningJobCount { get { lock (_sync) return _runningCount; } }
        public int PendingJobCount { get { lock (_sync) return _pendingCount; } }
        public int RegisteringJobCount { get { lock (_sync) return _registeringCount; } }

        public void Shutdown()
        {
            lock (_sync)
            {
                _cancelled = true;

                string[] ids = _uncancelledJobIds.ToArray();
                if (ids.Length > 0)
                {
                    Log($"cancelling {ids.Length} submitted cluster job(s) with parallelism " +
                        $"{Math.Max(1, _queue.CancelParallelism)}");
                    HashSet<string> cancelled = CancelJobIds(ids, "shutdown");
                    foreach (string id in cancelled)
                    {
                        _uncancelledJobIds.Remove(id);
                        _activeJobs.Remove(id);
                    }
                    Log($"cancelled {cancelled.Count}/{ids.Length} job(s); " +
                        $"{_uncancelledJobIds.Count} remain for a later shutdown retry",
                        error: _uncancelledJobIds.Count > 0);
                }

                RecalculateAndLogCounts();
                DisposeSignalHandlers();
            }
        }

        private HashSet<string> CancelJobIds(IEnumerable<string> jobIds, string reason)
        {
            var uniqueIds = new HashSet<string>(jobIds);
            var ids = new string[uniqueIds.Count];
            uniqueIds.CopyTo(ids);
            var succeeded = new ConcurrentBag<string>();
            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = Math.Max(1, _queue.CancelParallelism)
            };

            Parallel.ForEach(ids, options, id =>
            {
                try
                {
                    string cmd = TemplateRenderer.Render(
                        _queue.Cancel,
                        new Dictionary<string, string> { ["job_id"] = ShellQuote(id) });
                    ShellResult result = _runner(cmd);
                    if (result.ExitCode != 0)
                        throw new Exception(
                            $"exit code {result.ExitCode}; stdout: {Compact(result.StdOut)}; stderr: {Compact(result.StdErr)}");
                    succeeded.Add(id);
                }
                catch (Exception ex)
                {
                    Log($"failed to cancel cluster job {id} during {reason}: {ex.Message}", error: true);
                }
            });

            return succeeded.ToHashSet();
        }

        private void DisposeSignalHandlers()
        {
            if (_signalHandlersDisposed) return;
            _signalHandlersDisposed = true;
            foreach (IDisposable reg in _signalRegs)
                try { reg.Dispose(); } catch { }
            _signalRegs.Clear();
            if (_consoleCancelHandler != null)
            {
                try { Console.CancelKeyPress -= _consoleCancelHandler; } catch { }
                _consoleCancelHandler = null;
            }
        }

        private void Log(string message, bool error = false)
        {
            string consoleLine = "[cluster] " + message;
            if (error) Console.Error.WriteLine(consoleLine);
            else Console.WriteLine(consoleLine);

            if (string.IsNullOrEmpty(_provisionerLogPath)) return;
            string fileLine = $"{_utcNow():O} {consoleLine}{Environment.NewLine}";
            lock (_logSync)
            {
                try { File.AppendAllText(_provisionerLogPath, fileLine); }
                catch (Exception ex)
                {
                    if (!_fileLogFailureReported)
                    {
                        _fileLogFailureReported = true;
                        Console.Error.WriteLine(
                            $"[cluster] could not write provisioner log '{_provisionerLogPath}': {ex.Message}");
                    }
                }
            }
        }

        private static string Compact(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return "(empty)";
            string compact = value.Replace("\r", " ").Replace("\n", " ").Trim();
            return compact.Length <= 500 ? compact : compact.Substring(0, 500) + "...";
        }

        private static string ShellQuote(string value)
        {
            value ??= "";
            if (OperatingSystem.IsWindows())
                return "\"" + value.Replace("\"", "\\\"") + "\"";
            return "'" + value.Replace("'", "'\"'\"'") + "'";
        }

        /// <summary>
        /// Default shell runner. Reads stdout/stderr concurrently and enforces a 30-second
        /// timeout so a wedged scheduler command cannot block heartbeats or shutdown forever.
        /// </summary>
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

            using Process process = Process.Start(psi) ??
                throw new Exception($"Failed to start shell '{psi.FileName}'.");
            Task<string> stdout = process.StandardOutput.ReadToEndAsync();
            Task<string> stderr = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));

            try
            {
                process.WaitForExitAsync(timeout.Token).GetAwaiter().GetResult();
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                try { process.WaitForExit(); } catch { }
                throw new TimeoutException(
                    $"Shell command timed out after 30 seconds: {commandLine}; " +
                    $"stdout: {Compact(stdout.IsCompleted ? stdout.GetAwaiter().GetResult() : "")}; " +
                    $"stderr: {Compact(stderr.IsCompleted ? stderr.GetAwaiter().GetResult() : "")}");
            }

            return new ShellResult(
                process.ExitCode,
                stdout.GetAwaiter().GetResult(),
                stderr.GetAwaiter().GetResult());
        }
    }
}
