using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Tracks distinct cross-failures to separate bad hardware from bad tasks
    /// (spec §12.3) and applies the per-task retry cap (spec §10). Persisted to
    /// manager.state.json after every ProcessFailures tick so a manager restart
    /// resumes from the last known state (spec §A2).
    /// </summary>
    public class FailureMatrix
    {
        private readonly Dictionary<string, HashSet<string>> _hostFailures = new(); // host -> distinct task ids
        private readonly Dictionary<string, HashSet<string>> _taskFailures = new(); // task -> distinct hostnames

        private readonly int _hostBlacklistThreshold;
        private readonly int _taskPoisonThreshold;
        private readonly int _retryCap;

        public FailureMatrix(int hostBlacklistThreshold = 4, int taskPoisonThreshold = 4, int retryCap = 4)
        {
            _hostBlacklistThreshold = hostBlacklistThreshold;
            _taskPoisonThreshold = taskPoisonThreshold;
            _retryCap = retryCap;
        }

        public void RecordFailure(string hostname, string taskId)
        {
            if (!_hostFailures.TryGetValue(hostname, out var ht)) _hostFailures[hostname] = ht = new();
            ht.Add(taskId);
            if (!_taskFailures.TryGetValue(taskId, out var th)) _taskFailures[taskId] = th = new();
            th.Add(hostname);
        }

        public bool IsHostBlacklisted(string hostname) =>
            _hostFailures.TryGetValue(hostname, out var ht) && ht.Count >= _hostBlacklistThreshold;

        public bool ShouldPoison(string taskId) =>
            _taskFailures.TryGetValue(taskId, out var th) && th.Count >= _taskPoisonThreshold;

        public bool ShouldPoisonByRetry(int retryCount) => retryCount >= _retryCap;

        public IEnumerable<string> BlacklistedHosts()
        {
            foreach (var kv in _hostFailures)
                if (kv.Value.Count >= _hostBlacklistThreshold)
                    yield return kv.Key;
        }

        // ── Persistence ────────────────────────────────────────────────────────

        /// <summary>
        /// Return a new FailureMatrix with the same failure data but thresholds
        /// taken from <paramref name="thresholdSource"/>. Used when loading a
        /// persisted matrix: restore the data, apply the caller's configured
        /// thresholds (which may differ from the defaults used at save time).
        /// </summary>
        public FailureMatrix WithThresholds(FailureMatrix thresholdSource)
        {
            var result = new FailureMatrix(
                thresholdSource._hostBlacklistThreshold,
                thresholdSource._taskPoisonThreshold,
                thresholdSource._retryCap);
            foreach (var kv in _hostFailures)
                result._hostFailures[kv.Key] = new HashSet<string>(kv.Value);
            foreach (var kv in _taskFailures)
                result._taskFailures[kv.Key] = new HashSet<string>(kv.Value);
            return result;
        }

        /// <summary>Atomically write the failure data to <paramref name="path"/>.</summary>
        public void SaveToFile(string path)
        {
            var dto = new MatrixDto
            {
                HostFailures = _hostFailures.ToDictionary(kv => kv.Key, kv => kv.Value.ToList()),
                TaskFailures = _taskFailures.ToDictionary(kv => kv.Key, kv => kv.Value.ToList()),
            };
            string json = JsonSerializer.Serialize(dto,
                new JsonSerializerOptions { WriteIndented = true });
            string tmp = path + ".tmp." + System.Environment.ProcessId;
            File.WriteAllText(tmp, json);
            File.Move(tmp, path, overwrite: true);
        }

        /// <summary>
        /// Load failure data from <paramref name="path"/>. Returns null if the file
        /// does not exist or cannot be parsed (first run, fresh queue dir).
        /// Thresholds are defaults — call <see cref="WithThresholds"/> to apply
        /// the runtime configuration.
        /// </summary>
        public static FailureMatrix LoadFromFile(string path)
        {
            if (!File.Exists(path)) return null;
            try
            {
                var dto = JsonSerializer.Deserialize<MatrixDto>(File.ReadAllText(path));
                if (dto == null) return null;
                var m = new FailureMatrix();
                if (dto.HostFailures != null)
                    foreach (var kv in dto.HostFailures)
                        m._hostFailures[kv.Key] = new HashSet<string>(kv.Value ?? new());
                if (dto.TaskFailures != null)
                    foreach (var kv in dto.TaskFailures)
                        m._taskFailures[kv.Key] = new HashSet<string>(kv.Value ?? new());
                return m;
            }
            catch { return null; }  // corrupt file → treat as empty (safe; tasks re-accumulate)
        }

        private class MatrixDto
        {
            [JsonPropertyName("host_failures")]
            public Dictionary<string, List<string>> HostFailures { get; set; }
            [JsonPropertyName("task_failures")]
            public Dictionary<string, List<string>> TaskFailures { get; set; }
        }
    }
}
