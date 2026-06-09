using System.Collections.Generic;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Tracks distinct cross-failures to separate bad hardware from bad tasks
    /// (spec §12.3) and applies the per-task retry cap (spec §10). In-memory;
    /// the Scheduler persists/reloads it via manager.state.json.
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
    }
}
