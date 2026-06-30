using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Warp.Tools;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// One unit of work. Carries an init command sequence (resource loading,
    /// fingerprint-skipped by the worker) and a main command sequence. Commands
    /// are NamedSerializableObject, identical to the REST transport. Spec §5.
    /// </summary>
    public class TaskItem
    {
        public string TaskId { get; set; }
        public string Stage { get; set; } = "";
        public bool RequiresGpu { get; set; } = true;
        public NamedSerializableObject[] Init { get; set; } = Array.Empty<NamedSerializableObject>();
        public NamedSerializableObject[] Main { get; set; } = Array.Empty<NamedSerializableObject>();
        public string InitFingerprint { get; set; } = "";
        public int MaxRuntimeSeconds { get; set; } = 0;   // 0 = no self-imposed limit
        public int RetryCount { get; set; } = 0;
        public string CreatedAt { get; set; } = "";

        // Result payload (set by the worker on completion); inline scalars only.
        public object Result { get; set; }
        public string Error { get; set; }

        // Hostname the task most recently failed on. Stamped by the worker on
        // MarkFailed so the Scheduler can attribute failures to hosts for the
        // bad-node blacklist (spec §12.3).
        public string FailedOnHost { get; set; }

        private static readonly JsonSerializerOptions Opts = new() { WriteIndented = true };

        public string ToJson() => JsonSerializer.Serialize(this, Opts);

        public static TaskItem FromJson(string json) =>
            JsonSerializer.Deserialize<TaskItem>(json, Opts);

        /// <summary>SHA-256 over the serialized init array. Computed by the enqueuer.</summary>
        public void ComputeInitFingerprint()
        {
            string initJson = JsonSerializer.Serialize(Init, Opts);
            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(initJson));
            InitFingerprint = Convert.ToHexString(hash);
        }
    }
}
