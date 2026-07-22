using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Text.Json.Nodes;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// The user-supplied cluster configuration (pointed to by --cluster_config). Three
    /// required fields describe how to submit and cancel jobs:
    ///   submit              - command to submit the rendered script ({{script_path}})
    ///   submit_job_id_regex - first capture group extracts the job id from submit stdout
    ///   cancel              - command to cancel one job ({{job_id}}), run once per id
    /// Optional status fields enable scheduler-aware replenishment and running/pending
    /// counters. status_list must emit one "job_id,status" record per line.
    /// </summary>
    public class ClusterQueueDefinition
    {
        public string Submit { get; set; }
        public string SubmitJobIdRegex { get; set; }
        public string Cancel { get; set; }

        public string StatusList { get; set; }
        public string[] RunningStatuses { get; set; } = Array.Empty<string>();
        public string[] PendingStatuses { get; set; } = Array.Empty<string>();
        public int StatusPollSeconds { get; set; } = 10;
        public int RegistrationGraceSeconds { get; set; } = 120;
        public int MaxSubmissionsPerTick { get; set; } = 32;
        public int MaxSubmissionMultiplier { get; set; } = 4;
        public int CancelParallelism { get; set; } = 16;

        public bool HasStatusPolling => !string.IsNullOrWhiteSpace(StatusList);

        public static ClusterQueueDefinition Load(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Cluster config path is empty.");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Cluster config file not found: {path}");

            JsonNode root;
            try
            {
                root = JsonNode.Parse(File.ReadAllText(path));
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to parse cluster config JSON '{path}': {ex.Message}", ex);
            }

            string Req(string name)
            {
                string v = null;
                try { v = root?[name]?.GetValue<string>(); } catch { /* wrong type -> treat as missing */ }
                if (string.IsNullOrWhiteSpace(v))
                    throw new Exception($"Cluster config '{path}' is missing required string field '{name}'.");
                return v;
            }

            string Opt(string name)
            {
                try { return root?[name]?.GetValue<string>(); }
                catch { throw new Exception($"Cluster config '{path}' field '{name}' must be a string."); }
            }

            int OptPositive(string name, int defaultValue)
            {
                JsonNode node = root?[name];
                if (node == null) return defaultValue;
                int value;
                try { value = node.GetValue<int>(); }
                catch { throw new Exception($"Cluster config '{path}' field '{name}' must be an integer."); }
                if (value <= 0)
                    throw new Exception($"Cluster config '{path}' field '{name}' must be greater than 0.");
                return value;
            }

            string[] OptStrings(string name)
            {
                JsonNode node = root?[name];
                if (node == null) return Array.Empty<string>();
                if (node is not JsonArray array)
                    throw new Exception($"Cluster config '{path}' field '{name}' must be an array of strings.");

                var values = new List<string>();
                foreach (JsonNode item in array)
                {
                    string value;
                    try { value = item?.GetValue<string>(); }
                    catch { throw new Exception($"Cluster config '{path}' field '{name}' must contain only strings."); }
                    if (string.IsNullOrWhiteSpace(value))
                        throw new Exception($"Cluster config '{path}' field '{name}' cannot contain empty values.");
                    values.Add(value.Trim());
                }
                return values.ToArray();
            }

            string submit = Req("submit");
            string submitRegex = Req("submit_job_id_regex");
            string cancel = Req("cancel");
            try { _ = new Regex(submitRegex); }
            catch (Exception ex)
            {
                throw new Exception(
                    $"Cluster config '{path}' has invalid submit_job_id_regex: {ex.Message}", ex);
            }

            return new ClusterQueueDefinition
            {
                Submit = submit,
                SubmitJobIdRegex = submitRegex,
                Cancel = cancel,
                StatusList = Opt("status_list"),
                RunningStatuses = OptStrings("running_statuses"),
                PendingStatuses = OptStrings("pending_statuses"),
                StatusPollSeconds = OptPositive("status_poll_seconds", 10),
                RegistrationGraceSeconds = OptPositive("registration_grace_seconds", 120),
                MaxSubmissionsPerTick = OptPositive("max_submissions_per_tick", 32),
                MaxSubmissionMultiplier = OptPositive("max_submission_multiplier", 4),
                CancelParallelism = OptPositive("cancel_parallelism", 16),
            };
        }
    }
}
