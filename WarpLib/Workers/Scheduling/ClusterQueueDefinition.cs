using System;
using System.IO;
using System.Text.Json.Nodes;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// The user-supplied cluster configuration (pointed to by --cluster_config). Three
    /// fields describe how to talk to the batch scheduler:
    ///   submit              - command to submit the rendered script ({{script_path}})
    ///   submit_job_id_regex - first capture group extracts the job id from submit stdout
    ///   cancel              - command to cancel one job ({{job_id}}), run once per id
    /// One configurable regex covers any scheduler, so no per-scheduler logic ships here.
    /// </summary>
    public class ClusterQueueDefinition
    {
        public string Submit { get; set; }
        public string SubmitJobIdRegex { get; set; }
        public string Cancel { get; set; }

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

            return new ClusterQueueDefinition
            {
                Submit = Req("submit"),
                SubmitJobIdRegex = Req("submit_job_id_regex"),
                Cancel = Req("cancel"),
            };
        }
    }
}
