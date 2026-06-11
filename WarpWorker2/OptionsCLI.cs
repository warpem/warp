using System.Collections.Generic;
using CommandLine;

namespace WarpWorker2
{
    class OptionsCLI
    {
        [Option('d', "device", Required = true, HelpText = "GPU ID used for processing")]
        public int Device { get; set; }

        [Option('q', "queue-dir", Required = true, HelpText = "Path to the shared queue directory")]
        public string QueueDir { get; set; }

        [Option("log-dir", HelpText = "Directory for per-item processing logs (<task_id>.log). Defaults to <queue-dir>/logs")]
        public string LogDir { get; set; }

        [Option("stages", HelpText = "Space-separated stages this worker may claim; empty = any")]
        public IEnumerable<string> Stages { get; set; }

        [Option("worker-id", HelpText = "Explicit worker id; if empty, derived from PID and device")]
        public string WorkerId { get; set; }

        [Option('s', "silent", HelpText = "Suppress stdout")]
        public bool Silent { get; set; }

        [Option("mock", HelpText = "Mock mode: run MockCommand handlers instead of real GPU work")]
        public bool Mock { get; set; }

        [Option("debug", HelpText = "Debug output; do not exit on heartbeat stall")]
        public bool Debug { get; set; }

        [Option("debug_attach", HelpText = "Attach a debugger to this worker")]
        public bool DebugAttach { get; set; }
    }
}
