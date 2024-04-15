using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace WarpWorker
{
    class OptionsCLI
    {
        [Option('d', "device", Required = true, HelpText = "GPU ID used for processing everything")]
        public int Device { get; set; }

        [Option('p', "port", Required = true, HelpText = "Port to use for REST API calls")]
        public int Port { get; set; }

        [Option('s', "silent", HelpText = "Don't write anything to stdout, using only a virtual console that can be accessed through REST")]
        public bool Silent { get; set; }

        [Option("pipe", HelpText = "Named pipe to communicate assigned port back to master process (only needed when launching through WorkerWrapper)")]
        public string Pipe { get; set; }

        [Option("debug", HelpText = "Output debug information, and don't die in the absence of a heartbeat from the master process")]
        public bool Debug { get; set; }

        [Option("debug_attach", HelpText = "Attach a debugger to this worker process")]
        public bool DebugAttach { get; set; }
    }
}