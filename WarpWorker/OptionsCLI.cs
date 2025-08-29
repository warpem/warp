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


        [Option('s', "silent", HelpText = "Don't write anything to stdout, using only a virtual console")]
        public bool Silent { get; set; }


        [Option("debug", HelpText = "Output debug information, and don't die in the absence of a heartbeat from the master process")]
        public bool Debug { get; set; }

        [Option("debug_attach", HelpText = "Attach a debugger to this worker process")]
        public bool DebugAttach { get; set; }

        [Option('c', "controller", Required = true, HelpText = "Controller endpoint (host:port) to connect to")]
        public string Controller { get; set; }
        
        [Option("persistent", HelpText = "Keep trying to connect indefinitely until first successful connection (for external workers)")]
        public bool Persistent { get; set; }
    }
}