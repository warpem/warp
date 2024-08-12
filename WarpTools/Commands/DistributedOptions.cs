using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    abstract class DistributedOptions : BaseOptions
    {
        [OptionGroup("Work distribution", 100)]
        [Option("device_list", HelpText = "Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system")]
        public IEnumerable<int> DeviceList { get; set; }

        [Option("perdevice", Default = 1, HelpText = "Number of processes per GPU")]
        public int ProcessesPerDevice { get; set; }

        [OptionGroup("Advanced remote work distribution", 102)]
        [Option("workers", HelpText = "List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces")]
        public IEnumerable<string> Workers { get; set; }

        private WorkerWrapper[] ConnectedWorkers = null;

        public override void Evaluate()
        {
            base.Evaluate();
        }

        public virtual WorkerWrapper[] GetWorkers(bool attachDebugger = false)
        {
            if (ConnectedWorkers != null)
                return ConnectedWorkers;

            Console.WriteLine("Connecting to workers...");

            if (Workers == null || !Workers.Any())
            {
                List<int> UsedDevices = (DeviceList == null || !DeviceList.Any()) ?
                                        Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList() :
                                        DeviceList.ToList();
                int NDevices = UsedDevices.Count;

                if (NDevices <= 0)
                    throw new Exception("No devices found or specified");

                int MaxWorkers = Math.Min(InputSeries.Length, UsedDevices.Count * ProcessesPerDevice);
                List<WorkerWrapper> NewWorkers = new List<WorkerWrapper>();
                foreach (var id in UsedDevices)
                {
                    for (int i = 0; i < ProcessesPerDevice; i++)
                    {
                        if (NewWorkers.Count < MaxWorkers)
                        {
                            WorkerWrapper NewWorker = new WorkerWrapper(id,
                                                                        !Helper.IsDebug, 
                                                                        attachDebugger: attachDebugger);
                            NewWorkers.Add(NewWorker);
                        }
                    }
                }

                ConnectedWorkers = NewWorkers.ToArray();
            }
            else
            {
                ConnectedWorkers = new WorkerWrapper[Workers.Count()];

                for (int i = 0; i < Workers.Count(); i++)
                {
                    string[] Parts = Workers.ElementAt(i).Split(new[] { ':' }, StringSplitOptions.RemoveEmptyEntries);
                    string Host = Parts[0];
                    int Port = int.Parse(Parts[1]);

                    ConnectedWorkers[i] = new WorkerWrapper(Host, Port);
                }
            }

            if (Options != null)
            {
                Helper.ForCPU(0, ConnectedWorkers.Length, ConnectedWorkers.Length, null, (i, threadID) =>
                {
                    ConnectedWorkers[i].LoadGainRef(Options.Import.GainPath, 
                                                    Options.Import.GainFlipX, 
                                                    Options.Import.GainFlipY, 
                                                    Options.Import.GainTranspose, 
                                                    Options.Import.DefectsPath);
                }, null);
            }

            Console.WriteLine($"Connected to {ConnectedWorkers.Length} workers");

            return ConnectedWorkers;
        }
    }
}
