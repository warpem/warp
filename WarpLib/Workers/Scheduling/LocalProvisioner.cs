using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Local mode: maintains worker child processes pinned to the configured
    /// devices, pointing them at the queue dir. Respawns dead ones up to target.
    /// Mirrors the spawn path in WorkerWrapper (spec §7).
    /// </summary>
    public class LocalProvisioner : IWorkerProvisioner
    {
        private readonly string _queueDir;
        private readonly int[] _devices;
        private readonly int _perDevice;
        private readonly bool _mock;
        private readonly string _workerExeName;
        private readonly List<Process> _procs = new();
        private readonly object _sync = new();

        public LocalProvisioner(string queueDir, int[] devices, int perDevice,
                                bool mock = false, string workerExeName = "WarpWorker2")
        {
            _queueDir = queueDir;
            _devices = devices;
            _perDevice = perDevice;
            _mock = mock;
            _workerExeName = workerExeName;
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                _procs.RemoveAll(p => p.HasExited);
                var slots = new List<int>();
                foreach (int dev in _devices)
                    for (int i = 0; i < _perDevice; i++)
                        slots.Add(dev);

                while (_procs.Count < Math.Min(target, slots.Count))
                {
                    int dev = slots[_procs.Count];
                    _procs.Add(Spawn(dev));
                }
            }
        }

        private Process Spawn(int device)
        {
            string exe = Path.Combine(AppContext.BaseDirectory, _workerExeName);
            string args = $"-d {device} -q \"{_queueDir}\"{(_mock ? " --mock" : "")}";
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                Arguments = args,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            var p = new Process { StartInfo = psi };
            p.Start();
            return p;
        }

        public int LiveWorkerCount()
        {
            lock (_sync) { _procs.RemoveAll(p => p.HasExited); return _procs.Count; }
        }

        public void Shutdown()
        {
            lock (_sync)
                foreach (var p in _procs)
                    try { if (!p.HasExited) p.Kill(true); } catch { }
        }
    }
}
