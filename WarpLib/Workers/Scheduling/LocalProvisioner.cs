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
        private readonly string _logDir;
        private readonly List<(Process proc, int device)> _procs = new();
        private readonly object _sync = new();

        public LocalProvisioner(string queueDir, int[] devices, int perDevice,
                                bool mock = false, string workerExeName = "WarpWorker2",
                                string logDir = null)
        {
            _queueDir = queueDir;
            _devices = devices;
            _perDevice = perDevice;
            _mock = mock;
            _workerExeName = workerExeName;
            _logDir = logDir;
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                _procs.RemoveAll(e => e.proc.HasExited);

                // Build the full slot list (one entry per device×perDevice).
                var slots = new List<int>();
                foreach (int dev in _devices)
                    for (int i = 0; i < _perDevice; i++)
                        slots.Add(dev);
                int cap = Math.Min(target, slots.Count);

                // Find which device slots are already occupied by a live process.
                // Use a mutable copy so we can Remove() matches without consuming
                // the same slot entry for two live processes on the same device.
                var occupied = new List<int>();
                foreach (var e in _procs) occupied.Add(e.device);

                foreach (int dev in slots)
                {
                    if (_procs.Count >= cap) break;
                    if (occupied.Remove(dev)) continue;   // this slot is already filled
                    _procs.Add((Spawn(dev), dev));
                }
            }
        }

        private Process Spawn(int device)
        {
            string exe = Path.Combine(AppContext.BaseDirectory, _workerExeName);
            string args = $"-d {device} -q \"{_queueDir}\"{(_mock ? " --mock" : "")}" +
                          (string.IsNullOrEmpty(_logDir) ? "" : $" --log-dir \"{_logDir}\"");
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
            lock (_sync) { _procs.RemoveAll(e => e.proc.HasExited); return _procs.Count; }
        }

        public void Shutdown()
        {
            lock (_sync)
                foreach (var (proc, _) in _procs)
                    try { if (!proc.HasExited) proc.Kill(true); } catch { }
        }
    }
}
