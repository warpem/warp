using System;
using System.Diagnostics;
using System.IO;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// A marker in the queue directory saying one worker process holds a host's GPU for an exclusive task.
    /// Other workers on that GPU check it before claiming and wait; a marker whose process has exited is
    /// cleared, so a crashed worker cannot block its GPU.
    /// </summary>
    public sealed class DeviceLock : IDisposable
    {
        readonly string _path;

        DeviceLock(string path) { _path = path; }

        static string PathFor(QueueLayout layout, string host, int device) => Path.Combine(layout.Exclusive, $"{host}-gpu{device}");

        public static DeviceLock Acquire(QueueLayout layout, string host, int device, int processId)
        {
            Directory.CreateDirectory(layout.Exclusive);
            string path = PathFor(layout, host, device);
            File.WriteAllText(path + ".writing", processId.ToString());
            File.Move(path + ".writing", path, true);
            return new DeviceLock(path);
        }

        /// <summary>True while another live process on this host holds the GPU.</summary>
        public static bool IsHeldByOther(QueueLayout layout, string host, int device, int processId)
        {
            string path = PathFor(layout, host, device);
            int holder;
            try { holder = int.Parse(File.ReadAllText(path).Trim()); }
            catch (Exception e) when (e is IOException or FormatException or UnauthorizedAccessException) { return false; }
            if (holder == processId) return false;
            if (host == Environment.MachineName && !IsAlive(holder))
            {
                try { File.Delete(path); } catch (IOException) { }
                return false;
            }
            return true;
        }

        static bool IsAlive(int processId)
        {
            try { using var process = Process.GetProcessById(processId); return !process.HasExited; }
            catch (ArgumentException) { return false; }
            catch (InvalidOperationException) { return false; }
        }

        public void Dispose()
        {
            try { File.Delete(_path); } catch (IOException) { }
        }
    }
}
