using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// Writes monotonically increasing tick files (prefix + sequence) into a
    /// directory, keeping only the latest. Used for manager and worker
    /// heartbeats. Spec §8.
    /// </summary>
    public class HeartbeatWriter
    {
        private readonly string _dir;
        private readonly string _prefix;
        private long _seq;

        public HeartbeatWriter(string dir, string prefix)
        {
            _dir = dir;
            _prefix = prefix;
            Directory.CreateDirectory(dir);
            _seq = HeartbeatReader.MaxSequence(dir, prefix); // resume after restart
            if (_seq < 0) _seq = 0;
        }

        public void WriteTick()
        {
            _seq++;
            string newPath = Path.Combine(_dir, _prefix + _seq.ToString("D12"));
            File.WriteAllText(newPath, "");
            foreach (string old in Directory.GetFiles(_dir, _prefix + "*"))
                if (!string.Equals(old, newPath, StringComparison.Ordinal))
                    try { File.Delete(old); } catch { }
        }
    }

    public static class HeartbeatReader
    {
        /// <summary>Highest sequence number present, or -1 if none.</summary>
        public static long MaxSequence(string dir, string prefix)
        {
            if (!Directory.Exists(dir)) return -1;
            long max = -1;
            foreach (string f in Directory.GetFiles(dir, prefix + "*"))
            {
                string name = Path.GetFileName(f);
                string num = name.Substring(prefix.Length);
                if (long.TryParse(num, out long v) && v > max) max = v;
            }
            return max;
        }
    }

    /// <summary>
    /// Observes a heartbeat directory and decides, using the OBSERVER's own
    /// clock, whether the writer has stalled. No cross-node clock comparison.
    /// </summary>
    public class HeartbeatMonitor
    {
        private readonly string _dir;
        private readonly string _prefix;
        private readonly long _timeoutMs;
        private readonly long _startupGraceMs;
        private readonly Stopwatch _sinceStart = Stopwatch.StartNew();

        private long _lastSeq = -1;
        private long _lastAdvanceMs;

        public HeartbeatMonitor(string dir, string prefix, long timeoutMs, long startupGraceMs = 0)
        {
            _dir = dir;
            _prefix = prefix;
            _timeoutMs = timeoutMs;
            _startupGraceMs = startupGraceMs;
            _lastAdvanceMs = 0;
        }

        public void Observe()
        {
            long seq = HeartbeatReader.MaxSequence(_dir, _prefix);
            if (seq > _lastSeq)
            {
                _lastSeq = seq;
                _lastAdvanceMs = _sinceStart.ElapsedMilliseconds;
            }
        }

        public bool IsStalled()
        {
            // Still within startup grace and never saw a tick -> alive.
            if (_lastSeq < 0)
                return _sinceStart.ElapsedMilliseconds > _startupGraceMs;

            return _sinceStart.ElapsedMilliseconds - _lastAdvanceMs > _timeoutMs;
        }
    }
}
