using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;

namespace WarpWorker2;

static partial class WorkerProcess
{
    /// <summary>
    /// While a movie is processed, the TIFF or EER input of the task this worker will try to claim next is read into memory,
    /// so that task's LoadStack does not wait for storage. Movies stream in from network storage while the GPU works on
    /// the previous one, and storage is otherwise idle then. If another worker claims the task first, the read was
    /// not in vain on a shared node: it leaves the file in this node's file cache.
    /// </summary>
    static class MoviePrefetch
    {
        static FileInMemory Buffer = new();
        static Task Reading = Task.CompletedTask;
        static volatile string PendingPath = "";
        static long ReadLength;
        static DateTime ReadWriteTime;

        /// <summary>The task whose input is being read; ClaimOne tries it first.</summary>
        public static string PreferredTaskId => preferredTaskId;
        static volatile string preferredTaskId;

        /// <summary>
        /// Picks a pending task as ClaimOne would and reads its movie, unless a read is still running. All of it runs in
        /// the background and nothing escapes: a malformed or vanished pending task must not fail the current one.
        /// A session's movies share a format and only TIFF and EER movies are read into memory, so others skip the peek.
        /// </summary>
        public static void StartNext(TaskQueue queue, IEnumerable<string> stages, string currentPath)
        {
            if (queue == null || !FileInMemory.Enabled || !Reading.IsCompleted || !FileInMemory.IsMovie(currentPath)) return;
            preferredTaskId = null; PendingPath = "";
            var buffer = Buffer;
            string[] allowed = stages?.ToArray();
            Reading = Task.Run(() =>
            {
                try
                {
                    TaskItem next = queue.PeekOne(allowed);
                    string path = next?.Main?.FirstOrDefault(c => c.Name == WorkerCommandNames.LoadStack ||
                                                                  c.Name == WorkerCommandNames.MovieExportParticlesFromFile)?.Content?[0] as string;
                    if (path == null || path == currentPath || !FileInMemory.IsMovie(path)) return;
                    var info = new FileInfo(path);
                    if (info.Length > FileInMemory.MaxBytes) return;
                    ReadLength = info.Length; ReadWriteTime = info.LastWriteTimeUtc;
                    PendingPath = path; preferredTaskId = next.TaskId;
                    buffer.Read(path);
                }
                catch (Exception) { PendingPath = ""; preferredTaskId = null; } // LoadStack reads the file itself
            });
        }

        /// <summary>
        /// If the file being or already read is this path and has not changed since, swaps it into target (waiting
        /// for an unfinished read) and returns true.
        /// </summary>
        public static bool TryTake(string path, ref FileInMemory target)
        {
            if (PendingPath != path) return false;
            Reading.Wait();
            PendingPath = ""; preferredTaskId = null;
            if (Buffer.Path != path) return false;
            var info = new FileInfo(path);
            if (!info.Exists || info.Length != ReadLength || info.Length != Buffer.Length || info.LastWriteTimeUtc != ReadWriteTime) return false;
            (target, Buffer) = (Buffer, target);
            return true;
        }
    }
}
