using System.IO;

namespace Warp.Workers.Queue
{
    /// <summary>
    /// Resolves the subpaths of a queue directory. Pure path construction;
    /// no I/O except EnsureDirectories(). See the work-distribution spec §4.
    /// </summary>
    public class QueueLayout
    {
        public string Root { get; }

        public QueueLayout(string root) { Root = root; }

        public string Pending   => Path.Combine(Root, "pending");
        public string Running   => Path.Combine(Root, "running");
        public string Done      => Path.Combine(Root, "done");
        public string Failed    => Path.Combine(Root, "failed");
        public string Poisoned  => Path.Combine(Root, "poisoned");
        public string Heartbeat => Path.Combine(Root, "heartbeat");
        public string Sick      => Path.Combine(Root, "sick");
        public string Blacklist => Path.Combine(Root, "blacklisted_nodes");
        public string Logs        => Path.Combine(Root, "logs");
        public string Lock        => Path.Combine(Root, "pool.lock");
        public string ManagerState => Path.Combine(Root, "manager.state.json");

        public string RunningFor(string workerId) => Path.Combine(Running, workerId);

        public void EnsureDirectories()
        {
            foreach (string d in new[] { Pending, Running, Done, Failed, Poisoned, Heartbeat, Sick, Blacklist, Logs })
                Directory.CreateDirectory(d);
        }
    }
}
