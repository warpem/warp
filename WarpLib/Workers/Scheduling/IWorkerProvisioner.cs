namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Strategy for keeping the worker pool populated. Local mode spawns
    /// processes; cluster mode is a no-op because Relay provisions workers (spec §7).
    /// </summary>
    public interface IWorkerProvisioner
    {
        /// <summary>Ensure ~target workers exist. Implementation decides how.</summary>
        void EnsureWorkers(int target);

        /// <summary>How many workers this provisioner currently believes are alive.</summary>
        int LiveWorkerCount();

        /// <summary>Tear down any workers this provisioner owns.</summary>
        void Shutdown();
    }
}
