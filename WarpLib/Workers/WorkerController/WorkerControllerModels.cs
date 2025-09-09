using System;
using System.Collections.Generic;
using Warp.Tools;
using Warp.Workers.Distribution;

namespace Warp.Workers.WorkerController
{
    public enum WorkerStatus
    {
        Registering,
        Idle,
        Working,
        Offline,
        Failed
    }

    public class WorkerInfo
    {
        public string WorkerId { get; set; }
        public string Host { get; set; }
        public int DeviceId { get; set; }
        public WorkerStatus Status { get; set; }
        public DateTime LastHeartbeat { get; set; }
        public DateTime RegisteredAt { get; set; }
        public string CurrentWorkPackageId { get; set; }
        public string Token { get; set; }

        public WorkerInfo()
        {
            WorkerId = Guid.NewGuid().ToString();
            Status = WorkerStatus.Registering;
            RegisteredAt = DateTime.UtcNow;
            LastHeartbeat = DateTime.UtcNow;
            Token = Guid.NewGuid().ToString();
        }
    }


    public class WorkerRegistrationRequest
    {
        public string Host { get; set; }
        public int DeviceId { get; set; }
        public long FreeMemoryMB { get; set; }
    }

    public class WorkerRegistrationResponse
    {
        public string WorkerId { get; set; }
        public string Token { get; set; }
        public int PollIntervalMs { get; set; } = 5000;
        public int HeartbeatIntervalMs { get; set; } = 30000;
    }

    public class PollResponse
    {
        public WorkPackage WorkPackage { get; set; }
        public int NextPollDelayMs { get; set; } = 5000;
    }

    public class PollRequest
    {
        public List<LogEntry> ConsoleLines { get; set; } = new List<LogEntry>();
    }

    public class HeartbeatRequest
    {
        public WorkerStatus Status { get; set; }
        public long FreeMemoryMB { get; set; }
        public string CurrentWorkPackageId { get; set; }
    }
}