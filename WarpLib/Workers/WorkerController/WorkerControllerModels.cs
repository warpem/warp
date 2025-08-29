using System;
using System.Collections.Generic;
using Warp.Tools;

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

    public enum TaskStatus
    {
        Pending,
        Assigned,
        Running,
        Completed,
        Failed,
        Cancelled
    }

    public class WorkerInfo
    {
        public string WorkerId { get; set; }
        public string Host { get; set; }
        public int DeviceId { get; set; }
        public WorkerStatus Status { get; set; }
        public DateTime LastHeartbeat { get; set; }
        public DateTime RegisteredAt { get; set; }
        public string CurrentTaskId { get; set; }
        public long FreeMemoryMB { get; set; }
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

    public class TaskInfo
    {
        public string TaskId { get; set; }
        public string WorkerId { get; set; }
        public TaskStatus Status { get; set; }
        public NamedSerializableObject Command { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? AssignedAt { get; set; }
        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        public string ErrorMessage { get; set; }
        public object Result { get; set; }

        public TaskInfo(NamedSerializableObject command)
        {
            TaskId = Guid.NewGuid().ToString();
            Command = command;
            Status = TaskStatus.Pending;
            CreatedAt = DateTime.UtcNow;
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
        public TaskInfo Task { get; set; }
        public int NextPollDelayMs { get; set; } = 5000;
    }

    public class TaskUpdateRequest
    {
        public TaskStatus Status { get; set; }
        public string ErrorMessage { get; set; }
        public object Result { get; set; }
        public string ProgressMessage { get; set; }
        public double? ProgressPercentage { get; set; }
    }

    public class PollRequest
    {
        public List<LogEntry> ConsoleLines { get; set; } = new List<LogEntry>();
    }

    public class HeartbeatRequest
    {
        public WorkerStatus Status { get; set; }
        public long FreeMemoryMB { get; set; }
        public string CurrentTaskId { get; set; }
    }
}