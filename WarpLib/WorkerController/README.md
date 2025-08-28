# Controller-Based Worker Architecture

This implementation provides a reverse-direction REST architecture for Warp workers, designed specifically for HPC environments where workers may be on remote machines with restrictive firewalls.

## Architecture Overview

### Traditional Architecture (old)
- **Main Process** ← REST calls ← **Workers** (each serving own REST API on random ports)
- Workers need inbound ports open
- Named pipes for port communication
- Direct main-to-worker communication

### Controller Architecture (new)
- **Main Process** (with Controller Service) ← HTTP polls ← **Workers** (HTTP clients only)
- Workers only need outbound connections
- Single controller port for all communication
- Hub-and-spoke model with task queue

## Benefits

1. **Firewall-friendly**: Workers only make outbound connections
2. **HPC-compatible**: Works in restrictive network environments
3. **Scalable**: Single controller handles multiple workers
4. **Robust**: Built-in heartbeat monitoring and reconnection logic
5. **Queue management**: Centralized task distribution and monitoring

## Usage

### Using Controller-Based Workers

```csharp
// Create controller-based wrapper
var controllerWrapper = new ControllerBasedWorkerWrapper();

// Start controller service
int controllerPort = await controllerWrapper.StartControllerAsync(0); // 0 = auto-assign port
Console.WriteLine($"Controller started on port {controllerPort}");

// Add workers
string workerId1 = await controllerWrapper.AddWorkerAsync(0); // Device 0
string workerId2 = await controllerWrapper.AddWorkerAsync(1); // Device 1

// Use exactly like traditional WorkerWrapper
controllerWrapper.LoadStack("data/movie.mrc", 1.0m, 0);
controllerWrapper.MovieProcessCTF("data/movie.mrc", options);

// Monitor status
var status = controllerWrapper.GetControllerStatus();
var activeWorkers = controllerWrapper.GetActiveWorkers();
var queueLength = controllerWrapper.GetQueueLength();

// Cleanup
controllerWrapper.Dispose();
```

### Manual Worker Launch (for remote machines)

```bash
# Launch worker pointing to controller
./WarpWorker -d 0 --controller controller.hostname.com:8080 -s

# Traditional mode still works
./WarpWorker -d 0 -p 0 --pipe pipename -s
```

## Implementation Details

### Components

1. **WorkerControllerService**: Core task queue and worker management
2. **WorkerControllerAPI**: REST API endpoints for worker communication
3. **WorkerControllerHost**: ASP.NET Core host service
4. **ControllerBasedWorkerWrapper**: Drop-in replacement for WorkerWrapper
5. **ControllerClient**: Worker-side HTTP client for controller communication

### Communication Protocol

1. **Registration**: `POST /api/workers/register`
2. **Polling**: `GET /api/workers/{id}/poll` (long-polling)
3. **Heartbeat**: `POST /api/workers/{id}/heartbeat`
4. **Status Updates**: `POST /api/workers/{id}/tasks/{taskId}/status`

### Error Handling

- **Connection loss**: Automatic reconnection with exponential backoff
- **Worker failure**: Task reassignment to available workers
- **Controller failure**: Workers attempt reconnection
- **Timeout detection**: Automatic cleanup of stale workers and tasks

### Monitoring Endpoints

- `GET /api/workers/status` - Overall system status
- `GET /api/workers/active` - List of active workers
- `GET /api/workers/tasks` - Active tasks
- `GET /api/workers/tasks/{id}` - Specific task details

## Configuration

### Controller Settings
- Heartbeat timeout: 2 minutes
- Task timeout: 30 minutes
- Max reconnection attempts: 10
- Polling interval: 5 seconds (idle), 1 second (after task)

### Worker Settings
- Registration retries: 5 attempts with exponential backoff
- Heartbeat interval: 30 seconds
- Reconnection: Up to 10 attempts with exponential backoff

## Testing

```bash
# Run basic functionality test
./TestControllerArchitecture

# Run multi-worker test
./TestControllerArchitecture --multi

# Run stress test
./TestControllerArchitecture --stress
```

## Compatibility

- **Backward compatible**: Traditional WorkerWrapper continues to work
- **Feature complete**: All WorkerWrapper methods implemented
- **Cross-platform**: Works on Windows, Linux, macOS
- **HPC-ready**: Tested with SLURM and other job schedulers

## Migration Guide

### From Traditional to Controller-Based

```csharp
// Old way
var worker = new WorkerWrapper(deviceId: 0);
worker.LoadStack("file.mrc", 1.0m, 0);

// New way
var controllerWrapper = new ControllerBasedWorkerWrapper();
await controllerWrapper.StartControllerAsync(0);
await controllerWrapper.AddWorkerAsync(0);
controllerWrapper.LoadStack("file.mrc", 1.0m, 0);
```

The API is identical, making migration straightforward.

## Performance Considerations

- **Polling overhead**: Minimal due to long-polling and adaptive intervals
- **Memory usage**: Centralized task queue vs distributed state
- **Latency**: Slightly higher due to polling, but acceptable for most workloads
- **Throughput**: Comparable to traditional approach for typical Warp operations

## Security Notes

- Worker authentication via tokens (basic implementation)
- All communication over HTTP (HTTPS recommended for production)
- No sensitive data logged in normal operation
- Workers identified by generated UUIDs