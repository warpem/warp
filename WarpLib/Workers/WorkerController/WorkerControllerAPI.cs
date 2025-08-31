using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Warp.Tools;

namespace Warp.Workers.WorkerController
{
    [ApiController]
    [Route("api/workers")]
    public class WorkerControllerAPI : ControllerBase
    {
        private readonly WorkerControllerService _controllerService;
        private readonly ILogger<WorkerControllerAPI> _logger;

        public WorkerControllerAPI(WorkerControllerService controllerService, ILogger<WorkerControllerAPI> logger = null)
        {
            _controllerService = controllerService;
            _logger = logger;
        }

        [HttpPost("register")]
        public ActionResult<WorkerRegistrationResponse> RegisterWorker([FromBody] WorkerRegistrationRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.Host))
                    return BadRequest("Host is required");

                var response = _controllerService.RegisterWorker(request);
                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error registering worker");
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{workerId}/poll")]
        public ActionResult<PollResponse> PollForTask(string workerId, [FromBody] PollRequest pollRequest = null, [FromHeader] string authorization = null)
        {
            try
            {
                if (string.IsNullOrEmpty(workerId))
                    return BadRequest("Worker ID is required");

                // Basic token validation could be added here
                // if (!ValidateWorkerToken(workerId, authorization))
                //     return Unauthorized();

                var response = _controllerService.PollForTask(workerId, pollRequest);
                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error polling for task for worker {WorkerId}", workerId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{workerId}/heartbeat")]
        public IActionResult UpdateHeartbeat(string workerId, [FromBody] HeartbeatRequest heartbeat)
        {
            try
            {
                if (string.IsNullOrEmpty(workerId))
                    return BadRequest("Worker ID is required");

                var success = _controllerService.UpdateHeartbeat(workerId, heartbeat);
                if (!success)
                    return NotFound("Worker not found");

                return Ok();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error updating heartbeat for worker {WorkerId}", workerId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpPost("{workerId}/tasks/{taskId}/status")]
        public IActionResult UpdateTaskStatus(string workerId, string taskId, [FromBody] TaskUpdateRequest update)
        {
            try
            {
                if (string.IsNullOrEmpty(workerId))
                    return BadRequest("Worker ID is required");

                if (string.IsNullOrEmpty(taskId))
                    return BadRequest("Task ID is required");

                var success = _controllerService.UpdateTaskStatus(workerId, taskId, update);
                if (!success)
                    return NotFound("Worker or task not found");

                return Ok();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error updating task status for worker {WorkerId}, task {TaskId}", workerId, taskId);
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpGet("status")]
        public IActionResult GetStatus()
        {
            try
            {
                var status = _controllerService.GetStatus();
                return Ok(status);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error getting controller status");
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpGet("active")]
        public IActionResult GetActiveWorkers()
        {
            try
            {
                var workers = _controllerService.GetActiveWorkers();
                return Ok(workers);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error getting active workers");
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpGet("tasks")]
        public IActionResult GetActiveTasks()
        {
            try
            {
                var tasks = _controllerService.GetActiveTasks();
                return Ok(tasks);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error getting active tasks");
                return StatusCode(500, "Internal server error");
            }
        }

        [HttpGet("tasks/{taskId}")]
        public IActionResult GetTask(string taskId)
        {
            try
            {
                var task = _controllerService.GetTask(taskId);
                if (task == null)
                    return NotFound("Task not found");

                return Ok(task);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error getting task {TaskId}", taskId);
                return StatusCode(500, "Internal server error");
            }
        }
    }
}