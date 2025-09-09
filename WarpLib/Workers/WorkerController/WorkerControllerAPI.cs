using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Warp.Tools;
using Warp.Workers.Distribution;

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
                
                //Console.WriteLine("❤️ from heartbeat");

                return Ok();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error updating heartbeat for worker {WorkerId}", workerId);
                return StatusCode(500, "Internal server error");
            }
        }


        [HttpPost("{workerId}/workpackages/{workPackageId}/status")]
        public IActionResult UpdateWorkPackageStatus(string workerId, string workPackageId, [FromBody] WorkPackageUpdateRequest update)
        {
            try
            {
                if (string.IsNullOrEmpty(workerId))
                    return BadRequest("Worker ID is required");

                if (string.IsNullOrEmpty(workPackageId))
                    return BadRequest("Work package ID is required");

                var success = _controllerService.UpdateWorkPackageStatus(workerId, workPackageId, update);
                if (!success)
                    return NotFound("Worker not found");

                return Ok();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error updating work package status for worker {WorkerId}, package {WorkPackageId}", workerId, workPackageId);
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

    }
}