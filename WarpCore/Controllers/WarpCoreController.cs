using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Workers.WorkerController;
using WarpCore.Core;

namespace WarpCore.Controllers
{
    /// <summary>
    /// REST API controller that provides endpoints for managing the WarpCore distributed
    /// image processing system. Exposes functionality for settings management, processing control,
    /// worker monitoring, change tracking, and data export operations.
    /// </summary>
    [ApiController]
    [Route("api")]
    public class WarpCoreController : ControllerBase
    {
        private readonly ILogger<WarpCoreController> _logger;
        private readonly ProcessingOrchestrator _orchestrator;
        private readonly WorkerPool _workerPool;
        private readonly ChangeTracker _changeTracker;
        private readonly FileDiscoverer _fileDiscoverer;

        /// <summary>
        /// Initializes the controller with all required services for processing management.
        /// </summary>
        /// <param name="logger">Logger for recording API operations and errors</param>
        /// <param name="orchestrator">Main processing orchestrator that coordinates all operations</param>
        /// <param name="workerPool">Worker pool manager for distributed processing</param>
        /// <param name="changeTracker">Change tracking service for monitoring processing state</param>
        /// <param name="fileDiscoverer">File discovery service for monitoring input data</param>
        public WarpCoreController(
            ILogger<WarpCoreController> logger,
            ProcessingOrchestrator orchestrator,
            WorkerPool workerPool,
            ChangeTracker changeTracker,
            FileDiscoverer fileDiscoverer)
        {
            _logger = logger;
            _orchestrator = orchestrator;
            _workerPool = workerPool;
            _changeTracker = changeTracker;
            _fileDiscoverer = fileDiscoverer;
        }

        // Settings endpoints
        /// <summary>
        /// Updates processing settings for the system. Changes trigger re-evaluation of all
        /// movies to determine which ones need reprocessing based on the new settings.
        /// </summary>
        /// <param name="settings">New processing settings including CTF, movement, picking, and export options</param>
        /// <returns>Success message or error response</returns>
        [HttpPut("settings")]
        public async Task<IActionResult> UpdateSettings([FromBody] OptionsWarp settings)
        {
            try
            {
                _orchestrator.UpdateSettings(settings);
                return Ok(new { message = "Settings updated successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating settings");
                return StatusCode(500, new { error = "Failed to update settings" });
            }
        }

        /// <summary>
        /// Retrieves the current processing settings being used by the system.
        /// </summary>
        /// <returns>Current processing settings configuration</returns>
        [HttpGet("settings")]
        public async Task<IActionResult> GetSettings()
        {
            try
            {
                var settings = _orchestrator.GetCurrentSettings();
                return Ok(settings);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting settings");
                return StatusCode(500, new { error = "Failed to get settings" });
            }
        }

        // Processing control endpoints
        /// <summary>
        /// Starts the processing system. Initializes file discovery and begins the main processing loop
        /// that continuously processes discovered movies using available workers.
        /// </summary>
        /// <returns>Success message or error response</returns>
        [HttpPost("processing/start")]
        public async Task<IActionResult> StartProcessing()
        {
            try
            {
                await _orchestrator.StartProcessingAsync();
                return Ok(new { message = "Processing started" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting processing");
                return StatusCode(500, new { error = "Failed to start processing" });
            }
        }

        /// <summary>
        /// Pauses the processing system. Stops the main processing loop and waits for
        /// any currently running tasks to complete before fully stopping.
        /// </summary>
        /// <returns>Success message or error response</returns>
        [HttpPost("processing/pause")]
        public async Task<IActionResult> PauseProcessing()
        {
            try
            {
                await _orchestrator.PauseProcessingAsync();
                return Ok(new { message = "Processing paused" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error pausing processing");
                return StatusCode(500, new { error = "Failed to pause processing" });
            }
        }

        /// <summary>
        /// Gets the current processing status including statistics about processed, failed,
        /// and queued items, along with active worker count and processing state.
        /// </summary>
        /// <returns>Processing status with statistics and timestamps</returns>
        [HttpGet("processing/status")]
        public async Task<IActionResult> GetProcessingStatus()
        {
            try
            {
                var statistics = _orchestrator.GetStatistics();
                var response = new
                {
                    isProcessing = _orchestrator.IsProcessing,
                    statistics = statistics,
                    lastModified = _changeTracker.LastModified
                };
                return Ok(response);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting processing status");
                return StatusCode(500, new { error = "Failed to get processing status" });
            }
        }

        // Worker management endpoints
        /// <summary>
        /// Retrieves information about all registered workers including their status,
        /// connection details, and current task assignments.
        /// </summary>
        /// <returns>List of worker information objects</returns>
        [HttpGet("workers")]
        public async Task<IActionResult> GetWorkers()
        {
            try
            {
                var workers = _workerPool.GetWorkers();
                var workerInfos = workers.Select(w => new WorkerInfo
                {
                    WorkerId = w.WorkerId,
                    DeviceId = w.DeviceID,
                    Host = w.Host ?? "localhost",
                    Status = w.Status,
                    LastHeartbeat = w.LastHeartbeat,
                    RegisteredAt = w.ConnectedAt,
                    CurrentTaskId = w.CurrentTask,
                    FreeMemoryMB = 0 // WorkerWrapper doesn't track memory
                }).ToList();
                
                return Ok(workerInfos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting workers");
                return StatusCode(500, new { error = "Failed to get workers" });
            }
        }

        /// <summary>
        /// Retrieves execution logs from a specific worker for debugging and monitoring purposes.
        /// </summary>
        /// <param name="workerId">Unique identifier of the worker to get logs from</param>
        /// <returns>Worker logs as a list of log messages</returns>
        [HttpGet("workers/{workerId}/logs")]
        public async Task<IActionResult> GetWorkerLogs(string workerId)
        {
            try
            {
                var logs = await _workerPool.GetWorkerLogsAsync(workerId);
                return Ok(new { workerId = workerId, logs = logs });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting logs for worker {workerId}");
                return StatusCode(500, new { error = "Failed to get worker logs" });
            }
        }

        // Change tracking endpoints
        /// <summary>
        /// Gets the timestamp of the last change to the processing state.
        /// Used by clients to detect when processing results have been updated.
        /// </summary>
        /// <returns>Timestamp of the last processing state change</returns>
        [HttpGet("items/timestamp")]
        public async Task<IActionResult> GetTimestamp()
        {
            try
            {
                return Ok(new { timestamp = _changeTracker.LastModified });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting timestamp");
                return StatusCode(500, new { error = "Failed to get timestamp" });
            }
        }

        /// <summary>
        /// Gets a summary of processing results including counts of processed,
        /// failed, and queued movies along with the last modification timestamp.
        /// </summary>
        /// <returns>Processing summary with item counts and timestamps</returns>
        [HttpGet("items/summary")]
        public async Task<IActionResult> GetSummary()
        {
            try
            {
                var summary = await _changeTracker.GetSummaryAsync();
                return Ok(summary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting summary");
                return StatusCode(500, new { error = "Failed to get summary" });
            }
        }

        // Export endpoints
        /// <summary>
        /// Initiates export of processed micrographs in the specified format.
        /// Currently returns a placeholder response as implementation is pending.
        /// </summary>
        /// <param name="request">Export request specifying output path, format, and processing options</param>
        /// <returns>Export job information with unique job ID</returns>
        [HttpPost("export/micrographs")]
        public async Task<IActionResult> ExportMicrographs([FromBody] ExportMicrographsRequest request)
        {
            try
            {
                // TODO: Implement micrograph export logic
                return Ok(new { message = "Micrograph export started", jobId = Guid.NewGuid() });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting micrographs");
                return StatusCode(500, new { error = "Failed to export micrographs" });
            }
        }

        /// <summary>
        /// Initiates export of picked particles from processed movies.
        /// Currently returns a placeholder response as implementation is pending.
        /// </summary>
        /// <param name="request">Export request specifying output path, box size, and processing options</param>
        /// <returns>Export job information with unique job ID</returns>
        [HttpPost("export/particles")]
        public async Task<IActionResult> ExportParticles([FromBody] ExportParticlesRequest request)
        {
            try
            {
                // TODO: Implement particle export logic
                return Ok(new { message = "Particle export started", jobId = Guid.NewGuid() });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting particles");
                return StatusCode(500, new { error = "Failed to export particles" });
            }
        }

    }

    // Request models for export endpoints
    /// <summary>
    /// Request model for micrograph export operations. Specifies output location,
    /// file format, and processing options for exporting processed micrographs.
    /// </summary>
    public class ExportMicrographsRequest
    {
        /// <summary>
        /// Directory path where exported micrographs will be saved.
        /// </summary>
        public string OutputPath { get; set; }
        
        /// <summary>
        /// Output file format for micrographs (default: "mrc").
        /// </summary>
        public string Format { get; set; } = "mrc";
        
        /// <summary>
        /// Whether to export frame-averaged micrographs.
        /// </summary>
        public bool DoAverage { get; set; } = true;
        
        /// <summary>
        /// Whether to export full frame stacks.
        /// </summary>
        public bool DoStack { get; set; } = false;
    }

    /// <summary>
    /// Request model for particle export operations. Specifies output location,
    /// box size, and processing options for exporting picked particles.
    /// </summary>
    public class ExportParticlesRequest
    {
        /// <summary>
        /// Directory path where exported particles will be saved.
        /// </summary>
        public string OutputPath { get; set; }
        
        /// <summary>
        /// Size of the particle box in pixels (default: 128).
        /// </summary>
        public int BoxSize { get; set; } = 128;
        
        /// <summary>
        /// Whether to export frame-averaged particles.
        /// </summary>
        public bool DoAverage { get; set; } = true;
        
        /// <summary>
        /// Whether to normalize particle intensities.
        /// </summary>
        public bool Normalize { get; set; } = true;
    }
}