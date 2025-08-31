using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Warp;
using WarpCore.Core;

namespace WarpCore.Controllers
{
    [ApiController]
    [Route("api")]
    public class WarpCoreController : ControllerBase
    {
        private readonly ILogger<WarpCoreController> _logger;
        private readonly ProcessingOrchestrator _orchestrator;
        private readonly WorkerPool _workerPool;
        private readonly ChangeTracker _changeTracker;
        private readonly FileDiscoverer _fileDiscoverer;

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
        [HttpGet("workers")]
        public async Task<IActionResult> GetWorkers()
        {
            try
            {
                var workers = _workerPool.GetWorkers();
                return Ok(workers);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting workers");
                return StatusCode(500, new { error = "Failed to get workers" });
            }
        }

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
    public class ExportMicrographsRequest
    {
        public string OutputPath { get; set; }
        public string Format { get; set; } = "mrc";
        public bool DoAverage { get; set; } = true;
        public bool DoStack { get; set; } = false;
    }

    public class ExportParticlesRequest
    {
        public string OutputPath { get; set; }
        public int BoxSize { get; set; } = 128;
        public bool DoAverage { get; set; } = true;
        public bool Normalize { get; set; } = true;
    }
}