using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Warp;

namespace WarpCore.Core.Processing
{
    public class SettingsChangeHandler
    {
        private readonly ILogger<SettingsChangeHandler> _logger;

        public SettingsChangeHandler(ILogger<SettingsChangeHandler> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Analyze the impact of settings changes and trigger appropriate responses
        /// </summary>
        public SettingsChangeImpact AnalyzeSettingsChange(OptionsWarp oldSettings, OptionsWarp newSettings)
        {
            var impact = new SettingsChangeImpact();

            // Compare processing options to determine what's affected
            impact.CTFChanged = AreProcessingOptionsDifferent(
                oldSettings?.GetProcessingMovieCTF(),
                newSettings?.GetProcessingMovieCTF()
            );

            impact.MovementChanged = AreProcessingOptionsDifferent(
                oldSettings?.GetProcessingMovieMovement(),
                newSettings?.GetProcessingMovieMovement()
            );

            impact.PickingChanged = AreProcessingOptionsDifferent(
                oldSettings?.GetProcessingBoxNet(),
                newSettings?.GetProcessingBoxNet()
            );

            impact.ExportChanged = AreProcessingOptionsDifferent(
                oldSettings?.GetProcessingMovieExport(),
                newSettings?.GetProcessingMovieExport()
            );

            // Check if processing steps were enabled/disabled
            impact.CTFToggled = (oldSettings?.ProcessCTF != newSettings?.ProcessCTF);
            impact.MovementToggled = (oldSettings?.ProcessMovement != newSettings?.ProcessMovement);
            impact.PickingToggled = (oldSettings?.ProcessPicking != newSettings?.ProcessPicking);

            impact.HasAnyChanges = impact.CTFChanged || impact.MovementChanged || 
                                 impact.PickingChanged || impact.ExportChanged ||
                                 impact.CTFToggled || impact.MovementToggled || impact.PickingToggled;

            if (impact.HasAnyChanges)
            {
                _logger.LogInformation($"Settings change detected - CTF: {impact.CTFChanged}, " +
                                     $"Movement: {impact.MovementChanged}, Picking: {impact.PickingChanged}, " +
                                     $"Export: {impact.ExportChanged}");
            }

            return impact;
        }

        /// <summary>
        /// Apply settings change to the processing queue
        /// </summary>
        public void ApplySettingsChange(
            ProcessingQueue queue,
            OptionsWarp oldSettings,
            OptionsWarp newSettings,
            SettingsChangeImpact impact)
        {
            if (!impact.HasAnyChanges)
                return;

            _logger.LogInformation("Applying settings changes to processing queue");

            // Refresh all movie statuses with the new settings
            queue.RefreshAllStatuses(newSettings);

            // Log statistics about the impact
            var allMovies = queue.GetAllMovies();
            var statusCounts = allMovies
                .GroupBy(m => newSettings.GetMovieProcessingStatus(m, false))
                .ToDictionary(g => g.Key, g => g.Count());

            _logger.LogInformation($"After settings change: " +
                                 $"Processed: {statusCounts.GetValueOrDefault(Warp.ProcessingStatus.Processed, 0)}, " +
                                 $"Outdated: {statusCounts.GetValueOrDefault(Warp.ProcessingStatus.Outdated, 0)}, " +
                                 $"Unprocessed: {statusCounts.GetValueOrDefault(Warp.ProcessingStatus.Unprocessed, 0)}, " +
                                 $"LeaveOut: {statusCounts.GetValueOrDefault(Warp.ProcessingStatus.LeaveOut, 0)}");
        }

        /// <summary>
        /// Determine if settings change should trigger immediate redistribution
        /// </summary>
        public bool ShouldTriggerImmediateRedistribution(SettingsChangeImpact impact)
        {
            // Immediate redistribution if processing steps were toggled on/off
            return impact.CTFToggled || impact.MovementToggled || impact.PickingToggled;
        }

        private bool AreProcessingOptionsDifferent(object oldOptions, object newOptions)
        {
            // Handle null cases
            if (oldOptions == null && newOptions == null) return false;
            if (oldOptions == null || newOptions == null) return true;

            // Use the built-in equality comparison that exists in processing options classes
            return !oldOptions.Equals(newOptions);
        }
    }

    public class SettingsChangeImpact
    {
        public bool CTFChanged { get; set; }
        public bool MovementChanged { get; set; }
        public bool PickingChanged { get; set; }
        public bool ExportChanged { get; set; }
        
        public bool CTFToggled { get; set; }
        public bool MovementToggled { get; set; }
        public bool PickingToggled { get; set; }

        public bool HasAnyChanges { get; set; }

        public override string ToString()
        {
            return $"SettingsChangeImpact(CTF:{CTFChanged}/{CTFToggled}, " +
                   $"Movement:{MovementChanged}/{MovementToggled}, " +
                   $"Picking:{PickingChanged}/{PickingToggled}, " +
                   $"Export:{ExportChanged})";
        }
    }
}