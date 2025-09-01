using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Warp;

namespace WarpCore.Core.Processing
{
    /// <summary>
    /// Handles analysis and application of processing settings changes.
    /// Determines the impact of settings changes on existing movies and
    /// coordinates updates to processing queues and task priorities.
    /// </summary>
    public class SettingsChangeHandler
    {
        private readonly ILogger<SettingsChangeHandler> _logger;

        /// <summary>
        /// Initializes a new settings change handler.
        /// </summary>
        /// <param name="logger">Logger for recording settings change operations</param>
        public SettingsChangeHandler(ILogger<SettingsChangeHandler> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Analyzes the differences between old and new processing settings to determine
        /// what changes occurred and their potential impact on existing movies.
        /// Compares processing options and enabled/disabled states for each processing step.
        /// </summary>
        /// <param name="oldSettings">Previous processing settings</param>
        /// <param name="newSettings">New processing settings to apply</param>
        /// <returns>Impact analysis describing what changed and how it affects processing</returns>
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
        /// Applies the analyzed settings changes to the processing queue.
        /// Refreshes all movie statuses based on new settings and logs the resulting
        /// distribution of movies across different processing states.
        /// </summary>
        /// <param name="queue">Processing queue to update</param>
        /// <param name="oldSettings">Previous processing settings</param>
        /// <param name="newSettings">New processing settings</param>
        /// <param name="impact">Impact analysis from settings comparison</param>
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
        /// Determines if the settings change should trigger immediate work redistribution.
        /// Returns true when processing steps are enabled or disabled, as this significantly
        /// changes which movies need processing and their priority.
        /// </summary>
        /// <param name="impact">Impact analysis from settings comparison</param>
        /// <returns>True if immediate redistribution should be triggered, false otherwise</returns>
        public bool ShouldTriggerImmediateRedistribution(SettingsChangeImpact impact)
        {
            // Immediate redistribution if processing steps were toggled on/off
            return impact.CTFToggled || impact.MovementToggled || impact.PickingToggled;
        }

        /// <summary>
        /// Compares two processing options objects to determine if they are different.
        /// Handles null cases and uses built-in equality comparison for non-null objects.
        /// </summary>
        /// <param name="oldOptions">Previous processing options</param>
        /// <param name="newOptions">New processing options</param>
        /// <returns>True if the options are different, false if they are the same</returns>
        private bool AreProcessingOptionsDifferent(object oldOptions, object newOptions)
        {
            // Handle null cases
            if (oldOptions == null && newOptions == null) return false;
            if (oldOptions == null || newOptions == null) return true;

            // Use the built-in equality comparison that exists in processing options classes
            return !oldOptions.Equals(newOptions);
        }
    }

    /// <summary>
    /// Contains the analysis results of processing settings changes.
    /// Tracks what processing options changed and whether steps were enabled/disabled.
    /// </summary>
    public class SettingsChangeImpact
    {
        /// <summary>
        /// Gets or sets a value indicating whether CTF processing options changed.
        /// </summary>
        public bool CTFChanged { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether movement correction options changed.
        /// </summary>
        public bool MovementChanged { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether particle picking options changed.
        /// </summary>
        public bool PickingChanged { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether export options changed.
        /// </summary>
        public bool ExportChanged { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether CTF processing was enabled or disabled.
        /// </summary>
        public bool CTFToggled { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether movement correction was enabled or disabled.
        /// </summary>
        public bool MovementToggled { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether particle picking was enabled or disabled.
        /// </summary>
        public bool PickingToggled { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether any changes were detected.
        /// </summary>
        public bool HasAnyChanges { get; set; }

        /// <summary>
        /// Returns a string representation of the settings change impact for debugging.
        /// </summary>
        /// <returns>Formatted string showing all change indicators</returns>
        public override string ToString()
        {
            return $"SettingsChangeImpact(CTF:{CTFChanged}/{CTFToggled}, " +
                   $"Movement:{MovementChanged}/{MovementToggled}, " +
                   $"Picking:{PickingChanged}/{PickingToggled}, " +
                   $"Export:{ExportChanged})";
        }
    }
}