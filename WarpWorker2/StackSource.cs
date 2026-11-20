namespace WarpWorker2;

/// <summary>
/// The movie file and loading settings a stack was loaded with, and the task that loaded it. A command can only take a
/// stack for its movie when all of these match: another scale, EER grouping or gain correction gives other frames, and
/// a stack from an earlier task may be from a file that has changed since.
/// </summary>
sealed record StackSource(string Path, decimal ScaleFactor, int EerGroupFrames, bool CorrectGain, string TaskId);
