using Warp.Tools;

namespace Warp.Workers
{
    /// <summary>
    /// Typed factory methods for every worker command. Each method encapsulates the
    /// NamedSerializableObject construction and its argument order — the single source
    /// of truth for command signatures in the filesystem work-distribution path.
    ///
    /// nameof(WorkerWrapper.X) keeps command names tied to the wrapper method names so
    /// a rename is caught at compile time. When WorkerWrapper is eventually retired after
    /// the full port, move the nameof anchor here.
    /// </summary>
    public static class WorkerCommands
    {
        public static NamedSerializableObject LoadGainRef(
            string gainPath, bool flipX, bool flipY, bool transpose, string defectsPath) =>
            new(nameof(WorkerWrapper.LoadGainRef),
                gainPath ?? "", flipX, flipY, transpose, defectsPath ?? "");

        public static NamedSerializableObject LoadStack(
            string path, decimal scaleFactor, int eerGroupFrames, bool correctGain = true) =>
            new(nameof(WorkerWrapper.LoadStack), path, scaleFactor, eerGroupFrames, correctGain);

        public static NamedSerializableObject MovieProcessCTF(
            string path, ProcessingOptionsMovieCTF options) =>
            new(nameof(WorkerWrapper.MovieProcessCTF), path, options);

        public static NamedSerializableObject MovieProcessMovement(
            string path, ProcessingOptionsMovieMovement options) =>
            new(nameof(WorkerWrapper.MovieProcessMovement), path, options);

        public static NamedSerializableObject MovieExportMovie(
            string path, ProcessingOptionsMovieExport options) =>
            new(nameof(WorkerWrapper.MovieExportMovie), path, options);

        public static NamedSerializableObject MovieCreateThumbnail(
            string path, int size, float range) =>
            new(nameof(WorkerWrapper.MovieCreateThumbnail), path, size, range);

        public static NamedSerializableObject MoviePickBoxNet(
            string path, ProcessingOptionsBoxNet options) =>
            new(nameof(WorkerWrapper.MoviePickBoxNet), path, options);

        public static NamedSerializableObject MovieExportParticles(
            string path, ProcessingOptionsParticleExport options, float2[] coordinates) =>
            new(nameof(WorkerWrapper.MovieExportParticles), path, options, coordinates);

        public static NamedSerializableObject WaitAsyncTasks() =>
            new(nameof(WorkerWrapper.WaitAsyncTasks));

        public static NamedSerializableObject GcCollect() =>
            new(nameof(WorkerWrapper.GcCollect));

        public static NamedSerializableObject SetHeaderlessParams(
            int2 dims, long offset, string type) =>
            new(nameof(WorkerWrapper.SetHeaderlessParams), dims, offset, type);
    }
}
