using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace Warp.Workers
{
    /// <summary>
    /// Typed factory methods for every worker command. Each method encapsulates the
    /// NamedSerializableObject construction and its argument order — the single source
    /// of truth for command signatures in the filesystem work-distribution path.
    ///
    /// Command name strings live in <see cref="WorkerCommandNames"/>, which both these
    /// factories and the WarpWorker2 [Command]/[MockCommand] handlers reference, so the
    /// filesystem path is self-contained and no longer anchored on WorkerWrapper. The
    /// legacy WorkerWrapper RPC transport (still used by the Warp/M GUIs) keeps its own
    /// matching literal strings.
    /// </summary>
    public static class WorkerCommands
    {
        public static NamedSerializableObject LoadGainRef(
            string gainPath, bool flipX, bool flipY, bool transpose, string defectsPath) =>
            new(WorkerCommandNames.LoadGainRef,
                gainPath ?? "", flipX, flipY, transpose, defectsPath ?? "");

        public static NamedSerializableObject LoadStack(
            string path, decimal scaleFactor, int eerGroupFrames, bool correctGain = true) =>
            new(WorkerCommandNames.LoadStack, path, scaleFactor, eerGroupFrames, correctGain);

        public static NamedSerializableObject MovieProcessCTF(
            string path, ProcessingOptionsMovieCTF options) =>
            new(WorkerCommandNames.MovieProcessCTF, path, options);

        public static NamedSerializableObject MovieProcessMovement(
            string path, ProcessingOptionsMovieMovement options) =>
            new(WorkerCommandNames.MovieProcessMovement, path, options);

        public static NamedSerializableObject MovieExportMovie(
            string path, ProcessingOptionsMovieExport options) =>
            new(WorkerCommandNames.MovieExportMovie, path, options);

        public static NamedSerializableObject MovieCreateThumbnail(
            string path, int size, float range) =>
            new(WorkerCommandNames.MovieCreateThumbnail, path, size, range);

        public static NamedSerializableObject LoadBoxNet(
            string path, int boxSize, int batchSize = 1) =>
            new(WorkerCommandNames.LoadBoxNet, path, boxSize, batchSize);

        public static NamedSerializableObject DropBoxNet() =>
            new(WorkerCommandNames.DropBoxNet);

        public static NamedSerializableObject MoviePickBoxNet(
            string path, ProcessingOptionsBoxNet options) =>
            new(WorkerCommandNames.MoviePickBoxNet, path, options);

        /// <summary>Pick dark disk/ring blobs. All distances are in micrograph pixels.</summary>
        public static NamedSerializableObject MicrographPickBlobs(
            string path, float radius, float innerRadius = 0, float? minimumSeparation = null)
        {
            float separation = BlobPicker.Validate(radius, innerRadius, minimumSeparation);
            return new(WorkerCommandNames.MicrographPickBlobs, path, radius, innerRadius, separation);
        }

        public static NamedSerializableObject MovieExportParticles(
            string path, ProcessingOptionsParticleExport options, float2[] coordinates) =>
            new(WorkerCommandNames.MovieExportParticles, path, options, coordinates);

        /// <summary>Prepare deconvolved averages and independent training halves at >=5 Å/pixel.</summary>
        /// <summary>
        /// Noise2Mic training for epochs over the movies' half-averages, deconvolved on the fly with each movie's CTF at
        /// <paramref name="pixelSize"/> (the averages' pixel size); the deconvolution settings are saved with the model.
        /// </summary>
        /// <param name="eraseObjects">False trains on the half-averages with their high-contrast objects left in. Sent only
        /// when false, so a worker from before the option reads the command it knows.</param>
        public static NamedSerializableObject TrainMovieDenoiser(string[] moviePaths, string modelOutput, string previousModel, int epochs, float deconvStrength, float deconvFalloff, float pixelSize, bool eraseObjects = true) =>
            eraseObjects
                ? new(WorkerCommandNames.TrainMovieDenoiser, moviePaths, modelOutput, previousModel ?? "", epochs, deconvStrength, deconvFalloff, pixelSize)
                : new(WorkerCommandNames.TrainMovieDenoiser, moviePaths, modelOutput, previousModel ?? "", epochs, deconvStrength, deconvFalloff, pixelSize, false);

        /// <summary>A PNG of the micrograph that fits maxSize, in the browser viewer's contrast.</summary>
        public static NamedSerializableObject MicrographCreatePreview(string micrographPath, string pngPath, int maxSize) =>
            new(WorkerCommandNames.MicrographCreatePreview, micrographPath, pngPath, maxSize);

        /// <summary>
        /// Denoises the movie's average with the model, deconvolving on the fly with the movie's CTF and these settings
        /// (the ones the model was trained with) at <paramref name="pixelSize"/>, the average's pixel size.
        /// </summary>
        /// <param name="eraseObjects">False denoises the average with its high-contrast objects left in; sent only when false.</param>
        public static NamedSerializableObject MovieDenoise(string moviePath, string modelPath, float pixelSize, float deconvStrength, float deconvFalloff, bool eraseObjects = true) =>
            eraseObjects
                ? new(WorkerCommandNames.MovieDenoise, moviePath, modelPath, pixelSize, deconvStrength, deconvFalloff)
                : new(WorkerCommandNames.MovieDenoise, moviePath, modelPath, pixelSize, deconvStrength, deconvFalloff, false);

        /// <summary>Coordinates in the picker image are converted to physical Å before extraction.</summary>
        public static NamedSerializableObject MovieExportParticlesFromStar(string moviePath, ProcessingOptionsParticleExport options,
            string coordinateStarPath, float coordinatePixelSize) =>
            new(WorkerCommandNames.MovieExportParticlesFromStar, moviePath, options, coordinateStarPath, coordinatePixelSize);

        /// <summary>
        /// Extracts only the picks whose Pearson correlation and local power lie within the bounds (inclusive,
        /// <see cref="PickSelection"/>) and writes their rows to <paramref name="extractedStarPath"/> in the stack's order.
        /// <paramref name="maxPower"/> = <see cref="double.MaxValue"/> leaves power unbounded.
        /// </summary>
        public static NamedSerializableObject MovieExportParticlesFromStar(string moviePath, ProcessingOptionsParticleExport options,
            string coordinateStarPath, float coordinatePixelSize, double minCorrelation, double maxCorrelation, double minPower, double maxPower,
            string extractedStarPath) =>
            new(WorkerCommandNames.MovieExportParticlesFromStar, moviePath, options, coordinateStarPath, coordinatePixelSize,
                minCorrelation, maxCorrelation, minPower, maxPower, extractedStarPath);

        // --- Tilt series ---

        public static NamedSerializableObject TomoStack(
            string path, ProcessingOptionsTomoStack options) =>
            new(WorkerCommandNames.TomoStack, path, options);

        public static NamedSerializableObject TomoProcessCTF(
            string path, ProcessingOptionsMovieCTF options) =>
            new(WorkerCommandNames.TomoProcessCTF, path, options);

        public static NamedSerializableObject TomoMatch(
            string path, ProcessingOptionsTomoFullMatch options, string templatePath) =>
            new(WorkerCommandNames.TomoMatch, path, options, templatePath);

        public static NamedSerializableObject TomoAretomo(
            string path, ProcessingOptionsTomoAretomo options) =>
            new(WorkerCommandNames.TomoAretomo, path, options);

        public static NamedSerializableObject TomoAretomo3(
            string path, ProcessingOptionsTomoAretomo3 options) =>
            new(WorkerCommandNames.TomoAretomo3, path, options);

        public static NamedSerializableObject TomoEtomoFiducials(
            string path, ProcessingOptionsTomoEtomoFiducials options) =>
            new(WorkerCommandNames.TomoEtomoFiducials, path, options);

        public static NamedSerializableObject TomoEtomoPatchTrack(
            string path, ProcessingOptionsTomoEtomoPatch options) =>
            new(WorkerCommandNames.TomoEtomoPatchTrack, path, options);

        public static NamedSerializableObject TomoReconstruct(
            string path, ProcessingOptionsTomoFullReconstruction options) =>
            new(WorkerCommandNames.TomoReconstruct, path, options);

        public static NamedSerializableObject TomoAutoLevel(
            string path, ProcessingOptionsTomoAutoLevel options) =>
            new(WorkerCommandNames.TomoAutoLevel, path, options);

        public static NamedSerializableObject LoadTomoDenoiser(
            string path, int3 windowSize, int batchSize) =>
            new(WorkerCommandNames.LoadTomoDenoiser, path, windowSize, batchSize);

        public static NamedSerializableObject TomoDenoise(
            string path, ProcessingOptionsTomoDenoise options) =>
            new(WorkerCommandNames.TomoDenoise, path, options);

        public static NamedSerializableObject TomoPeakAlign(
            string path, ProcessingOptionsTomoPeakAlign options,
            string templatePath, float3[] positions, float3[] angles) =>
            new(WorkerCommandNames.TomoPeakAlign, path, options, templatePath, positions, angles);

        public static NamedSerializableObject TomoExportParticleSubtomos(
            string path, ProcessingOptionsTomoSubReconstruction options,
            float3[] coordinates, float3[] angles) =>
            new(WorkerCommandNames.TomoExportParticleSubtomos, path, options, coordinates, angles);

        public static NamedSerializableObject TomoExportParticleSeries(
            string path, ProcessingOptionsTomoSubReconstruction options,
            float3[] coordinates, float3[] angles, string pathsRelativeTo, string pathTableOut) =>
            new(WorkerCommandNames.TomoExportParticleSeries,
                path, options, coordinates, angles, pathsRelativeTo, pathTableOut);

        // --- Averaged reconstruction (filesystem map-reduce) ---

        public static NamedSerializableObject InitReconstructions(
            int nreconstructions, int boxSize, int oversample) =>
            new(WorkerCommandNames.InitReconstructions, nreconstructions, boxSize, oversample);

        public static NamedSerializableObject TomoAddToReconstructionAndSave(
            string path, ProcessingOptionsTomoAddToReconstruction options,
            float3[][] positions, float3[][] angles, string tempDir) =>
            new(WorkerCommandNames.TomoAddToReconstructionAndSave,
                path, options, positions, angles, tempDir);

        public static NamedSerializableObject TomoFinishReconstruction(
            string[][] partialPaths, string[] symmetries, string[] outputPaths,
            float pixelSize, int boxSize, int oversample) =>
            new(WorkerCommandNames.TomoFinishReconstruction,
                partialPaths, symmetries, outputPaths, pixelSize, boxSize, oversample);

        // --- Multi-particle refinement (filesystem map-reduce) ---
        //
        // Three sequential phases over one ephemeral worker pool: per-species pre-flight
        // (MPAPrepareSpecies → staging), per-source refinement (MPAPreparePopulation +
        // LoadGainRef as amortized init, MPARefineAndSave per item accumulating into the
        // resident population and safe-saving a per-worker partial), and per-species
        // post-flight (MPAFinishSpecies gathers every per-worker partial, reconstructs,
        // filters). Replaces the legacy three separate WorkerWrapper process pools.

        public static NamedSerializableObject MPAPrepareSpecies(string path, string stagingSave) =>
            new(WorkerCommandNames.MPAPrepareSpecies, path, stagingSave);

        public static NamedSerializableObject MPAPreparePopulation(string path, string stagingLoad) =>
            new(WorkerCommandNames.MPAPreparePopulation, path, stagingLoad);

        public static NamedSerializableObject MPARefineAndSave(
            string path, ProcessingOptionsMPARefine options, DataSource source, string tempDir) =>
            new(WorkerCommandNames.MPARefineAndSave, path, options, source, tempDir);

        public static NamedSerializableObject MPAFinishSpecies(
            string path, string stagingDirectory, string[] progressFolders) =>
            new(WorkerCommandNames.MPAFinishSpecies, path, stagingDirectory, progressFolders);

        public static NamedSerializableObject WaitAsyncTasks() =>
            new(WorkerCommandNames.WaitAsyncTasks);

        public static NamedSerializableObject GcCollect() =>
            new(WorkerCommandNames.GcCollect);

        public static NamedSerializableObject SetHeaderlessParams(
            int2 dims, long offset, string type) =>
            new(WorkerCommandNames.SetHeaderlessParams, dims, offset, type);
    }

    /// <summary>
    /// Wire identifiers for every worker command — the single source of truth for the
    /// filesystem work-distribution path. Referenced by the <see cref="WorkerCommands"/>
    /// factories (senders) and the WarpWorker2 [Command]/[MockCommand] handlers
    /// (receivers). For commands also supported by the legacy RPC transport
    /// (Warp/M GUIs → WarpWorker), values must stay byte-identical to the matching
    /// WorkerWrapper method names. Worker2-only commands need no legacy RPC handler.
    /// </summary>
    public static class WorkerCommandNames
    {
        // --- Movies / frame series ---
        public const string LoadGainRef = "LoadGainRef";
        public const string LoadStack = "LoadStack";
        public const string MovieProcessCTF = "MovieProcessCTF";
        public const string MovieProcessMovement = "MovieProcessMovement";
        public const string MovieExportMovie = "MovieExportMovie";
        public const string MovieCreateThumbnail = "MovieCreateThumbnail";
        public const string LoadBoxNet = "LoadBoxNet";
        public const string DropBoxNet = "DropBoxNet";
        public const string MoviePickBoxNet = "MoviePickBoxNet";
        public const string MicrographPickBlobs = "MicrographPickBlobs";
        public const string TrainMovieDenoiser = "TrainMovieDenoiser";
        public const string MovieDenoise = "MovieDenoise";
        public const string MicrographCreatePreview = "MicrographCreatePreview";
        public const string MovieExportParticlesFromStar = "MovieExportParticlesFromStar";
        public const string MovieExportParticles = "MovieExportParticles";

        // --- Tilt series ---
        public const string TomoStack = "TomoStack";
        public const string TomoProcessCTF = "TomoProcessCTF";
        public const string TomoMatch = "TomoMatch";
        public const string TomoAretomo = "TomoAretomo";
        public const string TomoAretomo3 = "TomoAretomo3";
        public const string TomoEtomoFiducials = "TomoEtomoFiducials";
        public const string TomoEtomoPatchTrack = "TomoEtomoPatchTrack";
        public const string TomoReconstruct = "TomoReconstruct";
        public const string TomoAutoLevel = "TomoAutoLevel";
        public const string LoadTomoDenoiser = "LoadTomoDenoiser";
        public const string TomoDenoise = "TomoDenoise";
        public const string TomoPeakAlign = "TomoPeakAlign";
        public const string TomoExportParticleSubtomos = "TomoExportParticleSubtomos";
        public const string TomoExportParticleSeries = "TomoExportParticleSeries";

        // --- Averaged reconstruction (filesystem map-reduce) ---
        public const string InitReconstructions = "InitReconstructions";
        public const string TomoAddToReconstructionAndSave = "TomoAddToReconstructionAndSave";
        public const string TomoFinishReconstruction = "TomoFinishReconstruction";

        // --- Multi-particle refinement (filesystem map-reduce) ---
        public const string MPAPrepareSpecies = "MPAPrepareSpecies";
        public const string MPAPreparePopulation = "MPAPreparePopulation";
        public const string MPARefineAndSave = "MPARefineAndSave";
        public const string MPAFinishSpecies = "MPAFinishSpecies";

        // --- Service ---
        public const string WaitAsyncTasks = "WaitAsyncTasks";
        public const string GcCollect = "GcCollect";
        public const string SetHeaderlessParams = "SetHeaderlessParams";
    }
}
