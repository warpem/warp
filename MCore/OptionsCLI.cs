using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace MCore
{
    class OptionsCLI
    {
        [Option("port", Default = 14300, HelpText = "Port to use for REST API calls, set to -1 to disable")]
        public int Port { get; set; }

        [Option("devicelist", HelpText = "Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system")]
        public IEnumerable<int> DeviceList { get; set; }

        [Option("perdevice_preprocess", HelpText = "Number of processes per GPU used for map pre-processing; leave blank = default to --perdevice_refine value")]
        public int? ProcessesPerDevicePreprocess { get; set; }

        [Option("perdevice_refine", Default = 1, HelpText = "Number of processes per GPU used for refinement; set to >1 to improve utilization if your GPUs have enough memory")]
        public int ProcessesPerDeviceRefine { get; set; }

        [Option("perdevice_postprocess", HelpText = "Number of processes per GPU used for map pre-processing; leave blank = default to --perdevice_refine value")]
        public int? ProcessesPerDevicePostprocess { get; set; }

        [Option("workers_preprocess", HelpText = "List of remote workers to be used instead of locally spawned processes for map pre-processing. Formatted as hostname:port, separated by spaces")]
        public IEnumerable<string> WorkersPreprocess { get; set; }

        [Option("workers_refine", HelpText = "List of remote workers to be used instead of locally spawned processes for refinement. Formatted as hostname:port, separated by spaces")]
        public IEnumerable<string> WorkersRefine { get; set; }

        [Option("workers_postprocess", HelpText = "List of remote workers to be used instead of locally spawned processes for map post-processing. Formatted as hostname:port, separated by spaces")]
        public IEnumerable<string> WorkersPostprocess { get; set; }


        #region General refinement

        [Option("population", Required = true, HelpText = "Path to the .population file containing descriptions of data sources and species")]
        public string Population { get; set; }

        [Option("iter", Default = 3, HelpText = "Number of refinement sub-iterations")]
        public int NIterations { get; set; }

        [Option("first_iteration_fraction", Default = 1.0f, HelpText = "Use this fraction of available resolution for alignment in first sub-iteration, increase linearly to 1.0 towards last sub-iterations")]
        public float FirstIterationFraction { get; set; }

        [Option("min_particles", Default = 1, HelpText = "Only use series with at least N particles in the field of view")]
        public int NParticles { get; set; }

        [Option("cpu_memory", HelpText = "Use CPU memory to store particle images during refinement (GPU by default)")]
        public bool UseHostMemory { get; set; }

        [Option("weight_threshold", Default = 0.05f, HelpText = "Refine each tilt/frame up to the resolution at which the exposure weighting function (B-factor) reaches this value")]
        public float WeightThreshold { get; set; }

        #endregion

        #region Geometry

        [Option("refine_imagewarp", HelpText = "Refine image warp with a grid of XxY dimensions. Examples: leave blank = don't refine, '1x1', '6x4'")]
        public string RefineImageWarp { get; set; }

        [Option("refine_particles", HelpText = "Refine particle poses")]
        public bool RefinePoses { get; set; }

        [Option("refine_mag", HelpText = "Refine anisotropic magnification")]
        public bool RefineMag { get; set; }

        [Option("refine_doming", HelpText = "Refine doming (frame series only)")]
        public bool RefineDoming { get; set; }

        [Option("refine_stageangles", HelpText = "Refine stage angles (tilt series only)")]
        public bool RefineStageAngles { get; set; }

        [Option("refine_volumewarp", HelpText = "Refine volume warp with a grid of XxYxZxT dimensions (tilt series only). Examples: leave blank = don't refine, '1x1x1x20', '4x6x1x41'")]
        public string RefineVolumeWarp { get; set; }

        [Option("refine_tiltmovies", HelpText = "Refine tilt movie alignments (tilt series only)")]
        public bool RefineTiltMovies { get; set; }

        #endregion

        #region CTF

        [Option("ctf_batch", Default = 32, HelpText = "Batch size for CTF refinements. Lower = less memory, higher = faster")]
        public int CTFBatch { get; set; }

        [Option("ctf_minresolution", Default = 8.0f, HelpText = "Use only species with at least this resolution (in Angstrom) for CTF refinement")]
        public float CTFMinResolution { get; set; }

        [Option("ctf_defocus", HelpText = "Refine defocus using a local search")]
        public bool CTFDefocus { get; set; }

        [Option("ctf_defocusexhaustive", HelpText = "Refine defocus using a more exhaustive grid search in the first sub-iteration; only works in combination with ctf_defocus")]
        public bool CTFDefocusExhaustive { get; set; }

        [Option("ctf_phase", HelpText = "Refine phase shift (phase plate data only)")]
        public bool CTFPhase { get; set; }

        [Option("ctf_cs", HelpText = "Refine spherical aberration, which is also a proxy for pixel size")]
        public bool CTFCs { get; set; }

        [Option("ctf_zernike3", HelpText = "Refine Zernike polynomials of 3rd order (beam tilt, trefoil – fast)")]
        public bool CTFZernike3 { get; set; }

        [Option("ctf_zernike5", HelpText = "Refine Zernike polynomials of 5th order (fast)")]
        public bool CTFZernike5 { get; set; }

        [Option("ctf_zernike2", HelpText = "Refine Zernike polynomials of 2nd order (slow)")]
        public bool CTFZernike2 { get; set; }

        [Option("ctf_zernike4", HelpText = "Refine Zernike polynomials of 4th order (slow)")]
        public bool CTFZernike4 { get; set; }

        #endregion
    }
}
