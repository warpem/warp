using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace EstimateWeights
{
    class Options
    {
        [Option("population", Required = true, HelpText = "Path to the .population file.")]
        public string PopulationPath { get; set; }

        [Option("source", Required = true, HelpText = "Name of the data source.")]
        public string SourceName { get; set; }

        [Option("resolve_frames", HelpText = "Estimate weights per frame or tilt. Can be combined with 'resolve_items', resulting in Nitems*Nframes weights.")]
        public bool ResolveFrames { get; set; }

        [Option("resolve_items", HelpText = "Estimate weights per item, i.e. frame series or tilt series. Can be combined with 'resolve_frames', resulting in Nitems*Nframes weights.")]
        public bool ResolveItems { get; set; }

        [Option("resolve_location", HelpText = "Estimate weights that depend on particle position within the image or volume. Cannot be combined with others.")]
        public bool ResolveLocation { get; set; }

        [Option("resolve_sources", HelpText = "Estimate weights per item, i.e. frame series or tilt series. Can be combined with 'resolve_frames', resulting in Nitems*Nframes weights.")]
        public bool ResolveSources { get; set; }

        [Option("fit_anisotropy", HelpText = "Fit anisotropic B-factors. Only makes sense when fitting per-item, per-frame/tilt (except maybe in tilted data where BIM is perpendicular to tilt axis?).")]
        public bool FitAnisotropy { get; set; }

        [Option("do_tiltframes", HelpText = "Estimate weights for tilt movies frames. Only works for tilt series where the original movies are available.")]
        public bool DoTiltFrames { get; set; }

        [Option("grid_width", HelpText = "Width of the parameter grid when fitting spatially resolved weights.")]
        public int GridWidth { get; set; }

        [Option("grid_height", HelpText = "Height of the parameter grid when fitting spatially resolved weights.")]
        public int GridHeight { get; set; }

        [Option("min_resolution", Default = 20f, HelpText = "Minimum resolution to consider for estimation.")]
        public float MinResolution { get; set; }

        [Option("reset", HelpText = "Resets all previously fitted weights for all items; not compatible with any other options.")]
        public bool DoReset { get; set; }
    }
}
