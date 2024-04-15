using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Frankenmap
{
    class Options
    {
        [Option("angpix", Required = true, HelpText = "Pixel size (must be identical for all maps).")]
        public float AngPix { get; set; }

        [Option("diameter", Required = true, HelpText = "Particle diameter in Angstrom.")]
        public int Diameter { get; set; }

        [Option("smooth", Default = 4f, HelpText = "Smoothing radius in pixels.")]
        public float SmoothingRadius { get; set; }

        [Option("windowsize", Default = 40, HelpText = "Size of the local window used for local resolution estimation, in pixels, even-numbered.")]
        public int WindowSize { get; set; }

        [Option("fscthreshold", Default = 0.143f, HelpText = "DANGER ZONE!")]
        public float FSCThreshold { get; set; }

        [Option("maskoverride", Default = "", HelpText = "Use a custom mask for certain maps instead of one created based on local resolution. Format: MapID:Path, separated by commas, e.g. 0:mask0.mrc,2:mask2.mrc")]
        public string MaskOverride { get; set; }
    }
}
