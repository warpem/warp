using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PCA3D
{
    class Options
    {
        [Option("i", Required = true, HelpText = "Path to a STAR file containing the results of RELION's 3D refinement (something like run_data.star)")]
        public string StarPath { get; set; }

        [Option("angpixori", Required = true, HelpText = "Pixel size of the original data in Angstrom")]
        public float AngPixOri { get; set; }

        [Option("angpix", Required = true, HelpText = "Rescale data to this pixel size for PCA")]
        public float AngPix { get; set; }

        [Option("mask", Required = true, HelpText = "Path to an MRC volume containing a hard mask to specify the region(s) for the analysis, must have same size as original particles")]
        public string MaskPath { get; set; }

        [Option("diameter", Required = true, HelpText = "Particle diameter in Angstrom")]
        public float Diameter { get; set; }

        [Option("symmetry", Required = false, Default = "C1", HelpText = "Symmetry of the particle (don't perform symmetry expansion in RELION, use this option instead)")]
        public string Symmetry { get; set; }

        [Option("n", Default = 6, HelpText = "Number of principal components to be calculated")]
        public int NComponents { get; set; }

        [Option("iterations", Default = 80, HelpText = "Number of iterations for each component")]
        public int NIterations { get; set; }

        [Option("gpu", Default = 0, HelpText = "ID of the GPU used for calculations")]
        public int DeviceID { get; set; }

        [Option("batchsize", Default = 1024, HelpText = "Batch size used in calculations, reduce if you run out of memory")]
        public int BatchSize { get; set; }

        [Option("threads", Default = 5, HelpText = "Number of threads used for calculations")]
        public int NThreads { get; set; }
    }
}
