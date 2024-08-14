using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_reconstruct", HelpText = "Reconstruct tomograms for various tasks and, optionally, half-tomograms for denoiser training")]
    [CommandRunner(typeof(ReconstructTiltseries))]
    class ReconstructTiltseriesOptions : DistributedOptions
    {
        [Option("angpix", Required = true, HelpText = "Pixel size of the reconstructed tomograms in Angstrom")]
        public double AngPix { get; set; }

        [Option("halfmap_frames", HelpText = "Also produce two half-tomograms, each reconstructed from half of the frames (requires running align_frameseries with --average_halves previously)")]
        public bool DoHalfmapFrames { get; set; }

        [Option("halfmap_tilts", HelpText = "Also produce two half-tomograms, each reconstructed from half of the tilts (doesn't work quite as well as --halfmap_frames)")]
        public bool DoHalfmapTilts { get; set; }

        [Option("deconv", HelpText = "Also produce a deconvolved version; all half-tomograms, if requested, will also be deconvolved")]
        public bool DoDeconv { get; set; }

        [Option("deconv_strength", Default = 1.0, HelpText = "Strength of the deconvolution filter, if requested")]
        public double DeconvStrength { get; set; }

        [Option("deconv_falloff", Default = 1.0, HelpText = "Fall-off of the deconvolution filter, if requested")]
        public double DeconvFalloff { get; set; }

        [Option("deconv_highpass", Default = 300.0, HelpText = "High-pass value (in Angstrom) of the deconvolution filter, if requested")]
        public double DeconvHighpass { get; set; }

        [Option("keep_full_voxels", HelpText = "Mask out voxels that aren't contained in some of the tilt images (due to excessive sample shifts); don't use if you intend to run template matching")]
        public bool KeepFullVoxels { get; set; }

        [Option("dont_invert", HelpText = "Don't invert the contrast; contrast inversion is needed for template matching on cryo data, i.e. when the density is dark in original images")]
        public bool NoInvert { get; set; }

        [Option("dont_normalize", HelpText = "Don't normalize the tilt images")]
        public bool NoNormalize { get; set; }

        [Option("dont_mask", HelpText = "Don't apply a mask to each tilt image if available; otherwise, masked areas will be filled with Gaussian noise")]
        public bool NoMask { get; set; }

        [Option("dont_overwrite", HelpText = "Don't overwrite existing tomograms in output directory")]
        public bool NoOverwrite { get; set; }

        [Option("subvolume_size", Default = 64, HelpText = "Reconstruction is performed locally using sub-volumes of this size in pixel")]
        public int SubVolumeSize { get; set; }

        [Option("subvolume_padding", Default = 3, HelpText = "Padding factor for the reconstruction sub-volumes (helps with aliasing effects at sub-volume borders)")]
        public int SubVolumePadding { get; set; }
    }

    class ReconstructTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ReconstructTiltseriesOptions CLI = options as ReconstructTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AngPix <= 0)
                throw new Exception("--angpix can't be 0 or negative");

            if (CLI.DoHalfmapFrames && CLI.DoHalfmapTilts)
                throw new Exception("Can't use both --halfmap_frames and --halfmap_tilts");

            if (CLI.DoHalfmapFrames)
            {
                Movie FirstMovie = new Movie(((TiltSeries)CLI.InputSeries.First()).TiltMoviePaths.First());
                if (!File.Exists(Path.Combine(CLI.InputSeries.First().DataOrProcessingDirectoryName, FirstMovie.AverageOddPath)) || 
                    !File.Exists(Path.Combine(CLI.InputSeries.First().DataOrProcessingDirectoryName, FirstMovie.AverageEvenPath)))
                    throw new Exception("Can't find half-averages for --halfmap_frames; run align_frameseries with --average_halves first");
            }

            if (CLI.DeconvHighpass < 10)
                throw new Exception("--deconv_highpass can't be lower than 10");

            if (CLI.SubVolumePadding < 1)
                throw new Exception("--subvolume_padding can't be lower than 1");

            if (CLI.SubVolumeSize < 16)
                throw new Exception("--subvolume_size can't be lower than 16");

            #endregion

            #region Create processing options

            Options.Tasks.TomoFullReconstructPixel = (decimal)CLI.AngPix;

            Options.Tasks.TomoFullReconstructDoDeconv = CLI.DoDeconv;
            Options.Tasks.TomoFullReconstructDeconvStrength = (decimal)CLI.DeconvStrength;
            Options.Tasks.TomoFullReconstructDeconvFalloff = (decimal)CLI.DeconvFalloff;
            Options.Tasks.TomoFullReconstructDeconvHighpass = (decimal)CLI.DeconvHighpass;

            Options.Tasks.TomoFullReconstructPrepareDenoising = CLI.DoHalfmapFrames || CLI.DoHalfmapTilts;
            Options.Tasks.TomoFullReconstructDenoisingFrames = CLI.DoHalfmapFrames;
            Options.Tasks.TomoFullReconstructDenoisingTilts = CLI.DoHalfmapTilts;

            Options.Tasks.InputInvert = !CLI.NoInvert;
            Options.Tasks.InputNormalize = !CLI.NoNormalize;

            Options.Tasks.TomoFullReconstructOnlyFullVoxels = CLI.KeepFullVoxels;

            var OptionsReconstruction = Options.GetProcessingTomoFullReconstruction();

            OptionsReconstruction.SubVolumeSize = CLI.SubVolumeSize;
            OptionsReconstruction.SubVolumePadding = CLI.SubVolumePadding;

            OptionsReconstruction.OverwriteFiles = !CLI.NoOverwrite;

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItems(Workers, CLI, (worker, m) =>
            {
                worker.TomoReconstruct(m.Path, OptionsReconstruction);
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }
    }
}
