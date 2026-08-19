using CommandLine;
using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;

namespace MTools.Commands
{
    [Verb("postprocess", HelpText = "Re-estimate resolution, sharpen and locally filter an existing species or " +
                                   "reconstruction, optionally with a different mask. Works from the half-maps " +
                                   "already on disk, so nothing is re-extracted or re-projected.")]
    [CommandRunner(typeof(Postprocess))]
    class PostprocessOptions
    {
        [Option('s', "species", Required = true, HelpText = "Path to a .species file, or to a directory containing one " +
                                                           "(e.g. a reconstruction directory).")]
        public string Species { get; set; }

        [Option('m', "mask", HelpText = "Path to a tight binary mask. M will expand and smooth it based on the resolution. " +
                                        "Omit to use the mask already in the species directory (<name>_mask.mrc), so you " +
                                        "can just drop a replacement in there. Rescaled and padded to the species' " +
                                        "sampling and box either way.")]
        public string Mask { get; set; }

        [Option("sphere", HelpText = "Replace the mask with a sphere of the molecule diameter — unbiased, but a more " +
                                     "conservative FSC. This is what a reconstruction made without --mask already uses.")]
        public bool Sphere { get; set; }

        [Option("lowpass", HelpText = "Skip FSC estimation and filter to this fixed resolution in Angstrom instead.")]
        public float? Lowpass { get; set; }

        [Option("denoise", HelpText = "Train and apply a denoiser. Off by default; it is by far the slowest part.")]
        public bool Denoise { get; set; }

        [Option("device", Default = -1, HelpText = "GPU ID to use. Default: let the runtime pick.")]
        public int Device { get; set; }
    }

    /// <summary>
    /// Standalone postprocessing.
    ///
    /// The mask is used nowhere in extraction or back-projection — grep the "Update
    /// reconstructions" regions of TiltSeries/Movie.MPARefinement and it never appears. It
    /// only enters CalculateResolutionAndFilter, which works entirely from HalfMap1/HalfMap2.
    /// Since those are written to disk, trying a different mask is seconds of work rather
    /// than a full re-reconstruction.
    /// </summary>
    class Postprocess : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            PostprocessOptions Options = options as PostprocessOptions;

            #region Resolve the species

            string SpeciesPath = Options.Species;

            if (Directory.Exists(SpeciesPath))
            {
                string[] Candidates = Directory.GetFiles(SpeciesPath, "*.species");
                if (Candidates.Length == 0)
                {
                    Console.Error.WriteLine($"No .species file found in {SpeciesPath}.");
                    return;
                }
                if (Candidates.Length > 1)
                {
                    Console.Error.WriteLine($"{SpeciesPath} contains more than one .species file; point at one of them directly.");
                    return;
                }
                SpeciesPath = Candidates[0];
            }
            else if (!File.Exists(SpeciesPath))
            {
                Console.Error.WriteLine($"Species not found: {SpeciesPath}");
                return;
            }

            if (!string.IsNullOrEmpty(Options.Mask) && Options.Sphere)
            {
                Console.Error.WriteLine("--mask and --sphere are mutually exclusive.");
                return;
            }

            if (!string.IsNullOrEmpty(Options.Mask) && !File.Exists(Options.Mask))
            {
                Console.Error.WriteLine($"Mask not found: {Options.Mask}");
                return;
            }

            Species S = Species.FromFile(SpeciesPath);

            // The half-map getters return null when the file is missing, which catches a
            // species whose maps were never produced or have since been deleted.
            if (S.HalfMap1 == null || S.HalfMap2 == null)
            {
                Console.Error.WriteLine($"{SpeciesPath} has no half-maps on disk ({S.NameSafe}_half1.mrc / _half2.mrc), " +
                                        "so there is nothing to postprocess.");
                return;
            }

            Console.WriteLine($"Postprocessing '{S.Name}' at {S.PixelSize:F3} A/px, " +
                              $"box {S.HalfMap1.Dims.X} px, diameter {S.DiameterAngstrom} A.");

            #endregion

            #region Mask

            Image Mask;

            if (Options.Sphere)
            {
                Console.WriteLine($"Using a sphere of {S.DiameterAngstrom} A as the mask.");
                Mask = Reconstruct.CreateSphereMask(S.HalfMap1.Dims.X,
                                                    (float)(S.DiameterAngstrom / S.PixelSize));
                Mask.PixelSize = (float)S.PixelSize;
            }
            else if (!string.IsNullOrEmpty(Options.Mask))
            {
                Console.WriteLine($"Using mask {Options.Mask}.");
                Mask = Image.FromFile(Options.Mask);
            }
            else if (S.Mask != null)
            {
                Console.WriteLine($"Using the mask already in the species directory ({Path.GetFileName(S.PathMask)}).");
                Mask = S.Mask;
            }
            else
            {
                Console.Error.WriteLine($"This species has no mask ({Path.GetFileName(S.PathMask)} is missing); " +
                                        "pass --mask or --sphere.");
                return;
            }

            // Normalise whichever mask we ended up with. This matters just as much for one
            // picked up from the species directory as for one passed on the command line:
            // dropping a replacement mask.mrc in by hand is a normal thing to do, and it may
            // well have been drawn on a map with a different box or sampling.
            S.Mask = NormalizeMask(Mask, S);

            #endregion

            #region Re-filter

            // Clear the previous estimate so everything is re-derived from the current
            // half-maps and mask. It is not merely cosmetic: CalculateResolutionAndFilter
            // bootstraps a resolution with a soft spherical mask only when GlobalResolution
            // is <= 0, and it sizes the mask softening as max(1, GlobalResolution /
            // (PixelSize * 2)) — so a value left over from a previous run would change how
            // much the new mask is smoothed.
            S.GlobalResolution = 0;
            S.ApplyDenoising = Options.Denoise;

            Console.WriteLine();
            S.CalculateResolutionAndFilter(Options.Lowpass ?? -1,
                                                 message => { VirtualConsole.ClearLastLine(); Console.Write(message); },
                                                 Options.Device);
            Console.WriteLine();

            // Save, but deliberately not Commit: the filtered maps are derived data, and
            // versioning them would spawn a versions/ tree for what is a cheap, repeatable step.
            S.Save();

            #endregion

            Console.WriteLine();
            Console.WriteLine($"Resolution {S.GlobalResolution:F2} A, global B-factor {S.GlobalBFactor}.");
            if (Options.Sphere)
                Console.WriteLine("The FSC used a spherical mask, so the resolution is a conservative estimate; " +
                                  "pass --mask with a tight mask for a tighter number.");
            Console.WriteLine();
            Console.WriteLine($"Updated maps written to {S.FolderPath}");
        }

        /// <summary>
        /// Bring a mask onto the species' sampling and box so it can be multiplied against
        /// the half-maps. Sampling is handled before the box: rescaling after padding would
        /// be too late, and padding a mask drawn at a finer pixel size would silently crop
        /// away its outside rather than resample it. A header pixel size of 0 means the file
        /// doesn't declare one, so it is taken to already match.
        /// </summary>
        private static Image NormalizeMask(Image mask, Species species)
        {
            bool Changed = false;

            if (mask.PixelSize > 0 && Math.Abs(mask.PixelSize - (float)species.PixelSize) > 0.02f)
            {
                int DimScaled = (int)MathF.Round(mask.Dims.X * mask.PixelSize / (float)species.PixelSize / 2) * 2;
                Console.WriteLine($"Rescaling mask from {mask.PixelSize:F3} to {(float)species.PixelSize:F3} A/px " +
                                  $"({mask.Dims.X} -> {DimScaled} px).");
                mask = mask.AsScaled(new int3(DimScaled)).AndDisposeParent();
                Changed = true;
            }

            if (mask.Dims != species.HalfMap1.Dims)
            {
                Console.WriteLine($"Padding or cropping mask from {mask.Dims.X} to {species.HalfMap1.Dims.X} px.");
                mask = mask.AsPadded(species.HalfMap1.Dims).AndDisposeParent();
                Changed = true;
            }

            // Interpolation leaves intermediate values behind; restore a binary mask the same
            // way create_species does.
            if (Changed)
                mask.Binarize(0.25f);

            mask.PixelSize = (float)species.PixelSize;
            return mask;
        }
    }
}
