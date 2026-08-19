using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Warp;
using System.Globalization;

namespace MTools.Commands
{
    [Verb("create_species", HelpText = "Create a new species")]
    [CommandRunner(typeof(CreateSpecies))]
    class CreateSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file to which to add the new data source.")]
        public string Population { get; set; }

        [Option('n', "name", Required = true, HelpText = "Name of the new species.")]
        public string Name { get; set; }

        [Option('d', "diameter", Required = true, HelpText = "Molecule diameter in Angstrom.")]
        public int Diameter { get; set; }

        [Option('s', "sym", Default = "C1", HelpText = "Point symmetry, e.g. C1, D7, O.")]
        public string Symmetry { get; set; }

        [Option("helical_units", Default = 1, HelpText = "Number of helical asymmetric units (only relevant for helical symmetry).")]
        public int HelicalUnits { get; set; }

        [Option("helical_twist", HelpText = "Helical twist in degrees, positive = right-handed (only relevant for helical symmetry).")]
        public double HelicalTwist { get; set; }

        [Option("helical_rise", HelpText = "Helical rise in Angstrom (only relevant for helical symmetry).")]
        public double HelicalRise { get; set; }

        [Option("helical_height", HelpText = "Height of the helical segment along the Z axis in Angstrom (only relevant for helical symmetry).")]
        public int HelicalHeight { get; set; }

        [Option('t', "temporal_samples", Default = 1, HelpText = "Number of temporal samples in each particle pose's trajectory.")]
        public int TemporalSamples { get; set; }

        [Option("half1", Required = true, HelpText = "Path to first half-map file.")]
        public string Half1 { get; set; }

        [Option("half2", Required = true, HelpText = "Path to second half-map file.")]
        public string Half2 { get; set; }

        [Option('m', "mask", Required = true, HelpText = "Path to a tight binary mask file. M will automatically expand and smooth it based on current resolution")]
        public string Mask { get; set; }

        [Option("angpix", HelpText = "Override pixel size value found in half-maps.")]
        public float? AngPix { get; set; }

        [Option("angpix_resample", HelpText = "Resample half-maps and masks to this pixel size.")]
        public float? AngPixResample { get; set; }

        [Option("lowpass", HelpText = "Optional low-pass filter (in Angstrom), applied to both half-maps.")]
        public float? Lowpass { get; set; }

        [Option("particles_relion", HelpText = "Path to _data.star-like particle metadata from RELION.")]
        public string ParticlesRelion { get; set; }

        [Option("particles_m", HelpText = "Path to particle metadata from M.")]
        public string ParticlesM { get; set; }

        [Option("angpix_coords", HelpText = "Override pixel size for RELION particle coordinates.")]
        public float? AngPixRelionPos { get; set; }

        [Option("angpix_shifts", HelpText = "Override pixel size for RELION particle shifts.")]
        public float? AngPixRelionShifts { get; set; }

        [Option("ignore_unmatched", HelpText = "Don't fail if there are particles that don't match any data sources.")]
        public bool IgnoreUnmatched { get; set; }

        [Option("dont_use_denoiser", HelpText = "Use low-pass filtering for regularization instead of a denoiser.")]
        public bool DontUseDenoiser { get; set; }
        
        [Option('o', "output", HelpText= "Optionally, override default path where the .species file and all data will be saved.")]
        public string OutputPath { get; set; }
        
        [Option("dont_version", HelpText = "If set, the source will not be versioned.")]
        public bool DontVersion { get; set; } = false;
    }

    class CreateSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            CreateSpeciesOptions Options = options as CreateSpeciesOptions;

            Population Population = new Population(Options.Population);

            #region Argument validation

            if (string.IsNullOrEmpty(Options.ParticlesRelion) == string.IsNullOrEmpty(Options.ParticlesM))
            {
                Console.Error.WriteLine("Must provide particle file from either RELION or M.");
                return;
            }

            if (Options.AngPix != null && Options.AngPix <= 0)
            {
                Console.Error.WriteLine("--angpix must be positive.");
                return;
            }

            if (Options.AngPixResample != null && Options.AngPixResample <= 0)
            {
                Console.Error.WriteLine("--angpix_resample must be positive.");
                return;
            }

            try
            {
                Symmetry S = new Symmetry(Options.Symmetry);
            }
            catch
            {
                Console.Error.WriteLine($"Unknown point symmetry: {Options.Symmetry}");
                return;
            }

            if (new[] { Options.HelicalUnits > 1,
                        Options.HelicalRise != 0,
                        Options.HelicalTwist != 0,
                        Options.HelicalHeight != 0 }.Any(v => v))
            {
                if (Options.HelicalUnits <= 1)
                {
                    Console.Error.WriteLine($"Helical symmetry requires at least 2 asymmetric units, specified {Options.HelicalUnits}.");
                    return;
                }

                if (Options.HelicalRise <= 0)
                {
                    Console.Error.WriteLine($"Helical rise must be positive, specified {Options.HelicalRise}.");
                    return;
                }

                if (Options.HelicalTwist == 0)
                {
                    Console.Error.WriteLine($"Helical twist must be non-zero, specified {Options.HelicalTwist}.");
                    return;
                }

                if (Options.HelicalHeight <= 0)
                {
                    Console.Error.WriteLine($"Helical height must be positive, specified {Options.HelicalHeight}.");
                    return;
                }

                double MinHeight = Options.HelicalRise * (Options.HelicalUnits - 1);
                if (Options.HelicalHeight < MinHeight)
                {
                    Console.Error.WriteLine($"Helical height must be at least {MinHeight:F2} A (but probably a bit more), specified {Options.HelicalHeight} A.");
                    return;
                }

                Console.WriteLine($"Will use helical symmetry: {Options.HelicalUnits} units, {Options.HelicalRise:F4} A rise, {Options.HelicalTwist:F4} deg twist, {Options.HelicalHeight} A height.");
            }

            if (Options.TemporalSamples < 1)
            {
                Console.Error.WriteLine("--temporal_samples can't be lower than 1.");
                return;
            }

            #endregion

            #region Half-maps and mask

            #region Read maps and and deal with pixel sizes

            Console.Write("Reading maps... ");

            Image Half1 = Image.FromFile(Options.Half1);
            Image Half2 = Image.FromFile(Options.Half2);
            Image Mask = Image.FromFile(Options.Mask);

            if (Half1.PixelSize != Mask.PixelSize)
            {
                Console.Error.WriteLine($"Half-map and mask pixel sizes don't match ({Half1.PixelSize} vs. {Mask.PixelSize}).");
                return;
            }

            if (!Half1.Dims.IsCubic || !Half2.Dims.IsCubic)
            {
                Console.Error.WriteLine($"Half-maps must be cubic.");
                return;
            }

            if (!Mask.Dims.IsCubic)
            {
                Console.Error.WriteLine($"Mask must be cubic.");
                return;
            }

            if (Half1.Dims != Half2.Dims)
            {
                Console.Error.WriteLine($"Half-map dimensions don't match ({Half1.Dims} vs. {Half2.Dims}).");
                return;
            }

            Console.WriteLine("Done");

            if (Options.AngPix == null)
            {
                Options.AngPix = Half1.PixelSize;
                Console.WriteLine($"--angpix not specified, using {Options.AngPix:F4} A/px from half-map.");
            }

            if (Options.AngPix != Half1.PixelSize)
            {
                Console.WriteLine($"WARNING: Pixel size in half maps ({Half1.PixelSize}) does not match --angpix ({Options.AngPix})");
            }
            if (Options.AngPixResample == null)
                Options.AngPixResample = Options.AngPix;

            #endregion

            #region Rescale and pad maps if needed

            float AngPix = Options.AngPix.Value;

            if (AngPix != Options.AngPixResample)
            {
                Console.Write($"Resampling maps to {(float)Options.AngPixResample:F4} A/px... ");

                int DimMapResampled = (int)MathF.Round(Half1.Dims.X * (float)Options.AngPix / (float)Options.AngPixResample / 2) * 2;
                Half1 = Half1.AsScaled(new int3(DimMapResampled)).AndDisposeParent();
                Half2 = Half2.AsScaled(new int3(DimMapResampled)).AndDisposeParent();

                int DimMaskResampled = (int)MathF.Round(Mask.Dims.X * (float)Options.AngPix / (float)Options.AngPixResample / 2) * 2;
                Mask = Mask.AsScaled(new int3(DimMaskResampled)).AndDisposeParent().AsPadded(Half1.Dims).AndDisposeParent();
                Mask.Binarize(0.25f);

                AngPix = (float)Options.AngPixResample;

                Console.WriteLine("Done");
            }

            Half1.PixelSize = AngPix;
            Half2.PixelSize = AngPix;
            Mask.PixelSize = AngPix;

            #endregion

            #region Pad maps to 2x diameter

            int DimPadded = (int)MathF.Round(Options.Diameter / AngPix) * 2;
            if (DimPadded != Half1.Dims.X)
            {
                Console.Write("Padding or cropping half-maps to 2x molecule diameter... ");

                Half1 = Half1.AsPadded(new int3(DimPadded)).AndDisposeParent();
                Half2 = Half2.AsPadded(new int3(DimPadded)).AndDisposeParent();

                Console.WriteLine("Done");
            }

            if (DimPadded != Mask.Dims.X)
            {
                Console.Write("Padding or cropping mask to 2x molecule diameter... ");

                Mask = Mask.AsPadded(new int3(DimPadded)).AndDisposeParent();

                Console.WriteLine("Done");
            }

            #endregion

            #region Low-pass, add a little noise to half-maps to avoid instability in FSC later, mask spherically

            Console.Write("Processing half-maps... ");

            if (Options.Lowpass != null)
            {
                if ((float)Options.Lowpass < AngPix * 2)
                {
                    Console.Error.WriteLine($"Low-pass can't be beyond Nyquist ({(AngPix * 2):F4} A)");
                    return;
                }

                Half1.Bandpass(0, (float)Options.Lowpass / AngPix / 2, true, 0.05f);
                Half2.Bandpass(0, (float)Options.Lowpass / AngPix / 2, true, 0.05f);
            }

            RandomNormal RandN = new RandomNormal(123);
            Half1.TransformValues(v => v + RandN.NextSingle(0, 1e-10f));
            Half2.TransformValues(v => v + RandN.NextSingle(0, 1e-10f));

            Half1.MaskSpherically(Half1.Dims.X - 32, 16, true);
            Half2.MaskSpherically(Half2.Dims.X - 32, 16, true);
            Mask.MaskSpherically(Mask.Dims.X - 32, 16, true);

            Console.WriteLine("Done");

            #endregion

            #endregion

            Species NewSpecies = new Species(Half1, Half2, Mask)
            {
                Name = Options.Name,
                PixelSize = (decimal)Options.AngPixResample,
                Symmetry = Options.Symmetry,
                HelicalUnits = Options.HelicalUnits,
                HelicalTwist = (decimal)Options.HelicalTwist,
                HelicalRise = (decimal)Options.HelicalRise,
                HelicalHeight = Options.HelicalHeight,
                DiameterAngstrom = Options.Diameter,
                TemporalResolutionMovement = Options.TemporalSamples,
                TemporalResolutionRotation = Options.TemporalSamples,
                ApplyDenoising = !Options.DontUseDenoiser,
                
                DontVersion = Options.DontVersion
            };

            NewSpecies.Path = string.IsNullOrWhiteSpace(Options.OutputPath) ?
                                  Path.Combine(Population.SpeciesDir,
                                               NewSpecies.NameSafe + "_" + NewSpecies.GUID.ToString().Substring(0, 8),
                                               NewSpecies.NameSafe + ".species") :
                                  Options.OutputPath;

            Directory.CreateDirectory(NewSpecies.FolderPath);
            
            if (File.Exists(NewSpecies.Path))
            {
                Console.Error.WriteLine($"{NewSpecies.Path} already exists. Please use a different name, or delete the old species.");
                return;
            }
            Directory.CreateDirectory(NewSpecies.FolderPath);

            #region Particles

            Console.Write("Parsing particle table... ");

            ParticleImportResult Imported = ParticleStarParsing.FromStar(Options.ParticlesM,
                                                                        Options.ParticlesRelion,
                                                                        Population.Sources,
                                                                        Options.TemporalSamples,
                                                                        Options.AngPixRelionPos,
                                                                        Options.AngPixRelionShifts);
            if (Imported == null)
                return;

            Particle[] ParticlesFinal = Imported.Particles;
            int ParticlesUnmatched = Imported.Unmatched;

            if (!Options.IgnoreUnmatched && ParticlesUnmatched > 0)
            {
                Console.Error.WriteLine($"{ParticlesUnmatched} particles couldn't be matched to data source. Please run again with --ignore_unmatched to proceed anyway.");
                return;
            }

            NewSpecies.AddParticles(ParticlesFinal);

            Console.WriteLine("Done");

            #endregion

            #region Calculate resolution

            Console.WriteLine("Calculating resolution and training denoiser model...");

            NewSpecies.CalculateResolutionAndFilter(Options.Lowpass ?? -1, (message) => { VirtualConsole.ClearLastLine(); Console.Write(message); });

            Console.Write("\nCalculating particle statistics... ");

            NewSpecies.CalculateParticleStats();

            Console.WriteLine("Done");
            Console.Write("Committing results... ");

            NewSpecies.Commit();
            NewSpecies.Save();

            Console.WriteLine("Done");

            #endregion

            Population.Species.Add(NewSpecies);
            Population.Save();

            Console.WriteLine($"Species created: '{NewSpecies.Name}' ({NewSpecies.GUID}), {NewSpecies.Path}");
            Console.WriteLine("To check if everything went alright, it's best to run M once with all refinements turned off and see if the new maps resemble your input.");
        }
    }
}
