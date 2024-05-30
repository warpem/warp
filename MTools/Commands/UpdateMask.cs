using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Warp;

namespace MTools.Commands
{
    [Verb("update_mask", HelpText = "Create a new mask for a species")]
    [CommandRunner(typeof(UpdateMap))]
    class UpdateMapOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }

        [Option('m', "map", Required = true, HelpText = "Path to the MRC map to be used to create the new mask.")]
        public string Map { get; set; }

        [Option('t', "threshold", Required = true, HelpText = "Binarization threshold to convert the input map to a mask.")]
        public float Threshold { get; set; }

        [Option('d', "dilate", Default = 0, HelpText = "Dilate the binary mask by this many voxels.")]
        public int Dilate { get; set; }

        [Option('c', "center", HelpText = "Center the species around the new mask's center of mass.")]
        public bool Center { get; set; }
    }

    class UpdateMap : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            UpdateMapOptions Options = options as UpdateMapOptions;

            Population Population = new Population(Options.Population);
            Species Species = null;

            if (File.Exists(Options.Species))
                Species = Species.FromFile(Options.Species);
            else
                try
                {
                    Guid Guid = Guid.Parse(Options.Species);
                    if (Population.Species.Any(s => s.GUID == Guid))
                        Species = Population.Species.First(s => s.GUID == Guid);
                    else
                    {
                        Console.Error.WriteLine($"No species with GUID {Guid} found in population.");
                        return;
                    }
                }
                catch
                {
                    Console.Error.WriteLine("Couldn't find species at specified path, and couldn't interpret the argument as a GUID.");
                    return;
                }

            Console.Write("Creating new mask... ");

            Image Map = Image.FromFile(Options.Map);
            if (Math.Abs(Map.PixelSize - (float)Species.PixelSize) > 1e-3)
                throw new Exception($"Map pixel size ({Map.PixelSize}) does not match species pixel size ({Species.PixelSize}).");

            Map.Binarize(Options.Threshold);
            if (Map.Dims != Species.HalfMap1.Dims)
                Map = Map.AsPadded(Species.HalfMap1.Dims).AndDisposeParent();

            if (Options.Dilate > 0)
                Map = Map.AsDilatedMask(Options.Dilate).AndDisposeParent();

            Species.Mask = Map;

            Console.WriteLine("Done");

            if (Options.Center)
            {
                Console.WriteLine("Centering species around new mask's center of mass...");

                float3 COM = Map.AsCenterOfMass();
                float3 Shifts = COM - new float3(Map.Dims / 2);
                Shifts *= (float)Species.PixelSize;

                Console.WriteLine($"Center will be shifted by {Shifts.X:F2}, {Shifts.Y:F2}, {Shifts.Z:F2} Angstrom");

                #region Rotate particles

                Console.Write("Shifting particles... ");

                var ParticlesOld = Species.Particles;
                var ShiftedParticles = new List<Particle>();

                foreach (var p in ParticlesOld)
                {
                    Matrix3 R0 = Matrix3.Euler(p.Angles[0] * Helper.ToRad);
                    float3 RotatedShift = R0 * Shifts;

                    Particle ShiftedParticle = p.GetCopy();

                    for (int t = 0; t < p.Coordinates.Length; t++)
                        ShiftedParticle.Coordinates[t] += RotatedShift;

                    ShiftedParticles.Add(ShiftedParticle);
                }

                var ParticlesFinal = ShiftedParticles.ToArray();
                Species.ReplaceParticles(ParticlesFinal);

                Console.WriteLine("Done");

                #endregion

                #region Rotate maps

                Console.Write("Shifting maps... ");

                Shifts = -Shifts / (float)Species.PixelSize;

                #region Remake mask from shifted map

                Map.Dispose();

                Map = Image.FromFile(Options.Map).AsShiftedVolume(Shifts).AndDisposeParent();

                Map.Binarize(Options.Threshold);
                if (Map.Dims != Species.HalfMap1.Dims)
                    Map = Map.AsPadded(Species.HalfMap1.Dims).AndDisposeParent();

                if (Options.Dilate > 0)
                    Map = Map.AsDilatedMask(Options.Dilate).AndDisposeParent();

                Species.Mask = Map;

                #endregion

                Species.MapFiltered = Species.MapFiltered.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.MapFilteredSharpened = Species.MapFilteredSharpened.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.MapFilteredAnisotropic = Species.MapFilteredAnisotropic.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.MapLocallyFiltered = Species.MapLocallyFiltered.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.MapDenoised = Species.MapDenoised.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.HalfMap1 = Species.HalfMap1.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
                Species.HalfMap2 = Species.HalfMap2.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            }

            Species.Commit();
            Species.Save();

            Console.WriteLine("Done");

            #endregion

            Console.WriteLine($"Update completed.");
        }
    }
}