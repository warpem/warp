using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace MTools.Commands
{
    [Verb("shift_species", HelpText = "Shift particles in a species to reflect a shift of the 3D map")]
    [CommandRunner(typeof(ShiftSpecies))]
    class ShiftSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }

        [Option('x', Required = true, HelpText = "Shift along the X axis in Angstrom. New map center will be at current center + this value.")]
        public float ShiftX { get; set; }

        [Option('y', Required = true, HelpText = "Shift along the X axis in Angstrom. New map center will be at current center + this value.")]
        public float ShiftY { get; set; }

        [Option('z', Required = true, HelpText = "Shift along Z axis in Angstrom. New map center will be at current center + this value.")]
        public float ShiftZ { get; set; }
    }

    class ShiftSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ShiftSpeciesOptions Options = options as ShiftSpeciesOptions;

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

            float3 Shifts = new float3(Options.ShiftX, Options.ShiftY, Options.ShiftZ);

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

            Species.MapFiltered = Species.MapFiltered.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.MapFilteredSharpened = Species.MapFilteredSharpened.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.MapFilteredAnisotropic = Species.MapFilteredAnisotropic.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.MapLocallyFiltered = Species.MapLocallyFiltered.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.MapDenoised = Species.MapDenoised.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.HalfMap1 = Species.HalfMap1.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();
            Species.HalfMap2 = Species.HalfMap2.AsShiftedVolume(Shifts).FreeDevice().AndDisposeParent();

            Species.Mask = Species.Mask.AsShiftedVolume(Shifts).AndDisposeParent();
            Species.Mask.Binarize(0.8f);
            Species.Mask.FreeDevice();

            Species.Commit();
            Species.Save();

            Console.WriteLine("Done");

            #endregion

            Console.WriteLine($"Shift completed.");
        }
    }
}
