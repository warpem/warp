using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace MTools.Commands
{
    [Verb("rotate_species", HelpText = "Modify particle poses in a species to reflect a rotation of the 3D map")]
    [CommandRunner(typeof(RotateSpecies))]
    class RotateSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }

        [Option("angle_rot", Required = true, HelpText = "First Euler angle (Rot in RELION) in degrees.")]
        public float AngleRot { get; set; }

        [Option("angle_tilt", Required = true, HelpText = "Second Euler angle (Tilt in RELION) in degrees.")]
        public float AngleTilt { get; set; }

        [Option("angle_psi", Required = true, HelpText = "Third Euler angle (Psi in RELION) in degrees.")]
        public float AnglePsi { get; set; }
    }

    class RotateSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            RotateSpeciesOptions Options = options as RotateSpeciesOptions;

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

            float3 Angles = new float3(Options.AngleRot, Options.AngleTilt, Options.AnglePsi) * Helper.ToRad;

            #region Rotate particles

            Console.Write("Rotating particles... ");

            var ParticlesOld = Species.Particles;

            List<Particle> RotatedParticles = new List<Particle>();
            Matrix3 R = Matrix3.Euler(Angles).Transposed();

            foreach (var p in ParticlesOld)
            {
                float3[] AnglesNew = p.Angles.Select(a => Matrix3.EulerFromMatrix(Matrix3.Euler(a * Helper.ToRad) * R) * Helper.ToDeg).ToArray();

                Particle Rotated = p.GetCopy();
                Rotated.Angles = AnglesNew;

                RotatedParticles.Add(Rotated);
            }

            var ParticlesFinal = RotatedParticles.ToArray();
            Species.ReplaceParticles(ParticlesFinal);

            Console.WriteLine("Done");

            #endregion

            #region Rotate maps

            Console.Write("Rotating maps... ");

            Species.MapFiltered = Species.MapFiltered.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.MapFilteredSharpened = Species.MapFilteredSharpened.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.MapFilteredAnisotropic = Species.MapFilteredAnisotropic.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.MapLocallyFiltered = Species.MapLocallyFiltered.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.MapDenoised = Species.MapDenoised.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.HalfMap1 = Species.HalfMap1.AsRotated3D(Angles).FreeDevice().AndDisposeParent();
            Species.HalfMap2 = Species.HalfMap2.AsRotated3D(Angles).FreeDevice().AndDisposeParent();

            Species.Mask = Species.Mask.AsRotated3D(Angles).AndDisposeParent();
            Species.Mask.Binarize(0.8f);
            Species.Mask.FreeDevice();

            Species.Commit();
            Species.Save();

            Console.WriteLine("Done");

            #endregion

            Console.WriteLine($"Rotation completed.");
        }
    }
}
