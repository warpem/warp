using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace MTools.Commands
{
    [Verb("expand_symmetry", HelpText = "Expand symmetry in a species by creating symmetrically equivalent particle copies")]
    [CommandRunner(typeof(ExpandSymmetry))]
    class ExpandSymmetryOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }

        [Option("expand_from", HelpText = "Symmetry to use for the expansion if it is different from the one specified in the species (e.g. expand only one sub-symmetry of a higher symmetry).")]
        public string ExpandFrom { get; set; }

        [Option("expand_to", HelpText = "Remaining symmetry that will be set as the species' symmetry, e.g. C1 (when using --expand_from to expand only part of the symmetry).")]
        public string ExpandTo { get; set; }

        [Option("helical_units", Default = 1, HelpText = "Number of asymmetric subunits in the helical symmetry to expand")]
        public int HelicalUnits { get; set; }

        [Option("helical_twist", HelpText = "Twist of the helical symmetry to expand, in degrees")]
        public double HelicalTwist { get; set; }

        [Option("helical_rise", HelpText = "Rise of the helical symmetry to expand, in Angstrom")]
        public double HelicalRise { get; set; }
    }

    class ExpandSymmetry : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ExpandSymmetryOptions Options = options as ExpandSymmetryOptions;

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

            #region Symmetry argument validation

            if (Options.HelicalUnits < 1)
            {
                Console.Error.WriteLine("--helical_units must be at least 1.");
                return;
            }

            bool DoingPointGroup = !string.IsNullOrWhiteSpace(Options.ExpandTo);
            bool DoingHelical = Options.HelicalUnits > 1;

            if (DoingPointGroup == DoingHelical)
            {
                Console.Error.WriteLine("Exactly one of --expand_to and --helical_units must be specified.");
                return;
            }

            if (DoingPointGroup)
            {
                if (string.IsNullOrEmpty(Options.ExpandFrom))
                {
                    Options.ExpandFrom = Species.Symmetry;
                    Console.WriteLine($"--expand_from not specified, using symmetry specified in species.");
                }

                try
                {
                    Symmetry S = new Symmetry(Options.ExpandTo);
                }
                catch
                {
                    Console.Error.WriteLine($"Unknown final symmetry: {Options.ExpandTo}");
                    return;
                }
            }
            else
            {
                if (Options.HelicalRise <= 0)
                {
                    Console.Error.WriteLine("--helical_rise must be positive.");
                    return;
                }
            }

            #endregion

            #region Get symmetry transforms and perform expansion

            Particle[] ParticlesFinal = null;

            if (DoingPointGroup)
            {
                Symmetry Sym = new Symmetry(Options.ExpandFrom);
                Matrix3[] SymMats = Sym.GetRotationMatrices();

                var ParticlesOld = Species.Particles;

                List<Particle> ExpandedParticles = new List<Particle>();
                Matrix3[] Angles = new Matrix3[ParticlesOld[0].Angles.Length];

                foreach (var p in ParticlesOld)
                {
                    for (int i = 0; i < Angles.Length; i++)
                        Angles[i] = Matrix3.Euler(p.Angles[i] * Helper.ToRad);

                    foreach (var m in SymMats)
                    {
                        float3[] AnglesNew = Angles.Select(a => Matrix3.EulerFromMatrix(a * m) * Helper.ToDeg).ToArray();
                        Particle RotatedParticle = p.GetCopy();
                        RotatedParticle.Angles = AnglesNew;

                        ExpandedParticles.Add(RotatedParticle);
                    }
                }

                ParticlesFinal = ExpandedParticles.ToArray();

                Species.Symmetry = Options.ExpandTo;
            }
            else
            {
                Species.ResampleParticleTemporalResolution(1, 1);
                Console.WriteLine("Temporal trajectory resolution will be reset to 1");

                Matrix3[] SymMats = new Matrix3[Options.HelicalUnits];
                float3[] SymOffsets = new float3[Options.HelicalUnits];

                for (int i = 0; i < Options.HelicalUnits; i++)
                {
                    float Angle = -(float)Options.HelicalTwist * (i - Options.HelicalUnits / 2);
                    SymMats[i] = Matrix3.Euler(new float3(0, 0, Angle) * Helper.ToRad);

                    SymOffsets[i] = new float3(0, 0, (float)Options.HelicalRise * i);
                }

                var ParticlesOld = Species.Particles;

                List<Particle> ExpandedParticles = new();

                foreach (var p in ParticlesOld)
                {
                    for (int i = 0; i < Options.HelicalUnits; i++)
                    {
                        Matrix3 R = Matrix3.Euler(p.Angles[0] * Helper.ToRad) * SymMats[i];

                        float3 Pose = Matrix3.EulerFromMatrix(R) * Helper.ToDeg;
                        float3 RotatedShift = R * SymOffsets[i];

                        Particle RotatedParticle = p.GetCopy();
                        RotatedParticle.Angles = [Pose];
                        RotatedParticle.Coordinates = [RotatedParticle.Coordinates[0] + RotatedShift];

                        ExpandedParticles.Add(RotatedParticle);
                    }
                }

                ParticlesFinal = ExpandedParticles.ToArray();
            }

            #endregion

            #region Set expanded particles and commit results

            Species.ReplaceParticles(ParticlesFinal);

            Console.Write("Calculating particle statistics and committing results... ");

            Species.CalculateParticleStats();

            Species.Commit();
            Species.Save();

            Console.WriteLine("Done");

            #endregion

            Console.WriteLine($"Expansion completed: species now has {Species.Symmetry} symmetry, {Species.Particles.Length} particles.");
        }
    }
}
