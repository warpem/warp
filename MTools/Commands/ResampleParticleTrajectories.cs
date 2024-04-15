using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Sociology;
using Warp.Tools;

namespace MTools.Commands
{
    [Verb("resample_trajectories", HelpText = "Resample the particle pose trajectories of a species")]
    [CommandRunner(typeof(ResampleTrajectories))]
    class ResampleTrajectoriesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }

        [Option("samples", Required = true, HelpText = "The new number of samples, usually between 1 (small particles) and 3 (very large particles).")]
        public int NewSamples { get; set; }
    }

    class ResampleTrajectories : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ResampleTrajectoriesOptions Options = options as ResampleTrajectoriesOptions;

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

            #region Argument validation

            if (Options.NewSamples < 1)
            {
                Console.Error.WriteLine("New number of samples must be at least 1.");
                return;
            }

            #endregion

            Species.ResampleParticleTemporalResolution(Options.NewSamples, Options.NewSamples);

            Console.Write("Committing results... ");

            Species.Commit();
            Species.Save();

            Console.WriteLine("Done");

            Console.WriteLine($"Resampling completed: poses now have {Species.TemporalResolutionMovement} temporal samples.");
        }
    }
}
