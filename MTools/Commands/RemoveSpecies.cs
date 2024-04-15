using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;

namespace MTools.Commands
{
    [Verb("remove_species", HelpText = "Remove a species from a population")]
    [CommandRunner(typeof(RemoveSpecies))]
    class RemoveSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "species", Required = true, HelpText = "Path to the .species file, or its GUID.")]
        public string Species { get; set; }
    }

    class RemoveSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            RemoveSpeciesOptions Options = options as RemoveSpeciesOptions;

            Population Population = new Population(Options.Population);

            Guid Guid;

            if (File.Exists(Options.Species))
                Guid = Species.FromFile(Options.Species).GUID;
            else
                try
                {
                    Guid = Guid.Parse(Options.Species);
                }
                catch
                {
                    Console.Error.WriteLine("Couldn't find species at specified path, and couldn't interpret the argument as a GUID.");
                    return;
                }

            if (Population.Species.Any(s => s.GUID == Guid))
            {
                Species Species = Population.Species.First(s => s.GUID == Guid);
                Population.Species.Remove(Species);

                Population.Save();

                Console.WriteLine($"Removed species '{Species.Name}' ({Species.GUID}). No species files have been deleted from disk.");
            }
            else
            {
                Console.Error.WriteLine($"Species with GUID {Guid} not found in population.");
            }
        }
    }
}
