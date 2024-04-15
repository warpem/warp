using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;

namespace MTools.Commands
{
    [Verb("list_species", HelpText = "List all species in a population")]
    [CommandRunner(typeof(ListSpecies))]
    class ListSpeciesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Path { get; set; }
    }

    class ListSpecies : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ListSpeciesOptions Options = options as ListSpeciesOptions;

            Population Population = new Population(Options.Path);

            foreach (var species in Population.Species)
                Console.WriteLine($"'{species.Name}' ({species.GUID}), {Path.GetFullPath(species.Path)}");
        }
    }
}
