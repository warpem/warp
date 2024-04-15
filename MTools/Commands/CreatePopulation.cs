using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;

namespace MTools.Commands
{
    [Verb("create_population", HelpText = "Create a new population")]
    [CommandRunner(typeof(CreatePopulation))]
    class CreatePopulationOptions
    {
        [Option('d', "directory", Required = true, HelpText = "Path to the directory where the new population will be located. All future species will also go there, so make sure there is enough space.")]
        public string Directory { get; set; }

        [Option('n', "name", Required = true, HelpText = "Name of the new population.")]
        public string Name { get; set; }
    }

    class CreatePopulation : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            CreatePopulationOptions Options = options as CreatePopulationOptions;

            Directory.CreateDirectory(Options.Directory);

            string PopulationName = Helper.RemoveInvalidChars(Options.Name);
            string PopulationPath = Path.Combine(Options.Directory, PopulationName + ".population");

            Population NewPopulation = new Population(PopulationPath);
            NewPopulation.Name = Options.Name;
            NewPopulation.Save();

            Console.WriteLine($"Population created: {PopulationPath}");
        }
    }
}
