using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;

namespace MTools.Commands
{
    [Verb("remove_source", HelpText = "Remove a data source from a population")]
    [CommandRunner(typeof(RemoveSource))]
    class RemoveSourceOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "source", Required = true, HelpText = "Path to the .source file, or its GUID.")]
        public string Source { get; set; }
    }

    class RemoveSource : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            RemoveSourceOptions Options = options as RemoveSourceOptions;

            Population Population = new Population(Options.Population);

            Guid Guid;

            if (File.Exists(Options.Source))
                Guid = DataSource.FromFile(Options.Source).GUID;
            else
                try
                {
                    Guid = Guid.Parse(Options.Source);
                }
                catch
                {
                    Console.Error.WriteLine("Couldn't find data source at specified path, and couldn't interpret the argument as a GUID.");
                    return;
                }

            if (Population.Sources.Any(s => s.GUID == Guid))
            {
                DataSource Source = Population.Sources.First(s => s.GUID == Guid);
                Population.Sources.Remove(Source);

                Population.Save();

                Console.WriteLine($"Removed data source '{Source.Name}' ({Source.GUID}).");
            }
            else
            {
                Console.Error.WriteLine($"Data source with GUID {Guid} not found in population.");
            }
        }
    }
}
