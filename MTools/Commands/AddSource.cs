using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;
using System.Linq;

namespace MTools.Commands
{
    [Verb("add_source", HelpText = "Add existing data source to a population")]
    [CommandRunner(typeof(AddSource))]
    class AddSourceOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Population { get; set; }

        [Option('s', "source", Required = true, HelpText = "Path to the .source file.")]
        public string Source { get; set; }
    }

    class AddSource : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            AddSourceOptions Options = options as AddSourceOptions;

            Population Population = new Population(Options.Population);
            DataSource Source = DataSource.FromFile(Options.Source);
            
            if (Population.Sources.Any(s => s.GUID == Source.GUID))
            {
                Console.Error.WriteLine($"Data source '{Source.Name}' ({Source.GUID}) already exists in this population.");
            }
            else
            {
                Population.Sources.Add(Source);
                Population.Save();

                Console.WriteLine($"Data source '{Source.Name}' ({Source.GUID}) added to population.");
            }    
        }
    }
}