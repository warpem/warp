using CommandLine;
using System;
using Warp.Sociology;
using Warp.Tools;
using System.IO;

namespace MTools.Commands
{
    [Verb("list_sources", HelpText = "List all data sources in a population")]
    [CommandRunner(typeof(ListSources))]
    class ListSourcesOptions
    {
        [Option('p', "population", Required = true, HelpText = "Path to the .population file.")]
        public string Path { get; set; }
    }

    class ListSources : BaseCommand
    {
        public override void Run(object options)
        {
            base.Run(options);
            ListSourcesOptions Options = options as ListSourcesOptions;

            Population Population = new Population(Options.Path);

            foreach (var source in Population.Sources)
                Console.WriteLine($"'{source.Name}' ({source.GUID}), {Path.GetFullPath(source.Path)}");
        }
    }
}
