using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Frame series")]
    [Verb("fs_tardis_segment_membranes", HelpText = "Semantic segmentation of membranes using tardis")]
    [CommandRunner(typeof(TardisSegmentMembranesFrameSeries))]
    class TardisSegmentMembranesFrameseriesOptions : DistributedOptions
    {
        
    }

    class TardisSegmentMembranesFrameSeries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            TardisSegmentMembranesFrameseriesOptions CLI = options as TardisSegmentMembranesFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;
            ProcessingOptionsTardisSegmentMembranes2D OptionsTardis = Options.GetProcessingTardisSegmentMembranes2D();

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItemsBatched(Workers, CLI, (worker, movies) =>
            {
                var paths = movies.Select(m => m.Path).ToArray();
                worker.TardisSegmentMembranes2D(paths, OptionsTardis);
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "tardis.settings"));
            Console.WriteLine(" Done");
        }
    }
}
