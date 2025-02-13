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

            #region Set options
            //
            // Options.Movement.RangeMin = (decimal)CLI.RangeMin;
            // Options.Movement.RangeMax = (decimal)CLI.RangeMax;
            // Options.Movement.Bfactor = (decimal)CLI.Bfactor;
            //
            // if (!string.IsNullOrEmpty(CLI.GridDims))
            // {
            //     try
            //     {
            //         var Dims = CLI.GridDims.Split('x');
            //
            //         Options.Grids.MovementX = int.Parse(Dims[0]);
            //         Options.Grids.MovementY = int.Parse(Dims[1]);
            //         Options.Grids.MovementZ = int.Parse(Dims[2]);
            //     }
            //     catch
            //     {
            //         throw new Exception("Grid dimensions must be specified as XxYxZ, e.g. 5x5x40, or left empty for auto");
            //     }
            // }
            // else
            // {
            //     Options.Grids.MovementX = 0;
            //     Options.Grids.MovementY = 0;
            //     Options.Grids.MovementZ = 0;
            // }
            //
            // Options.Export.DoAverage = CLI.Averages;
            // Options.Export.DoDenoise = CLI.AverageHalves;
            // Options.Export.SkipFirstN = CLI.SkipFirst;
            // Options.Export.SkipLastN = CLI.SkipLast;

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItemsBatched(Workers, CLI, (worker, movies) =>
            {
                worker.TardisSegmentMembranes2D(movies.Select(m => m.Path).ToArray(), OptionsTardis);
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
