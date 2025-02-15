using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_stack", HelpText = "Create tilt series stacks, i.e. put all of a series' tilt images in one .st file, to be used with IMOD, AreTomo etc.")]
    [CommandRunner(typeof(StackTiltseries))]
    class StackTiltseriesOptions : DistributedOptions
    {
        [Option("angpix", HelpText = "Rescale tilt images to this pixel size; leave out to keep the original pixel size")]
        public double? AngPix { get; set; }

        [Option("mask", HelpText = "Apply mask to each image if available; masked areas will be filled with Gaussian noise")]
        public bool ApplyMask { get; set; }
    }

    class StackTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            StackTiltseriesOptions CLI = options as StackTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AngPix.HasValue && CLI.AngPix.Value < (double)Options.Import.BinnedPixelSize)
                throw new Exception("--angpix can't be smaller than the binned pixel size of the original data");

            #endregion

            #region Create processing options

            var OptionsStack = (ProcessingOptionsTomoStack)Options.FillTomoProcessingBase(new ProcessingOptionsTomoStack());

            OptionsStack.ApplyMask = CLI.ApplyMask;

            if (CLI.AngPix.HasValue)
                OptionsStack.BinTimes = (decimal)Math.Log(CLI.AngPix.Value / (double)Options.Import.PixelSize, 2.0);

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            IterateOverItems<TiltSeries>(Workers, CLI, (worker, m) =>
            {
                worker.TomoStack(m.Path, OptionsStack);
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }
    }
}
