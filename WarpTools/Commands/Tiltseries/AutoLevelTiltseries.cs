using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands;


[VerbGroup("Tilt series")]
[Verb("ts_autolevel", HelpText = "Estimate the sample inclination around the X and Y axes to level it out")]
[CommandRunner(typeof(AutoLevelTiltseries))]
class AutoLevelTiltseriesOptions : DistributedOptions
{
    [Option("angpix", HelpText = "Rescale tilt images to this pixel size; leave out to keep the original pixel size")]
    public double? AngPix { get; set; }

    [Option("patch_size", Default = 500, HelpText = "Size of the patches in Angstroms to use during estimation")]
    public int PatchSize { get; set; }
}

class AutoLevelTiltseries : BaseCommand
{
    public override async Task Run(object options)
    {
        await base.Run(options);
        AutoLevelTiltseriesOptions CLI = options as AutoLevelTiltseriesOptions;
        CLI.Evaluate();

        OptionsWarp Options = CLI.Options;

        #region Validate options

        if (CLI.AngPix.HasValue && CLI.AngPix.Value < (double)Options.Import.BinnedPixelSize)
            throw new Exception("--angpix can't be smaller than the binned pixel size of the original data");

        if (CLI.PatchSize < 80)
            throw new Exception("--patchsize must be at least 80");

        #endregion

        #region Create processing options

        var OptionsLevel = (ProcessingOptionsTomoAutoLevel)Options.FillTomoProcessingBase(new ProcessingOptionsTomoAutoLevel());

        if (CLI.AngPix.HasValue)
            OptionsLevel.BinTimes = (decimal)Math.Log(CLI.AngPix.Value / (double)Options.Import.PixelSize, 2.0);

        OptionsLevel.RegionSize = Math.Max(16, (int)(CLI.PatchSize / OptionsLevel.BinnedPixelSizeMean + 1) / 2 * 2);

        #endregion

        WorkerWrapper[] Workers = CLI.GetWorkers();

        IterateOverItems<TiltSeries>(Workers, CLI, (worker, m) =>
        {
            worker.TomoAutoLevel(m.Path, OptionsLevel);

            worker.GcCollect();
        });

        Console.Write("Saying goodbye to all workers...");
        foreach (var worker in Workers)
            worker.Dispose();
        Console.WriteLine(" Done");
    }
}
