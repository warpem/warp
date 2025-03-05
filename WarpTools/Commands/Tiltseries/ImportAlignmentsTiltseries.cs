using CommandLine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_import_alignments", HelpText = "Import tilt series alignments from IMOD or AreTomo")]
    [CommandRunner(typeof(ImportAlignmentsTiltseries))]
    class ImportAlignmentsTiltseriesOptions : BaseOptions
    {
        [Option("alignments", Required = true, HelpText = "Path to a folder containing one sub-folder per tilt series with alignment results from IMOD or AreTomo")]
        public string AlignmentPath { get; set; }

        [Option("alignment_angpix", Required = true, HelpText = "Pixel size (in Angstrom) of the images used to create the alignments (used to convert the alignment shifts from pixels to Angstrom)")]
        public double AlignmentAngPix { get; set; }

        [Option("min_fov", Default = 0.0, HelpText = "Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts")]
        public double MinFOV { get; set; }
    }

    class ImportAlignmentsTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ImportAlignmentsTiltseriesOptions CLI = options as ImportAlignmentsTiltseriesOptions;

            CLI.Evaluate();
            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AlignmentAngPix <= 0)
                throw new Exception("--alignment_angpix can't be 0 or lower");

            if (CLI.MinFOV > 1)
                throw new Exception("--min_fov can't be higher than 1");

            #endregion

            var OptionsImport = (ProcessingOptionsTomoImportAlignments)Options.FillTomoProcessingBase(new ProcessingOptionsTomoImportAlignments());
            
            OptionsImport.MinFOV = (decimal)CLI.MinFOV;
            OptionsImport.BinTimes = (decimal)Math.Log(CLI.AlignmentAngPix / (double)Options.Import.PixelSize, 2.0);
            
            if (Helper.IsDebug && !CLI.StrictFormatting)
                Console.WriteLine($"override results dir: {OptionsImport.OverrideResultsDir}");

            IterateOverItems<TiltSeries>(null, CLI, (_, series) =>
            {
                OptionsImport.OverrideResultsDir = Path.Combine(CLI.AlignmentPath, series.RootName);
                series.ImportAlignments(OptionsImport);
                series.SaveMeta();
            });
        }
    }
}
