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
    [Verb("ts_aretomo3", HelpText = "Create tilt series stacks and run AreTomo3 to obtain tilt series alignments")]
    [CommandRunner(typeof(AreTomo3Tiltseries))]
    class AreTomo3TiltseriesOptions : DistributedOptions
    {
        [Option("angpix", HelpText = "Rescale tilt images to this pixel size; normally 10–15 for cryo data; leave out to keep the original pixel size")]
        public double? AngPix { get; set; }

        [Option("mask", HelpText = "Apply mask to each image if available; masked areas will be filled with Gaussian noise")]
        public bool ApplyMask { get; set; }

        [Option("alignz", Default = 0, HelpText = "Sample thickness in Angstrom for AreTomo3's AlignZ parameter (auto-converted to binned pixels). When 0 or not given, AreTomo3 will estimate sample thickness automatically")]
        public int AlignZ { get; set; }

        [Option("patches", Default = "4,4", HelpText = "Number of patches for local alignments in X, Y, separated by comma: e.g. 4,4")]
        public string AtPatch { get; set; }

        [Option("axis_iter", Default = 0, HelpText = "Number of tilt axis angle refinement iterations; each iteration will be started with median value from previous iteration, final iteration will use fixed angle")]
        public int AxisIterations { get; set; }

        [Option("axis_batch", Default = 0, HelpText = "Use only this many tilt series for the tilt axis angle search; only relevant if --axis_iter isn't 0")]
        public int AxisBatch { get; set; }

        [Option("min_fov", Default = 0.0, HelpText = "Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts")]
        public double MinFOV { get; set; }

        [Option("axis", HelpText = "Override tilt axis angle with this value")]
        public double? AxisAngle { get; set; }

        [Option("delete_intermediate", HelpText = "Delete tilt series stacks generated for AreTomo3")]
        public bool DeleteIntermediate { get; set; }

        [Option("thumbnails", HelpText = "Create thumbnails for each tilt image using the same pixel size as the stack")]
        public bool CreateThumbnails { get; set; }

        [Option("exe", Default = "AreTomo3", HelpText = "Name of the AreTomo3 executable; must be in $PATH")]
        public string Executable { get; set; }
    }

    class AreTomo3Tiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            AreTomo3TiltseriesOptions CLI = options as AreTomo3TiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AngPix.HasValue && CLI.AngPix.Value < (double)Options.Import.BinnedPixelSize)
                throw new Exception("--angpix can't be smaller than the binned pixel size of the original data");
            else if (!CLI.AngPix.HasValue)
                CLI.AngPix = (double)Options.Import.BinnedPixelSize;

            if (CLI.MinFOV > 1)
                throw new Exception("--min_fov can't be higher than 1");

            if (CLI.AxisIterations < 0)
                throw new Exception("--axis_iter can't be negative");

            if (CLI.AxisBatch < 0)
                throw new Exception("--axis_batch can't be negative");

            if (CLI.AlignZ != 0 && CLI.AlignZ < 1)
                throw new Exception("--alignz can't be lower than 1 (or use 0 as default)");

            bool RefiningAxis = CLI.AxisIterations > 0;

            if (!Helper.ExeutableIsOnPath(CLI.Executable))
                throw new Exception($"Executable '{CLI.Executable}' not found on PATH");

            #endregion

            #region Create processing options

            var OptionsStack = (ProcessingOptionsTomoStack)Options.FillTomoProcessingBase(new ProcessingOptionsTomoStack());
            OptionsStack.ApplyMask = CLI.ApplyMask;
            OptionsStack.CreateThumbnails = CLI.CreateThumbnails;
            OptionsStack.BinTimes = (decimal)Math.Log(CLI.AngPix.Value / (double)Options.Import.PixelSize, 2.0);

            var OptionsImport = (ProcessingOptionsTomoImportAlignments)Options.FillTomoProcessingBase(new ProcessingOptionsTomoImportAlignments());
            OptionsImport.MinFOV = (decimal)CLI.MinFOV;
            OptionsImport.BinTimes = OptionsStack.BinTimes;

            var OptionsAretomo3 = (ProcessingOptionsTomoAretomo3)Options.FillTomoProcessingBase(new ProcessingOptionsTomoAretomo3());
            OptionsAretomo3.Executable = CLI.Executable;
            OptionsAretomo3.AlignZ = (int)Math.Round(CLI.AlignZ / OptionsStack.BinnedPixelSizeMean);

            if (!string.IsNullOrEmpty(CLI.AtPatch))
            {
                try
                {
                    var AtPatchParts = CLI.AtPatch.Split(',').Select(int.Parse).ToArray();
                    if (AtPatchParts.Length == 2)
                        OptionsAretomo3.AtPatch = AtPatchParts;
                    else
                        throw new Exception("AtPatch must have exactly 2 values");
                }
                catch
                {
                    throw new Exception("AtPatch dimensions must be specified as X,Y, e.g. 4,4");
                }
            }
            else
            {
                OptionsAretomo3.AtPatch = new int[] { 4, 4 };
            }



            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            Func<float> CalculateAverageAxis = () =>
            {
                float2 MedianVec = new float2(1, 0);
                if (CLI.InputSeries.Count() > 3)
                    MedianVec = MathHelper.GeometricMedian(CLI.InputSeries.Select(m =>
                    {
                        float Axis = (m as TiltSeries).TiltAxisAngles.Average() * Helper.ToRad;
                        return new float2(MathF.Cos(Axis), MathF.Sin(Axis));
                    })).Normalized();
                else
                    MedianVec = MathHelper.Mean(CLI.InputSeries.Select(m =>
                    {
                        float Axis = (m as TiltSeries).TiltAxisAngles.Average() * Helper.ToRad;
                        return new float2(MathF.Cos(Axis), MathF.Sin(Axis));
                    })).Normalized();

                return MathF.Atan2(MedianVec.Y, MedianVec.X) * Helper.ToDeg;
            };

            float AxisAngle = CLI.AxisAngle.HasValue ? (float)CLI.AxisAngle.Value : CalculateAverageAxis();
            var AllSeries = CLI.InputSeries.ToArray();
            var UsedForSearch = CLI.AxisBatch > 0 ? AllSeries.Take(CLI.AxisBatch).ToArray() : AllSeries;
            var NotUsedForSearch = CLI.AxisBatch > 0 ? AllSeries.Where(s => !UsedForSearch.Contains(s)).ToArray() : Array.Empty<Movie>();

            for (int iiter = 0; iiter < CLI.AxisIterations + 1; iiter++)
            {
                bool LastIter = iiter == CLI.AxisIterations;
                OptionsAretomo3.AxisAngle = (decimal)AxisAngle;
                OptionsAretomo3.DoAxisSearch = !LastIter;

                Console.WriteLine($"Current tilt axis angle: {AxisAngle:F3} °");

                if (iiter < CLI.AxisIterations)
                {
                    Console.WriteLine($"Running iteration {iiter + 1} of tilt axis refinement:");

                    if (CLI.AxisBatch > 0)
                    {
                        CLI.InputSeries = UsedForSearch;
                        Console.WriteLine($"Using {CLI.InputSeries.Length} out of {AllSeries.Length} series for tilt axis refinement");
                    }
                }
                else
                {
                    Console.WriteLine("Running AreTomo3 with final average tilt axis angle:");

                    CLI.InputSeries = AllSeries;
                }

                IterateOverItems<TiltSeries>(Workers, CLI, (worker, t) =>
                {
                    if (iiter == 0 || NotUsedForSearch.Contains(t))
                        worker.TomoStack(t.Path, OptionsStack);

                    worker.TomoAretomo3(t.Path, OptionsAretomo3);

                    t.ImportAlignments(OptionsImport);

                    if (LastIter)
                        t.SaveMeta();
                });

                AxisAngle = CalculateAverageAxis();
            }

            if (CLI.DeleteIntermediate)
            {
                Console.Write("Deleting intermediate stacks... ");

                foreach (var t in CLI.InputSeries)
                {
                    foreach (var dir in Directory.GetDirectories((t as TiltSeries).TiltStackDir))
                        if (!dir.EndsWith("thumbnails"))
                            Directory.Delete(dir, true);
                    
                    foreach (var file in Directory.GetFiles((t as TiltSeries).TiltStackDir))
                        File.Delete(file);
                }

                Console.WriteLine("Done");
            }

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }
    }
} 