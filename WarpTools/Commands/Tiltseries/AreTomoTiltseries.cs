﻿using CommandLine;
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
    [Verb("ts_aretomo", HelpText = "Create tilt series stacks and run AreTomo2 to obtain tilt series alignments")]
    [CommandRunner(typeof(AreTomoTiltseries))]
    class AreTomoTiltseriesOptions : DistributedOptions
    {
        [Option("angpix", HelpText = "Rescale tilt images to this pixel size; normally 10–15 for cryo data; leave out to keep the original pixel size")]
        public double? AngPix { get; set; }

        [Option("mask", HelpText = "Apply mask to each image if available; masked areas will be filled with Gaussian noise")]
        public bool ApplyMask { get; set; }

        [Option("alignz", Required = true, HelpText = "Value for AreTomo's AlignZ parameter in Angstrom (will be auto-converted to binned pixels), i.e. the thickness of the reconstructed tomogram used for alignments")]
        public int AlignZ { get; set; }

        [Option("axis_iter", Default = 0, HelpText = "Number of tilt axis angle refinement iterations; each iteration will be started with median value from previous iteration, final iteration will use fixed angle")]
        public int AxisIterations { get; set; }

        [Option("axis_batch", Default = 0, HelpText = "Use only this many tilt series for the tilt axis angle search; only relevant if --axis_iter isn't 0")]
        public int AxisBatch { get; set; }

        [Option("min_fov", Default = 0.0, HelpText = "Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts")]
        public double MinFOV { get; set; }

        [Option("axis", HelpText = "Override tilt axis angle with this value")]
        public double? AxisAngle { get; set; }
        
        [Option("patches", HelpText = "Number of patches for local alignments in X, Y, separated by 'x': e.g. 6x4. Increases processing time.")]
        public string NPatchesXY { get; set; }
        
        [Option("delete_intermediate", HelpText = "Delete tilt series stacks generated for AreTomo")]
        public bool DeleteIntermediate { get; set; }

        [Option("thumbnails", HelpText = "Create thumbnails for each tilt image using the same pixel size as the stack")]
        public bool CreateThumbnails { get; set; }

        [Option("exe", Default = "AreTomo2", HelpText = "Name of the AreTomo2 executable; must be in $PATH")]
        public string Executable { get; set; }
    }

    class AreTomoTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            AreTomoTiltseriesOptions CLI = options as AreTomoTiltseriesOptions;
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

            if (CLI.AlignZ < 1)
                throw new Exception("--alignz can't be lower than 1");

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

            var OptionsAretomo = (ProcessingOptionsTomoAretomo)Options.FillTomoProcessingBase(new ProcessingOptionsTomoAretomo());
            OptionsAretomo.Executable = CLI.Executable;
            OptionsAretomo.AlignZ = (int)Math.Round(CLI.AlignZ / OptionsStack.BinnedPixelSizeMean);
            
            if (!string.IsNullOrEmpty(CLI.NPatchesXY))
            {
                try
                {
                    var NPatchesXY = CLI.NPatchesXY.Split('x').Select(int.Parse).ToArray();
                    OptionsAretomo.NPatchesXY = NPatchesXY;
                }
                catch
                {
                    throw new Exception("Patch grid dimensions must be specified as XxY, e.g. 5x5");
                }
            }
            else
            {
                OptionsAretomo.NPatchesXY = new int[] { 0, 0 };
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
                OptionsAretomo.AxisAngle = (decimal)AxisAngle;
                OptionsAretomo.DoAxisSearch = !LastIter;

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
                    Console.WriteLine("Running AreTomo with final average tilt axis angle:");

                    CLI.InputSeries = AllSeries;
                }

                IterateOverItems<TiltSeries>(Workers, CLI, (worker, t) =>
                {
                    if (iiter == 0 || NotUsedForSearch.Contains(t))
                        worker.TomoStack(t.Path, OptionsStack);

                    worker.TomoAretomo(t.Path, OptionsAretomo);

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
