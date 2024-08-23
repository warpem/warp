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
    [Verb("ts_etomo_fiducials", HelpText = "Create tilt series stacks and run Etomo fiducial tracking to obtain tilt series alignments")]
    [CommandRunner(typeof(EtomoFiducialsTiltseries))]
    class EtomoFiducialsTiltseriesOptions : DistributedOptions
    {
        [Option("angpix",
            HelpText =
                "Rescale tilt images to this pixel size; normally 10–15 for cryo data; leave out to keep the original pixel size")]
        public double? AngPix { get; set; }

        [Option("mask",
            HelpText = "Apply mask to each image if available; masked areas will be filled with Gaussian noise")]
        public bool ApplyMask { get; set; }

        [Option("min_fov", Default = 0.0,
            HelpText =
                "Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts")]
        public double MinFOV { get; set; }

        [Option("initial_axis", HelpText = "Override initial tilt axis angle with this value")]
        public double? InitialAxisAngle { get; set; }
        
        [Option("do_axis_search", HelpText = "Fit a new tilt axis angle for the whole dataset")]
        public bool DoAxisAngleSearch { get; set; }
        
        [Option("fiducial_size", HelpText = "size of gold fiducials in nanometers")]
        public double? FiducialSizeNanometers { get; set; }

        [Option("delete_intermediate", HelpText = "Delete tilt series stacks generated for ETomo")]
        public bool DeleteIntermediate { get; set; }
    }

    class EtomoFiducialsTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            EtomoFiducialsTiltseriesOptions CLI = options as EtomoFiducialsTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.AngPix.HasValue && CLI.AngPix.Value < (double)Options.Import.BinnedPixelSize)
                throw new Exception("--angpix can't be smaller than the binned pixel size of the original data");
            else if (!CLI.AngPix.HasValue)
                CLI.AngPix = (double)Options.Import.BinnedPixelSize;

            if (CLI.MinFOV > 1)
                throw new Exception("--min_fov can't be higher than 1");

            if (!Helper.ExeutableIsOnPath("batchruntomo"))
                throw new Exception("IMOD program batchruntomo not found on PATH");

            if (!CLI.FiducialSizeNanometers.HasValue)
                throw new Exception("--fiducial_size must be passed at the CLI");

            #endregion

            #region Create processing options

            var OptionsStack =
                (ProcessingOptionsTomoStack)Options.FillTomoProcessingBase(new ProcessingOptionsTomoStack());
            OptionsStack.ApplyMask = CLI.ApplyMask;
            OptionsStack.BinTimes = (decimal)Math.Log(CLI.AngPix.Value / (double)Options.Import.PixelSize, 2.0);

            var OptionsImport =
                (ProcessingOptionsTomoImportAlignments)Options.FillTomoProcessingBase(
                    new ProcessingOptionsTomoImportAlignments());
            OptionsImport.MinFOV = (decimal)CLI.MinFOV;
            OptionsImport.BinTimes = OptionsStack.BinTimes;

            var OptionsEtomo =
                (ProcessingOptionsTomoEtomoFiducials)Options.FillTomoProcessingBase(
                    new ProcessingOptionsTomoEtomoFiducials());
            OptionsEtomo.TiltStackAngPix = (decimal)CLI.AngPix;
            OptionsEtomo.FiducialSizeNanometers = (decimal)CLI.FiducialSizeNanometers;

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();
            
            // define a function to calculate the average tilt axis over the whole dataset
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
            
            // Set initial axis angle for all tilt series
            OptionsEtomo.AxisAngle = CLI.InitialAxisAngle.HasValue
                ? (decimal)CLI.InitialAxisAngle
                : (decimal)CalculateAverageAxis();
            
            // do patch tracking for all tilt series, optionally updating tilt axis angle
            OptionsEtomo.DoFiducialTracking = true;
            OptionsEtomo.DoTiltAlign = true;
            OptionsEtomo.DoAxisAngleSearch = CLI.DoAxisAngleSearch;
            
            
            Console.WriteLine("Performing fiducial tracking on all tilt-series...");
            IterateOverItems(Workers, CLI, (worker, t) =>
                {
                    worker.TomoStack(t.Path, OptionsStack);
                    worker.TomoEtomoFiducials(t.Path, OptionsEtomo);
                    
                    try
                    {
                        (t as TiltSeries).ImportAlignments(OptionsImport);
                    }
                    catch (Exception exc)
                    {
                        Console.WriteLine("\nFailed to import alignments:\n" + exc.Message);
                    }

                    t.SaveMeta();
                }
            );
            
            // second iteration, calculate new alignments with average tilt axis angle from full dataset
            if (CLI.DoAxisAngleSearch) // only update alignments if it was requested
            {
                OptionsEtomo.AxisAngle = (decimal)CalculateAverageAxis();
                OptionsEtomo.DoFiducialTracking = false;
                OptionsEtomo.DoTiltAlign = true;
                OptionsEtomo.DoAxisAngleSearch = false;  // fix the new tilt axis angle
                
                Console.WriteLine($"Average tilt axis angle from patch tracking: {OptionsEtomo.AxisAngle}");
                Console.WriteLine($"Recalculating alignments for all tilt series with new average tilt axis angle...");
            
                IterateOverItems(Workers, CLI, (worker, t) =>
                    {
                        worker.TomoEtomoFiducials(t.Path, OptionsEtomo);
                    
                        try
                        {
                            (t as TiltSeries).ImportAlignments(OptionsImport);
                        }
                        catch (Exception exc)
                        {
                            Console.WriteLine("\nFailed to import alignments:\n" + exc.Message);
                        }

                        t.SaveMeta();
                    }
                );

            }
            
            if (CLI.DeleteIntermediate)
            {
                Console.Write("Deleting intermediate stacks... ");

                foreach (var t in CLI.InputSeries)
                    Directory.Delete((t as TiltSeries).TiltStackDir, true);

                Console.WriteLine("Done");
            }

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }
    }
}