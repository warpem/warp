using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;


namespace WarpTools.Commands.Frameseries
{
    [VerbGroup("Frame series")]
    [Verb("fs_boxnet_infer", HelpText = "Run a trained BoxNet model on frameseries averages, producing particle positions and masks")]
    [CommandRunner(typeof(BoxNetInferFrameseries))]
    class BoxNetInferFrameseriesOptions : DistributedOptions
    {
        [Option("model", Required = true, HelpText = "Path to the .pt file containing the model weights")]
        public string Model { get; set; }

        [Option("perprocess", Default = 4, HelpText = "Number of threads per process; the model is loaded only once per process and oversubscription can save memory")]
        public int NThreads { get; set; }

        [Option("diameter", Default = 100.0, HelpText = "Approximate particle diameter in Angstrom")]
        public double Diameter { get; set; }

        [Option("threshold", Default = 0.5, HelpText = "Picking score threshold, between 0.0 and 1.0")]
        public double Threshold { get; set; }

        [Option("distance", Default = 0.0, HelpText = "Minimum distance in Angstrom to maintain between picked positions and masked pixels")]
        public double Distance { get; set; }

        [Option("negative", HelpText = "Expect negative stain-like contrast (mass = bright)")]
        public bool Negative { get; set; }

        [Option("suffix_star", HelpText = "Override the suffix added to the particle STAR file; leave empty to use model name")]
        public string SuffixStar { get; set; }

        [Option("patchsize", Default = 512, HelpText = "Size of the BoxNet input window that the model was trained with, a multiple of 256; the default for models shipped with Warp is 512")]
        public int PatchSize { get; set; }
    }

    class BoxNetInferFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            BoxNetInferFrameseriesOptions CLI = options as BoxNetInferFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            List<string> TryModelPaths = new List<string>() 
            {
                Helper.PathCombine(CLI.InputProcessing, CLI.Model),
                Helper.PathCombine(CLI.OutputProcessing, CLI.Model), 
                CLI.Model 
            };

            string ModelPath = TryModelPaths.FirstOrDefault(p => File.Exists(p));

            #region Validate options

            if (ModelPath == null)
                throw new Exception("Model file not found");

            if (CLI.Threshold < 0.0 || CLI.Threshold > 1.0)
                throw new Exception("Threshold must be between 0.0 and 1.0");

            if (CLI.Diameter <= 0.0)
                throw new Exception("Diameter must be positive");

            if (CLI.Distance < 0.0)
                throw new Exception("Distance must be non-negative");

            if (CLI.NThreads <= 0)
                throw new Exception("Number of threads must be positive");

            if (CLI.PatchSize % 256 != 0)
                throw new Exception("Patch size must be a multiple of 256");

            if (CLI.PatchSize <= 0)
                throw new Exception("Patch size must be positive");

            #endregion

            #region Set options

            Options.Picking.ModelPath = ModelPath;
            Options.Picking.DataStyle = CLI.Negative ? "negative" : "cryo";
            Options.Picking.Diameter = (int)CLI.Diameter;
            Options.Picking.MinimumScore = (decimal)CLI.Threshold;
            Options.Picking.MinimumMaskDistance = (decimal)CLI.Distance;

            ProcessingOptionsBoxNet OptionsPick = Options.GetProcessingBoxNet();

            OptionsPick.OverrideStarSuffix = CLI.SuffixStar;

            #endregion

            WorkerWrapper[] Workers = CLI.GetWorkers();

            #region Load model

            Console.Write("Loading model...");

            Helper.ForCPU(0, Workers.Length, Workers.Length, null, (iworker, threadID) =>
            {
                WorkerWrapper Worker = Workers[threadID];
                Worker.LoadBoxNet(ModelPath, CLI.PatchSize, 1);
            }, null);

            Console.WriteLine(" Done");

            #endregion

            IterateOverItems<Movie>(
                Workers,
                CLI,
                (worker, m) => { worker.MoviePickBoxNet(m.Path, OptionsPick); },
                oversubscribe: CLI.NThreads
            );

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");

            Console.Write("Saving settings...");
            Options.Save(Path.Combine(CLI.OutputProcessing, "align_frameseries.settings"));
            Console.WriteLine(" Done");
        }
    }
}
