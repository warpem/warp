using CommandLine;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands
{
    [VerbGroup("Tilt series")]
    [Verb("ts_template_match", HelpText = "Match previously reconstructed tomograms against a 3D template, producing a list of the highest-scoring matches")]
    [CommandRunner(typeof(TemplateMatchTiltseries))]
    class TemplateMatchTiltseriesOptions : DistributedOptions
    {
        [Option("tomo_angpix", Required = true, HelpText = "Pixel size of the reconstructed tomograms in Angstrom")]
        public double TomoAngPix { get; set; }

        [Option("template_path", HelpText = "Path to the template file")]
        public string TemplatePath { get; set; }
        public string FlippedTemplatePath { get; set; } = null;

        [Option("template_emdb", HelpText = "Instead of providing a local map, download the EMDB entry with this ID and use its main map")]
        public int? TemplateEMDB { get; set; }

        [Option("template_angpix", HelpText = "Pixel size of the template; leave empty to use value from map header")]
        public double? TemplateAngPix { get; set; }

        [Option("template_diameter", Required = true, HelpText = "Template diameter in Angstrom")]
        public int TemplateDiameter { get; set; }

        [Option("template_flip", HelpText = "Mirror the template along the X axis to flip the handedness; '_flipx' will be added to the template's name")]
        public bool TemplateFlip { get; set; }

        [Option("symmetry", Default = "C1", HelpText = "Symmetry of the template, e.g. C1, D7, O")]
        public string TemplateSymmetry { get; set; }

        [Option("subdivisions", Default = 3, HelpText = "Number of subdivisions defining the angular search step: 2 = 15° step, 3 = 7.5°, 4 = 3.75° and so on")]
        public int HealpixOrder { get; set; }

        [Option("tilt_range", HelpText = "Limit the range of angles between the reference's Z axis and the tomogram's XY plane to plus/minus this value, in °; useful for matching filaments lying flat in the XY plane")]
        public double? TiltRange { get; set; }

        [Option("batch_angles", Default = 32, HelpText = "How many orientations to evaluate at once; memory consumption scales linearly with this; higher than 32 probably won't lead to speed-ups")]
        public int BatchAngles { get; set; }

        [Option("peak_distance", HelpText = "Minimum distance (in Angstrom) between peaks; leave empty to use template diameter")]
        public int? PeakDistance { get; set; }

        [Option("npeaks", Default = 2000, HelpText = "Maximum number of peak positions to save")]
        public int PeakNumber { get; set; }

        [Option("dont_normalize", HelpText = "Don't set score distribution to median = 0, stddev = 1")]
        public bool DontNormalizeScores { get; set; }

        [Option("whiten", HelpText = "Perform spectral whitening to give higher-resolution information more weight; this can help when the alignments are already good and you need more selective matching")]
        public bool Whiten { get; set; }

        [Option("lowpass", Default = 1.0, HelpText = "Gaussian low-pass filter to be applied to template and tomogram, in fractions of Nyquist; 1.0 = no low-pass, <1.0 = low-pass")]
        public double Lowpass { get; set; }

        [Option("lowpass_sigma", Default = 0.1, HelpText = "Sigma (i.e. fall-off) of the Gaussian low-pass filter, in fractions of Nyquist; larger value = slower fall-off")]
        public double LowpassSigma { get; set; }

        [Option("max_missing_tilts", Default = 2, HelpText = "Dismiss positions not covered by at least this many tilts; set to -1 to disable position culling")]
        public int MaxMissingTilts { get; set; }

        [Option("reuse_results", HelpText = "Reuse correlation volumes from a previous run if available, only extract peak positions")]
        public bool ReuseResults { get; set; }

        [Option("check_hand", Default = 0, HelpText = "Also try a flipped version of the template on this many tomograms to see what geometric hand they have")]
        public int CheckHandN { get; set; }

        [Option("subvolume_size", Default = 192, HelpText = "Matching is performed locally using sub-volumes of this size in pixel")]
        public int SubVolumeSize { get; set; }
    }

    class TemplateMatchTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            TemplateMatchTiltseriesOptions CLI = options as TemplateMatchTiltseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (CLI.TomoAngPix <= 0)
                throw new Exception("--tomo_angpix can't be 0 or negative");

            if (string.IsNullOrEmpty(CLI.TemplatePath) && !CLI.TemplateEMDB.HasValue)
                throw new Exception("Either --template_path or --template_emdb must be specified");

            if (!string.IsNullOrEmpty(CLI.TemplatePath) && CLI.TemplateEMDB.HasValue)
                throw new Exception("Only one of --template_path and --template_emdb can be specified");

            if (!string.IsNullOrEmpty(CLI.TemplatePath) && !File.Exists(CLI.TemplatePath))
                throw new Exception("Template file doesn't exist");

            if (CLI.TemplateEMDB.HasValue && CLI.TemplateEMDB.Value <= 0)
                throw new Exception("--template_emdb can't be 0 or negative");

            if (CLI.TemplateAngPix.HasValue && CLI.TemplateAngPix.Value <= 0)
                throw new Exception("--template_angpix can't be 0 or negative");

            if (CLI.TemplateDiameter <= 0)
                throw new Exception("--template_diameter can't be 0 or negative");

            try
            {
                Symmetry TemplateSymmetry = new Symmetry(CLI.TemplateSymmetry);
            }
            catch (Exception e)
            {
                throw new Exception("Invalid --symmetry specified: " + e.Message);
            }

            if (CLI.HealpixOrder < 0)
                throw new Exception("--subdivisions can't be negative");

            if (CLI.TiltRange != null && (CLI.TiltRange.Value > 90 || CLI.TiltRange.Value < 0))
                throw new Exception("--tilt_range must be between 0 and 90");

            if (CLI.BatchAngles < 1)
                throw new Exception("--batch_angles must be positive");

            if (CLI.PeakDistance.HasValue && CLI.PeakDistance.Value <= 0)
                throw new Exception("--peak_distance can't be 0 or negative");

            if (CLI.PeakNumber <= 0)
                throw new Exception("--npeaks can't be 0 or negative");

            if (CLI.CheckHandN < 0)
                throw new Exception("--check_hand can't be negative");

            if (CLI.TemplateFlip && CLI.CheckHandN > 0)
                throw new Exception("--template_flip and --check_hand can't be used together");

            if (CLI.SubVolumeSize < 64)
                throw new Exception("--subvolume_size can't be lower than 64");

            if (CLI.Lowpass < 0 || CLI.Lowpass > 1)
                throw new Exception("--lowpass must be between 0 and 1");

            if (CLI.LowpassSigma < 0)
                throw new Exception("--lowpass_sigma can't be negative");


            #endregion

            #region Create processing options

            Options.Tasks.TomoFullReconstructPixel = (decimal)CLI.TomoAngPix;

            if (CLI.TemplateAngPix.HasValue)
                Options.Tasks.TomoMatchTemplatePixel = (decimal)CLI.TemplateAngPix;
            Options.Tasks.TomoMatchTemplateDiameter = CLI.TemplateDiameter;
            Options.Tasks.TomoMatchPeakDistance = CLI.PeakDistance.HasValue ? CLI.PeakDistance.Value : CLI.TemplateDiameter;
            Options.Tasks.TomoMatchTemplateFraction = 1;

            Options.Tasks.TomoMatchHealpixOrder = CLI.HealpixOrder;
            Options.Tasks.TomoMatchBatchAngles = CLI.BatchAngles;
            Options.Tasks.TomoMatchSymmetry = CLI.TemplateSymmetry;
            Options.Tasks.TomoMatchNResults = CLI.PeakNumber;

            Options.Tasks.ReuseCorrVolumes = CLI.ReuseResults;
            Options.Tasks.TomoMatchWhitenSpectrum = CLI.Whiten;

            var OptionsMatch = Options.GetProcessingTomoFullMatch();

            OptionsMatch.TiltRange = CLI.TiltRange != null ? (decimal)CLI.TiltRange.Value : -1;
            OptionsMatch.SubVolumeSize = CLI.SubVolumeSize;
            OptionsMatch.Supersample = 1;
            OptionsMatch.MaxMissingTilts = CLI.MaxMissingTilts;
            OptionsMatch.NormalizeScores = !CLI.DontNormalizeScores;
            OptionsMatch.Lowpass = (decimal)CLI.Lowpass;
            OptionsMatch.LowpassSigma = (decimal)CLI.LowpassSigma;

            #endregion

            string TemplateDir = Path.Combine(CLI.OutputProcessing, "template");
            Directory.CreateDirectory(TemplateDir);

            #region Download EMDB if necessary

            if (CLI.TemplateEMDB.HasValue)
            {
                // EMD IDs below 10000 are padded with zeros
                string ID = CLI.TemplateEMDB.Value.ToString("D4");
                string Url = $"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{ID}/map/emd_{ID}.map.gz";
                string OutputPath = Path.Combine(TemplateDir, $"emd_{ID}.mrc");

                if (!File.Exists(OutputPath))
                {
                    byte[] DownloadedData = await DownloadFileAsync(Url);

                    Console.Write("Extracting downloaded map...");
                    ExtractGzArchive(DownloadedData, OutputPath);
                    Console.WriteLine(" Done");
                }
                else
                {
                    Console.WriteLine($"EMD-{ID} already exists in {OutputPath}, skipping download");
                }

                CLI.TemplatePath = OutputPath;
            }
            else
            {
                CLI.TemplatePath = Helper.PathCombine(Environment.CurrentDirectory, CLI.TemplatePath);
            }

            #endregion

            #region Prepare template

            Image TemplateOri = Image.FromFile(CLI.TemplatePath);
            if (CLI.TemplateAngPix == null)
            {
                if (TemplateOri.PixelSize <= 0)
                    throw new Exception("Couldn't determine pixel size from template, please specify --template_angpix");

                CLI.TemplateAngPix = TemplateOri.PixelSize;
                OptionsMatch.TemplatePixel = (decimal)TemplateOri.PixelSize;
                Console.WriteLine($"Setting --template_angpix to {TemplateOri.PixelSize} based on template map");
            }

            Image TemplateFlipped = null;
            if (CLI.TemplateFlip || CLI.CheckHandN > 0)
            {
                Console.Write("Preparing flipped template... ");

                TemplateFlipped = TemplateOri.AsFlippedX();

                string FlippedPath = Path.Combine(TemplateDir, Path.GetFileNameWithoutExtension(CLI.TemplatePath) + "_flipx.mrc");
                CLI.FlippedTemplatePath = FlippedPath;
                TemplateFlipped.WriteMRC(FlippedPath, (float)CLI.TemplateAngPix, true);

                if (CLI.TemplateFlip)
                    CLI.TemplatePath = FlippedPath;

                Console.WriteLine("Done");
            }

            #endregion

            {
                var HealpixAngles = Helper.GetHealpixAngles(OptionsMatch.HealpixOrder, OptionsMatch.Symmetry);
                if (CLI.TiltRange >= 0)
                {
                    float Limit = MathF.Sin((float)CLI.TiltRange * Helper.ToRad);
                    HealpixAngles = HealpixAngles.Where(a => MathF.Abs(Matrix3.Euler(a).C3.Z) <= Limit).ToArray();
                }
                Console.WriteLine($"Using {HealpixAngles.Length} orientations for matching");
                if (HealpixAngles.Length == 0)
                    throw new Exception("Can't match with 0 orientations, please increase --tilt_range");
            }

            WorkerWrapper[] Workers = CLI.GetWorkers();

            if (CLI.CheckHandN > 0)
            {
                var AllItems = CLI.InputSeries;
                List<float> ScoresOriginal = new();
                List<float> ScoresFlipped = new();

                CLI.InputSeries = AllItems.Take(CLI.CheckHandN).ToArray();

                // Unflipped
                {
                    Console.WriteLine("Testing matching with original template:");

                    OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.TemplatePath);

                    IterateOverItems(Workers, CLI, (worker, m) =>
                    {
                        worker.TomoMatch(m.Path, OptionsMatch, CLI.TemplatePath);

                        string PeakTablePath = Path.Combine(m.MatchingDir, m.RootName +
                                                                           $"_{OptionsMatch.BinnedPixelSizeMean:F2}Apx" +
                                                                           "_" + OptionsMatch.TemplateName + ".star");
                        List<float> PeakValues = Star.LoadFloat(PeakTablePath, "rlnAutopickFigureOfMerit").ToList();
                        PeakValues.Sort();
                        PeakValues = PeakValues.TakeLast(20).ToList();

                        lock(Workers)
                            ScoresOriginal.AddRange(PeakValues);
                    });

                    Console.WriteLine($"Average top peak value with original template: {ScoresOriginal.Average():F3}");
                }

                // Flipped
                {
                    Console.WriteLine("Testing matching with flipped template:");

                    OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.FlippedTemplatePath);

                    IterateOverItems(Workers, CLI, (worker, m) =>
                    {
                        worker.TomoMatch(m.Path, OptionsMatch, CLI.FlippedTemplatePath);

                        string PeakTablePath = Path.Combine(m.MatchingDir, m.RootName +
                                                                           $"_{OptionsMatch.BinnedPixelSizeMean:F2}Apx" +
                                                                           "_" + OptionsMatch.TemplateName + ".star");
                        List<float> PeakValues = Star.LoadFloat(PeakTablePath, "rlnAutopickFigureOfMerit").ToList();
                        PeakValues.Sort();
                        PeakValues = PeakValues.TakeLast(20).ToList();

                        lock (Workers)
                            ScoresFlipped.AddRange(PeakValues);
                    });

                    Console.WriteLine($"Average top peak value with flipped template: {ScoresFlipped.Average():F3}");
                }

                if (ScoresFlipped.Average() > ScoresOriginal.Average())
                {
                    Console.WriteLine("Flipped template has higher peak values, using it for further processing");
                    CLI.TemplatePath = CLI.FlippedTemplatePath;
                }
                else
                {
                    Console.WriteLine("Original template has higher peak values, using it for further processing");
                }

                CLI.InputSeries = AllItems;
            }

            OptionsMatch.TemplateName = Path.GetFileNameWithoutExtension(CLI.TemplatePath);

            IterateOverItems(Workers, CLI, (worker, m) =>
            {
                worker.TomoMatch(m.Path, OptionsMatch, CLI.TemplatePath);
            });

            Console.Write("Saying goodbye to all workers...");
            foreach (var worker in Workers)
                worker.Dispose();
            Console.WriteLine(" Done");
        }

        async Task<byte[]> DownloadFileAsync(string url)
        {
            HttpClient HttpClient = new HttpClient();
            HttpResponseMessage Response = await HttpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);

            if (!Response.IsSuccessStatusCode)
            {
                Console.WriteLine($"Failed to download file: {Response.StatusCode}");
                return null;
            }

            Console.Write($"Downloading map from EMDB: 0%");
            using (MemoryStream memoryStream = new MemoryStream())
            {
                // Get content stream and length
                using (Stream contentStream = await Response.Content.ReadAsStreamAsync())
                {
                    long TotalBytes = Response.Content.Headers.ContentLength.GetValueOrDefault(0L);
                    long TotalReadBytes = 0L;

                    byte[] Buffer = new byte[8192];
                    int BytesRead;

                    while ((BytesRead = await contentStream.ReadAsync(Buffer, 0, Buffer.Length)) != 0)
                    {
                        await memoryStream.WriteAsync(Buffer, 0, BytesRead);
                        TotalReadBytes += BytesRead;

                        double Progress = (double)TotalReadBytes / TotalBytes * 100;

                        VirtualConsole.ClearLastLine();
                        Console.Write($"Downloading map from EMDB: {Progress:F2}%");
                    }
                    Console.WriteLine("");
                }

                return memoryStream.ToArray();
            }
        }

        void ExtractGzArchive(byte[] data, string outputPath)
        {
            using (MemoryStream memoryStr = new MemoryStream(data))
            {
                using (GZipStream decompressStream = new GZipStream(memoryStr, CompressionMode.Decompress))
                {
                    string OutputFile = outputPath;
                    using (FileStream OutputStr = new FileStream(OutputFile, FileMode.Create))
                    {
                        decompressStream.CopyTo(OutputStr);
                    }
                }
            }
        }
    }
}
