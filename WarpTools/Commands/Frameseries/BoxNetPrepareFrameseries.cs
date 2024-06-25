using CommandLine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace WarpTools.Commands.Frameseries
{
    [VerbGroup("Frame series")]
    [Verb("fs_boxnet_prepare", HelpText = "Prepare examples for BoxNet picker and/or denoiser training (fs_boxnet_train)")]
    [CommandRunner(typeof(BoxNetPrepareFrameseries))]
    class BoxNetPrepareFrameseriesOptions : BaseOptions
    {
        #region Segmentations

        [OptionGroup("Prepare segmentations for picker training")]
        [Option("out_segmentations", HelpText = "If specified, a TIFF stack containing the segmentation examples used to train the picking part of BoxNet will be saved under this path")]
        public string PathSegmentations { get; set; }

        [Option("use_masks", HelpText = "Include 💩 masks from the 'mask' folder in the segmentation. The names are expected to match the micrographs")]
        public bool UseMasks { get; set; }

        [Option("no_strict_masks", HelpText = "If --use_masks is specified, also use micrographs for which no mask files are available. Can't combine with --no_strict_particles")]
        public bool NoStrictMasks { get; set; }

        [Option("use_particles", HelpText = "Include particles positions from the 'matching' folder in the segmentation")]
        public bool UseParticles { get; set; }

        [Option("particles_suffix", HelpText = "If particle positions are available, use this suffix to match the STAR files to micrographs like this: {micrograph name}_{suffix}.star. If left empty, will look for {micrograph name}.star")]
        public string ParticlesSuffix { get; set; }

        [Option("particles_diameter", HelpText = "If particle positions are available, use this approximate particle diameter in Angstrom")]
        public int? ParticleDiameter { get; set; }

        [Option("no_strict_particles", HelpText = "If --use_particles is specified, also use micrographs for which no particle files are available. Can't combine with --no_strict_masks")]
        public bool NoStrictParticles { get; set; }

        [Option("negative_contrast", HelpText = "Use this when working with negative stain data, i.e. when mass = dark")]
        public bool NegativeContrast { get; set; }

        #endregion

        #region Half-averages

        [OptionGroup("Prepare half-averages for denoiser training")]
        [Option("out_half_averages", HelpText = "If specified, an MRC stack containing the half-averages used to train the denoising part of BoxNet will be saved under this path")]
        public string PathHalfAverages { get; set; }

        [Option("max_mics", Default = 256, HelpText = "The maximum number of micrographs to export as half-averages")]
        public int MaxMics { get; set; }

        #endregion
    }

    class BoxNetPrepareFrameseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            BoxNetPrepareFrameseriesOptions CLI = options as BoxNetPrepareFrameseriesOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (string.IsNullOrEmpty(CLI.PathSegmentations) && string.IsNullOrEmpty(CLI.PathHalfAverages))
                throw new Exception("At least one of --out_segmentations or --out_half_averages must be specified");

            if (!string.IsNullOrEmpty(CLI.ParticlesSuffix))
                CLI.ParticlesSuffix = "_" + CLI.ParticlesSuffix;

            if (!string.IsNullOrEmpty(CLI.PathSegmentations))
            {
                if (!CLI.UseMasks && !CLI.UseParticles)
                    throw new Exception("At least one of --use_masks or --use_particles must be specified");

                if (CLI.NoStrictMasks && CLI.NoStrictParticles)
                    throw new Exception("Can't use --no_strict_masks and --no_strict_particles at the same time");

                if (CLI.UseMasks)
                {
                    if (!CLI.InputSeries.Any(s => File.Exists(s.MaskPath)))
                        throw new Exception("--use_masks specified, but not a single micrograph has a mask associated with it");
                }

                if (CLI.UseParticles)
                {
                    if (!CLI.InputSeries.Any(s => File.Exists(Path.Combine(s.MatchingDir, $"{s.RootName}{CLI.ParticlesSuffix}.star"))))
                        throw new Exception("--use_particles specified, but not a single micrograph has particle positions associated with it");

                    if (!CLI.ParticleDiameter.HasValue || CLI.ParticleDiameter.Value <= 0)
                        throw new Exception("--particles_diameter must have a positive value if --use_particles is specified");
                }
            }

            if (!string.IsNullOrEmpty(CLI.PathHalfAverages))
            {
                if (CLI.MaxMics <= 0)
                    throw new Exception("--max_mics must be positive");
            }

            #endregion

            #region Prepare segmentations

            List<Movie> MicsPicking = null;
            
            if (!string.IsNullOrEmpty(CLI.PathSegmentations))
            {
                #region Figure out what micrographs to use

                MicsPicking = CLI.InputSeries.ToList();

                if (CLI.UseMasks && !CLI.NoStrictMasks)
                {
                    MicsPicking = MicsPicking.Where(m => File.Exists(m.MaskPath)).ToList();

                    if (MicsPicking.Count == 0)
                        throw new Exception("No micrographs left after removing those without masks.");
                }

                if (CLI.UseParticles && !CLI.NoStrictParticles)
                {
                    MicsPicking = MicsPicking.Where(m => File.Exists(Path.Combine(m.MatchingDir, $"{m.RootName}{CLI.ParticlesSuffix}.star"))).ToList();

                    if (MicsPicking.Count == 0)
                        throw new Exception("No micrographs left after removing those without particle positions.");
                }

                if (MicsPicking.Any(s => !File.Exists(s.AveragePath)))
                    throw new Exception("Not all micrographs have an aligned average file associated with them");

                if (MicsPicking.Count == 0)
                    throw new Exception("No micrographs left for preparing segmentations.");

                #endregion

                MapHeader MicHeader = MapHeader.ReadFromFile(MicsPicking.First().AveragePath);
                int2 Dims8Apx;
                {
                    float2 DimsMicAng = new float2(new int2(MicHeader.Dimensions)) * MicHeader.PixelSize.X;
                    Dims8Apx = new int2((DimsMicAng / BoxNetMulti.PixelSize + 1) / 2) * 2;
                }

                Image Stack = new Image(new int3(Dims8Apx.X, Dims8Apx.Y, MicsPicking.Count * 3));

                #region Paint all segmentations

                Helper.ForCPUGreedy(0, MicsPicking.Count, 8, null, (imic, threadID) =>
                {
                    Image MicScaled = Image.FromFile(MicsPicking[imic].AveragePath).AsScaled(Dims8Apx).AndDisposeParent();
                    if (CLI.NegativeContrast)
                        MicScaled.Multiply(-1f);
                    MicScaled.FreeDevice();
                    Stack.GetHost(Intent.Write)[imic * 3 + 0] = MicScaled.GetHost(Intent.Read)[0];

                    float[] MicMask = null;
                    {
                        if (CLI.UseMasks && File.Exists(MicsPicking[imic].MaskPath))
                            MicMask = Image.FromFile(MicsPicking[imic].MaskPath).GetHost(Intent.Read)[0];
                        else
                            MicMask = new float[MicScaled.ElementsReal];
                    }

                    byte[] MicParticles = new byte[MicScaled.ElementsReal];
                    {
                        string PathStar = Path.Combine(MicsPicking[imic].MatchingDir, $"{MicsPicking[imic].RootName}{CLI.ParticlesSuffix}.star");
                        if (File.Exists(PathStar))
                        {
                            float2[] Positions = Star.LoadFloat2(PathStar, "rlnCoordinateX", "rlnCoordinateY");
                            Positions = Positions.Select(p => p * MicHeader.PixelSize.X / BoxNetMulti.PixelSize).ToArray();

                            float R = Math.Max(1.5f, 
                                               CLI.ParticleDiameter.Value / 2f / BoxNetMulti.PixelSize / 4);
                            float R2 = R * R;

                            foreach (var pos in Positions)
                            {
                                int2 Min = new int2(Math.Max(0, (int)(pos.X - R)), Math.Max(0, (int)(pos.Y - R)));
                                int2 Max = new int2(Math.Min(MicScaled.Dims.X - 1, (int)(pos.X + R)), Math.Min(MicScaled.Dims.Y - 1, (int)(pos.Y + R)));

                                for (int y = Min.Y; y <= Max.Y; y++)
                                {
                                    float yy = y - pos.Y;
                                    yy *= yy;
                                    for (int x = Min.X; x <= Max.X; x++)
                                    {
                                        float xx = x - pos.X;
                                        xx *= xx;

                                        float r2 = xx + yy;
                                        if (r2 <= R2)
                                            MicParticles[y * MicScaled.Dims.X + x] = 1;
                                    }
                                }
                            }
                        }
                    }

                    float[] LabelsData = Stack.GetHost(Intent.Write)[imic * 3 + 1];
                    for (int i = 0; i < LabelsData.Length; i++)
                        LabelsData[i] = MicParticles[i] == 1 ? 1 : (MicMask[i] != 0 ? 2 : 0);

                    float[] CertaintyData = Stack.GetHost(Intent.Write)[imic * 3 + 2];
                    for (int i = 0; i < LabelsData.Length; i++)
                        CertaintyData[i] = 1;
                    
                }, null);

                #endregion

                Stack.WriteTIFF(CLI.PathSegmentations, 8, typeof(float));
            }

            #endregion
        }
    }
}
