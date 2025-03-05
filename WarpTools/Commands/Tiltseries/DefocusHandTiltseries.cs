using CommandLine;
using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace WarpTools.Commands.Tiltseries
{
    [VerbGroup("Tilt series")]
    [Verb("ts_defocus_hand", HelpText = "Check and/or set defocus handedness for all tilt series")]
    [CommandRunner(typeof(DefocusHandTiltseries))]
    class DefocusHandTiltseriesOptions : BaseOptions
    {
        [Option("check", HelpText = "Only check the defocus handedness, but don't set anything")]
        public bool CheckOnly { get; set; }

        [Option("set_auto", HelpText = "Check the defocus handedness and set the determined value for all tilt series")]
        public bool SetAuto { get; set; }

        [Option("set_flip", HelpText = "Set handedness to 'flip' for all tilt series")]
        public bool SetFlip { get; set; }

        [Option("set_noflip", HelpText = "Set handedness to 'no flip' for all tilt series")]
        public bool SetNoFlip { get; set; }

        [Option("set_switch", HelpText = "Switch whatever handedness value each tilt series has to the opposite value")]
        public bool SetSwitch { get; set; }
    }

    class DefocusHandTiltseries : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            DefocusHandTiltseriesOptions CLI = options as DefocusHandTiltseriesOptions;

            CLI.Evaluate();
            OptionsWarp Options = CLI.Options;

            #region Validate options

            if (new bool[] { CLI.CheckOnly, CLI.SetAuto, CLI.SetFlip, CLI.SetNoFlip, CLI.SetSwitch }.Count(v => v) != 1)
                throw new Exception("Choose exactly 1 of the options");

            #endregion

            float CorrelationAll = 0;

            if (CLI.CheckOnly || CLI.SetAuto)
            {
                Console.WriteLine("Checking defocus handedness for all tilt series...");
                Console.Write($"0/{CLI.InputSeries.Length}");

                var Correlations = new List<float>();

                IterateOverItems<TiltSeries>(null, CLI, (_, series) =>
                {
                    bool OriginalFlip = series.AreAnglesInverted;
                    series.AreAnglesInverted = false;

                    Movie[] TiltMovies = series.TiltMoviePaths
                                               .Select(s => new Movie(Path.Combine(series.DataOrProcessingDirectoryName,
                                                                                   s))).ToArray();

                    if (TiltMovies.Any(m => m.GridCTFDefocus.Values.Length < 2))
                        throw new Exception("One or more tilt movies don't have local defocus information. " +
                                            "Please run fs_ctf on all individual tilt movies using a 2x2x1 grid.");

                    series.VolumeDimensionsPhysical = new float3((float)Options.Tomo.DimensionsX,
                                                                 (float)Options.Tomo.DimensionsY,
                                                                 (float)Options.Tomo.DimensionsZ) *
                                                      (float)Options.Import.PixelSize;
                    series.ImageDimensionsPhysical =
                        new float2(series.VolumeDimensionsPhysical.X, series.VolumeDimensionsPhysical.Y);

                    float[] GradientsEstimated = new float[series.NTilts];
                    float[] GradientsAssumed = new float[series.NTilts];

                    float3[] Points =
                    [
                        new float3(0, series.VolumeDimensionsPhysical.Y / 2, series.VolumeDimensionsPhysical.Z / 2),
                        new float3(series.VolumeDimensionsPhysical.X, series.VolumeDimensionsPhysical.Y / 2,
                                   series.VolumeDimensionsPhysical.Z / 2)
                    ];

                    float3[] Projected0 = series.GetPositionInAllTilts(Points[0])
                                                .Select(v => v / new float3(series.ImageDimensionsPhysical.X,
                                                                            series.ImageDimensionsPhysical.Y, 1))
                                                .ToArray();
                    float3[] Projected1 = series.GetPositionInAllTilts(Points[1])
                                                .Select(v => v / new float3(series.ImageDimensionsPhysical.X,
                                                                            series.ImageDimensionsPhysical.Y, 1))
                                                .ToArray();

                    for (int t = 0; t < series.NTilts; t++)
                    {
                        float Interp0 = TiltMovies[t].GridCTFDefocus
                                                     .GetInterpolated(new float3(Projected0[t].X, Projected0[0].Y,
                                                                                 0.5f));
                        float Interp1 = TiltMovies[t].GridCTFDefocus
                                                     .GetInterpolated(new float3(Projected1[t].X, Projected1[0].Y,
                                                                                 0.5f));
                        GradientsEstimated[t] = Interp1 - Interp0;

                        GradientsAssumed[t] = Projected1[t].Z - Projected0[t].Z;
                    }

                    if (GradientsEstimated.Length > 1)
                    {
                        GradientsEstimated = MathHelper.Normalize(GradientsEstimated);
                        GradientsAssumed = MathHelper.Normalize(GradientsAssumed);
                    }
                    else
                    {
                        GradientsEstimated[0] = Math.Sign(GradientsEstimated[0]);
                        GradientsAssumed[0] = Math.Sign(GradientsAssumed[0]);
                    }

                    float Correlation = MathHelper.DotProduct(GradientsEstimated, GradientsAssumed) /
                                        GradientsEstimated.Length;
                    Correlations.Add(Correlation);

                    series.AreAnglesInverted = OriginalFlip;
                }, crashOnFail: true);

                CorrelationAll = Correlations.Average();

                Console.WriteLine($"Average correlation: {CorrelationAll:F3}");
                if (CorrelationAll < 0)
                    Console.WriteLine("The average correlation is negative, which means that the defocus handedness should be set to 'flip'");
                else if (CorrelationAll > 0)
                    Console.WriteLine("The average correlation is positive, which means that the defocus handedness should be set to 'no flip'");
                else
                    throw new Exception("The average correlation is 0, which shouldn't happen");

                if (CLI.SetAuto)
                {
                    Console.WriteLine($"Setting defocus handedness to '{(CorrelationAll > 0 ? "no flip" : "flip")}' for all tilt series...");
                    Console.Write($"0/{CLI.InputSeries.Length}");

                    int NDone = 0;
                    foreach (var series in CLI.InputSeries.Select(m => (TiltSeries)m))
                    {
                        series.AreAnglesInverted = CorrelationAll < 0;
                        series.SaveMeta();

                        VirtualConsole.ClearLastLine();
                        Console.Write($"{++NDone}/{CLI.InputSeries.Length}");
                    }

                    Console.WriteLine("");
                }
            }
            else
            {
                if (CLI.SetFlip)
                    Console.WriteLine("Setting defocus handedness to 'flip' for all tilt series...");
                else if (CLI.SetNoFlip)
                    Console.WriteLine("Setting defocus handedness to 'no flip' for all tilt series...");
                else if (CLI.SetSwitch)
                    Console.WriteLine("Switching defocus handedness for all tilt series...");
                else
                    throw new Exception("This shouldn't happen");

                Console.Write($"0/{CLI.InputSeries.Length}");
                int NDone = 0;
                foreach (var series in CLI.InputSeries.Select(m => (TiltSeries)m))
                {
                    if (CLI.SetFlip)
                        series.AreAnglesInverted = true;
                    else if (CLI.SetNoFlip)
                        series.AreAnglesInverted = false;
                    else if (CLI.SetSwitch)
                        series.AreAnglesInverted = !series.AreAnglesInverted;

                    series.SaveMeta();

                    VirtualConsole.ClearLastLine();
                    Console.Write($"{++NDone}/{CLI.InputSeries.Length}");
                }

                Console.WriteLine("");
            }
        }
    }
}
