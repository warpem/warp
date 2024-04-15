using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using OpenAI_API.Moderation;

namespace WarpTools.Commands.General
{
    [VerbGroup("General")]
    [Verb("move_data", HelpText = "Changes the location of raw data in XML metadata; use it when the raw data moves, or you switch between Windows and Linux")]
    [CommandRunner(typeof(MoveData))]
    class MoveDataOptions : BaseOptions
    {
        [Option("to", Required = true, HelpText = "New directory containing raw data; if raw data are located in sub-folders and require recursive search, specify the top directory here")]
        public string To { get; set; }

        [Option("new_settings", Required = true, HelpText = "Where to save an updated .settings file containing the new raw data location")]
        public string NewSettings { get; set; }
    }

    class MoveData : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            MoveDataOptions CLI = options as MoveDataOptions;

            OptionsWarp OldOptions = new OptionsWarp();
            OldOptions.Load(CLI.SettingsPath);

            OldOptions.Import.DataFolder = CLI.To;

            OldOptions.Save(CLI.NewSettings);

            CLI.Evaluate();

            Console.WriteLine("Saving updated metadata...");
            Console.Write($"0/{CLI.InputSeries.Length}");
            int NDone = 0;

            Helper.ForCPUGreedy(0, CLI.InputSeries.Length, 8, null, (i, threadID) =>
            {
                CLI.InputSeries[i].SaveMeta();

                lock (CLI.InputSeries)
                {
                    NDone++;

                    if (NDone % 10 == 0 || NDone == CLI.InputSeries.Length)
                    {
                        VirtualConsole.ClearLastLine();
                        Console.Write($"{NDone}/{CLI.InputSeries.Length}");
                    }
                }
            }, null);     

            Console.WriteLine();
        }
    }
}
