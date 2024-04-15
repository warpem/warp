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
    [VerbGroup("General")]
    [Verb("change_selection", HelpText = "Change the manual selection status (selected | deselected | null) for every item")]
    [CommandRunner(typeof(ChangeSelection))]
    class ChangeSelectionOptions : BaseOptions
    {
        [Option("select", HelpText = "Change status to selected")]
        public bool Select { get; set; }

        [Option("deselect", HelpText = "Change status to deselected")]
        public bool Deselect { get; set; }

        [Option("null", HelpText = "Change status to null, which is the default status")]
        public bool Null { get; set; }

        [Option("invert", HelpText = "Invert status if it isn't null")]
        public bool Invert { get; set; }
    }

    class ChangeSelection : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            ChangeSelectionOptions CLI = options as ChangeSelectionOptions;
            CLI.Evaluate();

            OptionsWarp Options = CLI.Options;

            #region Set options

            if (new bool[] { CLI.Select, CLI.Deselect, CLI.Null, CLI.Invert }.Count(v => v) != 1)
                throw new Exception("Choose exactly 1 of the options");

            #endregion

            bool? NewValue = null;
            if (CLI.Select)
                NewValue = false;
            else if (CLI.Deselect)
                NewValue = true;

            bool Invert = CLI.Invert;

            int[] StatsBefore = new int[3];
            int[] StatsAfter = new int[3];

            int NDone = 0;
            Console.Write($"0/{CLI.InputSeries.Length}");
            foreach (var item in CLI.InputSeries)
            {
                if (item.UnselectManual.HasValue)
                    StatsBefore[item.UnselectManual.Value ? 0 : 1]++;
                else
                    StatsBefore[2]++;

                if (!Invert)
                {
                    if (item.UnselectManual != NewValue)
                    {
                        item.UnselectManual = NewValue;
                        item.SaveMeta();
                    }
                }
                else
                {
                    if (item.UnselectManual.HasValue)
                    {
                        item.UnselectManual = !item.UnselectManual;
                        item.SaveMeta();
                    }
                }

                if (item.UnselectManual.HasValue)
                    StatsAfter[item.UnselectManual.Value ? 0 : 1]++;
                else
                    StatsAfter[2]++;

                VirtualConsole.ClearLastLine();
                Console.Write($"{++NDone}/{CLI.InputSeries.Length} processed");
            }
            Console.WriteLine("");
            Console.WriteLine($"Before change: {StatsBefore[0]} deselected, {StatsBefore[1]} selected, {StatsBefore[2]} null");
            Console.WriteLine($"After change: {StatsAfter[0]} deselected, {StatsAfter[1]} selected, {StatsAfter[2]} null");
        }
    }
}
