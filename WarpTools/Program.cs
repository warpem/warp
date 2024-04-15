using System;
using System.Linq;
using CommandLine;
using System.Reflection;
using WarpTools.Commands;
using Warp.Tools;
using System.Threading.Tasks;

namespace WarpTools
{
    class WarpTools
    {
        static async Task Main(string[] args)
        {
            VirtualConsole.AttachToConsole();

            await CommandLineParserHelper.ParseAndRun(args, Run, Verbs, "WarpTools - a collection of tools for EM data pre-processing");
        }

        //Load and alphabetically sort all verb types using reflection
        private static Type[] Verbs => Assembly.GetExecutingAssembly().GetTypes().Where(t => t.GetCustomAttribute<VerbAttribute>() != null).OrderBy(t => t.Name).ToArray();

        private static async Task Run(object options)
        {
            var Attributes = options.GetType().GetCustomAttributes(typeof(CommandRunner), false);
            if (Attributes.Length > 0)
            {
                Type RunnerType = ((CommandRunner)Attributes[0]).Type;
                var RunnerInstance = (BaseCommand)Activator.CreateInstance(RunnerType);
                await RunnerInstance.Run(options);
            }
            else
                Console.WriteLine($"Unknown command of type {options.GetType()}, exiting");
        }
    }
}
