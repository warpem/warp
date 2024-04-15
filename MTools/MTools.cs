using System;
using System.Linq;
using CommandLine;
using System.Reflection;
using MTools.Commands;
using Warp.Tools;

namespace MTools
{
    class MTools
    {
        static void Main(string[] args)
        {
            VirtualConsole.AttachToConsole();

            //List<string> VerbNames = Verbs.Select(v => v.GetCustomAttribute<VerbAttribute>().Name).ToList();
            //VerbNames.Sort();
            //foreach (var verb in VerbNames)
            //    Console.WriteLine(verb);

            Parser.Default.ParseArguments(args, Verbs).WithParsed(Run);
        }

        //Load all verb types using reflection
        private static Type[] Verbs => Assembly.GetExecutingAssembly().GetTypes().Where(t => t.GetCustomAttribute<VerbAttribute>() != null).ToArray();

        private static void Run(object options)
        {
            var Attributes = options.GetType().GetCustomAttributes(typeof(CommandRunner), false);
            if (Attributes.Length > 0)
            {
                Type RunnerType = ((CommandRunner)Attributes[0]).Type;
                var RunnerInstance = (BaseCommand)Activator.CreateInstance(RunnerType);
                RunnerInstance.Run(options);
            }
            else
                Console.WriteLine($"Unknown command of type {options.GetType()}, exiting");
        }
    }
}
