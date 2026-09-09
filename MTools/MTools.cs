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

            var Result = Parser.Default.ParseArguments(args, Verbs);
            Result.WithParsed(Run);
            CommandLineParserHelper.SetExitCode(Result);

            // MTools commands report handled validation/domain failures on stderr.
            // Preserve their concise messages while making the failure visible to the OS.
            if (Result.Tag == ParserResultType.Parsed &&
                VirtualConsole.GetAllLines().Any(line => line.Type == LogEntryType.Error))
                CommandLineParserHelper.SetErrorExitCode();
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
                throw new InvalidOperationException($"No command runner is registered for {options.GetType()}.");
        }
    }
}
