using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using MathNet.Numerics.LinearAlgebra;
using ZLinq;

namespace Warp.Tools
{
    public static class CommandLineParserHelper
    {
        public static async Task<ParserResult<object>> ParseAndRun(string[] args, Func<object, Task> run, Type[] verbs, string appName = "")
        {
            if (verbs == null || !verbs.Any())
                throw new ArgumentException("No verbs specified");

            var Parser = new Parser(c => c.HelpWriter = null);
            var Result = Parser.ParseArguments(args, verbs);

            if (Result.Tag == ParserResultType.Parsed)
            {
                if (run != null)
                    await run(Result.Value);
            }
            else
            {
                if (!string.IsNullOrEmpty(appName))
                    Console.WriteLine(appName);

                var Version = (Assembly.GetEntryAssembly() ?? Assembly.GetCallingAssembly()).GetName().Version;
                Console.WriteLine($"Version {Version.Major}.{Version.Minor}.{Version.Build}\n");

                if (verbs != null && verbs.Any() &&
                    Result.Errors.Any(e => e.Tag == ErrorType.NoVerbSelectedError ||
                                           e.Tag == ErrorType.BadVerbSelectedError ||
                                           e.Tag == ErrorType.HelpVerbRequestedError))
                {
                    #region Group verbs

                    Dictionary<string, List<Type>> VerbGroups = new Dictionary<string, List<Type>>();
                    foreach (var verb in verbs)
                    {
                        string Group = "General";
                        if (verb.GetCustomAttribute<VerbGroupAttribute>() != null)
                            Group = verb.GetCustomAttribute<VerbGroupAttribute>().Name;

                        if (!VerbGroups.ContainsKey(Group))
                            VerbGroups.Add(Group, new List<Type>());

                        VerbGroups[Group].Add(verb);
                    }
                    VerbGroups = VerbGroups.OrderBy(v => v.Key == "General" ? "" : v.Key).ToDictionary(v => v.Key, v => v.Value);

                    #endregion

                    if (Result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError))
                        Console.WriteLine("Help requested, showing all available commands:\n");
                    else if (Result.Errors.Any(e => e.Tag == ErrorType.BadVerbSelectedError))
                        Console.WriteLine("Unknown command, showing all available commands:\n");
                    else
                        Console.WriteLine("No command specified, showing all available commands:\n");

                    int MaxNameLength = VerbGroups.SelectMany(v => v.Value).Max(v => v.GetCustomAttribute<VerbAttribute>().Name.Length);

                    foreach (var group in VerbGroups)
                    {
                        int Prefix = (Parser.Settings.MaximumDisplayWidth - group.Key.Length) / 2;
                        int Suffix = Parser.Settings.MaximumDisplayWidth - Prefix - group.Key.Length;
                        Console.WriteLine(new string('-', Prefix) + group.Key + new string('-', Suffix));
                        Console.WriteLine();

                        foreach (var verb in group.Value)
                        {
                            var VerbAttribute = verb.GetCustomAttribute<VerbAttribute>();
                            WriteHelpLine($"{VerbAttribute.Name + new string(' ', MaxNameLength - VerbAttribute.Name.Length)}    ",
                                          VerbAttribute.HelpText,
                                          Parser.Settings.MaximumDisplayWidth);
                            Console.WriteLine();
                        }
                        Console.WriteLine("");
                    }
                }
                else
                {
                    string ParsingErrors = HelpText.RenderParsingErrorsText(Result,
                                                                            SentenceBuilder.Create().FormatError,
                                                                            SentenceBuilder.Create().FormatMutuallyExclusiveSetErrors,
                                                                            0);
                    if (!string.IsNullOrEmpty(ParsingErrors))
                        Console.WriteLine(ParsingErrors + '\n');

                    PrintOptions(Result, 
                                 Result.TypeInfo.Current.GetCustomAttribute<VerbAttribute>().Name, 
                                 Parser.Settings.MaximumDisplayWidth);
                }
            }

            return Result;
        }

        public static async Task<ParserResult<object>> ParseAndRun<T>(string[] args, Func<object, Task> run, string appName = "")
        {
            var Parser = new Parser(c => c.HelpWriter = null);
            var Result = Parser.ParseArguments(args);

            if (Result.Tag == ParserResultType.Parsed)
            {
                if (run != null)
                    await run(Result.Value);
            }
            else
            {
                if (!string.IsNullOrEmpty(appName))
                    Console.WriteLine(appName);

                var Version = (Assembly.GetEntryAssembly() ?? Assembly.GetCallingAssembly()).GetName().Version;
                Console.WriteLine($"Version {Version.Major}.{Version.Minor}.{Version.Build}\n");

                string ParsingErrors = HelpText.RenderParsingErrorsText(Result,
                                                                        SentenceBuilder.Create().FormatError,
                                                                        SentenceBuilder.Create().FormatMutuallyExclusiveSetErrors,
                                                                        0);
                if (!string.IsNullOrEmpty(ParsingErrors))
                    Console.WriteLine(ParsingErrors + '\n');

                PrintOptions(Result, null, Parser.Settings.MaximumDisplayWidth);
            }

            return Result;
        }

        private static void PrintOptions<T>(ParserResult<T> result, string command, int maxDisplayWidth)
        {
            if (string.IsNullOrEmpty(command))
                Console.WriteLine("Showing all available options:\n");
            else
                Console.WriteLine($"Showing all available options for command {command}:\n");

            #region Find options and group them

            var AllOptions = result.TypeInfo.Current.GetProperties().Where(p => p.GetCustomAttribute<OptionAttribute>() != null);
            var AllOptionAttributes = AllOptions.ToDictionary(p => p, p => p.GetCustomAttribute<OptionAttribute>());
            var AllOptionNames = AllOptionAttributes.ToDictionary(p => p.Key, p => string.IsNullOrEmpty(p.Value.ShortName) ? $"--{p.Value.LongName}" : $"-{p.Value.ShortName}, --{p.Value.LongName}");

            int MaxNameLength = AllOptionNames.Values.Max(n => n.Length);

            var OptionGroups = new Dictionary<string, List<PropertyInfo>>();
            var GroupPriorities = new Dictionary<string, int>() { { "", 0 } };
            {
                string CurrentGroup = "";

                foreach (var option in AllOptions)
                {
                    if (option.GetCustomAttribute<OptionGroupAttribute>() != null)
                    {
                        CurrentGroup = option.GetCustomAttribute<OptionGroupAttribute>().Name;
                        if (!GroupPriorities.ContainsKey(CurrentGroup))
                            GroupPriorities.Add(CurrentGroup, option.GetCustomAttribute<OptionGroupAttribute>().Priority);
                    }

                    if (!OptionGroups.ContainsKey(CurrentGroup))
                        OptionGroups.Add(CurrentGroup, new List<PropertyInfo>());

                    OptionGroups[CurrentGroup].Add(option);
                }

                OptionGroups = OptionGroups.OrderBy(g => GroupPriorities[g.Key]).ToDictionary(g => g.Key, g => g.Value);
            }

            #endregion

            #region Print options in groups

            foreach (var group in OptionGroups)
            {
                int Prefix = (maxDisplayWidth - group.Key.Length) / 2;
                int Suffix = maxDisplayWidth - Prefix - group.Key.Length;
                Console.WriteLine(new string('-', Prefix) + group.Key + new string('-', Suffix));
                Console.WriteLine();

                foreach (var property in group.Value)
                {
                    string Required = AllOptionAttributes[property].Required ? "REQUIRED " : "";
                    string Default = AllOptionAttributes[property].Default != null ? $"Default: {AllOptionAttributes[property].Default}. " : "";

                    WriteHelpLine($"{AllOptionNames[property] + new string(' ', MaxNameLength - AllOptionNames[property].Length)}    ",
                                  Required + Default + AllOptionAttributes[property].HelpText,
                                  maxDisplayWidth);
                    Console.WriteLine();
                }
                Console.WriteLine();
            }


            #endregion
        }

        private static void WriteHelpLine(string name, string help, int displayWidth)
        {
            int NameLength = name.Length;
            int HelpLength = help.Length;

            int MaxHelpLength = displayWidth - NameLength;

            Console.Write(name);

            if (HelpLength <= MaxHelpLength)
            {
                Console.WriteLine(help);
            }
            else
            {
                int HelpStart = 0;
                while (HelpStart < HelpLength)
                {
                    int HelpEnd = Math.Min(HelpLength, HelpStart + MaxHelpLength);

                    Console.WriteLine(help.Substring(HelpStart, HelpEnd - HelpStart));
                    HelpStart = HelpEnd;

                    if (HelpStart < HelpLength)
                        Console.Write(new string(' ', NameLength));
                }
            }
        }
    }
    public class VerbGroupAttribute : Attribute
    {
        public string Name { get; set; } = "";

        public VerbGroupAttribute(string name)
        {
            Name = name;
        }
    }

    public class OptionGroupAttribute : Attribute
    {
        public string Name { get; set; } = "";
        public int Priority { get; set; } = 0;

        public OptionGroupAttribute(string name, int priority = 0)
        {
            Name = name;
            Priority = priority;
        }
    }
}
