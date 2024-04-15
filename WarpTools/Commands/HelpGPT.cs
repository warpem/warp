using CommandLine;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;
using OpenAI_API;
using System.Reflection;
using OpenAI_API.Chat;
using System.Collections.Specialized;
using System.Text.RegularExpressions;

namespace WarpTools.Commands.Bal
{
    [VerbGroup("Help")]
    [Verb("helpgpt", HelpText = "Get help from ChatGPT; requires an OpenAI API key stored in ~/openai.key")]
    [CommandRunner(typeof(HelpGpt))]
    class HelpGptOptions
    {
        [Option("key", HelpText = "OpenAI key to be saved to ~/openai.key")]
        public string Key { get; set; }

        [Value(0)]
        public IEnumerable<string> Request { get; set; }
    }

    class HelpGpt : BaseCommand
    {
        public override async Task Run(object options)
        {
            await base.Run(options);
            HelpGptOptions CLI = options as HelpGptOptions;

            #region Key storage and retrieval

            string ApiKeyPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "openai.key");
            string ApiKey = null;

            if (!string.IsNullOrEmpty(CLI.Key))
            {
                File.WriteAllText(ApiKeyPath, CLI.Key);
                Console.WriteLine($"Saved {CLI.Key} to ~/openai.key\n");
            }

            if (File.Exists(ApiKeyPath))
            {
                IEnumerable<string> Lines = File.ReadLines(ApiKeyPath);
                if (Lines.Count() == 0 || Lines.First().Trim().Length != 51)
                {
                    Console.WriteLine("~/openai.key exists, but the key doesn't have the right length; exiting");
                    return;
                }

                ApiKey = Lines.First().Trim();
            }
            else
            {
                Console.WriteLine("No OpenAI key found in ~/openai.key, exiting");
                return;
            }

            #endregion

            #region API setup

            var Api = new OpenAIAPI(ApiKey);

            Api.Chat.DefaultChatRequestArgs.Model = OpenAI_API.Models.Model.GPT4_Turbo;
            Api.Chat.DefaultChatRequestArgs.Temperature = 0;
            Api.Chat.DefaultChatRequestArgs.TopP = 1;
            Api.Chat.DefaultChatRequestArgs.FrequencyPenalty = 0;
            Api.Chat.DefaultChatRequestArgs.PresencePenalty = 0;
            Api.Chat.DefaultChatRequestArgs.MaxTokens = 1024;

            #endregion

            #region Create system message

            string CommandString = null;
            {
                StringBuilder Builder = new StringBuilder();
                Type[] Verbs = Assembly.GetExecutingAssembly().GetTypes().Where(t => t.GetCustomAttribute<VerbAttribute>() != null).OrderBy(t => t.Name).ToArray();

                foreach (var verb in Verbs)
                    Builder.AppendLine(GetHelpString(verb, false));

                CommandString = Builder.ToString();
            }

            var Chat = Api.Chat.CreateConversation(new ChatRequest() { });

            StringBuilder SystemMessage = new StringBuilder();

            SystemMessage.AppendLine("Here is the full list of available commands:\n" +
                                     CommandString + 
                                     "\n");

            SystemMessage.AppendLine("The options for each command have been omitted for brevity. If you need to use a command in your response, " +
                                     "you MUST reply with ONLY {{{command_name}}} (replacing the command name, keeping the brackets), and the next user message will contain all options with " +
                                     "their descriptions. This part of the interaction will be hidden from the actual user. DO NOT ASSUME you know anything " +
                                     "about a command without requesting more information about its options from the system first.\n\n");

            SystemMessage.AppendLine("You are a chat assistant integrated in WarpTools, a command-line application for the pre-processing of cryo-EM data. " +
                                     "You have knowledge of the full list of commands and their options. Your goal is to assist the user by providing sequences " +
                                     "of commands that you think would be helpful to achieve their task, or further explanation for commands or options the user " +
                                     "doesn't understand. Sometimes a task may require running a bash script to do something that WarpTools wasn't made for. In that " +
                                     "case, write the bash script. SUPER IMPORTANT>>>Sometimes the user may not give you all required information to complete the task. If you would like " +
                                     "the user to provide more information, end your response with a question ending with a ?. This will allow the user to provide more information. " +
                                     "Work with the user step by step if their command is a call to action rather than a question. Don't just tell the user to replace a placeholder in your" +
                                     "response, but work with the user to build the correct commands<<<SUPER IMPORTANT " +
                                     "Your response will be printed in a Unix terminal with no advanced formatting capability. Don't enclose anything in ` or ``` code blocks. " +
                                     "Also don't write command examples as ```bash code blocks\n");

            SystemMessage.AppendLine("WarpTools can be used to process frame series (commonly called movies, but we prefer frame series!!!), and tilt series. Most processing commands require a " +
                                     ".settings file with information about where to look for data, and where to put the results. This is created by the create_settings " +
                                     "command. We usually separate data and processing results in different folders. For tilt series we make 2 separate settings files: " +
                                     "one frameseries.settings for the frame series comprising each tilt, and one tiltseries.settings for the .tomostar files that will " +
                                     "be created later. tiltseries.settings must also include the unbinned tomogram dimensions.\n");

            SystemMessage.AppendLine("Frame series processing usually includes aligning the frames, estimating the contrast transfer function (CTF), and picking particles using the BoxNet ML model. " +
                                     "Tilt series processing is more involved. It starts by processing the frame series that represent each of the tilts. The signal in " +
                                     "these movies is much lower than usual, so only alignment and CTF estimation are done. Tilt series are defined in separate .tomostar " +
                                     "files, which are considered the raw data for tilt series. Then the aligned averages are aligned in the context of the whole tilt series. " +
                                     "When using ts_aretomo, no explicit stacking step is needed, but tilt stacks may be generated for other workflows. After alignment, CTF " +
                                     "is estimated and the CTF handedness can be checked. This is followed by full tomogram reconstruction, usually at a large pixel size like " +
                                     "10 Angstrom. Template matching is performed to find particles. Once particle positions are available, particles are reconstructed either as 3D " +
                                     "sub-tomograms, or as 2D particle tilt series (the latter requires RELION 5 for further processing, while sub-tomograms work with RELION 3 or 4).\n");

            SystemMessage.AppendLine("Many commands run on GPUs. By default, one worker per GPU is spawned. For GPUs with more memory, more workers per device can be created, except " +
                                     "in ts_aretomo. Advanced users may run WarpWorker on remote hosts and supply the host name and port for each of them to a WarpTools command.\n");

            SystemMessage.AppendLine("If you don't know something, you are fully allowed to respond with \"I don't know\"");

            SystemMessage.AppendLine("[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n" +
                                     "[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n" +
                                     "[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n");

            //SystemMessage.AppendLine("Reminder for Assistant: Before providing detailed command usage examples or explanations, always verify the command's options by requesting more " +
            //                         "information using the format {{{command_name}}}. This ensures accuracy and relevance in the response.");

            string Bla = SystemMessage.ToString();
            Chat.AppendSystemMessage(SystemMessage.ToString());

            Chat.AppendExampleChatbotOutput("Here is an example of a user interaction:\n" +
                                             "User says: \"I'd like to estimate the CTF in frame series\"\n" +
                                             "You think: \"Looks like fs_ctf is the right command, but I don't know anything about its options. I need to ask for the full list\"\n" +
                                             "You say: \"{{{fs_ctf}}}\"\n" +
                                             "You receive hint:\n" +
                                             "<command>\n" +
                                             "Name: fs_ctf\n" +
                                             "Category: Frame series\n" +
                                             "Description: Estimate CTF parameters in frame series\n" +
                                             "Options:\n" +
                                             "<option_group name=\"Data import settings\">\n" +
                                             "--settings, required, Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.\n" +
                                             "</option_group>\n" +
                                             "<option_group name=\"\">\n" +
                                             "--window, default: 512, Patch size for CTF estimation in binned pixels\n" +
                                             "[more options would follow in a real example]\n" +
                                             "You say: \"The command you're looking for is fs_ctf. Here is an example of using it: [followed by generic example]\"\n" +
                                             "You think: \"The user hasn't provided some necessary parameters for composing the full command. I need to ask for them.\"\n" +
                                             "You say: \"What is the settings file, window size, frequency range, [insert more missing parameters]?\"\n" +
                                             "User says: \"The settings are in ../frameseries.settings, window size 512, estimated between 30 and 5 Angstrom [and so on]\"\n" +
                                             "You say: \"fs_ctf --settings ../frameseries.settings --window 512 --range_min 30 --range_max 5 [and so on]\"\n" +
                                             "User says: \"What would be the next command to use after that to pre-process the data?\"\n" +
                                             "You think: \"fs_align and fs_align_and_ctf both could work, but I don't know about them. I will first get more information, and " +
                                             "if it's still not clear then, I will ask the user which one they want to use.\"" +
                                             "You say: \"{{{fs_align}}} {{{fs_align_and_ctf}}}\"\n" +
                                             "You receive hint: [hint for fs_align]\n" +
                                             "You receive hint: [hint for fs_align_and_ctf]\n" +
                                             "You think: \"They should both work, but I still don't know which one the user wants to use. I will ask them.\"\n" +
                                             "You say: \"After creating the `frameseries.settings` file, the next step in pre-processing the frame series data typically involves " +
                                             "correcting motion in the frame series and estimating the contrast transfer function (CTF). In WarpTools, you can do both steps in one " +
                                             "go using the `fs_align_and_ctf` command, or only perform alignment using the `fs_align` command. Which one do you prefer?\"\n" +
                                             "User says: \"Let's just align for now\"\n" +
                                             "[You proceed to give help for fs_align]\n" +
                                             "[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n" +
                                             "[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n" +
                                             "[!!YOU NEVER EVER ASSUME YOU CAN GUESS A COMMAND'S PARAMETERS WITHOUT HAVING RECEIVED AN EXPLICIT HINT FOR IT!!]\n");

            #endregion

            string Request = string.Join(" ", CLI.Request);
            Chat.AppendUserInput(Request);

            //Chat.AppendUserInput("Reminder for Assistant: Before providing detailed command usage examples or explanations, always verify the command's options by requesting more " +
            //                     "information using the format {{{command_name}}}. This ensures accuracy and relevance in the response.");

            Console.WriteLine($"You asked: {Request}\n");

            await InteractionLoop(Chat);            

            Console.WriteLine();
        }

        async Task InteractionLoop(Conversation chat)
        {
            while (true)
            {
                IEnumerable<string> Commands = await StreamAndParse(chat);
                if (!Commands.Any())
                {
                    string UserInput = ReadLine.Read("\n\nREPLY (leave empty to end): ");

                    if (string.IsNullOrEmpty(UserInput))
                        break;
                    else
                    {
                        chat.AppendUserInput(UserInput);
                        Console.WriteLine();
                    }
                }
                else
                    foreach (var command in Commands)
                        RespondToCommand(chat, command);
            }
        }

        void RespondToCommand(Conversation chat, string commandName)
        {
            Type Verb = Assembly.GetExecutingAssembly().GetTypes().Where(t => t.GetCustomAttribute<VerbAttribute>() != null).FirstOrDefault(t => t.GetCustomAttribute<VerbAttribute>().Name == commandName);
            if (Verb == null)
                throw new Exception($"ChatGPT requested help for command \"{commandName}\"but it doesn't exist");

            chat.AppendUserInput(GetHelpString(Verb, true));
            Console.WriteLine($"[Added help for {commandName}]\n");
        }

        async Task<IEnumerable<string>> StreamAndParse(Conversation chat)
        {
            StringBuilder FullResponse = new StringBuilder();
            StringBuilder ResponseBuffer = new StringBuilder();
            bool ShouldPrint = true;

            await chat.StreamResponseFromChatbotAsync(res =>
            {
                ResponseBuffer.Append(res);
                FullResponse.Append(res);

                if (FullResponse.Length == 0 && ResponseBuffer.ToString().Contains("{{{"))
                    ShouldPrint = false;

                string FullCurrent = ResponseBuffer.ToString();
                FullCurrent = FullCurrent
                    .Replace("```bash\n", "")
                    .Replace("```\n", "");
                //.Replace("```", "")
                //.Replace("`", "");

                foreach (Match match in Regex.Matches(FullCurrent, @"\{{3}([a-zA-Z0-9_]+)\}{3}"))
                    FullCurrent = FullCurrent.Replace(match.Value, "");

                foreach (Match match in Regex.Matches(FullCurrent, "`([^`]+)`"))
                    FullCurrent = FullCurrent.Replace(match.Value, match.Value.Replace("`", ""));

                ResponseBuffer = new StringBuilder(FullCurrent);

                if (ResponseBuffer.Length > 64 && ShouldPrint)
                {
                    Console.Write(FullCurrent.Substring(0, FullCurrent.Length - 56));

                    ResponseBuffer = new StringBuilder(FullCurrent.Substring(FullCurrent.Length - 56));
                }
            });

            if (ShouldPrint)
            {
                string Full = FullResponse.ToString();
                Console.Write(ResponseBuffer.ToString());
                Console.WriteLine();
            }

            {
                string Response = FullResponse.ToString();
                string Pattern = @"\{{3}([a-zA-Z0-9_]+)\}{3}";

                MatchCollection Matches = Regex.Matches(Response, Pattern);

                foreach (Match match in Matches)
                    Console.WriteLine($"[Silently: {match.Groups[1].Value}]\n");

                return Matches.Select(m => m.Groups[1].Value);
            }
        }

        string GetHelpString(Type verb, bool includeOptions)
        {
            StringBuilder Builder = new StringBuilder();

            var Info = verb.GetCustomAttribute<VerbAttribute>();
            var Category = verb.GetCustomAttribute<VerbGroupAttribute>();

            Builder.AppendLine("<command>");
            Builder.AppendLine($"Name: {Info.Name}");
            Builder.AppendLine($"Category: {(Category == null ? "General" : Category.Name)}");
            Builder.AppendLine($"Description: {Info.HelpText}");

            if (includeOptions)
            {
                Builder.AppendLine("Options:");

                #region Group options

                var AllOptions = verb.GetProperties().Where(p => p.GetCustomAttribute<OptionAttribute>() != null);
                var AllOptionAttributes = AllOptions.ToDictionary(p => p, p => p.GetCustomAttribute<OptionAttribute>());
                var AllOptionNames = AllOptionAttributes.ToDictionary(p => p.Key, p => string.IsNullOrEmpty(p.Value.ShortName) ? $"--{p.Value.LongName}" : $"-{p.Value.ShortName}, --{p.Value.LongName}");

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
                    Builder.AppendLine($"<option_group name=\"{group.Key}\">");

                    foreach (var property in group.Value)
                    {
                        string Required = AllOptionAttributes[property].Required ? "required, " : "";
                        string Default = AllOptionAttributes[property].Default != null ? $"default: {AllOptionAttributes[property].Default}, " : "";

                        Builder.AppendLine($"{AllOptionNames[property]}, " + Required + Default + AllOptionAttributes[property].HelpText);
                    }

                    Builder.AppendLine($"</option_group>");
                }

                #endregion
            }

            Builder.AppendLine("</command>");

            return Builder.ToString();
        }
    }
}
