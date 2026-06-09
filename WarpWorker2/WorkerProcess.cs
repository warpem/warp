using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using Warp;
using Warp.Tools;
using WorkerWrapper = Warp.WorkerWrapper;

namespace WarpWorker2
{
    static partial class WorkerProcess
    {
        static readonly Dictionary<string, MethodInfo> CommandMethods = new();
        static readonly Dictionary<string, MethodInfo> MockCommandMethods = new();

        static int DeviceID = 0;
        static bool MockMode = false;
        static bool DebugMode = false;
        static bool IsSilent = false;

        // Worker resource state (loaded by init commands; survives across tasks).
        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";
        static float[][] RawLayers = null;
        static string OriginalStackOwner = "";
        static Image OriginalStack = null;

        static void RegisterCommands()
        {
            var methods = typeof(WorkerProcess).GetMethods(
                BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            foreach (var method in methods)
            {
                var cmd = method.GetCustomAttribute<CommandAttribute>();
                if (cmd != null) CommandMethods[cmd.Name] = method;
                var mock = method.GetCustomAttribute<MockCommandAttribute>();
                if (mock != null) MockCommandMethods[mock.Name] = method;
            }
        }

        /// <summary>
        /// Execute one command. In mock mode, only MockCommand handlers run — real
        /// GPU commands are skipped entirely so mock mode needs no GPU. Throws on an
        /// unknown (non-mock) command or on handler failure.
        /// </summary>
        static void EvaluateCommand(NamedSerializableObject command)
        {
            GPU.SetDevice(DeviceID);
            if (string.IsNullOrWhiteSpace(command?.Name))
                throw new ArgumentException("Command name cannot be null or empty");

            if (MockMode)
            {
                if (MockCommandMethods.TryGetValue(command.Name, out var mockMethod))
                    mockMethod.Invoke(null, new object[] { command });
                // No mock handler => no-op in mock mode (e.g. init commands like
                // LoadStack do no real GPU work). Intentionally does NOT fall through
                // to the real command.
                return;
            }

            if (CommandMethods.TryGetValue(command.Name, out var method))
                method.Invoke(null, new object[] { command });
            else
                throw new ArgumentException($"Unknown command: '{command.Name}'");
        }

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            RegisterCommands();
            Console.WriteLine($"WarpWorker2 registered {CommandMethods.Count} commands, " +
                              $"{MockCommandMethods.Count} mock commands");
        }
    }
}
