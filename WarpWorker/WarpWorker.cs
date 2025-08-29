using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using Warp;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;
using CommandLine;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using WorkerWrapper = Warp.Workers.WorkerWrapper;


namespace WarpWorker
{
    static partial class WarpWorkerProcess
    {
        static bool DebugMode = false;
        static bool IsSilent = false;

        static int DeviceID = 0;

        static bool Terminating = false;

        private static readonly Dictionary<string, MethodInfo> CommandMethods = new Dictionary<string, MethodInfo>();

        static WarpWorkerProcess()
        {
            RegisterCommands();
        }

        private static void RegisterCommands()
        {
            var methods = typeof(WarpWorkerProcess).GetMethods(BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
            
            foreach (var method in methods)
            {
                var commandAttr = method.GetCustomAttribute<CommandAttribute>();
                if (commandAttr != null)
                {
                    CommandMethods[commandAttr.Name] = method;
                }
            }
        }

        static Image GainRef = null;
        static DefectModel DefectMap = null;
        static int2 HeaderlessDims = new int2(2);
        static long HeaderlessOffset = 0;
        static string HeaderlessType = "float32";

        static float[][] RawLayers = null;

        static string OriginalStackOwner = "";
        static Image OriginalStack = null;

        static BoxNetTorch BoxNetModel = null;

        static Population MPAPopulation = null;

        static async Task Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            OptionsCLI OptionsCLI = null;
            Parser.Default.ParseArguments<OptionsCLI>(args).WithParsed(opts => OptionsCLI = opts);

            if (OptionsCLI.DebugAttach && !Debugger.IsAttached)
                Debugger.Launch();

            VirtualConsole.AttachToConsole();

            DeviceID = OptionsCLI.Device % GPU.GetDeviceCount();
            IsSilent = OptionsCLI.Silent;
            DebugMode = OptionsCLI.Debug;

            VirtualConsole.IsSilent = IsSilent;

            GPU.SetDevice(DeviceID);

            // Start in controller mode
            await RunControllerModeAsync(OptionsCLI);
        }


        #region Service

        public static void Exit()
        {
            Terminating = true;
        }

        public static void EvaluateCommand(NamedSerializableObject Command)
        {
            GPU.SetDevice(DeviceID);
            
            // Validate command name
            if (string.IsNullOrWhiteSpace(Command?.Name))
            {
                throw new ArgumentException("Command name cannot be null or empty");
            }
            
            Console.WriteLine($"Received \"{Command.Name}\", with {Command.Content.Length} arguments, for GPU #{GPU.GetDevice()}, {GPU.GetFreeMemory(DeviceID)} MB free:");
            if (DebugMode)
                foreach (var item in Command.Content)
                    Console.WriteLine($"{item.GetType().Name}: {item}");

            try
            {
                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                if (CommandMethods.TryGetValue(Command.Name, out MethodInfo method))
                    method.Invoke(null, new object[] { Command });
                else
                    throw new ArgumentException($"Unknown command: '{Command.Name}'");

                Watch.Stop();
                Console.WriteLine($"Execution took {(Watch.ElapsedMilliseconds / 1000f):F3} seconds");

                Console.WriteLine("");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());

                throw;
            }
        }

        #endregion

        #region Controller Mode

        static async Task RunControllerModeAsync(OptionsCLI options)
        {
            Console.WriteLine($"Starting worker in controller mode, connecting to {options.Controller}");
            Console.WriteLine($"Running on GPU #{DeviceID} ({GPU.GetFreeMemory(DeviceID)} MB free)");
            
            if (DebugMode)
                Console.WriteLine("Debug mode");

            var controllerClient = new ControllerClient(options.Controller, DeviceID, GPU.GetFreeMemory(DeviceID), options.Persistent);
            
            // Handle task execution
            controllerClient.TaskReceived += (command) => 
            {
                try
                {
                    EvaluateCommand(command);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Command execution failed: {ex}");
                    throw; // Re-throw so the controller client can report it as failed
                }
            };

            controllerClient.ErrorOccurred += (error) =>
            {
                Console.WriteLine($"Controller client error: {error}");
                if (!DebugMode)
                {
                    Console.WriteLine("Exiting due to controller error");
                    Process.GetCurrentProcess().Kill();
                }
            };

            // Register with controller
            bool registered = await controllerClient.RegisterAsync();
            if (!registered)
            {
                Console.WriteLine("Failed to register with controller, exiting");
                return;
            }

            // In controller mode, we don't need the heartbeat monitoring
            // since the controller handles worker lifecycle
            Console.WriteLine("Worker registered successfully, starting task processing...");

            // Keep the process alive
            while (true)
            {
                Thread.Sleep(1000);
                
                // Check if we should exit (this could be expanded with proper shutdown signaling)
                if (Terminating)
                {
                    Console.WriteLine("Exiting controller mode");
                    controllerClient.Dispose();
                    break;
                }
            }
        }

        #endregion
    }
}
