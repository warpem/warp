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
        static bool MockMode = false;

        static int DeviceID = 0;

        static bool Terminating = false;

        private static readonly Dictionary<string, MethodInfo> CommandMethods = new();
        private static readonly Dictionary<string, MethodInfo> MockCommandMethods = new();

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
                    CommandMethods[commandAttr.Name] = method;
                
                var mockCommandAttr = method.GetCustomAttribute<MockCommandAttribute>();
                if (mockCommandAttr != null)
                    MockCommandMethods[mockCommandAttr.Name] = method;
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
            MockMode = OptionsCLI.Mock;

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

                // Handle mock mode
                if (MockMode && Command.Name != "Exit")
                {
                    if (MockCommandMethods.TryGetValue(Command.Name, out MethodInfo method))
                        method.Invoke(null, new object[] { Command });
                    else
                        // Simulate processing time with a short delay
                        Thread.Sleep(100 + new Random().Next(400)); // 100-500ms random delay
                    
                    Console.WriteLine($"[MOCK] Command '{Command.Name}' completed successfully");
                }
                else
                {
                    // Execute real command
                    if (CommandMethods.TryGetValue(Command.Name, out MethodInfo method))
                        method.Invoke(null, new object[] { Command });
                    else
                        throw new ArgumentException($"Unknown command: '{Command.Name}'");
                }

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
            for (int i = 0; i < 10; i++)
                Console.WriteLine("HELLO");
            Console.WriteLine($"Starting worker in controller mode, connecting to {options.Controller}");
            Console.WriteLine($"Running on GPU #{DeviceID} ({GPU.GetFreeMemory(DeviceID)} MB free)");
            
            if (DebugMode)
                Console.WriteLine("Debug mode");
            if (MockMode)
                Console.WriteLine("Mock mode enabled");

            var controllerClient = new ControllerClient(options.Controller, DeviceID, GPU.GetFreeMemory(DeviceID), options.Persistent);
            
            // Handle work package execution (new system)
            controllerClient.WorkPackageReceived += (workPackage) =>
            {
                try
                {
                    ExecuteWorkPackage(workPackage, controllerClient);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Work package execution failed: {ex}");
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

        static void ExecuteWorkPackage(Warp.Workers.Distribution.WorkPackage workPackage, ControllerClient controllerClient)
        {
            Console.WriteLine($"Executing work package {workPackage.Id} with {workPackage.Commands.Count} commands");
            
            // Report package as started
            controllerClient.UpdateWorkPackageStatus(workPackage.Id, Warp.Workers.Distribution.WorkPackageStatus.Executing, 0);
            
            for (int i = 0; i < workPackage.Commands.Count; i++)
            {
                var command = workPackage.Commands[i];
                try
                {
                    Console.WriteLine($"Executing command {i + 1}/{workPackage.Commands.Count}: {command.Name}");
                    
                    // Update progress to current command
                    controllerClient.UpdateWorkPackageStatus(workPackage.Id, Warp.Workers.Distribution.WorkPackageStatus.Executing, i);
                    
                    // Execute the command
                    EvaluateCommand(command);
                    
                    Console.WriteLine($"Command {command.Name} completed successfully");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Command {command.Name} failed: {ex.Message}");
                    
                    // Report package as failed
                    controllerClient.UpdateWorkPackageStatus(workPackage.Id, Warp.Workers.Distribution.WorkPackageStatus.Failed, i, ex.Message);
                    throw; // Re-throw to stop package execution
                }
            }
            
            // All commands completed successfully
            Console.WriteLine($"Work package {workPackage.Id} completed successfully");
            controllerClient.UpdateWorkPackageStatus(workPackage.Id, Warp.Workers.Distribution.WorkPackageStatus.Completed, workPackage.Commands.Count);
        }

        #endregion
    }
}
