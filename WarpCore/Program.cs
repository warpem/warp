using System;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Warp;
using Warp.Tools;
using WarpCore.Core;
using WarpCore.Controllers;

namespace WarpCore;

/// <summary>
/// Configuration options for WarpCore startup, parsed from command line arguments.
/// Defines the input data directory, processing output directory, and server ports.
/// </summary>
public class StartupOptions
{
    /// <summary>
    /// Directory containing input files to be processed (e.g., TIFF movie files).
    /// </summary>
    [Option('d', "data", Required = true, HelpText = "Data directory containing input files")]
    public string DataDirectory { get; set; }

    /// <summary>
    /// Directory where processing outputs and metadata will be stored.
    /// </summary>
    [Option('p', "processing", Required = true, HelpText = "Processing directory for output files")]
    public string ProcessingDirectory { get; set; }

    /// <summary>
    /// Port number for the REST API server that provides the web interface.
    /// </summary>
    [Option("port", Default = 5000, HelpText = "Port for REST API server")]
    public int Port { get; set; }

    /// <summary>
    /// Port number for the worker controller that manages distributed workers.
    /// When set to 0, the system automatically assigns an available port.
    /// </summary>
    [Option("controller-port", Default = 0, HelpText = "Port for worker controller (0 = auto)")]
    public int ControllerPort { get; set; }
}

/// <summary>
/// Main entry point class for WarpCore - a distributed image processing system
/// for electron microscopy data. Coordinates file discovery, worker management,
/// and processing orchestration through a REST API.
/// </summary>
internal class WarpCore
{
    /// <summary>
    /// Application entry point. Parses command line arguments and starts the web host
    /// that provides the REST API and coordinates the processing system.
    /// </summary>
    /// <param name="args">Command line arguments containing data directory, processing directory, and port settings</param>
    static async Task Main(string[] args)
    {
        VirtualConsole.AttachToConsole();

        await Parser.Default.ParseArguments<StartupOptions>(args)
            .WithParsedAsync(async options =>
            {
                var host = CreateHostBuilder(options).Build();
                await host.RunAsync();
            });
    }

    /// <summary>
    /// Creates and configures the web host builder with the necessary services
    /// and startup configuration for the WarpCore processing system.
    /// </summary>
    /// <param name="options">Startup configuration options parsed from command line</param>
    /// <returns>Configured host builder ready to build and run the application</returns>
    private static IHostBuilder CreateHostBuilder(StartupOptions options) =>
        Host.CreateDefaultBuilder()
            .ConfigureLogging(logging =>
            {
                logging.ClearProviders();
                logging.AddConsole();
                logging.SetMinimumLevel(LogLevel.Warning);
                logging.AddFilter("Microsoft", LogLevel.Warning);
                logging.AddFilter("System", LogLevel.Warning);
                logging.AddFilter("WarpCore", LogLevel.Information);
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
                webBuilder.UseUrls($"http://*:{options.Port}");
            })
            .ConfigureServices(services =>
            {
                services.AddSingleton(options);
            });
}

