using System;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Warp;
using Warp.Tools;
using WarpCore.Core;
using WarpCore.Controllers;

namespace WarpCore;

public class StartupOptions
{
    [Option('d', "data", Required = true, HelpText = "Data directory containing input files")]
    public string DataDirectory { get; set; }

    [Option('p', "processing", Required = true, HelpText = "Processing directory for output files")]
    public string ProcessingDirectory { get; set; }

    [Option("port", Default = 5000, HelpText = "Port for REST API server")]
    public int Port { get; set; }

    [Option("controller-port", Default = 0, HelpText = "Port for worker controller (0 = auto)")]
    public int ControllerPort { get; set; }
}

class WarpCore
{
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

    static IHostBuilder CreateHostBuilder(StartupOptions options) =>
        Host.CreateDefaultBuilder()
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

