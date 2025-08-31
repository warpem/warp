using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Warp.Workers.WorkerController
{
    public class WorkerControllerHost : IDisposable
    {
        private WebApplication _host;
        private readonly WorkerControllerService _controllerService;
        
        public int Port { get; private set; }
        public bool IsRunning => _host != null;

        public WorkerControllerHost()
        {
            _controllerService = new WorkerControllerService();
        }

        public async Task<int> StartAsync(int port = 0)
        {
            if (_host != null)
                throw new InvalidOperationException("Controller host is already running");

            var builder = WebApplication.CreateBuilder();
            
            // Configure services
            builder.Services.AddSingleton(_controllerService);
            builder.Services.AddControllers()
                .AddApplicationPart(typeof(WorkerControllerAPI).Assembly)
                .AddNewtonsoftJson();
            builder.Services.AddLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Warning);
                logging.AddConsole();
            });

            // Configure Kestrel
            builder.WebHost.UseKestrel(options =>
            {
                options.ListenAnyIP(port);
                options.Limits.MaxRequestBodySize = 1 << 27; // 128 MB
            });

            _host = builder.Build();

            // Configure the HTTP request pipeline
            _host.UseRouting();
            _host.MapControllers();

            await _host.StartAsync();

            // Get the actual port if we used 0 (auto-assign)
            var server = _host.Services.GetService<IServer>();
            if (server != null)
            {
                var addressFeature = server.Features.Get<IServerAddressesFeature>();
                if (addressFeature?.Addresses?.Count > 0)
                {
                    var address = addressFeature.Addresses.First();
                    if (Uri.TryCreate(address, UriKind.Absolute, out var uri))
                    {
                        Port = uri.Port;
                    }
                }
            }

            if (Port == 0)
                Port = port; // Fallback to requested port

            Console.WriteLine($"Worker controller started on port {Port}");
            return Port;
        }

        public async Task StopAsync()
        {
            if (_host != null)
            {
                await _host.StopAsync();
                await _host.DisposeAsync();
                _host = null;
                Console.WriteLine("Worker controller stopped");
            }
        }

        public WorkerControllerService GetService()
        {
            return _controllerService;
        }

        public void Dispose()
        {
            StopAsync().GetAwaiter().GetResult();
            _controllerService?.Dispose();
        }
    }
}