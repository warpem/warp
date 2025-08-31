using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Newtonsoft.Json;
using WarpCore.Core;

namespace WarpCore
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddControllers()
                .AddNewtonsoftJson(options =>
                {
                    options.SerializerSettings.NullValueHandling = NullValueHandling.Ignore;
                    options.SerializerSettings.DefaultValueHandling = DefaultValueHandling.Include;
                });

            // Register core services
            services.AddSingleton<ProcessingOrchestrator>();
            services.AddSingleton<WorkerPool>();
            services.AddSingleton<ChangeTracker>();
            services.AddSingleton<FileDiscoverer>();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }

            // Eagerly initialize core services to ensure they subscribe to events before workers connect
            var workerPool = app.ApplicationServices.GetRequiredService<WorkerPool>();
            var orchestrator = app.ApplicationServices.GetRequiredService<ProcessingOrchestrator>();
            
            app.UseRouting();
            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}