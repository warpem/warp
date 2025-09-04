using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Text.Json;
using System.Text.Json.Serialization;
using Warp.Tools;
using WarpCore.Core;
using WarpCore.Core.Processing;

namespace WarpCore;

/// <summary>
/// ASP.NET Core startup class that configures dependency injection and the HTTP pipeline
/// for the WarpCore distributed image processing system. Sets up all core services
/// including orchestration, worker management, file discovery, and processing components.
/// </summary>
public class Startup
{
    /// <summary>
    /// Initializes the startup configuration with the provided application configuration.
    /// </summary>
    /// <param name="configuration">Application configuration from appsettings and other sources</param>
    public Startup(IConfiguration configuration)
    {
        Configuration = configuration;
    }

    /// <summary>
    /// Gets the application configuration used throughout the system.
    /// </summary>
    public IConfiguration Configuration { get; }

    /// <summary>
    /// Configures dependency injection services for the WarpCore system.
    /// Registers all core services including processing orchestration, worker management,
    /// file discovery, and JSON serialization settings for the REST API.
    /// </summary>
    /// <param name="services">Service collection to register dependencies with</param>
    public void ConfigureServices(IServiceCollection services)
    {
        services.AddControllers()
                .AddJsonOptions(options =>
                {
                    options.JsonSerializerOptions.DefaultIgnoreCondition = JsonSettings.Default.DefaultIgnoreCondition;
                    options.JsonSerializerOptions.PropertyNamingPolicy = JsonSettings.Default.PropertyNamingPolicy;
                    options.JsonSerializerOptions.PropertyNameCaseInsensitive = JsonSettings.Default.PropertyNameCaseInsensitive;
                });

        // Register core services
        services.AddSingleton<ProcessingOrchestrator>();
        services.AddSingleton<WorkerPool>();
        services.AddSingleton<ChangeTracker>();
        services.AddSingleton<FileDiscoverer>();
            
        // Register processing components
        services.AddSingleton<ProcessingQueue>();
        services.AddSingleton<ProcessingTaskDistributor>();
        services.AddSingleton<SettingsChangeHandler>();
    }

    /// <summary>
    /// Configures the HTTP request pipeline and initializes core services.
    /// Eagerly initializes the WorkerPool and ProcessingOrchestrator to ensure
    /// event subscriptions are established before any workers attempt to connect.
    /// </summary>
    /// <param name="app">Application builder for configuring the pipeline</param>
    /// <param name="env">Web hosting environment information</param>
    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }

        // Eagerly initialize core services to ensure they subscribe to events before workers connect
        _ = app.ApplicationServices.GetRequiredService<WorkerPool>();
        _ = app.ApplicationServices.GetRequiredService<ProcessingOrchestrator>();
            
        app.UseRouting();
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
    }
}