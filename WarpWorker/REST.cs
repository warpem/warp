using System;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi.Models;

namespace WarpWorker;

public class RESTStartup
{
    public void ConfigureServices(IServiceCollection services)
    {
        services.AddControllers().AddNewtonsoftJson();
        services.AddApiVersioning(opt => opt.ReportApiVersions = true);

        services.AddSwaggerGen(c => c.SwaggerDoc("v1", new OpenApiInfo
        {
            Title = "WarpWorker",
            Description = "",
            Version = "v1"
        }));
    }

    public void Configure(IApplicationBuilder app)
    {
        //app.UseMvc();

        app.UseSwagger();
        app.UseSwaggerUI(opt => opt.SwaggerEndpoint("/swagger/v1/swagger.json", "WarpWorker v1"));

        app.UseRouting();

        //app.UseAuthorization();

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllers();
        });
    }
}
