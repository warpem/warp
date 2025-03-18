using Asp.Versioning;
using Microsoft.AspNetCore.Mvc;
using System;
using System.IO;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Warp;
using Warp.Tools;

namespace WarpWorker.API
{
    [ApiController]
    [ApiVersion("1.0")]
    [Route("v{version:apiVersion}/[controller]")]
    [Produces("application/json")]
    public class ServiceController : Controller
    {
        [HttpPost]
        [Route("SendPulse")]
        public IActionResult SendPulse()
        {
            WarpWorker.SendPulse();
            return Ok();
        }

        [HttpPost]
        [Route("Exit")]
        public IActionResult Exit()
        {
            Console.WriteLine("Received exit command");
            WarpWorker.Exit();
            return Ok();
        }

        [HttpPost]
        [Route("EvaluateCommand")]
        public IActionResult EvaluateCommand()
        {
            using StreamReader reader = new StreamReader(Request.Body, Encoding.UTF8);
            string jsonString = reader.ReadToEndAsync().GetAwaiter().GetResult();

            NamedSerializableObject Command = JsonSerializer.Deserialize<NamedSerializableObject>(jsonString);

            try
            {
                WarpWorker.EvaluateCommand(Command);
            }
            catch (Exception ex) 
            { 
                return Problem(ex.ToString(), 
                               null, 
                               StatusCodes.Status500InternalServerError, 
                               "Command execution failed", 
                               "Processing error");
            }

            return Ok();
        }
    }
}