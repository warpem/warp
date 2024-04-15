using Asp.Versioning;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using Warp.Tools;

namespace WarpWorker.API
{
    [ApiController]
    [ApiVersion("1.0")]
    [Route("v{version:apiVersion}/[controller]")]
    [Produces("application/json")]
    public class ConsoleController : ControllerBase
    {
        [HttpGet("linecount")]
        public ActionResult<int> GetLineCount()
        {
            return VirtualConsole.LineCount;
        }

        [HttpGet("lines")]
        public ActionResult<List<LogEntry>> GetAllLines()
        {
            return VirtualConsole.GetAllLines();
        }

        [HttpGet("lines/last{count}")]
        public ActionResult<List<LogEntry>> GetLastNLines([FromRoute] int count)
        {
            return VirtualConsole.GetLastNLines(count);
        }

        [HttpGet("lines/first{count}")]
        public ActionResult<List<LogEntry>> GetFirstNLines([FromRoute] int count)
        {
            return VirtualConsole.GetFirstNLines(count);
        }

        [HttpGet("lines/range{start}_{end}")]
        public ActionResult<List<LogEntry>> GetLinesRange([FromRoute] int start, [FromRoute] int end)
        {
            return VirtualConsole.GetLinesRange(start, end);
        }

        [HttpPost("clear")]
        public ActionResult Clear()
        {
            VirtualConsole.ClearAll();
            return Ok();
        }

        [HttpPost("setfileoutput")]
        public ActionResult SetFileOutput([FromBody] string path)
        {
            VirtualConsole.FileOutputPath = path;
            return Ok();
        }

        [HttpPost("writetofile")]
        public ActionResult WriteToFile([FromBody] string path)
        {
            VirtualConsole.WriteToFile(path);
            return Ok();
        }
    }
}
