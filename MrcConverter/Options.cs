using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace MrcConverter
{
    class Options
    {
        [Option('r', "recursive", HelpText = "Search recursively")]
        public bool Recursive { get; set; }

        [Option("reverse", HelpText = "Convert from float16 to float32 instead")]
        public bool Reverse { get; set; }

        [Option('p', "pattern", Default = "*.mrc", HelpText = "File name pattern")]
        public string Pattern { get; set; }

        [Option('e', "exclude", Default = new string[0], HelpText = "Exclude files that contain any of these (space-separated) words in their path")]
        public IEnumerable<string> Exclude { get; set; }

        [Option('q', "quick", HelpText = "Skip entire folders if the last file in them has already been converted")]
        public bool Quick { get; set; }

        [Option('s', "simulate", HelpText = "Just calculate the space savings without converting anything")]
        public bool Simulate { get; set; }

        [Option('j', "threads", Default = 1, HelpText = "Number of threads for concurrent file handling")]
        public int Threads { get; set; }

        [Option('m', "memgb", Default = 0, HelpText = "Maximum amount of data in GB that should be kept in memory at any given point by all threads; 0 = no limit")]
        public int MemGB { get; set; }
    }
}
