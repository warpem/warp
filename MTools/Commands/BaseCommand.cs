using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace MTools.Commands
{
    abstract class BaseCommand
    {
        public virtual void Run(object options)
        {
            {
                var Attributes = options.GetType().GetCustomAttributes(typeof(CommandLine.VerbAttribute), false);
                if (Attributes.Length > 0)
                {
                    var Option = (CommandLine.VerbAttribute)Attributes[0];
                    Console.WriteLine($"Running command {Option.Name} with:");
                }
            }

            var Type = options.GetType();

            foreach (var field in Type.GetProperties())
            {
                var Attributes = field.GetCustomAttributes(typeof(CommandLine.OptionAttribute), false);
                if (Attributes.Length > 0)
                {
                    var Option = (CommandLine.OptionAttribute)Attributes[0];
                    Console.WriteLine($"{Option.LongName} = {field.GetValue(options)}");
                }
            }

            Console.WriteLine("");
        }
    }

    class CommandRunner : Attribute
    {
        public Type Type { get; set; }

        public CommandRunner(Type type) 
        { 
            Type = type;
        }
    }
}
