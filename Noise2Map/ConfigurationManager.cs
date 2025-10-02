using System;
using System.Linq;
using CommandLine;

namespace Noise2Map
{
    /// <summary>
    /// Handles configuration parsing and validation
    /// </summary>
    public static class ConfigurationManager
    {
        /// <summary>
        /// Parses command line arguments and validates the configuration
        /// </summary>
        /// <param name="args">Command line arguments</param>
        /// <param name="options">Parsed options output</param>
        /// <returns>True if parsing and validation succeeded, false otherwise</returns>
        public static bool ParseAndValidate(string[] args, out Options options)
        {
            Options parsedOptions = new Options();

            var result = Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => parsedOptions = opts);

            if (args.Length == 0 ||
                result.Tag == ParserResultType.NotParsed ||
                result.Errors.Any(e => e.Tag == ErrorType.HelpVerbRequestedError ||
                                       e.Tag == ErrorType.HelpRequestedError))
            {
                options = parsedOptions;
                return false;
            }

            ValidateOptions(parsedOptions);
            options = parsedOptions;

            return true;
        }

        /// <summary>
        /// Validates option combinations and constraints
        /// </summary>
        private static void ValidateOptions(Options options)
        {
            // Validate observation/half-map options
            if ((!string.IsNullOrEmpty(options.Observation1Path) || !string.IsNullOrEmpty(options.Observation2Path)) &&
                (!string.IsNullOrEmpty(options.HalfMap1Path) || !string.IsNullOrEmpty(options.HalfMap2Path)))
                throw new ArgumentException("Can't use --observation1/2 and --half1/2 at the same time");

            if (string.IsNullOrEmpty(options.Observation1Path) && string.IsNullOrEmpty(options.HalfMap1Path))
                throw new ArgumentException("You need to specify either two folders with half-maps (--observation1/2) or two single half-maps (--half1/2)");

            if (!string.IsNullOrEmpty(options.Observation1Path) && string.IsNullOrEmpty(options.Observation2Path))
                throw new ArgumentException("When specifying --observation1, you also need to specify --observation2");

            if (!string.IsNullOrEmpty(options.HalfMap1Path) && string.IsNullOrEmpty(options.HalfMap2Path))
                throw new ArgumentException("When specifying --half1, you also need to specify --half2");

            // Validate augmentation and CTF options
            if (!options.DontAugment && !string.IsNullOrEmpty(options.CTFPath))
                throw new ArgumentException("3D CTF cannot be combined with data augmentation.");

            // Validate batch size
            if (options.BatchSize < 1)
                throw new Exception("Batch size must be at least 1.");
        }
    }
}
