using System;
using System.IO;

namespace Warp
{
    /// <summary>
    /// Where this installation keeps its network weights: the folder WARP_MODELS names, else models/ beside the
    /// executables. Weights kept outside the build, with WARP_MODELS pointing to them, survive a clean rebuild.
    /// </summary>
    public static class InstalledModels
    {
        public const string Variable = "WARP_MODELS";

        public static string Folder =>
            Environment.GetEnvironmentVariable(Variable) is { Length: > 0 } folder ? folder : Path.Combine(AppContext.BaseDirectory, "models");

        /// <summary>The installed <paramref name="fileName"/>; <paramref name="description"/> names the model if it is missing.</summary>
        public static string Find(string fileName, string description)
        {
            string path = Path.Combine(Folder, fileName);
            if (!File.Exists(path))
                throw new FileNotFoundException($"{description} is not installed: put {fileName} in {Folder}, or set {Variable} to the folder holding it.", path);
            return path;
        }
    }
}
