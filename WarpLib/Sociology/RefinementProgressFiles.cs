using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Warp.Sociology
{
    /// <summary>Checks published refinement checkpoints without loading GPU data.</summary>
    public static class RefinementProgressFiles
    {
        public static string[] GetCompleteFolders(string[] folders, string speciesID)
        {
            var complete = new List<string>();
            foreach (string folder in folders)
            {
                // Scratch and .tmp files are not published results. Check all species:
                // a folder published for only another species is an incomplete save.
                bool published = Directory.EnumerateFiles(folder, "*_half1_*.mrc").Any() ||
                                 Directory.EnumerateFiles(folder, "*_half2_*.mrc").Any() ||
                                 Directory.EnumerateFiles(folder, "*_particles.star").Any();
                if (!published)
                {
                    Console.WriteLine($"Skipping refinement folder with no published results: {folder}");
                    continue;
                }

                string star = Path.Combine(folder, $"{speciesID}_particles.star");
                string[] half1 = Directory.GetFiles(folder, $"{speciesID}_half1_*.mrc");
                string[] half2 = Directory.GetFiles(folder, $"{speciesID}_half2_*.mrc");
                string prefix1 = $"{speciesID}_half1_";
                string prefix2 = $"{speciesID}_half2_";
                var indices1 = new HashSet<string>();
                foreach (string path in half1) indices1.Add(Path.GetFileName(path).Substring(prefix1.Length));
                var indices2 = new HashSet<string>();
                foreach (string path in half2) indices2.Add(Path.GetFileName(path).Substring(prefix2.Length));

                if (!File.Exists(star) || half1.Length == 0 || !indices1.SetEquals(indices2))
                    throw new InvalidDataException($"Incomplete refinement checkpoint in '{folder}' for species {speciesID}: " +
                        "expected a particle STAR and matching half1/half2 MRC files. No maps have been gathered for this species.");

                var publishedPaths = new List<string>(half1);
                publishedPaths.AddRange(half2);
                publishedPaths.Add(star);
                foreach (string path in publishedPaths)
                    if (new FileInfo(path).Length == 0)
                        throw new InvalidDataException($"Empty refinement checkpoint file '{path}'. No maps have been gathered for this species.");

                complete.Add(folder);
            }

            if (complete.Count == 0)
                throw new InvalidDataException($"No complete refinement checkpoints found for species {speciesID}.");

            return complete.ToArray();
        }
    }
}
