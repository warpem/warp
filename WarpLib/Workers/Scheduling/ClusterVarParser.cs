using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Reassembles the flat token stream from repeated --cluster_var options into a
    /// key=value dictionary. The shell splits on spaces before we see the args, so a
    /// single logical pair may arrive split across up to three tokens. We accept all
    /// four spellings of "key = value" and trim whitespace around both sides:
    ///   key=value | key= value | key =value | key = value
    /// A value that itself contains a space must be quoted into one shell token
    /// (e.g. --cluster_var "account=my project"). A pair with no '=' throws.
    /// </summary>
    public static class ClusterVarParser
    {
        private static readonly Regex ValidKey = new(@"^[A-Za-z0-9_.-]+$", RegexOptions.Compiled);

        public static Dictionary<string, string> Parse(IEnumerable<string> tokens)
        {
            var result = new Dictionary<string, string>();
            var list = tokens?.ToList() ?? new List<string>();

            int i = 0;
            while (i < list.Count)
            {
                string t = list[i];
                int eq = t.IndexOf('=');

                if (eq == 0)
                    throw new ArgumentException($"--cluster_var '{t}' has no key before '='.");

                string key, value;

                if (eq > 0 && eq < t.Length - 1)            // key=value in one token
                {
                    key = t.Substring(0, eq);
                    value = t.Substring(eq + 1);
                    i += 1;
                }
                else if (eq == t.Length - 1 && eq > 0)      // "key=" then value in next token
                {
                    key = t.Substring(0, eq);
                    if (i + 1 >= list.Count)
                        throw new ArgumentException($"--cluster_var '{t}' is missing a value after '='.");
                    value = list[i + 1];
                    i += 2;
                }
                else                                         // t is a bare key; the '=' is in the next token
                {
                    key = t;
                    if (i + 1 >= list.Count)
                        throw new ArgumentException($"--cluster_var '{t}' has no '=' (expected key=value).");
                    string next = list[i + 1];
                    if (next == "=")                         // key = value (three tokens)
                    {
                        if (i + 2 >= list.Count)
                            throw new ArgumentException($"--cluster_var '{t} =' is missing a value.");
                        value = list[i + 2];
                        i += 3;
                    }
                    else if (next.StartsWith("="))           // key =value (two tokens)
                    {
                        value = next.Substring(1);
                        i += 2;
                    }
                    else
                        throw new ArgumentException($"--cluster_var '{t}' has no '=' (expected key=value).");
                }

                key = key.Trim();
                value = value.Trim();
                if (key.Length == 0)
                    throw new ArgumentException("--cluster_var has an empty key.");
                if (!ValidKey.IsMatch(key))
                    throw new ArgumentException(
                        $"--cluster_var key '{key}' contains unsupported characters; " +
                        "use letters, digits, underscore, dot, or hyphen.");

                result[key] = value;
            }

            return result;
        }
    }
}
