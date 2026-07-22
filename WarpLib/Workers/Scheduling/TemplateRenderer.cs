using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// Fills {{ name }} placeholders in a template from a dictionary. Whitespace inside
    /// the braces is tolerated; replacement is literal (a '$' in a value is never treated
    /// as a regex backreference). Non-placeholder shell syntax such as $(hostname) or $$
    /// is left untouched. After substitution, any remaining {{...}} placeholder is an
    /// error: Render throws listing every unfilled name, rather than silently emptying it.
    /// </summary>
    public static class TemplateRenderer
    {
        // Keep placeholder names shell/config friendly while allowing common dotted and
        // hyphenated names. A separate broad scan below catches malformed/unfilled braces.
        private static readonly Regex Placeholder = new(
            @"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}", RegexOptions.Compiled);
        private static readonly Regex AnyPlaceholder = new(
            @"\{\{\s*([^{}]*?)\s*\}\}", RegexOptions.Compiled);

        public static string Render(string template, IReadOnlyDictionary<string, string> values)
        {
            if (template == null) return null;

            string result = Placeholder.Replace(template, m =>
            {
                string name = m.Groups[1].Value;
                return values != null && values.TryGetValue(name, out string v) ? v : m.Value;
            });

            var unfilled = new List<string>();
            var seen = new HashSet<string>();
            foreach (Match m in AnyPlaceholder.Matches(result))
            {
                string n = m.Groups[1].Value.Trim();
                if (n.Length == 0) n = "(empty)";
                if (seen.Add(n)) unfilled.Add("{{" + n + "}}");
            }
            if (unfilled.Count > 0)
                throw new ArgumentException(
                    "The submission template has unfilled placeholders: " +
                    string.Join(", ", unfilled) +
                    ". Provide them via --cluster_var key=value.");

            return result;
        }
    }
}
