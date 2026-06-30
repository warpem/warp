# WarpTools Cluster Pool Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained cluster provisioning mode to WarpTools so a user can submit a pool of `WarpWorker2` workers to a batch scheduler (SLURM/LSF/PBS/SGE/anything) that claim tasks from the shared filesystem queue — without Relay.

**Architecture:** A new `ClusterProvisioner : IWorkerProvisioner` (in WarpLib) submits `--pool_size` identical worker jobs once via a user-supplied submission-script template (`{{ }}` placeholders) and a 3-field queue-definition JSON (submit / cancel / job-id regex), then cancels them all on shutdown (drain, signals, heartbeat net). `DistributedOptions` (in WarpTools) gains four CLI options and a single `CreateProvisioner` helper that selects between local / external / cluster modes.

**Tech Stack:** C# / .NET 10, `System.Text.Json`, `System.Text.RegularExpressions`, `System.Runtime.InteropServices.PosixSignalRegistration`, xunit 2.9 for tests.

## Global Constraints

- All new reusable types live in `WarpLib/Workers/Scheduling/` under namespace `Warp.Workers.Scheduling`. The `Tests` project references **WarpLib** (and WarpWorker2's executable) but **NOT WarpTools**, so all unit-testable logic must live in WarpLib. WarpTools wiring is verified by build + the WarpLib tests behind it.
- Worker command device is always `0` (one cluster job = one GPU). Worker id is `"$(hostname)-$$"` (shell-expanded on the compute node) — never our default `local-{pid}-gpu{dev}` which collides across nodes.
- Workers run `--persistent`; the pool must always be cancelled on shutdown so jobs never linger.
- Unfilled `{{placeholder}}` after rendering is a hard error that lists the unfilled names (NOT silently emptied like Relay).
- `--cluster_var key=value` must be lenient about whitespace around `=`.
- Git commit trailer MUST be exactly: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Tests run with: `dotnet test Tests/Tests.csproj --filter "<FilterExpr>"`. The full suite is `dotnet test Tests/Tests.csproj`.
- Reference spec: `docs/superpowers/specs/2026-06-28-warptools-cluster-provisioning-design.md`.

---

## File Structure

**Create (WarpLib):**
- `WarpLib/Workers/Scheduling/ClusterVarParser.cs` — static `--cluster_var` token reassembler.
- `WarpLib/Workers/Scheduling/TemplateRenderer.cs` — static `{{ }}` substitution + unfilled-placeholder crash.
- `WarpLib/Workers/Scheduling/ClusterQueueDefinition.cs` — model + JSON loader for the 3-field config.
- `WarpLib/Workers/Scheduling/ClusterProvisioner.cs` — `ShellRunner` delegate + default shell runner + `ClusterProvisioner : IWorkerProvisioner` + `Create` factory.

**Create (Tests):**
- `Tests/Workers/ClusterVarParserTests.cs`
- `Tests/Workers/TemplateRendererTests.cs`
- `Tests/Workers/ClusterQueueDefinitionTests.cs`
- `Tests/Workers/ClusterProvisionerTests.cs`

**Modify (WarpTools):**
- `WarpTools/Commands/DistributedOptions.cs` — add 4 options; add `CreateProvisioner` helper; replace the two duplicated provisioner-selection blocks (lines ~101-125 and ~273-291) with calls to it.

---

## Task 1: ClusterVarParser

**Files:**
- Create: `WarpLib/Workers/Scheduling/ClusterVarParser.cs`
- Test: `Tests/Workers/ClusterVarParserTests.cs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Warp.Workers.Scheduling.ClusterVarParser.Parse(IEnumerable<string> tokens) -> Dictionary<string,string>`. Throws `ArgumentException` on a token group with no `=` or an empty/missing key.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Workers/ClusterVarParserTests.cs`:

```csharp
using System.Collections.Generic;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class ClusterVarParserTests
{
    [Fact]
    public void OneToken_KeyEqualsValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void TwoTokens_KeyEquals_ThenValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=", "gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void TwoTokens_Key_ThenEqualsValue()
    {
        var r = ClusterVarParser.Parse(new[] { "partition", "=gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void ThreeTokens_Key_Equals_Value()
    {
        var r = ClusterVarParser.Parse(new[] { "partition", "=", "gpu" });
        Assert.Equal("gpu", r["partition"]);
    }

    [Fact]
    public void MultiplePairs_MixedSpacing()
    {
        var r = ClusterVarParser.Parse(new[] { "partition=gpu", "walltime", "=", "04:00:00", "mem=", "16G" });
        Assert.Equal("gpu", r["partition"]);
        Assert.Equal("04:00:00", r["walltime"]);
        Assert.Equal("16G", r["mem"]);
    }

    [Fact]
    public void QuotedValueWithSpace_IsOneToken()
    {
        var r = ClusterVarParser.Parse(new[] { "account=my project" });
        Assert.Equal("my project", r["account"]);
    }

    [Fact]
    public void TrimsWhitespaceAroundEquals_WithinSingleToken()
    {
        var r = ClusterVarParser.Parse(new[] { "a = 1" });
        Assert.Equal("1", r["a"]);
    }

    [Fact]
    public void NullInput_ReturnsEmpty()
    {
        var r = ClusterVarParser.Parse(null);
        Assert.Empty(r);
    }

    [Fact]
    public void KeyWithNoEquals_Throws()
    {
        Assert.Throws<System.ArgumentException>(() => ClusterVarParser.Parse(new[] { "partition" }));
    }

    [Fact]
    public void LeadingEquals_Throws()
    {
        Assert.Throws<System.ArgumentException>(() => ClusterVarParser.Parse(new[] { "=gpu" }));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterVarParserTests"`
Expected: FAIL to compile / "ClusterVarParser does not exist".

- [ ] **Step 3: Write the implementation**

Create `WarpLib/Workers/Scheduling/ClusterVarParser.cs`:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

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

                result[key] = value;
            }

            return result;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterVarParserTests"`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Scheduling/ClusterVarParser.cs Tests/Workers/ClusterVarParserTests.cs
git commit -m "feat: ClusterVarParser for lenient --cluster_var key=value parsing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: TemplateRenderer

**Files:**
- Create: `WarpLib/Workers/Scheduling/TemplateRenderer.cs`
- Test: `Tests/Workers/TemplateRendererTests.cs`

**Interfaces:**
- Consumes: nothing.
- Produces: `Warp.Workers.Scheduling.TemplateRenderer.Render(string template, IReadOnlyDictionary<string,string> values) -> string`. Substitutes `{{ name }}` (whitespace-tolerant, literal value). Throws `ArgumentException` listing every distinct unfilled placeholder name remaining after substitution.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Workers/TemplateRendererTests.cs`:

```csharp
using System.Collections.Generic;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class TemplateRendererTests
{
    [Fact]
    public void SubstitutesSimplePlaceholder()
    {
        var r = TemplateRenderer.Render("a {{x}} b", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("a 1 b", r);
    }

    [Fact]
    public void ToleratesWhitespaceInsidePlaceholder()
    {
        var r = TemplateRenderer.Render("[{{  x  }}]", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("[1]", r);
    }

    [Fact]
    public void ReplacesAllOccurrences()
    {
        var r = TemplateRenderer.Render("{{x}}-{{x}}", new Dictionary<string, string> { ["x"] = "z" });
        Assert.Equal("z-z", r);
    }

    [Fact]
    public void ValueWithDollarIsLiteral_NotRegexBackref()
    {
        var r = TemplateRenderer.Render("{{x}}", new Dictionary<string, string> { ["x"] = "$1foo$$" });
        Assert.Equal("$1foo$$", r);
    }

    [Fact]
    public void LeavesNonPlaceholderShellSyntaxAlone()
    {
        var r = TemplateRenderer.Render("id=$(hostname)-$$ {{x}}", new Dictionary<string, string> { ["x"] = "1" });
        Assert.Equal("id=$(hostname)-$$ 1", r);
    }

    [Fact]
    public void UnfilledPlaceholder_ThrowsListingName()
    {
        var ex = Assert.Throws<System.ArgumentException>(() =>
            TemplateRenderer.Render("{{x}} {{y}}", new Dictionary<string, string> { ["x"] = "1" }));
        Assert.Contains("y", ex.Message);
    }

    [Fact]
    public void MultipleUnfilledPlaceholders_AllListed()
    {
        var ex = Assert.Throws<System.ArgumentException>(() =>
            TemplateRenderer.Render("{{a}} {{b}}", new Dictionary<string, string>()));
        Assert.Contains("a", ex.Message);
        Assert.Contains("b", ex.Message);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~TemplateRendererTests"`
Expected: FAIL — "TemplateRenderer does not exist".

- [ ] **Step 3: Write the implementation**

Create `WarpLib/Workers/Scheduling/TemplateRenderer.cs`:

```csharp
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
        // A placeholder name is one or more word characters (letters, digits, underscore).
        private static readonly Regex Placeholder = new(@"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", RegexOptions.Compiled);

        public static string Render(string template, IReadOnlyDictionary<string, string> values)
        {
            if (template == null) return null;

            string result = Placeholder.Replace(template, m =>
            {
                string name = m.Groups[1].Value;
                return values != null && values.TryGetValue(name, out string v) ? v : m.Value;
            });

            var unfilled = Placeholder.Matches(result)
                                      .Select(m => m.Groups[1].Value)
                                      .Distinct()
                                      .ToList();
            if (unfilled.Count > 0)
                throw new ArgumentException(
                    "The submission template has unfilled placeholders: " +
                    string.Join(", ", unfilled.Select(n => "{{" + n + "}}")) +
                    ". Provide them via --cluster_var key=value.");

            return result;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~TemplateRendererTests"`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Scheduling/TemplateRenderer.cs Tests/Workers/TemplateRendererTests.cs
git commit -m "feat: TemplateRenderer with crash-on-unfilled-placeholder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: ClusterQueueDefinition

**Files:**
- Create: `WarpLib/Workers/Scheduling/ClusterQueueDefinition.cs`
- Test: `Tests/Workers/ClusterQueueDefinitionTests.cs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Warp.Workers.Scheduling.ClusterQueueDefinition` with `string Submit`, `string SubmitJobIdRegex`, `string Cancel` (all `{ get; set; }`).
  - `static ClusterQueueDefinition Load(string path)` — throws `FileNotFoundException` if missing, and `Exception` on bad JSON or a missing/empty required field (message names the field).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Workers/ClusterQueueDefinitionTests.cs`:

```csharp
using System;
using System.IO;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class ClusterQueueDefinitionTests
{
    private static string WriteTemp(string contents)
    {
        string p = Path.Combine(Path.GetTempPath(), "clusterdef_" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(p, contents);
        return p;
    }

    [Fact]
    public void Load_ValidConfig_PopulatesAllFields()
    {
        string p = WriteTemp(@"{
            ""submit"": ""sbatch {{script_path}}"",
            ""submit_job_id_regex"": ""Submitted batch job (\\d+)"",
            ""cancel"": ""scancel {{job_id}}""
        }");
        try
        {
            var def = ClusterQueueDefinition.Load(p);
            Assert.Equal("sbatch {{script_path}}", def.Submit);
            Assert.Equal(@"Submitted batch job (\d+)", def.SubmitJobIdRegex);
            Assert.Equal("scancel {{job_id}}", def.Cancel);
        }
        finally { File.Delete(p); }
    }

    [Fact]
    public void Load_MissingField_ThrowsNamingField()
    {
        string p = WriteTemp(@"{ ""submit"": ""x"", ""submit_job_id_regex"": ""y"" }");
        try
        {
            var ex = Assert.Throws<Exception>(() => ClusterQueueDefinition.Load(p));
            Assert.Contains("cancel", ex.Message);
        }
        finally { File.Delete(p); }
    }

    [Fact]
    public void Load_EmptyField_Throws()
    {
        string p = WriteTemp(@"{ ""submit"": """", ""submit_job_id_regex"": ""y"", ""cancel"": ""z"" }");
        try
        {
            var ex = Assert.Throws<Exception>(() => ClusterQueueDefinition.Load(p));
            Assert.Contains("submit", ex.Message);
        }
        finally { File.Delete(p); }
    }

    [Fact]
    public void Load_MissingFile_Throws()
    {
        Assert.ThrowsAny<Exception>(() =>
            ClusterQueueDefinition.Load(Path.Combine(Path.GetTempPath(), "does_not_exist_" + Guid.NewGuid().ToString("N") + ".json")));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterQueueDefinitionTests"`
Expected: FAIL — "ClusterQueueDefinition does not exist".

- [ ] **Step 3: Write the implementation**

Create `WarpLib/Workers/Scheduling/ClusterQueueDefinition.cs`:

```csharp
using System;
using System.IO;
using System.Text.Json.Nodes;

namespace Warp.Workers.Scheduling
{
    /// <summary>
    /// The user-supplied cluster configuration (pointed to by --cluster_config). Three
    /// fields describe how to talk to the batch scheduler:
    ///   submit              - command to submit the rendered script ({{script_path}})
    ///   submit_job_id_regex - first capture group extracts the job id from submit stdout
    ///   cancel              - command to cancel one job ({{job_id}}), run once per id
    /// One configurable regex covers any scheduler, so no per-scheduler logic ships here.
    /// </summary>
    public class ClusterQueueDefinition
    {
        public string Submit { get; set; }
        public string SubmitJobIdRegex { get; set; }
        public string Cancel { get; set; }

        public static ClusterQueueDefinition Load(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Cluster config path is empty.");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Cluster config file not found: {path}");

            JsonNode root;
            try
            {
                root = JsonNode.Parse(File.ReadAllText(path));
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to parse cluster config JSON '{path}': {ex.Message}", ex);
            }

            string Req(string name)
            {
                string v = null;
                try { v = root?[name]?.GetValue<string>(); } catch { /* wrong type -> treat as missing */ }
                if (string.IsNullOrWhiteSpace(v))
                    throw new Exception($"Cluster config '{path}' is missing required string field '{name}'.");
                return v;
            }

            return new ClusterQueueDefinition
            {
                Submit = Req("submit"),
                SubmitJobIdRegex = Req("submit_job_id_regex"),
                Cancel = Req("cancel"),
            };
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterQueueDefinitionTests"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Scheduling/ClusterQueueDefinition.cs Tests/Workers/ClusterQueueDefinitionTests.cs
git commit -m "feat: ClusterQueueDefinition JSON loader (submit/cancel/job-id regex)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: ClusterProvisioner

**Files:**
- Create: `WarpLib/Workers/Scheduling/ClusterProvisioner.cs`
- Test: `Tests/Workers/ClusterProvisionerTests.cs`

**Interfaces:**
- Consumes:
  - `ClusterVarParser.Parse` (Task 1), `TemplateRenderer.Render` (Task 2), `ClusterQueueDefinition` + `.Load` (Task 3).
  - `Warp.Workers.Scheduling.IWorkerProvisioner` (existing: `void EnsureWorkers(int)`, `int LiveWorkerCount()`, `void Shutdown()`).
- Produces:
  - `readonly record struct ShellResult(int ExitCode, string StdOut, string StdErr)`.
  - `delegate ShellResult ShellRunner(string commandLine)`.
  - `class ClusterProvisioner : IWorkerProvisioner` with:
    - `ClusterProvisioner(ClusterQueueDefinition queue, string scriptPath, ShellRunner runner = null, bool registerSignalHandlers = true)`.
    - `static ClusterProvisioner Create(string clusterScriptPath, string clusterConfigPath, bool externalProvisioner, int poolSize, IEnumerable<string> clusterVars, string workerExePath, string queueDir, string logDir, ShellRunner runner = null, bool registerSignalHandlers = true)`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Workers/ClusterProvisionerTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using Warp.Workers.Scheduling;
using Xunit;

namespace Tests.Workers;

public class ClusterProvisionerTests
{
    private static ClusterQueueDefinition FakeQueue() => new ClusterQueueDefinition
    {
        Submit = "SUBMIT {{script_path}}",
        SubmitJobIdRegex = @"Submitted batch job (\d+)",
        Cancel = "CANCEL {{job_id}}",
    };

    // Fake shell: SUBMIT lines return an incrementing job id; CANCEL lines are recorded.
    private sealed class FakeShell
    {
        public int NextId = 100;
        public readonly List<string> Submits = new();
        public readonly List<string> Cancels = new();

        public ShellResult Run(string cmd)
        {
            if (cmd.StartsWith("SUBMIT"))
            {
                Submits.Add(cmd);
                return new ShellResult(0, $"Submitted batch job {NextId++}", "");
            }
            if (cmd.StartsWith("CANCEL"))
            {
                var m = Regex.Match(cmd, @"CANCEL (\d+)");
                Cancels.Add(m.Groups[1].Value);
                return new ShellResult(0, "", "");
            }
            return new ShellResult(1, "", "unexpected command");
        }
    }

    private static string WriteScript()
    {
        string p = Path.Combine(Path.GetTempPath(), "worker_" + Guid.NewGuid().ToString("N") + ".sh");
        File.WriteAllText(p, "#!/bin/bash\necho hi\n");
        return p;
    }

    [Fact]
    public void EnsureWorkers_SubmitsTargetCount_AndParsesIds()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(3);

        Assert.Equal(3, shell.Submits.Count);
        Assert.Equal(3, prov.LiveWorkerCount());
    }

    [Fact]
    public void EnsureWorkers_IsIdempotent_DoesNotOversubmit()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(2);
        prov.EnsureWorkers(2);

        Assert.Equal(2, shell.Submits.Count);
    }

    [Fact]
    public void Shutdown_CancelsEverySubmittedJobOnce()
    {
        var shell = new FakeShell();
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), shell.Run, registerSignalHandlers: false);

        prov.EnsureWorkers(3);
        prov.Shutdown();
        prov.Shutdown();   // idempotent: must not cancel again

        Assert.Equal(3, shell.Cancels.Count);
        Assert.Contains("100", shell.Cancels);
        Assert.Contains("101", shell.Cancels);
        Assert.Contains("102", shell.Cancels);
    }

    [Fact]
    public void EnsureWorkers_UnparseableSubmitOutput_Throws()
    {
        ShellRunner garbage = _ => new ShellResult(0, "no id here", "");
        var prov = new ClusterProvisioner(FakeQueue(), WriteScript(), garbage, registerSignalHandlers: false);

        Assert.ThrowsAny<Exception>(() => prov.EnsureWorkers(1));
    }

    [Fact]
    public void Create_ExternalProvisionerConflict_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: true,
            poolSize: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_PoolSizeZero_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "y", externalProvisioner: false,
            poolSize: 0, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_MissingConfigPath_Throws()
    {
        Assert.ThrowsAny<Exception>(() => ClusterProvisioner.Create(
            clusterScriptPath: "x", clusterConfigPath: "", externalProvisioner: false,
            poolSize: 1, clusterVars: null, workerExePath: "w", queueDir: Path.GetTempPath(),
            logDir: Path.GetTempPath(), runner: null, registerSignalHandlers: false));
    }

    [Fact]
    public void Create_RendersScriptWithCommandAndClusterVars_Submits()
    {
        // Real config file + template file; fake shell. Proves Create wires the whole chain.
        string cfgPath = Path.Combine(Path.GetTempPath(), "cfg_" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(cfgPath, @"{
            ""submit"": ""SUBMIT {{script_path}}"",
            ""submit_job_id_regex"": ""Submitted batch job (\\d+)"",
            ""cancel"": ""CANCEL {{job_id}}""
        }");
        string tmplPath = Path.Combine(Path.GetTempPath(), "tmpl_" + Guid.NewGuid().ToString("N") + ".sh");
        File.WriteAllText(tmplPath, "#!/bin/bash\n#SBATCH -p {{partition}}\n{{command}}\n");
        string queueDir = Path.Combine(Path.GetTempPath(), "q_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(queueDir);

        var shell = new FakeShell();
        try
        {
            var prov = ClusterProvisioner.Create(
                clusterScriptPath: tmplPath, clusterConfigPath: cfgPath, externalProvisioner: false,
                poolSize: 2, clusterVars: new[] { "partition=gpu" },
                workerExePath: "/opt/warp/WarpWorker2", queueDir: queueDir, logDir: Path.Combine(queueDir, "logs"),
                runner: shell.Run, registerSignalHandlers: false);

            string written = File.ReadAllText(Path.Combine(queueDir, "cluster", "worker.sh"));
            Assert.Contains("#SBATCH -p gpu", written);
            Assert.Contains("--device 0", written);
            Assert.Contains("$(hostname)-$$", written);
            Assert.Contains("--persistent", written);

            prov.EnsureWorkers(2);
            Assert.Equal(2, shell.Submits.Count);
        }
        finally
        {
            File.Delete(cfgPath); File.Delete(tmplPath);
            try { Directory.Delete(queueDir, true); } catch { }
        }
    }

    [Fact]
    public void DefaultShellRunner_RealShell_SubmitAndCancel()
    {
        // Exercises the real /bin/sh path (not the fake runner). Unix-only.
        if (OperatingSystem.IsWindows()) return;

        string marker = Path.Combine(Path.GetTempPath(), "cancelled_" + Guid.NewGuid().ToString("N") + ".txt");
        var queue = new ClusterQueueDefinition
        {
            Submit = "echo Submitted batch job 4242",
            SubmitJobIdRegex = @"Submitted batch job (\d+)",
            Cancel = $"echo {{{{job_id}}}} >> \"{marker}\"",
        };
        var prov = new ClusterProvisioner(queue, WriteScript(),
            runner: null /* default real shell */, registerSignalHandlers: false);
        try
        {
            prov.EnsureWorkers(2);
            Assert.Equal(2, prov.LiveWorkerCount());

            prov.Shutdown();
            string[] lines = File.ReadAllLines(marker);
            Assert.Equal(2, lines.Length);
            Assert.All(lines, l => Assert.Equal("4242", l.Trim()));
        }
        finally { try { File.Delete(marker); } catch { } }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterProvisionerTests"`
Expected: FAIL — "ClusterProvisioner does not exist".

- [ ] **Step 3: Write the implementation**

Create `WarpLib/Workers/Scheduling/ClusterProvisioner.cs`:

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace Warp.Workers.Scheduling
{
    /// <summary>Result of running one shell command line.</summary>
    public readonly record struct ShellResult(int ExitCode, string StdOut, string StdErr);

    /// <summary>Runs one shell command line and returns its result. Injectable for tests.</summary>
    public delegate ShellResult ShellRunner(string commandLine);

    /// <summary>
    /// Minimal cluster provisioning: submit a fixed pool of identical --persistent
    /// WarpWorker2 jobs to a batch scheduler, then cancel them all on shutdown. No status
    /// polling or resubmission — the filesystem queue's stall-sweep re-pends a dead
    /// worker's task for a surviving worker. See the design spec for the full rationale.
    /// </summary>
    public class ClusterProvisioner : IWorkerProvisioner
    {
        private readonly ClusterQueueDefinition _queue;
        private readonly string _scriptPath;
        private readonly ShellRunner _runner;

        private readonly List<string> _jobIds = new();
        private readonly object _sync = new();
        private bool _cancelled;
        private readonly List<IDisposable> _signalRegs = new();

        public ClusterProvisioner(ClusterQueueDefinition queue, string scriptPath,
                                  ShellRunner runner = null, bool registerSignalHandlers = true)
        {
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _scriptPath = scriptPath ?? throw new ArgumentNullException(nameof(scriptPath));
            _runner = runner ?? RunShell;

            if (registerSignalHandlers)
            {
                // --persistent workers won't self-stop, so make sure Ctrl-C / kill on the
                // manager cancels the pool before we die. Shutdown() is idempotent.
                TryRegister(PosixSignal.SIGINT);
                TryRegister(PosixSignal.SIGTERM);
            }
        }

        private void TryRegister(PosixSignal signal)
        {
            try { _signalRegs.Add(PosixSignalRegistration.Create(signal, _ => Shutdown())); }
            catch { /* signal not supported on this platform; ignore */ }
        }

        /// <summary>
        /// Build a ClusterProvisioner from raw CLI option values: validate the option
        /// combination, load the config, parse --cluster_var, build the built-in
        /// {{command}}, render the submission script once to &lt;queueDir&gt;/cluster/worker.sh.
        /// </summary>
        public static ClusterProvisioner Create(
            string clusterScriptPath, string clusterConfigPath, bool externalProvisioner,
            int poolSize, IEnumerable<string> clusterVars,
            string workerExePath, string queueDir, string logDir,
            ShellRunner runner = null, bool registerSignalHandlers = true)
        {
            // Validation first (no file IO), so misconfiguration fails fast and cheap.
            if (externalProvisioner)
                throw new Exception("--cluster_script cannot be combined with --external_provisioner; pick one provisioning mode.");
            if (poolSize <= 0)
                throw new Exception("--pool_size must be greater than 0 in cluster mode.");
            if (string.IsNullOrEmpty(clusterConfigPath))
                throw new Exception("--cluster_script requires --cluster_config (the queue-definition JSON).");
            if (string.IsNullOrEmpty(clusterScriptPath) || !File.Exists(clusterScriptPath))
                throw new Exception($"Cluster submission-script template not found: {clusterScriptPath}");

            ClusterQueueDefinition def = ClusterQueueDefinition.Load(clusterConfigPath);

            Dictionary<string, string> vars = ClusterVarParser.Parse(clusterVars);

            // Built-in command: one GPU per job (device 0), self-naming by hostname+pid so
            // ids never collide across nodes; --persistent so a transient empty queue does
            // not make the worker exit and hang a re-pended task. The compute node's shell
            // expands $(hostname) and $$ at runtime.
            string command =
                $"\"{workerExePath}\" --device 0 --queue-dir \"{queueDir}\" --log-dir \"{logDir}\" " +
                $"--persistent --worker-id \"$(hostname)-$$\"";
            vars["command"] = command;   // built-in wins over any user-supplied command var

            string template = File.ReadAllText(clusterScriptPath);
            string script = TemplateRenderer.Render(template, vars);

            string scriptDir = Path.Combine(queueDir, "cluster");
            Directory.CreateDirectory(scriptDir);
            string scriptPath = Path.Combine(scriptDir, "worker.sh");
            File.WriteAllText(scriptPath, script);

            return new ClusterProvisioner(def, scriptPath, runner, registerSignalHandlers);
        }

        public void EnsureWorkers(int target)
        {
            lock (_sync)
            {
                if (_cancelled) return;
                while (_jobIds.Count < target)
                {
                    string cmd = TemplateRenderer.Render(
                        _queue.Submit, new Dictionary<string, string> { ["script_path"] = _scriptPath });
                    ShellResult r = _runner(cmd);

                    Match m = Regex.Match(r.StdOut ?? "", _queue.SubmitJobIdRegex);
                    if (!m.Success || m.Groups.Count < 2)
                        throw new Exception(
                            $"Could not parse a job id from the scheduler's submit output using " +
                            $"submit_job_id_regex '{_queue.SubmitJobIdRegex}'.\nstdout: {r.StdOut}\nstderr: {r.StdErr}");

                    _jobIds.Add(m.Groups[1].Value);
                }
            }
        }

        public int LiveWorkerCount()
        {
            lock (_sync) return _jobIds.Count;
        }

        public void Shutdown()
        {
            lock (_sync)
            {
                if (_cancelled) return;
                _cancelled = true;

                foreach (string id in _jobIds)
                {
                    try
                    {
                        string cmd = TemplateRenderer.Render(
                            _queue.Cancel, new Dictionary<string, string> { ["job_id"] = id });
                        _runner(cmd);
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"Failed to cancel cluster job {id}: {ex.Message}");
                    }
                }

                foreach (IDisposable reg in _signalRegs)
                    try { reg.Dispose(); } catch { }
                _signalRegs.Clear();
            }
        }

        /// <summary>Default ShellRunner: run the command line through the platform shell.</summary>
        public static ShellResult RunShell(string commandLine)
        {
            bool win = OperatingSystem.IsWindows();
            var psi = new ProcessStartInfo
            {
                FileName = win ? "cmd.exe" : "/bin/sh",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.ArgumentList.Add(win ? "/c" : "-c");
            psi.ArgumentList.Add(commandLine);

            using var p = Process.Start(psi);
            string o = p.StandardOutput.ReadToEnd();
            string e = p.StandardError.ReadToEnd();
            p.WaitForExit();
            return new ShellResult(p.ExitCode, o, e);
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test Tests/Tests.csproj --filter "FullyQualifiedName~ClusterProvisionerTests"`
Expected: PASS (9 tests; the real-shell one self-skips on Windows).

- [ ] **Step 5: Commit**

```bash
git add WarpLib/Workers/Scheduling/ClusterProvisioner.cs Tests/Workers/ClusterProvisionerTests.cs
git commit -m "feat: ClusterProvisioner — submit-once pool + cancel-on-shutdown

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Wire cluster mode into DistributedOptions

**Files:**
- Modify: `WarpTools/Commands/DistributedOptions.cs` (add 4 options; add `CreateProvisioner`; replace the two selection blocks at ~101-125 and ~273-291).

**Interfaces:**
- Consumes: `ClusterProvisioner.Create` (Task 4); existing `ExternalProvisioner`, `LocalProvisioner`, `Scheduler`, `QueueLayout`.
- Produces: `private IWorkerProvisioner CreateProvisioner(QueueLayout layout, string logDir, int itemCount, out int target)`.

> This task's logic is verified by the WarpLib unit tests behind it (Task 4) plus a successful build, because the `Tests` project does not reference WarpTools. There is no per-step unit test here; the test step is a clean build of the whole solution.

- [ ] **Step 1: Add the four CLI options**

In `WarpTools/Commands/DistributedOptions.cs`, inside the `[OptionGroup("Advanced remote work distribution", 102)]` group (right after the existing `UseExternalProvisioner` property at lines ~33-35), add:

```csharp
        [Option("cluster_script", HelpText = "Path to a batch-scheduler submission-script template. " +
                                             "Use {{command}} where the WarpWorker2 invocation should go, plus any " +
                                             "{{custom}} placeholders filled by --cluster_var. Presence of this option " +
                                             "selects cluster mode. Assumes the queue dir is on a shared filesystem, " +
                                             "the WarpTools install is at the same path on compute nodes, and the script " +
                                             "shell expands $(hostname)/$$.")]
        public string ClusterScript { get; set; }

        [Option("cluster_config", HelpText = "Path to the cluster queue-definition JSON. Required with --cluster_script. " +
                                             "Fields: submit, submit_job_id_regex (first capture group = job id), cancel.")]
        public string ClusterConfig { get; set; }

        [Option("pool_size", HelpText = "Cluster mode: number of worker jobs to submit to the scheduler.")]
        public int PoolSize { get; set; }

        [Option("cluster_var", HelpText = "Cluster mode: a key=value pair substituted into the submission template " +
                                          "(repeatable). Whitespace around '=' is tolerated. Quote values containing " +
                                          "spaces, e.g. --cluster_var \"account=my project\".")]
        public IEnumerable<string> ClusterVars { get; set; }
```

- [ ] **Step 2: Add the `CreateProvisioner` helper**

In the same file, add this private method to the `DistributedOptions` class (e.g. immediately before `DistributeItems`). `Warp.Workers.Scheduling` is already imported (line 13):

```csharp
        /// <summary>
        /// Select and construct the worker provisioner for this run: cluster mode
        /// (--cluster_script), external mode (--external_provisioner), or local mode
        /// (default). Sets <paramref name="target"/> to the desired live worker count.
        /// </summary>
        private IWorkerProvisioner CreateProvisioner(
            QueueLayout layout, string logDir, int itemCount, out int target)
        {
            if (!string.IsNullOrEmpty(ClusterScript))
            {
                if ((DeviceList != null && DeviceList.Any()) || ProcessesPerDevice != 1)
                    Console.Error.WriteLine("Warning: --device_list/--perdevice are ignored in cluster mode " +
                                            "(one cluster job = one GPU = one worker).");

                target = Math.Min(itemCount, PoolSize);
                string workerExe = Path.Combine(AppContext.BaseDirectory, "WarpWorker2");
                var prov = ClusterProvisioner.Create(
                    clusterScriptPath: ClusterScript,
                    clusterConfigPath: ClusterConfig,
                    externalProvisioner: UseExternalProvisioner,
                    poolSize: PoolSize,
                    clusterVars: ClusterVars,
                    workerExePath: workerExe,
                    queueDir: layout.Root,
                    logDir: logDir);
                Console.WriteLine($"Distributing {itemCount} item(s) across a cluster pool of up to {target} worker(s)...");
                return prov;
            }

            if (UseExternalProvisioner)
            {
                target = 0;
                Console.WriteLine($"Distributing {itemCount} item(s); workers provisioned externally...");
                return new ExternalProvisioner();
            }

            List<int> devices = (DeviceList == null || !DeviceList.Any())
                ? Helper.ArrayOfSequence(0, GPU.GetDeviceCount(), 1).ToList()
                : DeviceList.ToList();
            if (devices.Count <= 0)
                throw new Exception("No devices found or specified");
            target = Math.Min(itemCount, devices.Count * ProcessesPerDevice);
            Console.WriteLine($"Distributing {itemCount} item(s) across up to {target} local worker(s)...");
            return new LocalProvisioner(layout.Root, devices.ToArray(), ProcessesPerDevice, logDir: logDir);
        }
```

- [ ] **Step 3: Replace the selection block in `DistributeItems`**

In `DistributeItems`, delete the entire block currently at lines ~101-125 (from `IWorkerProvisioner provisioner;` through the closing brace of the `else { ... }` that constructs the `LocalProvisioner`) and replace it with:

```csharp
            IWorkerProvisioner provisioner = CreateProvisioner(layout, logDir, InputSeries.Length, out int target);
```

Leave the following line `var scheduler = new Scheduler(layout, queue, provisioner, target);` unchanged.

- [ ] **Step 4: Replace the selection block in `DistributeTasks`**

In `DistributeTasks`, delete the entire block currently at lines ~273-291 (the `IWorkerProvisioner provisioner; int target;` declarations and the `if (UseExternalProvisioner) { ... } else { ... }`) and replace it with:

```csharp
            IWorkerProvisioner provisioner = CreateProvisioner(layout, logDir, tasks.Count, out int target);
```

Leave the following line `var scheduler = new Scheduler(layout, queue, provisioner, target);` unchanged.

- [ ] **Step 5: Build the whole solution**

Run: `dotnet build WarpTools/WarpTools.csproj`
Expected: Build succeeded, 0 errors. (Fix any compile errors — e.g. a leftover duplicate `target` declaration — before continuing.)

- [ ] **Step 6: Commit**

```bash
git add WarpTools/Commands/DistributedOptions.cs
git commit -m "feat: cluster provisioning mode in WarpTools DistributedOptions

Add --cluster_script/--cluster_config/--pool_size/--cluster_var and a single
CreateProvisioner helper selecting cluster/external/local; both DistributeItems
and DistributeTasks now route through it, so cluster mode works for every ported
command and for reduce-style whole-run tasks.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run: `dotnet test Tests/Tests.csproj`
Expected: All tests pass, including the new ClusterVarParser / TemplateRenderer / ClusterQueueDefinition / ClusterProvisioner tests and all pre-existing Workers tests.

- [ ] **Step 2: Manual end-to-end smoke with a fake scheduler (Unix, optional but recommended)**

This exercises the real WarpTools CLI path without a real cluster. Pick any ported WarpTool that uses `DistributedOptions` and a small input set, then point it at a fake scheduler:

Create `/tmp/fakeq.json`:
```json
{
  "submit": "sh {{script_path}} & echo Submitted batch job $!",
  "submit_job_id_regex": "Submitted batch job ([0-9]+)",
  "cancel": "kill {{job_id}}"
}
```
Create `/tmp/worker.sbatch`:
```
#!/bin/bash
# fake "scheduler" just runs the worker directly in the background
{{command}}
```
Run the chosen tool with `--cluster_script /tmp/worker.sbatch --cluster_config /tmp/fakeq.json --pool_size 2 --cluster_var dummy=1`.

Verify: `<output>/tasks/cluster/worker.sh` is written and contains `--device 0`, `--persistent`, and `$(hostname)-$$`; the workers actually drain the queue; and on completion (or Ctrl-C) the `kill` cancel command runs for each job id. Note this fake "submit" backgrounds the worker locally — it is only a smoke check of rendering/submit/cancel wiring, not a real cluster run.

- [ ] **Step 3: Confirm no regression in the existing distribution modes**

Confirm a normal local run (no cluster/external flags) and an `--external_provisioner` run still behave as before — both now route through `CreateProvisioner`. A quick local run of any ported tool on a tiny dataset is sufficient.

---

## Self-Review

**1. Spec coverage:**
- Submit-once + cancel-on-shutdown lifecycle → Task 4 (`EnsureWorkers`/`Shutdown`).
- `--cluster_script`/`--cluster_config`/`--pool_size`/`--cluster_var` options → Task 5 Step 1.
- Mode selection + validation (config required, pool_size>0, external conflict, device/perdevice warning) → Task 4 `Create` + Task 5 `CreateProvisioner`.
- `ClusterQueueDefinition` 3-field JSON → Task 3.
- `ClusterVarParser` lenient `=` → Task 1.
- `TemplateRenderer` substitution + unfilled crash → Task 2.
- `ClusterProvisioner` built-in `{{command}}`, device 0, `$(hostname)-$$`, `--persistent`, render-once to `<queue>/cluster/worker.sh`, signal handlers → Task 4.
- Job-id parse + throw-if-no-match → Task 4 `EnsureWorkers`.
- Integration into both `DistributeItems` and `DistributeTasks` via one helper → Task 5.
- Heartbeat-net fallback for hard crashes → already provided by existing worker behavior (no new code); documented in spec.
- Testing (unit for parser/renderer/loader; fake-runner + real-shell for provisioner) → Tasks 1-4.

No spec requirement is left without a task.

**2. Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N". Every code step shows full code; every command shows expected output.

**3. Type consistency:** `ClusterVarParser.Parse(IEnumerable<string>)→Dictionary<string,string>`, `TemplateRenderer.Render(string, IReadOnlyDictionary<string,string>)→string`, `ClusterQueueDefinition{Submit,SubmitJobIdRegex,Cancel}`+`Load(string)`, `ShellResult(int,string,string)`, `ShellRunner=ShellResult(string)`, `ClusterProvisioner(ClusterQueueDefinition,string,ShellRunner,bool)` + `Create(...)` — names/signatures are used identically across Tasks 1-5.
