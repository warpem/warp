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
    public void Load_StatusConfig_PopulatesReplenishmentSettings()
    {
        string p = WriteTemp(@"{
            ""submit"": ""sbatch {{script_path}}"",
            ""submit_job_id_regex"": ""Submitted batch job (\\d+)"",
            ""cancel"": ""scancel {{job_id}}"",
            ""status_list"": ""squeue -h -u {{user}} -o '%i,%T'"",
            ""running_statuses"": [""RUNNING"", ""COMPLETING""],
            ""pending_statuses"": [""PENDING""],
            ""status_poll_seconds"": 7,
            ""registration_grace_seconds"": 90,
            ""max_submissions_per_tick"": 12,
            ""max_submission_multiplier"": 3,
            ""cancel_parallelism"": 8
        }");
        try
        {
            var def = ClusterQueueDefinition.Load(p);
            Assert.True(def.HasStatusPolling);
            Assert.Equal("squeue -h -u {{user}} -o '%i,%T'", def.StatusList);
            Assert.Equal(new[] { "RUNNING", "COMPLETING" }, def.RunningStatuses);
            Assert.Equal(new[] { "PENDING" }, def.PendingStatuses);
            Assert.Equal(7, def.StatusPollSeconds);
            Assert.Equal(90, def.RegistrationGraceSeconds);
            Assert.Equal(12, def.MaxSubmissionsPerTick);
            Assert.Equal(3, def.MaxSubmissionMultiplier);
            Assert.Equal(8, def.CancelParallelism);
        }
        finally { File.Delete(p); }
    }

    [Fact]
    public void Load_NonPositiveReplenishmentSetting_ThrowsNamingField()
    {
        string p = WriteTemp(@"{
            ""submit"": ""x"", ""submit_job_id_regex"": ""(x)"", ""cancel"": ""z"",
            ""registration_grace_seconds"": 0
        }");
        try
        {
            var ex = Assert.Throws<Exception>(() => ClusterQueueDefinition.Load(p));
            Assert.Contains("registration_grace_seconds", ex.Message);
        }
        finally { File.Delete(p); }
    }

    [Fact]
    public void Load_InvalidSubmitRegex_ThrowsAtConfigLoad()
    {
        string p = WriteTemp(@"{
            ""submit"": ""x"", ""submit_job_id_regex"": ""("", ""cancel"": ""z""
        }");
        try
        {
            var ex = Assert.Throws<Exception>(() => ClusterQueueDefinition.Load(p));
            Assert.Contains("submit_job_id_regex", ex.Message);
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
