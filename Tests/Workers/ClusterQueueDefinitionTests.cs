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
