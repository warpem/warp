using System;
using System.IO;
using Warp.Sociology;
using Xunit;

namespace Tests;

public class RefinementProgressFilesTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "refinement-progress-" + Guid.NewGuid().ToString("N"));
    private const string Species = "9a0afb13";

    public RefinementProgressFilesTests() => Directory.CreateDirectory(root);
    public void Dispose() => Directory.Delete(root, true);

    private string Folder(string name)
    {
        string path = Path.Combine(root, name);
        Directory.CreateDirectory(path);
        return path;
    }

    private void Write(string folder, string suffix, string content = "test") =>
        File.WriteAllText(Path.Combine(folder, Species + suffix), content);

    private string Complete(string name)
    {
        string folder = Folder(name);
        Write(folder, "_half1_3.mrc");
        Write(folder, "_half2_3.mrc");
        Write(folder, "_particles.star");
        return folder;
    }

    [Fact]
    public void SkipsEmptyScratchAndTemporaryOnlyFolders()
    {
        string empty = Folder("empty");
        string scratch = Folder("scratch");
        File.WriteAllText(Path.Combine(scratch, "scratch.mrc"), "scratch");
        string temporary = Folder("temporary");
        Write(temporary, "_half1_3.mrc.tmp");
        Write(temporary, "_particles.star.tmp");
        string complete = Complete("complete");
        Assert.Equal(new[] { complete }, RefinementProgressFiles.GetCompleteFolders(
            new[] { empty, scratch, temporary, complete }, Species));
    }

    [Theory]
    [InlineData("_particles.star")]
    [InlineData("_half1_3.mrc")]
    [InlineData("_half2_3.mrc")]
    public void RejectsMissingPublishedFile(string missing)
    {
        string complete = Complete("complete");
        string partial = Complete("partial");
        File.Delete(Path.Combine(partial, Species + missing));
        var error = Assert.Throws<InvalidDataException>(() =>
            RefinementProgressFiles.GetCompleteFolders(new[] { complete, partial }, Species));
        Assert.Contains(partial, error.Message);
    }

    [Fact]
    public void RejectsMismatchedHalfMapIndices()
    {
        string folder = Complete("mismatched");
        File.Move(Path.Combine(folder, Species + "_half2_3.mrc"), Path.Combine(folder, Species + "_half2_0.mrc"));
        Assert.Throws<InvalidDataException>(() => RefinementProgressFiles.GetCompleteFolders(new[] { folder }, Species));
    }

    [Fact]
    public void RejectsCheckpointPublishedOnlyForAnotherSpecies()
    {
        string folder = Complete("other");
        Assert.Throws<InvalidDataException>(() => RefinementProgressFiles.GetCompleteFolders(new[] { folder }, "other-id"));
    }

    [Theory]
    [InlineData("_particles.star")]
    [InlineData("_half1_3.mrc")]
    public void RejectsEmptyPublishedFiles(string suffix)
    {
        string folder = Complete("empty-file");
        Write(folder, suffix, "");
        Assert.Throws<InvalidDataException>(() => RefinementProgressFiles.GetCompleteFolders(new[] { folder }, Species));
    }

    [Fact]
    public void RejectsNoUsableCheckpoints() => Assert.Throws<InvalidDataException>(() =>
        RefinementProgressFiles.GetCompleteFolders(new[] { Folder("empty") }, Species));

    [Fact]
    public void AcceptsMultipleGpuPairsAndIgnoresUnpublishedUpdates()
    {
        string folder = Complete("complete");
        Write(folder, "_half1_0.mrc");
        Write(folder, "_half2_0.mrc");
        Write(folder, "_half1_3.mrc.tmp", "");
        Assert.Equal(new[] { folder }, RefinementProgressFiles.GetCompleteFolders(new[] { folder }, Species));
    }
}
