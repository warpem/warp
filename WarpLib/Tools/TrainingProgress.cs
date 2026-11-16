using System;
using System.IO;
using System.Text.Json;

namespace Warp.Tools;

/// <summary>
/// Where a denoiser training run is, written beside the model it trains so whoever queued it can show progress
/// without reading worker logs.
/// </summary>
public sealed record TrainingProgress(int Step, int Steps, int Epoch, int Epochs, float Loss)
{
    public int Percent => Steps > 0 ? (int)Math.Clamp(100L * Step / Steps, 0, 100) : 0;

    public static string PathFor(string modelPath) => modelPath + ".progress.json";

    /// <summary>Atomic, so a reader never sees half a file.</summary>
    public void Write(string path)
    {
        string temporary = path + ".writing";
        File.WriteAllText(temporary, JsonSerializer.Serialize(this));
        File.Move(temporary, path, true);
    }

    public static TrainingProgress TryRead(string path)
    {
        try { return File.Exists(path) ? JsonSerializer.Deserialize<TrainingProgress>(File.ReadAllText(path)) : null; }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException) { return null; }
    }
}
