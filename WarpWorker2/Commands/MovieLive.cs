using System;
using System.Globalization;
using System.IO;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

static partial class WorkerProcess
{
    static Noise2MicTorch MicrographDenoiserModel;
    static string MicrographDenoiserPath = "";
    static JunkNetTorch JunkModel;
    static string JunkModelPath = "";

    /// <summary>
    /// Denoising's object erasure needs only the exported averages, so while a task fits a movie's CTF the erasure for
    /// its later MovieDenoise runs on another thread (CPU work, and its own FFT plans); denoising then deconvolves and
    /// runs the network. It is the same computation on the same files, which MovieDenoising.Denoise checks.
    /// </summary>
    static (string Path, float PixelSize, System.Threading.Tasks.Task<MovieDenoising.Erasure> Work) PendingErasure;

    static void StartErasureAhead(NamedSerializableObject[] commands, int index)
    {
        if (MockMode || commands[index].Name != WorkerCommandNames.MovieProcessCTF) return;
        string path = (string)commands[index].Content[0];
        for (int i = index + 1; i < commands.Length; i++)
            if (commands[i].Name == WorkerCommandNames.MovieDenoise && (string)commands[i].Content[0] == path)
            {
                FinishErasure();
                float pixelSize = (float)commands[i].Content[2];
                bool eraseObjects = (bool)commands[i].Content[5];
                int device = DeviceID;
                PendingErasure = (path, pixelSize, System.Threading.Tasks.Task.Run(() =>
                {
                    GPU.SetDevice(device);
                    using var plans = new PrivateFftPlans();
                    return MovieDenoising.Erase(path, pixelSize, plans, eraseObjects);
                }));
                return;
            }
    }


    /// <summary>The erasure made ahead for this movie, or null (denoising then makes it, and reports any error).</summary>
    static MovieDenoising.Erasure TakeErasure(string path, float pixelSize)
    {
        var pending = PendingErasure;
        if (pending.Work == null || pending.Path != path || pending.PixelSize != pixelSize) return null;
        PendingErasure = default;
        try { return pending.Work.GetAwaiter().GetResult(); }
        catch (Exception) { return null; }
    }

    /// <summary>Waits for and drops an erasure nobody took, so none runs into the next task.</summary>
    static void FinishErasure()
    {
        var pending = PendingErasure;
        PendingErasure = default;
        if (pending.Work == null) return;
        try { pending.Work.Wait(); } catch (Exception) { }
    }

    [Command(WorkerCommandNames.TrainMovieDenoiser)]
    static void TrainMovieDenoiser(NamedSerializableObject command)
    {
        // Training is a standalone queue task; release a prior movie/model before allocating its batches.
        OriginalStack?.Dispose(); OriginalStack = null; OriginalStackOwner = "";
        MicrographDenoiserModel?.Dispose(); MicrographDenoiserModel = null; MicrographDenoiserPath = "";
        MovieDenoising.TrainMovies((string[])command.Content[0], (string)command.Content[1], (string)command.Content[2],
            (int)command.Content[3], DeviceID, (float)command.Content[6], (float)command.Content[4], (float)command.Content[5],
            eraseObjects: (bool)command.Content[7]);
    }

    [Command(WorkerCommandNames.MovieSegmentJunk)]
    static void MovieSegmentJunk(NamedSerializableObject command)
    {
        string modelPath = JunkNetTorch.ResolvePath((string)command.Content[2]);
        if (JunkModel == null || JunkModelPath != modelPath)
        {
            JunkModel?.Dispose(); JunkModel = null; JunkModelPath = "";
            var model = new JunkNetTorch(DeviceID);
            try { model.Load(modelPath); }
            catch { model.Dispose(); throw; }
            JunkModel = model; JunkModelPath = modelPath;
        }
        MovieJunk.Segment((string)command.Content[0], (float)command.Content[1], JunkModel);
    }

    [Command(WorkerCommandNames.MovieDenoise)]
    static void MovieDenoise(NamedSerializableObject command)
    {
        string modelPath = (string)command.Content[1];
        if (MicrographDenoiserModel == null || MicrographDenoiserPath != modelPath)
        {
            MicrographDenoiserModel?.Dispose(); MicrographDenoiserModel = null; MicrographDenoiserPath = "";
            var model = new Noise2MicTorch(DeviceID);
            try { model.Load(modelPath); }
            catch { model.Dispose(); throw; }
            MicrographDenoiserModel = model; MicrographDenoiserPath = modelPath;
        }
        string path = (string)command.Content[0];
        float pixelSize = (float)command.Content[2];
        MovieDenoising.Denoise(path, pixelSize, (float)command.Content[3], (float)command.Content[4], MicrographDenoiserModel, TakeErasure(path, pixelSize), (bool)command.Content[5]);
    }

    [Command(WorkerCommandNames.MovieExportParticlesFromStar)]
    static void MovieExportParticlesFromStar(NamedSerializableObject command)
    {
        string path = (string)command.Content[0];
        var options = (ProcessingOptionsParticleExport)command.Content[1];
        string starPath = (string)command.Content[2];
        float sampling = (float)command.Content[3];
        if (!float.IsFinite(sampling) || sampling <= 0) throw new ArgumentOutOfRangeException(nameof(sampling));
        var table = new Star(starPath);
        // Only the picks within the score bounds are extracted, and their rows are recorded in stack order.
        string extractedStarPath = (string)command.Content[8];
        table = PickSelection.Select(table, (double)command.Content[4], (double)command.Content[5], (double)command.Content[6], (double)command.Content[7]);
        string[] x = table.GetColumn("rlnCoordinateX"), y = table.GetColumn("rlnCoordinateY");
        var coordinates = new float2[table.RowCount];
        for (int i = 0; i < coordinates.Length; i++)
        {
            float xx = float.Parse(x[i], CultureInfo.InvariantCulture), yy = float.Parse(y[i], CultureInfo.InvariantCulture);
            if (!float.IsFinite(xx) || !float.IsFinite(yy)) throw new InvalidDataException("Non-finite particle coordinates.");
            coordinates[i] = new float2(xx, yy) * sampling;
        }
        var movie = new Movie(path);
        if (coordinates.Length == 0)
        {
            // There is no stack for an empty list; one left from an earlier extraction is no longer valid.
            File.Delete(Path.Combine(movie.ParticlesDir, movie.RootName + options.Suffix + ".mrcs"));
        }
        else
        {
            options.Dimensions = OriginalStack.Dims.MultXY((float)options.BinnedPixelSizeMean);
            movie.ExportParticles(OriginalStack, coordinates, options, releaseStack: !StackResident);
        }
        PickSelection.Write(table, extractedStarPath);
        Console.WriteLine($"Exported {coordinates.Length} particles for {path}");
    }
}
