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

    [Command(WorkerCommandNames.MoviePrepareDenoising)]
    static void MoviePrepareDenoising(NamedSerializableObject command) =>
        MovieDenoising.Prepare((string)command.Content[0], (float)command.Content[1],
            (float)command.Content[2], (float)command.Content[3]);

    [Command(WorkerCommandNames.TrainMovieDenoiser)]
    static void TrainMovieDenoiser(NamedSerializableObject command)
    {
        // Training is a standalone queue task; release a prior movie/model before allocating its batches.
        OriginalStack?.Dispose(); OriginalStack = null; OriginalStackOwner = "";
        MicrographDenoiserModel?.Dispose(); MicrographDenoiserModel = null; MicrographDenoiserPath = "";
        MovieDenoising.Train((string[])command.Content[0], (string)command.Content[1],
            (string)command.Content[2], (int)command.Content[3], DeviceID, (float)command.Content[4], (float)command.Content[5]);
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
        MovieDenoising.Apply((string)command.Content[0], MicrographDenoiserModel);
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
        string[] x = table.GetColumn("rlnCoordinateX"), y = table.GetColumn("rlnCoordinateY");
        var coordinates = new float2[table.RowCount];
        for (int i = 0; i < coordinates.Length; i++)
        {
            float xx = float.Parse(x[i], CultureInfo.InvariantCulture), yy = float.Parse(y[i], CultureInfo.InvariantCulture);
            if (!float.IsFinite(xx) || !float.IsFinite(yy)) throw new InvalidDataException("Non-finite particle coordinates.");
            coordinates[i] = new float2(xx, yy) * sampling;
        }
        // There is no stack to write for an empty pick list. The source STAR remains an explicit empty result.
        if (coordinates.Length == 0) return;
        options.Dimensions = OriginalStack.Dims.MultXY((float)options.BinnedPixelSizeMean);
        new Movie(path).ExportParticles(OriginalStack, coordinates, options);
        Console.WriteLine($"Exported {coordinates.Length} particles for {path}");
    }
}
