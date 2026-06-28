using System;
using System.IO;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

// Averaged-reconstruction commands for the filesystem work-distribution path.
//
// Because workers are ephemeral (the manager/Relay may spawn and sweep them at
// will), there is no addressable, fixed worker pool to issue a final "reduce" to.
// Instead each worker accumulates into its own resident Projector(s) across the
// tilt series it claims (InitReconstructions runs once per worker via the task's
// amortized init), and atomically rewrites its running partial after every item.
// A crash therefore loses at most the current item. A separate single reduce task
// then sums every per-worker partial into a fresh accumulator and writes the maps.
static partial class WorkerProcess
{
    [Command(WorkerCommandNames.InitReconstructions)]
    static void InitReconstructions(NamedSerializableObject Command)
    {
        if (Reconstructions != null)
            foreach (var rec in Reconstructions) rec?.Dispose();

        int NReconstructions = (int)Command.Content[0];
        int BoxSize = (int)Command.Content[1];
        int Oversample = (int)Command.Content[2];

        Reconstructions = Helper.ArrayOfFunction(
            i => new Projector(new int3(BoxSize), Oversample), NReconstructions);

        Console.WriteLine($"Initialized {NReconstructions} reconstruction(s)");
    }

    [Command(WorkerCommandNames.TomoAddToReconstructionAndSave)]
    static void TomoAddToReconstructionAndSave(NamedSerializableObject Command)
    {
        if (Reconstructions == null)
            throw new Exception("Reconstructions not initialized");

        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoAddToReconstruction)Command.Content[1];
        float3[][] Positions = (float3[][])Command.Content[2];
        float3[][] Angles = (float3[][])Command.Content[3];
        string TempDir = (string)Command.Content[4];

        if (Positions.Length != Angles.Length || Positions.Length != Reconstructions.Length)
            throw new Exception("The number of reconstructions, and particle position and angle sets must match");

        TiltSeries T = new TiltSeries(Path);

        for (int irec = 0; irec < Reconstructions.Length; irec++)
        {
            if (Positions[irec].Length != Angles[irec].Length)
                throw new Exception("The number of particle positions and angles must match for each reconstruction");

            Console.WriteLine($"Adding {Positions[irec].Length / T.NTilts} particles to reconstruction {irec}");
        }

        T.AddToReconstruction(Options, Reconstructions, Positions, Angles);

        // Safe-save the running accumulators: write to a temp name, then atomically
        // replace this worker's partial. The replace is the only durability point, so
        // a crash mid-write leaves the previous (complete) partial intact and loses at
        // most this item — which the queue will re-pend and another worker will redo.
        for (int irec = 0; irec < Reconstructions.Length; irec++)
        {
            string finalPath = System.IO.Path.Combine(TempDir, $"partial_{WorkerId}_rec{irec}.mrc");
            string tmpPath = finalPath + ".tmp";

            Reconstructions[irec].WriteMRC(tmpPath);
            File.Move(tmpPath, finalPath, overwrite: true);
        }

        Console.WriteLine($"Added particles from {Path}; saved partial for worker {WorkerId}");
    }

    [Command(WorkerCommandNames.TomoFinishReconstruction)]
    static void TomoFinishReconstruction(NamedSerializableObject Command)
    {
        string[][] PartialPaths = (string[][])Command.Content[0];
        string[] Symmetries = (string[])Command.Content[1];
        string[] OutputPaths = (string[])Command.Content[2];
        float PixelSize = (float)Command.Content[3];
        int BoxSize = (int)Command.Content[4];
        int Oversample = (int)Command.Content[5];

        if (PartialPaths.Length != Symmetries.Length || Symmetries.Length != OutputPaths.Length)
            throw new Exception("The number of reconstructions, symmetry definitions, and output paths must match");

        // Reduce into a fresh zero accumulator — independent of any resident state this
        // worker may carry from a prior map task, so a partial is never double-counted.
        for (int irec = 0; irec < PartialPaths.Length; irec++)
        {
            using Projector Accumulator = new Projector(new int3(BoxSize), Oversample);

            foreach (var path in PartialPaths[irec])
            {
                Projector Partial = Projector.FromFile(path);
                Accumulator.Data.Add(Partial.Data);
                Accumulator.Weights.Add(Partial.Weights);
                Partial.Dispose();
            }

            using Image Reconstruction = Accumulator.Reconstruct(false, Symmetries[irec]);
            Reconstruction.WriteMRC(OutputPaths[irec], PixelSize, true);

            Console.WriteLine($"Wrote reconstruction {irec} to {OutputPaths[irec]} from {PartialPaths[irec].Length} partial(s)");
        }

        Console.WriteLine("Finished reconstructions");
    }
}
