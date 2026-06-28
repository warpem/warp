using System;
using System.IO;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

// Multi-particle refinement (MPA) commands for the filesystem work-distribution path.
//
// Replaces the legacy three separate WorkerWrapper process pools (pre-process, refine,
// post-process) with one ephemeral pool that runs three sequential phases:
//
//   1. Pre-flight  (per species):  MPAPrepareSpecies  -> staging
//   2. Refine      (per item):     MPAPreparePopulation + LoadGainRef as amortized init,
//                                  MPARefineAndSave per item accumulating into the
//                                  resident MPAPopulation and safe-saving a per-worker
//                                  progress partial
//   3. Post-flight (per species):  MPAFinishSpecies gathers every per-worker partial,
//                                  reconstructs, filters, commits
//
// Because workers are ephemeral, the refine phase mirrors the averaged-reconstruction
// map-reduce: each worker keeps accumulating across the items it claims and atomically
// rewrites its own progress after every item, so a crash loses at most the current item.
static partial class WorkerProcess
{
    [Command(WorkerCommandNames.MPAPrepareSpecies)]
    static void MPAPrepareSpecies(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        string StagingSave = (string)command.Content[1];

        Species S = Species.FromFile(Path);
        Console.Write($"Preparing {S.Name} for refinement... ");

        S.PrepareRefinementRequisites(true, DeviceID, StagingSave, null);

        // Pre-flight only writes denoised/filtered references to staging; the reference
        // projectors and reconstruction accumulators that PrepareRefinementRequisites
        // also allocates are unused here and the species is discarded. Free them so a
        // long-lived worker doesn't leak them into the refine phase (legacy reclaimed
        // this by killing the pre-process worker).
        S.FreeRefinementResources();

        Console.WriteLine("Done.");
    }

    [Command(WorkerCommandNames.MPAPreparePopulation)]
    static void MPAPreparePopulation(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        string StagingLoad = (string)command.Content[1];

        // Free the previous source's resident population before reloading. Previously the
        // refine worker process was killed between sources; with a long-lived worker the
        // species' projectors/reconstructions would otherwise accumulate on the device.
        if (MPAPopulation != null)
        {
            MPAPopulation.FreeRefinementResources();
            MPAPopulation = null;
        }

        MPAPopulation = new Population(Path);

        foreach (var species in MPAPopulation.Species)
        {
            Console.Write($"Preparing {species.Name} for refinement... ");

            species.PrepareRefinementRequisites(true, DeviceID, null, StagingLoad);

            Console.WriteLine("Done.");
        }
    }

    [Command(WorkerCommandNames.MPARefineAndSave)]
    static void MPARefineAndSave(NamedSerializableObject command)
    {
        if (MPAPopulation == null)
            throw new Exception("Population not prepared; MPAPreparePopulation must run in the task's init.");

        string Path = (string)command.Content[0];
        var Options = (ProcessingOptionsMPARefine)command.Content[1];
        var Source = (DataSource)command.Content[2];
        string TempDir = (string)command.Content[3];

        // Per-worker folder: both the refinement scratch and this worker's running
        // progress partial live here. Distinct per worker so partials never collide and
        // MPAFinishSpecies can gather them all.
        string WorkerDir = System.IO.Path.Combine(TempDir, $"worker_{WorkerId}");
        Directory.CreateDirectory(WorkerDir);

        Movie Item = Helper.PathToExtension(Path).ToLower() == ".tomostar"
            ? new TiltSeries(Path)
            : new Movie(Path);

        GPU.SetDevice(DeviceID);

        Item.PerformMultiParticleRefinement(WorkerDir, Options, MPAPopulation.Species.ToArray(), Source, GainRef, DefectMap, Console.WriteLine);

        Item.SaveMeta();

        GPU.CheckGPUExceptions();

        // Safe-save the running accumulation (half-map partials + updated poses) for this
        // worker. SaveRefinementProgress writes atomically (temp + rename per file), so a
        // crash mid-save leaves the previous complete partial intact and loses at most
        // this item — which the queue re-pends and another worker redoes.
        MPAPopulation.SaveRefinementProgress(WorkerDir);

        Console.WriteLine($"Finished refining {Item.Name}; saved progress for worker {WorkerId}");
    }

    [Command(WorkerCommandNames.MPAFinishSpecies)]
    static void MPAFinishSpecies(NamedSerializableObject command)
    {
        string Path = (string)command.Content[0];
        string StagingDirectory = (string)command.Content[1];
        string[] ProgressFolders = (string[])command.Content[2];

        Species S = Species.FromFile(Path);
        // singleGPU = 0 so the lone reconstruction accumulator lands at array index [0],
        // which GatherRefinementProgress/FinishRefinement read. The actual device binding
        // comes from GPU.SetDevice(DeviceID) (done per command) plus the DeviceID passed
        // to the gather/finish calls — matching the legacy post-process worker.
        S.PrepareRefinementRequisites(false, 0, null, StagingDirectory);
        S.GatherRefinementProgress(ProgressFolders, DeviceID);
        S.FinishRefinement(DeviceID);
        S.Commit();
        S.Save();

        Console.WriteLine($"Finished species {S.Name}: {S.GlobalResolution:F2} A");
    }
}
