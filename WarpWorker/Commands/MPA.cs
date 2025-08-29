using System;
using System.Linq;
using Warp;
using Warp.Sociology;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
{
    [Command(nameof(WorkerWrapper.MPAPrepareSpecies))]
    static void MPAPrepareSpecies(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        string StagingSave = (string)Command.Content[1];

        Species S = Species.FromFile(Path);
        Console.Write($"Preparing {S.Name} for refinement... ");

        S.PrepareRefinementRequisites(true, DeviceID, StagingSave, null);

        Console.WriteLine("Done.");
    }
    
    [Command(nameof(WorkerWrapper.MPAPreparePopulation))]
    static void MPAPreparePopulation(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        string StagingLoad = (string)Command.Content[1];

        MPAPopulation = new Population(Path);

        foreach (var species in MPAPopulation.Species)
        {
            Console.Write($"Preparing {species.Name} for refinement... ");

            species.PrepareRefinementRequisites(true, DeviceID, null, StagingLoad);

            Console.WriteLine("Done.");
        }
    }
    
    [Command(nameof(WorkerWrapper.MPARefine))]
    static void MPARefine(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        string WorkingDirectory = (string)Command.Content[1];
        ProcessingOptionsMPARefine Options = (ProcessingOptionsMPARefine)Command.Content[2];
        DataSource Source = (DataSource)Command.Content[3];

        Movie Item = null;

        if (Helper.PathToExtension(Path).ToLower() == ".tomostar")
            Item = new TiltSeries(Path);
        else
            Item = new Movie(Path);

        GPU.SetDevice(DeviceID);

        Item.PerformMultiParticleRefinement(WorkingDirectory, Options, MPAPopulation.Species.ToArray(), Source, GainRef, DefectMap, Console.WriteLine);

        Item.SaveMeta();

        GPU.CheckGPUExceptions();

        Console.WriteLine($"Finished refining {Item.Name}");
    }
    
    [Command(nameof(WorkerWrapper.MPASaveProgress))]
    static void MPASaveProgress(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];

        MPAPopulation.SaveRefinementProgress(Path);
    }
    
    [Command(nameof(WorkerWrapper.MPAFinishSpecies))]
    static void MPAFinishSpecies(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        string StagingDirectory = (string)Command.Content[1];
        string[] ProgressFolders = (string[])Command.Content[2];

        Species S = Species.FromFile(Path);
        S.PrepareRefinementRequisites(false, 0, null, StagingDirectory);
        S.GatherRefinementProgress(ProgressFolders, DeviceID);
        S.FinishRefinement(DeviceID);
        S.Commit();
        S.Save();
    }
}