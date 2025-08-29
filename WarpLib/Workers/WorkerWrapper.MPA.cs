using Warp.Sociology;
using Warp.Tools;

namespace Warp.Workers;

public partial class WorkerWrapper
{
    public void MPAPrepareSpecies(string path, string stagingSave)
    {
        SendCommand(new NamedSerializableObject(nameof(MPAPrepareSpecies),
                                                path,
                                                stagingSave));
    }

    public void MPAPreparePopulation(string path, string stagingLoad)
    {
        SendCommand(new NamedSerializableObject(nameof(MPAPreparePopulation),
                                                path,
                                                stagingLoad));
    }

    public void MPARefine(string path, string workingDirectory, ProcessingOptionsMPARefine options, DataSource source)
    {
        SendCommand(new NamedSerializableObject(nameof(MPARefine),
                                                path,
                                                workingDirectory,
                                                options,
                                                source));
    }

    public void MPASaveProgress(string path)
    {
        SendCommand(new NamedSerializableObject(nameof(MPASaveProgress),
                                                path));
    }

    public void MPAFinishSpecies(string path, string stagingDirectory, string[] progressFolders)
    {
        SendCommand(new NamedSerializableObject(nameof(MPAFinishSpecies),
                                                path,
                                                stagingDirectory,
                                                progressFolders));
    }
}