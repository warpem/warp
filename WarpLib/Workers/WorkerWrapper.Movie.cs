using Warp.Tools;

namespace Warp.Workers;

public partial class WorkerWrapper
{
    public void MovieProcessCTF(string path, ProcessingOptionsMovieCTF options)
    {
        SendCommand(new NamedSerializableObject(nameof(MovieProcessCTF),
                                                path,
                                                options));
    }

    public void MovieProcessMovement(string path, ProcessingOptionsMovieMovement options)
    {
        SendCommand(new NamedSerializableObject(nameof(MovieProcessMovement),
                                                path,
                                                options));
    }

    public void MoviePickBoxNet(string path, ProcessingOptionsBoxNet options)
    {
        SendCommand(new NamedSerializableObject(nameof(MoviePickBoxNet),
                                                path,
                                                options));
    }

    public void MovieExportMovie(string path, ProcessingOptionsMovieExport options)
    {
        SendCommand(new NamedSerializableObject(nameof(MovieExportMovie),
                                                path,
                                                options));
    }

    public void MovieCreateThumbnail(string path, int size, float range)
    {
        SendCommand(new NamedSerializableObject(nameof(MovieCreateThumbnail),
                                                path,
                                                size,
                                                range));
    }

    public void TardisSegmentMembranes2D(string[] paths, ProcessingOptionsTardisSegmentMembranes2D options)
    {
        SendCommand(new NamedSerializableObject(nameof(TardisSegmentMembranes2D),
                                                string.Join(';', paths),
                                                options));
    }

    public void MovieExportParticles(string path, ProcessingOptionsParticleExport options, float2[] coordinates)
    {
        SendCommand(new NamedSerializableObject(nameof(MovieExportParticles),
                                                path,
                                                options,
                                                coordinates));
    }
}