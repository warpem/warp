using Warp.Tools;

namespace Warp.Workers;

public partial class WorkerWrapper
{
    public void TomoStack(string path, ProcessingOptionsTomoStack options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoStack),
                                                path,
                                                options));
    }

    public void TomoAretomo(string path, ProcessingOptionsTomoAretomo options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoAretomo),
                                                path,
                                                options));
    }

    public void TomoEtomoPatchTrack(string path, ProcessingOptionsTomoEtomoPatch options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoEtomoPatchTrack),
                                                path,
                                                options));
    }

    public void TomoEtomoFiducials(string path, ProcessingOptionsTomoEtomoFiducials options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoEtomoFiducials),
                                                path,
                                                options));
    }

    public void TomoProcessCTF(string path, ProcessingOptionsMovieCTF options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoProcessCTF),
                                                path,
                                                options));
    }

    public void TomoAlignLocallyWithoutReferences(string path, ProcessingOptionsTomoFullReconstruction options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoAlignLocallyWithoutReferences),
                                                path,
                                                options));
    }

    public void TomoReconstruct(string path, ProcessingOptionsTomoFullReconstruction options)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoReconstruct),
                                                path,
                                                options));
    }

    public void TomoMatch(string path, ProcessingOptionsTomoFullMatch options, string templatePath)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoMatch),
                                                path,
                                                options,
                                                templatePath));
    }

    public void TomoExportParticleSubtomos(string path, ProcessingOptionsTomoSubReconstruction options, float3[] coordinates, float3[] angles)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoExportParticleSubtomos),
                                                path,
                                                options,
                                                coordinates,
                                                angles));
    }

    public void TomoExportParticleSeries(string path, ProcessingOptionsTomoSubReconstruction options, float3[] coordinates, float3[] angles, string pathsRelativeTo, string pathTableOut)
    {
        SendCommand(new NamedSerializableObject(nameof(TomoExportParticleSeries),
                                                path,
                                                options,
                                                coordinates,
                                                angles,
                                                pathsRelativeTo,
                                                pathTableOut));
    }
}