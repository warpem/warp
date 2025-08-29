using Warp.Tools;

namespace Warp.Workers;

public partial class WorkerWrapper
{
    public void WaitAsyncTasks()
    {
        SendCommand(new NamedSerializableObject(nameof(WaitAsyncTasks)));
    }

    public void GcCollect()
    {
        SendCommand(new NamedSerializableObject(nameof(GcCollect)));
    }

    public void SetHeaderlessParams(int2 dims, long offset, string type)
    {
        SendCommand(new NamedSerializableObject(nameof(SetHeaderlessParams),
                                                dims,
                                                offset,
                                                type));
    }

    public void LoadGainRef(string path, bool flipX, bool flipY, bool transpose, string defectsPath)
    {
        SendCommand(new NamedSerializableObject(nameof(LoadGainRef),
                                                path,
                                                flipX,
                                                flipY,
                                                transpose,
                                                defectsPath));
    }

    public void LoadStack(string path, decimal scaleFactor, int eerGroupFrames, bool correctGain = true)
    {
        SendCommand(new NamedSerializableObject(nameof(LoadStack),
                                                path,
                                                scaleFactor,
                                                eerGroupFrames,
                                                correctGain));
    }

    public void LoadBoxNet(string path, int boxSize, int batchSize)
    {
        SendCommand(new NamedSerializableObject(nameof(LoadBoxNet),
                                                path,
                                                boxSize,
                                                batchSize));
    }

    public void DropBoxNet()
    {
        SendCommand(new NamedSerializableObject(nameof(DropBoxNet)));
    }
}