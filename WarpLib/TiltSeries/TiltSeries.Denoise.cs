using System;
using System.IO;
using Warp.Tools;
using IOPath = System.IO.Path;

namespace Warp;

public partial class TiltSeries
{
    public void Denoise(ProcessingOptionsTomoDenoise options, NoiseNet3DTorch model)
    {
        Directory.CreateDirectory(ReconstructionDenoisedDir);

        // Load reconstruction
        string reconstructionPath = IOPath.Combine(ProcessingDirectoryName, 
                                                   ToReconstructionTomogramPath(RootName, 
                                                                                options.PixelSize));
        if (!File.Exists(reconstructionPath))
            throw new Exception($"Reconstruction not found at {reconstructionPath}");

        Image volume = Image.FromFile(reconstructionPath);

        // Calculate mean/std from center region
        Image volumeCenter = volume.AsPadded(volume.Dims / 2);
        float2 meanStd = MathHelper.MeanAndStd(volumeCenter.GetHost(Intent.Read)[0]);
        volumeCenter.Dispose();

        // Normalize (same as Noise2Map inference)
        volume.TransformValues(v => Math.Max(-30, Math.Min(30, (v - meanStd.X) / meanStd.Y)));

        // Denoise in-place
        NoiseNet3DTorch.Denoise(volume, [ model ]);

        // Denormalize
        volume.TransformValues(v => v * meanStd.Y + meanStd.X);

        // Set pixel size
        volume.PixelSize = (float)options.PixelSize;

        // Save denoised volume
        string denoisedPath = IOPath.Combine(ProcessingDirectoryName, 
                                             ToReconstructionDenoisedTomogramPath(RootName, 
                                                                                  options.PixelSize));
        volume.WriteMRC(denoisedPath, (float)options.PixelSize, true);

        volume.Dispose();
    }
}

[Serializable]
public class ProcessingOptionsTomoDenoise : TomoProcessingOptionsBase
{
    [WarpSerializable] public decimal PixelSize { get; set; }
}