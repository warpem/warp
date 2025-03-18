using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public void StackTilts(ProcessingOptionsTomoStack options)
    {
        Directory.CreateDirectory(TiltStackDir);

        Movie[] TiltMovies;
        Image[] TiltData;
        Image[] TiltMasks = null;
        LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
        if (options.ApplyMask)
            LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            if (options.ApplyMask)
            {
                EraseDirt(TiltData[z], TiltMasks[z]);
                TiltMasks[z]?.FreeDevice();
            }

            TiltData[z].FreeDevice();
        }

        var UsedIndices = Enumerable.Range(0, NTilts).Where(i => UseTilt[i]).ToArray();
        var UsedTilts = UsedIndices.Select(i => TiltData[i]).ToArray();
        var UsedAngles = UsedIndices.Select(i => Angles[i]).ToArray();

        using (Image stack = new Image(UsedTilts.Select(i => i.GetHost(Intent.Read)[0]).ToArray(), new int3(UsedTilts[0].Dims.X, UsedTilts[0].Dims.Y, UsedTilts.Length)))
            stack.WriteMRC(TiltStackPath, (float)options.BinnedPixelSizeMean, true);

        File.WriteAllLines(AngleFilePath, UsedAngles.Select(a => a.ToString("F2", CultureInfo.InvariantCulture)));
        
        #region Make thumbnails

        Directory.CreateDirectory(TiltStackDir);

        foreach (var t in UsedIndices)
            using (Image center = TiltData[t].AsPadded(new int2(TiltData[t].Dims) / 2))
            {
                float2 MeanStd = MathHelper.MedianAndStd(center.GetHost(Intent.Read)[0]);
                float Min = MeanStd.X;
                float Range = 0.5f / (MeanStd.Y * 3);

                TiltData[t].TransformValues(v => ((v - Min) * Range + 0.5f) * 255);
                TiltData[t].WritePNG(TiltStackThumbnailPath(TiltMoviePaths[t]));
            }
        
        #endregion
    }
}

[Serializable]
public class ProcessingOptionsTomoStack : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool ApplyMask { get; set; }
    [WarpSerializable] public bool CreateThumbnails { get; set; }
}