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

        var UsedTilts = TiltData.Where((d, i) => UseTilt[i]).ToArray();
        var UsedAngles = Angles.Where((d, i) => UseTilt[i]).ToArray();

        Image Stack = new Image(UsedTilts.Select(i => i.GetHost(Intent.Read)[0]).ToArray(), new int3(UsedTilts[0].Dims.X, UsedTilts[0].Dims.Y, UsedTilts.Length));
        Stack.WriteMRC(TiltStackPath, (float)options.BinnedPixelSizeMean, true);

        File.WriteAllLines(AngleFilePath, UsedAngles.Select(a => a.ToString("F2", CultureInfo.InvariantCulture)));
    }
}