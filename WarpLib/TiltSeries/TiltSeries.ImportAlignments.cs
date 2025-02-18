using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public void ImportAlignments(ProcessingOptionsTomoImportAlignments options)
    {
        string ResultsDir = string.IsNullOrEmpty(options.OverrideResultsDir) ? TiltStackDir : options.OverrideResultsDir;

        UseTilt = Helper.ArrayOfConstant(true, NTilts);

        #region Excluded tilts

        string CutviewsPath1 = System.IO.Path.Combine(ResultsDir, RootName + "_cutviews0.rawtlt");
        string CutviewsPath2 = System.IO.Path.Combine(ResultsDir, "../", RootName + "_cutviews0.rawtlt");
        string CutviewsPath3 = System.IO.Path.Combine(ResultsDir, RootName + "_Imod", RootName + "_cutviews0.rawtlt");
        string CutviewsPath = null;
        try
        {
            CutviewsPath = (new[] { CutviewsPath3, CutviewsPath1, CutviewsPath2 }).First(s => File.Exists(s));
        }
        catch
        {
        }

        if (CutviewsPath != null)
        {
            List<float> CutAngles = File.ReadAllLines(CutviewsPath).Where(l => !string.IsNullOrEmpty(l)).Select(l => float.Parse(l, CultureInfo.InvariantCulture)).ToList();

            UseTilt = UseTilt.Select((v, t) => !CutAngles.Any(a => Math.Abs(a - Angles[t]) < 0.2)).ToArray();
        }

        #endregion

        int NValid = UseTilt.Count(v => v);

        #region Transforms

        // .xf
        {
            string[] Directories = { "", "../", $"{RootName}_Imod", $"{RootName}_aligned_Imod", $"../{RootName}_Imod" };
            string[] FileNames =
            {
                $"{RootName}.xf",
                $"{RootName.Replace(".mrc", "")}.xf",
                $"{RootName}_st.xf",
                $"{RootName.Replace(".mrc", "")}_st.xf"
            };
            string[] XfPaths = new string[Directories.Length * FileNames.Length];
            int idx;
            for (int i = 0; i < Directories.Length; i++)
            for (int j = 0; j < FileNames.Length; j++)
            {
                idx = i * FileNames.Length + j;
                XfPaths[idx] = System.IO.Path.GetFullPath(System.IO.Path.Combine(ResultsDir, Directories[i], FileNames[j]));
            }

            if (Helper.IsDebug)
            {
                Console.WriteLine("Possible XF file paths:");
                foreach (string path in XfPaths)
                    Console.WriteLine($"{path}");
            }

            string XfPath = null;
            try
            {
                XfPath = XfPaths.First(s => File.Exists(s));
                if (Helper.IsDebug)
                    Console.WriteLine($"\nImporting 2D transforms from {XfPath}");
            }
            catch
            {
            }

            if (XfPath == null)
                throw new Exception($"Could not find {RootName}.xf");

            string[] Lines = File.ReadAllLines(XfPath).Where(l => !string.IsNullOrEmpty(l)).ToArray();
            if (Lines.Length != NValid)
                throw new Exception($"{NValid} active tilts in series, but {Lines.Length} lines in {XfPath}");

            for (int t = 0, iline = 0; t < NTilts; t++)
            {
                if (!UseTilt[t])
                    continue;

                string Line = Lines[iline];

                string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                float2 VecX = new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                    float.Parse(Parts[2], CultureInfo.InvariantCulture));
                float2 VecY = new float2(float.Parse(Parts[1], CultureInfo.InvariantCulture),
                    float.Parse(Parts[3], CultureInfo.InvariantCulture));

                Matrix3 Rotation = new Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1);
                float3 Euler = Matrix3.EulerFromMatrix(Rotation);

                TiltAxisAngles[t] = Euler.Z * Helper.ToDeg;

                //SortedAngle[i].Shift += VecX * float.Parse(Parts[4], CultureInfo.InvariantCulture) + VecY * float.Parse(Parts[5], CultureInfo.InvariantCulture);
                float3 Shift = new float3(-float.Parse(Parts[4], CultureInfo.InvariantCulture), -float.Parse(Parts[5], CultureInfo.InvariantCulture), 0);
                Shift = Rotation.Transposed() * Shift;

                Shift *= (float)options.BinnedPixelSizeMean;

                TiltAxisOffsetX[t] = Shift.X;
                TiltAxisOffsetY[t] = Shift.Y;

                iline++;
            }
        }

        // .tlt
        {
            string[] Directories = { "", "../", $"{RootName}_Imod", $"{RootName}_aligned_Imod", $"../{RootName}_Imod" };
            string[] FileNames =
            {
                $"{RootName}.tlt",
                $"{RootName.Replace(".mrc", "")}.tlt",
                $"{RootName}_st.tlt",
                $"{RootName.Replace(".mrc", "")}_st.tlt"
            };
            string[] TltPaths = new string[Directories.Length * FileNames.Length];
            int idx;
            for (int i = 0; i < Directories.Length; i++)
            for (int j = 0; j < FileNames.Length; j++)
            {
                idx = i * FileNames.Length + j;
                TltPaths[idx] = System.IO.Path.GetFullPath(System.IO.Path.Combine(ResultsDir, Directories[i], FileNames[j]));
            }

            if (Helper.IsDebug)
            {
                Console.WriteLine("Possible TLT file paths:");
                foreach (string path in TltPaths)
                    Console.WriteLine($"{path}");
            }

            string TltPath = null;
            try
            {
                TltPath = TltPaths.First(s => File.Exists(s));
                if (Helper.IsDebug)
                    Console.WriteLine($"\nImporting tilt angles from {TltPath}");
            }
            catch
            {
            }

            if (TltPath == null)
                throw new Exception($"Could not find {RootName}.tlt");

            string[] Lines = File.ReadAllLines(TltPath).Where(l => !string.IsNullOrEmpty(l)).ToArray();

            if (Lines.Length == NValid)
            {
                float[] ParsedTiltAngles = new float[NTilts];
                for (int t = 0; t < NTilts; t++)
                {
                    string Line = Lines[t];
                    ParsedTiltAngles[t] = float.Parse(Line, CultureInfo.InvariantCulture);
                }

                if (ParsedTiltAngles.All(angle => angle == 0))
                    throw new Exception($"all tilt angles are zero in {TltPath}");
                else
                {
                    for (int t = 0; t < NTilts; t++)
                    {
                        if (!UseTilt[t])
                            continue;
                        Angles[t] = ParsedTiltAngles[t];
                    }
                }
            }
        }

        #endregion

        #region FOV fraction

        if (options.MinFOV > 0)
        {
            VolumeDimensionsPhysical = new float3((float)options.DimensionsPhysical.X, (float)options.DimensionsPhysical.Y, 1);
            LoadMovieSizes();

            int NSteps = 100;
            var Positions = new float3[NSteps * NSteps];
            for (int y = 0; y < NSteps; y++)
            {
                float yy = VolumeDimensionsPhysical.Y * y / (NSteps - 1);
                for (int x = 0; x < NSteps; x++)
                {
                    float xx = VolumeDimensionsPhysical.X * x / (NSteps - 1);
                    Positions[y * NSteps + x] = new float3(xx, yy, 0);
                }
            }

            float[] FOVFractions = new float[NTilts];

            for (int t = 0; t < NTilts; t++)
            {
                if (!UseTilt[t])
                    continue;

                float3[] ImagePositions = GetPositionsInOneTilt(Positions, t);
                int NContained = 0;
                foreach (var pos in ImagePositions)
                    if (pos.X >= 0 && pos.Y >= 0 &&
                        pos.X <= ImageDimensionsPhysical.X - 1 &&
                        pos.Y <= ImageDimensionsPhysical.Y - 1)
                        NContained++;

                FOVFractions[t] = (float)NContained / ImagePositions.Length;
            }

            float FractionAt0 = Helper.ArrayOfFunction(i => FOVFractions[IndicesSortedAbsoluteAngle[i]], Math.Min(5, FOVFractions.Length)).Max();
            if (FractionAt0 > 0)
                FOVFractions = FOVFractions.Select(v => v / FractionAt0).ToArray();

            UseTilt = UseTilt.Select((v, t) => v && FOVFractions[t] >= (float)options.MinFOV).ToArray();
        }

        #endregion
    }
}