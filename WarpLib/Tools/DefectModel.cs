using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
    public class DefectModel : IDisposable
    {
        public readonly int2 Dims;
        public readonly int NDefects = 0;
        IntPtr DefectLocations;
        IntPtr DefectNeighbors;

        public DefectModel(Image defectImage, int minRadius)
        {
            Dims = new int2(defectImage.Dims);
            int MinSamples = (int)Math.Round(minRadius * minRadius * Math.PI) - 1;

            List<int> AllLocationsList = new List<int>();
            List<int[]> AllNeighborsList = new List<int[]>();

            float[] DefectData = defectImage.GetHost(Intent.Read)[0];
            List<int> CurNeighbors = new List<int>(MinSamples);

            for (int i = 0; i < DefectData.Length; i++)
            {
                if (DefectData[i] == 0)
                    continue;

                int CurRadius = minRadius;
                int X = i % Dims.X;
                int Y = i / Dims.X;

                #region Find suitable neighbor positions within at least minRadius

                while (CurRadius < Dims.X / 2)
                {
                    CurNeighbors.Clear();

                    for (int dy = -CurRadius; dy < CurRadius; dy++)
                    {
                        int YY = Y + dy;
                        if (YY < 0 || YY >= Dims.Y)
                            continue;

                        for (int dx = -CurRadius; dx <= CurRadius; dx++)
                        {
                            int XX = X + dx;
                            if (XX < 0 || XX >= Dims.X)
                                continue;

                            int R2 = dx * dx + dy * dy;
                            if (R2 > CurRadius * CurRadius)
                                continue;

                            if (DefectData[YY * Dims.X + XX] > 0)
                                continue;

                            CurNeighbors.Add(YY * Dims.X + XX);
                        }
                    }

                    if (CurNeighbors.Count >= MinSamples)
                        break;

                    CurRadius++;
                }

                #endregion

                AllLocationsList.Add(i);
                AllNeighborsList.Add(CurNeighbors.ToArray());
            }

            NDefects = AllLocationsList.Count;
            int[] AllLocations = new int[NDefects * 3];
            int[] AllNeighbors = new int[AllNeighborsList.Select(a => a.Length).Sum()];
            int NeighborStart = 0;

            for (int i = 0; i < NDefects; i++)
            {
                for (int n = 0; n < AllNeighborsList[i].Length; n++)
                    AllNeighbors[NeighborStart + n] = AllNeighborsList[i][n];

                AllLocations[i * 3 + 0] = AllLocationsList[i];
                AllLocations[i * 3 + 1] = NeighborStart;
                AllLocations[i * 3 + 2] = NeighborStart + AllNeighborsList[i].Length;

                NeighborStart += AllNeighborsList[i].Length;
            }

            if (NDefects > 0)
            {
                DefectLocations = GPU.MallocDeviceFromHostInt(AllLocations, AllLocations.Length);
                DefectNeighbors = GPU.MallocDeviceFromHostInt(AllNeighbors, AllNeighbors.Length);
            }
        }

        public void Correct(Image image, Image corrected)
        {
            if (Dims != new int2(image.Dims))
                throw new Exception("Image dimensions don't match those of the defect map");

            if (NDefects > 0)
                for (int z = 0; z < image.Dims.Z; z++)
                    GPU.CorrectDefects(image.GetDeviceSlice(z, Intent.Read),
                                        corrected.GetDeviceSlice(z, Intent.Write),
                                        DefectLocations,
                                        DefectNeighbors,
                                        NDefects);
        }

        public void Dispose()
        {
            if (DefectLocations != IntPtr.Zero)
            {
                GPU.FreeDevice(DefectLocations);
                DefectLocations = IntPtr.Zero;
            }
            if (DefectNeighbors != IntPtr.Zero)
            {
                GPU.FreeDevice(DefectNeighbors);
                DefectNeighbors = IntPtr.Zero;
            }
        }

        ~DefectModel()
        {
            Dispose();
        }
    }
}
