using System;
using System.Collections.Generic;
using Warp;
using Warp.Tools;

namespace Warp
{
    public partial class Movie
    {
        public void TraceMembranes()
        {
        }

        public void SubtractMembranes()
        {
        }
    }

    [Serializable]
    public class ProcessingOptionsTraceMembranes : ProcessingOptionsBase
    {
        // Placeholder properties for future use
        [WarpSerializable] public int MinComponentSize { get; set; } = 20; // px
        [WarpSerializable] public float BandpassLow { get; set; } = 0.002f;
        [WarpSerializable] public float BandpassHigh { get; set; } = 0.05f;
        [WarpSerializable] public bool EnableSplineRefinement { get; set; } = true;
    }
}

public static class TraceMembranesHelper
{
    public static Image ComputeDistanceMap(Image binaryMask)
    {
        int width = binaryMask.Dims.X;
        int height = binaryMask.Dims.Y;

        Image maskInv = new Image(IntPtr.Zero, new int3(width, height, 1));
        Image distanceMap = new Image(IntPtr.Zero, new int3(width, height, 1));

        maskInv.Fill(1f);
        maskInv.Subtract(binaryMask);

        GPU.DistanceMapExact(maskInv.GetDevice(Intent.Read), distanceMap.GetDevice(Intent.Write), new int3(width, height, 1), 20);

        maskInv.Dispose();
        return distanceMap;
    }

    public static Image DetectRidges(Image distanceMap, float minDistanceThreshold)
    {
        int width = distanceMap.Dims.X;
        int height = distanceMap.Dims.Y;
        Image ridgeMask = new Image(distanceMap.Dims);

        float[] distanceData = distanceMap.GetHost(Intent.Read)[0];
        float[] ridgeData = ridgeMask.GetHost(Intent.ReadWrite)[0];

        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = y * width + x;
                float centerValue = distanceData[index];

                if (centerValue < minDistanceThreshold)
                    continue;

                // Check local maxima in principal directions
                bool isRidge =
                    (distanceData[index - 1] < centerValue && distanceData[index + 1] < centerValue) || // Horizontal
                    (distanceData[index - width] < centerValue && distanceData[index + width] < centerValue) || // Vertical
                    (distanceData[index - width - 1] < centerValue && distanceData[index + width + 1] < centerValue) || // Diagonal 1
                    (distanceData[index - width + 1] < centerValue && distanceData[index + width - 1] < centerValue); // Diagonal 2

                if (isRidge)
                    ridgeData[index] = 1.0f;
            }
        }

        // Process junctions: Keep only the two longest branches
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = y * width + x;
                if (ridgeData[index] != 1.0f)
                    continue;

                int[] neighbors =
                {
                    index - 1, index + 1, index - width, index + width,
                    index - width - 1, index - width + 1, index + width - 1, index + width + 1
                };

                List<int> connected = new List<int>();
                foreach (int neighbor in neighbors)
                {
                    if (ridgeData[neighbor] == 1.0f)
                        connected.Add(neighbor);
                }

                if (connected.Count <= 2)
                    continue;

                // Trace each branch and determine its length
                List<(int start, int length)> branches = new List<(int, int)>();
                foreach (int start in connected)
                {
                    HashSet<int> visited = new HashSet<int>();
                    int length = TraceBranch(start, ridgeData, width, height, visited);
                    branches.Add((start, length));
                }

                // Keep only the two longest branches
                branches.Sort((a, b) => b.length.CompareTo(a.length));
                HashSet<int> keep = new HashSet<int> { branches[0].start, branches[1].start };

                foreach (int start in connected)
                {
                    if (!keep.Contains(start))
                        ridgeData[start] = 0.0f;
                }
            }
        }

        return ridgeMask;
    }

    private static int TraceBranch(int start, float[] ridgeData, int width, int height, HashSet<int> visited)
    {
        Queue<int> queue = new Queue<int>();
        queue.Enqueue(start);
        int length = 0;

        while(queue.Count > 0)
        {
            int index = queue.Dequeue();
            if (visited.Contains(index) || ridgeData[index] != 1.0f)
                continue;

            visited.Add(index);
            length++;

            int[] neighbors =
            {
                index - 1, index + 1, index - width, index + width,
                index - width - 1, index - width + 1, index + width - 1, index + width + 1
            };

            foreach (int neighbor in neighbors)
            {
                if (!visited.Contains(neighbor) && ridgeData[neighbor] == 1.0f)
                    queue.Enqueue(neighbor);
            }
        }

        return length;
    }
    
    public static List<List<(int x, int y)>> FindConnectedComponents(Image ridgeMask)
    {
        int width = ridgeMask.Dims.X;
        int height = ridgeMask.Dims.Y;
        float[] data = ridgeMask.GetHost(Intent.Read)[0];

        bool[,] visited = new bool[height, width];
        List<List<(int x, int y)>> components = new();
        
        // Neighbor offsets for 8-connectivity
        int[] dx = [-1, 0, 1, -1, 1, -1, 0, 1];
        int[] dy = [-1, -1, -1, 0, 0, 1, 1, 1];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (data[y * width + x] == 1 && !visited[y, x])
                {
                    List<(int x, int y)> component = new();
                    Queue<(int x, int y)> queue = new();

                    queue.Enqueue((x, y));
                    visited[y, x] = true;

                    while (queue.Count > 0)
                    {
                        (int cx, int cy) = queue.Dequeue();
                        component.Add((cx, cy));

                        for (int i = 0; i < 8; i++)
                        {
                            int nx = cx + dx[i];
                            int ny = cy + dy[i];

                            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                                data[ny * width + nx] == 1 && !visited[ny, nx])
                            {
                                queue.Enqueue((nx, ny));
                                visited[ny, nx] = true;
                            }
                        }
                    }

                    components.Add(component);
                }
            }
        }

        return components;
    }
}