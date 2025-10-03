using System;
using System.Threading;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    /// <summary>
    /// Worker class that prepares training batches in parallel and adds them to a queue
    /// </summary>
    public class BatchPreparationWorker
    {
        private readonly ProcessingContext context;
        private readonly Options options;
        private readonly ConcurrentTrainingQueue queue;
        private readonly int nMapsPerBatch;
        private readonly int mapSamples;
        private readonly int3 dim;
        private readonly int3 dim2;
        private readonly Random rand;

        public BatchPreparationWorker(ProcessingContext context, Options options, ConcurrentTrainingQueue queue, int seed)
        {
            this.context = context;
            this.options = options;
            this.queue = queue;
            this.nMapsPerBatch = Math.Min(8, context.MapPool.CurrentPoolSize);
            this.mapSamples = options.BatchSize;
            this.dim = context.TrainingDims;
            this.dim2 = dim * 2;
            this.rand = new Random(seed);
        }

        /// <summary>
        /// Prepares a single training batch and adds it to the queue
        /// </summary>
        public void PrepareBatch(int iterationNumber, CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested)
                return;

            GPU.SetDevice(options.GPUPreprocess);

            // Select random maps from current pool for this batch
            int[] shuffledMapIDs = Helper.RandomSubset(
                Helper.ArrayOfSequence(0, context.MapPool.CurrentPoolSize, 1),
                nMapsPerBatch,
                rand.Next(9999999));

            // Allocate temporary buffers for extraction
            Image[] extractedSource = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedCTF = Helper.ArrayOfFunction(i => new Image(new int3(dim2.X, dim2.Y, dim2.Z * mapSamples), true), nMapsPerBatch);

            // Allocate buffers for shuffled data
            Image[] extractedSourceRand = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedTargetRand = Helper.ArrayOfFunction(i => new Image(new int3(dim.X, dim.Y, dim.Z * mapSamples)), nMapsPerBatch);
            Image[] extractedCTFRand = Helper.ArrayOfFunction(i => new Image(new int3(dim2.X, dim2.Y, dim2.Z * mapSamples), true), nMapsPerBatch);

            ulong[] texture1 = new ulong[1];
            ulong[] texture2 = new ulong[1];
            ulong[] textureArray1 = new ulong[1];
            ulong[] textureArray2 = new ulong[1];

            try
            {
                // Extract training data from selected maps
                ExtractTrainingData(shuffledMapIDs, extractedSource, extractedTarget, extractedCTF,
                                   texture1, texture2, textureArray1, textureArray2);

                // Shuffle examples across maps
                ShuffleExamples(extractedSource, extractedTarget, extractedCTF,
                               extractedSourceRand, extractedTargetRand, extractedCTFRand);

                // Dispose temporary non-shuffled buffers
                foreach (var img in extractedSource) img.Dispose();
                foreach (var img in extractedTarget) img.Dispose();
                foreach (var img in extractedCTF) img.Dispose();

                // Create batch and add to queue
                var batch = new TrainingBatch
                {
                    ShuffledMapIDs = shuffledMapIDs,
                    ExtractedSourceRand = extractedSourceRand,
                    ExtractedTargetRand = extractedTargetRand,
                    ExtractedCTFRand = extractedCTFRand
                };

                queue.Enqueue(batch, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                // Clean up if cancelled
                foreach (var img in extractedSource) img?.Dispose();
                foreach (var img in extractedTarget) img?.Dispose();
                foreach (var img in extractedCTF) img?.Dispose();
                foreach (var img in extractedSourceRand) img?.Dispose();
                foreach (var img in extractedTargetRand) img?.Dispose();
                foreach (var img in extractedCTFRand) img?.Dispose();
                throw;
            }
            finally
            {
                // Textures are already cleaned up in the loop
            }
        }

        private void ExtractTrainingData(int[] shuffledMapIDs, Image[] extractedSource, Image[] extractedTarget, Image[] extractedCTF,
                                        ulong[] texture1, ulong[] texture2, ulong[] textureArray1, ulong[] textureArray2)
        {
            for (int m = 0; m < shuffledMapIDs.Length; m++)
            {
                int poolIndex = shuffledMapIDs[m];

                // Get maps from pool (thread-safe)
                context.MapPool.GetMap(poolIndex, out Image map1, out Image map2, out Image mapCTF);

                int3 dimsMap = map1.Dims;
                int3 margin = dim / 2;
                float3[] position = GenerateRandomPositions(dimsMap, margin);
                float3[] angle = GenerateRandomAngles();

                // Extract from map1
                lock (map1)  // Thread-safe access to shared map data
                {
                    GPU.CreateTexture3D(map1.GetDevice(Intent.Read), map1.Dims, texture1, textureArray1, true);

                    GPU.Rotate3DExtractAt(texture1[0], map1.Dims, extractedSource[m].GetDevice(Intent.Write),
                                         dim, Helper.ToInterleaved(angle), Helper.ToInterleaved(position), (uint)mapSamples);

                    GPU.DestroyTexture(texture1[0], textureArray1[0]);
                }

                // Extract from map2
                lock (map2)  // Thread-safe access to shared map data
                {
                    GPU.CreateTexture3D(map2.GetDevice(Intent.Read), map2.Dims, texture2, textureArray2, true);

                    GPU.Rotate3DExtractAt(texture2[0], map2.Dims, extractedTarget[m].GetDevice(Intent.Write),
                                         dim, Helper.ToInterleaved(angle), Helper.ToInterleaved(position), (uint)mapSamples);

                    GPU.DestroyTexture(texture2[0], textureArray2[0]);
                }

                // Copy CTF
                lock (mapCTF)  // Thread-safe access to shared CTF data
                {
                    for (int i = 0; i < mapSamples; i++)
                        GPU.CopyDeviceToDevice(mapCTF.GetDevice(Intent.Read),
                                              extractedCTF[m].GetDeviceSlice(i * dim2.Z, Intent.Write),
                                              mapCTF.ElementsReal);
                }
            }
        }

        private float3[] GenerateRandomPositions(int3 dimsMap, int3 margin)
        {
            if (context.BoundsMin == new int3(0))
                return Helper.ArrayOfFunction(i => new float3(
                    (float)rand.NextDouble() * (dimsMap.X - margin.X * 2) + margin.X,
                    (float)rand.NextDouble() * (dimsMap.Y - margin.Y * 2) + margin.Y,
                    (float)rand.NextDouble() * (dimsMap.Z - margin.Z * 2) + margin.Z), mapSamples);
            else
                return Helper.ArrayOfFunction(i => new float3(
                    (float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).X + context.BoundsMin.X,
                    (float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).Y + context.BoundsMin.Y,
                    (float)rand.NextDouble() * (context.BoundsMax - context.BoundsMin).Z + context.BoundsMin.Z), mapSamples);
        }

        private float3[] GenerateRandomAngles()
        {
            if (options.DontAugment)
                return Helper.ArrayOfFunction(i => new float3(
                    (float)Math.Round(rand.NextDouble()) * 0,
                    (float)Math.Round(rand.NextDouble()) * 0,
                    (float)Math.Round(rand.NextDouble()) * 0) * Helper.ToRad, mapSamples);
            else
                return Helper.ArrayOfFunction(i => new float3(
                    (float)rand.NextDouble() * 360,
                    (float)rand.NextDouble() * 360,
                    (float)rand.NextDouble() * 360) * Helper.ToRad, mapSamples);
        }

        private void ShuffleExamples(Image[] extractedSource, Image[] extractedTarget, Image[] extractedCTF,
                                    Image[] extractedSourceRand, Image[] extractedTargetRand, Image[] extractedCTFRand)
        {
            for (int b = 0; b < mapSamples; b++)
            {
                int[] order = Helper.RandomSubset(Helper.ArrayOfSequence(0, nMapsPerBatch, 1), nMapsPerBatch, rand.Next(9999999));
                for (int i = 0; i < order.Length; i++)
                {
                    GPU.CopyDeviceToDevice(extractedSource[i].GetDeviceSlice(b * dim.Z, Intent.Read),
                                          extractedSourceRand[order[i]].GetDeviceSlice(b * dim.Z, Intent.Write),
                                          dim.Elements());
                    GPU.CopyDeviceToDevice(extractedTarget[i].GetDeviceSlice(b * dim.Z, Intent.Read),
                                          extractedTargetRand[order[i]].GetDeviceSlice(b * dim.Z, Intent.Write),
                                          dim.Elements());
                    GPU.CopyDeviceToDevice(extractedCTF[i].GetDeviceSlice(b * dim2.Z, Intent.Read),
                                          extractedCTFRand[order[i]].GetDeviceSlice(b * dim2.Z, Intent.Write),
                                          (dim2.X / 2 + 1) * dim2.Y * dim2.Z);
                }
            }
        }
    }
}
