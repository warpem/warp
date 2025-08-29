using System;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace WarpWorker;

static partial class WarpWorkerProcess
{
    static Image LoadAndPrepareGainReference(string path, bool flipX, bool flipY, bool transpose)
        {
            Image Gain = Image.FromFilePatient(10, 500,
                                               path,
                                               HeaderlessDims,
                                               (int)HeaderlessOffset,
                                               ImageFormatsHelper.StringToType(HeaderlessType));

            float Mean = MathHelper.Mean(Gain.GetHost(Intent.Read)[0]);
            Gain.TransformValues(v => v == 0 ? 1 : v / Mean);

            if (flipX)
                Gain = Gain.AsFlippedX();
            if (flipY)
                Gain = Gain.AsFlippedY();
            if (transpose)
                Gain = Gain.AsTransposed();

            return Gain;
        }

        static DefectModel LoadAndPrepareDefectMap(string path, bool flipX, bool flipY, bool transpose)
        {
            Image Defects = Image.FromFilePatient(10, 500,
                                                  path,
                                                  HeaderlessDims,
                                                  (int)HeaderlessOffset,
                                                  ImageFormatsHelper.StringToType(HeaderlessType));

            if (flipX)
                Defects = Defects.AsFlippedX();
            if (flipY)
                Defects = Defects.AsFlippedY();
            if (transpose)
                Defects = Defects.AsTransposed();

            DefectModel Model = new DefectModel(Defects, 4);
            Defects.Dispose();

            return Model;
        }

        static Image LoadAndPrepareStack(string path, decimal scaleFactor, bool correctGain, int maxThreads = 8)
        {
            Image stack = null;

            MapHeader header = MapHeader.ReadFromFilePatient(10, 500,
                                                             path,
                                                             HeaderlessDims,
                                                             (int)HeaderlessOffset,
                                                             ImageFormatsHelper.StringToType(HeaderlessType));

            string Extension = Helper.PathToExtension(path).ToLower();
            bool IsTiff = header.GetType() == typeof(HeaderTiff);
            bool IsEER = header.GetType() == typeof(HeaderEER);

            if (GainRef != null && correctGain)
                if (!IsEER)
                    if (header.Dimensions.X != GainRef.Dims.X || header.Dimensions.Y != GainRef.Dims.Y)
                        throw new Exception($"Gain reference dimensions ({GainRef.Dims.X}x{GainRef.Dims.Y}) do not match image ({header.Dimensions.X}x{header.Dimensions.Y}).");

            int EERSupersample = 1;
            if (GainRef != null && correctGain && IsEER)
            {
                if (header.Dimensions.X == GainRef.Dims.X)
                    EERSupersample = 1;
                else if (header.Dimensions.X * 2 == GainRef.Dims.X)
                    EERSupersample = 2;
                else if (header.Dimensions.X * 4 == GainRef.Dims.X)
                    EERSupersample = 3;
                else
                    throw new Exception("Invalid supersampling factor requested for EER based on gain reference dimensions");
            }
            int EERGroupFrames = 1;
            if (IsEER)
            {
                if (HeaderEER.GroupNFrames > 0)
                    EERGroupFrames = HeaderEER.GroupNFrames;
                else if (HeaderEER.GroupNFrames < 0)
                {
                    int NFrames = -HeaderEER.GroupNFrames;
                    EERGroupFrames = header.Dimensions.Z / NFrames;
                }

                header.Dimensions.Z /= EERGroupFrames;
            }

            HeaderEER.SuperResolution = EERSupersample;

            int2 SourceDims = new int2(header.Dimensions);
            if (IsEER)
                SourceDims *= 4;

            if (IsEER && GainRef != null && correctGain)
            {
                header.Dimensions.X = GainRef.Dims.X;
                header.Dimensions.Y = GainRef.Dims.Y;
            }

            int NThreads = (IsTiff || IsEER) ? maxThreads : 1;
            int GPUThreads = 2;

            int CurrentDevice = GPU.GetDevice();

            if (RawLayers == null || RawLayers.Length < NThreads || RawLayers[0].Length < SourceDims.Elements())
            {
                Console.WriteLine($"Allocating {NThreads} raw layers of size {SourceDims.Elements()} for {path} on device {CurrentDevice}");
                if (RawLayers != null)
                    ArrayPool<float>.ReturnMultiple(RawLayers);
                RawLayers = ArrayPool<float>.RentMultiple((int)SourceDims.Elements(), NThreads);
            }

            Image[] GPULayers = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SourceDims)), GPUThreads);
            Image[] GPULayers2 = DefectMap != null ? Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SourceDims)), GPUThreads) : null;

            if (scaleFactor == 1M && !IsEER)
            {
                if (OriginalStack == null || OriginalStack.Dims != header.Dimensions)
                {
                    OriginalStack?.Dispose();
                    OriginalStack = new Image(header.Dimensions);
                    Console.WriteLine($"Allocating original stack of size {header.Dimensions} for {path} on device {CurrentDevice}");
                }

                stack = OriginalStack;
                float[][] OriginalStackData = stack.GetHost(Intent.Write);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, header.Dimensions.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(10, 500, path, z, true, RawLayers[threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(10, 500, path, z * EERGroupFrames, Math.Min(((HeaderEER)header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), EERSupersample, RawLayers[threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(10, 500,
                                                     path,
                                                     HeaderlessDims,
                                                     (int)HeaderlessOffset,
                                                     ImageFormatsHelper.StringToType(HeaderlessType),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), header.Dimensions.ElementsSlice());

                        if (GainRef != null && correctGain)
                        {
                            //if (IsEER)
                            //    GPULayers[GPUThreadID].DivideSlices(GainRef);
                            //else
                                GPULayers[GPUThreadID].MultiplySlices(GainRef); // EER .gain is now multiplicative??
                        }

                        if (DefectMap != null)
                        {
                            GPU.CopyDeviceToDevice(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                                   GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                                                   header.Dimensions.Elements());
                            DefectMap.Correct(GPULayers2[GPUThreadID], GPULayers[GPUThreadID]);
                        }

                        //GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                        //         GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                        //         20f,
                        //         new int2(header.Dimensions),
                        //         1);

                        GPU.CopyDeviceToHost(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                             OriginalStackData[z],
                                             header.Dimensions.ElementsSlice());
                    }

                }, null);
            }
            else
            {
                int3 ScaledDims = new int3((int)Math.Round(header.Dimensions.X * scaleFactor) / 2 * 2,
                                            (int)Math.Round(header.Dimensions.Y * scaleFactor) / 2 * 2,
                                            header.Dimensions.Z);

                if (OriginalStack == null || OriginalStack.Dims != ScaledDims)
                {
                    OriginalStack?.Dispose();
                    OriginalStack = new Image(ScaledDims);
                }

                stack = OriginalStack;
                float[][] OriginalStackData = stack.GetHost(Intent.Write);

                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SourceDims), 1), GPUThreads);
                int[] PlanBack = Helper.ArrayOfFunction(i => GPU.CreateIFFTPlan(ScaledDims.Slice(), 1), GPUThreads);

                Image[] GPULayersInputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SourceDims), true, true), GPUThreads);
                Image[] GPULayersOutputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice(), true, true), GPUThreads);

                Image[] GPULayersScaled = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice()), GPUThreads);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPU(0, ScaledDims.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    if (IsTiff)
                        TiffNative.ReadTIFFPatient(10, 500, path, z, true, RawLayers[threadID]);
                    else if (IsEER)
                        EERNative.ReadEERPatient(10, 500, path, z * EERGroupFrames, Math.Min(((HeaderEER)header).DimensionsUngrouped.Z, (z + 1) * EERGroupFrames), 3, RawLayers[threadID]);
                    else
                        IOHelper.ReadMapFloatPatient(10, 500,
                                                     path,
                                                     HeaderlessDims,
                                                     (int)HeaderlessOffset,
                                                     ImageFormatsHelper.StringToType(HeaderlessType),
                                                     new[] { z },
                                                     null,
                                                     new[] { RawLayers[threadID] });

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        GPU.CopyHostToDevice(RawLayers[threadID], GPULayers[GPUThreadID].GetDevice(Intent.Write), SourceDims.Elements());

                        if (GainRef != null && correctGain)
                        {
                            //if (IsEER)
                            //    GPULayers[GPUThreadID].DivideSlices(GainRef);
                            //else
                            if (!IsEER)
                                GPULayers[GPUThreadID].MultiplySlices(GainRef);
                        }

                        if (DefectMap != null && !IsEER)
                        {
                            GPU.CopyDeviceToDevice(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                                   GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                                                   SourceDims.Elements());
                            DefectMap.Correct(GPULayers2[GPUThreadID], GPULayers[GPUThreadID]);
                        }

                        //GPU.Xray(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                        //         GPULayers2[GPUThreadID].GetDevice(Intent.Write),
                        //         20f,
                        //         new int2(header.Dimensions),
                        //         1);

                        GPU.Scale(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                  GPULayersScaled[GPUThreadID].GetDevice(Intent.Write),
                                  new int3(SourceDims),
                                  ScaledDims.Slice(),
                                  1,
                                  PlanForw[GPUThreadID],
                                  PlanBack[GPUThreadID],
                                  GPULayersInputFT[GPUThreadID].GetDevice(Intent.Write),
                                  GPULayersOutputFT[GPUThreadID].GetDevice(Intent.Write));

                        GPU.CopyDeviceToHost(GPULayersScaled[GPUThreadID].GetDevice(Intent.Read),
                                             OriginalStackData[z],
                                             ScaledDims.ElementsSlice());
                    }

                }, null);

                for (int i = 0; i < GPUThreads; i++)
                {
                    GPU.DestroyFFTPlan(PlanForw[i]);
                    GPU.DestroyFFTPlan(PlanBack[i]);
                    GPULayersInputFT[i].Dispose();
                    GPULayersOutputFT[i].Dispose();
                    GPULayersScaled[i].Dispose();
                }
            }

            foreach (var layer in GPULayers)
                layer.Dispose();
            if (GPULayers2 != null)
                foreach (var layer in GPULayers2)
                    layer.Dispose();

            return stack;
        }
}