using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace WarpWorker2;

static partial class WorkerProcess
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

    /// <summary>
    /// A stack that fits comfortably stays on the GPU for its task: LoadStack leaves the corrected frames there and
    /// the movie commands read them in place instead of uploading the host copy for every step. The host copy is
    /// only made if something asks for it, and the device copy is dropped when the task ends (ReleaseResidentStack).
    /// It must be at most an eighth of the device and, as other workers may share the GPU, leave the rest of a movie
    /// task's working set free now. WARP_RESIDENT_STACK=0 keeps stacks on the host; =1 skips the free-memory check.
    /// </summary>
    static bool KeepStackOnDevice(int3 dims)
    {
        string forced = Environment.GetEnvironmentVariable("WARP_RESIDENT_STACK");
        long bytes = dims.Elements() * sizeof(float);
        if (forced == "0" || bytes > (GPU.GetTotalMemory(DeviceID) << 20) / 8) return false;
        return forced == "1" || bytes + ResidentStackReserveBytes <= GPU.GetFreeMemory(DeviceID) << 20;
    }

    /// <summary>
    /// The stack for a LoadStack of these dimensions, reusing the last one if it fits, and whether it stays on the GPU.
    /// A stack kept there is allocated there only: Image(dims) also allocates host arrays, which such a stack never
    /// fills and which turn into resident memory once the garbage collector reuses pages for them.
    /// </summary>
    static Image PrepareStack(int3 dims)
    {
        StackResident = KeepStackOnDevice(dims);
        if (OriginalStack == null || OriginalStack.Dims != dims)
        {
            OriginalStack?.Dispose();
            OriginalStack = StackResident ? new Image(IntPtr.Zero, dims) : new Image(dims);
            Console.WriteLine($"Allocating a stack of {dims} {(StackResident ? "on the GPU" : "on the host")}");
        }
        return OriginalStack;
    }

    /// <summary>A movie task's GPU memory besides its stack (motion patches, CTF spectra, the denoiser), with a margin.</summary>
    const long ResidentStackReserveBytes = 6L << 30;

    /// <summary>
    /// Drops a resident stack's device copy (and the stack) after its task. Every task that uses a stack starts with
    /// LoadStack, so nothing is lost, and a worker that is idle, or waiting while another holds the GPU for training,
    /// holds no movie there. Dispose, not FreeDevice: the device copy is the only current one and is not copied back.
    /// Host stacks stay as before.
    /// </summary>
    static void ReleaseResidentStack()
    {
        if (!StackResident || OriginalStack == null) return;
        OriginalStack.Dispose();
        OriginalStack = null; OriginalStackOwner = ""; OriginalStackSource = null; StackResident = false;
    }

    /// <summary>The movie file of the current LoadStack, when it was read into memory (TIFF or EER).</summary>
    static FileInMemory MovieFile = new();

    /// <summary>
    /// TIFF and EER movies are read into memory with a few large reads and decoded from there. Opened in place, every
    /// frame read walks the file's image directories, one small read at a scattered offset each, which a network file
    /// system serves one round trip at a time when the file is not cached. Any failure falls back to the file reader.
    /// </summary>
    static MapHeader TryReadMovieIntoMemory(string path)
    {
        if (!FileInMemory.Enabled || !FileInMemory.IsMovie(path)) return null;
        try
        {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            if (MoviePrefetch.TryTake(path, ref MovieFile))
                Console.WriteLine($"Movie file was prefetched, {timer.Elapsed.TotalMilliseconds:F0} ms waiting");
            else
            {
                if (!FileInMemory.Fits(path)) return null;
                MovieFile.Read(path);
                Console.WriteLine($"Read movie file into memory in {timer.Elapsed.TotalMilliseconds:F0} ms");
            }
            using var stream = MovieFile.AsStream();
            var header = MapHeader.ReadFromFile(path, HeaderlessDims, HeaderlessOffset, ImageFormatsHelper.StringToType(HeaderlessType), stream);
            return (header is HeaderTiff || header is HeaderEER) && header.Dimensions.Elements() > 0 ? header : null;
        }
        catch (Exception)
        {
            return null; // The file reader reports the problem, or reads what this could not.
        }
    }

    /// <summary>Frames a streamed load hands over at the same time, each from its own thread (see LoadAndPrepareStack).</summary>
    const int StreamLanes = 2;

    /// <summary>
    /// Loads a movie's frames, gain- and defect-corrected and scaled, into the stack (OriginalStack's buffer) and returns
    /// it. With <paramref name="frameSink"/>, the frames are streamed instead: no stack is made (null is returned), and
    /// the sink, made once the frames' size and count are known, takes each frame on the GPU as it is ready, with its
    /// index and lane (one of <see cref="StreamLanes"/>), from up to that many threads at a time. The frame's memory is
    /// reused once the sink returns. <paramref name="afterRead"/> runs once the file is read (or found unreadable into
    /// memory), before the frames are decoded.
    /// </summary>
    static Image LoadAndPrepareStack(string path, decimal scaleFactor, bool correctGain, int maxThreads = 8,
                                     Func<int3, Action<int, IntPtr, int>> frameSink = null, Action afterRead = null)
    {
        Image stack = null;

        MapHeader header = TryReadMovieIntoMemory(path);
        afterRead?.Invoke();
        bool InMemory = header != null;
        header ??= MapHeader.ReadFromFilePatient(10, 500,
                                                 path,
                                                 HeaderlessDims,
                                                 (int)HeaderlessOffset,
                                                 ImageFormatsHelper.StringToType(HeaderlessType));

        if (header is HeaderEER eer)
            return LoadAndPrepareEerStack(path, eer, InMemory, scaleFactor, correctGain, maxThreads, frameSink);

        bool TiffInMemory = InMemory;
        bool IsTiff = header.GetType() == typeof(HeaderTiff);

        if (GainRef != null && correctGain)
            if (header.Dimensions.X != GainRef.Dims.X || header.Dimensions.Y != GainRef.Dims.Y)
                throw new Exception($"Gain reference dimensions ({GainRef.Dims.X}x{GainRef.Dims.Y}) do not match image ({header.Dimensions.X}x{header.Dimensions.Y}).");

        int2 SourceDims = new int2(header.Dimensions);

        int NThreads = IsTiff ? maxThreads : 1;
        int GPUThreads = StreamLanes;

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

        // In-memory TIFF frames are decoded in their file type into the (pinned) raw layers, uploaded as such, and
        // converted and flipped on the GPU: the same floats ReadTIFF makes, for a quarter of the transfer at 8 bits.
        var RawPins = TiffInMemory ? RawLayers.Take(NThreads).Select(layer => System.Runtime.InteropServices.GCHandle.Alloc(layer, System.Runtime.InteropServices.GCHandleType.Pinned)).ToArray() : null;
        IntPtr[] GPURaw = TiffInMemory ? Helper.ArrayOfFunction(i => GPU.MallocDevice(SourceDims.Elements()), GPUThreads) : null;

        // Frame z into RawLayers[threadID]; returns the raw type, or -1 if it holds floats from the file readers.
        int ReadFrame(int z, int threadID)
        {
            if (TiffInMemory)
            {
                int type = TiffNative.ReadTIFFRawFromMemory(MovieFile.Pointer, MovieFile.Length, z, SourceDims.X, SourceDims.Y, RawPins[threadID].AddrOfPinnedObject());
                if (type >= 0)
                    return type;
            }

            if (IsTiff)
                TiffNative.ReadTIFFPatient(10, 500, path, z, true, RawLayers[threadID]);
            else
                IOHelper.ReadMapFloatPatient(10, 500,
                                             path,
                                             HeaderlessDims,
                                             (int)HeaderlessOffset,
                                             ImageFormatsHelper.StringToType(HeaderlessType),
                                             new[] { z },
                                             null,
                                             new[] { RawLayers[threadID] });
            return -1;
        }

        // Frame data from RawLayers[threadID] onto the GPU as float. Everything here runs on the legacy default stream,
        // which orders a lane's next frame after this one's kernels and device copies once its lock is released
        // (copies from pageable host memory also return only once the source can be reused). Building with per-thread
        // default streams would need an explicit synchronization before each lock is released.
        void UploadFrame(int threadID, int rawType, int gpuThreadID)
        {
            if (rawType < 0)
            {
                GPU.CopyHostToDevice(RawLayers[threadID], GPULayers[gpuThreadID].GetDevice(Intent.Write), SourceDims.Elements());
                return;
            }
            GPU.CopyHostPinnedToDevice(RawPins[threadID].AddrOfPinnedObject(), GPURaw[gpuThreadID], (SourceDims.Elements() * TiffNative.RawBytesPerSample(rawType) + 3) / 4);
            GPU.RawToFloat(GPURaw[gpuThreadID], rawType, GPULayers[gpuThreadID].GetDevice(Intent.Write), SourceDims.X, SourceDims.Y, true);
        }

        try
        {
            if (scaleFactor == 1M)
            {
                Action<int, IntPtr, int> Sink = frameSink?.Invoke(header.Dimensions);
                stack = Sink == null ? PrepareStack(header.Dimensions) : null;
                bool Resident = Sink == null && StackResident;
                float[][] OriginalStackData = Sink != null || Resident ? null : stack.GetHost(Intent.Write);
                IntPtr DeviceStack = Resident ? stack.GetDevice(Intent.Write) : IntPtr.Zero;

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                // Greedy: frames go to whichever reader thread is free; a fixed stride idles most threads at the end.
                Helper.ForCPUGreedy(0, header.Dimensions.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    int RawType = ReadFrame(z, threadID);

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        UploadFrame(threadID, RawType, GPUThreadID);

                        if (GainRef != null && correctGain)
                            GPULayers[GPUThreadID].MultiplySlices(GainRef);

                        if (DefectMap != null)
                        {
                            // One frame: these are per-frame layers (the whole movie's element count overran them).
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

                        if (Sink != null)
                            Sink(z, GPULayers[GPUThreadID].GetDevice(Intent.Read), GPUThreadID);
                        else if (Resident)
                            GPU.CopyDeviceToDevice(GPULayers[GPUThreadID].GetDevice(Intent.Read),
                                                   new IntPtr((long)DeviceStack + header.Dimensions.ElementsSlice() * z * sizeof(float)),
                                                   header.Dimensions.ElementsSlice());
                        else
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

                Action<int, IntPtr, int> Sink = frameSink?.Invoke(ScaledDims);
                stack = Sink == null ? PrepareStack(ScaledDims) : null;
                bool Resident = Sink == null && StackResident;
                float[][] OriginalStackData = Sink != null || Resident ? null : stack.GetHost(Intent.Write);
                IntPtr DeviceStack = Resident ? stack.GetDevice(Intent.Write) : IntPtr.Zero;

                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SourceDims), 1), GPUThreads);
                int[] PlanBack = Helper.ArrayOfFunction(i => GPU.CreateIFFTPlan(ScaledDims.Slice(), 1), GPUThreads);

                Image[] GPULayersInputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SourceDims), true, true), GPUThreads);
                Image[] GPULayersOutputFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice(), true, true), GPUThreads);

                Image[] GPULayersScaled = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, ScaledDims.Slice()), GPUThreads);

                object[] Locks = Helper.ArrayOfFunction(i => new object(), GPUThreads);

                Helper.ForCPUGreedy(0, ScaledDims.Z, NThreads, threadID => GPU.SetDevice(DeviceID), (z, threadID) =>
                {
                    int RawType = ReadFrame(z, threadID);

                    int GPUThreadID = threadID % GPUThreads;

                    lock (Locks[GPUThreadID])
                    {
                        UploadFrame(threadID, RawType, GPUThreadID);

                        if (GainRef != null && correctGain)
                            GPULayers[GPUThreadID].MultiplySlices(GainRef);

                        if (DefectMap != null)
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

                        if (Sink != null)
                            Sink(z, GPULayersScaled[GPUThreadID].GetDevice(Intent.Read), GPUThreadID);
                        else if (Resident)
                            GPU.CopyDeviceToDevice(GPULayersScaled[GPUThreadID].GetDevice(Intent.Read),
                                                   new IntPtr((long)DeviceStack + ScaledDims.ElementsSlice() * z * sizeof(float)),
                                                   ScaledDims.ElementsSlice());
                        else
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
        }
        finally
        {
            foreach (var layer in GPULayers)
                layer.Dispose();
            if (GPULayers2 != null)
                foreach (var layer in GPULayers2)
                    layer.Dispose();
            if (GPURaw != null)
                foreach (var raw in GPURaw)
                    GPU.FreeDevice(raw);
            if (RawPins != null)
                foreach (var pin in RawPins)
                    pin.Free();
        }

        return stack;
    }
    /// <summary>
    /// EER movies are rendered on the GPU from their electrons (EerRenderer): with the benefit of a 16K render's sub-pixel
    /// positions, but without its 1 GB image per frame group, the upload of that and a 16K transform. Frame groups follow
    /// HeaderEER.GroupNFrames as before, leaving out frames past the last whole group. The output is the 4K grid scaled by
    /// scaleFactor, whatever the gain's size, and the gain (4K, 8K or 16K) multiplies each electron by its pixel's value.
    /// Defects are not corrected in EER movies, as before.
    /// </summary>
    static Image LoadAndPrepareEerStack(string path, HeaderEER header, bool inMemory, decimal scaleFactor, bool correctGain, int maxThreads,
                                        Func<int3, Action<int, IntPtr, int>> frameSink = null)
    {
        int rawFrames = header.DimensionsUngrouped.Z;
        int groupFrames = HeaderEER.GroupNFrames > 0 ? HeaderEER.GroupNFrames :
                          HeaderEER.GroupNFrames < 0 ? Math.Max(1, rawFrames / -HeaderEER.GroupNFrames) : 1;
        int groups = rawFrames / groupFrames;
        if (groups < 1)
            throw new Exception($"{path} has {rawFrames} frames, fewer than one group of {groupFrames}.");

        int size = (int)Math.Round(EerRenderer.PhysicalSize * scaleFactor) / 2 * 2;
        var dims = new int3(size, size, groups);
        Image gain = correctGain ? GainRef : null;
        EerRenderer.CheckGain(gain);

        Action<int, IntPtr, int> sink = frameSink?.Invoke(dims);
        Image stack = sink == null ? PrepareStack(dims) : null;
        bool resident = sink == null && StackResident;

        // A file still being written, or a read that failed, gets another chance with a fresh read.
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                var timer = Stopwatch.StartNew();
                var index = EerIndex.Read(path, inMemory ? MovieFile : null, rawFrames);
                long electrons = EerRenderer.RenderStack(path, inMemory ? MovieFile : null, index, groupFrames, groups, gain, size, maxThreads,
                                                         resident ? stack.GetDevice(Intent.Write) : IntPtr.Zero,
                                                         sink == null && !resident ? stack.GetHost(Intent.Write) : null,
                                                         onGroup: sink == null ? null : (z, frame) => sink(z, frame, 0));
                Console.WriteLine($"Rendered {groups} EER groups of {groupFrames} frames at {size}x{size}, {electrons / (double)(groups * groupFrames) / 1e3:F0}k electrons a frame, in {timer.Elapsed.TotalMilliseconds:F0} ms");
                return stack;
            }
            // A streamed render can't start over: its sink already took the groups rendered before the failure.
            catch (Exception e) when (attempt < 3 && sink == null && e is IOException or NotSupportedException)
            {
                Console.WriteLine($"Rendering {path} failed, trying again: {e.Message}");
                System.Threading.Thread.Sleep(500);
                if (inMemory)
                    try { MovieFile.Read(path); } catch (Exception) { inMemory = false; }
            }
        }
    }
}
