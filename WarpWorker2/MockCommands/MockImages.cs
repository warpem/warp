using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

static partial class WorkerProcess
{
    static int3 MockStackDims = new(512, 512, 16);
    const string MockProvenance = "SYNTHETIC WarpWorker2 --mock; not scientific data";
    static void MockPause(int historicalDelay = 0) => Thread.Sleep(MockDelayMs < 0 ? historicalDelay : MockDelayMs);
    static uint MockSeed(string path) => BitConverter.ToUInt32(SHA256.HashData(Encoding.UTF8.GetBytes(Path.GetFileNameWithoutExtension(path))), 0);
    static bool MockVariant(string path, string value) => path.Contains(value, StringComparison.OrdinalIgnoreCase);
    static float MockSampling(decimal value) => value > 0 ? (float)value : 1;
    static void MockDirectory(string path) => Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);

    static float MockNoise(uint seed, int x, int y, int z)
    {
        uint v = unchecked(seed ^ (uint)x * 374761393u ^ (uint)y * 668265263u ^ (uint)z * 2246822519u);
        v = unchecked((v ^ (v >> 13)) * 1274126177u);
        return ((v ^ (v >> 16)) & 65535) / 32767.5f - 1;
    }
    static float MockMicrograph(uint seed, int x, int y, int width, int height, float noise = .2f, int half = 0)
    {
        double xx = (x + .5) / width, yy = (y + .5) / height;
        double intensity = .2 * Math.Sin(xx * 7 + seed % 7) + .15 * Math.Cos(yy * 9);
        for (int j = 0; j < 4; j++)
        for (int i = 0; i < 5; i++)
        {
            double cx = .1 + i * .2 + MockNoise(seed, i, j, 1) * .025;
            double cy = .12 + j * .25 + MockNoise(seed, i, j, 2) * .025;
            double radius2 = ((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) / .00032;
            intensity -= Math.Exp(-radius2 * .5) * (1.1 + .2 * Math.Cos(i + j));
            intensity += .3 * Math.Exp(-Math.Pow(Math.Sqrt(radius2) - 2, 2) * 2);
        }
        return (float)intensity + noise * MockNoise(seed, x, y, half);
    }

    // Stream one host plane at a time. No Image allocation, CUDA, FFT, Torch or native codecs.
    static void MockWriteMrc(string path, int3 dims, float sampling, Func<int, int, int, float> pixel)
    {
        if (dims.X < 1 || dims.Y < 1 || dims.Z < 1 || !float.IsFinite(sampling) || sampling <= 0)
            throw new ArgumentException("Invalid synthetic image dimensions or sampling.");
        MockDirectory(path);
        var header = new HeaderMRC { Dimensions = dims, PixelSize = new float3(sampling), NumLabels = 1 };
        MockProvenance.CopyTo(0, header.Labels[0], 0, MockProvenance.Length);
        using var writer = new BinaryWriter(File.Create(path));
        header.Write(writer);
        var plane = new float[checked(dims.X * dims.Y)];
        float min = float.MaxValue, max = float.MinValue; double total = 0, squares = 0;
        for (int z = 0; z < dims.Z; z++)
        {
            for (int y = 0; y < dims.Y; y++) for (int x = 0; x < dims.X; x++)
            {
                float value = pixel(x, y, z); plane[y * dims.X + x] = value;
                min = Math.Min(min, value); max = Math.Max(max, value); total += value; squares += value * value;
            }
            writer.BaseStream.Write(MemoryMarshal.AsBytes(plane.AsSpan()));
        }
        double n = (double)dims.X * dims.Y * dims.Z;
        header.MinValue = min; header.MaxValue = max; header.MeanValue = (float)(total / n);
        header.StdDevValue = (float)Math.Sqrt(Math.Max(0, squares / n - total * total / (n * n)));
        writer.BaseStream.Position = 0; header.Write(writer);
    }

    static void MockWritePng(string path, int width, int height, Func<int, int, byte> pixel)
    {
        MockDirectory(path);
        using var output = File.Create(path);
        output.Write(new byte[] {137,80,78,71,13,10,26,10});
        byte[] header = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(0,4),width);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4,4),height);
        header[8]=8; // grayscale, 8 bits, no interlacing
        MockPngChunk(output,"IHDR",header);
        MockPngChunk(output,"tEXt",Encoding.UTF8.GetBytes("Description\0"+MockProvenance));
        using var compressed = new MemoryStream();
        using (var zip = new ZLibStream(compressed,CompressionLevel.Fastest,true))
        {
            var row = new byte[width+1];
            for(int y=0;y<height;y++){for(int x=0;x<width;x++)row[x+1]=pixel(x,y);zip.Write(row);}
        }
        MockPngChunk(output,"IDAT",compressed.ToArray()); MockPngChunk(output,"IEND",[]);
    }
    static void MockPngChunk(Stream output,string kind,byte[] bytes)
    {
        byte[] name=Encoding.ASCII.GetBytes(kind), length=new byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(length,bytes.Length);output.Write(length);output.Write(name);output.Write(bytes);
        uint crc=uint.MaxValue;
        foreach(byte b in name.Concat(bytes)){crc^=b;for(int i=0;i<8;i++)crc=(crc>>1)^((crc&1)!=0?0xedb88320u:0);}
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(length,~crc);output.Write(length);
    }

    [MockCommand(WorkerCommandNames.WaitAsyncTasks)]
    static void MockWaitAsync(NamedSerializableObject command) { }
    [MockCommand(WorkerCommandNames.LoadGainRef)]
    static void MockLoadGain(NamedSerializableObject command) => Console.WriteLine("[MOCK] Gain/defect initialization ignored; synthetic pixels only.");
    [MockCommand(WorkerCommandNames.SetHeaderlessParams)]
    static void MockHeaderless(NamedSerializableObject command) => SetHeaderlessParams(command);
    [MockCommand(WorkerCommandNames.GcCollect)]
    static void MockGc(NamedSerializableObject command) => GC.Collect();
}
