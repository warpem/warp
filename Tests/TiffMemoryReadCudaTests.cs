using System;
using System.IO;
using BitMiracle.LibTiff.Classic;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

/// <summary>LoadStack's in-memory TIFF path must give exactly the floats ReadTIFF gives from the file.</summary>
public class TiffMemoryReadCudaTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "warp-tiffmem-" + Guid.NewGuid().ToString("N"));
    public TiffMemoryReadCudaTests() => Directory.CreateDirectory(root);
    public void Dispose() => Directory.Delete(root, true);

    const int Width = 61, Height = 45, Frames = 3;

    static double Sample(int z, int y, int x) => (x * 7 + y * 13 + z * 29) % 251 + (x == y ? 3 : 0);

    string WriteStack(string name, int bits, SampleFormat format, Compression compression)
    {
        string path = Path.Combine(root, name);
        using var tiff = Tiff.Open(path, "w");
        int bytes = bits / 8;
        for (int z = 0; z < Frames; z++)
        {
            tiff.SetField(TiffTag.IMAGEWIDTH, Width);
            tiff.SetField(TiffTag.IMAGELENGTH, Height);
            tiff.SetField(TiffTag.BITSPERSAMPLE, bits);
            tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
            tiff.SetField(TiffTag.SAMPLEFORMAT, format);
            tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
            tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            tiff.SetField(TiffTag.ROWSPERSTRIP, 8); // the last strip is short
            tiff.SetField(TiffTag.COMPRESSION, compression);
            tiff.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
            var row = new byte[Width * bytes];
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    double v = Sample(z, y, x);
                    byte[] b = format == SampleFormat.IEEEFP ? BitConverter.GetBytes((float)(v - 100.25)) :
                               bits == 16 ? BitConverter.GetBytes(format == SampleFormat.INT ? (short)(v * 100 - 12000) : (ushort)(v * 200)) :
                               new[] { format == SampleFormat.INT ? unchecked((byte)(sbyte)(v - 128)) : (byte)v };
                    Array.Copy(b, 0, row, x * bytes, bytes);
                }
                tiff.WriteScanline(row, y);
            }
            tiff.WriteDirectory();
        }
        return path;
    }

    [CtfCudaFact]
    public void InMemoryRawDecodeAndGpuConversionMatchReadTiff()
    {
        lock (GPU.Sync)
        {
            GPU.SetDevice(0);
            var memory = new FileInMemory();
            foreach (var (name, bits, format, compression, type) in new[]
            {
                ("u8-lzw.tif", 8, SampleFormat.UINT, Compression.LZW, TiffNative.RawType.UChar),
                ("s8.tif", 8, SampleFormat.INT, Compression.NONE, TiffNative.RawType.SChar),
                ("u16-lzw.tif", 16, SampleFormat.UINT, Compression.LZW, TiffNative.RawType.UShort),
                ("s16.tif", 16, SampleFormat.INT, Compression.ADOBE_DEFLATE, TiffNative.RawType.Short),
                ("f32.tif", 32, SampleFormat.IEEEFP, Compression.NONE, TiffNative.RawType.Float),
            })
            {
                string path = WriteStack(name, bits, format, compression);
                memory.Read(path);
                Assert.Equal(new FileInfo(path).Length, memory.Length);
                var raw = new float[Width * Height]; // as LoadStack's raw layers: room for 4-byte samples
                var pin = System.Runtime.InteropServices.GCHandle.Alloc(raw, System.Runtime.InteropServices.GCHandleType.Pinned);
                IntPtr deviceRaw = GPU.MallocDevice(Width * Height);
                using var converted = new Image(new int3(Width, Height, 1));
                try
                {
                    for (int z = 0; z < Frames; z++)
                    {
                        var expected = new float[Width * Height];
                        TiffNative.ReadTIFF(path, z, true, expected);
                        int read = TiffNative.ReadTIFFRawFromMemory(memory.Pointer, memory.Length, z, Width, Height, pin.AddrOfPinnedObject());
                        Assert.Equal((int)type, read);
                        GPU.CopyHostPinnedToDevice(pin.AddrOfPinnedObject(), deviceRaw, (Width * Height * TiffNative.RawBytesPerSample(read) + 3) / 4);
                        GPU.RawToFloat(deviceRaw, read, converted.GetDevice(Intent.Write), Width, Height, true);
                        Assert.Equal(expected, converted.GetHost(Intent.Read)[0]);
                    }
                    // Another size, a missing layer or truncated data are refused, so LoadStack falls back to ReadTIFF.
                    Assert.Equal(-1, TiffNative.ReadTIFFRawFromMemory(memory.Pointer, memory.Length, 0, Width + 1, Height, pin.AddrOfPinnedObject()));
                    Assert.Equal(-1, TiffNative.ReadTIFFRawFromMemory(memory.Pointer, memory.Length, Frames, Width, Height, pin.AddrOfPinnedObject()));
                    Assert.Equal(-1, TiffNative.ReadTIFFRawFromMemory(memory.Pointer, memory.Length / 2, Frames - 1, Width, Height, pin.AddrOfPinnedObject()));
                }
                finally { GPU.FreeDevice(deviceRaw); pin.Free(); }
            }
        }
    }
}
