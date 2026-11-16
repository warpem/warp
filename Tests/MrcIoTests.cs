using System;
using System.Buffers.Binary;
using System.IO;
using System.Linq;
using Warp;
using Warp.Tools;
using Xunit;

namespace Tests;

/// <summary>MRC reading straight into slices, exact reads, and the vectorized statistics written into headers.</summary>
public class MrcIoTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "warp-mrc-" + Guid.NewGuid().ToString("N"));
    public MrcIoTests() => Directory.CreateDirectory(root);
    public void Dispose() => Directory.Delete(root, true);

    static Image Sample(int3 dims, int seed, bool halfExact = false)
    {
        var random = new Random(seed);
        var image = new Image(dims);
        foreach (var slice in image.GetHost(Intent.Write))
            for (int i = 0; i < slice.Length; i++)
                slice[i] = halfExact ? (float)(Half)(float)(random.NextDouble() * 200 - 100) : (float)(random.NextDouble() * 2e4 - 1e4);
        return image;
    }

    [Fact]
    public void Float32AndFloat16RoundTripWithLayerSubsets()
    {
        var dims = new int3(37, 23, 4);
        using var image = Sample(dims, 1);
        string single = Path.Combine(root, "single.mrc");
        image.WriteMRC(single, 1.5f, true);
        using (var read = Image.FromFile(single))
            for (int z = 0; z < dims.Z; z++)
                Assert.Equal(image.GetHost(Intent.Read)[z], read.GetHost(Intent.Read)[z]);
        var layers = IOHelper.ReadMapFloat(single, new[] { 3, 1 });
        Assert.Equal(image.GetHost(Intent.Read)[3], layers[0]);
        Assert.Equal(image.GetHost(Intent.Read)[1], layers[1]);

        using var exact = Sample(dims, 2, halfExact: true);
        string half = Path.Combine(root, "half.mrc");
        exact.WriteMRC16b(half, 1.5f, true);
        Assert.Equal(1024 + dims.Elements() * 2, new FileInfo(half).Length);
        using (var read = Image.FromFile(half))
            for (int z = 0; z < dims.Z; z++)
                Assert.Equal(exact.GetHost(Intent.Read)[z], read.GetHost(Intent.Read)[z]);
    }

    [Fact]
    public void BigEndianFloatsAreSwappedInPlace()
    {
        var dims = new int3(5, 3, 2);
        using var image = Sample(dims, 3);
        string little = Path.Combine(root, "little.mrc"), big = Path.Combine(root, "big.mrc");
        image.WriteMRC(little, 1, true);

        // The header's numeric words (the label count and the rest are zero or text), then every value.
        byte[] bytes = File.ReadAllBytes(little);
        foreach (int word in Enumerable.Range(0, 24).Append(54).Append(55))
            BinaryPrimitives.WriteUInt32BigEndian(bytes.AsSpan(word * 4), BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(word * 4)));
        for (int offset = 1024; offset < bytes.Length; offset += 4)
            BinaryPrimitives.WriteUInt32BigEndian(bytes.AsSpan(offset), BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset)));
        File.WriteAllBytes(big, bytes);

        using var read = Image.FromFile(big);
        Assert.Equal(dims, read.Dims);
        for (int z = 0; z < dims.Z; z++)
            Assert.Equal(image.GetHost(Intent.Read)[z], read.GetHost(Intent.Read)[z]);
    }

    [Fact]
    public void TruncatedDataFailsInsteadOfReadingShort()
    {
        using var image = Sample(new int3(64, 64, 3), 4);
        string path = Path.Combine(root, "short.mrc");
        image.WriteMRC(path, 1, true);
        using (var stream = File.Open(path, FileMode.Open))
            stream.SetLength(stream.Length - 100);
        Assert.ThrowsAny<Exception>(() => Image.FromFile(path).Dispose());
    }

    [Fact]
    public void VectorizedMinMaxMatchesTheScalarPass()
    {
        var random = new Random(5);
        foreach (int length in new[] { 0, 1, 7, 8, 9, 31, 1000, 4097 })
        {
            float[] data = Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 100 - 50)).ToArray();
            var (min, max) = Image.MinMax(data);
            Assert.Equal(length == 0 ? float.MaxValue : data.Min(), min);
            Assert.Equal(length == 0 ? float.MinValue : data.Max(), max);
            if (length > 0)
            {
                data[random.Next(length)] = float.NaN;
                (min, max) = Image.MinMax(data);
                Assert.True(float.IsNaN(min) && float.IsNaN(max));
            }
        }
    }
}
