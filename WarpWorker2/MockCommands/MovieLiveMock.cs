using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

static partial class WorkerProcess
{
    const string MockModelFormat="warpworker2-mock-denoiser";

    [MockCommand(WorkerCommandNames.MoviePrepareDenoising)]
    static void MockMoviePrepareDenoising(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];float inputSampling=(float)command.Content[1];
        if(!float.IsFinite(inputSampling)||inputSampling<=0)throw new ArgumentOutOfRangeException(nameof(inputSampling));
        var movie=new Movie(path);
        if(movie.OptionsCTF==null)throw new InvalidOperationException("CTF metadata is required before mock enhancement.");
        var source=MapHeader.ReadFromFile(movie.AveragePath);float sampling=Math.Max(5,inputSampling);uint seed=MockSeed(path);
        int3 dims=new(Math.Max(4,(int)Math.Round(source.Dimensions.X*inputSampling/sampling/2)*2),Math.Max(4,(int)Math.Round(source.Dimensions.Y*inputSampling/sampling/2)*2),1);
        MockWriteMrc(movie.DeconvolvedPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.16f));
        if(File.Exists(movie.AverageOddPath)&&File.Exists(movie.AverageEvenPath))
        {
            MockWriteMrc(movie.DenoiseTrainingOddPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.24f,1));
            MockWriteMrc(movie.DenoiseTrainingEvenPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.24f,2));
        }
        else{File.Delete(movie.DenoiseTrainingOddPath);File.Delete(movie.DenoiseTrainingEvenPath);}
        MockPause();Console.WriteLine($"[MOCK] Prepared synthetic deconvolution/training halves for {path}");
    }

    [MockCommand(WorkerCommandNames.TrainMovieDenoiser)]
    static void MockTrainMovieDenoiser(NamedSerializableObject command)
    {
        string[] movies=(string[])command.Content[0];string output=(string)command.Content[1],previous=(string)command.Content[2];int iterations=(int)command.Content[3];
        if(movies.Length==0||iterations<1)throw new ArgumentException("Mock training requires movies and positive iterations.");
        foreach(string path in movies)
        {
            var movie=new Movie(path);
            var odd=MapHeader.ReadFromFile(movie.DenoiseTrainingOddPath);var even=MapHeader.ReadFromFile(movie.DenoiseTrainingEvenPath);
            if(odd.Dimensions!=even.Dimensions||odd.Dimensions.Z!=1||odd.PixelSize.X!=even.PixelSize.X)throw new InvalidDataException("Mock training halves have incompatible dimensions/sampling.");
        }
        if(!string.IsNullOrEmpty(previous))MockValidateModel(previous);
        MockDirectory(output);MockPause();
        File.WriteAllText(output,JsonSerializer.Serialize(new{format=MockModelFormat,synthetic=true,provenance=MockProvenance,trainingMovies=movies,iterations}));
        Console.WriteLine($"[MOCK] Wrote synthetic denoiser marker for {movies.Length} movies (not a Torch model): {output}");
    }
    static void MockValidateModel(string path)
    {
        using var json=JsonDocument.Parse(File.ReadAllText(path));
        if(!json.RootElement.TryGetProperty("format",out var format)||format.GetString()!=MockModelFormat)
            throw new InvalidDataException("Mock denoising requires a synthetic mock model marker.");
    }

    [MockCommand(WorkerCommandNames.MovieDenoise)]
    static void MockMovieDenoise(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];MockValidateModel((string)command.Content[1]);var movie=new Movie(path);
        var source=MapHeader.ReadFromFile(movie.DeconvolvedPath);uint seed=MockSeed(path);int3 dims=source.Dimensions;
        MockWriteMrc(movie.AverageDenoisedPath,dims,source.PixelSize.X,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.035f));
        MockPause();Console.WriteLine($"[MOCK] Synthetic denoised micrograph for {path}");
    }

    [MockCommand(WorkerCommandNames.MicrographPickBlobs)]
    static void MockMicrographPickBlobs(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];float radius=(float)command.Content[1];
        if(!float.IsFinite(radius)||radius<=0)throw new ArgumentOutOfRangeException(nameof(radius));
        var header=MapHeader.ReadFromFile(path);var movie=new Movie(path);uint seed=MockSeed(path);
        float margin=MathF.Ceiling(2*radius);
        if(margin*2>=Math.Min(header.Dimensions.X,header.Dimensions.Y))throw new ArgumentException("Mock micrograph is too small for the full blob support.");
        int count=MockVariant(path,"fewparticles")?3:20;var picks=new float4[count];
        for(int i=0;i<count;i++)
        {
            int gx=i%5,gy=i/5;
            float xx=.1f+gx*.2f+MockNoise(seed,gx,gy,1)*.025f,yy=.12f+gy*.25f+MockNoise(seed,gx,gy,2)*.025f;
            picks[i]=new float4(Math.Clamp(xx*header.Dimensions.X,margin,header.Dimensions.X-1-margin),Math.Clamp(yy*header.Dimensions.Y,margin,header.Dimensions.Y-1-margin),.2f+.75f*((i*7+seed%19)%19)/18f,
                MathF.Exp(.2f+4.8f*((i*11+seed%23)%23)/22f));
        }
        Directory.CreateDirectory(movie.MatchingDir);string output=Path.Combine(movie.MatchingDir,movie.RootName+"_blob.star");
        BlobPicker.WriteStar(output,movie.Name,picks);
        File.WriteAllText(output,"# "+MockProvenance+Environment.NewLine+File.ReadAllText(output));
        MockPause();Console.WriteLine($"[MOCK] Wrote {count} synthetic blob scores to {output}");
    }

    [MockCommand(WorkerCommandNames.MovieExportParticlesFromStar)]
    static void MockMovieExportParticlesFromStar(NamedSerializableObject command)
    {
        float sampling=(float)command.Content[3];if(!float.IsFinite(sampling)||sampling<=0)throw new ArgumentOutOfRangeException(nameof(sampling));
        var table=new Star((string)command.Content[2]);
        if(!table.HasColumn("rlnCoordinateX")||!table.HasColumn("rlnCoordinateY"))throw new InvalidDataException("Missing mock particle coordinates.");
        foreach(string value in table.GetColumn("rlnCoordinateX").Concat(table.GetColumn("rlnCoordinateY")))
            if(!float.IsFinite(float.Parse(value,CultureInfo.InvariantCulture)))throw new InvalidDataException("Invalid mock particle coordinate.");
        MockExtract((string)command.Content[0],(ProcessingOptionsParticleExport)command.Content[1],table.RowCount);
    }
    static void MockExtract(string path,ProcessingOptionsParticleExport options,int count)
    {
        var movie=new Movie(path);string output=Path.Combine(movie.ParticlesDir,movie.RootName+options.Suffix+".mrcs");
        int box=options.BoxSizeResample>0?options.BoxSizeResample:options.BoxSize;
        if(box<2)throw new ArgumentOutOfRangeException(nameof(options.BoxSize));
        if(count==0){File.Delete(output);return;}
        float sampling=MockSampling(options.BinnedPixelSizeMean)*options.BoxSize/box;uint seed=MockSeed(path);
        MockWriteMrc(output,new(box,box,count),sampling,(x,y,z)=>
        {double xx=(x-box*.5)/(box*.13),yy=(y-box*.5)/(box*.13),r=xx*xx+yy*yy;return (float)(-Math.Exp(-r*.5)+.25*Math.Exp(-Math.Pow(Math.Sqrt(r)-2,2)*2))+.1f*MockNoise(seed,x,y,z);});
        options.Dimensions=MockStackDims.MultXY(MockSampling(options.BinnedPixelSizeMean));movie.OptionsParticlesExport=options;
        MockPause();movie.SaveMeta();Console.WriteLine($"[MOCK] Wrote {count} synthetic particle images to {output}");
    }
}
