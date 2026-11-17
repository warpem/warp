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

    [MockCommand(WorkerCommandNames.TrainMovieDenoiser)]
    static void MockTrainMovieDenoiser(NamedSerializableObject command)
    {
        string[] movies=(string[])command.Content[0];string output=(string)command.Content[1],previous=(string)command.Content[2];int epochs=(int)command.Content[3];
        if(movies.Length==0||epochs<1)throw new ArgumentException("Mock training requires movies and positive epochs.");
        // Real training deconvolves the half-averages on the fly with each movie's CTF; the mock checks they are there.
        foreach(string path in movies)
        {
            var movie=new Movie(path);
            if(movie.OptionsCTF==null)throw new InvalidOperationException("CTF metadata is required before mock training.");
            var odd=MapHeader.ReadFromFile(movie.AverageOddPath);var even=MapHeader.ReadFromFile(movie.AverageEvenPath);
            if(odd.Dimensions!=even.Dimensions||odd.Dimensions.Z!=1||odd.PixelSize.X!=even.PixelSize.X)throw new InvalidDataException("Mock half-averages have incompatible dimensions/sampling.");
        }
        // The mock has no pretrained weights to read; it accepts the token real workers resolve to their installation's model.
        if(!string.IsNullOrEmpty(previous)&&previous!=Noise2MicTorch.PretrainedModel)MockValidateModel(previous);
        MockDirectory(output);
        // Real training reports progress beside its model; the mock walks the same file through its steps.
        for(int step=1;step<=epochs;step++){new TrainingProgress(step,epochs,step,epochs,1f/step).Write(TrainingProgress.PathFor(output));MockPause();}
        File.WriteAllText(output,JsonSerializer.Serialize(new{format=MockModelFormat,synthetic=true,provenance=MockProvenance,trainingMovies=movies,epochs}));
        File.Delete(TrainingProgress.PathFor(output));
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
        string path=(string)command.Content[0];MockValidateModel((string)command.Content[1]);float inputSampling=(float)command.Content[2];
        if(!float.IsFinite(inputSampling)||inputSampling<=0)throw new ArgumentOutOfRangeException(nameof(inputSampling));
        var movie=new Movie(path);
        if(movie.OptionsCTF==null)throw new InvalidOperationException("CTF metadata is required before mock denoising.");
        // Real denoising deconvolves the average on the fly at the denoiser's sampling; the mock writes that size.
        var source=MapHeader.ReadFromFile(movie.AveragePath);float sampling=MovieDenoising.Sampling(inputSampling);uint seed=MockSeed(path);
        int3 dims=new(Math.Max(4,(int)Math.Round(source.Dimensions.X*inputSampling/sampling/2)*2),Math.Max(4,(int)Math.Round(source.Dimensions.Y*inputSampling/sampling/2)*2),1);
        MockWriteMrc(movie.AverageDenoisedPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.035f));
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
        int count=MockVariant(path,"fewparticles")?3:20;var picks=new System.Collections.Generic.List<float4>(count);
        // Picks sit on the synthetic particles. Like the real picker, which scores positions without the
        // full blob support as invalid, particles within the margin of an edge are not picked.
        for(int i=0;i<count;i++)
        {
            int gx=i%5,gy=i/5;
            float xx=.1f+gx*.2f+MockNoise(seed,gx,gy,1)*.025f,yy=.12f+gy*.25f+MockNoise(seed,gx,gy,2)*.025f;
            float x=xx*header.Dimensions.X,y=yy*header.Dimensions.Y;
            if(x<margin||y<margin||x>=header.Dimensions.X-margin||y>=header.Dimensions.Y-margin)continue;
            picks.Add(new float4(x,y,.2f+.75f*((i*7+seed%19)%19)/18f,MathF.Exp(.2f+4.8f*((i*11+seed%23)%23)/22f)));
        }
        Directory.CreateDirectory(movie.MatchingDir);string output=Path.Combine(movie.MatchingDir,movie.RootName+"_blob.star");
        BlobPicker.WriteStar(output,movie.Name,picks.ToArray());
        File.WriteAllText(output,"# "+MockProvenance+Environment.NewLine+File.ReadAllText(output));
        MockPause();Console.WriteLine($"[MOCK] Wrote {picks.Count} synthetic blob scores to {output}");
    }

    [MockCommand(WorkerCommandNames.MovieExportParticlesFromStar)]
    static void MockMovieExportParticlesFromStar(NamedSerializableObject command)
    {
        float sampling=(float)command.Content[3];if(!float.IsFinite(sampling)||sampling<=0)throw new ArgumentOutOfRangeException(nameof(sampling));
        var table=new Star((string)command.Content[2]);
        if(!table.HasColumn("rlnCoordinateX")||!table.HasColumn("rlnCoordinateY"))throw new InvalidDataException("Missing mock particle coordinates.");
        string extracted=command.Content.Length>8?(string)command.Content[8]:null;
        if(extracted!=null)table=PickSelection.Select(table,(double)command.Content[4],(double)command.Content[5],(double)command.Content[6],(double)command.Content[7]);
        foreach(string value in table.GetColumn("rlnCoordinateX").Concat(table.GetColumn("rlnCoordinateY")))
            if(!float.IsFinite(float.Parse(value,CultureInfo.InvariantCulture)))throw new InvalidDataException("Invalid mock particle coordinate.");
        MockExtract((string)command.Content[0],(ProcessingOptionsParticleExport)command.Content[1],table.RowCount);
        if(extracted!=null)PickSelection.Write(table,extracted);
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
