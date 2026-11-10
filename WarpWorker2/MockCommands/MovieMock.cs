using System;
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
    [MockCommand(WorkerCommandNames.LoadStack)]
    static void MockLoadStack(NamedSerializableObject command)
    {
        string path=(string)command.Content[0]; decimal scale=(decimal)command.Content[1];
        int3 raw=new(512,512,16);
        if(File.Exists(path) && new[]{".mrc",".mrcs",".map",".st"}.Contains(Path.GetExtension(path).ToLowerInvariant()))
            raw=MapHeader.ReadFromFile(path).Dimensions;
        if(scale<=0)throw new ArgumentOutOfRangeException(nameof(scale));
        MockStackDims=new(Math.Max(2,(int)Math.Round(raw.X*scale)/2*2),Math.Max(2,(int)Math.Round(raw.Y*scale)/2*2),Math.Max(1,raw.Z));
        OriginalStack?.Dispose();OriginalStack=null;OriginalStackOwner=Path.GetFileName(path);
        Console.WriteLine($"[MOCK] Synthetic stack geometry {MockStackDims}; source {path}");
    }

    [MockCommand(WorkerCommandNames.MovieProcessCTF)]
    static void MockMovieProcessCTF(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];var options=(ProcessingOptionsMovieCTF)command.Content[1];
        uint seed=MockSeed(path);float sampling=MockSampling(options.BinnedPixelSizeMean);
        options.Dimensions=MockStackDims.MultXY(sampling);
        var movie=new Movie(path) {OptionsCTF=options};
        movie.CTF=new CTF {PixelSize=(decimal)sampling,Amplitude=options.Amplitude>0?options.Amplitude:.07m,
            Defocus=.7m+(seed%2300)/1000m,DefocusDelta=.03m+(seed%120)/1000m,DefocusAngle=seed%180,
            Cs=options.Cs>0?options.Cs:2.7m,Voltage=options.Voltage>0?options.Voltage:300,PhaseShift=options.DoPhase?.15m+(seed%30)/100m:0};
        movie.GridCTFDefocus=new CubicGrid(new int3(1),[(float)movie.CTF.Defocus]);
        movie.GridCTFPhase=new CubicGrid(new int3(1),[(float)movie.CTF.PhaseShift]);
        movie.CTFResolutionEstimate=MockVariant(path,"badctf")?8+(seed%20)/10m:2.5m+(seed%18)/10m;
        movie.CTFSpecimenThicknessAngstrom=options.FitThickness?(MockVariant(path,"thickice")?2500+seed%800:150+seed%700):0;
        movie.SimulatedBackground=new Cubic1D([new float2(0,.1f),new float2(.5f,.03f)]);
        movie.SimulatedScale=new Cubic1D([new float2(0,1),new float2(.5f,.25f)]);
        movie.PS1D=Enumerable.Range(0,128).Select(i=>{float f=i/256f;return new float2(f,.1f+(float)(.5+.45*Math.Cos(f*f*900+(seed%11)))*MathF.Exp(-f*2));}).ToArray();
        movie.CTFQuality=Enumerable.Range(0,128).Select(i=>{float f=i/256f;float cutoff=sampling/(float)movie.CTFResolutionEstimate;return new float2(f,Math.Clamp(.98f-.9f/(1+MathF.Exp(-(f-cutoff)*22)),0,1));}).ToArray();
        int window=options.Window>0?Math.Clamp(options.Window,64,512):256;
        MockWriteMrc(movie.PowerSpectrumPath,new(window,window/2,1),sampling,(x,y,_)=>
        {double xx=(x-window/2.0)/window,yy=(double)y/window,r2=xx*xx+yy*yy;return (float)(.15+(.6+.4*Math.Cos(r2*900+seed%11))*Math.Exp(-r2*3))+.035f*MockNoise(seed,x,y,0);});
        MockPause(3000+(int)(seed%1000));movie.SaveMeta();
        Console.WriteLine($"[MOCK] Synthetic CTF, power spectrum and fit metadata for {path}");
    }

    [MockCommand(WorkerCommandNames.MovieProcessMovement)]
    static void MockMovieProcessMovement(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];var options=(ProcessingOptionsMovieMovement)command.Content[1];uint seed=MockSeed(path);
        options.Dimensions=MockStackDims.MultXY(MockSampling(options.BinnedPixelSizeMean));
        var movie=new Movie(path){OptionsMovement=options,MeanFrameMovement=MockVariant(path,"highmotion")?5+(seed%30)/10m:.3m+(seed%13)/10m};
        movie.GridMovementX=new CubicGrid(new int3(1),[0]);movie.GridMovementY=new CubicGrid(new int3(1),[0]);
        var track=Enumerable.Range(0,MockStackDims.Z).Select(i=>i/(float)Math.Max(1,MockStackDims.Z-1)).ToArray();
        MockDirectory(movie.MotionTracksPath);
        // The viewer expects a dictionary keyed by spatial grid cell.
        File.WriteAllText(movie.MotionTracksPath,JsonSerializer.Serialize(new System.Collections.Generic.Dictionary<string,object>{["0_0"]=new{x=track.Select(t=>t*(float)movie.MeanFrameMovement*3).ToArray(),y=track.Select(t=>MathF.Sin(t*3)*(float)movie.MeanFrameMovement).ToArray()}}));
        MockPause(100+(int)(seed%50));movie.SaveMeta();
        Console.WriteLine($"[MOCK] Synthetic motion for {path}");
    }

    [MockCommand(WorkerCommandNames.MovieExportMovie)]
    static void MockMovieExportMovie(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];var options=(ProcessingOptionsMovieExport)command.Content[1];
        if(options.DoStack)throw new NotSupportedException("Aligned movie-stack export is not implemented by the real exporter and cannot be simulated as successful.");
        var movie=new Movie(path);
        uint seed=MockSeed(path);float sampling=MockSampling(options.BinnedPixelSizeMean);options.Dimensions=MockStackDims.MultXY(sampling);
        int3 dims=new(MockStackDims.X,MockStackDims.Y,1);
        if(options.DoAverage)MockWriteMrc(movie.AveragePath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y));
        if(options.DoDenoise)
        {
            int frames=MockStackDims.Z-Math.Max(0,options.SkipFirstN)-Math.Max(0,options.SkipLastN);
            if(frames>=2){MockWriteMrc(movie.AverageOddPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.3f,1));MockWriteMrc(movie.AverageEvenPath,dims,sampling,(x,y,z)=>MockMicrograph(seed,x,y,dims.X,dims.Y,.3f,2));}
            else{File.Delete(movie.AverageOddPath);File.Delete(movie.AverageEvenPath);File.Delete(movie.DenoiseTrainingOddPath);File.Delete(movie.DenoiseTrainingEvenPath);}
        }
        movie.OptionsMovieExport=options;MockPause();movie.SaveMeta();Console.WriteLine($"[MOCK] Synthetic average/halves for {path}");
    }

    [MockCommand(WorkerCommandNames.MovieCreateThumbnail)]
    static void MockMovieCreateThumbnail(NamedSerializableObject command)
    {
        string path=(string)command.Content[0];var movie=new Movie(path);int size=Math.Clamp((int)command.Content[1],16,1024);uint seed=MockSeed(path);
        MockWritePng(movie.ThumbnailsPath,size,size,(x,y)=>(byte)Math.Clamp(155+MockMicrograph(seed,x,y,size,size,.08f)*70,0,255));
        Console.WriteLine($"[MOCK] Synthetic thumbnail for {path}");
    }
    [MockCommand(WorkerCommandNames.LoadBoxNet)] static void MockLoadBoxNet(NamedSerializableObject command)=>Console.WriteLine("[MOCK] BoxNet initialization ignored.");
    [MockCommand(WorkerCommandNames.DropBoxNet)] static void MockDropBoxNet(NamedSerializableObject command) { }
    [MockCommand(WorkerCommandNames.MoviePickBoxNet)] static void MockMoviePickBoxNet(NamedSerializableObject command)
    { new Movie((string)command.Content[0]).SaveMeta();Console.WriteLine("[MOCK] Legacy BoxNet inference skipped; use blob picking for synthetic coordinates."); }
    [MockCommand(WorkerCommandNames.MovieExportParticles)]
    static void MockMovieExportParticles(NamedSerializableObject command)=>MockExtract((string)command.Content[0],(ProcessingOptionsParticleExport)command.Content[1],((float2[])command.Content[2]).Length);
}
