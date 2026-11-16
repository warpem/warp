using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Warp;
using Warp.Headers;
using Warp.Tools;
using Warp.Workers;
using Warp.Workers.Queue;
using Xunit;

namespace Tests.Workers;

public sealed class WarpMockWorkerTests : IDisposable
{
    private readonly string root=Path.Combine(Path.GetTempPath(),"warp session worker mock "+Guid.NewGuid().ToString("N"));
    public void Dispose(){if(Directory.Exists(root))Directory.Delete(root,true);}

    [Fact]
    public void CompleteCpuPipelineWritesDeterministicReadableArtifacts()
    {
        Directory.CreateDirectory(root);
        string path=Path.Combine(root,"synthetic.mrc"), model=Path.Combine(root,"model.pt");
        IOHelper.WriteMapFloat(path,new HeaderMRC{Dimensions=new int3(96,64,4),PixelSize=new float3(1)},new float[96*64*4]);
        var movie=new Movie(path);
        string picks=Path.Combine(movie.AverageDenoisedDir,"matching",movie.RootName+"_blob.star");
        var commands=new[]{
            WorkerCommands.LoadStack(path,1,1),
            WorkerCommands.MovieProcessMovement(path,new ProcessingOptionsMovieMovement{PixelSize=1}),
            WorkerCommands.MovieProcessCTF(path,new ProcessingOptionsMovieCTF{PixelSize=1,Window=128,FitThickness=true,DoPhase=true}),
            WorkerCommands.MovieExportMovie(path,new ProcessingOptionsMovieExport{PixelSize=1,DoAverage=true,DoDenoise=true}),
            WorkerCommands.MovieCreateThumbnail(path,64,2),
            WorkerCommands.TrainMovieDenoiser([path],model,"",2,.9f,.1f,1),
            WorkerCommands.MovieDenoise(path,model,1,.9f,.1f),
            WorkerCommands.MicrographPickBlobs(movie.AverageDenoisedPath,1),
            WorkerCommands.MovieExportParticlesFromStar(path,new ProcessingOptionsParticleExport{PixelSize=1,BoxSize=16,BoxSizeResample=-1,Suffix="_warp"},picks,5),
            WorkerCommands.WaitAsyncTasks()
        };
        RunTask("first",commands);
        movie=new Movie(path);
        Assert.NotNull(movie.OptionsMovement);Assert.NotNull(movie.OptionsCTF);Assert.NotNull(movie.OptionsMovieExport);Assert.NotNull(movie.OptionsParticlesExport);
        Assert.Equal(128,movie.PS1D.Length);Assert.Equal(128,movie.Simulated1D.Length);Assert.Equal(128,movie.CTFQuality.Length);
        Assert.InRange(movie.CTF.DefocusAngle,0,179);Assert.True(movie.MeanFrameMovement>0);Assert.True(movie.CTFSpecimenThicknessAngstrom>0);
        var average=MapHeader.ReadFromFile(movie.AveragePath);Assert.Equal(new int3(96,64,1),average.Dimensions);Assert.Equal(1,average.PixelSize.X);
        Assert.Contains("SYNTHETIC",new string(((HeaderMRC)average).Labels[0]));
        var enhanced=MapHeader.ReadFromFile(movie.AverageDenoisedPath);Assert.Equal(new int3(16,10,1),enhanced.Dimensions);Assert.Equal(6,enhanced.PixelSize.X);
        // Denoising deconvolves on the fly: nothing deconvolved is left on disk.
        Assert.False(File.Exists(movie.DeconvolvedPath));Assert.False(File.Exists(movie.DenoiseTrainingOddPath));
        Assert.Equal(new int3(128,64,1),MapHeader.ReadFromFile(movie.PowerSpectrumPath).Dimensions);
        // Radius 1 needs a 2-pixel margin on the 16x10 enhanced image (6 Å/pixel); particles closer to an edge are not picked.
        var table=new Star(picks);Assert.InRange(table.RowCount,1,19);Assert.True(table.HasColumn("wrpAutopickPower"));
        Assert.All(table.GetFloat("rlnCoordinateX"),x=>Assert.True(x>=2&&x<14,$"x={x}"));
        Assert.All(table.GetFloat("rlnCoordinateY"),y=>Assert.True(y>=2&&y<8,$"y={y}"));
        string stack=Path.Combine(movie.ParticlesDir,movie.RootName+"_warp.mrcs");
        Assert.Equal(new int3(16,16,table.RowCount),MapHeader.ReadFromFile(stack).Dimensions);
        using(var marker=JsonDocument.Parse(File.ReadAllText(model))){Assert.Equal("warpworker2-mock-denoiser",marker.RootElement.GetProperty("format").GetString());Assert.True(marker.RootElement.GetProperty("synthetic").GetBoolean());}
        ValidatePng(movie.ThumbnailsPath,64,64);
        string[] files=[movie.AveragePath,movie.PowerSpectrumPath,movie.ThumbnailsPath,movie.AverageDenoisedPath,picks,stack];
        var hashes=files.Select(file=>Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file)))).ToArray();
        RunTask("second",commands);
        Assert.Equal(hashes,files.Select(file=>Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file)))).ToArray());

        // With score bounds only the picks within them are extracted, and their rows are recorded in stack order.
        var correlations=table.GetColumn("rlnAutopickFigureOfMerit").Select(v=>double.Parse(v,CultureInfo.InvariantCulture)).ToArray();
        double median=correlations.OrderBy(v=>v).ElementAt(correlations.Length/2);
        string extracted=Path.Combine(movie.ParticlesDir,movie.RootName+"_warp.star");
        var particleOptions=new ProcessingOptionsParticleExport{PixelSize=1,BoxSize=16,BoxSizeResample=-1,Suffix="_warp"};
        RunTask("bounded",[WorkerCommands.LoadStack(path,1,1),WorkerCommands.MovieExportParticlesFromStar(path,particleOptions,picks,5,median,1,0,double.MaxValue,extracted),WorkerCommands.WaitAsyncTasks()]);
        var kept=new Star(extracted);
        Assert.Equal(correlations.Count(c=>c>=median),kept.RowCount);
        Assert.Equal(correlations.Where(c=>c>=median),kept.GetColumn("rlnAutopickFigureOfMerit").Select(v=>double.Parse(v,CultureInfo.InvariantCulture)));
        Assert.Equal(new int3(16,16,kept.RowCount),MapHeader.ReadFromFile(stack).Dimensions);
        // Bounds that keep nothing leave an empty list and no stack from the earlier extraction.
        RunTask("none",[WorkerCommands.LoadStack(path,1,1),WorkerCommands.MovieExportParticlesFromStar(path,particleOptions,picks,5,1.5,2,0,double.MaxValue,extracted),WorkerCommands.WaitAsyncTasks()]);
        Assert.Equal(0,new Star(extracted).RowCount);Assert.True(new Star(extracted).HasColumn("wrpAutopickPower"));Assert.False(File.Exists(stack));
    }

    [Fact]
    public void UnsupportedRealAlignedStackExportAlsoFailsInMock()
    {
        var layout=new QueueLayout(root);layout.EnsureDirectories();var queue=new TaskQueue(layout);
        string path=Path.Combine(root,"movie.mrc");
        var task=new TaskItem{TaskId="aligned-stack",Main=[WorkerCommands.LoadStack(path,1,1),WorkerCommands.MovieExportMovie(path,new ProcessingOptionsMovieExport{PixelSize=1,DoAverage=true,DoStack=true})]};
        task.ComputeInitFingerprint();queue.Enqueue(task);RunWorker();
        string failed=Path.Combine(layout.Failed,"aligned-stack.json");Assert.True(File.Exists(failed));
        Assert.Contains("not implemented by the real exporter",File.ReadAllText(failed));
        Assert.False(File.Exists(new Movie(path).AveragePath));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnsupportedMockCommandFailsRatherThanReportingSuccessfulProcessing(bool duringInitialization)
    {
        var layout=new QueueLayout(root);layout.EnsureDirectories();var queue=new TaskQueue(layout);
        var bad=new TaskItem{TaskId="unsupported",Init=duringInitialization?[new NamedSerializableObject("FutureProcessingCommand")]:[],Main=duringInitialization?[WorkerCommands.WaitAsyncTasks()]:[new NamedSerializableObject("FutureProcessingCommand")]};bad.ComputeInitFingerprint();queue.Enqueue(bad);
        var good=new TaskItem{TaskId="service",Main=[WorkerCommands.WaitAsyncTasks()]};good.ComputeInitFingerprint();queue.Enqueue(good);
        RunWorker();
        Assert.True(File.Exists(Path.Combine(layout.Failed,"unsupported.json")));
        Assert.Contains("No CPU mock handler",File.ReadAllText(Path.Combine(layout.Failed,"unsupported.json")));
        Assert.True(File.Exists(Path.Combine(layout.Done,"service.json")));
    }

    [Fact]
    public void PreviewCommandWritesAFittedGrayPng()
    {
        Directory.CreateDirectory(root);
        string path=Path.Combine(root,"synthetic.mrc"),preview=Path.Combine(root,"previews","synthetic.png");
        IOHelper.WriteMapFloat(path,new HeaderMRC{Dimensions=new int3(2048,1024,1),PixelSize=new float3(1)},Enumerable.Range(0,2048*1024).Select(i=>(float)Math.Sin(i*.01)).ToArray());
        RunTask("preview",[WorkerCommands.MicrographCreatePreview(path,preview,1024),WorkerCommands.WaitAsyncTasks()]);
        ValidatePng(preview,1024,512);
    }

    [Fact]
    public void FailedTaskRecordsHowManyCommandsCompleted()
    {
        Directory.CreateDirectory(root);
        string path=Path.Combine(root,"synthetic.mrc");
        IOHelper.WriteMapFloat(path,new HeaderMRC{Dimensions=new int3(96,64,4),PixelSize=new float3(1)},new float[96*64*4]);
        var layout=new QueueLayout(Path.Combine(root,"tasks"));layout.EnsureDirectories();var queue=new TaskQueue(layout);
        var task=new TaskItem{TaskId="partial",Main=[
            WorkerCommands.LoadStack(path,1,1),
            WorkerCommands.MovieProcessMovement(path,new ProcessingOptionsMovieMovement{PixelSize=1}),
            WorkerCommands.MovieProcessCTF(path,new ProcessingOptionsMovieCTF{PixelSize=1,Window=128}),
            WorkerCommands.WaitAsyncTasks()]};
        task.ComputeInitFingerprint();queue.Enqueue(task);
        RunWorker(layout.Root,"--mock-fail-command",WorkerCommandNames.MovieProcessCTF);
        var failed=TaskItem.FromJson(File.ReadAllText(Path.Combine(layout.Failed,"partial.json")));
        Assert.Equal(2,failed.CompletedCommands);
    }

    private void RunTask(string id,NamedSerializableObject[] commands)
    {
        var layout=new QueueLayout(Path.Combine(root,"tasks"));layout.EnsureDirectories();var queue=new TaskQueue(layout);
        var task=new TaskItem{TaskId=id,Main=commands};task.ComputeInitFingerprint();queue.Enqueue(task);
        RunWorker(layout.Root);
        string error=Path.Combine(layout.Failed,id+".json");
        Assert.True(File.Exists(Path.Combine(layout.Done,id+".json")),File.Exists(error)?File.ReadAllText(error):"Worker did not complete task");
    }
    private void RunWorker(string queueDirectory=null,params string[] extra)
    {
        var info=new ProcessStartInfo{FileName=Path.Combine(AppContext.BaseDirectory,"WarpWorker2"),UseShellExecute=false,RedirectStandardOutput=true,RedirectStandardError=true};
        foreach(string arg in new[]{"--device","0","--queue-dir",queueDirectory??root,"--mock","--mock-delay-ms","0","--debug"}.Concat(extra))info.ArgumentList.Add(arg);
        using var process=Process.Start(info)!;
        var stdout=process.StandardOutput.ReadToEndAsync();var stderr=process.StandardError.ReadToEndAsync();
        try{Assert.True(process.WaitForExit(30000),"Mock worker timed out");Assert.True(process.ExitCode==0,$"exit {process.ExitCode}\n{stdout.Result}\n{stderr.Result}");}
        finally{if(!process.HasExited){process.Kill(true);process.WaitForExit();}}
    }
    private static void ValidatePng(string path,int width,int height)
    {
        using var file=File.OpenRead(path);using var reader=new BinaryReader(file);
        Assert.Equal(new byte[]{137,80,78,71,13,10,26,10},reader.ReadBytes(8));
        using var compressed=new MemoryStream();bool provenance=false;
        while(file.Position<file.Length)
        {
            int length=System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(reader.ReadBytes(4));string kind=Encoding.ASCII.GetString(reader.ReadBytes(4));byte[] bytes=reader.ReadBytes(length);reader.ReadUInt32();
            if(kind=="IHDR"){Assert.Equal(width,System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(0,4)));Assert.Equal(height,System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(4,4)));}
            if(kind=="tEXt")provenance=Encoding.UTF8.GetString(bytes).Contains("SYNTHETIC");
            if(kind=="IDAT")compressed.Write(bytes);
        }
        Assert.True(provenance);compressed.Position=0;using var zip=new ZLibStream(compressed,CompressionMode.Decompress);using var pixels=new MemoryStream();zip.CopyTo(pixels);
        Assert.Equal((width+1)*height,pixels.Length);Assert.True(pixels.ToArray().Distinct().Count()>20);
    }
}
