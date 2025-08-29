using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker;

static partial class WarpWorkerProcess
{
    [Command(nameof(WorkerWrapper.TomoStack))]
    static void TomoStack(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoStack Options = (ProcessingOptionsTomoStack)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.StackTilts(Options);
        T.SaveMeta();

        Console.WriteLine($"Created tilt stack for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoAretomo))]
    static void TomoAretomo(NamedSerializableObject Command)
    {
        string SeriesPath = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoAretomo)Command.Content[1];

        TiltSeries T = new TiltSeries(SeriesPath);

        string StackDir = T.TiltStackDir;
        string StackPath = Path.GetFileName(T.TiltStackPath);
        string AnglePath = Path.GetFileNameWithoutExtension(StackPath) + ".rawtlt";
        string OutStack = T.RootName + "_aligned.mrc";
        string Axis = Options.AxisAngle.ToString() + (Options.DoAxisSearch ? " 0" : " -1");
        string AlignZ = Options.AlignZ.ToString();
        string NPatchesX = Options.NPatchesXY[0].ToString();
        string NPatchesY = Options.NPatchesXY[1].ToString();

        string Arguments = $"-InMrc {StackPath} -AngFile {AnglePath} -VolZ 0 -OutBin 0 -TiltAxis {Axis} -AlignZ {AlignZ} -TiltCor 1 -OutImod 1 -DarkTol 0 -OutMrc {OutStack} -Gpu {DeviceID} -Patch {NPatchesX} {NPatchesY}";

        Console.WriteLine($"Executing {Options.Executable} in {StackDir} with arguments: {Arguments}");

        Process AreTomo = new Process
        {
            StartInfo =
            {
                FileName = Options.Executable,
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Minimized,
                WorkingDirectory = StackDir,
                Arguments = Arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }
        };
        DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
        AreTomo.OutputDataReceived += Handler;
        AreTomo.ErrorDataReceived += Handler;

        AreTomo.Start();

        AreTomo.BeginOutputReadLine();
        AreTomo.BeginErrorReadLine();

        AreTomo.WaitForExit();

        Console.WriteLine($"Executed AreTomo for {SeriesPath}");
    }
    
    [Command(nameof(WorkerWrapper.TomoEtomoPatchTrack))]
    static void TomoEtomoPatchTrack(NamedSerializableObject Command)
    {
        string TiltSeriesPath = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoEtomoPatch)Command.Content[1];

        TiltSeries T = new TiltSeries(TiltSeriesPath);
        
        // First generate a directive file to run Etomo automatically through batchruntomo
        int PatchSize = (int)(Options.PatchSizeAngstroms / Options.TiltStackAngPix);
        int RotOption = Options.DoAxisAngleSearch ? -1 : 0; // Fit single value (-1) or leave fixed (0)
        var DirectiveFile = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
        DirectiveFile = Path.ChangeExtension(DirectiveFile, ".adoc");

        var BRTConfig = $"setupset.copyarg.userawtlt = 1\n" +
                        $"setupset.copyarg.stackext = st\n" +
                        $"setupset.copyarg.rotation = {Options.AxisAngle}\n" +
                        $"setupset.copyarg.pixel = {Options.TiltStackAngPix / 10}\n" +
                        $"setupset.copyarg.dual = 0\n" +
                        $"runtime.Fiducials.any.trackingMethod = 1\n" +
                        $"comparam.xcorr.tiltxcorr.FilterRadius2 = 0.5\n" +
                        $"comparam.xcorr.tiltxcorr.ExcludeCentralPeak = 1\n" +
                        $"comparam.xcorr_pt.tiltxcorr.IterateCorrelations = 4\n" +
                        $"comparam.xcorr_pt.imodchopconts.LengthOfPieces = -1\n" +
                        $"comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY = {PatchSize},{PatchSize}\n" +
                        $"comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY = 0.8,0.8\n" +
                        $"comparam.align.tiltalign.MagOption = 0\n" +
                        $"comparam.align.tiltalign.TiltOption = 0\n" +
                        $"comparam.align.tiltalign.RotOption = {RotOption}\n" +
                        $"comparam.align.tiltalign.RobustFitting = 1\n" +
                        $"comparam.align.tiltalign.WeightWholeTracks = 1\n";
        File.WriteAllText(path: DirectiveFile, contents: BRTConfig);
        
        // Then run batchruntomo
        // Only do setup and alignment calculation if on second pass, otherwise do fiducial model generation too
        int EndingStep = Options.DoPatchTracking ? 5 : 0;
        string Arguments = $"-DirectiveFile {DirectiveFile} -CurrentLocation {T.TiltStackDir} " +
                           $"-RootName {T.RootName} -EndingStep {EndingStep}";

        if (Options.DoPatchTracking)
            Console.WriteLine($"Performing patch tracking in {T.TiltStackDir} with arguments: {Arguments}");

        // we execute batchruntomo even if not doing patch tracking to create com file for alignment

        bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string BatchRunTomoExecutable = IsWindows ? "batchruntomo.cmd" : "batchruntomo";
        Console.WriteLine($"Running '{BatchRunTomoExecutable} {Arguments}'");
        Process BatchRunTomo = new Process 
        {
            StartInfo =
            {
                FileName = BatchRunTomoExecutable,
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Minimized,
                Arguments = Arguments,
                WorkingDirectory = T.TiltStackDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }
        };
        DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
        BatchRunTomo.OutputDataReceived += Handler;
        BatchRunTomo.ErrorDataReceived += Handler;

        BatchRunTomo.Start();

        BatchRunTomo.BeginOutputReadLine();
        BatchRunTomo.BeginErrorReadLine();

        BatchRunTomo.WaitForExit();
        
        // Run alignment separately from batchruntomo to avoid expensive cross-validation calculations
        if (Options.DoTiltAlign)
        {
            Console.WriteLine($"Calculating projection parameters from patch tracking results in {T.TiltStackDir}");
            string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
            Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
            Process TiltAlign = new Process
            {
                StartInfo =
                {
                    FileName = SubMfgExecutable,
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = "align.com",
                    WorkingDirectory = T.TiltStackDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                }
            };
            TiltAlign.OutputDataReceived += Handler;
            TiltAlign.ErrorDataReceived += Handler;

            TiltAlign.Start();

            TiltAlign.BeginOutputReadLine();
            TiltAlign.BeginErrorReadLine();

            TiltAlign.WaitForExit();
        }

        if (Options.DoPatchTracking)
            Console.WriteLine($"Finished for {TiltSeriesPath}");
    }
    
    [Command(nameof(WorkerWrapper.TomoEtomoFiducials))]
    static void TomoEtomoFiducials(NamedSerializableObject Command)
    {
        string TiltSeriesPath = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoEtomoFiducials)Command.Content[1];

        TiltSeries T = new TiltSeries(TiltSeriesPath);
        
        // first generate a directive file to run Etomo automatically through batchruntomo
        int RotOption = Options.DoAxisAngleSearch ? -1 : 0; // fit single value (-1) or leave fixed (0)
        var BRTConfig = $"setupset.copyarg.userawtlt = 1\n" +
                        $"setupset.copyarg.stackext = st\n" +
                        $"setupset.copyarg.rotation = {Options.AxisAngle}\n" +
                        $"setupset.copyarg.pixel = {Options.TiltStackAngPix / 10}\n" +
                        $"setupset.copyarg.gold = {Options.FiducialSizeNanometers}\n" +
                        $"setupset.copyarg.dual = 0\n" +
                        $"comparam.xcorr.tiltxcorr.FilterRadius2 = 0.5\n" +
                        $"comparam.xcorr.tiltxcorr.ExcludeCentralPeak = 1\n" +
                        $"runtime.Fiducials.any.trackingMethod = 0\n" +
                        $"runtime.Fiducials.any.seedingMethod = 1\n" +
                        $"comparam.autofidseed.autofidseed.TargetNumberOfBeads = {Options.TargetNBeads}\n" +
                        $"comparam.align.tiltalign.MagOption = 0\n" +
                        $"comparam.align.tiltalign.TiltOption = 0\n" +
                        $"comparam.align.tiltalign.RotOption = {RotOption}\n" +
                        $"comparam.align.tiltalign.RobustFitting = 1\n" +
                        $"comparam.align.tiltalign.WeightWholeTracks = 1\n";

        var DirectiveFile = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
        DirectiveFile = Path.ChangeExtension(DirectiveFile, ".adoc");
        File.WriteAllText(path: DirectiveFile, contents: BRTConfig);
        
        // then run batchruntomo
        // only do setup and alignment calculation if on second pass, otherwise do fiducial model generation too
        int EndingStep = Options.DoFiducialTracking ? 5 : 0;
        string Arguments = $"-DirectiveFile {DirectiveFile} -CurrentLocation {T.TiltStackDir} -RootName {T.RootName} -EndingStep {EndingStep}";

        if (Options.DoFiducialTracking)
            Console.WriteLine($"Performing fiducial tracking in {T.TiltStackDir} with arguments: {Arguments}");

        // we execute batchruntomo even if not doing fiducial tracking to create com file for alignment
        bool IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        string BatchRunTomoExecutable = IsWindows ? "batchruntomo.cmd" : "batchruntomo";
        Console.WriteLine($"Running '{BatchRunTomoExecutable} {Arguments}'");
        Process BatchRunTomo = new Process 
        {
            StartInfo =
            {
                FileName = BatchRunTomoExecutable,
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Minimized,
                Arguments = Arguments,
                WorkingDirectory = T.TiltStackDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            }
        };
        DataReceivedEventHandler Handler = (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };
        BatchRunTomo.OutputDataReceived += Handler;
        BatchRunTomo.ErrorDataReceived += Handler;

        BatchRunTomo.Start();

        BatchRunTomo.BeginOutputReadLine();
        BatchRunTomo.BeginErrorReadLine();

        BatchRunTomo.WaitForExit();

        // run alignment separately from batchruntomo to avoid expensive cross-validation calculations
        if (Options.DoTiltAlign)
        {
            Console.WriteLine($"Calculating projection parameters from fiducial tracking results in {T.TiltStackDir}");
            string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
            Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
            Process TiltAlign = new Process
            {
                StartInfo =
                {
                    FileName = SubMfgExecutable,
                    CreateNoWindow = false,
                    WindowStyle = ProcessWindowStyle.Minimized,
                    Arguments = "align.com",
                    WorkingDirectory = T.TiltStackDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                }
            };
            TiltAlign.OutputDataReceived += Handler;
            TiltAlign.ErrorDataReceived += Handler;

            TiltAlign.Start();

            TiltAlign.BeginOutputReadLine();
            TiltAlign.BeginErrorReadLine();

            TiltAlign.WaitForExit();
        }

        if (Options.DoFiducialTracking)
            Console.WriteLine($"Finished for {TiltSeriesPath}");
    }
    
    [Command(nameof(WorkerWrapper.TomoProcessCTF))]
    static void TomoProcessCTF(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsMovieCTF Options = (ProcessingOptionsMovieCTF)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.ProcessCTFSimultaneous(Options);
        T.SaveMeta();

        Console.WriteLine($"Processed CTF for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoAlignLocallyWithoutReferences))]
    static void TomoAlignLocallyWithoutReferences(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoFullReconstruction Options = (ProcessingOptionsTomoFullReconstruction)Command.Content[1];

        TiltSeries T = new TiltSeries(Path);
        T.AlignLocallyWithoutReferences(Options);
        T.SaveMeta();

        Console.WriteLine($"Aligned tilts for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoReconstruct))]
    static void TomoReconstruct(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoFullReconstruction)Command.Content[1];

        string LastMessage = "";
        TiltSeries T = new TiltSeries(Path);
        T.ReconstructFull(Options, (grid, gridElements, message) =>
        {
            if (message != LastMessage)
            {
                LastMessage = message;
                Console.WriteLine(message);
            }
            return false;
        });

        Console.WriteLine($"Reconstructed full tomogram for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoMatch))]
    static void TomoMatch(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoFullMatch)Command.Content[1];
        var TemplatePath = (string)Command.Content[2];

        Image Template = Image.FromFile(TemplatePath);

        string LastMessage = "";
        TiltSeries T = new TiltSeries(Path);
        T.MatchFull(Options, Template, (grid, gridElements, message) =>
        {
            if (message != LastMessage)
            {
                LastMessage = message;
                Console.WriteLine(message);
            }
            Console.WriteLine($"{(float)gridElements / grid.Elements() * 100}%");
            return false;
        });

        Template.Dispose();

        Console.WriteLine($"Template-matched {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoExportParticleSubtomos))]
    static void TomoExportParticleSubtomos(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoSubReconstruction Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
        float3[] Coordinates = (float3[])Command.Content[2];
        float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;

        TiltSeries T = new TiltSeries(Path);
        T.ReconstructSubtomos(Options, Coordinates, Angles);
        T.SaveMeta();

        Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
    }
    
    [Command(nameof(WorkerWrapper.TomoExportParticleSeries))]
    static void TomoExportParticleSeries(NamedSerializableObject Command)
    {
        string Path = (string)Command.Content[0];
        ProcessingOptionsTomoSubReconstruction Options = (ProcessingOptionsTomoSubReconstruction)Command.Content[1];
        float3[] Coordinates = (float3[])Command.Content[2];
        float3[] Angles = Command.Content[3] != null ? (float3[])Command.Content[3] : null;
        string PathsRelativeTo = (string)Command.Content[4];
        string PathTableOut = (string)Command.Content[5];

        Star TableOut;

        TiltSeries T = new TiltSeries(Path);
        T.ReconstructParticleSeries(Options, Coordinates, Angles, PathsRelativeTo, out TableOut);
        T.SaveMeta();

        if (!string.IsNullOrEmpty(PathTableOut))
            TableOut.Save(PathTableOut);

        Console.WriteLine($"Exported {Coordinates.Length / T.NTilts} particles for {Path}");
    }
}