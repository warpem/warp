using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Warp;
using Warp.Tools;
using Warp.Workers;

namespace WarpWorker2;

// Handlers that shell out to external alignment tools (AreTomo2/AreTomo3, IMOD's
// batchruntomo/submfg). Each ports the corresponding legacy WarpWorker dispatch
// verbatim — the heavy lifting is the external process, run synchronously to
// completion with stdout/stderr streamed to the per-item log.
static partial class WorkerProcess
{
    // Forwards a child process' stdout/stderr to the worker's console (per-item log).
    static DataReceivedEventHandler StreamToConsole =>
        (sender, args) => { if (args.Data != null) Console.WriteLine(args.Data); };

    static void RunToCompletion(Process proc, string executableName = null)
    {
        proc.OutputDataReceived += StreamToConsole;
        proc.ErrorDataReceived += StreamToConsole;

        proc.Start();

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        proc.WaitForExit();

        if (proc.ExitCode != 0)
        {
            string name = executableName ?? proc.StartInfo.FileName;
            throw new Exception($"{name} exited with code {proc.ExitCode}");
        }
    }

    [Command(WorkerCommandNames.TomoAretomo)]
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

        RunToCompletion(new Process
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
        });

        Console.WriteLine($"Executed AreTomo for {SeriesPath}");
    }

    [Command(WorkerCommandNames.TomoAretomo3)]
    static void TomoAretomo3(NamedSerializableObject Command)
    {
        string SeriesPath = (string)Command.Content[0];
        var Options = (ProcessingOptionsTomoAretomo3)Command.Content[1];

        TiltSeries T = new TiltSeries(SeriesPath);

        string StackDir = T.TiltStackDir;
        string StackPath = Path.GetFileName(T.TiltStackPath);

        // Build the AreTomo3 command
        // Use the actual stack path as the input for AreTomo3
        string BaseName = Path.GetFileNameWithoutExtension(StackPath);
        string InPrefix = BaseName;
        string InSuffix = ".st";
        string AtPatch = $"{Options.AtPatch[0]} {Options.AtPatch[1]}";
        string Axis = Options.AxisAngle.ToString() + (Options.DoAxisSearch ? " 0" : " -1");
        string AlignZ = Options.AlignZ.ToString();

        // VolZ is forced to 0 as per AreTomo3 requirements
        string Arguments = $"-InPrefix {InPrefix} -InSuffix {InSuffix} -OutDir {StackDir} " +
                           $"-TiltAxis {Axis} -AlignZ {AlignZ} -AtPatch {AtPatch} -VolZ 0 " +
                           $"-ExtZ 0 -OutImod 2 -TiltCor 1 -Cmd 1 -Serial 1 " +
                           $"-CorrCTF 0 0 -DarkTol 0 -Gpu {DeviceID}";

        Console.WriteLine($"Executing {Options.Executable} in {StackDir} with arguments: {Arguments}");

        RunToCompletion(new Process
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
        });

        Console.WriteLine($"Executed AreTomo3 for {SeriesPath}");
    }

    [Command(WorkerCommandNames.TomoEtomoFiducials)]
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
                        $"comparam.restrictalign.restrictalign.UseCrossValidation = 0\n" +           
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
        RunToCompletion(new Process
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
        });

        // run alignment separately from batchruntomo to avoid expensive cross-validation calculations
        if (Options.DoTiltAlign)
        {
            Console.WriteLine($"Calculating projection parameters from fiducial tracking results in {T.TiltStackDir}");
            string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
            Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
            RunToCompletion(new Process
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
            });
        }

        if (Options.DoFiducialTracking)
            Console.WriteLine($"Finished for {TiltSeriesPath}");
    }

    [Command(WorkerCommandNames.TomoEtomoPatchTrack)]
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
                        $"comparam.restrictalign.restrictalign.UseCrossValidation = 0\n" +    
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
        RunToCompletion(new Process
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
        });

        // Run alignment separately from batchruntomo to avoid expensive cross-validation calculations
        if (Options.DoTiltAlign)
        {
            Console.WriteLine($"Calculating projection parameters from patch tracking results in {T.TiltStackDir}");
            string SubMfgExecutable = IsWindows ? "submfg.cmd" : "submfg";
            Console.WriteLine($"Running '{SubMfgExecutable} align.com'");
            RunToCompletion(new Process
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
            });
        }

        if (Options.DoPatchTracking)
            Console.WriteLine($"Finished for {TiltSeriesPath}");
    }
}
