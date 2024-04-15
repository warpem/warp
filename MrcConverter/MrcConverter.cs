using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using Warp;
using Warp.Headers;
using Warp.Tools;

namespace MrcConverter
{
    class MrcConverter
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            #region Command line options

            Options Options = new Options();
            string WorkingDirectory;

            string ProgramFolder = System.AppContext.BaseDirectory;
            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";
            }

            #endregion

            List<string> TopDirectories = new List<string>();

            if (Options.Recursive)
            {
                Console.Write($"Looking for 3 levels of directories in {WorkingDirectory}... ");
                TopDirectories = EnumerateDirectoriesSafely(WorkingDirectory, Options.Pattern, SearchOption.AllDirectories, 2).ToList();
                TopDirectories.RemoveAll(d => Options.Exclude.Any(d.Contains));
                Console.WriteLine($"{TopDirectories.Count} found");
            }
            else
            {
                TopDirectories.Add(WorkingDirectory);
            }

            var FoundFiles = new Dictionary<string, List<string>>();
            var DiscoveredReady = new Queue<string>();
            int TopDirectoriesProcessed = 0;

            int Done = 0;
            int OverallFound = 0;
            int NeededConversion = 0;
            long BytesSaved = 0;
            long CurrentMemory = 0;
            long MaxMemory = Options.MemGB > 0 ? ((long)Options.MemGB << 30) : long.MaxValue;
            long MaxConsumedMemory = 0;
            object SyncStats = new object();
            object SyncConsole = new object();
            object SyncMemoryWait = new object();
            object SyncMemoryChange = new object();

            Func<string, bool> SkipCondition = null;
            if (Options.Quick)
                SkipCondition = (path) =>
                {
                    HeaderMRC Header = (HeaderMRC)MapHeader.ReadFromFile(path);

                    if ((Header.Mode == MRCDataType.Float && !Options.Reverse) ||
                        (Header.Mode == MRCDataType.Half && Options.Reverse))
                        return false;
                    else
                        return true;
                };

            Thread DiscoveryThread = new Thread(() =>
            {
                Helper.ForCPU(0, TopDirectories.Count, 8, null, (i, threadID) =>
                {
                    string TopDir = TopDirectories[i];
                    //Console.WriteLine($"{TopDir} start");

                    List<string> DiscoveredFiles = new List<string>();

                    if (TopDirectories.Any(d => d != TopDir && d.IndexOf(TopDir) == 0))
                    {
                        // We already have sub-directories in TopDirectories, so only need non-recursive file search in this directory
                        DiscoveredFiles.AddRange(EnumerateFilesSafely(TopDir, Options.Pattern, SearchOption.TopDirectoryOnly, SkipCondition));
                    }
                    else
                    {
                        // We're at the bottom of the TopDirectories tree, so can search recursively if needed
                        DiscoveredFiles.AddRange(EnumerateFilesSafely(TopDir, Options.Pattern, Options.Recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly, SkipCondition));
                    }

                    lock (FoundFiles)
                    {
                        FoundFiles.Add(TopDir, DiscoveredFiles);
                        OverallFound += DiscoveredFiles.Count;
                    }
                    lock (DiscoveredReady)
                        DiscoveredReady.Enqueue(TopDir);

                    //Console.WriteLine($"{TopDir} end");
                }, null);
            });
            DiscoveryThread.Start();

            while (TopDirectoriesProcessed <  TopDirectories.Count)
            {
                while (DiscoveredReady.Count == 0)
                {
                    ClearCurrentConsoleLine();
                    Console.Write($"{NeededConversion} converted, {BytesSaved / ((long)1 << 30)} GB saved{(Options.Simulate ? " (simulated)" : "")}, " +
                                  $"looking for files in {(TopDirectories.Count - TopDirectoriesProcessed)} directories...");
                    Thread.Sleep(100);
                }

                string TopDir;
                lock (DiscoveredReady)
                    TopDir = DiscoveredReady.Dequeue();

                List<string> Paths = FoundFiles[TopDir];

                Helper.ForCPU(0, Paths.Count, Options.Threads, null, (i, threadID) =>
                {
                    string path = Paths[i];
                    bool Converted = false;
                    bool MemoryLocked = false;
                    long MemRequired = 0;
                    string Folder = Helper.PathToFolder(path);

                    bool Excluded = false;
                    if (Options.Exclude != null)
                        Excluded = Options.Exclude.Any(s => path.Contains(s));

                    try
                    {
                        if (!Excluded && !IsSymbolicLink(path))
                            try
                            {
                                HeaderMRC Header = (HeaderMRC)MapHeader.ReadFromFile(path);

                                if ((Header.Mode == MRCDataType.Float && !Options.Reverse) ||
                                    (Header.Mode == MRCDataType.Half && Options.Reverse))
                                {
                                    if (!Options.Simulate)
                                    {
                                        MemRequired = Header.Dimensions.Elements() * sizeof(float);

                                        if (MemRequired > MaxMemory)
                                            lock (SyncMemoryChange)
                                                MaxMemory = Math.Max(MaxMemory, MemRequired);

                                        lock (SyncMemoryWait)
                                        {
                                            while (CurrentMemory + MemRequired > MaxMemory)
                                                Thread.Sleep(50);

                                            lock (SyncMemoryChange)
                                            {
                                                CurrentMemory += MemRequired;
                                                MaxConsumedMemory = Math.Max(MaxConsumedMemory, CurrentMemory);
                                            }
                                        }

                                        MemoryLocked = true;

                                        Image MRC = Image.FromFile(path);
                                        string NewPath = path + ".conversiontemp";

                                        if (Options.Reverse)
                                        {
                                            Header.Mode = MRCDataType.Float;
                                            MRC.WriteMRC(NewPath, true, Header);
                                        }
                                        else
                                        {
                                            Header.Mode = MRCDataType.Half;
                                            MRC.WriteMRC16b(NewPath, true, Header);
                                        }

                                        File.Move(NewPath, path, true);

                                        MRC.Dispose();

                                        lock (SyncMemoryChange)
                                            CurrentMemory -= MemRequired;

                                        MemoryLocked = false;
                                        Converted = true;
                                    }

                                    lock (SyncStats)
                                    {
                                        NeededConversion++;
                                        if (!Options.Reverse)
                                            BytesSaved += Header.Dimensions.Elements() * 2;
                                        else
                                            BytesSaved -= Header.Dimensions.Elements() * 2;
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                lock (SyncConsole)
                                    Console.Error.WriteLine($"\nCouldn't convert {path}: {ex.Message}");

                                if (MemoryLocked)
                                    lock (SyncMemoryChange)
                                        CurrentMemory -= MemRequired;
                            }
                    }
                    catch { }

                    lock (SyncConsole)
                    {
                        Done++;
                        if (Converted || Done % 10 == 0)
                        {
                            ClearCurrentConsoleLine();
                            Console.Write($"{Done}/{OverallFound}, {NeededConversion} converted, {BytesSaved / ((long)1 << 30)} GB saved{(Options.Simulate ? " (simulated)" : "")}, {Folder}");
                        }
                    }
                }, null);

                TopDirectoriesProcessed++;
            }

            Console.Write("\n");
            Console.WriteLine($"{NeededConversion} files converted");
        }

        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth - 2));
            Console.SetCursorPosition(0, currentLineCursor);
        }

        public static bool IsSymbolicLink(string path)
        {
            var attributes = File.GetAttributes(path);
            return (attributes & FileAttributes.ReparsePoint) == FileAttributes.ReparsePoint;
        }

        public static IEnumerable<string> EnumerateDirectoriesSafely(string path, string searchPattern = "*", SearchOption searchOption = SearchOption.AllDirectories, int maxDepth = -1)
        {
            var ImmediateDirs = new List<string>();

            List<string> SubDirs = new List<string>();

            try
            {
                SubDirs = Directory.EnumerateDirectories(path).ToList();
            }
            catch (UnauthorizedAccessException) { }
            catch (PathTooLongException) { }

            ImmediateDirs.AddRange(SubDirs);

            foreach (var subDir in SubDirs)
                if (maxDepth != 0 && searchOption == SearchOption.AllDirectories)
                    ImmediateDirs.AddRange(EnumerateDirectoriesSafely(subDir, searchPattern, searchOption, maxDepth - 1));

            return ImmediateDirs;
        }

        public static IEnumerable<string> EnumerateFilesSafely(string path, string searchPattern = "*", SearchOption searchOption = SearchOption.AllDirectories, Func<string, bool> skipCondition = null)
        {
            var ImmediateFiles = new List<string>();

            if (searchOption == SearchOption.AllDirectories)
            {
                IEnumerable<string> SubDirs = new List<string>();

                try
                {
                    SubDirs = Directory.EnumerateDirectories(path);
                }
                catch (UnauthorizedAccessException) { }
                catch (PathTooLongException) { }

                foreach (var subDir in SubDirs)
                    ImmediateFiles.AddRange(EnumerateFilesSafely(subDir, searchPattern, searchOption, skipCondition));
            }

            try
            {
                var Discovered = Directory.EnumerateFiles(path, searchPattern).ToList();

                if (Discovered.Count > 0)
                {
                    if (skipCondition == null)
                        ImmediateFiles.AddRange(Discovered);
                    else
                    {
                        try
                        {
                            if (!skipCondition(Discovered.Last()))
                                ImmediateFiles.AddRange(Discovered);
                        }
                        catch
                        {
                            ImmediateFiles.AddRange(Discovered);
                        }
                    }
                }
            }
            catch (UnauthorizedAccessException) { }
            catch (PathTooLongException) { }

            return ImmediateFiles;
        }
    }
}
