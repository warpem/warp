using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp
{
    public class MdocDiscoverer
    {
        int ExceptionsLogged = 0;

        List<MdocIncubatorEntry> Incubator = new List<MdocIncubatorEntry>();    // Files that could still change in size go here.
        string FolderPath = "";
        const string FileExtension = "*.mdoc";
        BackgroundWorker DiscoveryThread;
        bool ShouldAbort = false;

        FileSystemWatcher FileWatcher = null;
        bool FileWatcherRaised = false;

        List<Mdoc> Ripe = new List<Mdoc>();   // Files that haven't changed in size for a while go here.

        public event Action IncubationStarted;
        public event Action IncubationEnded;
        public event Action FilesChanged;

        Dictionary<string, Task> CreationTasks = new Dictionary<string, Task>();

        public int ExpectedNTilts = 60;
        public int AcceptAnywayAfter = -1;
        public float DefaultTiltDose = 2;

        public MdocDiscoverer()
        {
            FileWatcher = new FileSystemWatcher();
            FileWatcher.IncludeSubdirectories = false;
            FileWatcher.Changed += FileWatcher_Changed;
            FileWatcher.Created += FileWatcher_Changed;
            FileWatcher.Renamed += FileWatcher_Changed;
        }

        private void FileWatcher_Changed(object sender, FileSystemEventArgs e)
        {
            FileWatcherRaised = true;
        }

        public void ChangePath(string newPath)
        {
            IncubationEnded?.Invoke();
            
            lock (Incubator)
            {
                if (DiscoveryThread != null && DiscoveryThread.IsBusy)
                {
                    DiscoveryThread.RunWorkerCompleted += (sender, args) =>
                    {
                        FileWatcher.EnableRaisingEvents = false;
                        FileWatcherRaised = false;
                        ShouldAbort = false;

                        FolderPath = newPath;

                        if (FolderPath == "" || !Directory.Exists(FolderPath) || !IOHelper.CheckFolderPermission(FolderPath))
                            return;

                        Task.WaitAll(CreationTasks.Values.ToArray());
                        //Thread.Sleep(500);  // There might still be item creation tasks running asynchro

                        Incubator.Clear();
                        lock (Ripe)
                            Ripe.Clear();

                        FilesChanged?.Invoke();

                        FileWatcher.Path = FolderPath;
                        FileWatcher.Filter = FileExtension;
                        FileWatcher.EnableRaisingEvents = true;

                        DiscoveryThread = new BackgroundWorker();
                        DiscoveryThread.DoWork += WorkLoop;
                        DiscoveryThread.RunWorkerAsync();
                    };

                    ShouldAbort = true;
                }
                else
                {
                    FolderPath = newPath;

                    if (FolderPath == "" || !Directory.Exists(FolderPath) || !IOHelper.CheckFolderPermission(FolderPath))
                        return;

                    Incubator.Clear();
                    lock (Ripe)
                        Ripe.Clear();

                    FilesChanged?.Invoke();

                    FileWatcher.Path = FolderPath;
                    FileWatcher.Filter = FileExtension;
                    FileWatcher.EnableRaisingEvents = true;

                    DiscoveryThread = new BackgroundWorker();
                    DiscoveryThread.DoWork += WorkLoop;
                    DiscoveryThread.RunWorkerAsync();
                }
            }
        }

        void WorkLoop(object sender, EventArgs e)
        {
            while (true)
            {
                if (ShouldAbort)
                    return;

                Stopwatch Watch = new Stopwatch();
                bool EventNeedsFiring = false;
                FileWatcherRaised = false;

                try
                {
                    foreach (var fileName in Directory.EnumerateFiles(FolderPath, FileExtension, SearchOption.TopDirectoryOnly).ToArray())
                    {
                        if (ShouldAbort)
                            return;

                        if (GetMdoc(fileName) != null)
                            continue;

                        FileInfo Info = new FileInfo(fileName);
                        MdocIncubatorEntry CurrentState = GetIncubating(fileName);
                        Mdoc ParsedEntry = null;
                        int AvailableTilts = 0;
                        try
                        {
                            ParsedEntry = Mdoc.FromFile(new[] { fileName }, DefaultTiltDose);
                            AvailableTilts = ParsedEntry.Entries.Count;
                        } catch { }

                        if (CurrentState == null)
                        {
                            Stopwatch Timer = new Stopwatch();
                            Timer.Start();
                            lock (Incubator)
                            {
                                Incubator.Add(new MdocIncubatorEntry(fileName, Timer, Info.Length, AvailableTilts));
                                if (Incubator.Count == 1)
                                    IncubationStarted?.Invoke();
                            }
                        }
                        else
                        {
                            // Check if 
                            bool CanRead = false;
                            try
                            {
                                File.OpenRead(fileName).Close();
                                CanRead = true;
                            }
                            catch
                            {
                            }

                            if (ParsedEntry == null || Info.Length != CurrentState.Size || CurrentState.NTilts != AvailableTilts || !CanRead)
                            {
                                lock (Incubator)
                                    Incubator.Remove(CurrentState);
                                Stopwatch Timer = new Stopwatch();
                                Timer.Start();
                                lock (Incubator)
                                    Incubator.Add(new MdocIncubatorEntry(fileName, Timer, Info.Length, AvailableTilts));
                            }
                            else if ((CurrentState.Lifetime.ElapsedMilliseconds > 1000 && AvailableTilts >= ExpectedNTilts) ||
                                     (AcceptAnywayAfter > 0 && CurrentState.Lifetime.ElapsedMilliseconds > AcceptAnywayAfter))
                            {
                                //lock (Ripe)
                                {
                                    if (!Ripe.Exists(m => m.Path == fileName))
                                    {
                                        while (CreationTasks.Count > 15)
                                            Thread.Sleep(1);

                                        Task CreationTask = new Task(() =>
                                        {
                                            Mdoc Created = ParsedEntry;

                                            lock (Ripe)
                                            {
                                                // Make sure the list is sorted
                                                int InsertAt = 0;
                                                while (InsertAt < Ripe.Count && Ripe[InsertAt].Path.CompareTo(fileName) < 0)
                                                    InsertAt++;

                                                Ripe.Insert(InsertAt, Created);
                                            }

                                            lock (CreationTasks)
                                                CreationTasks.Remove(fileName);
                                        });

                                        lock (CreationTasks)
                                            CreationTasks.Add(fileName, CreationTask);

                                        CreationTask.Start();
                                    }
                                }

                                EventNeedsFiring = true;
                                if (!Watch.IsRunning)
                                    Watch.Start();

                                lock (Incubator)
                                {
                                    Incubator.Remove(CurrentState);
                                    if (Incubator.Count == 0)
                                        IncubationEnded?.Invoke();
                                }
                            }
                        }

                        if (EventNeedsFiring && Watch.ElapsedMilliseconds > 500)
                        {
                            //Task.WaitAll(CreationTasks.ToArray());
                            //CreationTasks.Clear();

                            Watch.Stop();
                            Watch.Reset();
                            EventNeedsFiring = false;

                            FilesChanged?.Invoke();
                        }
                    }
                }
                catch (Exception exc)
                {
                    Debug.WriteLine("FileDiscoverer crashed:");
                    Debug.WriteLine(exc);

                    if (ExceptionsLogged < 100)
                        using (TextWriter Writer = File.AppendText("d_filediscoverer.txt"))
                        {
                            Writer.WriteLine(DateTime.Now + ":");
                            Writer.WriteLine(exc.ToString());
                            Writer.WriteLine("");
                            ExceptionsLogged++;
                        }
                }

                while (CreationTasks.Count > 0)
                    Thread.Sleep(1);

                if (EventNeedsFiring)
                    FilesChanged?.Invoke();

                while (!IsIncubating() && !FileWatcherRaised)
                {
                    if (ShouldAbort)
                        return;
                    Thread.Sleep(50);
                }
            }
        }

        public void Shutdown()
        {
            lock (Incubator)
                ShouldAbort = true;
        }

        MdocIncubatorEntry GetIncubating(string path)
        {
            lock (Incubator)
                return Incubator.FirstOrDefault(entry => entry.Path == path);
        }

        Mdoc GetMdoc(string path)
        {
            lock (Ripe)
                return Ripe.FirstOrDefault(movie => movie.Path == path);
        }

        public Mdoc[] GetImmutableFiles()
        {
            lock (Ripe)
                return Ripe.ToArray();
        }

        public bool IsIncubating()
        {
            return Incubator.Count > 0;
        }
    }

    public class MdocIncubatorEntry
    {
        public string Path;
        public Stopwatch Lifetime;
        public long Size;
        public int NTilts;

        public MdocIncubatorEntry(string path, Stopwatch lifetime, long size, int ntilts)
        {
            Path = path;
            Lifetime = lifetime;
            Size = size;
            NTilts = ntilts;
        }
    }
}
