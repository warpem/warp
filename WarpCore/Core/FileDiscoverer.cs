using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace WarpCore.Core
{
    public class FileDiscoverer : IDisposable
    {
        private readonly ILogger<FileDiscoverer> _logger;
        
        private string _dataDirectory;
        private string _filePattern;
        private bool _recursiveSearch;
        private int _incubationMilliseconds;
        
        private BackgroundWorker _discoveryThread;
        private BackgroundWorker _incubationThread;
        private bool _shouldAbort;
        
        private readonly ConcurrentQueue<string> _newFilesQueue = new ConcurrentQueue<string>();
        private readonly List<FileIncubationInfo> _incubator = new List<FileIncubationInfo>();
        private readonly List<string> _ripeFiles = new List<string>();
        private readonly object _incubatorLock = new object();
        private readonly object _ripeFilesLock = new object();
        
        private FileSystemWatcher _fileWatcher;
        private volatile bool _fileWatcherRaised;
        
        private int _exceptionsLogged = 0;
        private const int MaxExceptionsToLog = 100;
        
        public event EventHandler<FileDiscoveredEventArgs> FileDiscovered;
        public event EventHandler IncubationStarted;
        public event EventHandler IncubationEnded;
        public event EventHandler FilesChanged;

        public FileDiscoverer(ILogger<FileDiscoverer> logger)
        {
            _logger = logger;
            _incubationMilliseconds = 1000;
        }

        public Task InitializeAsync(string dataDirectory, string filePattern, bool recursiveSearch = false)
        {
            _dataDirectory = dataDirectory ?? throw new ArgumentNullException(nameof(dataDirectory));
            _filePattern = filePattern ?? throw new ArgumentNullException(nameof(filePattern));
            _recursiveSearch = recursiveSearch;

            if (!Directory.Exists(_dataDirectory))
                throw new DirectoryNotFoundException($"Data directory not found: {_dataDirectory}");

            _logger.LogInformation($"Initializing file discoverer for directory: {_dataDirectory}, pattern: {_filePattern}");

            SetupFileWatcher();
            StartDiscoveryThreads();

            return Task.CompletedTask;
        }

        public void ChangeConfiguration(string dataDirectory, string filePattern, bool recursiveSearch = false)
        {
            StopThreads();
            
            _dataDirectory = dataDirectory;
            _filePattern = filePattern;
            _recursiveSearch = recursiveSearch;
            
            if (string.IsNullOrEmpty(_dataDirectory) || !Directory.Exists(_dataDirectory))
                return;

            ClearState();
            SetupFileWatcher();
            StartDiscoveryThreads();
        }

        private void SetupFileWatcher()
        {
            try
            {
                _fileWatcher?.Dispose();
                _fileWatcher = new FileSystemWatcher(_dataDirectory)
                {
                    Filter = _filePattern,
                    IncludeSubdirectories = _recursiveSearch,
                    NotifyFilter = NotifyFilters.FileName | NotifyFilters.CreationTime
                };

                _fileWatcher.Created += OnFileSystemEvent;
                _fileWatcher.Changed += OnFileSystemEvent;
                _fileWatcher.Renamed += OnFileSystemEvent;
                _fileWatcher.EnableRaisingEvents = true;

                _logger.LogDebug("File system watcher enabled");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not set up file system watcher");
            }
        }

        private void OnFileSystemEvent(object sender, FileSystemEventArgs e)
        {
            _fileWatcherRaised = true;
        }

        private void StartDiscoveryThreads()
        {
            _shouldAbort = false;
            
            _discoveryThread = new BackgroundWorker();
            _discoveryThread.DoWork += DiscoveryThreadWork;
            _discoveryThread.RunWorkerAsync();

            _incubationThread = new BackgroundWorker();
            _incubationThread.DoWork += IncubationThreadWork;
            _incubationThread.RunWorkerAsync();
        }

        private void StopThreads()
        {
            _shouldAbort = true;
            
            if (_discoveryThread?.IsBusy == true)
            {
                _discoveryThread.RunWorkerCompleted += (sender, args) => { };
            }
            
            if (_incubationThread?.IsBusy == true)
            {
                _incubationThread.RunWorkerCompleted += (sender, args) => { };
            }
        }

        private void DiscoveryThreadWork(object sender, DoWorkEventArgs e)
        {
            while (!_shouldAbort)
            {
                try
                {
                    bool foundNewFiles = false;
                    var searchOption = _recursiveSearch ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                    
                    foreach (var filePath in Directory.EnumerateFiles(_dataDirectory, _filePattern, searchOption))
                    {
                        if (_shouldAbort) return;

                        var fileName = Path.GetFileName(filePath);
                        if (fileName.StartsWith(".")) continue;

                        if (!IsFileKnown(filePath))
                        {
                            _newFilesQueue.Enqueue(filePath);
                            foundNewFiles = true;
                        }
                    }

                    if (foundNewFiles)
                    {
                        FilesChanged?.Invoke(this, EventArgs.Empty);
                    }
                }
                catch (Exception ex)
                {
                    LogException(ex, "Error in discovery thread");
                }

                WaitForNextDiscoveryRound();
            }
        }

        private void IncubationThreadWork(object sender, DoWorkEventArgs e)
        {
            var eventTimer = new Stopwatch();
            bool eventNeedsFiring = false;

            while (!_shouldAbort)
            {
                try
                {
                    ProcessNewFiles();
                    ProcessIncubatingFiles(ref eventTimer, ref eventNeedsFiring);
                    
                    if (eventNeedsFiring && eventTimer.ElapsedMilliseconds > 500)
                    {
                        eventTimer.Reset();
                        eventNeedsFiring = false;
                        FilesChanged?.Invoke(this, EventArgs.Empty);
                    }
                }
                catch (Exception ex)
                {
                    LogException(ex, "Error in incubation thread");
                }

                Thread.Sleep(50);
            }
        }

        private void ProcessNewFiles()
        {
            while (_newFilesQueue.TryDequeue(out string filePath))
            {
                if (_shouldAbort) return;

                try
                {
                    if (!File.Exists(filePath)) continue;

                    var fileInfo = new FileInfo(filePath);
                    var incubationInfo = new FileIncubationInfo(filePath, fileInfo.Length);

                    lock (_incubatorLock)
                    {
                        bool wasEmpty = _incubator.Count == 0;
                        _incubator.Add(incubationInfo);
                        
                        if (wasEmpty)
                        {
                            IncubationStarted?.Invoke(this, EventArgs.Empty);
                        }
                    }

                    _logger.LogDebug($"Started incubating file: {Path.GetFileName(filePath)}");
                }
                catch (Exception ex)
                {
                    LogException(ex, $"Error processing new file {filePath}");
                }
            }
        }

        private void ProcessIncubatingFiles(ref Stopwatch eventTimer, ref bool eventNeedsFiring)
        {
            List<FileIncubationInfo> toRemove = new List<FileIncubationInfo>();
            List<string> newRipeFiles = new List<string>();

            lock (_incubatorLock)
            {
                foreach (var incubatingFile in _incubator.ToList())
                {
                    if (_shouldAbort) return;

                    try
                    {
                        if (!File.Exists(incubatingFile.FilePath))
                        {
                            toRemove.Add(incubatingFile);
                            continue;
                        }

                        var fileInfo = new FileInfo(incubatingFile.FilePath);
                        bool canRead = CanReadFile(incubatingFile.FilePath);

                        if (fileInfo.Length != incubatingFile.FileSize || !canRead)
                        {
                            incubatingFile.FileSize = fileInfo.Length;
                            incubatingFile.LastChecked = DateTime.UtcNow;
                            incubatingFile.Timer.Restart();
                        }
                        else if (incubatingFile.Timer.ElapsedMilliseconds > _incubationMilliseconds)
                        {
                            toRemove.Add(incubatingFile);
                            newRipeFiles.Add(incubatingFile.FilePath);
                            
                            eventNeedsFiring = true;
                            if (!eventTimer.IsRunning)
                                eventTimer.Start();
                        }
                    }
                    catch (Exception ex)
                    {
                        LogException(ex, $"Error checking incubating file {incubatingFile.FilePath}");
                        toRemove.Add(incubatingFile);
                    }
                }

                foreach (var item in toRemove)
                {
                    _incubator.Remove(item);
                }

                if (_incubator.Count == 0 && toRemove.Any())
                {
                    IncubationEnded?.Invoke(this, EventArgs.Empty);
                }
            }

            lock (_ripeFilesLock)
            {
                _ripeFiles.AddRange(newRipeFiles);
            }

            foreach (var ripeFile in newRipeFiles)
            {
                try
                {
                    var fileInfo = new FileInfo(ripeFile);
                    var fileName = Path.GetFileName(ripeFile);
                    
                    FileDiscovered?.Invoke(this, new FileDiscoveredEventArgs(
                        ripeFile, fileName, fileInfo.Length, DateTime.UtcNow));
                        
                    _logger.LogDebug($"File ready: {fileName}");
                }
                catch (Exception ex)
                {
                    LogException(ex, $"Error notifying about ripe file {ripeFile}");
                }
            }
        }

        private bool CanReadFile(string filePath)
        {
            try
            {
                using (var stream = File.OpenRead(filePath))
                {
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private bool IsFileKnown(string filePath)
        {
            lock (_incubatorLock)
            {
                if (_incubator.Any(i => i.FilePath == filePath))
                    return true;
            }

            lock (_ripeFilesLock)
            {
                if (_ripeFiles.Contains(filePath))
                    return true;
            }

            return false;
        }

        private void WaitForNextDiscoveryRound()
        {
            var startTime = DateTime.UtcNow;
            
            while (!_shouldAbort && DateTime.UtcNow - startTime < TimeSpan.FromSeconds(5))
            {
                if (_fileWatcherRaised)
                {
                    _fileWatcherRaised = false;
                    break;
                }
                Thread.Sleep(100);
            }
        }

        private void ClearState()
        {
            lock (_incubatorLock)
            {
                _incubator.Clear();
            }

            lock (_ripeFilesLock)
            {
                _ripeFiles.Clear();
            }

            while (_newFilesQueue.TryDequeue(out _)) { }
        }

        private void LogException(Exception ex, string context)
        {
            if (_exceptionsLogged < MaxExceptionsToLog)
            {
                _logger.LogWarning(ex, context);
                _exceptionsLogged++;
            }
        }

        public bool IsIncubating()
        {
            lock (_incubatorLock)
            {
                return _incubator.Count > 0;
            }
        }

        public string[] GetRipeFiles()
        {
            lock (_ripeFilesLock)
            {
                return _ripeFiles.ToArray();
            }
        }

        public Task RescanAsync()
        {
            _logger.LogInformation("Rescanning for files");
            _fileWatcherRaised = true;
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            _shouldAbort = true;
            _fileWatcher?.Dispose();
            _discoveryThread?.Dispose();
            _incubationThread?.Dispose();
        }
    }
}