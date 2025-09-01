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
    /// <summary>
    /// Discovers and monitors files in a specified directory, ensuring they are stable
    /// before notifying listeners. Uses an incubation system to prevent processing
    /// files that are still being written. Supports file system watching for real-time
    /// discovery and periodic scanning for robustness.
    /// </summary>
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
        
        /// <summary>
        /// Event raised when a stable file has been discovered and is ready for processing.
        /// </summary>
        public event EventHandler<FileDiscoveredEventArgs> FileDiscovered;
        
        /// <summary>
        /// Event raised when file incubation begins (when files start being monitored for stability).
        /// </summary>
        public event EventHandler IncubationStarted;
        
        /// <summary>
        /// Event raised when file incubation ends (when all files have been determined stable or removed).
        /// </summary>
        public event EventHandler IncubationEnded;
        
        /// <summary>
        /// Event raised when the file system state changes (new files detected).
        /// </summary>
        public event EventHandler FilesChanged;

        /// <summary>
        /// Initializes a new file discoverer with default incubation period of 1 second.
        /// </summary>
        /// <param name="logger">Logger for recording file discovery operations</param>
        public FileDiscoverer(ILogger<FileDiscoverer> logger)
        {
            _logger = logger;
            _incubationMilliseconds = 1000;
        }

        /// <summary>
        /// Initializes the file discoverer with the specified directory, pattern, and search options.
        /// Sets up file system watching and starts the discovery and incubation threads.
        /// </summary>
        /// <param name="dataDirectory">Directory to monitor for files</param>
        /// <param name="filePattern">File pattern to match (e.g., "*.tiff")</param>
        /// <param name="recursiveSearch">Whether to search subdirectories recursively</param>
        /// <returns>Task representing the initialization operation</returns>
        /// <exception cref="ArgumentNullException">Thrown when required parameters are null</exception>
        /// <exception cref="DirectoryNotFoundException">Thrown when the data directory doesn't exist</exception>
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

        /// <summary>
        /// Changes the file discovery configuration. Stops current operations and reinitializes
        /// with the new settings. Used for dynamic reconfiguration during runtime.
        /// </summary>
        /// <param name="dataDirectory">New directory to monitor</param>
        /// <param name="filePattern">New file pattern to match</param>
        /// <param name="recursiveSearch">Whether to search subdirectories recursively</param>
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

        /// <summary>
        /// Sets up the file system watcher for real-time file detection.
        /// Configures the watcher to monitor file creation, changes, and renames.
        /// </summary>
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

        /// <summary>
        /// Event handler for file system events. Sets a flag to trigger
        /// immediate discovery scanning.
        /// </summary>
        /// <param name="sender">Event sender (FileSystemWatcher)</param>
        /// <param name="e">File system event arguments</param>
        private void OnFileSystemEvent(object sender, FileSystemEventArgs e)
        {
            _fileWatcherRaised = true;
        }

        /// <summary>
        /// Starts the background threads for file discovery and incubation processing.
        /// The discovery thread finds new files, while the incubation thread monitors
        /// them for stability before declaring them ready.
        /// </summary>
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

        /// <summary>
        /// Background thread worker that continuously scans the data directory for new files.
        /// Enumerates files matching the pattern and adds unknown files to the processing queue.
        /// Runs until termination is requested.
        /// </summary>
        /// <param name="sender">Event sender (BackgroundWorker)</param>
        /// <param name="e">Background worker event arguments</param>
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

        /// <summary>
        /// Background thread worker that processes files through the incubation system.
        /// Monitors newly discovered files and files being incubated, determining when
        /// they are stable and ready for processing. Runs until termination is requested.
        /// </summary>
        /// <param name="sender">Event sender (BackgroundWorker)</param>
        /// <param name="e">Background worker event arguments</param>
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

        /// <summary>
        /// Tests whether a file can be opened for reading, which indicates it's not
        /// currently being written to by another process.
        /// </summary>
        /// <param name="filePath">Path to the file to test</param>
        /// <returns>True if the file can be read, false if it's locked or inaccessible</returns>
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

        /// <summary>
        /// Gets a value indicating whether any files are currently being incubated
        /// (monitored for stability).
        /// </summary>
        /// <returns>True if files are being incubated, false otherwise</returns>
        public bool IsIncubating()
        {
            lock (_incubatorLock)
            {
                return _incubator.Count > 0;
            }
        }

        /// <summary>
        /// Gets an array of all files that have completed incubation and are ready for processing.
        /// Thread-safe operation that returns a snapshot of the ripe files list.
        /// </summary>
        /// <returns>Array of file paths that are ready for processing</returns>
        public string[] GetRipeFiles()
        {
            lock (_ripeFilesLock)
            {
                return _ripeFiles.ToArray();
            }
        }

        /// <summary>
        /// Triggers an immediate rescan of the data directory for new files.
        /// Useful for forcing discovery without waiting for the next scheduled scan.
        /// </summary>
        /// <returns>Task representing the rescan trigger operation</returns>
        public Task RescanAsync()
        {
            _logger.LogInformation("Rescanning for files");
            _fileWatcherRaised = true;
            return Task.CompletedTask;
        }

        /// <summary>
        /// Disposes the file discoverer, stopping all background threads and cleaning up resources.
        /// Ensures proper shutdown of file system monitoring and thread operations.
        /// </summary>
        public void Dispose()
        {
            _shouldAbort = true;
            _fileWatcher?.Dispose();
            _discoveryThread?.Dispose();
            _incubationThread?.Dispose();
        }
    }
}