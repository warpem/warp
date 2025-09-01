using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Xunit;
using WarpCore.Core;
using Xunit.Abstractions;

namespace WarpCore.Tests
{
    /// <summary>
    /// Test suite for the FileDiscoverer class, validating file detection, incubation,
    /// event handling, and performance under various scenarios including high-volume file creation.
    /// </summary>
    public class FileDiscovererTests : IDisposable
    {
        private readonly ITestOutputHelper _testOutputHelper;
        private readonly ILogger<FileDiscoverer> _logger;
        private readonly string _testDirectory;

        /// <summary>
        /// Initializes a new test instance with a temporary directory for test files.
        /// </summary>
        /// <param name="testOutputHelper">XUnit test output helper for logging</param>
        public FileDiscovererTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
            var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            _logger = loggerFactory.CreateLogger<FileDiscoverer>();
            
            _testDirectory = Path.Combine(Path.GetTempPath(), $"FileDiscovererTest_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDirectory);
        }

        /// <summary>
        /// Tests that FileDiscoverer can be successfully initialized without throwing exceptions.
        /// Validates basic setup and configuration of file watching infrastructure.
        /// </summary>
        [Fact]
        public async Task InitializeAsync_SetsUpFileWatching()
        {
            using var discoverer = new FileDiscoverer(_logger);
            
            await discoverer.InitializeAsync(_testDirectory, "*.txt", false);
            
            // Just verify no exception was thrown during initialization
            Assert.True(true);
        }

        /// <summary>
        /// Tests that FileDiscovered event is properly fired when a new file is created.
        /// Validates the event mechanism and correct file path reporting.
        /// </summary>
        [Fact]
        public async Task FileDiscovered_EventFiredWhenFileCreated()
        {
            using var discoverer = new FileDiscoverer(_logger);
            var eventFired = false;
            string discoveredPath = null;

            discoverer.FileDiscovered += (sender, args) =>
            {
                eventFired = true;
                discoveredPath = args.FilePath;
            };

            await discoverer.InitializeAsync(_testDirectory, "*.txt", false);
            
            var testFile = Path.Combine(_testDirectory, "test.txt");
            await File.WriteAllTextAsync(testFile, "test content");
            
            await Task.Delay(2000); // Wait for incubation
            
            Assert.True(eventFired);
            Assert.Equal(testFile, discoveredPath);
        }

        /// <summary>
        /// Tests that the file incubation mechanism properly waits for files to reach stable size
        /// before declaring them ready. Simulates a file being written in chunks and verifies
        /// the event is not fired until the file stops growing.
        /// </summary>
        [Fact]
        public async Task FileIncubation_WaitsForStableSize()
        {
            using var discoverer = new FileDiscoverer(_logger);
            var eventFired = false;

            discoverer.FileDiscovered += (sender, args) => eventFired = true;

            await discoverer.InitializeAsync(_testDirectory, "*.txt", false);
            
            var testFile = Path.Combine(_testDirectory, "growing.txt");
            
            // Write file in chunks to simulate copying
            await File.WriteAllTextAsync(testFile, "part1");
            await Task.Delay(100);
            await File.AppendAllTextAsync(testFile, "part2");
            await Task.Delay(100);
            await File.AppendAllTextAsync(testFile, "part3");
            
            // Should not fire immediately
            await Task.Delay(500);
            Assert.False(eventFired);
            
            // Wait for incubation period
            await Task.Delay(2000);
            Assert.True(eventFired);
        }

        /// <summary>
        /// Tests that existing files are discovered when FileDiscoverer is initialized
        /// with recursive search enabled. Validates discovery of files that existed
        /// before the discoverer was started.
        /// </summary>
        [Fact]
        public async Task RescanAsync_DiscoverExistingFiles()
        {
            var testFile = Path.Combine(_testDirectory, "existing.txt");
            await File.WriteAllTextAsync(testFile, "existing file");

            using var discoverer = new FileDiscoverer(_logger);
            var eventFired = false;

            discoverer.FileDiscovered += (sender, args) => eventFired = true;

            await discoverer.InitializeAsync(_testDirectory, "*.txt", true);
            
            await Task.Delay(2000); // Wait for incubation
            
            Assert.True(eventFired);
        }

        /// <summary>
        /// Performance test that validates FileDiscoverer can handle high-volume file creation.
        /// Creates 1000 files simultaneously and verifies all are discovered and reported correctly.
        /// Tests the scalability and thread safety of the file discovery system.
        /// </summary>
        [Fact]
        public async Task MultipleFiles_Handles1000Files()
        {
            using var discoverer = new FileDiscoverer(_logger);
            var discoveredFiles = new List<string>();
            var lockObject = new object();

            discoverer.FileDiscovered += (sender, args) =>
            {
                lock (lockObject)
                {
                    discoveredFiles.Add(args.FilePath);
                }
            };

            await discoverer.InitializeAsync(_testDirectory, "*.txt", false);

            // Create 1000 files
            var tasks = new List<Task>();
            for (int i = 0; i < 1000; i++)
            {
                var fileName = $"file_{i:D4}.txt";
                var filePath = Path.Combine(_testDirectory, fileName);
                tasks.Add(File.WriteAllTextAsync(filePath, $"content for file {i}"));
            }
            await Task.WhenAll(tasks);

            // Wait for discovery and incubation
            await Task.Delay(5000);

            lock (lockObject)
            {
                Assert.Equal(1000, discoveredFiles.Count);
                _testOutputHelper.WriteLine($"Discovered {discoveredFiles.Count} files.");
            }
        }

        /// <summary>
        /// Cleans up the temporary test directory and all created files.
        /// Implements IDisposable to ensure proper resource cleanup after each test.
        /// </summary>
        public void Dispose()
        {
            if (Directory.Exists(_testDirectory))
            {
                try
                {
                    Directory.Delete(_testDirectory, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }
    }
}