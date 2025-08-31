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
    public class FileDiscovererTests : IDisposable
    {
        private readonly ITestOutputHelper _testOutputHelper;
        private readonly ILogger<FileDiscoverer> _logger;
        private readonly string _testDirectory;

        public FileDiscovererTests(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
            var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            _logger = loggerFactory.CreateLogger<FileDiscoverer>();
            
            _testDirectory = Path.Combine(Path.GetTempPath(), $"FileDiscovererTest_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDirectory);
        }

        [Fact]
        public async Task InitializeAsync_SetsUpFileWatching()
        {
            using var discoverer = new FileDiscoverer(_logger);
            
            await discoverer.InitializeAsync(_testDirectory, "*.txt", false);
            
            // Just verify no exception was thrown during initialization
            Assert.True(true);
        }

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