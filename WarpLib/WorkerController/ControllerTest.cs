using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Warp.Tools;
using Warp.WorkerController;

namespace Warp.WorkerController
{
    /// <summary>
    /// Simple test class to verify the controller-based worker architecture
    /// </summary>
    public class ControllerTest
    {
        public static async Task RunBasicTest()
        {
            Console.WriteLine("=== Controller-Based Worker Test ===");
            
            try
            {
                // Create worker using the new integrated controller architecture
                var worker = new WorkerWrapper(0, silent: true);
                
                Console.WriteLine("1. Worker created and controller started automatically");
                Console.WriteLine($"   Controller is running on port {worker.Port}");

                Console.WriteLine("2. Worker already registered automatically");
                Console.WriteLine($"   Worker ready on device {worker.DeviceID}");

                // Wait a moment for worker to fully initialize
                await Task.Delay(2000);

                Console.WriteLine("3. Testing simple commands...");
                
                // Test GC collection (simple command)
                Console.WriteLine("   - Testing GcCollect...");
                worker.GcCollect();
                Console.WriteLine("   ✓ GcCollect completed");

                // Test waiting for async tasks
                Console.WriteLine("   - Testing WaitAsyncTasks...");
                worker.WaitAsyncTasks();
                Console.WriteLine("   ✓ WaitAsyncTasks completed");

                Console.WriteLine("4. Cleaning up...");
                worker.Dispose();

                Console.WriteLine("\n=== Test completed successfully! ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Test failed: {ex.Message}");
                Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                throw;
            }
        }

        public static async Task RunMultiWorkerTest()
        {
            Console.WriteLine("=== Multi-Worker Test ===");
            
            try
            {
                var workers = new List<WorkerWrapper>();

                Console.WriteLine("1. Creating multiple workers...");

                // Add workers for devices 0, 1, 2 (if available)
                for (int deviceId = 0; deviceId < Math.Min(3, GPU.GetDeviceCount()); deviceId++)
                {
                    try
                    {
                        var worker = new WorkerWrapper(deviceId, silent: true);
                        workers.Add(worker);
                        Console.WriteLine($"   ✓ Worker created for device {deviceId}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"   ✗ Failed to create worker for device {deviceId}: {ex.Message}");
                    }
                }

                // Wait for workers to initialize
                await Task.Delay(3000);

                Console.WriteLine("2. Submitting multiple tasks across workers...");
                for (int i = 0; i < 5; i++)
                {
                    var worker = workers[i % workers.Count];
                    worker.GcCollect();
                    Console.WriteLine($"   Submitted task {i + 1}/5 to device {worker.DeviceID}");
                }

                // Wait for tasks to complete
                await Task.Delay(5000);

                Console.WriteLine("3. Final statistics...");
                Console.WriteLine($"   Workers created: {workers.Count}");

                Console.WriteLine("4. Cleaning up...");
                foreach (var worker in workers)
                {
                    worker.Dispose();
                }
                
                Console.WriteLine("\n=== Multi-worker test completed! ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Multi-worker test failed: {ex.Message}");
                throw;
            }
        }

        public static async Task RunStressTest()
        {
            Console.WriteLine("=== Stress Test ===");
            
            try
            {
                var worker = new WorkerWrapper(0, silent: true);
                
                await Task.Delay(2000); // Let worker initialize

                Console.WriteLine("Submitting 50 tasks...");
                var startTime = DateTime.UtcNow;
                
                for (int i = 0; i < 50; i++)
                {
                    worker.GcCollect();
                    if ((i + 1) % 10 == 0)
                        Console.WriteLine($"   Submitted {i + 1} tasks...");
                }

                var elapsed = DateTime.UtcNow - startTime;
                Console.WriteLine($"Completed 50 tasks in {elapsed.TotalSeconds:F2} seconds");
                Console.WriteLine($"Average: {elapsed.TotalMilliseconds / 50:F2}ms per task");

                worker.Dispose();
                Console.WriteLine("\n=== Stress test completed! ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Stress test failed: {ex.Message}");
                throw;
            }
        }
    }
}