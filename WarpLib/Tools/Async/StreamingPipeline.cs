using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Warp.Tools.Async
{
    /// <summary>
    /// Multi-stage streaming pipeline with automatic queue management.
    /// Enables load → process → save patterns with concurrent execution.
    ///
    /// IMPORTANT: Stage processors that use GPU operations must call GPU.SetDevice()
    /// as background threads default to GPU 0.
    /// </summary>
    public class StreamingPipeline<TData> : IDisposable
    {
        private class PipelineStage
        {
            public string Name { get; set; }
            public Delegate Processor { get; set; }
            public BoundedQueue<object> InputQueue { get; set; }  // Input queue for this stage
            public BoundedQueue<object> OutputQueue { get; set; } // Output queue for this stage
            public Task Task { get; set; }
            public bool RunInBackground { get; set; }
            public int GpuDevice { get; set; }
            public Type InputType { get; set; }
            public Type OutputType { get; set; }
        }

        private readonly List<PipelineStage> stages = new List<PipelineStage>();
        private readonly CancellationTokenSource cancellationSource = new CancellationTokenSource();

        /// <summary>
        /// Builder for constructing multi-stage pipelines
        /// </summary>
        public class Builder
        {
            private readonly List<PipelineStage> stages = new List<PipelineStage>();

            /// <summary>
            /// Adds processing stage to pipeline
            /// </summary>
            /// <param name="name">Stage name for debugging</param>
            /// <param name="processor">Processing function</param>
            /// <param name="queueCapacity">Queue capacity between this and next stage</param>
            /// <param name="runInBackground">Run in background thread (true) or main thread (false)</param>
            /// <param name="gpuDevice">GPU device ID for this stage. -1 to not set device.</param>
            public Builder AddStage<TIn, TOut>(
                string name,
                Func<TIn, CancellationToken, TOut> processor,
                int queueCapacity = 2,
                bool runInBackground = true,
                int gpuDevice = -1)
            {
                stages.Add(new PipelineStage
                {
                    Name = name,
                    Processor = processor,
                    InputQueue = new BoundedQueue<object>(queueCapacity),
                    OutputQueue = new BoundedQueue<object>(queueCapacity),
                    RunInBackground = runInBackground,
                    GpuDevice = gpuDevice,
                    InputType = typeof(TIn),
                    OutputType = typeof(TOut)
                });

                return this;
            }

            /// <summary>
            /// Builds the configured pipeline
            /// </summary>
            public StreamingPipeline<TData> Build()
            {
                return new StreamingPipeline<TData>(stages);
            }
        }

        private StreamingPipeline(List<PipelineStage> stages)
        {
            this.stages = stages;
        }

        /// <summary>
        /// Processes all source items through the pipeline
        /// </summary>
        public void ProcessAll(IEnumerable<TData> source)
        {
            // Start all background stages
            for (int i = 0; i < stages.Count; i++)
            {
                if (stages[i].RunInBackground)
                {
                    int stageIndex = i;
                    stages[i].Task = Task.Run(() => RunStage(stageIndex));
                }
            }

            // Feed source data into first stage's input queue
            if (stages.Count > 0)
            {
                var firstStage = stages[0];

                if (firstStage.RunInBackground)
                {
                    // First stage is background, feed source to its input queue
                    foreach (var item in source)
                    {
                        firstStage.InputQueue.Enqueue(item, cancellationSource.Token);
                    }
                    firstStage.InputQueue.CompleteAdding();
                }
                else
                {
                    // First stage runs on main thread
                    RunFirstStageOnMainThread(source);
                }
            }

            // Wait for all background tasks
            foreach (var stage in stages)
            {
                if (stage.Task != null)
                {
                    try
                    {
                        stage.Task.Wait();
                    }
                    catch (AggregateException) { }
                }
            }
        }

        private void RunFirstStageOnMainThread(IEnumerable<TData> source)
        {
            var firstStage = stages[0];
            var processor = (Func<TData, CancellationToken, object>)firstStage.Processor;

            if (firstStage.GpuDevice >= 0)
                GPU.SetDevice(firstStage.GpuDevice);

            foreach (var item in source)
            {
                var result = processor(item, cancellationSource.Token);

                if (stages.Count > 1)
                {
                    firstStage.OutputQueue.Enqueue(result, cancellationSource.Token);
                    stages[1].InputQueue.Enqueue(result, cancellationSource.Token);
                }
            }

            if (stages.Count > 1)
            {
                firstStage.OutputQueue.CompleteAdding();
                stages[1].InputQueue.CompleteAdding();
            }
        }

        private void RunStage(int stageIndex)
        {
            try
            {
                var stage = stages[stageIndex];

                // Set GPU device for this stage
                if (stage.GpuDevice >= 0)
                    GPU.SetDevice(stage.GpuDevice);

                // Get processor (we need to cast based on input/output types)
                var processor = stage.Processor;
                var processMethod = processor.GetType().GetMethod("Invoke");

                BoundedQueue<object> input = stage.InputQueue;
                BoundedQueue<object> output = stage.OutputQueue;
                bool isLastStage = stageIndex == stages.Count - 1;

                foreach (var item in input.GetConsumingEnumerable(cancellationSource.Token))
                {
                    // Invoke processor
                    var result = processMethod.Invoke(processor, new object[] { item, cancellationSource.Token });

                    // Pass to next stage if not last
                    if (!isLastStage)
                    {
                        output.Enqueue(result, cancellationSource.Token);

                        // Also feed to next stage's input if it exists
                        if (stageIndex + 1 < stages.Count)
                            stages[stageIndex + 1].InputQueue.Enqueue(result, cancellationSource.Token);
                    }
                }

                // Complete output queue
                if (!isLastStage)
                {
                    output.CompleteAdding();

                    // Also complete next stage's input
                    if (stageIndex + 1 < stages.Count)
                        stages[stageIndex + 1].InputQueue.CompleteAdding();
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during cancellation
            }
        }

        public void Dispose()
        {
            cancellationSource?.Cancel();

            // Dispose all queues
            foreach (var stage in stages)
            {
                stage.InputQueue?.Dispose();
                stage.OutputQueue?.Dispose();
            }

            cancellationSource?.Dispose();
        }
    }
}
