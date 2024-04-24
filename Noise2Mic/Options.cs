using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Noise2Mic
{
    class Options
    {
        [Option("processingdir", Required = true, HelpText = "Relative path to a the root folder used during Warp's processing, i.e. where the XML metadata are.")]
        public string ProcessingDirPath { get; set; }

        [Option("start_model", Default = "", HelpText = "Name of the file with the initial (pre-trained) model.")]
        public string StartModelName { get; set; }

        [Option("old_model", Default = "", HelpText = "Name of the folder with the pre-trained model. Leave empty to train a new one.")]
        public string OldModelName { get; set; }

        [Option("learningrate_start", Default = 0.0001, HelpText = "Initial learning rate that will be decreased exponentially to reach the final learning rate.")]
        public double LearningRateStart { get; set; }

        [Option("learningrate_finish", Default = 0.000001, HelpText = "Final learning rate, after exponential decrease from the initial rate.")]
        public double LearningRateFinish { get; set; }

        [Option("dont_augment", Default = false, HelpText = "Don't augment data through random rotations. Only rotations by multiples of 180 degrees will be used.")]
        public bool DontAugment { get; set; }

        [Option("rescale_angpix", Default = -1f, HelpText = "Pixel size to scale the micrographs to during training and denoising.")]
        public float PixelSize { get; set; }

        [Option("lowpass", Default = -1f, HelpText = "Low-pass filter to be applied to denoised micrographs (in Angstroms).")]
        public float Lowpass { get; set; }

        [Option("iterations", Default = 600, HelpText = "Number of iterations.")]
        public int NIterations { get; set; }

        [Option("window", Default = 512, HelpText = "Size of the model's input window in pixels; should be a multiple of 256.")]
        public int WindowSize { get; set; }

        [Option("batchsize", Default = 64, HelpText = "Batch size for model training. Decrease if you run out of memory. The number of iterations will be adjusted automatically. Should be a multiple of the number of GPUs used in training.")]
        public int BatchSize { get; set; }

        [Option("gpuid_network", Default = new int[] { 0 }, HelpText = "Comma-separated GPU IDs used for denoiser training.")]
        public IEnumerable<int> GPUNetwork { get; set; }

        [Option("gpuid_preprocess", Default = 1, HelpText = "GPU ID used for data preprocessing. Ideally not the GPU used for training")]
        public int GPUPreprocess { get; set; }
    }
}
