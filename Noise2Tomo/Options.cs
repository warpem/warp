using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;

namespace Noise2Tomo
{
    class Options
    {
        [Option("angpix", Default = -1f, HelpText = "Pixel size used for denoising.")]
        public float PixelSize { get; set; }

        [Option("angpixori", Default = -1f, HelpText = "Pixel size of the original data.")]
        public float PixelSizeOri { get; set; }

        [Option("lowpassstart", Default = 0.1f, HelpText = "Lowpass as fraction of Nyquist at the beginning of training.")]
        public float LowpassStart { get; set; }

        [Option("lowpassend", Default = 1.0f, HelpText = "Lowpass as fraction of Nyquist at the end of training.")]
        public float LowpassEnd { get; set; }

        [Option("angpixmask", Default = 40f, HelpText = "Pixel size for calculating content mask.")]
        public float PixelSizeMask { get; set; }

        [Option("maskedpercentage", Default = 50, HelpText = "Percentage of highest-intensity voxels to include in mask.")]
        public int MaskPercentage { get; set; }

        [Option("start_model", Default = "", HelpText = "Name of the file with an initial pre-trained model that will be further trained on the new data.")]
        public string StartModelName { get; set; }

        [Option("old_model", Default = "", HelpText = "Name of the file with the pre-trained model that will be used for denoising without training. Leave empty to train a new one.")]
        public string OldModelName { get; set; }

        [Option("learningrate_start", Default = 0.0001, HelpText = "Initial learning rate that will be decreased exponentially to reach the final learning rate.")]
        public double LearningRateStart { get; set; }

        [Option("learningrate_finish", Default = 0.000001, HelpText = "Final learning rate, after exponential decrease from the initial rate.")]
        public double LearningRateFinish { get; set; }

        [Option("dont_augment", Default = false, HelpText = "Don't augment data through random rotations.")]
        public bool DontAugment { get; set; }

        [Option("iterations", Default = 1000, HelpText = "Number of iterations per epoch.")]
        public int NIterations { get; set; }

        [Option("epochs", Default = 30, HelpText = "Number of epochs.")]
        public int NEpochs { get; set; }

        [Option("batchsize", Default = 4, HelpText = "Batch size for model training. Decrease if you run out of memory. The number of iterations will be adjusted automatically.")]
        public int BatchSize { get; set; }

        [Option("gpuid_network", Default = "0", HelpText = "GPU IDs used for network training, separate multiple IDs with commas, e.g. 0,1,2,3.")]
        public string GPUNetwork { get; set; }

        [Option("gpuid_preprocess", Default = 1, HelpText = "GPU ID used for data preprocessing. Ideally not the GPU used for training")]
        public int GPUPreprocess { get; set; }
    }
}
