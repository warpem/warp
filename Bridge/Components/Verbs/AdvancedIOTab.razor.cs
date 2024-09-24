namespace Bridge.Components.Verbs
{
    public class AdvancedIOTabParams
    {
        public string InputData { get; set; } = "";
        public bool DoRecursive { get; set; } = false;
        public string InputProcessing { get; set; } = "";
        public string OutputProcessing { get; set; } = "";
        public bool NoRawData { get; set; } = false;

        public void PopulateArguments(VerbCommand command)
        {
            if (!string.IsNullOrEmpty(InputData))
                command.AddArgument($"--input_data {InputData}");

            if (DoRecursive)
                command.AddArgument("--input_data_recursive");

            if (!string.IsNullOrEmpty(InputProcessing))
                command.AddArgument($"--input_processing {InputProcessing}");

            if (!string.IsNullOrEmpty(OutputProcessing))
                command.AddArgument($"--output_processing {OutputProcessing}");

            if (NoRawData)
                command.AddArgument("--input_norawdata");
        }
    }
}
