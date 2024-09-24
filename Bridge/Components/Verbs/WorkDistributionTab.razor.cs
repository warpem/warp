namespace Bridge.Components.Verbs
{
    public class WorkDistributionTabParams
    {
        public int WorkersPerGPU { get; set; } = 1;
        public string GPUList { get; set; } = "";
        public string RemoteWorkers { get; set; } = "";

        public void PopulateArguments(VerbCommand command)
        {
            if (WorkersPerGPU != 1)
                command.AddArgument($"--perdevice {WorkersPerGPU}");
            if (!string.IsNullOrEmpty(GPUList))
                command.AddArgument($"--device_list {GPUList}");
            if (!string.IsNullOrEmpty(RemoteWorkers))
                command.AddArgument($"--workers {RemoteWorkers}");
        }
    }
}
