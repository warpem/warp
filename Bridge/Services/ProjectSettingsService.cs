using System.Text.Json;
using System.Text.Json.Serialization;

namespace Bridge.Services
{
    public class ProjectSettingsService
    {
        public readonly string VerbSettingsDirectory = ".bridge";
        private readonly JsonSerializerOptions SerializerOptions = new JsonSerializerOptions 
        { 
            WriteIndented = true
        };
        private readonly JsonSerializerOptions DeserializerOptions = new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        public string FrameSeriesSettingsName { get; set; } = "warp_fs.settings";
        public string TiltSeriesSettingsName { get; set; } = "warp_ts.settings";

        public string GetSettingsName(SettingsType dataType)
        {
            return dataType switch
            {
                SettingsType.FrameSeries => FrameSeriesSettingsName,
                SettingsType.TiltSeries => TiltSeriesSettingsName,
                _ => string.Empty
            };
        }

        public void SaveVerbSettings(string name, object settings)
        {
            if (!Directory.Exists(VerbSettingsDirectory))
                Directory.CreateDirectory(VerbSettingsDirectory);

            File.WriteAllText(Path.Combine(VerbSettingsDirectory, name + ".json"), JsonSerializer.Serialize(settings, SerializerOptions));
        }

        public T LoadVerbSettings<T>(string name)
        {             
            if (!Directory.Exists(VerbSettingsDirectory))
                return default;

            var path = Path.Combine(VerbSettingsDirectory, name + ".json");
            if (!File.Exists(path))
                return default;

            return JsonSerializer.Deserialize<T>(File.ReadAllText(path), DeserializerOptions);
        }
    }

    public enum SettingsType
    {
        FrameSeries = 0,
        TiltSeries = 1,
        Undefined = 2
    }
}
