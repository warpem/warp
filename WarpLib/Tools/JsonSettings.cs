using System.Text.Json;
using System.Text.Json.Serialization;
using Warp.Tools;

namespace Warp.Tools
{
    /// <summary>
    /// Provides common JSON serialization options for the WarpCore application.
    /// Ensures consistent serialization behavior across API responses, test clients, and internal communication.
    /// </summary>
    public static class JsonSettings
    {
        /// <summary>
        /// Common JsonSerializerOptions used throughout the application.
        /// Configured with camelCase property naming, null value exclusion, and custom converters
        /// including the NamedSerializableObject converter for worker communication.
        /// </summary>
        public static readonly JsonSerializerOptions Default = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }
}