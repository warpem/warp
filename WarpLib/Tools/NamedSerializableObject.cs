using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Warp.Tools
{
    [Serializable]
    [JsonConverter(typeof(NamedSerializableObjectConverter))]
    public class NamedSerializableObject
    {
        public string Name { get; set; }
        public object[] Content { get; set; }

        public NamedSerializableObject() { }

        public NamedSerializableObject(string name, params object[] content)
        {
            Name = name;
            Content = content;
        }
    }

    public class NamedSerializableObjectConverter : JsonConverter<NamedSerializableObject>
    {
        public override NamedSerializableObject Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            var namedSerializableObject = new NamedSerializableObject();
            var contentList = new List<object>();

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                {
                    namedSerializableObject.Content = contentList.ToArray();
                    return namedSerializableObject;
                }

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var propertyName = reader.GetString();
                    reader.Read();

                    switch (propertyName?.ToLower())
                    {
                        case "name":
                            namedSerializableObject.Name = reader.GetString();
                            break;
                        case "content":
                            if (reader.TokenType != JsonTokenType.StartArray)
                            {
                                throw new JsonException();
                            }

                            while (reader.Read())
                            {
                                if (reader.TokenType == JsonTokenType.EndArray)
                                {
                                    break;
                                }

                                var typeName = reader.GetString();
                                var objectType = Type.GetType(typeName);

                                reader.Read();
                                var item = JsonSerializer.Deserialize(ref reader, objectType, options);
                                contentList.Add(item);
                            }

                            break;
                    }
                }
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, NamedSerializableObject value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WriteString("Name", value.Name);

            writer.WritePropertyName("Content");
            writer.WriteStartArray(); 
            
            foreach (var item in value.Content)
            {
                writer.WriteStringValue(item.GetType().AssemblyQualifiedName);
                JsonSerializer.Serialize(writer, item, item.GetType(), options);
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }
    }
}
