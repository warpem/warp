using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.XPath;

namespace Warp.Tools
{
    public static class XMLHelper
    {
        public static void WriteAttribute(XmlTextWriter writer, string name, string value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(value);
            writer.WriteEndAttribute();
        }

        public static void WriteAttribute(XmlTextWriter writer, string name, int value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndAttribute();
        }

        public static void WriteAttribute(XmlTextWriter writer, string name, float value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndAttribute();
        }

        public static void WriteAttribute(XmlTextWriter writer, string name, float[] value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(string.Join(";", value.Select(v => v.ToString(CultureInfo.InvariantCulture))));
            writer.WriteEndAttribute();
        }

        public static void WriteAttribute(XmlTextWriter writer, string name, double value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndAttribute();
        }

        public static void WriteAttribute(XmlTextWriter writer, string name, decimal value)
        {
            writer.WriteStartAttribute(name);
            writer.WriteValue(value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndAttribute();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, string value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value);
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, bool value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, int value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, long value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, float value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, float[] value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", string.Join(";", value.Select(v => v.ToString(CultureInfo.InvariantCulture))));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, double value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, decimal value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString(CultureInfo.InvariantCulture));
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, int2 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", $"{value.X},{value.Y}");
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, int3 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", $"{value.X},{value.Y},{value.Z}");
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, int4 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", $"{value.X},{value.Y},{value.Z},{value.W}");
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, float2 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", $"{value.X.ToString(CultureInfo.InvariantCulture)},{value.Y.ToString(CultureInfo.InvariantCulture)}");
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, float3 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", $"{value.X.ToString(CultureInfo.InvariantCulture)},{value.Y.ToString(CultureInfo.InvariantCulture)},{value.Z.ToString(CultureInfo.InvariantCulture)}");
            writer.WriteEndElement();
        }

        public static void WriteParamNode(XmlTextWriter writer, string name, Matrix2 value)
        {
            writer.WriteStartElement("Param");
            XMLHelper.WriteAttribute(writer, "Name", name);
            XMLHelper.WriteAttribute(writer, "Value", value.ToString());
            writer.WriteEndElement();
        }

        public static string LoadParamNode(XPathNavigator nav, string name, string defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            return Value;
        }

        public static object LoadParamNode(XPathNavigator nav, Type type, string name, object defaultValue)
        {
            if (!type.IsEnum)
                throw new Exception("Unknown type must be an enumeration.");

            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return Enum.Parse(type, Value);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static bool LoadParamNode(XPathNavigator nav, string name, bool defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return bool.Parse(Value);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static int LoadParamNode(XPathNavigator nav, string name, int defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return int.Parse(Value, CultureInfo.InvariantCulture);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static long LoadParamNode(XPathNavigator nav, string name, long defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return long.Parse(Value, CultureInfo.InvariantCulture);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static float LoadParamNode(XPathNavigator nav, string name, float defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return float.Parse(Value, CultureInfo.InvariantCulture);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static float[] LoadParamNode(XPathNavigator nav, string name, float[] defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
                    return Parts.Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static double LoadParamNode(XPathNavigator nav, string name, double defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return double.Parse(Value, CultureInfo.InvariantCulture);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static decimal LoadParamNode(XPathNavigator nav, string name, decimal defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return decimal.Parse(Value, CultureInfo.InvariantCulture);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static int2 LoadParamNode(XPathNavigator nav, string name, int2 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                    return new int2(int.Parse(Parts[0]), int.Parse(Parts[1]));
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static int3 LoadParamNode(XPathNavigator nav, string name, int3 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                    return new int3(int.Parse(Parts[0]), int.Parse(Parts[1]), int.Parse(Parts[2]));
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static int4 LoadParamNode(XPathNavigator nav, string name, int4 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                    return new int4(int.Parse(Parts[0]), int.Parse(Parts[1]), int.Parse(Parts[2]), int.Parse(Parts[3]));
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static float2 LoadParamNode(XPathNavigator nav, string name, float2 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                    return new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture), float.Parse(Parts[1], CultureInfo.InvariantCulture));
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static float3 LoadParamNode(XPathNavigator nav, string name, float3 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                    return new float3(float.Parse(Parts[0], CultureInfo.InvariantCulture), float.Parse(Parts[1], CultureInfo.InvariantCulture), float.Parse(Parts[2], CultureInfo.InvariantCulture));
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static Matrix2 LoadParamNode(XPathNavigator nav, string name, Matrix2 defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    float[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                    return new Matrix2(Parts[0], Parts[1], Parts[2], Parts[3]);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static Guid LoadParamNode(XPathNavigator nav, string name, Guid defaultValue)
        {
            XPathNodeIterator Iterator = nav.Select($"Param[@Name = \"{name}\"]");
            if (Iterator.Count == 0)
                return defaultValue;

            Iterator.MoveNext();
            string Value = Iterator.Current.GetAttribute("Value", "");
            if (Value.Length > 0)
                try
                {
                    return Guid.Parse(Value);
                }
                catch (Exception)
                { }

            return defaultValue;
        }

        public static string LoadAttribute(XPathNavigator nav, string name, string defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return Value;
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static int LoadAttribute(XPathNavigator nav, string name, int defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return int.Parse(Value, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static long LoadAttribute(XPathNavigator nav, string name, long defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return long.Parse(Value, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static float LoadAttribute(XPathNavigator nav, string name, float defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return float.Parse(Value, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static double LoadAttribute(XPathNavigator nav, string name, double defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return double.Parse(Value, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static bool LoadAttribute(XPathNavigator nav, string name, bool defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return bool.Parse(Value);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static decimal LoadAttribute(XPathNavigator nav, string name, decimal defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                return decimal.Parse(Value, CultureInfo.InvariantCulture);
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static float2 LoadAttribute(XPathNavigator nav, string name, float2 defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                return new float2(float.Parse(Parts[0], CultureInfo.InvariantCulture), float.Parse(Parts[1], CultureInfo.InvariantCulture));
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static float3 LoadAttribute(XPathNavigator nav, string name, float3 defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                return new float3(float.Parse(Parts[0], CultureInfo.InvariantCulture),
                                  float.Parse(Parts[1], CultureInfo.InvariantCulture),
                                  float.Parse(Parts[2], CultureInfo.InvariantCulture));
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static Matrix2 LoadAttribute(XPathNavigator nav, string name, Matrix2 defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                float[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                if (Parts.Length != 4)
                    throw new Exception("Expected 4 floats");

                return new Matrix2(Parts);
            }
            catch (Exception exc)
            {
                Console.WriteLine(exc.ToString());
            }

            return defaultValue;
        }

        public static int2 LoadAttribute(XPathNavigator nav, string name, int2 defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                return new int2(int.Parse(Parts[0]), int.Parse(Parts[1]));
            }
            catch (Exception)
            { }

            return defaultValue;
        }

        public static int3 LoadAttribute(XPathNavigator nav, string name, int3 defaultValue)
        {
            string Value = nav.GetAttribute(name, "");
            if (string.IsNullOrEmpty(Value))
                return defaultValue;

            try
            {
                string[] Parts = Value.Split(new[] { "," }, StringSplitOptions.RemoveEmptyEntries);
                return new int3(int.Parse(Parts[0]),
                                int.Parse(Parts[1]),
                                int.Parse(Parts[2]));
            }
            catch (Exception)
            { }

            return defaultValue;
        }
    }
}
