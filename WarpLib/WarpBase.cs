using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Xml;
using System.Xml.XPath;
using Warp.Tools;

namespace Warp
{
    [Serializable]
    public class WarpBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;

        public void OnPropertyChanged([CallerMemberName] string fieldName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(fieldName));
        }

        public virtual void WriteToXML(XmlTextWriter writer)
        {
            PropertyInfo[] Properties = GetType().GetProperties();
            List<PropertyInfo> SerializableProps = Properties.Where(p => p.GetCustomAttribute(typeof(WarpSerializable)) != null).ToList();
            SerializableProps.Sort((a, b) => ((WarpSerializable)a.GetCustomAttribute(typeof(WarpSerializable))).Order.CompareTo(((WarpSerializable)b.GetCustomAttribute(typeof(WarpSerializable))).Order));

            foreach (var property in SerializableProps)
            {
                if (property.PropertyType == typeof(string))
                    XMLHelper.WriteParamNode(writer, property.Name, (string)property.GetValue(this));
                else if (property.PropertyType == typeof(bool))
                    XMLHelper.WriteParamNode(writer, property.Name, (bool)property.GetValue(this));
                else if (property.PropertyType == typeof(int))
                    XMLHelper.WriteParamNode(writer, property.Name, (int)property.GetValue(this));
                else if (property.PropertyType == typeof(long))
                    XMLHelper.WriteParamNode(writer, property.Name, (long)property.GetValue(this));
                else if (property.PropertyType == typeof(float))
                    XMLHelper.WriteParamNode(writer, property.Name, (float)property.GetValue(this));
                else if (property.PropertyType == typeof(float[]))
                    XMLHelper.WriteParamNode(writer, property.Name, (float[])property.GetValue(this));
                else if (property.PropertyType == typeof(double))
                    XMLHelper.WriteParamNode(writer, property.Name, (double)property.GetValue(this));
                else if (property.PropertyType == typeof(decimal))
                    XMLHelper.WriteParamNode(writer, property.Name, (decimal)property.GetValue(this));
                else if (property.PropertyType == typeof(int2))
                    XMLHelper.WriteParamNode(writer, property.Name, (int2)property.GetValue(this));
                else if (property.PropertyType == typeof(int3))
                    XMLHelper.WriteParamNode(writer, property.Name, (int3)property.GetValue(this));
                else if (property.PropertyType == typeof(int4))
                    XMLHelper.WriteParamNode(writer, property.Name, (int4)property.GetValue(this));
                else if (property.PropertyType == typeof(float2))
                    XMLHelper.WriteParamNode(writer, property.Name, (float2)property.GetValue(this));
                else if (property.PropertyType == typeof(float3))
                    XMLHelper.WriteParamNode(writer, property.Name, (float3)property.GetValue(this));
                else if (property.PropertyType == typeof(Matrix2))
                    XMLHelper.WriteParamNode(writer, property.Name, (Matrix2)property.GetValue(this));
                else if (property.PropertyType == typeof(Guid))
                    XMLHelper.WriteParamNode(writer, property.Name, ((Guid)property.GetValue(this)).ToString());
                else if (property.PropertyType.IsEnum)
                    XMLHelper.WriteParamNode(writer, property.Name, ((int)property.GetValue(this)).ToString());
                else
                    throw new Exception("Value type not supported.");
            }
        }

        public virtual void ReadFromXML(XPathNavigator nav)
        {
            if (nav == null)
                return;

            PropertyInfo[] Properties = GetType().GetProperties();
            List<PropertyInfo> SerializableProps = Properties.Where(p => p.GetCustomAttribute(typeof(WarpSerializable)) != null).ToList();
            SerializableProps.Sort((a, b) => ((WarpSerializable)a.GetCustomAttribute(typeof(WarpSerializable))).Order.CompareTo(((WarpSerializable)b.GetCustomAttribute(typeof(WarpSerializable))).Order));

            foreach (var property in SerializableProps)
            {
                if (property.PropertyType == typeof(string))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, property.GetValue(this) == null ? "" : (string)property.GetValue(this)));
                else if (property.PropertyType == typeof(bool))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (bool)property.GetValue(this)));
                else if (property.PropertyType == typeof(int))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (int)property.GetValue(this)));
                else if (property.PropertyType == typeof(long))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (long)property.GetValue(this)));
                else if (property.PropertyType == typeof(float))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (float)property.GetValue(this)));
                else if (property.PropertyType == typeof(float[]))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (float[])property.GetValue(this)));
                else if (property.PropertyType == typeof(double))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (double)property.GetValue(this)));
                else if (property.PropertyType == typeof(decimal))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (decimal)property.GetValue(this)));
                else if (property.PropertyType == typeof(int2))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (int2)property.GetValue(this)));
                else if (property.PropertyType == typeof(int3))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (int3)property.GetValue(this)));
                else if (property.PropertyType == typeof(int4))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (int4)property.GetValue(this)));
                else if (property.PropertyType == typeof(float2))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (float2)property.GetValue(this)));
                else if (property.PropertyType == typeof(float3))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (float3)property.GetValue(this)));
                else if (property.PropertyType == typeof(Matrix2))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (Matrix2)property.GetValue(this)));
                else if (property.PropertyType == typeof(Guid))
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.Name, (Guid)property.GetValue(this)));
                else if (property.PropertyType.IsEnum)
                    property.SetValue(this, XMLHelper.LoadParamNode(nav, property.PropertyType, property.Name, property.GetValue(this)));
                else
                    throw new Exception("Value type not supported.");
            }
        }
    }

    public delegate void NotifiedPropertyChanged(object sender, object newValue);

    public class WarpSerializable : Attribute
    {
        public int Order = 0;
    }
}
