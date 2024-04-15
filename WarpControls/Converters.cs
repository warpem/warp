using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using Warp;
using Warp.Tools;

namespace Sparta
{
    public class ImagePathConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (((string)value).Length == 0)
                return "";
            FileInfo Info = new FileInfo((string)value);
            return Info.Name;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class BoolToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return Visibility.Hidden;

            bool Value = (bool)value;
            return Value ? Visibility.Visible : Visibility.Hidden;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            Visibility Value = (Visibility)value;
            return Value == Visibility.Visible;
        }
    }

    public class BoolToVisibilityCollapsedConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null)
                return Visibility.Collapsed;

            bool Value = (bool)value;
            return Value ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            Visibility Value = (Visibility)value;
            return Value == Visibility.Visible;
        }
    }

    public class ProjectNameConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            string Original = (string)value;
            FileInfo Info = new FileInfo(Original);
            return Info.Name;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class ByteToGigabyteConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double Original = (double)(long)value;
            return Original / (double)(2L << 29);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double Converted = (double)value;
            return (long)(Converted * (double)(2L << 29));
        }
    }

    public class IsEmptyToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (value == null)
                return Visibility.Collapsed;
            else if (((ICollection)value).Count == 0)
                return Visibility.Collapsed;
            else
                return Visibility.Visible;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class RoundDoubleConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double Value = (double)value;
            int Digits = int.Parse(parameter.ToString());

            return Math.Round(Value, Digits);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return 0.0;
        }
    }

    public class DoubleArrayToFloatConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            double[] Original = (double[])value;
            float[] Converted = new float[Original.Length];
            unsafe
            {
                fixed (double* OriginalPtr = Original)
                fixed (float* ConvertedPtr = Converted)
                {
                    double* OriginalP = OriginalPtr;
                    float* ConvertedP = ConvertedPtr;
                    int Length = Original.Length;
                    for (int i = 0; i < Length; i++)
                        *ConvertedP++ = (float)(*OriginalP++);
                }
            }

            return Converted;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class TextFormatConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (values[0] == null)
                return null;

            string ValueString = values[0].ToString();
            string Format = (string)values[1];

            int ValueInt = 0;
            bool IsInt = int.TryParse(ValueString, out ValueInt);
            if (IsInt)
                return string.Format(Helper.NativeFormat, Format, ValueInt);

            double ValueDouble = 0.0;
            bool IsDouble = double.TryParse(ValueString, NumberStyles.Any, Helper.NativeFormat, out ValueDouble);
            if (IsDouble)
                return string.Format(Helper.NativeFormat, Format, ValueDouble);

            return string.Format(Helper.NativeFormat, Format, ValueString);
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class LeftPartFormatConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            string Format = (string)value;

            if (Format.Contains("{0}"))
                return Format.Split(new string[] { "{0}" }, StringSplitOptions.None)[0];
            else
                return "";
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class RightPartFormatConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            string Format = (string)value;

            if (Format.Contains("{0}"))
                return Format.Split(new string[] { "{0}" }, StringSplitOptions.None)[1];
            else
                return "";
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return null;
        }
    }

    public class ColorToInt3Converter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            int3 Value = (int3)value;

            return Color.FromRgb((byte)Value.X, (byte)Value.Y, (byte)Value.Z);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            Color Value = (Color)value;

            return new int3(Value.R, Value.G, Value.B);
        }
    }

    public class ColorToSolidColorBrushValueConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            if (null == value)
                return null;

            // For a more sophisticated converter, check also the targetType and react accordingly..
            if (value is Color color)
                return new SolidColorBrush(color);

            return null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            // If necessary, here you can convert back. Check if which brush it is (if its one),
            // get its Color-value and return it.

            throw new NotImplementedException();
        }
    }
}
