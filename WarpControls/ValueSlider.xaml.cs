using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Sparta
{
    /// <summary>
    /// Interaction logic for ValueSlider.xaml
    /// </summary>
    public partial class ValueSlider : UserControl
    {
        string Format = "{0:0}";

        #region Dependency properties

        public decimal Value
        {
            get { return (decimal)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty =
            DependencyProperty.Register("Value", typeof(decimal), typeof(ValueSlider), new UIPropertyMetadata(-1m, (sender, e) =>
            {
                ValueSlider Sender = (ValueSlider)sender;
                Sender.TempValue = (decimal)e.NewValue;
            }));

        public int Decimals
        {
            get { return (int)GetValue(DecimalsProperty); }
            set { SetValue(DecimalsProperty, value); }
        }
        public static readonly DependencyProperty DecimalsProperty =
            DependencyProperty.Register("Decimals", typeof(int), typeof(ValueSlider), new PropertyMetadata(0));

        public string StringValue
        {
            get { return (string)GetValue(StringValueProperty); }
            set { SetValue(StringValueProperty, value); }
        }
        public static readonly DependencyProperty StringValueProperty =
            DependencyProperty.Register("StringValue", typeof(string), typeof(ValueSlider), new PropertyMetadata("", StringValueChanged));
        private static void StringValueChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ValueSlider Sender = (ValueSlider)sender;
            if (Sender.IsPerformingManualEdit)
            {
                Sender.IsPerformingManualEdit = false;
                try
                {
                    decimal NewValue = decimal.Parse((string)e.NewValue);
                    decimal OldValue = Sender.Value;
                    Sender.Value = Math.Max(Math.Min(Math.Round(NewValue / Sender.DisplayMultiplicator / Sender.StepSize) * Sender.StepSize, Sender.MaxValue), Sender.MinValue);

                    Sender.OnValueChanged(Sender, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, NewValue));
                }
                catch
                { }
                Sender.StringValue = Sender.FormatValueForDisplay(Sender.TempValue);
            }
        }

        public string TextFormat
        {
            get { return (string)GetValue(TextFormatProperty); }
            set { SetValue(TextFormatProperty, value); }
        }
        public static readonly DependencyProperty TextFormatProperty =
            DependencyProperty.Register("TextFormat", typeof(string), typeof(ValueSlider), new PropertyMetadata("{0}"));

        public decimal TempValue
        {
            get { return (decimal)GetValue(TempValueProperty); }
            set { SetValue(TempValueProperty, value); }
        }

        public static readonly DependencyProperty TempValueProperty =
            DependencyProperty.Register("TempValue", typeof(decimal), typeof(ValueSlider),
                new UIPropertyMetadata(-1m, (sender, e) =>
                {
                    ValueSlider Sender = (ValueSlider)sender;
                    Sender.StringValue = Sender.FormatValueForDisplay(Sender.TempValue);
                }));

        public decimal StepSize
        {
            get { return (decimal)GetValue(StepSizeProperty); }
            set { SetValue(StepSizeProperty, value); }
        }
        public static readonly DependencyProperty StepSizeProperty =
            DependencyProperty.Register("StepSize", typeof(decimal), typeof(ValueSlider), new UIPropertyMetadata(1m, StepSizeChanged));
        private static void StepSizeChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ValueSlider Sender = (ValueSlider)sender;
            string DecimalSeparator = NumberFormatInfo.CurrentInfo.CurrencyDecimalSeparator;
            string StepString = (Sender.StepSize * Sender.DisplayMultiplicator).ToString();
            int Decimals = 0;
            if (StepString.Contains(DecimalSeparator))
            {
                string DecimalPart = StepString.Split(new string[] { DecimalSeparator }, StringSplitOptions.None)[1];
                while (DecimalPart.Length > 0 && DecimalPart[DecimalPart.Length - 1] == '0')
                    DecimalPart = DecimalPart.Substring(0, DecimalPart.Length - 1);
                Decimals = DecimalPart.Length;
            }

            Sender.Decimals = Decimals;
            string Format = "{0:0}";
            if (Decimals > 0)
            {
                Format = "{0:0.";
                for (int i = 0; i < Decimals; i++)
                    Format += "0";
                Format += "}";
            }
            Sender.Format = Format;
            Sender.StringValue = Sender.FormatValueForDisplay(Sender.TempValue);
        }

        public decimal MinValue
        {
            get { return (decimal)GetValue(MinValueProperty); }
            set { SetValue(MinValueProperty, value); }
        }
        public static readonly DependencyProperty MinValueProperty =
            DependencyProperty.Register("MinValue", typeof(decimal), typeof(ValueSlider),
                new UIPropertyMetadata(0m, (sender, e) =>
                {
                    ValueSlider Sender = (ValueSlider)sender;
                    if (Sender.Value < (decimal)e.NewValue)
                        Sender.Value = (decimal)e.NewValue;
                }));

        public decimal MaxValue
        {
            get { return (decimal)GetValue(MaxValueProperty); }
            set { SetValue(MaxValueProperty, value); }
        }
        public static readonly DependencyProperty MaxValueProperty =
            DependencyProperty.Register("MaxValue", typeof(decimal), typeof(ValueSlider),
                new UIPropertyMetadata(100m, (sender, e) =>
                {
                    ValueSlider Sender = (ValueSlider)sender;
                    if (Sender.Value > (decimal)e.NewValue)
                        Sender.Value = (decimal)e.NewValue;
                }));

        public bool IsExponential
        {
            get { return (bool)GetValue(IsExponentialProperty); }
            set { SetValue(IsExponentialProperty, value); }
        }
        public static readonly DependencyProperty IsExponentialProperty =
            DependencyProperty.Register("IsExponential", typeof(bool), typeof(ValueSlider), new UIPropertyMetadata(false));

        public decimal ExponentialBase
        {
            get { return (decimal)GetValue(ExponentialBaseProperty); }
            set { SetValue(ExponentialBaseProperty, value); }
        }
        public static readonly DependencyProperty ExponentialBaseProperty =
            DependencyProperty.Register("ExponentialBase", typeof(decimal), typeof(ValueSlider), new UIPropertyMetadata(2m));

        public decimal DisplayMultiplicator
        {
            get { return (decimal)GetValue(DisplayMultiplicatorProperty); }
            set { SetValue(DisplayMultiplicatorProperty, value); }
        }
        public static readonly DependencyProperty DisplayMultiplicatorProperty =
            DependencyProperty.Register("DisplayMultiplicator", typeof(decimal), typeof(ValueSlider),
                new PropertyMetadata(1.0m, (sender, e) =>
                {
                    ValueSlider Sender = (ValueSlider)sender;
                    Sender.StringValue = Sender.FormatValueForDisplay(Sender.TempValue);
                    StepSizeChanged(sender, e);
                }));

        public UpdateSourceTrigger UpdateTrigger
        {
            get { return (UpdateSourceTrigger)GetValue(UpdateTriggerProperty); }
            set { SetValue(UpdateTriggerProperty, value); }
        }
        public static readonly DependencyProperty UpdateTriggerProperty =
            DependencyProperty.Register("UpdateTrigger", typeof(UpdateSourceTrigger), typeof(ValueSlider), new PropertyMetadata(UpdateSourceTrigger.Explicit));
               
        public SolidColorBrush UnderlineColor
        {
            get { return (SolidColorBrush)GetValue(UnderlineColorProperty); }
            set { SetValue(UnderlineColorProperty, value); }
        }
        public static readonly DependencyProperty UnderlineColorProperty =
            DependencyProperty.Register("UnderlineColor", typeof(SolidColorBrush), typeof(ValueSlider), new PropertyMetadata(new SolidColorBrush(Colors.CornflowerBlue)));

        #endregion

        public event PropertyChangedCallback PreviewValueChanged;
        public event PropertyChangedCallback ValueChanged;
        public event Action ValueChangeStarted;
        public event Action ValueChangeFinished;

        private bool IsDragging = false;
        private bool WillTriggerManual = false;
        private bool IsPerformingManualEdit = false;
        private Point OldPosition;
        private System.Drawing.Point OldFormsPosition;
        private int DeltaY;
        private decimal OldValue;
        private decimal OldStepSize;

        public ValueSlider()
        {
            IsEnabledChanged += ValueSlider_IsEnabledChanged;
            InitializeComponent();
        }

        private void ValueSlider_IsEnabledChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            ValueBlock.Opacity = IsEnabled ? 1.0 : 0.5;
            LineDashes.Opacity = IsEnabled ? 1.0 : 0.5;
        }

        public void OnPreviewValueChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            PreviewValueChanged?.Invoke(sender, e);
        }

        public void OnValueChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ValueChanged?.Invoke(sender, e);
        }

        private void ValueBlock_MouseDown(object sender, MouseButtonEventArgs e)
        {
            IsDragging = true;
            WillTriggerManual = true;
            OldPosition = e.GetPosition(this);
            OldFormsPosition = System.Windows.Forms.Cursor.Position;
            OldValue = Value;
            OldStepSize = StepSize;
            DeltaY = 0;

            e.Handled = true;
        }

        private void ValueBlock_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (IsDragging)
            {
                IsDragging = false;

                System.Windows.Forms.Cursor.Show();

                decimal OldValue = Value;
                OnPreviewValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, TempValue));
                Value = TempValue;
                OnValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, Value));
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            if (Parent != null)
            {
                FrameworkElement RootParent = (FrameworkElement)Parent;
                while (RootParent.Parent != null)
                    RootParent = (FrameworkElement)RootParent.Parent;

                try
                {
                    RootParent.PreviewMouseMove += Window_PreviewMouseMove;
                    RootParent.PreviewMouseUp += Window_PreviewMouseUp;
                }
                catch
                {
                    // ignored
                }
            }
        }

        void Window_PreviewMouseUp(object sender, MouseButtonEventArgs e)
        {
            ValueChangeFinished?.Invoke();

            if (WillTriggerManual)
            {
                IsDragging = false;
                ValueBlock.TriggerEdit();
                WillTriggerManual = false;
                IsPerformingManualEdit = true;

                return;
            }

            ValueBlock_MouseUp(sender, e);
        }

        void Window_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (!IsDragging)
                return;

            if (e.LeftButton == MouseButtonState.Released)
                ValueBlock_MouseUp(sender, null);

            // Go into actual dragging mode
            if (WillTriggerManual)
            {
                WillTriggerManual = false;
                System.Windows.Forms.Cursor.Hide();
                ValueChangeStarted?.Invoke();
            }

            e.Handled = true;

            Point NewPosition = e.GetPosition(this);
            DeltaY += (int)(NewPosition.Y - OldPosition.Y);

            System.Windows.Forms.Cursor.Position = OldFormsPosition;

            int DeltaSteps = -DeltaY / 6;
            decimal DeltaValue = DeltaSteps * StepSize;
            //if (!IsExponential)
                TempValue = Math.Min(Math.Max(OldValue + DeltaValue, MinValue), MaxValue);
            //else
            //{
            //    decimal OldExponent = OldValue >= 1m ? (decimal)Math.Log((double)OldValue, (double)ExponentialBase) : -1m;
            //    decimal NewExponent = (decimal)Math.Min(Math.Max(((double)OldExponent + (double)DeltaValue), -1), Math.Log((double)MaxValue, (double)ExponentialBase));
            //    if (NewExponent >= 0m)
            //        TempValue = (decimal)Math.Max(Math.Pow((double)ExponentialBase, (double)NewExponent), (double)MinValue);
            //    else
            //        TempValue = (decimal)Math.Max(0m, MinValue);
            //}

            TempValue = Math.Round(TempValue, Decimals + 1);
            if (UpdateTrigger == UpdateSourceTrigger.PropertyChanged)
            {
                OnPreviewValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, TempValue));
                Value = TempValue;
                OnValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, Value));
            }
        }

        void Window_PreviewDragEnter(object sender, DragEventArgs e)
        {
            e.Handled = true;
        }

        private void EditPanel_OnPreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            e.Handled = true;

            if (IsDragging)
                return;

            OldValue = Value;
            OldStepSize = StepSize;
            DeltaY = 0;

            int DeltaSteps = Math.Sign(e.Delta);
            decimal DeltaValue = DeltaSteps * StepSize;
            //if (!IsExponential)
                TempValue = Math.Min(Math.Max(OldValue + DeltaValue, MinValue), MaxValue);
            //else
            //{
            //    decimal OldExponent = OldValue >= 1m ? (decimal)Math.Log((double)OldValue, (double)ExponentialBase) : -1m;
            //    decimal NewExponent = (decimal)Math.Min(Math.Max(((double)OldExponent + (double)DeltaValue), -1), Math.Log((double)MaxValue, (double)ExponentialBase));
            //    if (NewExponent >= 0m)
            //        TempValue = (decimal)Math.Max(Math.Pow((double)ExponentialBase, (double)NewExponent), (double)MinValue);
            //    else
            //        TempValue = (decimal)Math.Max(0m, MinValue);
            //}

            TempValue = Math.Round(TempValue, Decimals + 1);
            if (UpdateTrigger == UpdateSourceTrigger.PropertyChanged)
            {
                OnPreviewValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, TempValue));
                Value = TempValue;
                OnValueChanged(this, new DependencyPropertyChangedEventArgs(ValueProperty, OldValue, Value));
            }

            ValueBlock_MouseUp(null, null);
        }

        private string FormatValueForDisplay(decimal value)
        {
            if (IsExponential)
            {
                decimal OutValue = (decimal)Math.Pow((double)ExponentialBase, (double)value) * DisplayMultiplicator;

                string DecimalSeparator = NumberFormatInfo.CurrentInfo.CurrencyDecimalSeparator;
                string TempString = OutValue.ToString();
                int TempDecimals = 0;
                if (TempString.Contains(DecimalSeparator))
                {
                    string DecimalPart = TempString.Split(new [] { DecimalSeparator }, StringSplitOptions.None)[1];
                    while (DecimalPart.Length > 0 && DecimalPart[DecimalPart.Length - 1] == '0')
                        DecimalPart = DecimalPart.Substring(0, DecimalPart.Length - 1);
                    TempDecimals = DecimalPart.Length;
                }
                
                string TempFormat = "{0:0}";
                if (TempDecimals > 0)
                {
                    TempFormat = "{0:0.";
                    for (int i = 0; i < TempDecimals; i++)
                        TempFormat += "0";
                    TempFormat += "}";
                }

                return string.Format(TempFormat, OutValue);
            }
            else
                return string.Format(Format, value * DisplayMultiplicator);
        }

        private void ValueBlock_OnEditCompleted()
        {
            
        }
    }
}
