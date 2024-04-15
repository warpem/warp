using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Warp.Tools;

namespace Warp.Controls
{
    /// <summary>
    /// Interaction logic for ReconstructionMiniature.xaml
    /// </summary>
    public partial class ReconstructionMiniature : UserControl
    {
        public int3 GridSize
        {
            get { return (int3)GetValue(GridSizeProperty); }
            set { SetValue(GridSizeProperty, value); }
        }
        public static readonly DependencyProperty GridSizeProperty = DependencyProperty.Register("GridSize", typeof(int3), typeof(ReconstructionMiniature), new PropertyMetadata(new int3(1), (o, args) => ((ReconstructionMiniature)o).Render()));

        public int Value
        {
            get { return (int)GetValue(ValueProperty); }
            set { SetValue(ValueProperty, value); }
        }
        public static readonly DependencyProperty ValueProperty = DependencyProperty.Register("Value", typeof(int), typeof(ReconstructionMiniature), new PropertyMetadata(0, (o, args) => ((ReconstructionMiniature)o).Render()));

        public float AxisRatio
        {
            get { return (float)GetValue(AxisRatioProperty); }
            set { SetValue(AxisRatioProperty, value); }
        }
        public static readonly DependencyProperty AxisRatioProperty = DependencyProperty.Register("AxisRatio", typeof(float), typeof(ReconstructionMiniature), new PropertyMetadata(3f, (o, args) => ((ReconstructionMiniature)o).Render()));

        public ReconstructionMiniature()
        {
            InitializeComponent();
            this.SizeChanged += (o, args) => Render();
        }

        void Render()
        {
            CanvasFill.Children.Clear();
            CanvasLines.Children.Clear();

            if (GridSize.Elements() < 1 || this.ActualWidth == 0)
                return;

            float StepLength = (float)(this.ActualWidth / (GridSize.X + GridSize.Y)) / (float)Math.Sqrt(1f - 1f / AxisRatio / AxisRatio);
            float2 StepZ = new float2(0, -StepLength);
            float2 StepY = new float2(-(float)Math.Sqrt(1f - 1f / AxisRatio / AxisRatio) * StepLength, StepLength / AxisRatio);
            float2 StepX = new float2((float)Math.Sqrt(1f - 1f / AxisRatio / AxisRatio) * StepLength, StepLength / AxisRatio);

            double DesiredHeight = GridSize.X * Math.Abs(StepX.Y) + GridSize.Y * Math.Abs(StepY.Y) + GridSize.Z * Math.Abs(StepZ.Y);
            if (DesiredHeight != this.Height)
            {
                this.Height = DesiredHeight;
                return;
            }

            float2 Origin = new float2(GridSize.X * StepX.X, -GridSize.Z * StepZ.Y);

            Func<int, int, int, bool> CubeExists = (x, y, z) => (z * GridSize.Y + y) * GridSize.X + x < Value && x < GridSize.X && y < GridSize.Y && z < GridSize.Z && x >= 0 && y >= 0 && z >= 0;
            SolidColorBrush LineBrush = new SolidColorBrush(Colors.Black);
            SolidColorBrush BoundsBrush = new SolidColorBrush(Color.FromArgb(40, 0, 0, 0));
            SolidColorBrush FaceLeftBrush = new SolidColorBrush(Color.FromArgb(60, Colors.CornflowerBlue.R, Colors.CornflowerBlue.G, Colors.CornflowerBlue.B));
            SolidColorBrush FaceRightBrush = new SolidColorBrush(Color.FromArgb(120, Colors.CornflowerBlue.R, Colors.CornflowerBlue.G, Colors.CornflowerBlue.B));
            SolidColorBrush FaceTopBrush = new SolidColorBrush(Color.FromArgb(30, Colors.CornflowerBlue.R, Colors.CornflowerBlue.G, Colors.CornflowerBlue.B));

            for (int z = 0; z < GridSize.Z; z++)
            {
                for (int y = 0; y < GridSize.Y; y++)
                {
                    for (int x = 0; x < GridSize.X; x++)
                    {
                        if (!CubeExists(x, y, z))
                            break;

                        #region Vertical Edges

                        // Left edge
                        if (!CubeExists(x - 1, y, z) && !CubeExists(x - 1, y + 1, z) && !CubeExists(x, y + 1, z))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                Y1 = Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y,

                                X2 = Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        // Front edge
                        if ((!CubeExists(x + 1, y, z) && !CubeExists(x + 1, y + 1, z) && !CubeExists(x, y + 1, z)) || (CubeExists(x + 1, y, z) && !CubeExists(x + 1, y + 1, z) && CubeExists(x, y + 1, z)))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                Y1 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        // Right edge
                        if (!CubeExists(x + 1, y, z) && !CubeExists(x + 1, y - 1, z) && !CubeExists(x, y - 1, z))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 0) * StepZ.X,
                                Y1 = Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 0) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        #endregion

                        #region Horizontal edges

                        #region Bottom edges

                        // Left edge
                        if ((!CubeExists(x, y + 1, z) && !CubeExists(x, y + 1, z - 1) && !CubeExists(x, y, z - 1)) || (!CubeExists(x, y + 1, z) && CubeExists(x, y + 1, z - 1)))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                Y1 = Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        // Right edge
                        if ((!CubeExists(x + 1, y, z) && !CubeExists(x + 1, y, z - 1) && !CubeExists(x, y, z - 1)) || (!CubeExists(x + 1, y, z) && CubeExists(x + 1, y, z - 1)))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 0) * StepZ.X,
                                Y1 = Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 0) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        #endregion

                        #region Middle edges

                        // Left edge
                        if (!CubeExists(x, y + 1, z) && !CubeExists(x, y + 1, z + 1) && !CubeExists(x, y, z + 1))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y1 = Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        // Right edge
                        if (!CubeExists(x + 1, y, z) && !CubeExists(x + 1, y, z + 1) && !CubeExists(x, y, z + 1))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                Y1 = Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        #endregion

                        #region Top edges

                        // Left edge
                        if (!CubeExists(x - 1, y, z) && !CubeExists(x - 1, y, z + 1) && !CubeExists(x, y, z + 1))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 0) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                Y1 = Origin.Y + (x + 0) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y,

                                X2 = Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        // Right edge
                        if (!CubeExists(x, y - 1, z) && !CubeExists(x, y - 1, z + 1) && !CubeExists(x, y, z + 1))
                        {
                            Line EdgeLine = new Line()
                            {
                                X1 = Origin.X + (x + 0) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                Y1 = Origin.Y + (x + 0) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y,

                                X2 = Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                Y2 = Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y,

                                StrokeThickness = 1,
                                Stroke = LineBrush
                            };
                            CanvasLines.Children.Add(EdgeLine);
                        }

                        #endregion

                        #endregion

                        #region Faces

                        // Left
                        if (!CubeExists(x, y + 1, z))
                        {
                            Polygon Poly = new Polygon()
                            {
                                Points = new PointCollection()
                                {
                                    new Point(Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                              Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y),
                                    new Point(Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y)
                                },
                                Fill = new SolidColorBrush(Color.FromArgb(60,
                                                                          (byte)((x + 1) / (float)GridSize.X * 255),
                                                                          (byte)((y + 1) / (float)GridSize.Y * 255),
                                                                          (byte)((z + 1) / (float)GridSize.Z * 255)))
                            };
                            CanvasFill.Children.Add(Poly);
                        }

                        // Right
                        if (!CubeExists(x + 1, y, z))
                        {
                            Polygon Poly = new Polygon()
                            {
                                Points = new PointCollection()
                                {
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 0) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 0) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 0) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 0) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y)
                                },
                                Fill = new SolidColorBrush(Color.FromArgb(120,
                                                                          (byte)((x + 1) / (float)GridSize.X * 255),
                                                                          (byte)((y + 1) / (float)GridSize.Y * 255),
                                                                          (byte)((z + 1) / (float)GridSize.Z * 255)))
                            };
                            CanvasFill.Children.Add(Poly);
                        }

                        // Top
                        if (!CubeExists(x, y, z + 1))
                        {
                            Polygon Poly = new Polygon()
                            {
                                Points = new PointCollection()
                                {
                                    new Point(Origin.X + (x + 0) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 0) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 0) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 0) * StepY.Y + (z + 1) * StepZ.Y),
                                    new Point(Origin.X + (x + 1) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 1) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y),
                                    new Point(Origin.X + (x + 0) * StepX.X + (y + 1) * StepY.X + (z + 1) * StepZ.X,
                                              Origin.Y + (x + 0) * StepX.Y + (y + 1) * StepY.Y + (z + 1) * StepZ.Y)
                                },
                                Fill = new SolidColorBrush(Color.FromArgb(30,
                                                                          (byte)((x + 1) / (float)GridSize.X * 255),
                                                                          (byte)((y + 1) / (float)GridSize.Y * 255),
                                                                          (byte)((z + 1) / (float)GridSize.Z * 255)))
                            };
                            CanvasFill.Children.Add(Poly);
                        }

                        #endregion
                    }
                }
            }

            #region Bounding box

            // Top left
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 0 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 0 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 0 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 0 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Top right
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 0 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 0 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Middle left
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 0 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 0 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Middle right
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 1 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 1 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Bottom left
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 0 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 0 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Bottom right
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 1 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 1 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Vertical left
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 0 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 0 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 0 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 0 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Vertical middle
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 1 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 1 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            // Vertical right
            {
                Line EdgeLine = new Line
                {
                    X1 = Origin.X + 1 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 0 * GridSize.Z * StepZ.X,
                    Y1 = Origin.Y + 1 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 0 * GridSize.Z * StepZ.Y,

                    X2 = Origin.X + 1 * GridSize.X * StepX.X + 0 * GridSize.Y * StepY.X + 1 * GridSize.Z * StepZ.X,
                    Y2 = Origin.Y + 1 * GridSize.X * StepX.Y + 0 * GridSize.Y * StepY.Y + 1 * GridSize.Z * StepZ.Y,

                    StrokeThickness = 1,
                    Stroke = BoundsBrush
                };
                CanvasLines.Children.Add(EdgeLine);
            }

            #endregion
        }
    }
}
