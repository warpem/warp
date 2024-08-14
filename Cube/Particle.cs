using System.Windows.Controls;
using System.Windows.Shapes;
using Warp;
using Warp.Tools;

namespace Cube
{
    public class Particle : WarpBase
    {
        private float3 _Position = new float3(0, 0, 0);
        public float3 Position
        {
            get { return _Position; }
            set { if (value != _Position) { _Position = value; OnPropertyChanged(); } }
        }

        private float3 _Angle = new float3(0, 0, 0);
        public float3 Angle
        {
            get { return _Angle; }
            set { if (value != _Angle) { _Angle = value; OnPropertyChanged(); } }
        }

        private float _Score = 0;
        public float Score
        {
            get { return _Score; }
            set { if (value != _Score) { _Score = value; OnPropertyChanged(); } }
        }

        private bool _IsSelected = false;
        public bool IsSelected
        {
            get { return _IsSelected; }
            set { if (value != _IsSelected) { _IsSelected = value; OnPropertyChanged(); } }
        }

        private Ellipse _BoxXY = null;
        public Ellipse BoxXY
        {
            get { return _BoxXY; }
            set { if (value != _BoxXY) { _BoxXY = value; OnPropertyChanged(); } }
        }

        private Ellipse _BoxZY = null;
        public Ellipse BoxZY
        {
            get { return _BoxZY; }
            set { if (value != _BoxZY) { _BoxZY = value; OnPropertyChanged(); } }
        }

        private Ellipse _BoxXZ = null;
        public Ellipse BoxXZ
        {
            get { return _BoxXZ; }
            set { if (value != _BoxXZ) { _BoxXZ = value; OnPropertyChanged(); } }
        }

        public Particle(float3 position, float3 angle)
        {
            Position = position;
            Angle = angle;
        }
    }
}