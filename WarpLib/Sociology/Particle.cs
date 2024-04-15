using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp.Sociology
{
    public class Particle
    {
        private float3[] _Coordinates;
        public float3[] Coordinates
        {
            get { return _Coordinates; }
            set
            {
                _Coordinates = value;
                _CoordinatesMean = MathHelper.Mean(value);
                TimeStepCoordinates = 1f / Math.Max(1, value.Length - 1);
            }
        }

        private float TimeStepCoordinates;
        private float3 _CoordinatesMean;
        public float3 CoordinatesMean => _CoordinatesMean;

        private float3[] _Angles;
        public float3[] Angles
        {
            get { return _Angles; }
            set
            {
                _Angles = value;
                _AnglesMean = MathHelper.Mean(value);
                TimeStepAngles = 1f / Math.Max(1, value.Length - 1);
            }
        }

        private float TimeStepAngles;
        private float3 _AnglesMean;
        public float3 AnglesMean => _AnglesMean;

        public int RandomSubset;

        public string SourceName;
        public string SourceHash;

        public float FOM;

        public Particle(float3[] coordinates, float3[] angles, int randomSubset, string sourceName, string sourceHash, float fom = 0)
        {
            Coordinates = coordinates;
            Angles = angles;
            //_CoordinatesMean = MathHelper.Mean(Coordinates);
            //_AnglesMean = MathHelper.Mean(Angles);

            RandomSubset = randomSubset;

            SourceName = sourceName;
            SourceHash = sourceHash;

            FOM = fom;
        }

        public Cubic1D GetSplineCoordinateX()
        {
            float2[] Data = new float2[Coordinates.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepCoordinates * i, Coordinates[i].X);

            return new Cubic1D(Data);
        }

        public Cubic1D GetSplineCoordinateY()
        {
            float2[] Data = new float2[Coordinates.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepCoordinates * i, Coordinates[i].Y);

            return new Cubic1D(Data);
        }

        public Cubic1D GetSplineCoordinateZ()
        {
            float2[] Data = new float2[Coordinates.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepCoordinates * i, Coordinates[i].Z);

            return new Cubic1D(Data);
        }

        public Cubic1D GetSplineAngleX()
        {
            float2[] Data = new float2[Angles.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepAngles * i, Angles[i].X);

            return new Cubic1D(Data);
        }

        public Cubic1D GetSplineAngleY()
        {
            float2[] Data = new float2[Angles.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepAngles * i, Angles[i].Y);

            return new Cubic1D(Data);
        }

        public Cubic1D GetSplineAngleZ()
        {
            float2[] Data = new float2[Angles.Length];
            for (int i = 0; i < Data.Length; i++)
                Data[i] = new float2(TimeStepAngles * i, Angles[i].Z);

            return new Cubic1D(Data);
        }

        public void ResampleCoordinates(int newResolution)
        {
            if (newResolution < 1)
                throw new Exception("New resolution should include at least 1 point.");
            if (Coordinates.Length == newResolution)
                return;

            float NewTimeStepCoordinates = 1f / Math.Max(1, newResolution - 1);
            float[] NewSamples = Helper.ArrayOfFunction(i => i * NewTimeStepCoordinates, newResolution);
            float[] NewCoordinatesX = GetSplineCoordinateX().Interp(NewSamples);
            float[] NewCoordinatesY = GetSplineCoordinateY().Interp(NewSamples);
            float[] NewCoordinatesZ = GetSplineCoordinateZ().Interp(NewSamples);

            Coordinates = Helper.Zip(NewCoordinatesX, NewCoordinatesY, NewCoordinatesZ);
            TimeStepCoordinates = NewTimeStepCoordinates;
        }

        public void ResampleAngles(int newResolution)
        {
            if (newResolution < 1)
                throw new Exception("New resolution should include at least 1 point.");
            if (Angles.Length == newResolution)
                return;

            float NewTimeStepAngles = 1f / Math.Max(1, newResolution - 1);
            float[] NewSamples = Helper.ArrayOfFunction(i => i * NewTimeStepAngles, newResolution);
            float[] NewAnglesX = GetSplineAngleX().Interp(NewSamples);
            float[] NewAnglesY = GetSplineAngleY().Interp(NewSamples);
            float[] NewAnglesZ = GetSplineAngleZ().Interp(NewSamples);

            Angles = Helper.Zip(NewAnglesX, NewAnglesY, NewAnglesZ);
            TimeStepAngles = NewTimeStepAngles;
        }

        public float3[] GetCoordinateSeries(float[] interpolationSteps)
        {
            if (Coordinates.Length == 1)
                return Helper.ArrayOfConstant(Coordinates[0], interpolationSteps.Length);
            else
                return Helper.Zip(GetSplineCoordinateX().Interp(interpolationSteps),
                                  GetSplineCoordinateY().Interp(interpolationSteps),
                                  GetSplineCoordinateZ().Interp(interpolationSteps));
        }

        public float3 GetCoordinatesAt(float interpolationStep)
        {
            if (Coordinates.Length == 1)
                return Coordinates[0];
            else
                return new float3(GetSplineCoordinateX().Interp(interpolationStep),
                                  GetSplineCoordinateY().Interp(interpolationStep),
                                  GetSplineCoordinateZ().Interp(interpolationStep));
        }

        public float3[] GetAngleSeries(float[] interpolationSteps)
        {
            if (Angles.Length == 1)
                return Helper.ArrayOfConstant(Angles[0], interpolationSteps.Length);
            else
                return Helper.Zip(GetSplineAngleX().Interp(interpolationSteps),
                                  GetSplineAngleY().Interp(interpolationSteps),
                                  GetSplineAngleZ().Interp(interpolationSteps));
        }

        public float3 GetAnglesAt(float interpolationStep)
        {
            if (Angles.Length == 1)
                return Angles[0];
            else
                return new float3(GetSplineAngleX().Interp(interpolationStep),
                                  GetSplineAngleY().Interp(interpolationStep),
                                  GetSplineAngleZ().Interp(interpolationStep));
        }

        public Particle GetCopy()
        {
            return new Particle(Coordinates.ToArray(), Angles.ToArray(), RandomSubset, SourceName, SourceHash);
        }

        public bool IsSameMicrograph(Particle other)
        {
            if (!string.IsNullOrEmpty(SourceHash) && !string.IsNullOrEmpty(other.SourceHash))
                return SourceHash.Equals(other.SourceHash);
            else
                return SourceName.Equals(other.SourceName);
        }

        public float DistanceSqFrom(Particle other)
        {
            return (CoordinatesMean - other.CoordinatesMean).LengthSq();
        }

        public float DistanceFrom(Particle other)
        {
            return (float)Math.Sqrt(DistanceSqFrom(other));
        }

        public bool IsWithinDistance(Particle other, float threshold)
        {
            threshold *= threshold;
            return DistanceSqFrom(other) <= threshold;
        }
    }
}
