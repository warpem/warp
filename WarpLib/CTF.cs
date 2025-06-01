using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp
{
    public class CTF : WarpBase
    {
        private decimal _PixelSize = 1.0M;
        /// <summary>
        /// Pixel size in Angstrom
        /// </summary>
        [WarpSerializable]
        public decimal PixelSize
        {
            get { return _PixelSize; }
            set { if (value != _PixelSize) { _PixelSize = value; OnPropertyChanged(); } }
        }

        private decimal _PixelSizeDeltaPercent = 0M;
        /// <summary>
        /// Pixel size anisotropy delta in percent (so it is invariant to scaling of PixelSize)
        /// </summary>
        [WarpSerializable]
        public decimal PixelSizeDeltaPercent
        {
            get { return _PixelSizeDeltaPercent; }
            set { if (value != _PixelSizeDeltaPercent) { _PixelSizeDeltaPercent = value; OnPropertyChanged(); } }
        }

        private decimal _PixelSizeAngle = 0M;
        /// <summary>
        /// Pixel size anisotropy angle in degrees
        /// </summary>
        [WarpSerializable]
        public decimal PixelSizeAngle
        {
            get { return _PixelSizeAngle; }
            set
            {
                value = (decimal)Accord.Math.Tools.Mod((double)value + 180.0, 180.0);
                if (value != _PixelSizeAngle)
                {
                    _PixelSizeAngle = value;
                    OnPropertyChanged();
                }
            }
        }

        private Matrix2 _Distortion = new Matrix2(1, 0, 0, 1);
        /// <summary>
        /// Linear distortion matrix
        /// </summary>
        [WarpSerializable]
        public Matrix2 Distortion
        {
            get { return _Distortion; }
            set { if (value != _Distortion) { _Distortion = value; OnPropertyChanged(); } }
        }

        private decimal _Cs = 2.7M;
        /// <summary>
        /// Spherical aberration in mm
        /// </summary>
        [WarpSerializable]
        public decimal Cs
        {
            get { return _Cs; }
            set { if (value != _Cs) { _Cs = value; OnPropertyChanged(); } }
        }

        private decimal _Cc = 2.7M;
        /// <summary>
        /// Chromatic aberration in mm
        /// </summary>
        [WarpSerializable]
        public decimal Cc
        {
            get { return _Cc; }
            set { if (value != _Cc) { _Cc = value; OnPropertyChanged(); } }
        }

        private decimal _Voltage = 300.0M;
        /// <summary>
        /// Voltage in kV
        /// </summary>
        [WarpSerializable]
        public decimal Voltage
        {
            get { return _Voltage; }
            set { if (value != _Voltage) { _Voltage = value; OnPropertyChanged(); } }
        }

        private decimal _Defocus = 1.0M;
        /// <summary>
        /// Defocus in um, underfocus (first peak positive) is positive
        /// </summary>
        [WarpSerializable]
        public decimal Defocus
        {
            get { return _Defocus; }
            set { if (value != _Defocus) { _Defocus = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusDelta = 0M;
        /// <summary>
        /// Astigmatism delta defocus in um
        /// </summary>
        [WarpSerializable]
        public decimal DefocusDelta
        {
            get { return _DefocusDelta; }
            set { if (value != _DefocusDelta) { _DefocusDelta = value; OnPropertyChanged(); } }
        }

        private decimal _DefocusAngle = 0M;
        /// <summary>
        /// Astigmatism angle in degrees
        /// </summary>
        [WarpSerializable]
        public decimal DefocusAngle
        {
            get { return _DefocusAngle; }
            set
            {
                //value = (decimal) Accord.Math.Tools.Mod((double)value + 180.0, 180.0);
                if (value != _DefocusAngle)
                {
                    _DefocusAngle = value; OnPropertyChanged();
                }
            }
        }

        private decimal _Amplitude = 0.07M;
        /// <summary>
        /// Amplitude contrast
        /// </summary>
        [WarpSerializable]
        public decimal Amplitude
        {
            get { return _Amplitude; }
            set { if (value != _Amplitude) { _Amplitude = value; OnPropertyChanged(); } }
        }

        private decimal _Bfactor = 0M;
        /// <summary>
        /// B factor in Angstrom^2
        /// </summary>
        [WarpSerializable]
        public decimal Bfactor
        {
            get { return _Bfactor; }
            set { if (value != _Bfactor) { _Bfactor = value; OnPropertyChanged(); } }
        }

        private decimal _BfactorDelta = 0M;
        /// <summary>
        /// B factor anisotropy in Angstrom^2
        /// </summary>
        [WarpSerializable]
        public decimal BfactorDelta
        {
            get { return _BfactorDelta; }
            set { if (value != _BfactorDelta) { _BfactorDelta = value; OnPropertyChanged(); } }
        }

        private decimal _BfactorAngle = 0M;
        /// <summary>
        /// B factor anisotropy X axis angle in degrees
        /// </summary>
        [WarpSerializable]
        public decimal BfactorAngle
        {
            get { return _BfactorAngle; }
            set { if (value != _BfactorAngle) { _BfactorAngle = value; OnPropertyChanged(); } }
        }

        private decimal _Scale = 1.0M;
        /// <summary>
        /// Scale, i. e. CTF oscillates within [-Scale; +Scale]
        /// </summary>
        [WarpSerializable]
        public decimal Scale
        {
            get { return _Scale; }
            set { if (value != _Scale) { _Scale = value; OnPropertyChanged(); } }
        }

        private decimal _PhaseShift = 0M;
        /// <summary>
        /// Phase shift in Pi
        /// </summary>
        [WarpSerializable]
        public decimal PhaseShift
        {
            get { return _PhaseShift; }
            set { if (value != _PhaseShift) { _PhaseShift = value; OnPropertyChanged(); } }
        }

        private float2 _BeamTilt = new float2(0, 0);
        /// <summary>
        /// Beam tilt in millirad
        /// </summary>
        [WarpSerializable]
        public float2 BeamTilt
        {
            get { return _BeamTilt; }
            set { if (value != _BeamTilt) { _BeamTilt = value; OnPropertyChanged(); } }
        }

        private float3 _BeamTilt2 = new float3(1, 0, 1);
        /// <summary>
        /// Higher-order beam tilt in millirad
        /// </summary>
        [WarpSerializable]
        public float3 BeamTilt2
        {
            get { return _BeamTilt2; }
            set { if (value != _BeamTilt2) { _BeamTilt2 = value; OnPropertyChanged(); } }
        }

        private float[] _ZernikeCoeffsOdd = new float[0];
        [WarpSerializable]
        public float[] ZernikeCoeffsOdd
        {
            get { return _ZernikeCoeffsOdd; }
            set { if (value != _ZernikeCoeffsOdd) { _ZernikeCoeffsOdd = value; OnPropertyChanged(); } }
        }

        private float[] _ZernikeCoeffsEven = new float[0];
        [WarpSerializable]
        public float[] ZernikeCoeffsEven
        {
            get { return _ZernikeCoeffsEven; }
            set { if (value != _ZernikeCoeffsEven) { _ZernikeCoeffsEven = value; OnPropertyChanged(); } }
        }

        private decimal _Thickness = 0;
        /// <summary>
        /// Thickness in nm
        /// </summary>
        [WarpSerializable]
        public decimal Thickness
        {
            get { return _Thickness; }
            set { if (value != _Thickness) { _Thickness = value; OnPropertyChanged(); } }
        }

        private static List<ZernikeCacheEntry> ZernikeOddCache = new List<ZernikeCacheEntry>();
        private static List<ZernikeCacheEntry> ZernikeEvenCache = new List<ZernikeCacheEntry>();

        public void FromStruct(CTFStruct s)
        {
            _PixelSize = (decimal)s.PixelSize * 1e10M;
            _PixelSizeDeltaPercent = (decimal)s.PixelSizeDelta;
            _PixelSizeAngle = (decimal)s.PixelSizeAngle * (180M / (decimal)Math.PI);

            _Cs = (decimal)s.Cs * 1e3M;
            _Voltage = (decimal)s.Voltage * 1e-3M;
            _Amplitude = (decimal)s.Amplitude;

            _Defocus = (decimal)-s.Defocus * 1e6M;
            _DefocusDelta = (decimal)-s.DefocusDelta * 1e6M;
            _DefocusAngle = (decimal)s.AstigmatismAngle * (180M / (decimal)Math.PI);

            //_Bfactor = (decimal) s.Bfactor * 1e20M;
            _Scale = (decimal)s.Scale;

            _PhaseShift = (decimal)s.PhaseShift / (decimal)Math.PI;

            OnPropertyChanged("");
        }

        public CTFStruct ToStruct()
        {
            CTFStruct Result = new CTFStruct();

            Result.PixelSize = (float)(PixelSize * 1e-10M);
            Result.PixelSizeDelta = (float)(PixelSize * PixelSizeDeltaPercent * 1e-10M);
            Result.PixelSizeAngle = (float)PixelSizeAngle * Helper.ToRad;// / (float)(180.0 / Math.PI);

            Result.Cs = (float)(Cs * 1e-3M);
            Result.Voltage = (float)(Voltage * 1e3M);
            Result.Amplitude = (float)Amplitude;

            Result.Defocus = (float)(-Defocus * 1e-6M);
            Result.DefocusDelta = (float)(-DefocusDelta * 1e-6M);
            Result.AstigmatismAngle = (float)DefocusAngle * Helper.ToRad;// / (float)(180.0 / Math.PI);

            Result.Bfactor = (float)(Bfactor * 1e-20M);
            Result.BfactorDelta = (float)(BfactorDelta * 1e-20M);
            Result.BfactorAngle = (float)BfactorAngle * Helper.ToRad;// / (float)(180.0 / Math.PI);
            Result.Scale = (float)Scale;

            Result.PhaseShift = (float)PhaseShift * Helper.PI;

            return Result;
        }

        public float[] Get1D(int width, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = ArrayPool<float>.Rent(width);

            double ny = 0.5 / (double)PixelSize / width;

            float defocus = -(float)Defocus * 1e4f;
            float amplitude = (float)Amplitude;
            float scale = ignorescale ? 1 : (float)Scale;
            float phaseshift = (float)PhaseShift * MathF.PI;

            (float K1, float K2, float K3, float K4) = GetKs();

            for (int i = 0; i < width; i++)
            {
                float freq = i * (float)ny;
                float r2 = freq * freq;
                float r4 = r2 * r2;

                float deltaf = defocus;
                float argument = (float)(K1 * deltaf * r2 + K2 * r4 - phaseshift);
                float retval = amplitude * MathF.Cos(argument) - K3 * MathF.Sin(argument);

                if (K4 != 0 && !ignorebfactor)
                    retval *= MathF.Exp(K4 * r2);

                if (ampsquared)
                    retval = Math.Abs(retval);

                Output[i] = retval * scale;
            }

            return Output;
        }

        public float Get1D(float freq, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float defocus = -(float)Defocus * 1e4f;
            float amplitude = (float)Amplitude;
            float scale = ignorescale ? 1 : (float)Scale;
            float phaseshift = (float)PhaseShift * MathF.PI;

            (float K1, float K2, float K3, float K4) = GetKs();

            float r2 = freq * freq;
            float r4 = r2 * r2;

            float deltaf = defocus;
            float argument = (float)(K1 * deltaf * r2 + K2 * r4 - phaseshift);
            float retval = amplitude * MathF.Cos(argument) - K3 * MathF.Sin(argument);

            if (K4 != 0 && !ignorebfactor)
                retval *= MathF.Exp(K4 * r2);

            if (ampsquared)
                retval = Math.Abs(retval);

            return scale * retval;
        }

        private (float K1, float K2, float K3, float K4) GetKs()
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = (float)(12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6)));
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float K1 = MathF.PI * lambda;
            float K2 = MathF.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = MathF.Sqrt(1f - amplitude * amplitude);
            float K4 = (float)Bfactor * 0.25f;

            return (K1, K2, K3, K4);
        }

        public double GetPhaseAtFrequency(double freq)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double defocus = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double K1 = Math.PI * lambda;
            double K2 = Math.PI * 0.5f * cs * lambda * lambda * lambda;

            double r2 = freq * freq;
            double r4 = r2 * r2;

            double deltaf = defocus;
            double argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;

            return argument;
        }

        public double GetFrequencyAtPhase(double phase)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double z = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double a = Math.PI * lambda;
            double b = Math.PI * 0.5f * cs * lambda * lambda * lambda;

            double Root1 = Math.Sqrt(a * a * z * z + 4 * b * (phase + phaseshift));
            double Frac1 = -(Root1 + a * z) / b;
            double Frac2 = Math.Sqrt(Frac1) / Math.Sqrt(2.0);

            return Frac2;
        }

        public double[] Get1DDouble(int width, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            double[] Output = new double[width];

            double ny = 0.5 / (double)PixelSize / width;

            for (int i = 0; i < width; i++)
                Output[i] = Get1DDouble(i * ny, ampsquared, ignorebfactor, ignorescale);

            return Output;
        }

        public double Get1DDouble(double freq, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            double voltage = (double)Voltage * 1e3;
            double lambda = 12.2643247 / Math.Sqrt(voltage * (1.0 + voltage * 0.978466e-6));
            double defocus = -(double)Defocus * 1e4;
            double cs = (double)Cs * 1e7;
            double amplitude = (double)Amplitude;
            double scale = (double)Scale;
            double phaseshift = (double)PhaseShift * Math.PI;
            double K1 = Math.PI * lambda;
            double K2 = Math.PI * 0.5f * cs * lambda * lambda * lambda;
            double K3 = Math.Sqrt(1f - amplitude * amplitude);
            double K4 = (double)Bfactor * 0.25f;

            double r2 = freq * freq;
            double r4 = r2 * r2;

            double deltaf = defocus;
            double argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
            double retval = amplitude * Math.Cos(argument) - K3 * Math.Sin(argument);

            if (K4 != 0)
                retval *= Math.Exp(K4 * r2);

            if (ampsquared)
                retval = Math.Abs(retval);

            return scale * retval;
        }

        public float[] Get2D(float2[] coordinates, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = new float[coordinates.Length];

            float pixelsize = (float)PixelSize;
            float pixeldelta = (float)(PixelSize * PixelSizeDeltaPercent);
            float pixelangle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float scale = (float)Scale;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Atan(amplitude / Math.Sqrt(1 - amplitude * amplitude));
            //float K4 = (float)Bfactor * 0.25f;

            Parallel.For(0, coordinates.Length, i =>
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float)Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3;
                float retval = -(float)Math.Sin(argument);

                if (!ignorebfactor && (Bfactor != 0 || BfactorDelta != 0))
                {
                    float BfactorAniso = (float)Bfactor * 0.25f;
                    if (BfactorDelta != 0)
                        BfactorAniso += ((float)BfactorDelta * 0.25f) * (float)Math.Cos(2.0 * (angle - (float)BfactorAngle));

                    retval *= (float)Math.Exp(BfactorAniso * r2);
                }

                if (ampsquared)
                    retval = Math.Abs(retval);// * retval);

                if (!ignorescale)
                    Output[i] = scale * retval;
                else
                    Output[i] = retval;
            });

            return Output;
        }

        public void Get2DP(float2[] coordinates, float2[] coordinatesP, bool isPositive, float pAngle, float[] result)
        {
            float pixelsize = (float)PixelSize;
            float pixeldelta = (float)(PixelSize * PixelSizeDeltaPercent);
            float pixelangle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Atan(amplitude / Math.Sqrt(1 - amplitude * amplitude));
            //float K4 = (float)Bfactor * 0.25f;

            if (pAngle >= (float)Math.PI)
            {
                pAngle -= (float)Math.PI;
                isPositive = !isPositive;
            }

            for (int i = 0; i < coordinates.Length; i++)
            {
                float angle = coordinates[i].Y;
                float angleP = coordinatesP[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float)Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3 + (float)Math.PI / 2;

                float reverse = isPositive ? 1 : -1;
                if (angleP <= pAngle)
                    reverse *= -1;
                if (pAngle == 0 && Math.Abs(angleP - Math.PI) < 1e-5)
                    reverse *= -1;

                result[i * 2] = (float)Math.Cos(argument);
                result[i * 2 + 1] = (float)Math.Sin(argument) * reverse;
            }
        }

        public void Get2DPQFromPrecomp(float2[] coordinates, float[] pqSign, float[] resultP)
        {
            float pixelsize = (float)PixelSize;
            float pixeldelta = (float)(PixelSize * PixelSizeDeltaPercent);
            float pixelangle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Atan(amplitude / Math.Sqrt(1 - amplitude * amplitude));
            //float K4 = (float)Bfactor * 0.25f;

            for (int i = 0; i < coordinates.Length; i++)
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float)Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3 + (float)Math.PI / 2;

                float reverse = pqSign[i];

                resultP[i * 2] = (float)Math.Cos(argument);
                resultP[i * 2 + 1] = (float)Math.Sin(argument) * reverse;
            }
        }

        public float[] Get2DP(float2[] coordinates, float2[] coordinatesP, bool isPositive, float pAngle)
        {
            float[] Output = new float[coordinates.Length * 2];

            Get2DP(coordinates, coordinatesP, isPositive, pAngle, Output);

            return Output;
        }

        public void GetEwaldWeights(float2[] coordinates, float particleDiameter, float[] result)
        {
            float pixelsize = (float)PixelSize;
            float pixeldelta = (float)(PixelSize * PixelSizeDeltaPercent);
            float pixelangle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Atan(amplitude / Math.Sqrt(1 - amplitude * amplitude));

            for (int i = 0; i < coordinates.Length; i++)
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float)Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3;

                float aux = 2 * Math.Abs(deltaf) * lambda * r / particleDiameter;
                float A = (float)((aux > 1) ? 0 : (2 / Math.PI) * (Math.Acos(aux) - aux * Math.Sin(Math.Acos(aux))));

                result[i] = (float)(1 + A * (2 * Math.Abs(-Math.Sin(argument)) - 1)) * 0.5f;
            }
        }

        public float[] GetEwaldWeights(float2[] coordinates, float particleDiameter)
        {
            float[] Output = new float[coordinates.Length];

            float pixelsize = (float)PixelSize;
            float pixeldelta = (float)(PixelSize * PixelSizeDeltaPercent);
            float pixelangle = (float)PixelSizeAngle / (float)(180.0 / Math.PI);
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Atan(amplitude / Math.Sqrt(1 - amplitude * amplitude));

            for (int i = 0; i < coordinates.Length; i++)
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X / (pixelsize + pixeldelta * (float)Math.Cos(2.0 * (angle - pixelangle)));
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3;

                float aux = 2 * Math.Abs(deltaf) * lambda * r / particleDiameter;
                float A = (float)((aux > 1) ? 0 : (2 / Math.PI) * (Math.Acos(aux) - aux * Math.Sin(Math.Acos(aux))));

                Output[i] = (float)(1 + A * (2 * Math.Abs(-Math.Sin(argument)) - 1)) * 0.5f;
            }

            return Output;
        }

        public float[] Get2DFromScaledCoords(float2[] coordinates, bool ampsquared, bool ignorebfactor = false, bool ignorescale = false)
        {
            float[] Output = ArrayPool<float>.Rent(coordinates.Length);

            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float defocus = -(float)Defocus * 1e4f;
            float defocusdelta = -(float)DefocusDelta * 1e4f * 0.5f;
            float astigmatismangle = (float)DefocusAngle / (float)(180.0 / Math.PI);
            float cs = (float)Cs * 1e7f;
            float amplitude = (float)Amplitude;
            float scale = (float)Scale;
            float phaseshift = (float)PhaseShift * (float)Math.PI;
            float K1 = (float)Math.PI * lambda;
            float K2 = (float)Math.PI * 0.5f * cs * lambda * lambda * lambda;
            float K3 = (float)Math.Sqrt(1f - amplitude * amplitude);
            //float K4 = (float)Bfactor * 0.25f;

            Parallel.For(0, coordinates.Length, i =>
            {
                float angle = coordinates[i].Y;
                float r = coordinates[i].X;
                float r2 = r * r;
                float r4 = r2 * r2;

                float deltaf = defocus + defocusdelta * (float)Math.Cos(2.0 * (angle - astigmatismangle));
                float argument = K1 * deltaf * r2 + K2 * r4 - phaseshift;
                float retval = amplitude * (float)Math.Cos(argument) - K3 * (float)Math.Sin(argument);

                if (!ignorebfactor && (Bfactor != 0 || BfactorDelta != 0))
                {
                    float BfactorAniso = (float)Bfactor * 0.25f;
                    if (BfactorDelta != 0)
                        BfactorAniso += ((float)BfactorDelta * 0.25f) * (float)Math.Cos(2.0 * (angle - (float)BfactorAngle));

                    retval *= (float)Math.Exp(BfactorAniso * r2);
                }

                if (ampsquared)
                    retval = Math.Abs(retval);// * retval);

                if (!ignorescale)
                    Output[i] = scale * retval;
                else
                    Output[i] = retval;
            });

            return Output;
        }

        public float[] GetPeaks()
        {
            List<float> Result = new List<float>();

            float[] Values = Get1D(1 << 12, true);
            float[] dValues = MathHelper.Diff(Values);

            for (int i = 0; i < dValues.Length - 1; i++)
                if (Math.Sign(dValues[i]) > 0 && Math.Sign(dValues[i + 1]) < 0)
                    Result.Add(0.5f * i / Values.Length);

            return Result.ToArray();
        }

        public float[] GetZeros()
        {
            List<float> Result = new List<float>();

            float[] Values = Get1D(1 << 12, true);
            float[] dValues = MathHelper.Diff(Values);

            for (int i = 0; i < dValues.Length - 1; i++)
                if (Math.Sign(dValues[i]) < 0 && Math.Sign(dValues[i + 1]) > 0)
                    Result.Add(0.5f * i / Values.Length);

            return Result.ToArray();
        }

        public float GetEwaldRadius(int size, float pixelSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));

            return size * pixelSize / lambda;
        }

        public Image GetEwaldLerpWeights(int size, float pixelSize)
        {
            float Radius = GetEwaldRadius(size, pixelSize);
            float DiameterInv = Radius == 0 ? 0 : (1.0f / (2.0f * Radius));

            Image Result = new Image(new int3(size, size, 1), true, true);
            float[] ResultData = Result.GetHost(Intent.Write)[0];

            int i = 0;
            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float r2 = xx * xx + yy * yy;
                float z = r2 * DiameterInv;

                ResultData[i * 2] = 1;
                ResultData[i * 2 + 1] = Math.Min(1, z);

                i++;
            });

            return Result;
        }

        public Image GetBeamTilt(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[] Data = new float2[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                float delta_phase = factor * q * (yy * _BeamTilt.Y + xx * _BeamTilt.X);
                delta_phase = delta_phase * Helper.ToRad;
                Data[y * (size / 2 + 1) + x] = new float2((float)Math.Cos(delta_phase), (float)Math.Sin(delta_phase));
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public Image GetBeamTilt(int size, int originalSize, float2[] beamTilts)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[][] Data = Helper.ArrayOfFunction(i => new float2[new int2(size).ElementsFFT()], beamTilts.Length);

            for (int z = 0; z < beamTilts.Length; z++)
            {
                float2 t = beamTilts[z];
                Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
                {
                    float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                    float delta_phase = factor * q * (yy * t.Y + xx * t.X);
                    delta_phase = delta_phase * Helper.ToRad;
                    Data[z][y * (size / 2 + 1) + x] = new float2((float)Math.Cos(delta_phase), (float)Math.Sin(delta_phase));
                });
            }

            return new Image(Data, new int3(size, size, beamTilts.Length), true);
        }

        public Image GetBeamTiltCoords(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float2[] Data = new float2[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                Data[y * (size / 2 + 1) + x] = new float2(factor * q * xx, factor * q * yy) * Helper.ToRad;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public Image GetBeamTiltPhase(int size, int originalSize)
        {
            float voltage = (float)Voltage * 1e3f;
            float lambda = 12.2643247f / (float)Math.Sqrt(voltage * (1.0f + voltage * 0.978466e-6f));
            float boxsize = (float)PixelSize * originalSize;

            float factor = 0.360f * (float)Cs * 10000000 * lambda * lambda / (boxsize * boxsize * boxsize);

            float[] Data = new float[new int2(size).ElementsFFT()];

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float q = _BeamTilt2.X * xx * xx + 2f * _BeamTilt2.Y * yy * xx + _BeamTilt2.Z * yy * yy;
                float delta_phase = factor * q * (yy * _BeamTilt.Y + xx * _BeamTilt.X);
                delta_phase = delta_phase * Helper.ToRad;
                Data[y * (size / 2 + 1) + x] = delta_phase;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public float[] GetZernikeOddPhases(float angPix, int size, int icoeff)
        {
            lock (ZernikeOddCache)
            {
                int EntryID = ZernikeOddCache.FindIndex(v => v.AngPix == angPix && v.Size == size && v.iCoeff == icoeff);
                if (EntryID >= 0)
                    return ZernikeOddCache[EntryID].Data;

                int m, n;
                Zernike.OddIndexToMN(icoeff, out m, out n);

                float[] NewData = new float[new int2(size).ElementsFFT()];
                int i = 0;
                float Norm = 1f / (size * angPix);
                Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
                {
                    NewData[i++] = (float)Zernike.Zcart(m, n, xx * Norm, yy * Norm);
                });

                ZernikeOddCache.Add(new ZernikeCacheEntry(angPix, size, icoeff, NewData));
                return NewData;
            }
        }

        public float[] GetZernikeEvenPhases(float angPix, int size, int icoeff)
        {
            lock (ZernikeEvenCache)
            {
                int EntryID = ZernikeEvenCache.FindIndex(v => v.AngPix == angPix && v.Size == size && v.iCoeff == icoeff);
                if (EntryID >= 0)
                    return ZernikeEvenCache[EntryID].Data;

                int m, n;
                Zernike.EvenIndexToMN(icoeff, out m, out n);

                float[] NewData = new float[new int2(size).ElementsFFT()];
                int i = 0;
                float Norm = 1f / (size * angPix);
                Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
                {
                    NewData[i++] = (float)Zernike.Zcart(m, n, xx * Norm, yy * Norm);
                });

                ZernikeEvenCache.Add(new ZernikeCacheEntry(angPix, size, icoeff, NewData));
                return NewData;
            }
        }

        public Image GetPhaseCorrection(float angPix, int size)
        {
            float[][] ZernikePhases = new float[ZernikeCoeffsOdd.Length][];
            for (int i = 0; i < ZernikePhases.Length; i++)
                ZernikePhases[i] = GetZernikeOddPhases(angPix, size, i);

            Image Result = new Image(new int3(size, size, 1), true, true);
            float[] ResultData = Result.GetHost(Intent.Write)[0];

            Parallel.For(0, ResultData.Length / 2, i =>
            {
                float PhaseSum = 0;
                for (int icoeff = 0; icoeff < ZernikePhases.Length; icoeff++)
                    PhaseSum += ZernikePhases[icoeff][i] * _ZernikeCoeffsOdd[icoeff];

                ResultData[i * 2] = (float)Math.Cos(PhaseSum);
                ResultData[i * 2 + 1] = (float)Math.Sin(PhaseSum);
            });

            return Result;
        }

        public Image GetGammaCorrection(float angPix, int size, Image reuseImage = null)
        {
            float[][] ZernikePhases = new float[ZernikeCoeffsEven.Length][];
            for (int i = 0; i < ZernikePhases.Length; i++)
                ZernikePhases[i] = GetZernikeEvenPhases(angPix, size, i + 1);   // Skip phase shift, defocus and astigmatism

            Image Result = reuseImage == null ? new Image(new int3(size, size, 1), true) : reuseImage;
            float[] ResultData = Result.GetHost(Intent.Write)[0];

            if (reuseImage != null)
                for (int i = 0; i < ResultData.Length; i++)
                    ResultData[i] = 0;

            Parallel.For(0, ZernikePhases.Length, icoeff =>
            {
                unsafe
                {
                    fixed (float* ResultDataP = ResultData)
                    fixed (float* ZernikePhasesP = ZernikePhases[icoeff])
                    {
                        float Coeff = _ZernikeCoeffsEven[icoeff];

                        if (Coeff != 0)
                            for (int i = 0; i < ResultData.Length; i++)
                                ResultDataP[i] += ZernikePhasesP[i] * Coeff;
                    }
                }
            });

            return Result;
        }

        public int GetAliasingFreeSize(float maxResolution, float particleDiameterPixel)
        {
            double Freq0 = 1.0 / maxResolution;
            float[] P = Helper.ArrayOfFunction(i => (float)(GetPhaseAtFrequency((i + 1) / 1000.0 * Freq0) / Math.PI), 1000);
            float[] dP = MathHelper.Diff(P).Select(v => Math.Abs(v)).ToArray();
            float dPMax = MathHelper.Max(dP);

            return (int)Math.Ceiling(dPMax / 0.5f * 1000 * 1.1 + particleDiameterPixel);
        }

        public Image GetMotionSinc(int size, int originalSize, float2 motion)
        {
            float[] Data = new float[new int2(size).ElementsFFT()];
            motion /= (float)PixelSize;

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float2 Pos = new float2(xx, yy) / new float2(originalSize) * 2;
                float Projected = float2.Dot(Pos, motion);
                float Sinc = MathHelper.Sinc(Projected);

                Data[y * (size / 2 + 1) + x] = Sinc;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public Image GetMotionSinc2(int size, int originalSize, float2 motion)
        {
            float[] Data = new float[new int2(size).ElementsFFT()];
            motion /= (float)PixelSize;

            Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
            {
                float2 Pos = new float2(xx, yy) / new float2(originalSize) * 2;
                float Projected = float2.Dot(Pos, motion);
                float Sinc = MathHelper.Sinc(Projected);

                Data[y * (size / 2 + 1) + x] = Sinc * Sinc;
            });

            return new Image(Data, new int3(size, size, 1), true);
        }

        public static Image GetCTFCoords(int size, int originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = ArrayPool<float2>.Rent((size / 2 + 1) * size);
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size / 2 + 1 ? y : y - size;

                        float xs = xx / (float)originalSize;
                        float ys = yy / (float)originalSize;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size / 2 + 1) + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size, size, 1), true);
                ArrayPool<float2>.Return(CTFCoordsData);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoords(int size, int originalSize, Matrix2 distortion, Image buffer = null)
        {
            Image CTFCoords = buffer != null ? buffer : new Image(new int3(size, size, 1), true, true);
            float[] CTFCoordsData = CTFCoords.GetHost(Intent.Write)[0];

            Parallel.For(0, size, y =>
            {
                for (int x = 0; x < size / 2 + 1; x++)
                {
                    float xx = x;
                    float yy = y < size / 2 + 1 ? y : y - size;

                    float2 Dist = distortion * new float2(xx, yy);

                    float xs = Dist.X / (float)originalSize;
                    float ys = Dist.Y / (float)originalSize;
                    float r = (float)Math.Sqrt(xs * xs + ys * ys);
                    float angle = (float)Math.Atan2(yy, xx);

                    CTFCoordsData[(y * (size / 2 + 1) + x) * 2] = r;
                    CTFCoordsData[(y * (size / 2 + 1) + x) * 2 + 1] = angle;
                }
            });

            return CTFCoords;
        }

        public static Image GetCTFPCoords(int size, int originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size / 2 + 1) * size];
                for (int y = 0; y < size; y++)
                    for (int x = 0; x < size / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size / 2 + 1 ? y : y - size;

                        float xs = xx / (float)originalSize;
                        float ys = yy / (float)originalSize;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = r > 0 ? (float)Math.Acos(yy / Math.Sqrt(xx * xx + yy * yy)) : 0;

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size / 2 + 1) + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size, size, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFPCoords(int size, int originalSize, Matrix2 distortion, Image buffer = null)
        {
            Image CTFCoords = buffer != null ? buffer : new Image(new int3(size, size, 1), true, true);
            float[] CTFCoordsData = CTFCoords.GetHost(Intent.Write)[0];

            Parallel.For(0, size, y =>
            {
                for (int x = 0; x < size / 2 + 1; x++)
                {
                    int xx = x;
                    int yy = y < size / 2 + 1 ? y : y - size;

                    float2 Dist = distortion * new float2(xx, yy);

                    float xs = Dist.X / (float)originalSize;
                    float ys = Dist.Y / (float)originalSize;
                    float r = (float)Math.Sqrt(xs * xs + ys * ys);
                    float angle = r > 0 ? (float)Math.Acos(yy / Math.Sqrt(xx * xx + yy * yy)) : 0;

                    CTFCoordsData[(y * (size / 2 + 1) + x) * 2] = r;
                    CTFCoordsData[(y * (size / 2 + 1) + x) * 2 + 1] = angle;
                }
            });

            return CTFCoords;
        }

        public static Image GetCTFCoords(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size.X / 2 + 1) * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size.X / 2 + 1) + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsParallel(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size.X / 2 + 1) * size.Y];
                Helper.ForCPU(0, size.Y, 4, null, (y, threadID) =>
                {
                    for (int x = 0; x < size.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size.X / 2 + 1) + x] = new float2(r, angle);
                    }
                }, null);

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsFull(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[size.X * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X; x++)
                    {
                        int xx = x < size.X / 2 + 1 ? x : x - size.X;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * size.X + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1));
            }

            return CTFCoords;
        }

        public static Image GetCTFCoordsFullCentered(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[size.X * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X; x++)
                    {
                        int xx = x - size.X / 2;
                        int yy = y - size.Y / 2;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * size.X + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1));
            }

            return CTFCoords;
        }

        public static void ApplyPandQ(Image particlesFT,
                                      CTF[] ctfs,
                                      Image intermediateFTCorr,
                                      Image intermediateParticlesSuper,
                                      Image intermediateParticles,
                                      Image intermediateCTFP,
                                      Image intermediateMaskAngles,
                                      float2[] coordsCTF,
                                      float2[] coordsCTFP,
                                      Image mask,
                                      int planForw,
                                      int planBack,
                                      int nSectors,
                                      bool isReverse,
                                      Image outP,
                                      Image outQ)
        {
            decimal AngleStep = 180M / nSectors;
            int SizeFull = outP.Dims.X;
            int SizeFullSuper = particlesFT.Dims.X;
            int BatchSize = ctfs.Length;

            outP.Fill(0);
            outQ.Fill(0);

            int iangle = 0;
            for (decimal angle = 0; angle < 180; angle += AngleStep, iangle++)
            {
                for (int ipass = 0; ipass < 2; ipass++)
                {
                    bool isPositive = (ipass == 1) ? isReverse : !isReverse;

                    float[][] CTFPData = intermediateCTFP.GetHost(Intent.Write);
                    Parallel.For(0, BatchSize, i =>
                    {
                        ctfs[i].Get2DP(coordsCTF, coordsCTFP, isPositive, (float)angle * Helper.ToRad, CTFPData[i]);
                    });

                    GPU.MultiplyComplexSlicesByComplex(particlesFT.GetDevice(Intent.Read),
                                                       intermediateCTFP.GetDevice(Intent.Read),
                                                       intermediateFTCorr.GetDevice(Intent.Write),
                                                       particlesFT.ElementsComplex,
                                                       1);

                    if (false)
                    {
                        Image CTFPReal = intermediateCTFP.AsReal();
                        CTFPReal.WriteMRC($"rec_{angle}_{ipass}_ctfpreal.mrc", true);
                        CTFPReal.Dispose();
                        Image CTFPImag = intermediateCTFP.AsImaginary();
                        CTFPImag.WriteMRC($"rec_{angle}_{ipass}_ctfpimag.mrc", true);
                        CTFPImag.Dispose();
                    }

                    #region Mask and crop in real space

                    GPU.IFFT(intermediateFTCorr.GetDevice(Intent.Read),
                             intermediateParticlesSuper.GetDevice(Intent.Write),
                             new int3(SizeFullSuper, SizeFullSuper, 1),
                             (uint)BatchSize,
                             planBack,
                             false);

                    if (mask != null)
                        intermediateParticlesSuper.MultiplySlices(mask);

                    if (false)
                    {
                        intermediateParticlesSuper.RemapFromFT();
                        intermediateParticlesSuper.WriteMRC($"rec_{angle}_{ipass}.mrc", true);
                        intermediateParticlesSuper.RemapToFT();
                    }

                    GPU.CropFTFull(intermediateParticlesSuper.GetDevice(Intent.Read),
                                   intermediateParticles.GetDevice(Intent.Write),
                                   new int3(SizeFullSuper, SizeFullSuper, 1),
                                   new int3(SizeFull, SizeFull, 1),
                                   (uint)BatchSize);

                    GPU.FFT(intermediateParticles.GetDevice(Intent.Read),
                            intermediateFTCorr.GetDevice(Intent.Write),
                            new int3(SizeFull, SizeFull, 1),
                            (uint)BatchSize,
                            planForw);
                    intermediateFTCorr.Multiply(1f / (SizeFull * SizeFull));

                    #endregion

                    float AngleMin = (float)(angle + 90 - (0.5M * AngleStep));
                    float AngleMax = (float)(angle + 90 + (0.5M * AngleStep));

                    // angles larger than 180
                    bool AngleReverse = false;
                    if (AngleMin >= 180)
                    {
                        AngleMin -= 180;
                        AngleMax -= 180;
                        AngleReverse = true;
                    }

                    Image myCTFPorQa, myCTFPorQb;
                    if (AngleReverse)
                    {
                        myCTFPorQa = (ipass == 0) ? outQ : outP;
                        myCTFPorQb = (ipass == 0) ? outP : outQ;
                    }
                    else
                    {
                        myCTFPorQa = (ipass == 0) ? outP : outQ;
                        myCTFPorQb = (ipass == 0) ? outQ : outP;
                    }

                    // Deal with sectors with the Y-axis in the middle of the sector...
                    bool DoWrapMax = false;
                    if (AngleMin < 180 && AngleMax > 180)
                    {
                        AngleMax -= 180;
                        DoWrapMax = true;
                    }

                    AngleMin *= Helper.ToRad;
                    AngleMax *= Helper.ToRad;

                    float[][] MaskData = intermediateMaskAngles.GetHost(Intent.Write);
                    int imask = 0;
                    Helper.ForEachElementFT(new int2(SizeFull), (x, y, xx, yy) =>
                    {
                        float r2 = xx * xx + yy * yy;
                        float Angle = r2 > 0 ? (float)(Math.Acos(yy / Math.Sqrt(r2))) : 0;

                        MaskData[0][imask] = 0;
                        MaskData[1][imask] = 0;

                        if (DoWrapMax)
                        {
                            if (Angle >= AngleMin)
                                MaskData[0][imask] = 1;
                            else if (Angle < AngleMax)
                                MaskData[1][imask] = 1;
                        }
                        else
                        {
                            if (Angle >= AngleMin && Angle < AngleMax)
                                MaskData[0][imask] = 1;
                        }

                        imask++;
                    });

                    if (false)
                        intermediateMaskAngles.WriteMRC($"rec_{angle}_{ipass}_maskpq.mrc");

                    GPU.MultiplyComplexSlicesByScalar(intermediateFTCorr.GetDevice(Intent.Read),
                                                      intermediateMaskAngles.GetDeviceSlice(0, Intent.Read),
                                                      intermediateCTFP.GetDevice(Intent.Write),
                                                      new int2(SizeFull).ElementsFFT(),
                                                      (uint)BatchSize);
                    GPU.AddToSlices(myCTFPorQa.GetDevice(Intent.Read),
                                    intermediateCTFP.GetDevice(Intent.Read),
                                    myCTFPorQa.GetDevice(Intent.Write),
                                    new int2(SizeFull).ElementsFFT() * 2 * BatchSize,
                                    1);

                    GPU.MultiplyComplexSlicesByScalar(intermediateFTCorr.GetDevice(Intent.Read),
                                                      intermediateMaskAngles.GetDeviceSlice(1, Intent.Read),
                                                      intermediateCTFP.GetDevice(Intent.Write),
                                                      new int2(SizeFull).ElementsFFT(),
                                                      (uint)BatchSize);
                    GPU.AddToSlices(myCTFPorQb.GetDevice(Intent.Read),
                                    intermediateCTFP.GetDevice(Intent.Read),
                                    myCTFPorQb.GetDevice(Intent.Write),
                                    new int2(SizeFull).ElementsFFT() * 2 * BatchSize,
                                    1);
                }
            }
        }

        public static void ApplyPandQPrecomp(Image particlesFT,
                                              CTF[] ctfs,
                                              Image intermediateFTCorr,
                                              Image intermediateParticlesSuper,
                                              Image intermediateParticles,
                                              Image intermediateCTFP,
                                              Image coordsCTF,
                                              Image gammaCorrection,
                                              bool reverse,
                                              Image mask,
                                              int planForw,
                                              int planBack,
                                              Image outP,
                                              Image outQ)
        {
            int SizeFull = outP.Dims.X;
            int SizeFullSuper = particlesFT.Dims.X;
            int BatchSize = ctfs.Length;

            //float[][] CTFPData = intermediateCTFP.GetHost(Intent.Write);
            //Parallel.For(0, BatchSize, i =>
            //{
            //    ctfs[i].Get2DPQFromPrecomp(coordsCTF, pqSigns, CTFPData[i]);
            //});

            //if (true)
            //{
            //    Image CTFPReal = intermediateCTFP.AsReal();
            //    CTFPReal.WriteMRC($"prec_ctfpreal.mrc", true);
            //    CTFPReal.Dispose();
            //    Image CTFPImag = intermediateCTFP.AsImaginary();
            //    CTFPImag.WriteMRC($"prec_ctfpimag.mrc", true);
            //    CTFPImag.Dispose();
            //}

            GPU.CreateCTFComplex(intermediateCTFP.GetDevice(Intent.Write),
                                coordsCTF.GetDevice(Intent.Read),
                                gammaCorrection == null ? IntPtr.Zero : gammaCorrection.GetDevice(Intent.Read),
                                (uint)coordsCTF.ElementsComplex,
                                ctfs.Select(v => v.ToStruct()).ToArray(),
                                reverse,
                                (uint)BatchSize);

            if (false)
            {
                Image CTFPReal = intermediateCTFP.AsReal();
                CTFPReal.WriteMRC($"prec_ctfpreal_gpu.mrc", true);
                CTFPReal.Dispose();
                Image CTFPImag = intermediateCTFP.AsImaginary();
                CTFPImag.WriteMRC($"prec_ctfpimag_gpu.mrc", true);
                CTFPImag.Dispose();
            }

            #region Make P

            GPU.MultiplyComplexSlicesByComplex(particlesFT.GetDevice(Intent.Read),
                                               intermediateCTFP.GetDevice(Intent.Read),
                                               intermediateFTCorr.GetDevice(Intent.Write),
                                               particlesFT.ElementsComplex,
                                               1);

            GPU.IFFT(intermediateFTCorr.GetDevice(Intent.Read),
                     intermediateParticlesSuper.GetDevice(Intent.Write),
                     new int3(SizeFullSuper, SizeFullSuper, 1),
                     (uint)BatchSize,
                     planBack,
                     false);

            if (mask != null)
                intermediateParticlesSuper.MultiplySlices(mask);

            if (false)
            {
                intermediateParticlesSuper.RemapFromFT();
                intermediateParticlesSuper.WriteMRC($"prec.mrc", true);
                intermediateParticlesSuper.RemapToFT();
            }

            GPU.CropFTFull(intermediateParticlesSuper.GetDevice(Intent.Read),
                           intermediateParticles.GetDevice(Intent.Write),
                           new int3(SizeFullSuper, SizeFullSuper, 1),
                           new int3(SizeFull, SizeFull, 1),
                           (uint)BatchSize);

            GPU.FFT(intermediateParticles.GetDevice(Intent.Read),
                    outP.GetDevice(Intent.Write),
                    new int3(SizeFull, SizeFull, 1),
                    (uint)BatchSize,
                    planForw);
            outP.Multiply(1f / (SizeFull * SizeFull));

            #endregion      

            #region Make Q

            GPU.MultiplyComplexSlicesByComplexConj(particlesFT.GetDevice(Intent.Read),
                                                   intermediateCTFP.GetDevice(Intent.Read),
                                                   intermediateFTCorr.GetDevice(Intent.Write),
                                                   particlesFT.ElementsComplex,
                                                   1);

            GPU.IFFT(intermediateFTCorr.GetDevice(Intent.Read),
                     intermediateParticlesSuper.GetDevice(Intent.Write),
                     new int3(SizeFullSuper, SizeFullSuper, 1),
                     (uint)BatchSize,
                     planBack,
                     false);

            if (mask != null)
                intermediateParticlesSuper.MultiplySlices(mask);

            if (false)
            {
                intermediateParticlesSuper.RemapFromFT();
                intermediateParticlesSuper.WriteMRC($"qrec.mrc", true);
                intermediateParticlesSuper.RemapToFT();
            }

            GPU.CropFTFull(intermediateParticlesSuper.GetDevice(Intent.Read),
                           intermediateParticles.GetDevice(Intent.Write),
                           new int3(SizeFullSuper, SizeFullSuper, 1),
                           new int3(SizeFull, SizeFull, 1),
                           (uint)BatchSize);

            GPU.FFT(intermediateParticles.GetDevice(Intent.Read),
                    outQ.GetDevice(Intent.Write),
                    new int3(SizeFull, SizeFull, 1),
                    (uint)BatchSize,
                    planForw);
            outQ.Multiply(1f / (SizeFull * SizeFull));

            #endregion        
        }

        public static void GetPQSign(float2[] coordinates, float2[] coordinatesP, bool isPositive, float pAngle, float[] result)
        {
            if (pAngle >= (float)Math.PI)
            {
                pAngle -= (float)Math.PI;
                isPositive = !isPositive;
            }

            for (int i = 0; i < coordinates.Length; i++)
            {
                float angleP = coordinatesP[i].Y;

                float reverse = isPositive ? 1 : -1;
                if (angleP <= pAngle)
                    reverse *= -1;
                if (pAngle == 0 && Math.Abs(angleP - Math.PI) < 1e-5)
                    reverse *= -1;

                result[i] = reverse;
            }
        }

        public static void PrecomputePQSigns(int size,
                                             int nSectors,
                                             bool isReverse,
                                             float2[] coordsCTF,
                                             float2[] coordsCTFP,
                                             float[] result)
        {
            decimal AngleStep = 180M / nSectors;
            float[] Dummy = new float[result.Length];

            for (decimal angle = 0; angle < 180; angle += AngleStep)
            {
                for (int ipass = 0; ipass < 2; ipass++)
                {
                    float AngleMin = (float)(angle + 90 - (0.5M * AngleStep));
                    float AngleMax = (float)(angle + 90 + (0.5M * AngleStep));

                    // angles larger than 180
                    bool AngleReverse = false;
                    if (AngleMin >= 180)
                    {
                        AngleMin -= 180;
                        AngleMax -= 180;
                        AngleReverse = true;
                    }

                    float[] resultP, resultQ;
                    if (AngleReverse)
                    {
                        resultP = (ipass == 0) ? Dummy : result;
                        resultQ = (ipass == 0) ? result : Dummy;
                    }
                    else
                    {
                        resultP = (ipass == 0) ? result : Dummy;
                        resultQ = (ipass == 0) ? Dummy : result;
                    }

                    // Deal with sectors with the Y-axis in the middle of the sector...
                    bool DoWrapMax = false;
                    if (AngleMin < 180 && AngleMax > 180)
                    {
                        AngleMax -= 180;
                        DoWrapMax = true;
                    }

                    AngleMin *= Helper.ToRad;
                    AngleMax *= Helper.ToRad;

                    bool isPositive = (ipass == 1) ? isReverse : !isReverse;

                    float[] CurrentSigns = new float[coordsCTF.Length];
                    GetPQSign(coordsCTF, coordsCTFP, isPositive, (float)angle * Helper.ToRad, CurrentSigns);

                    int imask = 0;
                    Helper.ForEachElementFT(new int2(size), (x, y, xx, yy) =>
                    {
                        float r2 = xx * xx + yy * yy;
                        float Angle = r2 > 0 ? (float)(Math.Acos(yy / Math.Sqrt(r2))) : 0;

                        if (DoWrapMax)
                        {
                            if (Angle >= AngleMin)
                                resultP[imask] = CurrentSigns[imask];
                            else if (Angle < AngleMax)
                                resultQ[imask] = CurrentSigns[imask];
                        }
                        else
                        {
                            if (Angle >= AngleMin && Angle < AngleMax)
                                resultP[imask] = CurrentSigns[imask];
                        }

                        imask++;
                    });
                }
            }
        }

        public CTF GetCopy()
        {
            return new CTF
            {
                _Amplitude = Amplitude,
                _BeamTilt = BeamTilt,
                _BeamTilt2 = BeamTilt2,
                _Bfactor = Bfactor,
                _Cs = Cs,
                _Defocus = Defocus,
                _DefocusAngle = DefocusAngle,
                _DefocusDelta = DefocusDelta,
                _PhaseShift = PhaseShift,
                _PixelSize = PixelSize,
                _PixelSizeAngle = PixelSizeAngle,
                _PixelSizeDeltaPercent = PixelSizeDeltaPercent,
                _Distortion = Distortion.GetCopy(),
                _Scale = Scale,
                _Voltage = Voltage
            };
        }

        public float[] EstimateQuality(float[] experimental, float[] experimentalScale, float minFreq, int minSamples)
        {
            try
            {
                int N = experimental.Length;
                int MinN = (int)(minFreq * N);

                float[] Quality = Helper.ArrayOfConstant(float.NaN, N);
                float[] Zeros = GetZeros().Select(v => v * 2 * N).ToArray();
                if (Zeros.Length < 2)
                    return Quality;

                // Calculate indices and widths of all peaks
                int[] Peaks = new int[Zeros.Length - 1];
                int[] PeakWidths = new int[Peaks.Length];
                for (int i = 0; i < Zeros.Length - 1; i++)
                {
                    Peaks[i] = (int)((Zeros[i] + Zeros[i + 1]) * 0.5f);
                    PeakWidths[i] = (int)(Zeros[i + 1] - Zeros[i]);
                }

                // For each index, find the closest peak
                int[] ClosestPeak = new int[N];
                int[] WindowLength = new int[N];
                for (int i = 0, peak = 0; i < N; i++)
                {
                    int ClosestDist = Math.Abs(i - Peaks[peak]);
                    if (Math.Abs(i - Peaks[peak + 1]) < ClosestDist)
                        peak = Math.Min(Peaks.Length - 2, peak + 1);

                    ClosestPeak[i] = peak;
                    WindowLength[i] = Math.Max(minSamples, PeakWidths[peak]);
                }

                // Calculate simulated CTF and multiply it by the scale curve if needed
                float[] Simulated = Get1D(N, true);
                if (experimentalScale != null)
                {
                    if (experimental.Length != experimentalScale.Length)
                        throw new Exception("Experimental values and scale arrays should have equal length.");
                    for (int i = 0; i < Simulated.Length; i++)
                        Simulated[i] *= experimentalScale[i];
                }

                for (int i = MinN; i < N - minSamples / 2; i++)
                {
                    int WindowStart = Math.Max(0, i - WindowLength[i] / 2);
                    int WindowEnd = Math.Min(N, i + WindowLength[i] / 2);
                    float[] WindowExperimental = Helper.Subset(experimental, WindowStart, WindowEnd);
                    float[] WindowSimulated = Helper.Subset(Simulated, WindowStart, WindowEnd);

                    Quality[i] = MathHelper.Mult(MathHelper.Normalize(WindowExperimental), MathHelper.Normalize(WindowSimulated)).Sum() / WindowExperimental.Length;
                }

                return Quality;
            }
            catch
            {
                return Helper.ArrayOfConstant(1f, experimental.Length);
            }
        }
    }

    /// <summary>
    /// Everything is in SI units
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CTFStruct
    {
        public float PixelSize;
        public float PixelSizeDelta;
        public float PixelSizeAngle;
        public float Cs;
        public float Voltage;
        public float Defocus;
        public float AstigmatismAngle;
        public float DefocusDelta;
        public float Amplitude;
        public float Bfactor;
        public float BfactorDelta;
        public float BfactorAngle;
        public float Scale;
        public float PhaseShift;
    }

    /// <summary>
    /// Everything is in SI units
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CTFFitStruct
    {
        public float3 Pixelsize;
        public float3 Pixeldelta;
        public float3 Pixelangle;
        public float3 Cs;
        public float3 Voltage;
        public float3 Defocus;
        public float3 Astigmatismangle;
        public float3 Defocusdelta;
        public float3 Amplitude;
        public float3 Bfactor;
        public float3 Scale;
        public float3 Phaseshift;

        public int2 DimsPeriodogram;
        public int MaskInnerRadius;
        public int MaskOuterRadius;
    }

    class ZernikeCacheEntry
    {
        public float AngPix;
        public int Size;
        public int iCoeff;
        public float[] Data;

        public ZernikeCacheEntry(float angpix, int size, int icoeff, float[] data)
        {
            AngPix = angpix;
            Size = size;
            iCoeff = icoeff;
            Data = data;
        }
    }
}