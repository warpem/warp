using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ZLinq;

namespace Warp.Tools;

/// <summary>Windowed GPU periodograms with independent spatial aperture and Fourier sampling.</summary>
public static class CtfPowerSpectrum
{
    public sealed record Observation(CtfSpectrumFit Spectrum, float3 Position, int Group);
    public sealed record Extraction(List<Observation> Observations, float[] Display, int2 PositionGrid, int FourierSize);

    public static Extraction Extract(Image image, ProcessingOptionsMovieCTF options, int groups = 1, int groupOffset = 0, CtfSpectrumFit basisSource = null)
    {
        using var extractor = new Extractor(new int2(image.Dims), options);
        return extractor.Extract(image, groups, groupOffset, basisSource);
    }

    /// <summary>Reuse one extractor across a tilt series. Window, Fourier-bin geometry,
    /// FFT plan and device scratch buffers are allocated once. Only compact spectra return to the host.</summary>
    public sealed class Extractor : IDisposable
    {
        IntPtr context;
        readonly ProcessingOptionsMovieCTF options;
        readonly int2 dimensions, grid;
        readonly int window, fftSize, bins;
        readonly int3[] origins;
        readonly double[] count, q2, q4, ax, ay;
        CtfSpectrumFit sharedBasis;

        public Extractor(int2 dimensions, ProcessingOptionsMovieCTF options, bool fullSpectrum = false)
        {
            this.options = options; this.dimensions = dimensions; window = options.Window;
            double pixel = (double)options.BinnedPixelSizeMean;
            if (window < 64 || window > Math.Min(dimensions.X, dimensions.Y) || window % 2 != 0)
                throw new ArgumentException("CTF window must be even, at least 64 pixels, and fit inside the image.");
            if (!(pixel > 0) || !(options.RangeMin > 0 && options.RangeMax > options.RangeMin && options.RangeMax <= 1))
                throw new ArgumentException("Invalid CTF pixel size or frequency range.");
            if (!(options.ZMin >= 0 && options.ZMax > options.ZMin)) throw new ArgumentException("The CTF defocus search range must be nonnegative and nonempty.");
            double qmax = (fullSpectrum ? 1 : (double)options.RangeMax) / (2 * pixel);
            fftSize = 1;
            double minimum = Math.Max(window * 2, 6 * CtfSpectrumFit.Wavelength((double)options.Voltage) * (double)options.ZMax * 1e4 * qmax / pixel);
            while (fftSize < minimum) fftSize *= 2;
            if (fftSize > 8192) throw new ArgumentException("CTF fitting requires an impractically large Fourier grid. Reduce the fitting frequency or maximum defocus.");
            const int sectors = 24;
            int radialBins = fftSize / 2, binCount = radialBins * sectors;
            var pixels = Enumerable.Range(0, binCount).Select(_ => new List<int>()).ToArray();
            double[] c = new double[binCount], u = new double[binCount], v = new double[binCount], r2 = new double[binCount], r4 = new double[binCount];
            for (int y = 0; y < fftSize; y++) for (int x = 0; x <= fftSize / 2; x++)
            {
                int yy = y <= fftSize / 2 ? y : y - fftSize;
                if (x == 0 && yy < 0) continue;
                double r = Math.Sqrt(x*x+yy*yy), q = r/(fftSize*pixel);
                if (r == 0 || (!fullSpectrum && q < (double)options.RangeMin/(2*pixel)) || q >= qmax || r >= radialBins) continue;
                double angle = Math.Atan2(yy,x); if (angle < 0) angle += Math.PI;
                int b = Math.Min(sectors-1,(int)(angle*sectors/Math.PI))*radialBins+(int)r;
                pixels[b].Add(y*(fftSize/2+1)+x);
                c[b]++; r2[b] += q*q; r4[b] += q*q*q*q;
                u[b] += (x*x-yy*yy)/Math.Pow(fftSize*pixel,2); v[b] += 2.0*x*yy/Math.Pow(fftSize*pixel,2);
            }
            int[] valid = Enumerable.Range(0,binCount).Where(b => c[b]>0).ToArray();
            bins = valid.Length;
            if (bins < 16) throw new ArgumentException("The CTF fitting band contains too few Fourier samples.");
            count = valid.Select(b => c[b]).ToArray(); q2 = valid.Select(b => r2[b]/c[b]).ToArray(); q4 = valid.Select(b => r4[b]/c[b]).ToArray();
            ax = valid.Select(b => u[b]/c[b]).ToArray(); ay = valid.Select(b => v[b]/c[b]).ToArray();
            int[] starts = new int[bins+1];
            for (int b = 0; b < bins; b++) starts[b+1] = starts[b]+pixels[valid[b]].Count;
            int[] indices = new int[starts[bins]];
            for (int b = 0; b < bins; b++) pixels[valid[b]].CopyTo(indices,starts[b]);
            origins = Helper.GetEqualGridSpacing(dimensions,new int2(window),.5f,out grid);
            var hann = new float[window];
            for (int x = 0; x < window; x++) hann[x] = (float)(.5-.5*Math.Cos(2*Math.PI*(x+.5)/window));
            var displayIndices = new int[window*window/2];
            for (int y = 0; y < window/2; y++) for (int x = 0; x < window; x++)
            {
                int xx = x-window/2, yy = window/2-1-y;
                if (xx < 0) {xx = -xx; yy = -yy;}
                int iy = (int)Math.Round((double)yy*fftSize/window); if (iy < 0) iy += fftSize;
                int ix = (int)Math.Round((double)xx*fftSize/window);
                displayIndices[y*window+x] = iy*(fftSize/2+1)+ix;
            }
            int batch = Math.Max(1,Math.Min(8,32*1024*1024/(fftSize*fftSize)));
            CtfNative.Check(CtfNative.PowerCreate(dimensions.X,dimensions.Y,window,fftSize,batch,origins.Length,bins,origins,hann,starts,indices,displayIndices,out context),"create spectrum extractor");
        }

        public Extraction Extract(Image image, int groups = 1, int groupOffset = 0, CtfSpectrumFit basisSource = null, int firstFrame = 0, int frameCount = -1)
        {
            ObjectDisposedException.ThrowIf(context == IntPtr.Zero,this);
            if (image.Dims.X != dimensions.X || image.Dims.Y != dimensions.Y) throw new ArgumentException("CTF extractor/image dimensions differ.");
            groups = Math.Clamp(groups,1,image.Dims.Z);
            sharedBasis ??= basisSource;
            float[][] allFrames = image.GetHost(Intent.Read);
            if (frameCount < 0) frameCount = allFrames.Length - firstFrame;
            if (firstFrame < 0 || frameCount < 1 || firstFrame + frameCount > allFrames.Length)
                throw new ArgumentOutOfRangeException(nameof(firstFrame));
            float[][] frames = allFrames.Skip(firstFrame).Take(frameCount).ToArray();
            groups = Math.Min(groups, frames.Length);
            var records = new List<Observation>(); var display = new float[window*window/2];
            var power = new double[checked(origins.Length*bins)];
            for (int group = 0; group < groups; group++)
            {
                int first = group*frames.Length/groups, end = (group+1)*frames.Length/groups;
                CtfNative.Check(CtfNative.PowerBegin(context,group == 0 ? 1 : 0),"begin spectrum group");
                for (int frame = first; frame < end; frame++) CtfNative.Check(CtfNative.PowerAdd(context,frames[frame]),"prepare spectra");
                CtfNative.Check(CtfNative.PowerRead(context,power,display),"read spectra (input pixels must be finite)");
                var observations = new Observation[origins.Length];
                Observation Make(int p)
                {
                    var samples = new CtfSpectrumFit.Sample[bins];
                    for (int b = 0; b < bins; b++) samples[b] = new(q2[b],q4[b],ax[b],ay[b],power[p*bins+b]/(count[b]*(end-first)),count[b]*(end-first)*Math.Pow((double)window/fftSize,2));
                    var spectrum = new CtfSpectrumFit(samples,(double)options.Voltage,(double)options.Cs,(double)options.Amplitude,sharedBasis);
                    var origin = origins[p];
                    return new(spectrum,new float3((origin.X+window*.5f)/dimensions.X,(origin.Y+window*.5f)/dimensions.Y,
                        frames.Length>1 ? (first+end-1)*.5f/(frames.Length-1) : .5f),group+groupOffset);
                }
                observations[0] = Make(0); sharedBasis ??= observations[0].Spectrum;
                Parallel.For(1,origins.Length,p => observations[p] = Make(p));
                records.AddRange(observations);
            }
            for (int i = 0; i < display.Length; i++) display[i] /= (float)((long)frames.Length*origins.Length);
            return new(records,display,grid,fftSize);
        }
        public void Dispose()
        {
            if (context != IntPtr.Zero) {CtfNative.PowerDestroy(context);context = IntPtr.Zero;}
            GC.SuppressFinalize(this);
        }
        ~Extractor() {Dispose();}
    }
}
