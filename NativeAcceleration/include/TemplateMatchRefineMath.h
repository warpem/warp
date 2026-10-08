#ifndef WARP_TEMPLATE_MATCH_REFINE_MATH_H
#define WARP_TEMPLATE_MATCH_REFINE_MATH_H

// Shared by the CUDA scorer and a CUDA-free derivative test. Fourier coordinates
// are in the wrapped RFFT layout used by Warp's Projector, not an fftshifted grid.
#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define TM_HD __host__ __device__
#else
#define TM_HD
#endif

namespace warp_template_match
{
template <class T> struct Complex
{
    T re, im;
    TM_HD Complex(T real = T(0), T imaginary = T(0)) : re(real), im(imaginary) {}
    TM_HD Complex operator+(const Complex& b) const { return Complex(re + b.re, im + b.im); }
    TM_HD Complex operator-(const Complex& b) const { return Complex(re - b.re, im - b.im); }
    TM_HD Complex operator*(T b) const { return Complex(re * b, im * b); }
    TM_HD Complex operator*(const Complex& b) const
    { return Complex(re * b.re - im * b.im, re * b.im + im * b.re); }
};

template <class T> TM_HD T Inner(const Complex<T>& a, const Complex<T>& b)
{ return a.re * b.re + a.im * b.im; }

template <class T> struct Sample
{
    Complex<T> value;
    Complex<T> gradient[3];
};

TM_HD inline int Wrap(int index, int size)
{
    index %= size;
    return index < 0 ? index + size : index;
}

// Deliberately omit DC, the x=0 conjugate half-line, and all Nyquist samples.
// Every selected sample is one independent proper-complex observation.
TM_HD inline bool IndependentSample(int x, int y, int box, double cutoffRadius)
{
    if (x == 0 && y <= 0) return false;
    if (x >= box / 2 || y <= -box / 2 || y >= box / 2) return false;
    return double(x) * x + double(y) * y <= cutoffRadius * cutoffRadius;
}

template <class T, class Fetch>
TM_HD Sample<T> Interpolate(Fetch fetch, int dim, T x, T y, T z)
{
    Sample<T> result;
    const T reflection = x < T(0) ? T(-1) : T(1);
    x *= reflection;
    y *= reflection;
    z *= reflection;
    // The scorer uses an inscribed disk and an oversampled projection grid, so
    // valid projections have a complete interpolation halo. Outside that grid
    // is zero signal; never clamp an out-of-range model sample to an edge value.
    if (!(x >= T(0) && x < T(dim / 2))) return result;
    const int ix = int(::floor(x)), iy = int(::floor(y)), iz = int(::floor(z));
    const T f[3] = {x - T(ix), y - T(iy), z - T(iz)};
    // Wrap each axis once, then share the interpolation intermediates between
    // the value and all three derivatives. This is the same trilinear model.
    const int y0 = Wrap(iy, dim), z0 = Wrap(iz, dim);
    const int y1 = y0 + 1 == dim ? 0 : y0 + 1, z1 = z0 + 1 == dim ? 0 : z0 + 1;
    const Complex<T> v000 = fetch(ix, y0, z0), v100 = fetch(ix+1, y0, z0);
    const Complex<T> v010 = fetch(ix, y1, z0), v110 = fetch(ix+1, y1, z0);
    const Complex<T> v001 = fetch(ix, y0, z1), v101 = fetch(ix+1, y0, z1);
    const Complex<T> v011 = fetch(ix, y1, z1), v111 = fetch(ix+1, y1, z1);
    const Complex<T> dx00 = v100-v000, dx10 = v110-v010, dx01 = v101-v001, dx11 = v111-v011;
    const Complex<T> x00 = v000 + dx00*f[0], x10 = v010 + dx10*f[0];
    const Complex<T> x01 = v001 + dx01*f[0], x11 = v011 + dx11*f[0];
    const Complex<T> dy0 = x10-x00, dy1 = x11-x01;
    const Complex<T> xy0 = x00+dy0*f[1], xy1 = x01+dy1*f[1];
    const Complex<T> dx0 = dx00+(dx10-dx00)*f[1], dx1 = dx01+(dx11-dx01)*f[1];
    result.gradient[0] = dx0+(dx1-dx0)*f[2];
    result.gradient[1] = dy0+(dy1-dy0)*f[2];
    result.gradient[2] = xy1-xy0;
    result.value = xy0+result.gradient[2]*f[2];
    // F(q)=conj(F(-q)) on the reflected half-plane. Both the conjugation and
    // the derivative of -q are essential for correct orientation gradients.
    result.value.im *= reflection;
    for (int axis = 0; axis < 3; ++axis)
    {
        result.gradient[axis].re *= reflection;
        // Imaginary gradient has two reflection factors, which cancel.
    }
    return result;
}

// One Fourier sample's model and its six derivatives. Matrices are column-major
// and already include projector oversampling; translations are in image pixels.
template <class T, class Fetch>
TM_HD void ModelAndDerivatives(Fetch fetch, int dim, int box, int x, int y,
                              const T* matrix, const T* matrixDerivatives,
                              const T* shift, const T* shiftDerivatives,
                              T beta, const T* betaDerivatives,
                              T ctfBase, T ctfQuadrature,
                              Complex<T>& model, Complex<T>* derivatives,
                              T phaseRadiusSquared = T(-1))
{
    const T px = matrix[0] * x + matrix[3] * y;
    const T py = matrix[1] * x + matrix[4] * y;
    const T pz = matrix[2] * x + matrix[5] * y;
    const Sample<T> sample = Interpolate<T>(fetch, dim, px, py, pz);
    // Production supplies the exact squared physical frequency used to form
    // the anchor CTF (including anisotropic pixels/magnification). The default
    // integer-grid value is useful for isolated interpolation tests.
    const T frequencySquared = phaseRadiusSquared >= T(0) ? phaseRadiusSquared : T(x) * x + T(y) * y;
    const T angle = beta * frequencySquared;
    const T ca = T(::cos(angle)), sa = T(::sin(angle));
    const T transfer = ctfBase * ca + ctfQuadrature * sa;
    const T transferPhaseDerivative = (-ctfBase * sa + ctfQuadrature * ca) * frequencySquared;
    const T phaseScale = T(-6.283185307179586476925286766559) / box;
    const T phase = phaseScale * (x * shift[0] + y * shift[1]);
    const Complex<T> phaseFactor(T(::cos(phase)), T(::sin(phase)));
    const Complex<T> phased = sample.value * phaseFactor;
    model = phased * transfer;
    for (int parameter = 0; parameter < 6; ++parameter)
    {
        const T* dm = matrixDerivatives + parameter * 9;
        const T* ds = shiftDerivatives + parameter * 2;
        Complex<T> dp = sample.gradient[0] * (dm[0] * x + dm[3] * y)
                      + sample.gradient[1] * (dm[1] * x + dm[4] * y)
                      + sample.gradient[2] * (dm[2] * x + dm[5] * y);
        const T dphase = phaseScale * (x * ds[0] + y * ds[1]);
        dp = dp * phaseFactor + Complex<T>(-phased.im, phased.re) * dphase;
        derivatives[parameter] = dp * transfer + phased * (transferPhaseDerivative * betaDerivatives[parameter]);
    }
}
}

#undef TM_HD
#endif
