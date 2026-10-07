#include "Functions.h"
#include "TemplateMatchRefineBatch.h"
#include "TemplateMatchBfgs.h"
#include "TemplateMatchRefineMath.h"
#include <math_constants.h>
#include <climits>
#include <limits>

namespace
{
using warp_template_match::Complex;
using warp_template_match::Inner;
using namespace warp_template_match_batch;
constexpr int Threads = 128;

struct BatchContext
{
    cudaTextureObject_t textureRe, textureIm;
    int dim, box, views, particles, hypotheses, maxIterations;
    float pixel, cutoff, diameter;
    unsigned long long cutoffSquaredLimit;
    const float2* data;
    const float *ctf, *quadrature, *noise, *phaseRadii;
    float *geometry, *bounds, *symmetry, *poses;
    int *seedIds, *diagnostics, *order;
    double *summary, *tiltStats;
};

struct TextureFetch
{
    cudaTextureObject_t re, im;
    __device__ Complex<float> operator()(int x, int y, int z) const
    {
        return Complex<float>(tex3D<float>(re, x + 0.5f, y + 0.5f, z + 0.5f),
                              tex3D<float>(im, x + 0.5f, y + 0.5f, z + 0.5f));
    }
};

// A score-only instantiation omits the six Jacobians.
template <bool Derivatives>
__device__ void Model(const BatchContext& c, const float* geometry, const float* pose,
                     size_t sampleIndex, int x, int y, Complex<float>& model,
                     Complex<float>* derivatives)
{
    const float gx = geometry[0] * x + geometry[3] * y;
    const float gy = geometry[1] * x + geometry[4] * y;
    const float gz = geometry[2] * x + geometry[5] * y;
    const float* rotation = pose + 3;
    const float qx = rotation[0] * gx + rotation[1] * gy + rotation[2] * gz;
    const float qy = rotation[3] * gx + rotation[4] * gy + rotation[5] * gz;
    const float qz = rotation[6] * gx + rotation[7] * gy + rotation[8] * gz;
    const TextureFetch fetch = {c.textureRe, c.textureIm};
    const warp_template_match::Sample<float> sample =
        warp_template_match::Interpolate<float>(fetch, c.dim, qx, qy, qz);
    const float shiftX = geometry[9] * pose[0] + geometry[11] * pose[1] + geometry[13] * pose[2];
    const float shiftY = geometry[10] * pose[0] + geometry[12] * pose[1] + geometry[14] * pose[2];
    const float beta = geometry[15] * pose[0] + geometry[16] * pose[1] + geometry[17] * pose[2];
    const float radius2 = c.phaseRadii[sampleIndex];
    float ca, sa, cp, sp;
    sincosf(beta * radius2, &sa, &ca);
    const float phaseScale = float(-2.0 * Pi) / c.box;
    sincosf(phaseScale * (x * shiftX + y * shiftY), &sp, &cp);
    const Complex<float> phase(cp, sp);
    const Complex<float> phased = sample.value * phase;
    const float transfer = c.ctf[sampleIndex] * ca + c.quadrature[sampleIndex] * sa;
    model = phased * transfer;
    if (Derivatives)
    {
        const float depthDerivative = (-c.ctf[sampleIndex] * sa + c.quadrature[sampleIndex] * ca) * radius2;
        for (int parameter = 0; parameter < 3; ++parameter)
        {
            const float dphase = phaseScale * (x * geometry[9 + parameter * 2] + y * geometry[10 + parameter * 2]);
            derivatives[parameter] = Complex<float>(-model.im, model.re) * dphase +
                phased * (depthDerivative * geometry[15 + parameter]);
        }
        // R' = R exp([omega]x), q' = exp(-[omega]x) q.
        derivatives[3] = (sample.gradient[1] * qz - sample.gradient[2] * qy) * phase * transfer;
        derivatives[4] = (sample.gradient[2] * qx - sample.gradient[0] * qz) * phase * transfer;
        derivatives[5] = (sample.gradient[0] * qy - sample.gradient[1] * qx) * phase * transfer;
    }
}

#include "TemplateMatchBfgs.cuh"

// One fixed pose per particle, sharing the exact projection/CTF model used by
// refinement. Linear deposition in q² bounds discretization error when the
// host later evaluates exp(-B*q²/4). All native arithmetic remains FP32.
__global__ void EnvelopeSpectra(BatchContext c, int bins, float minimumQ2, float maximumQ2, bool byTilt, float* output)
{
    const int particle = blockIdx.y;
    if (c.seedIds[particle] < 0) return;
    const size_t frequencies = size_t(c.box) * (c.box / 2 + 1);
    const size_t samples = frequencies * c.views;
    for (size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x; id < samples; id += size_t(gridDim.x) * blockDim.x)
    {
        const int view = int(id / frequencies);
        float* cross = output + (byTilt ? size_t(particle) * c.views + view : size_t(particle)) * bins * 2;
        float* power = cross + bins;
        const size_t f = id % frequencies, index = size_t(particle) * samples + id;
        const int x = int(f % (c.box / 2 + 1));
        int y = int(f / (c.box / 2 + 1));
        if (y > c.box / 2) y -= c.box;
        if (!bfgs::IndependentSample(x, y, c.box, c.cutoffSquaredLimit)) continue;
        const float weight = c.noise[index];
        if (weight == 0) continue;
        const float q2 = c.phaseRadii[index];
        if (!(weight > 0) || !isfinite(weight) || !isfinite(q2) || q2 < 0 || q2 > maximumQ2)
        { atomicAdd(cross, CUDART_NAN_F); continue; }
        // Apply the physical-frequency boundary before binning, so even samples
        // sharing a histogram bin cannot leak across the requested high-pass.
        if (q2 <= minimumQ2) continue;
        Complex<float> model, unused[1];
        Model<false>(c, c.geometry + (size_t(particle) * c.views + view) * 18,
                     c.poses + particle * 12, index, x, y, model, unused);
        const float2 value = c.data[index];
        const float cc = weight * Inner(Complex<float>(value.x, value.y), model);
        const float pp = weight * Inner(model, model);
        const float coordinate = fminf(float(bins - 1), q2 / maximumQ2 * (bins - 1));
        const int bin = min(bins - 2, int(coordinate));
        const float fraction = coordinate - bin;
        atomicAdd(cross + bin, cc * (1 - fraction));
        atomicAdd(cross + bin + 1, cc * fraction);
        atomicAdd(power + bin, pp * (1 - fraction));
        atomicAdd(power + bin + 1, pp * fraction);
    }
}

__global__ void FinalTiltStatistics(BatchContext c)
{
    const size_t slotView = blockIdx.x;
    const size_t hypothesis = slotView / c.views;
    const int view = int(slotView % c.views), particle = int(hypothesis / c.hypotheses);
    float cross = 0, power = 0;
    const size_t frequencies = size_t(c.box) * (c.box / 2 + 1);
    const size_t offset = (size_t(particle) * c.views + view) * frequencies;
    if (c.seedIds[hypothesis] >= 0)
        for (size_t id = threadIdx.x; id < frequencies; id += blockDim.x)
        {
            const int x = int(id % (c.box / 2 + 1));
            int y = int(id / (c.box / 2 + 1));
            if (y > c.box / 2) y -= c.box;
            const bool included = bfgs::IndependentSample(x, y, c.box, c.cutoffSquaredLimit);
            if (!included) continue;
            const float weight = c.noise[offset + id];
            if (weight == 0) continue;
            Complex<float> model, unused[1];
            Model<false>(c, c.geometry + (size_t(particle) * c.views + view) * 18,
                c.poses + hypothesis * 12, offset + id, x, y, model, unused);
            const float2 observation = c.data[offset + id];
            const Complex<float> data(observation.x, observation.y), predicted(model.re, model.im);
            cross += weight * Inner(data, predicted);
            power += weight * Inner(predicted, predicted);
        }
    __shared__ float partial[2][4];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        cross += __shfl_down_sync(0xffffffffu, cross, offset);
        power += __shfl_down_sync(0xffffffffu, power, offset);
    }
    if ((threadIdx.x & 31) == 0) { partial[0][threadIdx.x / 32] = cross; partial[1][threadIdx.x / 32] = power; }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        cross = power = 0;
        for (int i = 0; i < 4; ++i) { cross += partial[0][i]; power += partial[1][i]; }
        c.tiltStats[slotView * 2] = cross;
        c.tiltStats[slotView * 2 + 1] = power;
    }
}

__device__ bool Precedes(const BatchContext& c, int first, int second)
{
    const float firstZ = float(c.summary[size_t(first) * 4]) / sqrtf(float(c.summary[size_t(first) * 4 + 1]));
    const float secondZ = float(c.summary[size_t(second) * 4]) / sqrtf(float(c.summary[size_t(second) * 4 + 1]));
    return firstZ > secondZ || (firstZ == secondZ &&
        (c.seedIds[first] < c.seedIds[second] || (c.seedIds[first] == c.seedIds[second] && first < second)));
}

// Greedy suppression uses only retained hypotheses, with stable original seed
// ties. Poses stay in their original slots so diagnostic provenance is preserved.
__global__ void Merge(BatchContext c, int symmetryCount, float distance, float angle)
{
    const int particle = blockIdx.x, base = particle * c.hypotheses;
    int* order = c.order + base;
    __shared__ int count, retained, mergedInto;
    if (threadIdx.x == 0) { count = 0; retained = 0; }
    __syncthreads();
    for (int slot = threadIdx.x; slot < c.hypotheses; slot += blockDim.x)
        if (c.seedIds[base + slot] >= 0)
        {
            int rank = 0;
            for (int other = 0; other < c.hypotheses; ++other)
                if (c.seedIds[base + other] >= 0 && other != slot && Precedes(c, base + other, base + slot)) ++rank;
            order[rank] = slot;
            atomicAdd(&count, 1);
        }
    __syncthreads();
    const float sine = sinf(0.5f * angle);
    const float differenceThreshold = 8.0f * sine * sine;
    for (int position = 0; position < count; ++position)
    {
        const int candidate = order[position];
        const float* pose = c.poses + size_t(base + candidate) * 12;
        if (threadIdx.x == 0) mergedInto = INT_MAX;
        __syncthreads();
        for (size_t comparison = threadIdx.x; comparison < size_t(retained) * symmetryCount; comparison += blockDim.x)
        {
            const int retainedPosition = int(comparison / symmetryCount);
            const int symmetry = int(comparison % symmetryCount);
            const int better = order[retainedPosition];
            const float* other = c.poses + size_t(base + better) * 12;
            float displacement2 = 0;
            for (int axis = 0; axis < 3; ++axis) { const float delta = pose[axis] - other[axis]; displacement2 = fmaf(delta, delta, displacement2); }
            if (displacement2 < distance * distance && CloseRotationBfgs(other + 3, pose + 3, c.symmetry + symmetry * 9, differenceThreshold))
                atomicMin(&mergedInto, retainedPosition);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            if (mergedInto == INT_MAX) order[retained++] = candidate;
            else
            {
                c.diagnostics[size_t(base + candidate) * 4 + 3] = order[mergedInto];
                c.seedIds[base + candidate] = -1;
            }
        }
        __syncthreads();
    }
}

bool Finite(const float* values, size_t count)
{
    if (!values) return false;
    for (size_t i = 0; i < count; ++i) if (!std::isfinite(values[i])) return false;
    return true;
}

void Release(BatchContext& c)
{
    cudaFree(c.geometry); cudaFree(c.bounds); cudaFree(c.symmetry); cudaFree(c.poses);
    cudaFree(c.seedIds); cudaFree(c.diagnostics); cudaFree(c.order);
    cudaFree(c.summary); cudaFree(c.tiltStats);
}

template <class T> cudaError_t AllocateCopy(T*& target, const T* source, size_t count)
{
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&target), count * sizeof(T));
    if (status == cudaSuccess && source) status = cudaMemcpy(target, source, count * sizeof(T), cudaMemcpyHostToDevice);
    return status;
}
}

static int RunTemplateMatchRefineBatch(
    unsigned long long textureRe, unsigned long long textureIm,
    int dim, int box, int views, int particles, int hypotheses,
    const float2* d_data, const float* d_ctf, const float* d_quad,
    const float* d_inverseNoise, const float* d_phaseRadii,
    const float* h_geometry, const float* h_bounds, const float* h_symmetry,
    int symmetryCount, float* h_poses, int* h_seedIds,
    float pixel, float cutoff, float diameter, int maxIterations,
    float mergeDistance, float mergeAngle,
    double* h_summary, int* h_diagnostics, double* h_tiltStats)
{
    if (!textureRe || !textureIm || dim < 4 || box < 4 || box % 2 || views < 1 ||
        particles < 1 || particles > INT_MAX / 6 || hypotheses < 1 ||
        symmetryCount < 1 || symmetryCount > INT_MAX / 9 || maxIterations < 0 ||
        !d_data || !d_ctf || !d_quad || !d_inverseNoise || !d_phaseRadii || !h_seedIds || !h_poses ||
        !h_summary || !h_diagnostics || !h_tiltStats ||
        !std::isfinite(pixel) || !(pixel > 0) || !std::isfinite(cutoff) || !(cutoff > 0) || cutoff > box / 2.0f ||
        !std::isfinite(diameter) || !(diameter > 0) || !std::isfinite(mergeDistance) || mergeDistance < 0 ||
        !std::isfinite(mergeAngle) || mergeAngle < 0 || mergeAngle > Pi) return -1;
    const size_t modes = size_t(particles) * hypotheses, particleViews = size_t(particles) * views;
    if (modes > INT_MAX || modes > size_t(INT_MAX) / views || particleViews > SIZE_MAX / 18 / sizeof(float) ||
        size_t(box) > SIZE_MAX / size_t(box / 2 + 1) / particleViews / sizeof(float2) ||
        size_t(symmetryCount) > SIZE_MAX / 9 / sizeof(float)) return -1;
    if (!Finite(h_geometry, particleViews * 18) || !Finite(h_bounds, size_t(particles) * 6) ||
        !Finite(h_symmetry, size_t(symmetryCount) * 9)) return -1;
    for (int symmetry = 0; symmetry < symmetryCount; ++symmetry)
        if (!ProperRotation(h_symmetry + size_t(symmetry) * 9)) return -1;
    for (int particle = 0; particle < particles; ++particle)
    {
        for (int axis = 0; axis < 3; ++axis)
            if (h_bounds[particle * 6 + axis] > h_bounds[particle * 6 + axis + 3]) return -1;
        for (int slot = 0; slot < hypotheses; ++slot)
        {
            const size_t mode = size_t(particle) * hypotheses + slot;
            if (h_seedIds[mode] < -1) return -1;
        }
    }
    BatchContext c = {};
    c.textureRe = textureRe; c.textureIm = textureIm; c.dim = dim; c.box = box;
    c.views = views; c.particles = particles; c.hypotheses = hypotheses; c.maxIterations = maxIterations;
    c.pixel = pixel; c.cutoff = cutoff; c.diameter = diameter;
    // Integer frequency coordinates let FP32 kernels preserve the original
    // double-precision cutoff decision without FP64 work per Fourier sample.
    c.cutoffSquaredLimit = static_cast<unsigned long long>(std::floor(double(cutoff) * cutoff));
    c.data = d_data; c.ctf = d_ctf; c.quadrature = d_quad; c.noise = d_inverseNoise; c.phaseRadii = d_phaseRadii;
    cudaError_t status = AllocateCopy(c.geometry, h_geometry, particleViews * 18);
    if (status == cudaSuccess) status = AllocateCopy(c.bounds, h_bounds, size_t(particles) * 6);
    if (status == cudaSuccess) status = AllocateCopy(c.symmetry, h_symmetry, size_t(symmetryCount) * 9);
    if (status == cudaSuccess) status = AllocateCopy(c.poses, h_poses, modes * 12);
    if (status == cudaSuccess) status = AllocateCopy(c.seedIds, h_seedIds, modes);
    if (status == cudaSuccess) status = AllocateCopy(c.diagnostics, static_cast<const int*>(nullptr), modes * 4);
    if (status == cudaSuccess) status = AllocateCopy(c.order, static_cast<const int*>(nullptr), modes);
    if (status == cudaSuccess) status = AllocateCopy(c.summary, static_cast<const double*>(nullptr), modes * 4);
    if (status == cudaSuccess) status = AllocateCopy(c.tiltStats, static_cast<const double*>(nullptr), modes * views * 2);
    if (status == cudaSuccess)
    {
        OptimizeBfgs<<<unsigned(modes), Threads>>>(c);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
    {
        FinalTiltStatistics<<<unsigned(modes * views), Threads>>>(c);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
    {
        Merge<<<particles, Threads>>>(c, symmetryCount, mergeDistance, mergeAngle);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpy(h_poses, c.poses, modes * 12 * sizeof(float), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(h_seedIds, c.seedIds, modes * sizeof(int), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(h_summary, c.summary, modes * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(h_diagnostics, c.diagnostics, modes * 4 * sizeof(int), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) status = cudaMemcpy(h_tiltStats, c.tiltStats, modes * views * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    Release(c);
    return int(status);
}

extern "C" __declspec(dllexport) int TemplateMatchRefineBatchBfgs(
    unsigned long long textureRe, unsigned long long textureIm,
    int dim, int box, int views, int particles, int hypotheses,
    const float2* d_data, const float* d_ctf, const float* d_quad,
    const float* d_inverseNoise, const float* d_phaseRadii,
    const float* h_geometry, const float* h_bounds, const float* h_symmetry,
    int symmetryCount, float* h_poses, int* h_seedIds,
    float pixel, float cutoff, float diameter, int maxIterations,
    float mergeDistance, float mergeAngle,
    double* h_summary, int* h_diagnostics, double* h_tiltStats)
{
    return RunTemplateMatchRefineBatch(textureRe, textureIm, dim, box, views, particles, hypotheses,
        d_data, d_ctf, d_quad, d_inverseNoise, d_phaseRadii, h_geometry, h_bounds, h_symmetry,
        symmetryCount, h_poses, h_seedIds, pixel, cutoff, diameter, maxIterations,
        mergeDistance, mergeAngle, h_summary, h_diagnostics, h_tiltStats);
}

static int RunTemplateMatchEnvelopeSpectra(bool byTilt,
    unsigned long long textureRe, unsigned long long textureIm,
    int dim, int box, int views, int particles,
    const float2* d_data, const float* d_ctf, const float* d_quad,
    const float* d_inverseNoise, const float* d_phaseRadii,
    const float* h_geometry, const float* h_poses, const int* h_active,
    float cutoff, int bins, float minimumQ2, float maximumQ2, float* h_spectra)
{
    if (!textureRe || !textureIm || dim < 4 || box < 4 || box % 2 || views < 1 ||
        particles < 1 || particles > 65535 || bins < 2 || bins > 1048576 ||
        !d_data || !d_ctf || !d_quad || !d_inverseNoise || !d_phaseRadii || !h_active || !h_spectra ||
        !std::isfinite(cutoff) || cutoff <= 0 || cutoff > box / 2.0f ||
        !std::isfinite(maximumQ2) || maximumQ2 <= 0 ||
        !std::isfinite(minimumQ2) || minimumQ2 < 0 || minimumQ2 >= maximumQ2) return -1;
    const size_t pv = size_t(particles) * views;
    if (pv > SIZE_MAX / size_t(bins) / 2 / sizeof(float)) return -1;
    const size_t outputCount = (byTilt ? pv : size_t(particles)) * bins * 2;
    if (pv > SIZE_MAX / 18 / sizeof(float) || outputCount > SIZE_MAX / sizeof(float) ||
        size_t(box) > SIZE_MAX / (box / 2 + 1) / pv / sizeof(float2) ||
        !Finite(h_geometry, pv * 18) || !h_poses) return -1;
    for (int p = 0; p < particles; p++)
        if (h_active[p] >= 0 && (!Finite(h_poses + p * 12, 12) || !ProperRotation(h_poses + p * 12 + 3))) return -1;
    BatchContext c = {};
    c.textureRe = textureRe; c.textureIm = textureIm; c.dim = dim; c.box = box;
    c.views = views; c.particles = particles;
    c.cutoffSquaredLimit = static_cast<unsigned long long>(std::floor(double(cutoff) * cutoff));
    c.data = d_data; c.ctf = d_ctf; c.quadrature = d_quad; c.noise = d_inverseNoise; c.phaseRadii = d_phaseRadii;
    float* spectra = nullptr;
    cudaError_t status = AllocateCopy(c.geometry, h_geometry, pv * 18);
    if (status == cudaSuccess) status = AllocateCopy(c.poses, h_poses, size_t(particles) * 12);
    if (status == cudaSuccess) status = AllocateCopy(c.seedIds, h_active, size_t(particles));
    if (status == cudaSuccess) status = AllocateCopy(spectra, static_cast<const float*>(nullptr), outputCount);
    if (status == cudaSuccess) status = cudaMemset(spectra, 0, outputCount * sizeof(float));
    if (status == cudaSuccess)
    {
        EnvelopeSpectra<<<dim3(64, particles), 128>>>(c, bins, minimumQ2, maximumQ2, byTilt, spectra);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaMemcpy(h_spectra, spectra, outputCount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(spectra); Release(c);
    return int(status);
}

extern "C" __declspec(dllexport) int TemplateMatchEnvelopeSpectra(
    unsigned long long textureRe, unsigned long long textureIm, int dim, int box, int views, int particles,
    const float2* data, const float* ctf, const float* quad, const float* inverseNoise, const float* phaseRadii,
    const float* geometry, const float* poses, const int* active,
    float cutoff, int bins, float minimumQ2, float maximumQ2, float* spectra)
{
    return RunTemplateMatchEnvelopeSpectra(false, textureRe, textureIm, dim, box, views, particles,
        data, ctf, quad, inverseNoise, phaseRadii, geometry, poses, active, cutoff, bins, minimumQ2, maximumQ2, spectra);
}

extern "C" __declspec(dllexport) int TemplateMatchEnvelopeSpectraByTilt(
    unsigned long long textureRe, unsigned long long textureIm, int dim, int box, int views, int particles,
    const float2* data, const float* ctf, const float* quad, const float* inverseNoise, const float* phaseRadii,
    const float* geometry, const float* poses, const int* active,
    float cutoff, int bins, float minimumQ2, float maximumQ2, float* spectra)
{
    return RunTemplateMatchEnvelopeSpectra(true, textureRe, textureIm, dim, box, views, particles,
        data, ctf, quad, inverseNoise, phaseRadii, geometry, poses, active, cutoff, bins, minimumQ2, maximumQ2, spectra);
}
