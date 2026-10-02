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
constexpr int Statistics = 35;

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

// Keep the expensive sampler/evaluation out of the trust controller's register
// lifetime. A score-only instantiation omits the six Jacobians and Gram matrix.
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

struct Workspace
{
    float current[12], trial[12];
    double stats[Statistics], trialStats[2];
    double reduction[4][Statistics];
    double gradient[6], hessian[21];
    double matrix[36], vectors[36], values[6], rotatedGradient[6], step[6];
    double trust, maxTrust, predicted, z, initialZ, stepNorm, rotationNorm, translationNorm;
    int active, terminate, termination, usable, accepted, evaluations, rejected, trustInterior;
};

template <bool Derivatives>
__device__ __noinline__ void Evaluate(const BatchContext& c, int particle,
                                    const float* pose, Workspace& workspace,
                                    double* result)
{
    constexpr int Count = Derivatives ? Statistics : 2;
    double sums[Count] = {};
    const size_t frequencies = size_t(c.box) * (c.box / 2 + 1);
    const size_t samples = frequencies * c.views;
    const size_t particleOffset = size_t(particle) * samples;
    for (size_t index = threadIdx.x; index < samples; index += blockDim.x)
    {
        const int view = int(index / frequencies);
        const size_t frequency = index - size_t(view) * frequencies;
        const int x = int(frequency % (c.box / 2 + 1));
        int y = int(frequency / (c.box / 2 + 1));
        if (y > c.box / 2) y -= c.box;
        if (!warp_template_match::IndependentSample(x, y, c.box, c.cutoff)) continue;
        const size_t sampleIndex = particleOffset + index;
        const float weight = c.noise[sampleIndex];
        if (weight == 0) continue; // Excluded observations must not be read.
        if (!(weight > 0) || !isfinite(weight)) { sums[0] = CUDART_NAN; continue; }
        Complex<float> model, derivatives[Derivatives ? 6 : 1];
        Model<Derivatives>(c, c.geometry + (size_t(particle) * c.views + view) * 18,
                           pose, sampleIndex, x, y, model, derivatives);
        const float2 observation = c.data[sampleIndex];
        const Complex<double> observed(observation.x, observation.y), predicted(model.re, model.im);
        sums[0] += double(weight) * Inner(observed, predicted);
        sums[1] += double(weight) * Inner(predicted, predicted);
        if (Derivatives)
            for (int row = 0; row < 6; ++row)
            {
                const Complex<double> derivative(derivatives[row].re, derivatives[row].im);
                sums[2 + row] += double(weight) * Inner(observed, derivative);
                sums[8 + row] += double(weight) * Inner(predicted, derivative);
                for (int column = row; column < 6; ++column)
                {
                    const Complex<double> other(derivatives[column].re, derivatives[column].im);
                    sums[14 + UpperIndex(row, column)] += double(weight) * Inner(derivative, other);
                }
            }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int statistic = 0; statistic < Count; ++statistic)
    {
        double value = sums[statistic];
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) workspace.reduction[warp][statistic] = value;
    }
    __syncthreads();
    if (threadIdx.x < Count)
    {
        double value = 0;
        for (int worker = 0; worker < 4; ++worker) value += workspace.reduction[worker][threadIdx.x];
        result[threadIdx.x] = value;
    }
    __syncthreads();
}

__device__ double Norm3(const double* values)
{
    return sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

__device__ void Shrink(Workspace& w)
{
    w.trust = fmax(w.maxTrust * (8.0 * FLT_EPSILON), w.trust * 0.25);
}

__global__ void Optimize(BatchContext c)
{
    const int hypothesis = int(blockIdx.x);
    const int particle = hypothesis / c.hypotheses;
    __shared__ Workspace w;
    if (threadIdx.x == 0)
    {
        w.active = c.seedIds[hypothesis] >= 0;
        w.accepted = w.evaluations = w.rejected = 0;
        w.terminate = !w.active;
        w.termination = w.active ? 0 : 3;
        w.z = w.initialZ = 0;
        w.stats[0] = w.stats[1] = 0;
        w.maxTrust = hypot(0.5 * c.diameter * MaximumRotation, 2.0 * c.pixel);
        w.trust = w.maxTrust;
        for (int entry = 0; entry < 12; ++entry) w.current[entry] = c.poses[size_t(hypothesis) * 12 + entry];
        if (w.active)
        {
            bool valid = ProperRotation(w.current + 3);
            for (int axis = 0; axis < 3; ++axis)
                valid = valid && isfinite(w.current[axis]) &&
                    w.current[axis] >= c.bounds[particle * 6 + axis] &&
                    w.current[axis] <= c.bounds[particle * 6 + 3 + axis];
            if (!valid) { w.terminate = 1; w.termination = 4; }
        }
    }
    __syncthreads();
    if (w.active && !w.terminate)
    {
        Evaluate<true>(c, particle, w.current, w, w.stats);
        if (threadIdx.x == 0)
        {
            ++w.evaluations;
            if (!SignedNormal(w.stats, w.z, w.gradient, w.hessian))
            { w.termination = 4; w.terminate = 1; }
            w.initialZ = w.z;
            if (c.maxIterations == 0) w.terminate = 1;
        }
        __syncthreads();
    }
    while (!w.terminate)
    {
        if (threadIdx.x == 0)
        {
            w.usable = TrustStep(w.gradient, w.hessian, 0.5 * c.diameter, w.trust,
                w.matrix, w.vectors, w.values, w.rotatedGradient, w.step);
            if (w.usable)
            {
                const double translation = Norm3(w.step), rotation = Norm3(w.step + 3);
                // A contracted trust radius can force a small step far from an
                // optimum. Record interiority before pose caps, box clipping,
                // and the trial's gain-based trust-radius update.
                w.trustInterior = hypot(translation, 0.5 * c.diameter * rotation) < 0.8 * w.trust;
                if (translation > 2.0 * c.pixel)
                    for (int i = 0; i < 3; ++i) w.step[i] *= 2.0 * c.pixel / translation;
                if (rotation > MaximumRotation)
                    for (int i = 3; i < 6; ++i) w.step[i] *= MaximumRotation / rotation;
                for (int i = 0; i < 3; ++i)
                {
                    w.trial[i] = float(Maximum(double(c.bounds[particle * 6 + i]),
                        Minimum(double(c.bounds[particle * 6 + 3 + i]), w.current[i] + w.step[i])));
                    w.step[i] = double(w.trial[i]) - w.current[i];
                }
                RotateRight(w.current + 3, w.step + 3, w.trial + 3);
                w.predicted = 0;
                for (int row = 0; row < 6; ++row)
                {
                    w.predicted -= w.gradient[row] * w.step[row];
                    for (int column = 0; column < 6; ++column)
                        w.predicted -= 0.5 * w.step[row] * w.hessian[UpperIndex(row, column)] * w.step[column];
                }
                w.translationNorm = Norm3(w.step);
                w.rotationNorm = Norm3(w.step + 3);
                w.stepNorm = hypot(w.translationNorm, 0.5 * c.diameter * w.rotationNorm);
                w.usable = isfinite(w.predicted) && w.predicted > 0;
            }
            if (!w.usable)
            {
                Shrink(w);
                if (++w.rejected >= 5) { w.terminate = 1; w.termination = 2; }
            }
        }
        __syncthreads();
        if (!w.usable) continue;
        Evaluate<false>(c, particle, w.trial, w, w.trialStats);
        if (threadIdx.x == 0)
        {
            ++w.evaluations;
            const double trialZ = w.trialStats[0] / sqrt(w.trialStats[1]);
            w.usable = 0; // Now denotes an accepted step requiring new derivatives.
            if (!(w.trialStats[1] > 0) || !isfinite(trialZ) || !isfinite(w.trialStats[1]))
            {
                // A finite current pose is still usable when a trial leaves
                // Fourier support or overflows. Reject it and contract the trust.
                Shrink(w);
                if (++w.rejected >= 5) { w.terminate = 1; w.termination = 2; }
            }
            else
            {
                const double actual = trialZ - w.z, gain = actual / w.predicted;
                if (!isfinite(gain) || gain < 0.25) Shrink(w);
                else if (gain > 0.75 && w.stepNorm >= 0.8 * w.trust) w.trust = fmin(w.maxTrust, 2 * w.trust);
                // The forward model is float32, reductions are float64. Avoid
                // treating rounding-level signed-Z fluctuations as improvements.
                const double tolerance = 8.0 * FLT_EPSILON * fmax(1.0, fabs(w.z));
                if (actual > tolerance && isfinite(gain) && gain >= 1.0e-4)
                {
                    for (int entry = 0; entry < 12; ++entry) w.current[entry] = w.trial[entry];
                    w.stats[0] = w.trialStats[0]; w.stats[1] = w.trialStats[1];
                    w.z = trialZ;
                    ++w.accepted; w.rejected = 0;
                    if (w.trustInterior && w.rotationNorm <= RotationTolerance &&
                        w.translationNorm <= TranslationTolerancePixels * c.pixel)
                    { w.terminate = 1; w.termination = 1; }
                    else if (w.accepted >= c.maxIterations) { w.terminate = 1; w.termination = 0; }
                    else w.usable = 1;
                }
                else if (++w.rejected >= 5) { w.terminate = 1; w.termination = 2; }
            }
        }
        __syncthreads();
        if (w.usable)
        {
            Evaluate<true>(c, particle, w.current, w, w.stats);
            if (threadIdx.x == 0)
            {
                ++w.evaluations;
                if (!SignedNormal(w.stats, w.z, w.gradient, w.hessian))
                { w.terminate = 1; w.termination = 4; }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
    {
        for (int entry = 0; entry < 12; ++entry) c.poses[size_t(hypothesis) * 12 + entry] = w.current[entry];
        const bool valid = w.active && w.termination != 4;
        c.summary[size_t(hypothesis) * 4] = valid ? w.stats[0] : 0;
        c.summary[size_t(hypothesis) * 4 + 1] = valid ? w.stats[1] : 0;
        c.summary[size_t(hypothesis) * 4 + 2] = valid ? w.initialZ : 0;
        c.summary[size_t(hypothesis) * 4 + 3] = valid ? w.trust : 0;
        c.diagnostics[size_t(hypothesis) * 4] = w.accepted;
        c.diagnostics[size_t(hypothesis) * 4 + 1] = w.evaluations;
        c.diagnostics[size_t(hypothesis) * 4 + 2] = w.termination;
        c.diagnostics[size_t(hypothesis) * 4 + 3] = valid ? -1 : -2;
        if (!valid) c.seedIds[hypothesis] = -1;
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

template <class T>
__global__ void FinalTiltStatistics(BatchContext c)
{
    const size_t slotView = blockIdx.x;
    const size_t hypothesis = slotView / c.views;
    const int view = int(slotView % c.views), particle = int(hypothesis / c.hypotheses);
    T cross = 0, power = 0;
    const size_t frequencies = size_t(c.box) * (c.box / 2 + 1);
    const size_t offset = (size_t(particle) * c.views + view) * frequencies;
    if (c.seedIds[hypothesis] >= 0)
        for (size_t id = threadIdx.x; id < frequencies; id += blockDim.x)
        {
            const int x = int(id % (c.box / 2 + 1));
            int y = int(id / (c.box / 2 + 1));
            if (y > c.box / 2) y -= c.box;
            const bool included = sizeof(T) == sizeof(float)
                ? bfgs::IndependentSample(x, y, c.box, c.cutoffSquaredLimit)
                : warp_template_match::IndependentSample(x, y, c.box, c.cutoff);
            if (!included) continue;
            const float weight = c.noise[offset + id];
            if (weight == 0) continue;
            Complex<float> model, unused[1];
            Model<false>(c, c.geometry + (size_t(particle) * c.views + view) * 18,
                c.poses + hypothesis * 12, offset + id, x, y, model, unused);
            const float2 observation = c.data[offset + id];
            const Complex<T> data(observation.x, observation.y), predicted(model.re, model.im);
            cross += T(weight) * Inner(data, predicted);
            power += T(weight) * Inner(predicted, predicted);
        }
    __shared__ T partial[2][4];
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

template <bool FloatMath>
__device__ bool Precedes(const BatchContext& c, int first, int second)
{
    if (FloatMath)
    {
        const float firstZ = float(c.summary[size_t(first) * 4]) / sqrtf(float(c.summary[size_t(first) * 4 + 1]));
        const float secondZ = float(c.summary[size_t(second) * 4]) / sqrtf(float(c.summary[size_t(second) * 4 + 1]));
        return firstZ > secondZ || (firstZ == secondZ &&
            (c.seedIds[first] < c.seedIds[second] || (c.seedIds[first] == c.seedIds[second] && first < second)));
    }
    const double firstZ = c.summary[size_t(first) * 4] / sqrt(c.summary[size_t(first) * 4 + 1]);
    const double secondZ = c.summary[size_t(second) * 4] / sqrt(c.summary[size_t(second) * 4 + 1]);
    return firstZ > secondZ || (firstZ == secondZ &&
        (c.seedIds[first] < c.seedIds[second] || (c.seedIds[first] == c.seedIds[second] && first < second)));
}

__device__ bool CloseRotation(const float* first, const float* candidate, const float* symmetry, double cosineThreshold)
{
    double trace = 0;
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
        {
            double equivalent = 0;
            for (int k = 0; k < 3; ++k) equivalent += candidate[row + k * 3] * double(symmetry[k + column * 3]);
            trace += first[row + column * 3] * equivalent;
        }
    return fmax(-1.0, fmin(1.0, (trace - 1.0) * 0.5)) > cosineThreshold;
}

// Greedy suppression uses only retained hypotheses, with stable original seed
// ties. Poses stay in their original slots so diagnostic provenance is preserved.
template <bool FloatMath>
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
                if (c.seedIds[base + other] >= 0 && other != slot && Precedes<FloatMath>(c, base + other, base + slot)) ++rank;
            order[rank] = slot;
            atomicAdd(&count, 1);
        }
    __syncthreads();
    const double cosineThreshold = FloatMath ? 0 : cos(double(angle)), distance2 = FloatMath ? 0 : double(distance) * distance;
    const float sine = FloatMath ? sinf(0.5f * angle) : 0;
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
            if (FloatMath)
            {
                float displacement2 = 0;
                for (int axis = 0; axis < 3; ++axis) { const float delta = pose[axis] - other[axis]; displacement2 = fmaf(delta, delta, displacement2); }
                if (displacement2 < distance * distance && CloseRotationBfgs(other + 3, pose + 3, c.symmetry + symmetry * 9, differenceThreshold))
                    atomicMin(&mergedInto, retainedPosition);
            }
            else
            {
                double displacement2 = 0;
                for (int axis = 0; axis < 3; ++axis) { const double delta = double(pose[axis]) - other[axis]; displacement2 += delta * delta; }
                if (displacement2 < distance2 && CloseRotation(other + 3, pose + 3, c.symmetry + symmetry * 9, cosineThreshold))
                    atomicMin(&mergedInto, retainedPosition);
            }
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

static int RunTemplateMatchRefineBatch(bool useBfgs,
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
        if (useBfgs) OptimizeBfgs<<<unsigned(modes), Threads>>>(c);
        else Optimize<<<unsigned(modes), Threads>>>(c);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
    {
        if (useBfgs) FinalTiltStatistics<float><<<unsigned(modes * views), Threads>>>(c);
        else FinalTiltStatistics<double><<<unsigned(modes * views), Threads>>>(c);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
    {
        if (useBfgs) Merge<true><<<particles, Threads>>>(c, symmetryCount, mergeDistance, mergeAngle);
        else Merge<false><<<particles, Threads>>>(c, symmetryCount, mergeDistance, mergeAngle);
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

extern "C" __declspec(dllexport) int TemplateMatchRefineBatch(
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
    return RunTemplateMatchRefineBatch(false, textureRe, textureIm, dim, box, views, particles, hypotheses,
        d_data, d_ctf, d_quad, d_inverseNoise, d_phaseRadii, h_geometry, h_bounds, h_symmetry,
        symmetryCount, h_poses, h_seedIds, pixel, cutoff, diameter, maxIterations,
        mergeDistance, mergeAngle, h_summary, h_diagnostics, h_tiltStats);
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
    return RunTemplateMatchRefineBatch(true, textureRe, textureIm, dim, box, views, particles, hypotheses,
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
