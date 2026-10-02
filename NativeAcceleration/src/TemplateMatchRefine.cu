#include "Functions.h"
#include "TemplateMatchRefineMath.h"
#include <new>
#include <limits>

namespace
{
using warp_template_match::Complex;
using warp_template_match::Inner;
const int Threads = 128;
const int Statistics = 14;
const int ParametersPerView = 84;

struct TemplateMatchContext
{
    cudaTextureObject_t textureRe, textureIm;
    int dim, box, views, blocks, device;
    const float2* data;
    const float* ctfBase;
    const float* ctfQuadrature;
    const float* inverseNoise;
    const float* phaseRadiusSquared;
    float* parameters = nullptr;
    double* partial = nullptr;
    double* result = nullptr;
};

struct TextureFetch
{
    cudaTextureObject_t re, im;
    __device__ Complex<float> operator()(int x, int y, int z) const
    {
        // Fetch exact voxel centers; manual interpolation below avoids the
        // texture unit's quantized interpolation weights and derivatives.
        return Complex<float>(tex3D<float>(re, x + 0.5f, y + 0.5f, z + 0.5f),
                              tex3D<float>(im, x + 0.5f, y + 0.5f, z + 0.5f));
    }
};

__global__ void ScoreTemplateViews(TemplateMatchContext context, float cutoffRadius)
{
    const int view = blockIdx.y;
    const size_t elements = size_t(context.box) * (context.box / 2 + 1);
    const size_t offset = size_t(view) * elements;
    const float* matrix = context.parameters + view * 9;
    const float* matrixDerivatives = context.parameters + context.views * 9 + view * 54;
    const float* shift = context.parameters + context.views * 63 + view * 2;
    const float* shiftDerivatives = context.parameters + context.views * 65 + view * 12;
    const float beta = context.parameters[context.views * 77 + view];
    const float* betaDerivatives = context.parameters + context.views * 78 + view * 6;
    double sums[Statistics] = {};
    const TextureFetch fetch = {context.textureRe, context.textureIm};
    for (size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         id < elements; id += size_t(gridDim.x) * blockDim.x)
    {
        const int x = int(id % (context.box / 2 + 1));
        int y = int(id / (context.box / 2 + 1));
        if (y > context.box / 2) y -= context.box;
        if (!warp_template_match::IndependentSample(x, y, context.box, cutoffRadius)) continue;
        const float weight = context.inverseNoise[offset + id];
        if (weight == 0.0f) continue;
        Complex<float> model, derivatives[6];
        warp_template_match::ModelAndDerivatives<float>(
            fetch, context.dim, context.box, x, y, matrix, matrixDerivatives,
            shift, shiftDerivatives, beta, betaDerivatives,
            context.ctfBase[offset + id], context.ctfQuadrature[offset + id], model, derivatives,
            context.phaseRadiusSquared[offset + id]);
        const float2 observed = context.data[offset + id];
        const Complex<double> data(observed.x, observed.y), predicted(model.re, model.im);
        sums[0] += double(weight) * Inner(data, predicted);
        sums[1] += double(weight) * Inner(predicted, predicted);
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            const Complex<double> derivative(derivatives[parameter].re, derivatives[parameter].im);
            sums[2 + parameter] += double(weight) * Inner(data, derivative);
            sums[8 + parameter] += 2.0 * double(weight) * Inner(predicted, derivative);
        }
    }
    __shared__ double reduction[Statistics][Threads];
    for (int statistic = 0; statistic < Statistics; ++statistic)
        reduction[statistic][threadIdx.x] = sums[statistic];
    __syncthreads();
    for (int stride = Threads / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            for (int statistic = 0; statistic < Statistics; ++statistic)
                reduction[statistic][threadIdx.x] += reduction[statistic][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x < Statistics)
        context.partial[(size_t(view) * context.blocks + blockIdx.x) * Statistics + threadIdx.x]
            = reduction[threadIdx.x][0];
}

__global__ void ReduceTemplateViews(const double* partial, double* result, int blocks)
{
    const int statistic = threadIdx.x;
    if (statistic >= Statistics) return;
    double sum = 0;
    for (int block = 0; block < blocks; ++block)
        sum += partial[(size_t(blockIdx.x) * blocks + block) * Statistics + statistic];
    result[size_t(blockIdx.x) * Statistics + statistic] = sum;
}

bool FiniteArray(const float* values, size_t count)
{
    if (!values) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

void Release(TemplateMatchContext* context)
{
    if (context->parameters) cudaFree(context->parameters);
    if (context->partial) cudaFree(context->partial);
    if (context->result) cudaFree(context->result);
    delete context;
}
}

// Device data and texture lifetimes remain caller-owned. A context is tied to
// the creating CUDA device and must not be evaluated concurrently.
extern "C" __declspec(dllexport) int TemplateMatchRefineCreate(
    unsigned long long textureRe, unsigned long long textureIm,
    int dimProjector, int box, int views, const float2* data,
    const float* ctfBase, const float* ctfQuadrature, const float* inverseNoise,
    const float* phaseRadiusSquared, void** result)
{
    if (!result) return -1;
    *result = nullptr;
    if (!textureRe || !textureIm || dimProjector < 4 || box < 4 || box % 2 ||
        views < 1 || views > 65535 || !data || !ctfBase || !ctfQuadrature || !inverseNoise || !phaseRadiusSquared) return -1;
    TemplateMatchContext* context = new (std::nothrow) TemplateMatchContext();
    if (!context) return -2;
    context->textureRe = textureRe;
    context->textureIm = textureIm;
    context->dim = dimProjector;
    context->box = box;
    context->views = views;
    const size_t elements = size_t(box) * (box / 2 + 1);
    context->blocks = int(std::min(size_t(64), (elements + Threads - 1) / Threads));
    context->data = data;
    context->ctfBase = ctfBase;
    context->ctfQuadrature = ctfQuadrature;
    context->inverseNoise = inverseNoise;
    context->phaseRadiusSquared = phaseRadiusSquared;
    cudaError_t status = cudaGetDevice(&context->device);
    if (status == cudaSuccess)
        status = cudaMalloc(reinterpret_cast<void**>(&context->parameters), size_t(views) * ParametersPerView * sizeof(float));
    if (status == cudaSuccess)
        status = cudaMalloc(reinterpret_cast<void**>(&context->partial), size_t(views) * context->blocks * Statistics * sizeof(double));
    if (status == cudaSuccess)
        status = cudaMalloc(reinterpret_cast<void**>(&context->result), size_t(views) * Statistics * sizeof(double));
    if (status != cudaSuccess)
    {
        Release(context);
        return int(status);
    }
    *result = context;
    return 0;
}

extern "C" __declspec(dllexport) int TemplateMatchRefineEvaluate(
    void* handle, const float* matrices, const float* matrixDerivatives,
    const float* shifts, const float* shiftDerivatives, const float* phaseCoefficients,
    const float* phaseDerivatives, float cutoffRadius, double* output)
{
    TemplateMatchContext* context = static_cast<TemplateMatchContext*>(handle);
    if (!context || !output || !std::isfinite(cutoffRadius) || cutoffRadius <= 0 || cutoffRadius > context->box / 2.0f) return -1;
    const size_t count = size_t(context->views);
    const float* arrays[] = {matrices, matrixDerivatives, shifts, shiftDerivatives, phaseCoefficients, phaseDerivatives};
    const size_t widths[] = {9, 54, 2, 12, 1, 6};
    for (int i = 0; i < 6; ++i)
        if (!FiniteArray(arrays[i], count * widths[i])) return -1;
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return int(status);
    if (device != context->device) return -1;
    size_t offset = 0;
    for (int i = 0; i < 6; ++i)
    {
        status = cudaMemcpy(context->parameters + offset, arrays[i], count * widths[i] * sizeof(float), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return int(status);
        offset += count * widths[i];
    }
    ScoreTemplateViews<<<dim3(context->blocks, context->views), Threads>>>(*context, cutoffRadius);
    status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    ReduceTemplateViews<<<context->views, 32>>>(context->partial, context->result, context->blocks);
    status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
    status = cudaMemcpy(output, context->result, count * Statistics * sizeof(double), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return int(status);
    for (size_t i = 0; i < count * Statistics; ++i)
        if (!std::isfinite(output[i])) return -3;
    return 0;
}

extern "C" __declspec(dllexport) int TemplateMatchRefineDestroy(void* handle)
{
    if (!handle) return 0;
    TemplateMatchContext* context = static_cast<TemplateMatchContext*>(handle);
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return int(status);
    if (device != context->device) return -1;
    // Free every allocation even when CUDA reports an earlier asynchronous error.
    cudaError_t first = cudaSuccess;
    cudaError_t current = cudaFree(context->parameters);
    if (current != cudaSuccess) first = current;
    current = cudaFree(context->partial);
    if (first == cudaSuccess && current != cudaSuccess) first = current;
    current = cudaFree(context->result);
    if (first == cudaSuccess && current != cudaSuccess) first = current;
    delete context;
    return int(first);
}
