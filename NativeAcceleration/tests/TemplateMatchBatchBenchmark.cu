// Standalone bounded synthetic benchmark (not a timed unit test).
// nvcc -O3 -std=c++14 -I NativeAcceleration/include \
//   NativeAcceleration/tests/TemplateMatchBatchBenchmark.cu \
//   -L NativeAcceleration/build/lib -l:libNativeAcceleration.so \
//   -Xlinker -rpath -Xlinker NativeAcceleration/build/lib \
//   -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" --cudart shared -o /tmp/tm_batch_benchmark
// Defaults: box128,41tilts,32particles,32hypotheses,30 accepted-step budget.
// Smoke: --box 96 --particles 4 --iterations 5; --score-only skips optimization.
#include <cuda_runtime.h>
#include "TemplateMatchRefineBatch.h"
#include "TemplateMatchRefineMath.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using warp_template_match::Complex;
using namespace warp_template_match_batch;

namespace
{
void Check(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceBuffer
{
    void* pointer = nullptr;
    DeviceBuffer(const void* data, size_t bytes)
    {
        Check(cudaMalloc(&pointer, bytes));
        try { Check(cudaMemcpy(pointer, data, bytes, cudaMemcpyHostToDevice)); }
        catch (...) { cudaFree(pointer); pointer = nullptr; throw; }
    }
    ~DeviceBuffer() { cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

struct Texture
{
    cudaArray_t array = nullptr;
    cudaTextureObject_t texture = 0;
    Texture(const std::vector<float>& data, int width, int dim)
    {
        const cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();
        Check(cudaMalloc3DArray(&array, &channel, make_cudaExtent(width, dim, dim)));
        try
        {
            cudaMemcpy3DParms copy = {};
            copy.srcPtr = make_cudaPitchedPtr(const_cast<float*>(data.data()), width * sizeof(float), width, dim);
            copy.dstArray = array;
            copy.extent = make_cudaExtent(width, dim, dim);
            copy.kind = cudaMemcpyHostToDevice;
            Check(cudaMemcpy3D(&copy));
            cudaResourceDesc resource = {};
            resource.resType = cudaResourceTypeArray;
            resource.res.array.array = array;
            cudaTextureDesc description = {};
            description.filterMode = cudaFilterModePoint;
            description.readMode = cudaReadModeElementType;
            description.addressMode[0] = description.addressMode[1] = description.addressMode[2] = cudaAddressModeClamp;
            Check(cudaCreateTextureObject(&texture, &resource, &description, nullptr));
        }
        catch (...) { cudaFreeArray(array); array = nullptr; throw; }
    }
    ~Texture() { if (texture) cudaDestroyTextureObject(texture); if (array) cudaFreeArray(array); }
    Texture(const Texture&) = delete;
};

struct HostFetch
{
    const float* re;
    const float* im;
    int width, dim;
    __host__ __device__ Complex<float> operator()(int x, int y, int z) const
    {
        const size_t index = (size_t(z) * dim + y) * width + x;
        return Complex<float>(re[index], im[index]);
    }
};

Complex<float> Model(HostFetch fetch, int dim, int box, int x, int y,
                     const float* geometry, const float* pose,
                     float ctf, float quad, float radius2)
{
    const float gx = geometry[0] * x + geometry[3] * y;
    const float gy = geometry[1] * x + geometry[4] * y;
    const float gz = geometry[2] * x + geometry[5] * y;
    const float* r = pose + 3;
    const auto sample = warp_template_match::Interpolate<float>(fetch, dim,
        r[0] * gx + r[1] * gy + r[2] * gz,
        r[3] * gx + r[4] * gy + r[5] * gz,
        r[6] * gx + r[7] * gy + r[8] * gz);
    const float sx = geometry[9] * pose[0] + geometry[11] * pose[1] + geometry[13] * pose[2];
    const float sy = geometry[10] * pose[0] + geometry[12] * pose[1] + geometry[14] * pose[2];
    const float beta = geometry[15] * pose[0] + geometry[16] * pose[1] + geometry[17] * pose[2];
    const float phase = float(-2 * Pi) / box * (x * sx + y * sy);
    const float transfer = ctf * std::cos(beta * radius2) + quad * std::sin(beta * radius2);
    return sample.value * Complex<float>(std::cos(phase), std::sin(phase)) * transfer;
}

int IntegerArgument(int argc, char** argv, const char* key, int fallback)
{
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], key) == 0)
        {
            if (i + 1 == argc) throw std::runtime_error(std::string("Missing value for ") + key);
            char* end = nullptr;
            const long value = std::strtol(argv[i + 1], &end, 10);
            if (*end || value < 1 || value > 4096) throw std::runtime_error(std::string("Invalid value for ") + key);
            return int(value);
        }
    return fallback;
}
}

int main(int argc, char** argv)
{
    try
    {
        const int box = IntegerArgument(argc, argv, "--box", 128);
        const int views = IntegerArgument(argc, argv, "--views", 41);
        const int particles = IntegerArgument(argc, argv, "--particles", 32);
        const int hypotheses = IntegerArgument(argc, argv, "--hypotheses", 32);
        const int iterations = IntegerArgument(argc, argv, "--iterations", 30);
        bool scoreOnly = false;
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--score-only") == 0) scoreOnly = true;
        }
        const char* optimizerName = "bfgs";
        const auto refine = TemplateMatchRefineBatchBfgs;
        if (box % 2 || box < 8) throw std::runtime_error("Box must be even and >=8");
        const int dim = 2 * box + 3, width = dim / 2 + 1;
        const size_t frequencies = size_t(box) * (box / 2 + 1);
        const size_t samples = size_t(particles) * views * frequencies;
        const size_t modes = size_t(particles) * hypotheses;
        const size_t volumeElements = size_t(width) * dim * dim;
        const size_t inputBytes = samples * (sizeof(float2) + 4 * sizeof(float)) + volumeElements * 2 * sizeof(float);
        if (inputBytes > size_t(2) * 1024 * 1024 * 1024 || modes > 65536)
            throw std::runtime_error("Benchmark is bounded to 2GiB inputs and65536 hypotheses");
        Check(cudaSetDevice(0));
        cudaDeviceProp properties = {};
        Check(cudaGetDeviceProperties(&properties, 0));
        std::printf("device=%s optimizer=%s box=%d views=%d particles=%d hypotheses=%d iterations=%d inputs_MiB=%.1f\n",
            properties.name, optimizerName, box, views, particles, hypotheses, iterations, inputBytes / 1048576.0);
        std::fflush(stdout);

        std::vector<float> volumeRe(volumeElements), volumeIm(volumeElements);
        const double envelopeScale = 0.5 * box;
        for (int z = 0; z < dim; ++z)
            for (int y = 0; y < dim; ++y)
                for (int x = 0; x < width; ++x)
                {
                    const int ky = y <= dim / 2 ? y : y - dim, kz = z <= dim / 2 ? z : z - dim;
                    const double envelope = std::exp(-(double(x) * x + double(ky) * ky + double(kz) * kz) / (envelopeScale * envelopeScale));
                    const size_t id = (size_t(z) * dim + y) * width + x;
                    volumeRe[id] = float(envelope * (1 + .25 * std::cos(.3 * x) + .2 * std::cos(.23 * ky) + .1 * std::cos(.17 * kz)));
                    volumeIm[id] = float(envelope * (.2 * std::sin(.2 * x) + .15 * std::sin(.3 * kz) + .1 * std::sin(.2 * ky)));
                }
        Texture textureRe(volumeRe, width, dim), textureIm(volumeIm, width, dim);
        const HostFetch fetch = {volumeRe.data(), volumeIm.data(), width, dim};
        std::vector<float2> data(samples);
        std::vector<float> ctf(samples), quad(samples), noise(samples), radii(samples);
        std::vector<float> geometry(size_t(particles) * views * 18), poses(modes * 12), bounds(size_t(particles) * 6);
        std::vector<int> seeds(modes);
        const float pixel = 2.0f, cutoff = float(box / 2 - 1), diameter = 90.0f;
        const float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        for (int p = 0; p < particles; ++p)
        {
            float truth[12] = {float(.25 * std::sin(p + .3)), float(.2 * std::cos(p + .5)), float(.15 * std::sin(p + .8))};
            const double rotation[3] = {.15 + .007 * p, -.23 + .011 * p, .18 - .006 * p};
            RotateRight(identity, rotation, truth + 3);
            for (int axis = 0; axis < 3; ++axis) { bounds[p * 6 + axis] = -8; bounds[p * 6 + axis + 3] = 8; }
            for (int h = 0; h < hypotheses; ++h)
            {
                float* pose = poses.data() + (size_t(p) * hypotheses + h) * 12;
                for (int axis = 0; axis < 3; ++axis) pose[axis] = truth[axis] + float(.9 * std::sin(.37 * h + axis + .13 * p));
                const double perturbation[3] = {.075 * std::sin(.43 * h + .1), .07 * std::cos(.39 * h + .2), .06 * std::sin(.31 * h + .8)};
                RotateRight(truth + 3, perturbation, pose + 3);
                seeds[size_t(p) * hypotheses + h] = h;
            }
            for (int t = 0; t < views; ++t)
            {
                const double tilt = views == 1 ? 0 : (-60.0 + 120.0 * t / (views - 1)) * Pi / 180;
                const double tiltOmega[3] = {.012 * std::sin(t + .1), tilt, .006 * p};
                float tiltRotation[9];
                RotateRight(identity, tiltOmega, tiltRotation);
                float* g = geometry.data() + (size_t(p) * views + t) * 18;
                for (int column = 0; column < 3; ++column)
                    for (int row = 0; row < 3; ++row)
                        g[row + column * 3] = 2 * tiltRotation[column + row * 3] * (column == 0 ? 1.01f : column == 1 ? .99f : 1.0f);
                for (int axis = 0; axis < 3; ++axis)
                {
                    g[9 + axis * 2] = tiltRotation[axis * 3] / pixel;
                    g[10 + axis * 2] = tiltRotation[axis * 3 + 1] / pixel;
                    g[15 + axis] = .08f * tiltRotation[axis * 3 + 2];
                }
                for (int row = 0; row < box; ++row)
                    for (int x = 0; x <= box / 2; ++x)
                    {
                        const int y = row <= box / 2 ? row : row - box;
                        const size_t id = (size_t(p) * views + t) * frequencies + size_t(row) * (box / 2 + 1) + x;
                        const double radius2 = (1.02 * x * x + .98 * y * y + .03 * x * y) / (double(box) * box * pixel * pixel);
                        const double phase = (220.0 + 2 * t + p) * radius2;
                        ctf[id] = float(.9 * std::sin(phase)); quad[id] = float(-.9 * std::cos(phase));
                        radii[id] = float(radius2); noise[id] = float(.8 + .2 * std::cos(2 * tilt) + .05 * std::sin(.13 * x + .09 * y + .1 * p));
                        const Complex<float> value = Model(fetch, dim, box, x, y, g, truth, ctf[id], quad[id], radii[id]);
                        const double amplitude = 1 + .025 * p;
                        data[id] = make_float2(float(amplitude * value.re + .012 * std::sin(.37 * x + .29 * y + .17 * t + .3 * p)),
                                              float(amplitude * value.im + .012 * std::cos(.23 * x - .33 * y + .21 * t + .4 * p)));
                    }
            }
        }
        DeviceBuffer dData(data.data(), samples * sizeof(float2)), dCtf(ctf.data(), samples * sizeof(float));
        DeviceBuffer dQuad(quad.data(), samples * sizeof(float)), dNoise(noise.data(), samples * sizeof(float)), dRadii(radii.data(), samples * sizeof(float));
        std::vector<double> summary(modes * 4), tiltStats(modes * views * 2);
        std::vector<int> diagnostics(modes * 4);
        const std::vector<float> initialPoses = poses;
        const std::vector<int> initialSeeds = seeds;
        cudaEvent_t start, end;
        Check(cudaEventCreate(&start)); Check(cudaEventCreate(&end));
        for (int run = 0; run < (scoreOnly ? 1 : 2); ++run)
        {
            poses = initialPoses; seeds = initialSeeds;
            Check(cudaDeviceSynchronize());
            const auto begin = std::chrono::steady_clock::now();
            Check(cudaEventRecord(start));
            const int status = refine(textureRe.texture, textureIm.texture,
                dim, box, views, particles, hypotheses, static_cast<const float2*>(dData.pointer),
                static_cast<const float*>(dCtf.pointer), static_cast<const float*>(dQuad.pointer),
                static_cast<const float*>(dNoise.pointer), static_cast<const float*>(dRadii.pointer),
                geometry.data(), bounds.data(), identity, 1, poses.data(), seeds.data(),
                pixel, cutoff, diameter, run == 0 ? 0 : iterations, 0, 0,
                summary.data(), diagnostics.data(), tiltStats.data());
            if (status) throw std::runtime_error(std::string(optimizerName) + " batch refinement returned " + std::to_string(status));
            Check(cudaEventRecord(end)); Check(cudaEventSynchronize(end));
            const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            float eventMs = 0; Check(cudaEventElapsedTime(&eventMs, start, end));
            long long accepted = 0, evaluations = 0;
            size_t valid = 0, invalid = 0, worse = 0;
            int maximumAccepted = 0;
            double improvement = 0, initial = 0, final = 0;
            for (size_t mode = 0; mode < modes; ++mode)
            {
                accepted += diagnostics[mode * 4]; evaluations += diagnostics[mode * 4 + 1];
                maximumAccepted = std::max(maximumAccepted, diagnostics[mode * 4]);
                if (seeds[mode] < 0) { ++invalid; continue; }
                const double z = summary[mode * 4] / std::sqrt(summary[mode * 4 + 1]);
                if (!std::isfinite(z)) throw std::runtime_error("Nonfinite retained score");
                const double z0 = summary[mode * 4 + 2];
                // BFGS stores initial Z rounded to float, while diagnostics are
                // reconstructed from promoted C/P here; allow that last rounding.
                const double scoreTolerance = 8.0 * FLT_EPSILON;
                if (z < z0 - scoreTolerance * std::max(1.0, std::fabs(z0))) ++worse;
                initial += z0; final += z; improvement += z - z0; ++valid;
            }
            std::printf("mode=%s optimizer=%s api_ms=%.3f cuda_interval_ms=%.3f valid=%zu invalid=%zu worse=%zu accepted=%lld max_accepted=%d evaluations=%lld mean_initial_Z=%.6f mean_final_Z=%.6f mean_delta_Z=%.6f\n",
                run == 0 ? "initial_score" : "refine", optimizerName, wallMs, eventMs, valid, invalid, worse, accepted,
                maximumAccepted, evaluations, initial / std::max(size_t(1), valid), final / std::max(size_t(1), valid), improvement / std::max(size_t(1), valid));
            std::fflush(stdout);
            if (invalid || worse) throw std::runtime_error("Invalid or worsened hypotheses");
        }
        Check(cudaEventDestroy(start)); Check(cudaEventDestroy(end));

        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "benchmark failed: %s\n", error.what());
        return 1;
    }
}
