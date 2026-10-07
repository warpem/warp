// Included inside TemplateMatchRefineBatch.cu's anonymous namespace, sharing the
// frozen-geometry forward model. All BFGS arithmetic is
// FP32; conversion to double occurs only at the existing diagnostic ABI.
namespace bfgs = warp_template_match_bfgs;
constexpr int BfgsStatistics = 14;

__device__ inline void CompensatedAdd(float value, float& sum, float& correction)
{
    // Explicit rounding prevents contraction/reassociation from destroying Kahan
    // compensation, including builds that enable CUDA fast math.
    const float adjusted = __fsub_rn(value, correction);
    const float total = __fadd_rn(sum, adjusted);
    correction = __fsub_rn(__fsub_rn(total, sum), adjusted);
    sum = total;
}

struct BfgsWorkspace
{
    float current[12], trial[12], basis[9];
    float q[6], trialQ[6], gradient[6], oldGradient[6];
    float direction[6], displacement[6], difference[6], omega[3];
    float inverse[36], updateWork[bfgs::UpdateWorkspace];
    float stats[BfgsStatistics], trialStats[2], reduction[4][BfgsStatistics];
    float z, initialZ, alpha, slope, rawTranslation, rawRotation;
    int active, terminate, termination, accepted, evaluations, acceptedTrial, capped;
};

template <bool Derivatives>
__device__ __noinline__ void EvaluateBfgs(const BatchContext& c, int particle,
    const float* pose, BfgsWorkspace& w, float* result)
{
    constexpr int Count = Derivatives ? BfgsStatistics : 2;
    float sums[Count] = {}, correction[Count] = {};
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
        if (!bfgs::IndependentSample(x, y, c.box, c.cutoffSquaredLimit)) continue;
        const size_t sampleIndex = particleOffset + index;
        const float weight = c.noise[sampleIndex];
        if (weight == 0) continue;
        if (!(weight > 0) || !isfinite(weight)) { sums[0] = CUDART_NAN_F; continue; }
        Complex<float> model, derivatives[Derivatives ? 6 : 1];
        Model<Derivatives>(c, c.geometry + (size_t(particle) * c.views + view) * 18,
            pose, sampleIndex, x, y, model, derivatives);
        const float2 observation = c.data[sampleIndex];
        const Complex<float> observed(observation.x, observation.y);
        CompensatedAdd(weight * Inner(observed, model), sums[0], correction[0]);
        CompensatedAdd(weight * Inner(model, model), sums[1], correction[1]);
        if (Derivatives)
            for (int parameter = 0; parameter < 6; ++parameter)
            {
                CompensatedAdd(weight * Inner(observed, derivatives[parameter]),
                    sums[2 + parameter], correction[2 + parameter]);
                CompensatedAdd(weight * Inner(model, derivatives[parameter]),
                    sums[8 + parameter], correction[8 + parameter]);
            }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (int statistic = 0; statistic < Count; ++statistic)
    {
        float value = sums[statistic];
        for (int offset = 16; offset > 0; offset /= 2)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) w.reduction[warp][statistic] = value;
    }
    __syncthreads();
    if (threadIdx.x < Count)
        result[threadIdx.x] = (w.reduction[0][threadIdx.x] + w.reduction[1][threadIdx.x]) +
                             (w.reduction[2][threadIdx.x] + w.reduction[3][threadIdx.x]);
    __syncthreads();
}

__device__ float Norm3f(const float* v)
{
    return sqrtf(fmaf(v[0], v[0], fmaf(v[1], v[1], v[2] * v[2])));
}

__device__ bool BfgsGradient(const BatchContext& c, BfgsWorkspace& w)
{
    if (!(w.stats[1] > 0) || !isfinite(w.stats[0]) || !isfinite(w.stats[1])) return false;
    const float root = sqrtf(w.stats[1]);
    w.z = w.stats[0] / root;
    const float amplitude = w.stats[0] / w.stats[1];
    for (int j = 0; j < 6; ++j)
        w.difference[j] = -fmaf(-amplitude, w.stats[8 + j], w.stats[2 + j]) / root;
    for (int j = 0; j < 3; ++j) w.omega[j] = w.q[3 + j] * (2.0f * c.pixel / c.diameter);
    bfgs::ScaledGradient(w.difference, w.omega, c.pixel, 0.5f * c.diameter, w.gradient);
    if (!isfinite(w.z)) return false;
    for (int j = 0; j < 6; ++j) if (!isfinite(w.gradient[j])) return false;
    return true;
}

__device__ void ResetBfgs(BfgsWorkspace& w)
{
    float norm2 = 0;
    for (int i = 0; i < 6; ++i) norm2 = fmaf(w.gradient[i], w.gradient[i], norm2);
    // Start with an approximately one-pixel displacement in the scaled metric.
    bfgs::ResetInverse(w.inverse, 1.0f / fmaxf(sqrtf(norm2), 1.0e-4f));
}

__device__ bool BfgsDirection(const BatchContext& c, int particle, BfgsWorkspace& w, bool fallback)
{
    for (int i = 0; i < 6; ++i) w.difference[i] = w.gradient[i];
    for (int i = 0; i < 3; ++i)
        if ((w.current[i] <= c.bounds[particle * 6 + i] && w.difference[i] > 0) ||
            (w.current[i] >= c.bounds[particle * 6 + 3 + i] && w.difference[i] < 0)) w.difference[i] = 0;
    if (fallback) ResetBfgs(w);
    if (!bfgs::Direction(w.inverse, w.difference, w.direction)) return false;
    // Projection also removes outward directions introduced by Hessian coupling.
    for (int i = 0; i < 3; ++i)
        if ((w.current[i] <= c.bounds[particle * 6 + i] && w.direction[i] < 0) ||
            (w.current[i] >= c.bounds[particle * 6 + 3 + i] && w.direction[i] > 0)) w.direction[i] = 0;
    w.rawTranslation = Norm3f(w.direction) * c.pixel;
    w.rawRotation = Norm3f(w.direction + 3) * (2.0f * c.pixel / c.diameter);
    // A uniform cap preserves the direction. The exponential map's differential
    // has norm <=1, so limiting chart displacement also limits geodesic rotation.
    const float factor = fminf(1.0f, fminf(2.0f * c.pixel / fmaxf(w.rawTranslation, FLT_MIN),
        (5.0f * 3.14159265358979323846f / 180.0f) / fmaxf(w.rawRotation, FLT_MIN)));
    w.capped = factor < 1.0f;
    float slope = 0;
    for (int i = 0; i < 6; ++i)
    {
        w.direction[i] *= factor;
        slope = fmaf(w.gradient[i], w.direction[i], slope);
    }
    return isfinite(slope) && slope < 0;
}

__global__ void OptimizeBfgs(BatchContext c)
{
    const int hypothesis = int(blockIdx.x), particle = hypothesis / c.hypotheses;
    __shared__ BfgsWorkspace w;
    if (threadIdx.x == 0)
    {
        w.active = c.seedIds[hypothesis] >= 0;
        w.accepted = w.evaluations = 0;
        w.terminate = !w.active;
        w.termination = w.active ? 0 : 3;
        w.z = w.initialZ = w.stats[0] = w.stats[1] = 0;
        for (int i = 0; i < 12; ++i) w.current[i] = c.poses[size_t(hypothesis) * 12 + i];
        for (int i = 0; i < 9; ++i) w.basis[i] = w.current[3 + i];
        for (int i = 0; i < 6; ++i) w.q[i] = i < 3 ? w.current[i] / c.pixel : 0;
        if (w.active)
        {
            bool valid = bfgs::ProperRotation(w.basis);
            for (int i = 0; i < 3; ++i)
                valid = valid && isfinite(w.current[i]) &&
                    w.current[i] >= c.bounds[particle * 6 + i] && w.current[i] <= c.bounds[particle * 6 + 3 + i];
            if (!valid) { w.terminate = 1; w.termination = 4; }
        }
    }
    __syncthreads();
    if (w.active && !w.terminate)
    {
        EvaluateBfgs<true>(c, particle, w.current, w, w.stats);
        if (threadIdx.x == 0)
        {
            ++w.evaluations;
            if (!BfgsGradient(c, w)) { w.termination = 4; w.terminate = 1; }
            else ResetBfgs(w);
            w.initialZ = w.z;
            if (c.maxIterations == 0) w.terminate = 1;
        }
        __syncthreads();
    }
    while (!w.terminate)
    {
        if (threadIdx.x == 0) w.acceptedTrial = 0;
        __syncthreads();
        // A failed quasi-Newton direction gets one fresh projected-gradient
        // attempt. Both searches stay entirely inside this hypothesis's block.
        for (int attempt = 0; attempt < 2 && !w.acceptedTrial; ++attempt)
        {
            if (threadIdx.x == 0)
            {
                w.alpha = BfgsDirection(c, particle, w, attempt != 0) ? 1.0f : 0;
            }
            __syncthreads();
            for (int trial = 0; trial < 16 && w.alpha > 0 && !w.acceptedTrial; ++trial)
            {
                if (threadIdx.x == 0)
                {
                    w.slope = 0;
                    for (int i = 0; i < 6; ++i)
                    {
                        w.trialQ[i] = fmaf(w.alpha, w.direction[i], w.q[i]);
                        if (i < 3)
                        {
                            w.trial[i] = fminf(c.bounds[particle * 6 + 3 + i],
                                fmaxf(c.bounds[particle * 6 + i], w.trialQ[i] * c.pixel));
                            w.trialQ[i] = w.trial[i] / c.pixel;
                        }
                        w.displacement[i] = w.trialQ[i] - w.q[i];
                        w.slope = fmaf(w.gradient[i], w.displacement[i], w.slope);
                    }
                    for (int i = 0; i < 3; ++i) w.omega[i] = w.trialQ[3 + i] * (2.0f * c.pixel / c.diameter);
                    bfgs::RotateRight(w.basis, w.omega, w.trial + 3);
                }
                __syncthreads();
                // Clipping can remove the descending part of a coupled BFGS
                // direction; a shorter trial may become descent again.
                if (!(w.slope < 0) || !isfinite(w.slope))
                {
                    if (threadIdx.x == 0) w.alpha *= 0.5f;
                    __syncthreads();
                    continue;
                }
                EvaluateBfgs<false>(c, particle, w.trial, w, w.trialStats);
                if (threadIdx.x == 0)
                {
                    ++w.evaluations;
                    const float trialZ = w.trialStats[0] / sqrtf(w.trialStats[1]);
                    const float improvement = trialZ - w.z;
                    const float tolerance = 8.0f * FLT_EPSILON * fmaxf(1.0f, fabsf(w.z));
                    if (w.trialStats[1] > 0 && isfinite(w.trialStats[1]) && isfinite(trialZ) &&
                        improvement > tolerance && improvement >= -1.0e-4f * w.slope)
                        w.acceptedTrial = 1;
                    else w.alpha *= 0.5f;
                }
                __syncthreads();
            }
            // Every lane must consume the completed search's shared loop
            // condition before thread zero resets alpha for the fallback.
            __syncthreads();
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            if (!w.acceptedTrial) { w.terminate = 1; w.termination = 2; }
            else
            {
                ++w.accepted;
                for (int i = 0; i < 12; ++i) w.current[i] = w.trial[i];
                for (int i = 0; i < 6; ++i) { w.oldGradient[i] = w.gradient[i]; w.q[i] = w.trialQ[i]; }
                w.stats[0] = w.trialStats[0]; w.stats[1] = w.trialStats[1];
                w.z = w.stats[0] / sqrtf(w.stats[1]);
                // Backtracking, clipping, or caps can force tiny steps away from
                // a stationary point; do not report those as step convergence.
                if (w.alpha == 1.0f && !w.capped && w.rawTranslation <= 0.005f * c.pixel &&
                    w.rawRotation <= 0.01f * 3.14159265358979323846f / 180.0f)
                { w.terminate = 1; w.termination = 1; }
                else if (w.accepted >= c.maxIterations) { w.terminate = 1; w.termination = 0; }
            }
        }
        __syncthreads();
        if (!w.terminate)
        {
            EvaluateBfgs<true>(c, particle, w.current, w, w.stats);
            if (threadIdx.x == 0)
            {
                ++w.evaluations;
                if (!BfgsGradient(c, w)) { w.terminate = 1; w.termination = 4; }
                else
                {
                    for (int i = 0; i < 6; ++i) w.difference[i] = w.gradient[i] - w.oldGradient[i];
                    bfgs::UpdateInverse(w.inverse, w.displacement, w.difference, w.updateWork);
                    // Stay in a regular local chart without comparing gradients
                    // from different tangent frames. Recentering resets H.
                    if (Norm3f(w.omega) > 0.5f)
                    {
                        for (int i = 0; i < 9; ++i) w.basis[i] = w.current[3 + i];
                        for (int i = 3; i < 6; ++i) w.q[i] = 0;
                        BfgsGradient(c, w);
                        ResetBfgs(w);
                    }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 12; ++i) c.poses[size_t(hypothesis) * 12 + i] = w.current[i];
        const bool valid = w.active && w.termination != 4;
        c.summary[size_t(hypothesis) * 4] = valid ? w.stats[0] : 0;
        c.summary[size_t(hypothesis) * 4 + 1] = valid ? w.stats[1] : 0;
        c.summary[size_t(hypothesis) * 4 + 2] = valid ? w.initialZ : 0;
        c.summary[size_t(hypothesis) * 4 + 3] = 0; // BFGS has no trust radius.
        c.diagnostics[size_t(hypothesis) * 4] = w.accepted;
        c.diagnostics[size_t(hypothesis) * 4 + 1] = w.evaluations;
        c.diagnostics[size_t(hypothesis) * 4 + 2] = w.termination;
        c.diagnostics[size_t(hypothesis) * 4 + 3] = valid ? -1 : -2;
        if (!valid) c.seedIds[hypothesis] = -1;
    }
}

__device__ bool CloseRotationBfgs(const float* first, const float* candidate, const float* symmetry, float threshold)
{
    // ||A-B||_F^2 = 8 sin^2(theta/2): unlike acos((trace-1)/2),
    // this resolves the very small merge angles without subtracting from one.
    float difference2 = 0;
    for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
        {
            float equivalent = 0;
            for (int k = 0; k < 3; ++k) equivalent = fmaf(candidate[row + k * 3], symmetry[k + column * 3], equivalent);
            const float difference = first[row + column * 3] - equivalent;
            difference2 = fmaf(difference, difference, difference2);
        }
    return difference2 < threshold;
}
