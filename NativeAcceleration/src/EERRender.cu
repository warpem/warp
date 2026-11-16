#include "include/Functions.h"

using namespace gtom;

// EER frames rendered as a type 1 non-uniform FFT. Rendering electrons at 16K and Fourier cropping that image to the
// output size keeps their sub-pixel positions: the output is each electron's band-limited (Dirichlet) kernel, sampled
// at the output pixels. The same coefficients come from spreading the electrons onto a grid twice the output size with
// a compact exponential-of-semicircle kernel, transforming that grid, keeping the output's frequencies and dividing the
// kernel's transform out (Barnett et al., FINUFFT). Width 4 is accurate to ~2e-3 of render-at-16K-then-crop, with a
// grid a sixteenth of the 16K one or smaller; width 1 on the 16K grid itself is that render, exactly.
//
// A renderer holds the grid, the transforms and two device buffers of electron positions. Batches of frames are
// uploaded on a copy stream into one buffer while the default stream spreads the other, and a batch's frames are
// spread together, interleaved in chunks, so that the threads running at any time work on the same rows of every
// frame: the grid is swept once per batch rather than once per frame, which matters once it outgrows the L2 cache.
// The grid accumulates in 32-bit fixed point: integer atomics add up the same in any order, so a movie renders to the
// same floats every time (float atomics did not, and a CTF fit can turn that noise into a different defocus). The
// scale is a power of two below a strict bound on any node's sum, given by the caller.

namespace
{
	const int SuperSize = 16384, SuperShift = 14;
	const int MaxSlots = 64, ChunkElectrons = 32;
	const double Pi = 3.14159265358979323846;

	// FINUFFT's shape parameter for an upsampling factor of 2.
	inline float KernelBeta(int width)
	{
		float perWidth = width == 2 ? 2.20f : width == 3 ? 2.26f : width == 4 ? 2.38f : 2.30f;
		return perWidth * width;
	}

	__device__ inline float Kernel(float distance, float beta, int width)
	{
		float z = 2.f * distance / width;
		return __expf(beta * (sqrtf(fmaxf(0.f, 1.f - z * z)) - 1.f));
	}

	__device__ inline int Wrap(int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); }

	struct SlotCounts
	{
		long long counts[MaxSlots];
		long long chunks; // per slot, enough for the fullest one
		int slots;
	};

	// Electron positions are (y << 14) | x on the 16K grid, slot s's at positions + s * slotStride; the gain is
	// gainSize square (4K, 8K or 16K) and multiplies each electron by its pixel's value. On the grid, an electron sits
	// at position * gridSize / 16K. Chunks of the slots alternate, so a warp's electrons share a slot. L lanes share an
	// electron, lane a adding its kernel's column a: each row's taps are then one contiguous write from the warp.
	template<int W, int L> __global__ void SpreadKernel(const unsigned int* positions, long long slotStride, SlotCounts counts, const float* gain, int gainShift, int gainSize, int* grid, int n, float beta, float scale)
	{
		const int a = threadIdx.x % L;
		const long long total = counts.chunks * counts.slots * ChunkElectrons;
		const long long step = (long long)gridDim.x * blockDim.x / L;
		for (long long v = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / L; v < total; v += step)
		{
			long long chunk = v / ChunkElectrons;
			int slot = (int)(chunk % counts.slots);
			long long i = chunk / counts.slots * ChunkElectrons + v % ChunkElectrons;
			if (i >= counts.counts[slot])
				continue;

			unsigned int p = positions[slot * slotStride + i];
			int x = p & (SuperSize - 1), y = p >> SuperShift;
			float weight = (gain != NULL ? gain[(size_t)(y >> gainShift) * gainSize + (x >> gainShift)] : 1.f) * scale;

			if (W == 1)
			{
				atomicAdd(grid + (size_t)y * n + x, __float2int_rn(weight));
				continue;
			}
			if (a >= W)
				continue;

			// Grid coordinates: whole cell and fraction; the kernel's first node is ceil(fraction - W/2) cells away.
			int gx = x * n, gy = y * n; // < 2^28: exact
			float fx = (gx & (SuperSize - 1)) * (1.f / SuperSize), fy = (gy & (SuperSize - 1)) * (1.f / SuperSize);
			int firstx = (int)ceilf(fx - W * 0.5f), firsty = (int)ceilf(fy - W * 0.5f);
			float wx = weight * Kernel((firstx + a) - fx, beta, W);
			int column = Wrap(firstx + a + (gx >> SuperShift), n), row = firsty + (gy >> SuperShift);

			#pragma unroll
			for (int b = 0; b < W; b++)
				atomicAdd(grid + (size_t)Wrap(row + b, n) * n + column, __float2int_rn(Kernel((firsty + b) - fy, beta, W) * wx));
		}
	}

	// The fixed-point sums as floats, in place.
	__global__ void ToFloatKernel(int* grid, size_t elements, float inverseScale)
	{
		for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < elements; i += (size_t)gridDim.x * blockDim.x)
			((float*)grid)[i] = grid[i] * inverseScale;
	}

	// The output's frequencies from the grid's transform, as d_FFTCrop keeps them (row N/2 is +N/2), times the
	// correction, which divides the kernel's transform and the 16K transform's size out.
	__global__ void CropKernel(const float2* gridft, int n, const float* correction, float2* outputft, int N)
	{
		int y = blockIdx.x;
		int ky = y <= N / 2 ? y : y - N;
		int source = ky >= 0 ? ky : ky + n;
		float cy = correction[ky >= 0 ? ky : -ky];
		for (int x = threadIdx.x; x <= N / 2; x += blockDim.x)
		{
			float2 v = gridft[(size_t)source * (n / 2 + 1) + x];
			float c = cy * correction[x];
			outputft[(size_t)y * (N / 2 + 1) + x] = make_float2(v.x * c, v.y * c);
		}
	}

	// 1 / (kernel transform at k * 16K) for k = 0..outputSize/2. The transform, (W/2) ∫ φ(t) cos(π k W t / n) dt over
	// [-1, 1] with φ(t) = exp(β (sqrt(1 - t²) - 1)), is W times the integral over [0, 1], taken with t = sin θ, which
	// makes the integrand smooth.
	std::vector<float> KernelCorrection(int gridSize, int outputSize, int kernelWidth)
	{
		int K = outputSize / 2;
		std::vector<double> transform(K + 1, kernelWidth > 1 ? 0.0 : 1.0);
		if (kernelWidth > 1)
		{
			// Simpson over θ for every k at once: cos(k α) for the node's α by the recurrence
			// cos((k + 1) α) = 2 cos α cos(k α) - cos((k - 1) α), with no trigonometry per k.
			const int M = 512;
			double beta = KernelBeta(kernelWidth);
			for (int i = 0; i <= M; i++)
			{
				double theta = (double)i / M * (Pi / 2), t = sin(theta);
				double f = exp(beta * (cos(theta) - 1)) * cos(theta) * (i == 0 || i == M ? 1 : (i % 2 == 1 ? 4 : 2));
				double c1 = cos(Pi * kernelWidth * t / gridSize), previous = 1, current = c1;
				transform[0] += f;
				for (int k = 1; k <= K; k++)
				{
					transform[k] += f * current;
					double next = 2 * c1 * current - previous;
					previous = current;
					current = next;
				}
			}
			for (int k = 0; k <= K; k++)
				transform[k] *= kernelWidth * (Pi / 2) / M / 3;
		}
		std::vector<float> correction(K + 1);
		for (int k = 0; k <= K; k++)
			correction[k] = (float)(1.0 / (transform[k] * SuperSize));
		return correction;
	}

	struct Renderer
	{
		int outputSize, gridSize, kernelWidth, slots;
		long long slotCapacity;
		float scale;
		int* grid; // fixed point while spreading, floats once a group is finished
		float2* gridft;
		float2* outputft;
		float* correction;
		unsigned int* positions[2];
		long long counts[2][MaxSlots];
		int current;
		cufftHandle forward, back;
		cudaStream_t copy;
		cudaEvent_t uploaded[2], consumed[2];
	};
}

// A renderer for outputSize² groups from a gridSize² grid (16K when kernelWidth is 1), taking batches of up to
// slots frames of up to slotCapacity electrons, with fixed-point sums of weights times scale (a power of two, so that
// whole counts stay exact). Returns NULL if the arguments are out of range or memory is short.
__declspec(dllexport) void* EERCreateRenderer(int outputSize, int gridSize, int kernelWidth, int slots, long long slotCapacity, float scale)
{
	if (outputSize < 2 || outputSize % 2 != 0 || gridSize < outputSize || gridSize > SuperSize || slots < 1 || slots > MaxSlots || slotCapacity < 1 || !(scale >= 1) ||
		!(kernelWidth == 1 ? gridSize == SuperSize : kernelWidth >= 4 && kernelWidth <= 8))
		return NULL;

	Renderer* r = new Renderer();
	r->outputSize = outputSize; r->gridSize = gridSize; r->kernelWidth = kernelWidth; r->slots = slots; r->slotCapacity = slotCapacity; r->scale = scale;
	r->current = 1;

	bool ok = cudaMalloc((void**)&r->grid, (size_t)gridSize * gridSize * sizeof(int)) == cudaSuccess &&
			  cudaMalloc((void**)&r->gridft, (size_t)(gridSize / 2 + 1) * gridSize * sizeof(float2)) == cudaSuccess &&
			  cudaMalloc((void**)&r->outputft, (size_t)(outputSize / 2 + 1) * outputSize * sizeof(float2)) == cudaSuccess &&
			  cudaMalloc((void**)&r->correction, (outputSize / 2 + 1) * sizeof(float)) == cudaSuccess &&
			  cudaMalloc((void**)&r->positions[0], (size_t)slots * slotCapacity * sizeof(unsigned int)) == cudaSuccess &&
			  cudaMalloc((void**)&r->positions[1], (size_t)slots * slotCapacity * sizeof(unsigned int)) == cudaSuccess;
	if (!ok)
	{
		cudaGetLastError();
		cudaFree(r->grid); cudaFree(r->gridft); cudaFree(r->outputft); cudaFree(r->correction); cudaFree(r->positions[0]); cudaFree(r->positions[1]);
		delete r;
		return NULL;
	}

	cudaMemset(r->grid, 0, (size_t)gridSize * gridSize * sizeof(int));
	std::vector<float> correction = KernelCorrection(gridSize, outputSize, kernelWidth);
	cudaMemcpy(r->correction, correction.data(), correction.size() * sizeof(float), cudaMemcpyHostToDevice);
	memset(r->counts, 0, sizeof(r->counts));

	r->forward = d_FFTR2CGetPlan(2, toInt3(gridSize, gridSize, 1), 1);
	r->back = d_IFFTC2RGetPlan(2, toInt3(outputSize, outputSize, 1), 1);

	// Not blocking: the legacy default stream would otherwise wait for every copy, and they for it.
	cudaStreamCreateWithFlags(&r->copy, cudaStreamNonBlocking);
	for (int b = 0; b < 2; b++)
	{
		cudaEventCreateWithFlags(&r->uploaded[b], cudaEventDisableTiming);
		cudaEventCreateWithFlags(&r->consumed[b], cudaEventDisableTiming);
		cudaEventRecord(r->consumed[b], 0);
	}
	return r;
}

// Copies a batch's frames from pinned host memory (slot s at h_positions + s * slotCapacity, h_counts[s] electrons)
// into the next device buffer, once the spreading of the batch before last there is done. The host memory must stay
// as it is until EERRendererWaitForUploads returns.
__declspec(dllexport) void EERRendererUpload(void* handle, const unsigned int* h_positions, const long long* h_counts, int nslots)
{
	Renderer* r = (Renderer*)handle;
	int b = r->current ^ 1;
	r->current = b;
	cudaStreamWaitEvent(r->copy, r->consumed[b], 0);
	for (int s = 0; s < r->slots; s++)
	{
		long long n = s < nslots ? h_counts[s] : 0;
		if (n < 0 || n > r->slotCapacity)
			n = 0;
		r->counts[b][s] = n;
		if (n > 0)
			cudaMemcpyAsync(r->positions[b] + s * r->slotCapacity, h_positions + s * r->slotCapacity, n * sizeof(unsigned int), cudaMemcpyHostToDevice, r->copy);
	}
	cudaEventRecord(r->uploaded[b], r->copy);
	cudaStreamWaitEvent(0, r->uploaded[b], 0);
}

// Adds slots first..first+nslots of the uploaded batch to the current group, each electron times its gain pixel.
__declspec(dllexport) void EERRendererSpread(void* handle, int firstSlot, int nslots, const float* d_gain, int gainSize)
{
	Renderer* r = (Renderer*)handle;
	int b = r->current;
	if (firstSlot < 0 || nslots < 1 || firstSlot + nslots > r->slots)
		return;

	SlotCounts counts;
	long long most = 0;
	counts.slots = nslots;
	for (int s = 0; s < nslots; s++)
	{
		counts.counts[s] = r->counts[b][firstSlot + s];
		most = counts.counts[s] > most ? counts.counts[s] : most;
	}
	if (most == 0)
		return;
	counts.chunks = (most + ChunkElectrons - 1) / ChunkElectrons;

	int gainShift = 0;
	if (d_gain != NULL)
		while ((gainSize << gainShift) < SuperSize)
			gainShift++;

	const int threads = 256;
	int lanes = r->kernelWidth == 1 ? 1 : r->kernelWidth <= 4 ? 4 : 8;
	long long needed = (counts.chunks * nslots * ChunkElectrons * lanes + threads - 1) / threads;
	unsigned int blocks = (unsigned int)(needed < 65536 ? needed : 65536);
	float beta = KernelBeta(r->kernelWidth);
	const unsigned int* positions = r->positions[b] + firstSlot * r->slotCapacity;
	switch (r->kernelWidth)
	{
	case 1: SpreadKernel<1, 1><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	case 4: SpreadKernel<4, 4><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	case 5: SpreadKernel<5, 8><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	case 6: SpreadKernel<6, 8><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	case 7: SpreadKernel<7, 8><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	case 8: SpreadKernel<8, 8><<<blocks, threads>>>(positions, r->slotCapacity, counts, d_gain, gainShift, gainSize, r->grid, r->gridSize, beta, r->scale); break;
	}
}

// Marks the uploaded batch's buffer free for the upload after next, once what was queued to spread it has run.
__declspec(dllexport) void EERRendererBatchDone(void* handle)
{
	Renderer* r = (Renderer*)handle;
	cudaEventRecord(r->consumed[r->current], 0);
}

// Waits until every upload so far has left host memory.
__declspec(dllexport) void EERRendererWaitForUploads(void* handle)
{
	cudaStreamSynchronize(((Renderer*)handle)->copy);
}

// Transforms the grid, crops and corrects its transform into the output's, transforms that back into d_output
// (outputSize² floats on the device), and clears the grid for the next group.
__declspec(dllexport) void EERRendererFinish(void* handle, float* d_output)
{
	Renderer* r = (Renderer*)handle;
	size_t elements = (size_t)r->gridSize * r->gridSize;
	ToFloatKernel<<<(unsigned int)((elements + 255) / 256 < 65536 ? (elements + 255) / 256 : 65536), 256>>>(r->grid, elements, 1.f / r->scale);
	d_FFTR2C((float*)r->grid, r->gridft, &r->forward);
	CropKernel<<<r->outputSize, min(256, NextMultipleOf(r->outputSize / 2 + 1, 32))>>>(r->gridft, r->gridSize, r->correction, r->outputft, r->outputSize);
	d_IFFTC2R(r->outputft, d_output, &r->back);
	cudaMemsetAsync(r->grid, 0, elements * sizeof(int), 0);
}

__declspec(dllexport) void EERDestroyRenderer(void* handle)
{
	Renderer* r = (Renderer*)handle;
	if (r == NULL)
		return;
	cudaStreamSynchronize(r->copy);
	cudaStreamSynchronize(0);
	cufftDestroy(r->forward);
	cufftDestroy(r->back);
	cudaFree(r->grid); cudaFree(r->gridft); cudaFree(r->outputft); cudaFree(r->correction); cudaFree(r->positions[0]); cudaFree(r->positions[1]);
	for (int b = 0; b < 2; b++)
	{
		cudaEventDestroy(r->uploaded[b]);
		cudaEventDestroy(r->consumed[b]);
	}
	cudaStreamDestroy(r->copy);
	delete r;
}
