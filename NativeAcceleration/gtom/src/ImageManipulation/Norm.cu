#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"


namespace gtom
{
#define MonoTpB 192

	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
	__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple);
	__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
	__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf);

	template<bool outputmu> __global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements);
	template<bool outputmu> __global__ void NormMeanStdDevMonoMaskedKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, tfloat* d_mask, size_t elements);

	template<bool outputmu> __global__ void NormMeanStdDevWarpMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, uchar elements, size_t n);

	template<bool flipsign> __global__ void NormBackgroundMonoKernel(tfloat* d_input, tfloat* d_output, int3 dims, uint particleradius2);

	__global__ void Mean0MonoKernel(tfloat* d_input, tfloat* d_output, size_t elements);

	__global__ void NormFTMonoKernel(tcomplex* d_input, tcomplex* d_output, size_t elements);


	///////////////////////////////////////
	//Equivalent of TOM's tom_norm method//
	///////////////////////////////////////

	template <class Tmask> void d_Norm(tfloat* d_input, tfloat* d_output, size_t elements, Tmask* d_mask, T_NORM_MODE mode, tfloat scf, int batch)
	{
		imgstats5* d_imagestats;
		cudaMalloc((void**)&d_imagestats, batch * sizeof(imgstats5));
		d_Dev(d_input, d_imagestats, elements, d_mask, batch);

		imgstats5* h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));

		size_t TpB = tmin((size_t)192, NextMultipleOf(elements, 32));
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks, batch);

		if (mode == T_NORM_PHASE)
			NormPhaseKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements);
		else if (mode == T_NORM_STD1)
			NormStdDevKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements, (tfloat)1);
		else if (mode == T_NORM_STD2)
			NormStdDevKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements, (tfloat)2);
		else if (mode == T_NORM_STD3)
			NormStdDevKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements, (tfloat)3);
		else if (mode == T_NORM_MEAN01STD)
			NormMeanStdDevKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements);
		else if (mode == T_NORM_OSCAR)
		{
			NormStdDevKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements, (tfloat)3);
			d_AddScalar(d_output, d_output, elements * batch, (tfloat)100);
			d_Dev(d_output, d_imagestats, elements, d_mask, batch);
			NormPhaseKernel << <grid, (uint)TpB >> > (d_output, d_output, d_imagestats, elements);
		}
		else if (mode == T_NORM_CUSTOM)
			NormCustomScfKernel << <grid, (uint)TpB >> > (d_input, d_output, d_imagestats, elements, scf);

		free(h_imagestats);
		cudaFree(d_imagestats);
	}
	template void d_Norm<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
	template void d_Norm<int>(tfloat* d_input, tfloat* d_output, size_t elements, int* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
	template void d_Norm<char>(tfloat* d_input, tfloat* d_output, size_t elements, char* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
	template void d_Norm<bool>(tfloat* d_input, tfloat* d_output, size_t elements, bool* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);

	void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, T_NORM_MODE mode, int batch)
	{
		if (elements > 256)
			for (int b = 0; b < batch; b += 32768)
			{
				dim3 grid = dim3(tmin(batch - b, 32768));
				NormMeanStdDevMonoKernel<false> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, NULL, elements);
			}
		else
			for (int b = 0; b < batch; b += 32768)
			{
				dim3 TpB = dim3(32, 6);
				dim3 grid = dim3((tmin(batch - b, 32768) + 5) / 6);
				NormMeanStdDevWarpMonoKernel<false> << <grid, TpB >> > (d_input + elements * b, d_output + elements * b, NULL, elements, min(batch - b, 32768));
			}
	}

	void d_NormMonolithic(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements, T_NORM_MODE mode, int batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(tmin(batch - b, 32768));
			NormMeanStdDevMonoKernel<true> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, d_mu + b, elements);
		}
	}

	void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, int batch)
	{
		if (d_mask != NULL)
			for (int b = 0; b < batch; b += 32768)
			{
				dim3 grid = dim3(tmin(batch - b, 32768));
				NormMeanStdDevMonoMaskedKernel<false> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, NULL, d_mask, elements);
			}
		else
			d_NormMonolithic(d_input, d_output, elements, mode, batch);
	}

	void d_NormMonolithic(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements, tfloat* d_mask, T_NORM_MODE mode, int batch)
	{
		if (d_mask != NULL)
			for (int b = 0; b < batch; b += 32768)
			{
				dim3 grid = dim3(tmin(batch - b, 32768));
				NormMeanStdDevMonoMaskedKernel<true> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, d_mu + b, d_mask, elements);
			}
		else
			d_NormMonolithic(d_input, d_output, d_mu, elements, mode, batch);
	}

	void d_NormBackground(tfloat* d_input, tfloat* d_output, int3 dims, uint particleradius, bool flipsign, uint batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(tmin(batch - b, 32768));
			if (flipsign)
				NormBackgroundMonoKernel<true> << <grid, MonoTpB >> > (d_input + Elements(dims) * b, d_output + Elements(dims) * b, dims, particleradius * particleradius);
			else
				NormBackgroundMonoKernel<false> << <grid, MonoTpB >> > (d_input + Elements(dims) * b, d_output + Elements(dims) * b, dims, particleradius * particleradius);
		}
	}

	void d_Mean0Monolithic(tfloat* d_input, tfloat* d_output, size_t elements, int batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(tmin(batch - b, 32768));
			Mean0MonoKernel << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, elements);
		}
	}

	void d_NormFTMonolithic(tcomplex* d_input, tcomplex* d_output, size_t elements, int batch)
	{
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(tmin(batch - b, 32768));
			NormFTMonoKernel << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, elements);
		}
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements)
	{
		__shared__ tfloat mean;
		if (threadIdx.x == 0)
			mean = d_imagestats[blockIdx.y].mean;
		__syncthreads();

		size_t offset = elements * blockIdx.y;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = (d_input[id + offset] - mean) / (mean + (tfloat)0.000000000001);
	}

	__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple)
	{
		__shared__ tfloat mean, stddev;
		if (threadIdx.x == 0)
			stddev = d_imagestats[blockIdx.y].stddev * stddevmultiple;
		else if (threadIdx.x == 1)
			mean = d_imagestats[blockIdx.y].mean;
		__syncthreads();

		size_t offset = elements * blockIdx.y;
		tfloat upper = mean + stddev, lower = mean - stddev;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = tmax(tmin(d_input[id + offset], upper), lower) - mean;
	}

	__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements)
	{
		__shared__ tfloat mean, stddev;
		if (threadIdx.x == 0)
			stddev = d_imagestats[blockIdx.y].stddev;
		else if (threadIdx.x == 1)
			mean = d_imagestats[blockIdx.y].mean;
		__syncthreads();

		size_t offset = elements * blockIdx.y;
		if (stddev == (tfloat)0)
			for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
				id < elements;
				id += blockDim.x * gridDim.x)
				d_output[id + offset] = d_input[id + offset] - mean;
		else
			for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
				id < elements;
				id += blockDim.x * gridDim.x)
				d_output[id + offset] = (d_input[id + offset] - mean) / stddev;
	}

	__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf)
	{
		__shared__ imgstats5 stats;
		if (threadIdx.x == 0)
			stats = d_imagestats[blockIdx.y];
		__syncthreads();

		size_t offset = elements * blockIdx.y;
		if (stats.stddev != (tfloat)0 && stats.mean != scf)
		{
			tfloat range = stats.max - stats.min;
			for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
				id < elements;
				id += blockDim.x * gridDim.x)
				d_output[id + offset] = scf * (d_input[id + offset] - stats.min) / range;
		}
		else if (stats.stddev == (tfloat)0 && stats.mean != scf)
			for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
				id < elements;
				id += blockDim.x * gridDim.x)
				d_output[id + offset] = d_input[id + offset] / scf;
		else
			for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
				id < elements;
				id += blockDim.x * gridDim.x)
				d_output[id + offset] = d_input[id + offset];
	}

	template<bool outputmu> __global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements)
	{
		__shared__ double s_sums1[MonoTpB];
		__shared__ double s_sums2[MonoTpB];
		__shared__ double s_mean, s_stddev;

		d_input += elements * blockIdx.x;
		d_output += elements * blockIdx.x;

		double sum1 = 0.0, sum2 = 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			double val = d_input[i];
			sum1 += val;
			sum2 += val * val;
		}
		s_sums1[threadIdx.x] = sum1;
		s_sums2[threadIdx.x] = sum2;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < MonoTpB; i++)
			{
				sum1 += s_sums1[i];
				sum2 += s_sums2[i];
			}

			s_mean = sum1 / (double)elements;
			s_stddev = sqrt(tmax(0, ((double)elements * sum2 - (sum1 * sum1)))) / (double)elements;
		}
		__syncthreads();

		tfloat mean = s_mean;
		tfloat stddev = s_stddev > 0.0 ? 1.0 / s_stddev : 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			d_output[i] = (d_input[i] - mean) * stddev;

		if (outputmu && threadIdx.x == 0)
			d_mu[blockIdx.x] = tfloat2(mean, stddev);
	}

	template<bool outputmu> __global__ void NormMeanStdDevMonoMaskedKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, tfloat* d_mask, size_t elements)
	{
		__shared__ double s_sums1[MonoTpB];
		__shared__ double s_sums2[MonoTpB];
		__shared__ double s_samples[MonoTpB];
		__shared__ double s_mean, s_stddev;

		d_input += elements * blockIdx.x;
		d_output += elements * blockIdx.x;

		double sum1 = 0.0, sum2 = 0.0, samples = 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			double val = d_input[i];
			double mask = d_mask[i];
			sum1 += val * mask;
			sum2 += val * val * mask;
			samples += mask;
		}
		s_sums1[threadIdx.x] = sum1;
		s_sums2[threadIdx.x] = sum2;
		s_samples[threadIdx.x] = samples;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < MonoTpB; i++)
			{
				sum1 += s_sums1[i];
				sum2 += s_sums2[i];
				samples += s_samples[i];
			}

			s_mean = sum1 / samples;
			s_stddev = sqrt(tmax(0, samples * sum2 - (sum1 * sum1))) / samples;
		}
		__syncthreads();

		tfloat mean = s_mean;
		tfloat stddev = s_stddev > 0.0 ? 1.0 / s_stddev : 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			d_output[i] = (d_input[i] - mean) * stddev;

		if (outputmu && threadIdx.x == 0)
			d_mu[blockIdx.x] = tfloat2(mean, stddev);
	}

	template<bool outputmu> __global__ void NormMeanStdDevWarpMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, uchar elements, size_t n)
	{
		__shared__ double s_sums1[6][32];
		__shared__ double s_sums2[6][32];
		__shared__ double s_mean[6], s_stddev[6];

		size_t id = blockIdx.x * 6 + threadIdx.y;

		d_input += elements * (blockIdx.x * 6 + threadIdx.y);
		d_output += elements * (blockIdx.x * 6 + threadIdx.y);

		double sum1 = 0.0, sum2 = 0.0;

		if (id < n)
			for (uchar i = threadIdx.x; i < elements; i += 32)
			{
				double val = d_input[i];
				sum1 += val;
				sum2 += val * val;
			}
		s_sums1[threadIdx.y][threadIdx.x] = sum1;
		s_sums2[threadIdx.y][threadIdx.x] = sum2;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < 32; i++)
			{
				sum1 += s_sums1[threadIdx.y][i];
				sum2 += s_sums2[threadIdx.y][i];
			}

			s_mean[threadIdx.y] = sum1 / (double)elements;
			s_stddev[threadIdx.y] = sqrt((double)elements * sum2 - (sum1 * sum1)) / (double)elements;
		}
		__syncthreads();

		if (id >= n)
			return;

		double mean = s_mean[threadIdx.y];
		double stddev = s_stddev[threadIdx.y];

		for (uint i = threadIdx.x; i < elements; i += 32)
			d_output[i] = (d_input[i] - mean) / stddev;

		if (outputmu && threadIdx.x == 0)
			d_mu[blockIdx.x * 6 + threadIdx.y] = tfloat2(mean, stddev);
	}
	
	template<bool flipsign> __global__ void NormBackgroundMonoKernel(tfloat* d_input, tfloat* d_output, int3 dims, uint particleradius2)
	{
		__shared__ tfloat s_sums1[MonoTpB];
		__shared__ tfloat s_sums2[MonoTpB];
		__shared__ uint s_samples[MonoTpB];
		__shared__ tfloat s_mean, s_stddev;

		uint elements = Elements(dims);

		d_input += elements * blockIdx.x;
		d_output += elements * blockIdx.x;

		tfloat sum1 = 0.0, sum2 = 0.0;
		uint samples = 0;
		
		for (int z = 0; z < dims.z; z++)
			for (int y = 0; y < dims.y; y++)
				for (int x = threadIdx.x; x < dims.x; x += blockDim.x)
				{
					int zz = y - dims.z / 2;
					int yy = y - dims.y / 2;
					int xx = x - dims.x / 2;

					uint r = zz * zz + yy * yy + xx * xx;
					if (r <= particleradius2)
						continue;

					tfloat val = d_input[(z * dims.y + y) * dims.x + x];
					sum1 += val;
					sum2 += val * val;
					samples++;
				}
		s_sums1[threadIdx.x] = sum1;
		s_sums2[threadIdx.x] = sum2;
		s_samples[threadIdx.x] = samples;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < MonoTpB; i++)
			{
				sum1 += s_sums1[i];
				sum2 += s_sums2[i];
				samples += s_samples[i];
			}

			s_mean = sum1 / (tfloat)samples;
			s_stddev = sqrt(((tfloat)samples * sum2 - (sum1 * sum1))) / (tfloat)samples;
		}
		__syncthreads();

		tfloat mean = s_mean;
		tfloat stddev = s_stddev > 0 ? (tfloat)1 / s_stddev * (flipsign ? -1 : 1) : (tfloat)0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			d_output[i] = (d_input[i] - mean) * stddev;
	}

	__global__ void Mean0MonoKernel(tfloat* d_input, tfloat* d_output, size_t elements)
	{
		__shared__ double s_sums1[MonoTpB];
		__shared__ double s_mean;

		d_input += elements * blockIdx.x;
		d_output += elements * blockIdx.x;

		double sum1 = 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			double val = d_input[i];
			sum1 += val;
		}
		s_sums1[threadIdx.x] = sum1;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < MonoTpB; i++)
				sum1 += s_sums1[i];

			s_mean = sum1 / (double)elements;
		}
		__syncthreads();

		tfloat mean = s_mean;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			d_output[i] = d_input[i] - mean;
	}

	__global__ void NormFTMonoKernel(tcomplex* d_input, tcomplex* d_output, size_t elements)
	{
		__shared__ tfloat s_sums[MonoTpB];
		__shared__ tfloat s_mean;

		d_input += elements * blockIdx.x;
		d_output += elements * blockIdx.x;

		double sum = 0.0;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
		{
			tcomplex val = d_input[i];
			sum += dotp2(val, val);
		}
		s_sums[threadIdx.x] = sum;
		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < MonoTpB; i++)
				sum += s_sums[i];

			s_mean = sqrt(sum / elements);
		}
		__syncthreads();

		tfloat mean = s_mean;
		mean = tmax(mean, 1e-20f);
		mean = 1 / mean;

		for (int i = threadIdx.x; i < elements; i += blockDim.x)
			d_output[i] = d_input[i] * mean;
	}
}