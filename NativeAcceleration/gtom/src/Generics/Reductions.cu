#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors, int vectorlength);
	template<class T> __global__ void ReduceMeanKernel(T* d_input, T* d_output, int nvectors, int vectorlength);
	template<class T> __global__ void ReduceMeanWeightedKernel(T* d_input, tfloat* d_inputweights, T* d_output, int nvectors, int vectorlength);
	template<class T> __global__ void ReduceOrKernel(T* d_input, T* d_output, uint nvectors, uint vectorlength, uint batch);
	template<class T> __global__ void ReduceMaxKernel(T* d_input, T* d_output, int nvectors, int vectorlength);


	////////////
	//Addition//
	////////////

	template<class T> void d_ReduceAdd(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
	{
		int TpB = min(NextMultipleOf(vectorlength, 32), 128);
		dim3 grid = dim3(min((vectorlength + TpB - 1) / TpB, 1024), batch);
		ReduceAddKernel<T> << <grid, TpB >> > (d_input, d_output, nvectors, vectorlength);
	}
	template void d_ReduceAdd<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceAdd<float2>(float2* d_input, float2* d_output, int vectorlength, int nvectors, int batch);

	template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;
		d_output += blockIdx.y * vectorlength;

		for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			T sum = (T)0;

			for (int n = 0; n < nvectors; n++)
				sum += d_input[n * vectorlength + id];

			d_output[id] = sum;
		}
	}

	template<> __global__ void ReduceAddKernel<float2>(float2* d_input, float2* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;
		d_output += blockIdx.y * vectorlength;

		for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			float2 sum = make_float2(0.0f, 0.0f);

			for (int n = 0; n < nvectors; n++)
				sum += d_input[n * vectorlength + id];

			d_output[id] = sum;
		}
	}


	////////
	//Mean//
	////////

	template<class T> void d_ReduceMean(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
	{
		int TpB = tmin(NextMultipleOf(vectorlength, 32), 256);
		dim3 grid = dim3(tmin((vectorlength + TpB - 1) / TpB, 2048), batch);
		ReduceMeanKernel<T> << <grid, TpB >> > (d_input, d_output, nvectors, vectorlength);
	}
	template void d_ReduceMean<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<half>(half* d_input, half* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMean<float2>(float2* d_input, float2* d_output, int vectorlength, int nvectors, int batch);

	template<class T> __global__ void ReduceMeanKernel(T* d_input, T* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			T sum = (T)0;

			for (uint n = 0; n < nvectors; n++)
				sum += d_input[n * vectorlength + id];

			d_output[blockIdx.y * vectorlength + id] = sum / (T)nvectors;
		}
	}

	template<> __global__ void ReduceMeanKernel<half>(half* d_input, half* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			float sum = 0.0f;

			for (uint n = 0; n < nvectors; n++)
				sum += __half2float(d_input[n * vectorlength + id]);

			d_output[blockIdx.y * vectorlength + id] = __float2half(sum / (float)nvectors);
		}
	}

	template<> __global__ void ReduceMeanKernel<float2>(float2* d_input, float2* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			float2 sum = make_float2(0.0f, 0.0f);

			for (uint n = 0; n < nvectors; n++)
				sum += d_input[n * vectorlength + id];

			d_output[blockIdx.y * vectorlength + id] = sum / (float)nvectors;
		}
	}


	/////////////////
	//Mean weighted//
	/////////////////

	template<class T> void d_ReduceMeanWeighted(T* d_input, tfloat* d_inputweights, T* d_output, int vectorlength, int nvectors, int batch)
	{
		int TpB = tmin(NextMultipleOf(vectorlength, 32), 256);
		dim3 grid = dim3(tmin((vectorlength + TpB - 1) / TpB, 2048), batch);
		ReduceMeanWeightedKernel<T> << <grid, TpB >> > (d_input, d_inputweights, d_output, nvectors, vectorlength);
	}
	template void d_ReduceMeanWeighted<char>(char* d_input, tfloat* d_inputweights, char* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMeanWeighted<short>(short* d_input, tfloat* d_inputweights, short* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMeanWeighted<int>(int* d_input, tfloat* d_inputweights, int* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMeanWeighted<uint>(uint* d_input, tfloat* d_inputweights, uint* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMeanWeighted<float>(float* d_input, tfloat* d_inputweights, float* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMeanWeighted<double>(double* d_input, tfloat* d_inputweights, double* d_output, int vectorlength, int nvectors, int batch);

	template<class T> __global__ void ReduceMeanWeightedKernel(T* d_input, tfloat* d_inputweights, T* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			T sum = (T)0;
			tfloat weightsum = 0;

			for (uint n = 0; n < nvectors; n++)
			{
				tfloat weight = d_inputweights[n * vectorlength + id];
				weightsum += weight;
				sum += d_input[n * vectorlength + id] * weight;
			}

			if (weightsum != 0)
				d_output[blockIdx.y * vectorlength + id] = sum / weightsum;
			else
				d_output[blockIdx.y * vectorlength + id] = (T)0;
		}
	}


	//////
	//Or//
	//////

	template<class T> void d_ReduceOr(T* d_input, T* d_output, uint vectorlength, uint nvectors, uint batch)
	{
		int TpB = min(NextMultipleOf(nvectors, 32), 256);
		dim3 grid = dim3(min(vectorlength, 2048), min(batch, 32768));
		ReduceOrKernel<T> << <grid, TpB >> > (d_input, d_output, nvectors, vectorlength, batch);
	}
	template void d_ReduceOr<char>(char* d_input, char* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<uchar>(uchar* d_input, uchar* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<short>(short* d_input, short* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<ushort>(ushort* d_input, ushort* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<int>(int* d_input, int* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<uint>(uint* d_input, uint* d_output, uint vectorlength, uint nvectors, uint batch);
	template void d_ReduceOr<bool>(bool* d_input, bool* d_output, uint vectorlength, uint nvectors, uint batch);

	template<class T> __global__ void ReduceOrKernel(T* d_input, T* d_output, uint nvectors, uint vectorlength, uint batch)
	{
		d_input += blockIdx.y * nvectors * vectorlength;
		d_output += blockIdx.y * vectorlength;

		for (uint b = blockIdx.y; b < batch; b += gridDim.y, d_input += gridDim.y * nvectors * vectorlength, d_output += gridDim.y * vectorlength)
		{
			for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
			{
				T sum = (T)0;

				for (uint n = 0; n < nvectors; n++)
					sum |= d_input[n * vectorlength + id];

				d_output[id] = sum;
			}
		}
	}


	///////
	//Max//
	///////

	template<class T> void d_ReduceMax(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
	{
		int TpB = min(NextMultipleOf(vectorlength, 32), 128);
		dim3 grid = dim3(min((vectorlength + TpB - 1) / TpB, 1024), batch);
		ReduceMaxKernel<T> << <grid, TpB >> > (d_input, d_output, nvectors, vectorlength);
	}
	template void d_ReduceMax<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMax<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMax<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMax<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMax<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
	template void d_ReduceMax<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);

	template<class T> __global__ void ReduceMaxKernel(T* d_input, T* d_output, int nvectors, int vectorlength)
	{
		d_input += blockIdx.y * nvectors * vectorlength;
		d_output += blockIdx.y * vectorlength;

		for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
		{
			T max = d_input[id];

			for (int n = 1; n < nvectors; n++)
				max = tmax(max, d_input[n * vectorlength + id]);

			d_output[id] = max;
		}
	}
}