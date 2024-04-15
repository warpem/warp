#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/FFT.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void FFTCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims);
	template <class T> __global__ void FFTFullCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims);
	template <class T> __global__ void FFTPadEvenKernel(T* d_input, T* d_output, int3 olddims, int3 newdims);
	template <class T> __global__ void FFTFullPadEvenKernel(T* d_input, T* d_output, int3 olddims, int3 newdims);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_FFTCrop(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch)
	{
		size_t elementsnew = ElementsFFT(newdims);
		size_t elementsold = ElementsFFT(olddims);

		T* d_intermediate;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(newdims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
		dim3 grid = dim3(newdims.y, newdims.z, batch);
		FFTCropKernel << <grid, TpB >> > (d_input, d_intermediate, olddims, newdims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(newdims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_FFTCrop<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch);
	template void d_FFTCrop<tfloat>(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, int batch);

	template <class T> void d_FFTFullCrop(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch)
	{
		size_t elementsnew = Elements(newdims);
		size_t elementsold = Elements(olddims);

		T* d_intermediate;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(newdims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(newdims.x, 32));
		dim3 grid = dim3(newdims.y, newdims.z, batch);
		FFTFullCropKernel << <grid, TpB >> > (d_input, d_intermediate, olddims, newdims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(newdims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_FFTFullCrop<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch);
	template void d_FFTFullCrop<tfloat>(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, int batch);

	template <class T> void d_FFTPad(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch)
	{
		size_t elementsnew = ElementsFFT(newdims);
		size_t elementsold = ElementsFFT(olddims);

		T* d_intermediate;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(newdims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
		dim3 grid = dim3(newdims.y, newdims.z, batch);
		FFTPadEvenKernel << <grid, TpB >> > (d_input, d_intermediate, olddims, newdims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(newdims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_FFTPad<tfloat>(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, int batch);
	template void d_FFTPad<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch);

	template <class T> void d_FFTFullPad(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch)
	{
		size_t elementsnew = Elements(newdims);
		size_t elementsold = Elements(olddims);

		T* d_intermediate;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(newdims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(newdims.x, 32));
		dim3 grid = dim3(newdims.y, newdims.z, batch);
		FFTFullPadEvenKernel << <grid, TpB >> > (d_input, d_output, olddims, newdims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(newdims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_FFTFullPad<tfloat>(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, int batch);
	template void d_FFTFullPad<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void FFTCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims)
	{
		d_input += ElementsFFT(olddims) * blockIdx.z;
		d_output += ElementsFFT(newdims) * blockIdx.z;

		for (int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
		{
			int y = blockIdx.x;
			int yy = y < newdims.y / 2 + 1 ? y : y - newdims.y + olddims.y;
			int z = blockIdx.y;
			int zz = z < newdims.z / 2 + 1 ? z : z - newdims.z + olddims.z;

			/*yy = tmax(0, tmin(yy, olddims.y - 1));
			zz = tmax(0, tmin(zz, olddims.z - 1));*/

			d_output[(z * newdims.y + y) * (newdims.x / 2 + 1) + x] = d_input[(zz * olddims.y + yy) * (olddims.x / 2 + 1) + x];
		}
	}

	template <class T> __global__ void FFTFullCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims)
	{
		int oldy = blockIdx.x;
		if (oldy >= newdims.y / 2)
			oldy += olddims.y - newdims.y;
		int oldz = blockIdx.y;
		if (oldz >= newdims.z / 2)
			oldz += olddims.z - newdims.z;

		d_input += Elements(olddims) * blockIdx.z + (oldz * olddims.y + oldy) * olddims.x;
		d_output += Elements(newdims) * blockIdx.z + (blockIdx.y * newdims.y + blockIdx.x) * newdims.x;

		for (int x = threadIdx.x; x < newdims.x; x += blockDim.x)
		{
			int oldx = x;
			if (oldx >= newdims.x / 2)
				oldx += olddims.x - newdims.x;

			d_output[x] = d_input[oldx];
		}
	}

	template <class T> __global__ void FFTPadEvenKernel(T* d_input, T* d_output, int3 olddims, int3 newdims)
	{
		d_input += ElementsFFT(olddims) * blockIdx.z;
		d_output += ElementsFFT(newdims) * blockIdx.z;

		for (int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
		{
			int newry = ((blockIdx.x + newdims.y / 2) % newdims.y);
			int newrz = ((blockIdx.y + newdims.z / 2) % newdims.z);

			int oldry = newry + (olddims.y - newdims.y) / 2;
			int oldrz = newrz + (olddims.z - newdims.z) / 2;

			if (x < (olddims.x + 1) / 2 && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
			{
				int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
				int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

				d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
			}
			else
				d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = (T)0;
		}
	}

	template<> __global__ void FFTPadEvenKernel<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
	{
		d_input += ElementsFFT(olddims) * blockIdx.z;
		d_output += ElementsFFT(newdims) * blockIdx.z;

		for (int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
		{
			int newry = ((blockIdx.x + newdims.y / 2) % newdims.y);
			int newrz = ((blockIdx.y + newdims.z / 2) % newdims.z);

			int oldry = newry + (olddims.y - newdims.y) / 2;
			int oldrz = newrz + (olddims.z - newdims.z) / 2;

			if (x < (olddims.x + 1) / 2 && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
			{
				int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
				int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

				d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
			}
			else
				d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = make_cuComplex(0.0f, 0.0f);
		}
	}

	template <class T> __global__ void FFTFullPadEvenKernel(T* d_input, T* d_output, int3 olddims, int3 newdims)
	{
		d_input += Elements(olddims) * blockIdx.z;
		d_output += Elements(newdims) * blockIdx.z;

		for (int x = threadIdx.x; x < newdims.x; x += blockDim.x)
		{
			int newrx = ((x + (newdims.x) / 2) % newdims.x);
			int newry = ((blockIdx.x + (newdims.y) / 2) % newdims.y);
			int newrz = ((blockIdx.y + (newdims.z) / 2) % newdims.z);

			int oldrx = newrx + (olddims.x - newdims.x - ((olddims.x & 1 - (newdims.x & 1)) % 2)) / 2;
			int oldry = newry + (olddims.y - newdims.y - ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2;
			int oldrz = newrz + (olddims.z - newdims.z - ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2;

			if (oldrx >= 0 && oldrx < olddims.x && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
			{
				int oldx = ((oldrx + (olddims.x + 1) / 2) % olddims.x);
				int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
				int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

				d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = d_input[(oldz * olddims.y + oldy) * olddims.x + oldx];
			}
			else
				d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = (T)0;
		}
	}

	template<> __global__ void FFTFullPadEvenKernel<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
	{
		d_input += Elements(olddims) * blockIdx.z;
		d_output += Elements(newdims) * blockIdx.z;

		for (int x = threadIdx.x; x < newdims.x; x += blockDim.x)
		{
			int newrx = ((x + (newdims.x) / 2) % newdims.x);
			int newry = ((blockIdx.x + (newdims.y) / 2) % newdims.y);
			int newrz = ((blockIdx.y + (newdims.z) / 2) % newdims.z);

			int oldrx = newrx + (olddims.x - newdims.x - ((olddims.x & 1 - (newdims.x & 1)) % 2)) / 2;
			int oldry = newry + (olddims.y - newdims.y - ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2;
			int oldrz = newrz + (olddims.z - newdims.z - ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2;

			if (oldrx >= 0 && oldrx < olddims.x && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
			{
				int oldx = ((oldrx + (olddims.x + 1) / 2) % olddims.x);
				int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
				int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

				d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = d_input[(oldz * olddims.y + oldy) * olddims.x + oldx];
			}
			else
				d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = make_cuComplex(0.0f, 0.0f);
		}
	}
}