#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CubicInterp.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void RemapKernel(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch);
	template <class T> __global__ void RemapReverseKernel(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch);
	template<bool cubicinterp> __global__ void RemapInterpolated2DKernel(cudaTex t_input, tfloat* d_output, float2* d_addresses, int n);


	//////////////////
	//Data remapping//
	//////////////////

	template <class T> void d_Remap(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
	{
		size_t TpB = 192;
		size_t totalblocks = tmin((elementsmapped + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks);
		RemapKernel<T> << <grid, (uint)TpB >> > (d_input, d_map, d_output, elementsmapped, elementsoriginal, defvalue, batch);
	}
	template void d_Remap<tfloat>(tfloat* d_input, size_t* d_map, tfloat* d_output, size_t elementsmapped, size_t elementsoriginal, tfloat defvalue, int batch);
	template void d_Remap<tcomplex>(tcomplex* d_input, size_t* d_map, tcomplex* d_output, size_t elementsmapped, size_t elementsoriginal, tcomplex defvalue, int batch);
	template void d_Remap<int>(int* d_input, size_t* d_map, int* d_output, size_t elementsmapped, size_t elementsoriginal, int defvalue, int batch);

	template <class T> void d_RemapReverse(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch)
	{
		size_t TpB = 192;
		size_t totalblocks = tmin((elementsmapped + TpB - 1) / TpB, (size_t)32768);
		dim3 grid = dim3((uint)totalblocks);
		RemapReverseKernel<T> << <grid, (uint)TpB >> > (d_input, d_map, d_output, elementsmapped, elementsdestination, defvalue, batch);
	}
	template void d_RemapReverse<tfloat>(tfloat* d_input, size_t* d_map, tfloat* d_output, size_t elementsmapped, size_t elementsdestination, tfloat defvalue, int batch);
	template void d_RemapReverse<int>(int* d_input, size_t* d_map, int* d_output, size_t elementsmapped, size_t elementsdestination, int defvalue, int batch);

	template <class T> void h_Remap(T* h_input, size_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
	{
		T* d_input = (T*)CudaMallocFromHostArray(h_input, elementsoriginal * batch * sizeof(T));
		size_t* d_map = (size_t*)CudaMallocFromHostArray(h_map, elementsmapped * sizeof(size_t));
		T* d_output;
		cudaMalloc((void**)&d_output, elementsmapped * batch * sizeof(T));

		d_Remap(d_input, d_map, d_output, elementsmapped, elementsoriginal, defvalue, batch);

		cudaMemcpy(h_output, d_output, elementsmapped * batch * sizeof(T), cudaMemcpyDeviceToHost);

		cudaFree(d_input);
		cudaFree(d_map);
		cudaFree(d_output);
	}
	template void h_Remap<tfloat>(tfloat* d_input, size_t* d_map, tfloat* d_output, size_t elementsmapped, size_t elementsoriginal, tfloat defvalue, int batch);
	template void h_Remap<int>(int* d_input, size_t* d_map, int* d_output, size_t elementsmapped, size_t elementsoriginal, int defvalue, int batch);

	template <class T> __global__ void RemapKernel(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
	{
		size_t address;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elementsmapped;
			id += blockDim.x * gridDim.x)
		{
			address = d_map[id];
			for (size_t b = 0; b < batch; b++)
				d_output[id + elementsmapped * b] = d_input[address + elementsoriginal * b];
		}
	}

	template <class T> __global__ void RemapReverseKernel(T* d_input, size_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch)
	{
		size_t address;
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elementsmapped;
			id += blockDim.x * gridDim.x)
		{
			address = d_map[id];
			for (size_t b = 0; b < batch; b++)
				d_output[address + elementsdestination * b] = d_input[id + elementsmapped * b];
		}
	}


	/////////////////////
	//Texture remapping//
	/////////////////////

	void d_RemapInterpolated2D(tfloat* d_input, int2 dimsinput, tfloat* d_output, float2* d_addresses, int n, T_INTERP_MODE mode)
	{
		cudaArray* a_input;
		cudaTex t_input;
		if (mode == T_INTERP_LINEAR)
			d_BindTextureToArray(d_input, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
		else if (mode == T_INTERP_CUBIC)
		{
			tfloat* d_temp;
			cudaMalloc((void**)&d_temp, Elements2(dimsinput) * sizeof(tfloat));
			cudaMemcpy(d_temp, d_input, Elements2(dimsinput) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter2D(d_temp, dimsinput);
			d_BindTextureToArray(d_temp, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
			cudaFree(d_temp);
		}

		int TpB = tmin(256, NextMultipleOf(n, 32));
		int grid = tmin((n + TpB - 1) / TpB, 8192);
		if (mode == T_INTERP_LINEAR)
			RemapInterpolated2DKernel<false> << <grid, TpB >> > (t_input, d_output, d_addresses, n);
		else if (mode == T_INTERP_CUBIC)
			RemapInterpolated2DKernel<true> << <grid, TpB >> > (t_input, d_output, d_addresses, n);

		cudaDestroyTextureObject(t_input);
		cudaFreeArray(a_input);
	}

	template<bool cubicinterp> __global__ void RemapInterpolated2DKernel(cudaTex t_input, tfloat* d_output, float2* d_addresses, int n)
	{
		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
		{
			float2 address = d_addresses[idx];
			if (cubicinterp)
				d_output[idx] = cubicTex2D(t_input, address.x, address.y);
			else
				d_output[idx] = tex2D<tfloat>(t_input, address.x, address.y);
		}
	}


	///////////////////////////////////
	//Sparse mask to dense conversion//
	///////////////////////////////////

	template <class T> void h_MaskSparseToDense(T* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal)
	{
		size_t lastaddress = 0;
		size_t* h_tempforward = (size_t*)malloc(elementsoriginal * sizeof(size_t));

		if (h_mapbackward != NULL)
			for (size_t i = 0; i < elementsoriginal; i++)
				if (h_input[i] > 0)
				{
					h_tempforward[lastaddress] = i;
					h_mapbackward[i] = lastaddress;
					lastaddress++;
				}
				else
					h_mapbackward[i] = -1;
		else
			for (size_t i = 0; i < elementsoriginal; i++)
			{
				if (h_input[i] > 0)
				{
					h_tempforward[lastaddress] = i;
					lastaddress++;
				}
			}

		if (lastaddress == 0)
		{
			*h_mapforward = NULL;
			elementsmapped = 0;
		}
		else
		{
			*h_mapforward = (size_t*)malloc(lastaddress * sizeof(size_t));
			memcpy(*h_mapforward, h_tempforward, lastaddress * sizeof(size_t));
			elementsmapped = lastaddress;
		}

		free(h_tempforward);
	}
	template void h_MaskSparseToDense<float>(float* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void h_MaskSparseToDense<double>(double* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void h_MaskSparseToDense<int>(int* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void h_MaskSparseToDense<bool>(bool* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void h_MaskSparseToDense<char>(char* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);


	template <class T> void d_MaskSparseToDense(T* d_input, size_t** d_mapforward, size_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal)
	{
		T* h_input = (T*)MallocFromDeviceArray(d_input, elementsoriginal * sizeof(T));
		size_t* h_mapforward = NULL;
		size_t* h_mapbackward = d_mapbackward == NULL ? NULL : (size_t*)malloc(elementsoriginal * sizeof(size_t));
		size_t elements = 0;

		h_MaskSparseToDense(h_input, &h_mapforward, h_mapbackward, elements, elementsoriginal);

		*d_mapforward = h_mapforward == NULL ? NULL : (size_t*)CudaMallocFromHostArray(h_mapforward, elements * sizeof(size_t));
		if (d_mapbackward != NULL && h_mapbackward != NULL)
			cudaMemcpy(d_mapbackward, h_mapbackward, elementsoriginal * sizeof(size_t), cudaMemcpyHostToDevice);

		elementsmapped = elements;

		free(h_input);
		if (h_mapbackward != NULL)
			free(h_mapbackward);
		if (h_mapforward != NULL)
			free(h_mapforward);
	}
	template void d_MaskSparseToDense<float>(float* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void d_MaskSparseToDense<double>(double* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void d_MaskSparseToDense<int>(int* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void d_MaskSparseToDense<bool>(bool* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
	template void d_MaskSparseToDense<char>(char* h_input, size_t** h_mapforward, size_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
}