#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void MemcpyMultiKernel(T* d_output, T* d_input, size_t elements, int copies);
	template <class T> __global__ void MemcpyStridedKernel(T* d_output, T* d_input, size_t elements, int stridedst, int stridesrc);
	template <class T> __global__ void MemcpyBlockStridedKernel(T* d_output, T* d_input, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template <class T> __global__ void ValueFillKernel(T* d_output, size_t elements, T value);
	template <class T, int fieldcount> __global__ void JoinInterleavedKernel(T** d_fields, T* d_output, size_t elements);
	template <class T1, class T2> __global__ void TypeConversionKernel(T1* d_input, T2* d_output, size_t elements);


	///////////////
	//Host memory//
	///////////////

	void* MallocFromDeviceArray(void* d_array, size_t size)
	{
		void* h_array = malloc(size);
		cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
		cudaStreamQuery(0);

		return h_array;
	}

	void* MallocPinnedFromDeviceArray(void* d_array, size_t size)
	{
		void* h_array;
		cudaMallocHost((void**)&h_array, size);
		cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
		cudaStreamQuery(0);

		return h_array;
	}

	tfloat* MallocZeroFilledFloat(size_t elements)
	{
		return MallocValueFilled<tfloat>(elements, (tfloat)0.0);
	}

	template <class T> T* MallocValueFilled(size_t elements, T value)
	{
		T* h_array = (T*)malloc(elements * sizeof(T));

		intptr_t s_elements = (intptr_t)elements;
		//#pragma omp for schedule(dynamic, 1024)
		for (intptr_t i = 0; i < s_elements; i++)
			h_array[i] = value;

		return h_array;
	}
	template tfloat* MallocValueFilled<tfloat>(size_t elements, tfloat value);
	template double* MallocValueFilled<double>(size_t elements, double value);
	template tcomplex* MallocValueFilled<tcomplex>(size_t elements, tcomplex value);
	template char* MallocValueFilled<char>(size_t elements, char value);
	template bool* MallocValueFilled<bool>(size_t elements, bool value);
	template int* MallocValueFilled<int>(size_t elements, int value);

	tfloat* MixedToHostTfloat(void* h_input, EM_DATATYPE datatype, size_t elements)
	{
		tfloat* h_output;
		cudaMallocHost((void**)&h_output, elements * sizeof(tfloat));

		if (datatype == EM_BYTE)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((unsigned char*)h_input)[i];
		else if (datatype == EM_SHORT)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((short*)h_input)[i];
		else if (datatype == EM_LONG)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((int*)h_input)[i];
		else if (datatype == EM_SINGLE)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((float*)h_input)[i];
		else if (datatype == EM_DOUBLE)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((double*)h_input)[i];
		else
			throw;

		return h_output;
	}

	tfloat* MixedToHostTfloat(void* h_input, MRC_DATATYPE datatype, size_t elements)
	{
		tfloat* h_output;
		cudaMallocHost((void**)&h_output, elements * sizeof(tfloat));

		if (datatype == MRC_BYTE)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((unsigned char*)h_input)[i];
		else if (datatype == MRC_SHORT)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((short*)h_input)[i];
		else if (datatype == MRC_UNSIGNEDSHORT)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((unsigned short*)h_input)[i];
		else if (datatype == MRC_FLOAT)
#pragma omp parallel for schedule(dynamic, 1024)
			for (intptr_t i = 0; i < elements; i++)
				h_output[i] = (tfloat)((float*)h_input)[i];
		else
			throw;

		return h_output;
	}

	void WriteToBinaryFile(std::string path, void* data, size_t bytes)
	{
		FILE* outputfile = fopen(path.c_str(), "wb");
		fwrite(data, sizeof(char), bytes, outputfile);
		fclose(outputfile);
	}

	template <class T1, class T2> void MemcpyFromDeviceArrayConverted(T1* d_array, T2* h_output, size_t elements)
	{
		T2* d_output;
		cudaMalloc((void**)&d_output, elements * sizeof(T2));

		size_t TpB = tmin((size_t)768, elements);
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		TypeConversionKernel<T1, T2> << <grid, (uint)TpB >> > (d_array, d_output, elements);

		cudaMemcpy(h_output, d_output, elements * sizeof(T2), cudaMemcpyDeviceToHost);

		cudaFree(d_output);
	}
	template void MemcpyFromDeviceArrayConverted<tfloat, char>(tfloat* d_array, char* h_output, size_t elements);
	template void MemcpyFromDeviceArrayConverted<tfloat, short>(tfloat* d_array, short* h_output, size_t elements);
	template void MemcpyFromDeviceArrayConverted<tfloat, int>(tfloat* d_array, int* h_output, size_t elements);
	template void MemcpyFromDeviceArrayConverted<tfloat, long>(tfloat* d_array, long* h_output, size_t elements);
	template void MemcpyFromDeviceArrayConverted<tfloat, float>(tfloat* d_array, float* h_output, size_t elements);
	template void MemcpyFromDeviceArrayConverted<tfloat, double>(tfloat* d_array, double* h_output, size_t elements);


	/////////////////
	//Device memory//
	/////////////////

	template<class T> void CudaMemcpyMulti(T* dst, T* src, uint elements, uint copies, uint batch)
	{
		size_t TpB = min((uint)256, elements);
		dim3 grid = dim3(tmin((elements + TpB - 1) / TpB, (size_t)8192), batch);
		MemcpyMultiKernel << <grid, TpB >> > (dst, src, elements, copies);
	}
	template void CudaMemcpyMulti<char>(char* dst, char* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<short>(short* dst, short* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<int>(int* dst, int* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<long>(long* dst, long* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<float>(float* dst, float* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<double>(double* dst, double* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<float2>(float2* dst, float2* src, uint elements, uint copies, uint batch);
	template void CudaMemcpyMulti<double2>(double2* dst, double2* src, uint elements, uint copies, uint batch);

	template<class T> void CudaMemcpyStrided(T* dst, T* src, size_t elements, int stridedst, int stridesrc)
	{
		size_t TpB = tmin((size_t)256, elements);
		dim3 grid = dim3(tmin((elements + TpB - 1) / TpB, (size_t)8192));
		MemcpyStridedKernel << <grid, TpB >> > (dst, src, elements, stridedst, stridesrc);
	}
	template void CudaMemcpyStrided<char>(char* dst, char* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<short>(short* dst, short* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<int>(int* dst, int* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<long>(long* dst, long* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<float>(float* dst, float* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<double>(double* dst, double* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<float2>(float2* dst, float2* src, size_t elements, int stridedst, int stridesrc);
	template void CudaMemcpyStrided<double2>(double2* dst, double2* src, size_t elements, int stridedst, int stridesrc);

	template<class T> void CudaMemcpyBlockStrided(T* dst, T* src, size_t blockelements, int stridedst, int stridesrc, int nblocks)
	{
		int TpB = tmin(256, blockelements);
		dim3 grid = dim3(tmin((blockelements + TpB - 1) / TpB, 8192), nblocks);
		MemcpyBlockStridedKernel << <grid, TpB >> > (dst, src, blockelements, stridedst, stridesrc, nblocks);
	}
	template void CudaMemcpyBlockStrided<char>(char* dst, char* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<short>(short* dst, short* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<int>(int* dst, int* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<long>(long* dst, long* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<float>(float* dst, float* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<double>(double* dst, double* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<float2>(float2* dst, float2* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);
	template void CudaMemcpyBlockStrided<double2>(double2* dst, double2* src, size_t blockelements, int stridedst, int stridesrc, int nblocks);

	void* CudaMallocAligned2D(size_t widthbytes, size_t height, int* pitch, int alignment)
	{
		if ((widthbytes % alignment) != 0)
			widthbytes += (alignment - (widthbytes % alignment));

		(*pitch) = widthbytes;

		void* ptr;
		cudaMalloc((void**)&ptr, widthbytes* height);

		return ptr;
	}

	void* CudaMallocFromHostArray(void* h_array, size_t size)
	{
		void* d_array;
		cudaMalloc((void**)&d_array, size);
		cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

		return d_array;
	}

	void* CudaMallocFromHostArray(void* h_array, size_t devicesize, size_t hostsize)
	{
		void* d_array;
		cudaMalloc((void**)&d_array, devicesize);
		cudaMemcpy(d_array, h_array, hostsize, cudaMemcpyHostToDevice);

		return d_array;
	}

	template <class T1, class T2> T2* CudaMallocFromHostArrayConverted(T1* h_array, size_t elements)
	{
		T2* d_output;

		CudaMallocFromHostArrayConverted(h_array, &d_output, elements);

		return d_output;
	}
	template tfloat* CudaMallocFromHostArrayConverted<char, tfloat>(char* h_array, size_t elements);
	template tfloat* CudaMallocFromHostArrayConverted<unsigned char, tfloat>(unsigned char* h_array, size_t elements);
	template tfloat* CudaMallocFromHostArrayConverted<short, tfloat>(short* h_array, size_t elements);
	template tfloat* CudaMallocFromHostArrayConverted<unsigned short, tfloat>(unsigned short* h_array, size_t elements);
	template tfloat* CudaMallocFromHostArrayConverted<int, tfloat>(int* h_array, size_t elements);
	template tfloat* CudaMallocFromHostArrayConverted<double, tfloat>(double* h_array, size_t elements);

	template <class T1, class T2> void CudaMemcpyFromHostArrayConverted(T1* h_array, T2* d_output, size_t elements)
	{
		T1* d_input = (T1*)CudaMallocFromHostArray(h_array, elements * sizeof(T1));

		size_t TpB = tmin((size_t)768, elements);
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		TypeConversionKernel<T1, T2> << <grid, (uint)TpB >> > (d_input, d_output, elements);

		cudaFree(d_input);
	}
	template void CudaMemcpyFromHostArrayConverted<char, tfloat>(char* h_array, tfloat* d_output, size_t elements);
	template void CudaMemcpyFromHostArrayConverted<short, tfloat>(short* h_array, tfloat* d_output, size_t elements);
	template void CudaMemcpyFromHostArrayConverted<unsigned short, tfloat>(unsigned short* h_array, tfloat* d_output, size_t elements);
	template void CudaMemcpyFromHostArrayConverted<int, tfloat>(int* h_array, tfloat* d_output, size_t elements);
	template void CudaMemcpyFromHostArrayConverted<double, tfloat>(double* h_array, tfloat* d_output, size_t elements);

	template <class T1, class T2> void CudaMallocFromHostArrayConverted(T1* h_array, T2** d_output, size_t elements)
	{
		T1* d_input = (T1*)CudaMallocFromHostArray(h_array, elements * sizeof(T1));
		cudaMalloc((void**)d_output, elements * sizeof(T2));

		size_t TpB = tmin((size_t)768, elements);
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		TypeConversionKernel<T1, T2> << <grid, (uint)TpB >> > (d_input, *d_output, elements);

		cudaFree(d_input);
	}
	template void CudaMallocFromHostArrayConverted<unsigned char, tfloat>(unsigned char* h_array, tfloat** d_output, size_t elements);
	template void CudaMallocFromHostArrayConverted<short, tfloat>(short* h_array, tfloat** d_output, size_t elements);
	template void CudaMallocFromHostArrayConverted<unsigned short, tfloat>(unsigned short* h_array, tfloat** d_output, size_t elements);
	template void CudaMallocFromHostArrayConverted<int, tfloat>(int* h_array, tfloat** d_output, size_t elements);
	template void CudaMallocFromHostArrayConverted<double, tfloat>(double* h_array, tfloat** d_output, size_t elements);

	tfloat* CudaMallocZeroFilledFloat(size_t elements)
	{
		return CudaMallocValueFilled<tfloat>(elements, (tfloat)0.0);
	}

	template <class T> T* CudaMallocValueFilled(size_t elements, T value)
	{
		T* d_array;
		cudaMalloc((void**)&d_array, elements * sizeof(T));

		d_ValueFill(d_array, elements, value);

		return d_array;
	}
	template float* CudaMallocValueFilled<float>(size_t elements, float value);
	template double* CudaMallocValueFilled<double>(size_t elements, double value);
	template tcomplex* CudaMallocValueFilled<tcomplex>(size_t elements, tcomplex value);
	template char* CudaMallocValueFilled<char>(size_t elements, char value);
	template unsigned char* CudaMallocValueFilled<unsigned char>(size_t elements, unsigned char value);
	template short* CudaMallocValueFilled<short>(size_t elements, short value);
	template unsigned short* CudaMallocValueFilled<unsigned short>(size_t elements, unsigned short value);
	template int* CudaMallocValueFilled<int>(size_t elements, int value);
	template int2* CudaMallocValueFilled<int2>(size_t elements, int2 value);
	template int3* CudaMallocValueFilled<int3>(size_t elements, int3 value);
	template uint* CudaMallocValueFilled<uint>(size_t elements, uint value);
	template bool* CudaMallocValueFilled<bool>(size_t elements, bool value);
	template tfloat2* CudaMallocValueFilled<tfloat2>(size_t elements, tfloat2 value);
	template tfloat3* CudaMallocValueFilled<tfloat3>(size_t elements, tfloat3 value);
	template tfloat4* CudaMallocValueFilled<tfloat4>(size_t elements, tfloat4 value);

	template <class T> void d_ValueFill(T* d_array, size_t elements, T value)
	{
		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		ValueFillKernel<T> << <grid, (uint)TpB >> > (d_array, elements, value);
		cudaStreamQuery(0);
	}

	tfloat* CudaMallocRandomFilled(size_t elements, tfloat mean, tfloat stddev, curandGenerator_t generator)
	{
		tfloat* d_array;
		cudaMalloc((void**)&d_array, elements * sizeof(tfloat));
		
		d_RandomFill(d_array, elements, mean, stddev, generator);

		return d_array;
	}

	tfloat* CudaMallocRandomFilled(size_t elements, tfloat mean, tfloat stddev, unsigned long long seed)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, seed);

		tfloat* d_array = CudaMallocRandomFilled(elements, mean, stddev, generator);

		curandDestroyGenerator(generator);

		return d_array;
	}

	void d_RandomFill(tfloat* d_array, size_t elements, tfloat mean, tfloat stddev, curandGenerator_t generator)
	{
#ifndef GTOM_DOUBLE
		curandGenerateNormal(generator, d_array, elements, mean, stddev);
#else
		curandGenerateLogNormalDouble(generator, d_array, elements, mean, stddev);
#endif
	}

	void d_RandomFill(tfloat* d_array, size_t elements, tfloat mean, tfloat stddev, unsigned long long seed)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, seed);

		d_RandomFill(d_array, elements, mean, stddev, generator);

		curandDestroyGenerator(generator);
	}

	template <class T, int fieldcount> T* d_JoinInterleaved(T** d_fields, size_t elements)
	{
		T* d_output;
		cudaMalloc((void**)&d_output, elements * fieldcount * sizeof(T));

		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		JoinInterleavedKernel<T, fieldcount> << <grid, (uint)TpB >> > (d_fields, d_output, elements);
		cudaStreamQuery(0);

		return d_output;
	}
	template tfloat* d_JoinInterleaved<tfloat, 2>(tfloat** d_fields, size_t elements);
	template tfloat* d_JoinInterleaved<tfloat, 3>(tfloat** d_fields, size_t elements);
	template tfloat* d_JoinInterleaved<tfloat, 4>(tfloat** d_fields, size_t elements);
	template tfloat* d_JoinInterleaved<tfloat, 5>(tfloat** d_fields, size_t elements);
	template tfloat* d_JoinInterleaved<tfloat, 6>(tfloat** d_fields, size_t elements);
	template int* d_JoinInterleaved<int, 2>(int** d_fields, size_t elements);
	template int* d_JoinInterleaved<int, 3>(int** d_fields, size_t elements);
	template int* d_JoinInterleaved<int, 4>(int** d_fields, size_t elements);
	template int* d_JoinInterleaved<int, 5>(int** d_fields, size_t elements);
	template int* d_JoinInterleaved<int, 6>(int** d_fields, size_t elements);

	template <class T, int fieldcount> void d_JoinInterleaved(T** d_fields, T* d_output, size_t elements)
	{
		size_t TpB = 256;
		size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
		dim3 grid = dim3((uint)totalblocks);
		JoinInterleavedKernel<T, fieldcount> << <grid, (uint)TpB >> > (d_fields, d_output, elements);
		cudaStreamQuery(0);
	}
	template void d_JoinInterleaved<tfloat, 2>(tfloat** d_fields, tfloat* d_output, size_t elements);
	template void d_JoinInterleaved<tfloat, 3>(tfloat** d_fields, tfloat* d_output, size_t elements);
	template void d_JoinInterleaved<tfloat, 4>(tfloat** d_fields, tfloat* d_output, size_t elements);
	template void d_JoinInterleaved<tfloat, 5>(tfloat** d_fields, tfloat* d_output, size_t elements);
	template void d_JoinInterleaved<tfloat, 6>(tfloat** d_fields, tfloat* d_output, size_t elements);
	template void d_JoinInterleaved<int, 2>(int** d_fields, int* d_output, size_t elements);
	template void d_JoinInterleaved<int, 3>(int** d_fields, int* d_output, size_t elements);
	template void d_JoinInterleaved<int, 4>(int** d_fields, int* d_output, size_t elements);
	template void d_JoinInterleaved<int, 5>(int** d_fields, int* d_output, size_t elements);
	template void d_JoinInterleaved<int, 6>(int** d_fields, int* d_output, size_t elements);


	void MixedToDeviceTfloat(void* h_input, tfloat* d_output, EM_DATATYPE datatype, size_t elements)
	{
		if (datatype == EM_BYTE)
			CudaMemcpyFromHostArrayConverted<unsigned char, tfloat>((unsigned char*)h_input, d_output, elements);
		else if (datatype == EM_SHORT)
			CudaMemcpyFromHostArrayConverted<short, tfloat>((short*)h_input, d_output, elements);
		else if (datatype == EM_LONG)
			CudaMemcpyFromHostArrayConverted<int, tfloat>((int*)h_input, d_output, elements);
		else if (datatype == EM_SINGLE)
			cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
		else if (datatype == EM_DOUBLE)
			CudaMemcpyFromHostArrayConverted<double, tfloat>((double*)h_input, d_output, elements);
		else
			throw;
	}

	void MixedToDeviceTfloat(void* h_input, tfloat* d_output, MRC_DATATYPE datatype, size_t elements)
	{
		if (datatype == MRC_BYTE)
			CudaMemcpyFromHostArrayConverted<unsigned char, tfloat>((unsigned char*)h_input, d_output, elements);
		else if (datatype == MRC_SHORT)
			CudaMemcpyFromHostArrayConverted<short, tfloat>((short*)h_input, d_output, elements);
		else if (datatype == MRC_UNSIGNEDSHORT)
			CudaMemcpyFromHostArrayConverted<unsigned short, tfloat>((unsigned short*)h_input, d_output, elements);
		else if (datatype == MRC_FLOAT)
			cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
		else
			throw;
	}

	tfloat* MixedToDeviceTfloat(void* h_input, EM_DATATYPE datatype, size_t elements)
	{
		tfloat* d_output;

		if (datatype == EM_BYTE)
			CudaMallocFromHostArrayConverted<unsigned char, tfloat>((unsigned char*)h_input, &d_output, elements);
		else if (datatype == EM_SHORT)
			CudaMallocFromHostArrayConverted<short, tfloat>((short*)h_input, &d_output, elements);
		else if (datatype == EM_LONG)
			CudaMallocFromHostArrayConverted<int, tfloat>((int*)h_input, &d_output, elements);
		else if (datatype == EM_SINGLE)
		{
			cudaMalloc((void**)&d_output, elements * sizeof(tfloat));
			cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
		}
		else if (datatype == EM_DOUBLE)
			CudaMallocFromHostArrayConverted<double, tfloat>((double*)h_input, &d_output, elements);
		else
			throw;

		return d_output;
	}

	tfloat* MixedToDeviceTfloat(void* h_input, MRC_DATATYPE datatype, size_t elements)
	{
		tfloat* d_output;

		if (datatype == MRC_BYTE)
			CudaMallocFromHostArrayConverted<unsigned char, tfloat>((unsigned char*)h_input, &d_output, elements);
		else if (datatype == MRC_SHORT)
			CudaMallocFromHostArrayConverted<short, tfloat>((short*)h_input, &d_output, elements);
		else if (datatype == MRC_UNSIGNEDSHORT)
			CudaMallocFromHostArrayConverted<unsigned short, tfloat>((unsigned short*)h_input, &d_output, elements);
		else if (datatype == MRC_FLOAT)
		{
			cudaMalloc((void**)&d_output, elements * sizeof(tfloat));
			cudaMemcpy(d_output, h_input, elements * sizeof(tfloat), cudaMemcpyHostToDevice);
		}
		else
			throw;

		return d_output;
	}

	int GetFileSize(std::string path)
	{
		std::ifstream inputfile(path, std::ios::in | std::ios::binary | std::ios::ate);
		int size = inputfile.tellg();
		inputfile.close();

		return size;
	}

	void* MallocFromBinaryFile(std::string path)
	{
		std::ifstream inputfile(path, std::ios::in | std::ios::binary | std::ios::ate);
		int size = inputfile.tellg();
		void* output = malloc(size);
		inputfile.seekg(0, std::ios::beg);
		inputfile.read((char*)output, size);
		inputfile.close();

		return output;
	}

	void* CudaMallocFromBinaryFile(std::string path)
	{
		void* h_array = MallocFromBinaryFile(path);
		void* d_array = CudaMallocFromHostArray(h_array, GetFileSize(path));
		free(h_array);

		return d_array;
	}

	void CudaWriteToBinaryFile(std::string path, void* d_data, size_t elements)
	{
		void* h_data = MallocFromDeviceArray(d_data, elements);
		WriteToBinaryFile(path, h_data, elements);

		free(h_data);
	}


	////////////////////
	//3D device memory//
	////////////////////

	cudaPitchedPtr CopyVolumeDeviceToDevice(tfloat* d_input, int3 dims)
	{
		cudaPitchedPtr deviceTo = { 0 };
		const cudaExtent extent = make_cudaExtent(dims.x * sizeof(tfloat), dims.y, dims.z);
		cudaMalloc3D(&deviceTo, extent);
		cudaMemcpy3DParms p = { 0 };
		p.srcPtr = make_cudaPitchedPtr((void*)d_input, dims.x * sizeof(tfloat), dims.x, dims.y);
		p.dstPtr = deviceTo;
		p.extent = extent;
		p.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&p);
		return deviceTo;
	}

	cudaPitchedPtr CopyVolumeHostToDevice(tfloat* h_input, int3 dims)
	{
		cudaPitchedPtr deviceTo = { 0 };
		const cudaExtent extent = make_cudaExtent(dims.x * sizeof(tfloat), dims.y, dims.z);
		cudaMalloc3D(&deviceTo, extent);
		cudaMemcpy3DParms p = { 0 };
		p.srcPtr = make_cudaPitchedPtr((void*)h_input, dims.x * sizeof(tfloat), dims.x, dims.y);
		p.dstPtr = deviceTo;
		p.extent = extent;
		p.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&p);
		return deviceTo;
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void MemcpyMultiKernel(T* d_output, T* d_input, size_t elements, int copies)
	{
		d_output += elements * copies * blockIdx.y;
		d_input += elements * blockIdx.y;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			T value = d_input[id];
			for (uint i = 0; i < copies; i++)
				d_output[i * elements + id] = value;
		}
	}

	template <class T> __global__ void MemcpyStridedKernel(T* d_output, T* d_input, size_t elements, int stridedst, int stridesrc)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
		{
			d_output[id * stridedst] = d_input[id * stridesrc];
		}
	}

	template <class T> __global__ void MemcpyBlockStridedKernel(T* d_output, T* d_input, size_t blockelements, int stridedst, int stridesrc, int nblocks)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < blockelements;
			id += blockDim.x * gridDim.x)
		{
			d_output[blockIdx.y * stridedst + id] = d_input[blockIdx.y * stridesrc + id];
		}
	}

	template <class T> __global__ void ValueFillKernel(T* d_output, size_t elements, T value)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
			d_output[id] = value;
	}

	template <class T, int fieldcount> __global__ void JoinInterleavedKernel(T** d_fields, T* d_output, size_t elements)
	{
		size_t startid = blockIdx.x * blockDim.x + threadIdx.x;
		int gridsize = blockDim.x * gridDim.x;

		//Roll out the first 5 iterations and put the rest into a for-loop since more than 5 fields are unlikely.
		//Maybe the compiler is smart enough to roll out the rest itself.
		if (fieldcount > 0)
		{
			T* d_field = d_fields[0];
			for (size_t id = startid;
				id < elements;
				id += gridsize)
				d_output[id * fieldcount] = d_field[id];
		}

		if (fieldcount > 1)
		{
			T* d_field = d_fields[1];
			for (size_t id = startid;
				id < elements;
				id += gridsize)
				d_output[id * fieldcount + 1] = d_field[id];
		}

		if (fieldcount > 2)
		{
			T* d_field = d_fields[2];
			for (size_t id = startid;
				id < elements;
				id += gridsize)
				d_output[id * fieldcount + 2] = d_field[id];
		}

		if (fieldcount > 3)
		{
			T* d_field = d_fields[3];
			for (size_t id = startid;
				id < elements;
				id += gridsize)
				d_output[id * fieldcount + 3] = d_field[id];
		}

		if (fieldcount > 4)
		{
			T* d_field = d_fields[4];
			for (size_t id = startid;
				id < elements;
				id += gridsize)
				d_output[id * fieldcount + 4] = d_field[id];
		}

		if (fieldcount > 5)
		{
			for (int f = 5; f < fieldcount; f++)
			{
				T* d_field = d_fields[f];
				for (size_t id = startid;
					id < elements;
					id += gridsize)
					d_output[id * fieldcount + f] = d_field[id];
			}
		}
	}

	template <class T1, class T2> __global__ void TypeConversionKernel(T1* d_input, T2* d_output, size_t elements)
	{
		for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
			id < elements;
			id += blockDim.x * gridDim.x)
			d_output[id] = (T2)d_input[id];
	}
}