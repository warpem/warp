#include "include/Functions.h"
using namespace gtom;

__declspec(dllexport) float* __stdcall MallocDevice(long long elements)
{
	float* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(float));

	return d_memory;
}

__declspec(dllexport) float* __stdcall MallocDeviceFromHost(float* h_data, long long elements)
{
	float* d_memory = (float*)gtom::CudaMallocFromHostArray(h_data, elements * sizeof(float));
	return d_memory;
}

__declspec(dllexport) int* __stdcall MallocDeviceFromHostInt(int* h_data, long long elements)
{
    int* d_memory = (int*)gtom::CudaMallocFromHostArray(h_data, elements * sizeof(int));
    return d_memory;
}

__declspec(dllexport) void* __stdcall MallocDeviceHalf(long long elements)
{
	half* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(half));

	return d_memory;
}

__declspec(dllexport) void* __stdcall MallocDeviceHalfFromHost(float* h_data, long long elements)
{
	half* d_memory;
	cudaMalloc((void**)&d_memory, elements * sizeof(half));

	CopyHostToDeviceHalf(h_data, d_memory, elements);

	return d_memory;
}

__declspec(dllexport) void __stdcall FreeDevice(void* d_data)
{
	cudaFree(d_data);
}

__declspec(dllexport) void __stdcall CopyDeviceToHost(float* d_source, float* h_dest, long long elements)
{
	cudaMemcpy(h_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToHost);
}

__declspec(dllexport) void __stdcall CopyDeviceToHostPinned(float* d_source, float* hp_dest, long long elements)
{
    cudaMemcpy(hp_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToHost);
}

__declspec(dllexport) void __stdcall CopyDeviceHalfToHost(half* d_source, float* h_dest, long long elements)
{
	float* d_source32;
	cudaMallocHost((void**)&d_source32, elements * sizeof(float));

	gtom::d_ConvertToTFloat(d_source, d_source32, elements);
	cudaMemcpy(h_dest, d_source32, elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_source32);
}

__declspec(dllexport) void __stdcall CopyDeviceToDevice(float* d_source, float* d_dest, long long elements)
{
	cudaMemcpy(d_dest, d_source, elements * sizeof(float), cudaMemcpyDeviceToDevice);
}

__declspec(dllexport) void __stdcall CopyDeviceHalfToDeviceHalf(half* d_source, half* d_dest, long long elements)
{
	cudaMemcpy(d_dest, d_source, elements * sizeof(half), cudaMemcpyDeviceToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToDevice(float* h_source, float* d_dest, long long elements)
{
	cudaMemcpy(d_dest, h_source, elements * sizeof(float), cudaMemcpyHostToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToHost(float* h_source, float* h_dest, long long elements)
{
	memcpy(h_dest, h_source, elements * sizeof(float));
}

__declspec(dllexport) void __stdcall CopyHostPinnedToDevice(float* hp_source, float* d_dest, long long elements)
{
    cudaMemcpy(d_dest, hp_source, elements * sizeof(float), cudaMemcpyHostToDevice);
}

__declspec(dllexport) void __stdcall CopyHostToDeviceHalf(float* h_source, half* d_dest, long long elements)
{
	float* d_source = (float*)gtom::CudaMallocFromHostArray(h_source, elements * sizeof(float));

	gtom::d_ConvertTFloatTo(d_source, d_dest, elements);

	cudaFree(d_source);
}

__declspec(dllexport) void __stdcall SingleToHalf(float* d_source, half* d_dest, long long elements)
{
	gtom::d_ConvertTFloatTo(d_source, d_dest, elements);
}

__declspec(dllexport) void __stdcall HalfToSingle(half* d_source, float* d_dest, long long elements)
{
	gtom::d_ConvertToTFloat(d_source, d_dest, elements);
}

__declspec(dllexport) float* __stdcall MallocHostPinned(long long elements)
{
    float* hp_memory;
    cudaMallocHost((void**)&hp_memory, elements * sizeof(float));

    return hp_memory;
}

__declspec(dllexport) void __stdcall FreeHostPinned(void* hp_data)
{
    cudaFreeHost(hp_data);
}

__declspec(dllexport) void* __stdcall HostMalloc(long long elements)
{
	return malloc(elements * sizeof(float));
}

__declspec(dllexport) void __stdcall HostFree(void* h_pointer)
{
    free(h_pointer);
}

__declspec(dllexport) cudaArray_t MallocArray(int2 dims)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaArray_t a_input;
	cudaMallocArray(&a_input, &desc, dims.x, dims.y);

	return a_input;
}

__declspec(dllexport) void CopyDeviceToArray(float* d_input, cudaArray_t a_output, int2 dims)
{
	cudaMemcpyToArray(a_output, 0, 0, d_input, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice);
}

__declspec(dllexport) void FreeArray(cudaArray_t a_input)
{
	cudaFreeArray(a_input);
}