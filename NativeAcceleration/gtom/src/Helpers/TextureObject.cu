#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"

namespace gtom
{
	cudaArray_t d_MallocArray(int2 dims)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaArray_t a_input;
		cudaMallocArray(&a_input, &desc, dims.x, dims.y);

		return a_input;
	}

	void d_MemcpyToArray(tfloat* d_input, cudaArray_t a_output, int2 dims)
	{
		cudaMemcpyToArray(a_output, 0, 0, d_input, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	}

	void d_BindTextureToArray(tfloat* d_input, cudaArray_t &createdarray, cudaTex &createdtexture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaArray* a_input;
		cudaMallocArray(&a_input, &desc, dims.x, dims.y);
		cudaMemcpyToArray(a_input, 0, 0, d_input, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = a_input;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = filtermode;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = normalizedcoords;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.addressMode[2] = cudaAddressModeWrap;
		cudaTex texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		createdarray = a_input;
		createdtexture = texObj;
	}

	void d_BindTextureToArray(cudaArray_t a_input, cudaTex& createdtexture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
	{
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = a_input;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = filtermode;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = normalizedcoords;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.addressMode[2] = cudaAddressModeWrap;
		cudaTex texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		createdtexture = texObj;
	}

	void d_BindTextureToArray(tfloat* d_input, cudaArray_t* &h_createdarrays, cudaTex* &h_createdtextures, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords, int nimages)
	{
		for (int n = 0; n < nimages; n++)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
			cudaArray* a_input;
			cudaMallocArray(&a_input, &desc, dims.x, dims.y);
			cudaMemcpyToArray(a_input, 0, 0, d_input + Elements2(dims) * n, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);

			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = a_input;

			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.filterMode = filtermode;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = normalizedcoords;
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.addressMode[2] = cudaAddressModeWrap;
			cudaTex texObj = 0;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

			h_createdarrays[n] = a_input;
			h_createdtextures[n] = texObj;
		}
	}

	void d_BindTextureTo3DArray(tfloat* d_input, cudaArray_t &createdarray, cudaTex &createdtexture, int3 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaArray* a_input;
		cudaMalloc3DArray(&a_input, &desc, make_cudaExtent(dims.x, dims.y, dims.z));

		cudaMemcpy3DParms p = { 0 };
		p.extent = make_cudaExtent(dims.x, dims.y, dims.z);
		p.srcPtr = make_cudaPitchedPtr(d_input, dims.x * sizeof(tfloat), dims.x, dims.y);
		p.dstArray = a_input;
		p.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&p);

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = a_input;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.filterMode = filtermode;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = normalizedcoords;
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.addressMode[2] = cudaAddressModeWrap;
		cudaTex texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

		createdarray = a_input;
		createdtexture = texObj;
	}

	void d_BindTextureTo3DArray(tfloat* d_input, cudaArray_t* &h_createdarrays, cudaTex* &h_createdtextures, int3 dims, cudaTextureFilterMode filtermode, bool normalizedcoords, int nvolumes)
	{
		for (int n = 0; n < nvolumes; n++)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
			cudaArray* a_input;
			cudaMalloc3DArray(&a_input, &desc, make_cudaExtent(dims.x, dims.y, dims.z));

			cudaMemcpy3DParms p = { 0 };
			p.extent = make_cudaExtent(dims.x, dims.y, dims.z);
			p.srcPtr = make_cudaPitchedPtr(d_input + Elements(dims) * n, dims.x * sizeof(tfloat), dims.x, dims.y);
			p.dstArray = a_input;
			p.kind = cudaMemcpyDeviceToDevice;
			cudaMemcpy3D(&p);

			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = a_input;

			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			texDesc.filterMode = filtermode;
			texDesc.readMode = cudaReadModeElementType;
			texDesc.normalizedCoords = normalizedcoords;
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.addressMode[2] = cudaAddressModeWrap;
			cudaTex texObj = 0;
			cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

			h_createdarrays[n] = a_input;
			h_createdtextures[n] = texObj;
		}
	}
}