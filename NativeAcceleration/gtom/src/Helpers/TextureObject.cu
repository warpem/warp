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

	// Helper function to replace deprecated cudaMemcpyToArray with cudaMemcpy3D
	void d_CopyToArray(void* d_src, cudaArray_t dst, size_t width, size_t height, size_t depth, size_t elemSize, cudaMemcpyKind kind)
	{
		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr = make_cudaPitchedPtr(d_src, width * elemSize, width, height);
		copyParams.dstArray = dst;
		copyParams.extent = make_cudaExtent(width, height, depth);
		copyParams.kind = kind;
		cudaMemcpy3D(&copyParams);
	}
	
	// Helper function to create a texture object from a CUDA array
	cudaTex d_CreateTextureObject(cudaArray_t array, 
								  cudaTextureFilterMode filterMode, 
								  cudaTextureReadMode readMode, 
								  bool normalizedCoords,
								  cudaTextureAddressMode addressMode = cudaAddressModeWrap)
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;
		
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.filterMode = filterMode;
		texDesc.readMode = readMode;
		texDesc.normalizedCoords = normalizedCoords;
		texDesc.addressMode[0] = addressMode;
		texDesc.addressMode[1] = addressMode;
		texDesc.addressMode[2] = addressMode;
		
		cudaTex texObj = 0;
		cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
		return texObj;
	}

	void d_MemcpyToArray(tfloat* d_input, cudaArray_t a_output, int2 dims)
	{
		// Use the helper function
		d_CopyToArray(d_input, a_output, dims.x, dims.y, 1, sizeof(tfloat), cudaMemcpyDeviceToDevice);
	}

	void d_BindTextureToArray(tfloat* d_input, cudaArray_t &createdarray, cudaTex &createdtexture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaArray* a_input;
		cudaMallocArray(&a_input, &desc, dims.x, dims.y);
		
		// Use our helper function to copy to array
		d_CopyToArray(d_input, a_input, dims.x, dims.y, 1, sizeof(tfloat), cudaMemcpyDeviceToDevice);
		
		// Use our helper function to create texture object
		cudaTex texObj = d_CreateTextureObject(a_input, filtermode, cudaReadModeElementType, normalizedcoords);

		createdarray = a_input;
		createdtexture = texObj;
	}

	void d_BindTextureToArray(cudaArray_t a_input, cudaTex& createdtexture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
	{
		// Use our helper function to create texture object
		createdtexture = d_CreateTextureObject(a_input, filtermode, cudaReadModeElementType, normalizedcoords);
	}

	void d_BindTextureToArray(tfloat* d_input, cudaArray_t* &h_createdarrays, cudaTex* &h_createdtextures, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords, int nimages)
	{
		for (int n = 0; n < nimages; n++)
		{
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
			cudaArray* a_input;
			cudaMallocArray(&a_input, &desc, dims.x, dims.y);
			
			// Use our helper function to copy to array
			d_CopyToArray(d_input + Elements2(dims) * n, a_input, dims.x, dims.y, 1, sizeof(tfloat), cudaMemcpyDeviceToDevice);
			
			// Use our helper function to create texture object
			cudaTex texObj = d_CreateTextureObject(a_input, filtermode, cudaReadModeElementType, normalizedcoords);

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