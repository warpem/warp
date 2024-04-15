#include "gtom/include/Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<int mode, int ndims> __global__ void WindowMaskKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat radius, tfloat3 center, int batch);
	template<int mode, int ndims> __global__ void WindowMaskBorderDistanceKernel(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch);


	////////////////
	//Host methods//
	////////////////

	void d_HannMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch)
	{
		tfloat _radius = radius != NULL ? *radius : tmin(dims.z > 1 ? tmin(dims.x, dims.z) : dims.x, dims.y) / 2 - 1;
		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = tmin(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, 1);
		if (DimensionCount(dims) == 1)
			WindowMaskKernel<0, 1> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
		if (DimensionCount(dims) == 2)
			WindowMaskKernel<0, 2> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
		if (DimensionCount(dims) == 3)
			WindowMaskKernel<0, 3> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
	}

	void d_HammingMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch)
	{
		tfloat _radius = radius != NULL ? *radius : tmin(dims.z > 1 ? tmin(dims.x, dims.z) : dims.x, dims.y) / 2 - 1;
		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = tmin(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, 1);
		if (DimensionCount(dims) == 1)
			WindowMaskKernel<1, 1> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
		if (DimensionCount(dims) == 2)
			WindowMaskKernel<1, 2> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
		if (DimensionCount(dims) == 3)
			WindowMaskKernel<1, 3> << <grid, TpB >> > (d_input, d_output, dims, _radius, _center, batch);
	}

	void d_GaussianMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* sigma, tfloat3* center, int batch)
	{
		tfloat _sigma = sigma != NULL ? *sigma : (tfloat)1;
		tfloat3 _center = center != NULL ? *center : tfloat3(dims.x / 2, dims.y / 2, dims.z / 2);

		int TpB = tmin(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, 1);
		if (DimensionCount(dims) == 1)
			WindowMaskKernel<2, 1> << <grid, TpB >> > (d_input, d_output, dims, (tfloat)2 * _sigma * _sigma, _center, batch);
		if (DimensionCount(dims) == 2)
			WindowMaskKernel<2, 2> << <grid, TpB >> > (d_input, d_output, dims, (tfloat)2 * _sigma * _sigma, _center, batch);
		if (DimensionCount(dims) == 3)
			WindowMaskKernel<2, 3> << <grid, TpB >> > (d_input, d_output, dims, (tfloat)2 * _sigma * _sigma, _center, batch);
	}

	void d_HannMaskBorderDistance(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch)
	{
		int TpB = tmin(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, 1);
		if (DimensionCount(dims) == 1)
			WindowMaskBorderDistanceKernel<0, 1> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
		if (DimensionCount(dims) == 2)
			WindowMaskBorderDistanceKernel<0, 2> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
		if (DimensionCount(dims) == 3)
			WindowMaskBorderDistanceKernel<0, 3> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
	}

	void d_HammingMaskBorderDistance(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch)
	{
		int TpB = tmin(NextMultipleOf(dims.x, 32), 256);
		dim3 grid = dim3(dims.y, dims.z, 1);
		if (DimensionCount(dims) == 1)
			WindowMaskBorderDistanceKernel<1, 1> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
		if (DimensionCount(dims) == 2)
			WindowMaskBorderDistanceKernel<1, 2> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
		if (DimensionCount(dims) == 3)
			WindowMaskBorderDistanceKernel<1, 3> << <grid, TpB >> > (d_input, d_output, dims, falloff, batch);
	}

	////////////////
	//CUDA kernels//
	////////////////

	template<int mode, int ndims> __global__ void WindowMaskKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat radius, tfloat3 center, int batch)
	{
		tfloat xsq, ysq, zsq, length;

		if (ndims > 1)
		{
			ysq = (tfloat)blockIdx.x - center.y;
			ysq *= ysq;
		}
		else
			ysq = 0;

		if (ndims > 2)
		{
			zsq = (tfloat)blockIdx.y - center.z;
			zsq *= zsq;
		}
		else
			zsq = 0;

		for (int x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			xsq = (tfloat)x - center.x;
			xsq *= xsq;

			length = sqrt(xsq + ysq + zsq);

			tfloat val = 0;
			//Hann
			if (mode == 0)
				val = (tfloat)0.5 * ((tfloat)1 + cos(min(length / radius, (tfloat)1) * PI));
			//Hamming
			else if (mode == 1)
				val = (tfloat)0.54 - (tfloat)0.46 * cos(((tfloat)1 - min(length / radius, (tfloat)1)) * PI);
			//Gaussian
			else if (mode == 2)
				val = exp(-(pow(length, (tfloat)2) / radius));

			for (int b = 0; b < batch; b++)
			{
				if (ndims > 2)
					d_output[Elements(dims) * b + (blockIdx.y * dims.y + blockIdx.x) * dims.x + x] = val * d_input[Elements(dims) * b + (blockIdx.y * dims.y + blockIdx.x) * dims.x + x];
				else
					d_output[Elements(dims) * b + blockIdx.x * dims.x + x] = val * d_input[Elements(dims) * b + blockIdx.x * dims.x + x];
			}
		}
	}

	template<int mode, int ndims> __global__ void WindowMaskBorderDistanceKernel(tfloat* d_input, tfloat* d_output, int3 dims, int falloff, int batch)
	{
		int distx = 0, disty = 0, distz = 0;

		if (ndims > 1)
		{
			int y = blockIdx.x;
			int fromtop = max(0, falloff - y);
			int frombottom = max(0, falloff - (dims.y - 1 - y));
			disty = max(fromtop, frombottom);
		}

		if (ndims > 2)
		{
			int z = blockIdx.y;
			int fromback = max(0, falloff - z);
			int fromfront = max(0, falloff - (dims.z - 1 - z));
			distz = max(fromback, fromfront);
		}

		for (int idx = threadIdx.x; idx < dims.x; idx += blockDim.x)
		{
			int fromleft = max(0, falloff - idx);
			int fromright = max(0, falloff - (dims.x - 1 - idx));
			distx = max(fromleft, fromright);

			float dist;
			if (ndims == 3)
				dist = sqrt(float(distx * distx + disty * disty + distz * distz));
			else if (ndims == 2)
				dist = sqrt(float(distx * distx + disty * disty));
			else
				dist = (float)distx;


			tfloat val = 0;
			//Hann
			if (mode == 0)
			{
				val = 0.5f * (1.0f + cos(min(dist / (float)falloff, 1.0f) * PI));
			}
			//Hamming
			else if (mode == 1)
			{
				val = 0.54f - 0.46f * cos((1.0f - min(dist / (float)falloff, 1.0f)) * PI);
			}

			for (int b = 0; b < batch; b++)
			{
				if (ndims == 3)
					d_output[Elements(dims) * b + (blockIdx.y * dims.y + blockIdx.x) * dims.x + idx] = val * d_input[Elements(dims) * b + (blockIdx.y * dims.y + blockIdx.x) * dims.x + idx];
				else
					d_output[Elements(dims) * b + blockIdx.x * dims.x + idx] = val * d_input[Elements(dims) * b + blockIdx.x * dims.x + idx];
			}
		}
	}
}