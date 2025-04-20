#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Correlation.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<int nsamples> __global__ void SincMax1DKernel(tfloat* d_input, tfloat2* d_output);
	template<int ndims> __global__ void SincEvalMaxKernel(tfloat* d_input, uint nsamples, tfloat3* d_positions, tfloat* d_values);

	__global__ void PeakOne2DKernel(tfloat* d_input, float3* d_positions, tfloat* d_values, int2 dims, int2 dimsregion, bool subtractcenter);


	///////////////////////////////////////
	//Equivalent of TOM's tom_peak method//
	///////////////////////////////////////

	void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, cufftHandle* planforw, cufftHandle* planback, int batch)
	{
		tuple2<tfloat, size_t>* d_integerindices;
		cudaMalloc((void**)&d_integerindices, batch * sizeof(tuple2<tfloat, size_t>));

		if (batch <= 1)
			d_Max(d_input, d_integerindices, Elements(dims), batch);
		else
			d_MaxMonolithic(d_input, d_integerindices, Elements(dims), batch);
		tuple2<tfloat, size_t>* h_integerindices = (tuple2<tfloat, size_t>*)MallocFromDeviceArray(d_integerindices, batch * sizeof(tuple2<tfloat, size_t>));

		tfloat3* h_positions = (tfloat3*)malloc(batch * sizeof(tfloat3));
		tfloat* h_values = (tfloat*)malloc(batch * sizeof(tfloat));

		for (int b = 0; b < batch; b++)
		{
			size_t index = h_integerindices[b].t2;
			size_t z = index / (dims.x * dims.y);
			index -= z * (dims.x * dims.y);
			size_t y = index / dims.x;
			index -= y * dims.x;

			h_positions[b] = tfloat3((tfloat)index, (tfloat)y, (tfloat)z);
			h_values[b] = h_integerindices[b].t1;
		}
		if (mode == T_PEAK_INTEGER)
		{
			cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
		}
		else if (mode == T_PEAK_SUBCOARSE)
		{
			int ndims = DimensionCount(dims);
			int nsamples = min(ndims == 3 ? 10 : 20, NextMultipleOf(dims.x, 2));
			int3 dimspeak = toInt3(nsamples, 1, 1);
			if (ndims > 1)
				dimspeak.y = nsamples;
			if (ndims > 2)
				dimspeak.z = nsamples;

			int3* h_origins = (int3*)malloc(ndims * batch * sizeof(int3));
			for (int b = 0; b < batch; b++)
			{
				h_origins[b] = toInt3((int)h_positions[b].x - nsamples / 2, (int)h_positions[b].y, (int)h_positions[b].z);
				if (ndims > 1)
					h_origins[b + batch] = toInt3((int)h_positions[b].x, (int)h_positions[b].y - nsamples / 2, (int)h_positions[b].z);
				if (ndims > 2)
					h_origins[b + batch * 2] = toInt3((int)h_positions[b].x, (int)h_positions[b].y, (int)h_positions[b].z - nsamples / 2);
			}
			int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, ndims * batch * sizeof(int3));
			tfloat* d_extracts;
			cudaMalloc((void**)&d_extracts, batch * max(Elements(dimspeak), ndims * nsamples) * sizeof(tfloat));
			tfloat2* d_peaks;
			cudaMalloc((void**)&d_peaks, ndims * batch * sizeof(tfloat2));

			d_Extract(d_input, d_extracts, dims, toInt3(nsamples, 1, 1), d_origins, batch);
			if (ndims > 1)
				d_Extract(d_input, d_extracts + batch * nsamples, dims, toInt3(1, nsamples, 1), d_origins + batch, batch);
			if (ndims > 2)
				d_Extract(d_input, d_extracts + 2 * batch * nsamples, dims, toInt3(1, 1, nsamples), d_origins + 2 * batch, batch);

			// sinc-interpolate along each dimension to find sub-pixel position
			dim3 TpB = 256;
			dim3 grid = ndims * batch;
			if (nsamples == 20)
				SincMax1DKernel<20> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 18)
				SincMax1DKernel<18> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 16)
				SincMax1DKernel<16> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 14)
				SincMax1DKernel<14> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 12)
				SincMax1DKernel<12> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 10)
				SincMax1DKernel<10> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 8)
				SincMax1DKernel<8> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 6)
				SincMax1DKernel<6> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 4)
				SincMax1DKernel<4> << <grid, TpB >> > (d_extracts, d_peaks);
			if (nsamples == 2)
				SincMax1DKernel<2> << <grid, TpB >> > (d_extracts, d_peaks);

			for (int b = 0; b < batch; b++)
				h_origins[b] = toInt3((int)h_positions[b].x - dimspeak.x / 2, (int)h_positions[b].y - dimspeak.y / 2, (int)h_positions[b].z - dimspeak.z / 2);
			cudaMemcpy(d_origins, h_origins, batch * sizeof(int3), cudaMemcpyHostToDevice);
			d_Extract(d_input, d_extracts, dims, dimspeak, d_origins, batch);

			tfloat2* h_peaks = (tfloat2*)MallocFromDeviceArray(d_peaks, ndims * batch * sizeof(tfloat2));
			tfloat3* h_evalpositions = (tfloat3*)malloc(batch * sizeof(tfloat3));
			for (int b = 0; b < batch; b++)
			{
				h_positions[b].x += h_peaks[b].x;
				h_evalpositions[b] = tfloat3((tfloat)(nsamples / 2) + h_peaks[b].x, (tfloat)0, (tfloat)0);
				if (ndims > 1)
				{
					h_positions[b].y += h_peaks[b + batch].x;
					h_evalpositions[b].y = (tfloat)(nsamples / 2) + h_peaks[b + batch].x;
				}
				if (ndims > 2)
				{
					h_positions[b].z += h_peaks[b + 2 * batch].x;
					h_evalpositions[b].z = (tfloat)(nsamples / 2) + h_peaks[b + 2 * batch].x;
				}
			}
			cudaMemcpy(d_origins, h_evalpositions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
			free(h_evalpositions);

			// sinc-interpolate at sub-pixel position to determine its value
			grid = batch;
			if (ndims == 1)
				SincEvalMaxKernel<1> << <grid, TpB >> > (d_extracts, nsamples, (tfloat3*)d_origins, (tfloat*)d_peaks);
			else if (ndims == 2)
				SincEvalMaxKernel<2> << <grid, TpB >> > (d_extracts, nsamples, (tfloat3*)d_origins, (tfloat*)d_peaks);
			if (ndims == 3)
				SincEvalMaxKernel<3> << <grid, TpB >> > (d_extracts, nsamples, (tfloat3*)d_origins, (tfloat*)d_peaks);

			cudaMemcpy(h_values, d_peaks, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

			free(h_peaks);
			free(h_origins);
			cudaFree(d_peaks);
			cudaFree(d_extracts);
			cudaFree(d_origins);

			cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
		}
		else if (mode == T_PEAK_SUBFINE)
		{
			int samples = DimensionCount(dims) < 3 ? 9 : 5;	//Region around the peak to be extracted
			for (int i = 0; i < DimensionCount(dims); i++)	//Samples shouldn't be bigger than smallest relevant dimension
				samples = min(samples, ((int*)&dims)[i]);
			int subdivisions = DimensionCount(dims) < 3 ? 105 : 63;		//Theoretical precision is 1/subdivisions; scaling 3D map is more expensive, thus less precision there
			int centerindex = samples / 2 * subdivisions;	//Where the peak is within the extracted, up-scaled region

			tfloat* d_original;
			cudaMalloc((void**)&d_original, pow(samples, DimensionCount(dims)) * sizeof(tfloat));
			tfloat* d_interpolated;
			cudaMalloc((void**)&d_interpolated, pow(samples * subdivisions, DimensionCount(dims)) * sizeof(tfloat));
			tuple2<tfloat, size_t>* d_maxtuple;
			cudaMalloc((void**)&d_maxtuple, sizeof(tuple2<tfloat, size_t>));
			tuple2<tfloat, size_t>* h_maxtuple = (tuple2<tfloat, size_t>*)malloc(sizeof(tuple2<tfloat, size_t>));

			for (int b = 0; b < batch; b++)
			{
				int3 coarseposition = toInt3((int)h_positions[b].x, (int)h_positions[b].y, (int)h_positions[b].z);

				d_Extract(d_input + Elements(dims) * b, d_original, dims, toInt3(samples, min(dims.y, samples), min(dims.z, samples)), coarseposition);
				d_Scale(d_original,
					d_interpolated,
					toInt3(samples, min(dims.y, samples), min(dims.z, samples)),
					toInt3(samples * subdivisions, dims.y == 1 ? 1 : samples * subdivisions, dims.z == 1 ? 1 : samples * subdivisions),
					T_INTERP_FOURIER,
					planforw,
					planback);
				d_Max(d_interpolated, d_maxtuple, pow(samples * subdivisions, DimensionCount(dims)));
				cudaMemcpy(h_maxtuple, d_maxtuple, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);

				h_values[b] = max(h_values[b], (*h_maxtuple).t1);

				size_t index = (*h_maxtuple).t2;
				size_t z = index / (samples * samples * subdivisions * subdivisions);
				index -= z * (samples * samples * subdivisions * subdivisions);
				size_t y = index / (samples * subdivisions);
				index -= y * (samples * subdivisions);

				h_positions[b].x += (tfloat)((int)index - centerindex) / (tfloat)subdivisions;
				if (dims.y > 1)
					h_positions[b].y += (tfloat)((int)y - centerindex) / (tfloat)subdivisions;
				if (dims.z > 1)
					h_positions[b].z += (tfloat)((int)z - centerindex) / (tfloat)subdivisions;
			}

			cudaFree(d_original);
			cudaFree(d_interpolated);
			cudaFree(d_maxtuple);
			free(h_maxtuple);

			cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
			cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
		}


		free(h_integerindices);
		free(h_positions);
		free(h_values);
		cudaFree(d_integerindices);
	}

	void d_PeakOne2D(tfloat* d_input, float3* d_positions, tfloat* d_values, int2 dims, int2 dimsregion, bool subtractcenter, int batch)
	{
		dim3 TpB = tmin(128, Elements2(dimsregion));
		dim3 grid = batch;

		PeakOne2DKernel << <grid, TpB >> > (d_input, d_positions, d_values, dims, dimsregion, subtractcenter);
	}

	////////////////
	//CUDA kernels//
	////////////////

	template <int nsamples> __global__ void SincMax1DKernel(tfloat* d_input, tfloat2* d_output)
	{
		d_input += nsamples * blockIdx.x;
		d_output += blockIdx.x;

		__shared__ tfloat s_p[nsamples];
		__shared__ tfloat2 s_best[256];

		if (threadIdx.x < nsamples)
			s_p[threadIdx.x] = d_input[threadIdx.x];
		__syncthreads();

		tfloat2 best = tfloat2(0, -1e30);
		for (int id = threadIdx.x; id < 1024; id += 256)
		{
			double x = (double)id / 512.0 - 1.0;
			double y = 0;
			if (nsamples > 18)
			{
				y += sinc(-9.0 - x) * s_p[nsamples / 2 - 9];
				y += sinc(9.0 - x) * s_p[nsamples / 2 + 9];
			}
			if (nsamples > 16)
			{
				y += sinc(-8.0 - x) * s_p[nsamples / 2 - 8];
				y += sinc(8.0 - x) * s_p[nsamples / 2 + 8];
			}
			if (nsamples > 14)
			{
				y += sinc(-7.0 - x) * s_p[nsamples / 2 - 7];
				y += sinc(7.0 - x) * s_p[nsamples / 2 + 7];
			}
			if (nsamples > 12)
			{
				y += sinc(-6.0 - x) * s_p[nsamples / 2 - 6];
				y += sinc(6.0 - x) * s_p[nsamples / 2 + 6];
			}
			if (nsamples > 10)
			{
				y += sinc(-5.0 - x) * s_p[nsamples / 2 - 5];
				y += sinc(5.0 - x) * s_p[nsamples / 2 + 5];
			}
			if (nsamples > 8)
			{
				y += sinc(-4.0 - x) * s_p[nsamples / 2 - 4];
				y += sinc(4.0 - x) * s_p[nsamples / 2 + 4];
			}
			if (nsamples > 6)
			{
				y += sinc(-3.0 - x) * s_p[nsamples / 2 - 3];
				y += sinc(3.0 - x) * s_p[nsamples / 2 + 3];
			}
			if (nsamples > 4)
			{
				y += sinc(-2.0 - x) * s_p[nsamples / 2 - 2];
				y += sinc(2.0 - x) * s_p[nsamples / 2 + 2];
			}
			if (nsamples > 2)
			{
				y += sinc(-1.0 - x) * s_p[nsamples / 2 - 1];
				y += sinc(1.0 - x) * s_p[nsamples / 2 + 1];
			}
			y += sinc(x) * s_p[nsamples / 2];

			if (best.y < y)
				best = tfloat2(x, y);
		}
		s_best[threadIdx.x] = best;
		__syncthreads();

		if (threadIdx.x < 128)
			if (s_best[threadIdx.x + 128].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 128];
		__syncthreads();
		if (threadIdx.x < 64)
			if (s_best[threadIdx.x + 64].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 64];
		__syncthreads();
		if (threadIdx.x < 32)
			if (s_best[threadIdx.x + 32].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 32];
		__syncthreads();
		if (threadIdx.x < 16)
			if (s_best[threadIdx.x + 16].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 16];
		__syncthreads();
		if (threadIdx.x < 8)
			if (s_best[threadIdx.x + 8].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 8];
		__syncthreads();
		if (threadIdx.x < 4)
			if (s_best[threadIdx.x + 4].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 4];
		__syncthreads();
		if (threadIdx.x < 2)
			if (s_best[threadIdx.x + 2].y > s_best[threadIdx.x].y)
				s_best[threadIdx.x] = s_best[threadIdx.x + 2];
		__syncthreads();
		if (threadIdx.x == 0)
		{
			if (s_best[threadIdx.x + 1].y > s_best[0].y)
				s_best[0] = s_best[threadIdx.x + 1];
			*d_output = s_best[0];
		}
	}

	template<int ndims> __global__ void SincEvalMaxKernel(tfloat* d_input, uint nsamples, tfloat3* d_positions, tfloat* d_values)
	{
		__shared__ tfloat s_sum[256];

		int elements;
		if (ndims == 3)
			elements = nsamples * nsamples * nsamples;
		else if (ndims == 2)
			elements = nsamples * nsamples;
		else if (ndims == 1)
			elements = nsamples;
		d_input += blockIdx.x * elements;
		d_values += blockIdx.x;

		tfloat3 position = d_positions[blockIdx.x];

		tfloat sum = 0;
		uint nsamples2 = nsamples * nsamples;
		for (uint idx = threadIdx.x; idx < elements; idx += blockDim.x)
		{
			uint x = idx % nsamples, y = 0, z = 0;
			if (ndims > 1)
				y = (idx / nsamples) % nsamples;
			if (ndims > 2)
				z = idx / nsamples2;

			float weight = sinc((float)x - position.x);
			if (ndims > 1)
				weight *= sinc((float)y - position.y);
			if (ndims > 2)
				weight *= sinc((float)z - position.z);

			sum += d_input[idx] * (tfloat)weight;
		}
		s_sum[threadIdx.x] = sum;
		__syncthreads();

		if (threadIdx.x < 128)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 128];
		__syncthreads();
		if (threadIdx.x < 64)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 64];
		__syncthreads();
		if (threadIdx.x < 32)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 32];
		__syncthreads();
		if (threadIdx.x < 16)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 16];
		__syncthreads();
		if (threadIdx.x < 8)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 8];
		__syncthreads();
		if (threadIdx.x < 4)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 4];
		__syncthreads();
		if (threadIdx.x < 2)
			s_sum[threadIdx.x] += s_sum[threadIdx.x + 2];
		__syncthreads();
		if (threadIdx.x == 0)
		{
			*d_values = s_sum[0] + s_sum[1];
		}
	}

	__global__ void PeakOne2DKernel(tfloat* d_input, float3* d_positions, tfloat* d_values, int2 dims, int2 dimsregion, bool subtractcenter)
	{
		__shared__ tfloat s_maxval[128];
		__shared__ int2 s_maxpos[128];

		d_input += Elements2(dims) * blockIdx.x;

		int2 regionoffset = dims - dimsregion;
		regionoffset.x /= 2;
		regionoffset.y /= 2;

		uint elements = Elements2(dimsregion);
		tfloat maxval = -1e20;
		int2 maxpos = make_int2(0, 0);

		for (uint id = threadIdx.x; id < elements; id += blockDim.x)
		{
			uint y = id / dimsregion.x;
			uint x = id - y * dimsregion.x;

			x += regionoffset.x;
			y += regionoffset.y;

			tfloat val = d_input[y * dims.x + x];
			if (val > maxval)
			{
				maxval = val;
				maxpos = make_int2((int)x, (int)y);
			}
		}

		s_maxval[threadIdx.x] = maxval;
		s_maxpos[threadIdx.x] = maxpos;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			for (int i = 1; i < blockDim.x; i++)
			{
				if (s_maxval[i] > maxval)
				{
					maxval = s_maxval[i];
					maxpos = s_maxpos[i];
				}
			}

			float2 weightedcenter = make_float2(0, 0);
			float sumweights = 0;

			for (int y = -1; y <= 1; y++)
			{
				if (maxpos.y + y >= 0 && maxpos.y + y < dims.y)
				{
					for (int x = -1; x <= 1; x++)
					{
						if (maxpos.x + x >= 0 && maxpos.x + x < dims.x)
						{
							float val = (float)d_input[(maxpos.y + y) * dims.x + maxpos.x + x];
							weightedcenter.x += val * x;
							weightedcenter.y += val * y;
							sumweights += val;
						}
					}
				}
			}

			if (sumweights != 0)
				weightedcenter /= sumweights;

			float2 finalpos = make_float2(maxpos.x, maxpos.y) + weightedcenter;
			if (subtractcenter)
			{
				finalpos.x -= dims.x / 2;
				finalpos.y -= dims.y / 2;
			}

			d_positions[blockIdx.x] = make_float3(finalpos.x, finalpos.y, 0);
			d_values[blockIdx.x] = maxval;
		}
	}
}