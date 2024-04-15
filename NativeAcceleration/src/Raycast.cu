#include "include/Functions.h"
#include "gtom/include/CubicInterp.cuh"
using namespace gtom;

template <bool cubicinterp> __global__ void RaycastKernel(cudaTex t_intensities, 
															int sizevolume, 
															float surfacethreshold, 
															int2 dimsimage, 
															float3 camera, 
															float3 pixelx, 
															float3 pixely, 
															float3 view, 
															float3* d_intersectionpoints, 
															float3* d_intersectionnormals, 
															char* d_hittest, 
															float* d_accumulated);

__global__ void RaycastShadingKernel(int2 dimsimage, 
									int sizevolume, 
									cudaTex t_coloring, 
									int sizecoloring, 
									char* d_hittest, 
									float* d_accumulated,
									float3 view,
									float3* d_intersectionpoints, 
									float3* d_intersectionnormals, 
									float3* d_colormap,
									int ncolors,
									float2 colorrange,
									float2 shadingrange,
									float3 intensitycolor,
									float2 intensityrange,
									uchar4* d_bgra);


__declspec(dllexport) void RenderVolume(cudaTex t_intensities,
										int3 dimsvolume,
										float surfacethreshold,
										cudaTex t_coloring,
										int3 dimscoloring,
										int2 dimsimage,
										float3 camera,
										float3 pixelx,
										float3 pixely,
										float3 view,
										float3* d_colormap,
										int ncolors,
										float2 colorrange,
										float2 shadingrange,
										float3 intensitycolor,
										float2 intensityrange,
										float3* h_intersectionpoints,
										char* h_hittest,
										uchar4* h_bgra)
{
	char* d_hittest = CudaMallocValueFilled(Elements2(dimsimage), (char)0);
	float3* d_intersectionpoints;
	cudaMalloc((void**)&d_intersectionpoints, Elements2(dimsimage) * sizeof(float3));
	float3* d_intersectionnormals;
	cudaMalloc((void**)&d_intersectionnormals, Elements2(dimsimage) * sizeof(float3));
	float* d_accumulated;
	cudaMalloc((void**)&d_accumulated, Elements2(dimsimage) * sizeof(float));
	uchar4* d_bgra;
	cudaMalloc((void**)&d_bgra, Elements2(dimsimage) * sizeof(uchar4));

	dim3 grid = dim3((dimsimage.x + 7) / 8, (dimsimage.y + 7) / 8, 1);
	dim3 TpB = dim3(8, 8, 1);
	
	RaycastKernel<true> <<<grid, TpB>>> (t_intensities, 
											dimsvolume.x,
											surfacethreshold,
											dimsimage,
											camera,
											pixelx,
											pixely,
											view,
											d_intersectionpoints,
											d_intersectionnormals,
											d_hittest,
											d_accumulated);

	RaycastShadingKernel <<<grid, TpB>>> (dimsimage, 
											dimsvolume.x, 
											t_coloring, 
											dimscoloring.x, 
											d_hittest, 
											d_accumulated, 
											view,
											d_intersectionpoints,
											d_intersectionnormals,
											d_colormap, 
											ncolors, 
											colorrange, 
											shadingrange, 
											intensitycolor, 
											intensityrange, 
											d_bgra);
	
	cudaMemcpy(h_bgra, d_bgra, Elements2(dimsimage) * sizeof(uchar4), cudaMemcpyDeviceToHost);
	cudaFree(d_bgra);
	cudaMemcpy(h_intersectionpoints, d_intersectionpoints, Elements2(dimsimage) * sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(d_intersectionpoints);
	cudaMemcpy(h_hittest, d_hittest, Elements2(dimsimage) * sizeof(char), cudaMemcpyDeviceToHost);
	cudaFree(d_hittest);

	cudaFree(d_accumulated);
	cudaFree(d_intersectionnormals);
}

template<bool cubic> __device__ float Tex(cudaTex t_tex, float3 pos)
{
    if (cubic)
		return cubicTex3D(t_tex, pos.x, pos.y, pos.z);
	else
		return tex3D<tfloat>(t_tex, pos.x, pos.y, pos.z);
}

template<bool cubic> __device__ float Grad(cudaTex t_tex, float3 pos, float3 h)
{
    return Tex<cubic>(t_tex, pos + h) - Tex<cubic>(t_tex, pos - h);
}

template <bool cubicinterp> __global__ void RaycastKernel(cudaTex t_intensities, 
															int sizevolume, 
															float surfacethreshold, 
															int2 dimsimage, 
															float3 camera, 
															float3 pixelx, 
															float3 pixely, 
															float3 view, 
															float3* d_intersectionpoints, 
															float3* d_intersectionnormals, 
															char* d_hittest, 
															float* d_accumulated)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= dimsimage.x || idy >= dimsimage.y)
		return;

	float3 origin = camera + (idx - dimsimage.x / 2) * pixelx + (idy - dimsimage.y / 2) * pixely;
	double sphereradius = sizevolume / 2;
	sphereradius *= sphereradius;

	float t0, t1; // solutions for t if the ray intersects 

    // ray-sphere intersection
    double3 L = make_double3(-origin.x, -origin.y, -origin.z); 
    double tca = dotp(L, view);
    double d2 = dotp(L, L) - tca * tca; 
    if (d2 > sphereradius) 
		return; // no intersection 
    double thc = sqrt(sphereradius - d2); 
    t0 = tca - (float)thc; 
    t1 = tca + (float)thc;
	if (abs(t0 - t1) < 1e-5f)
		return;	// tangent, 0 path length

	if (t0 > t1)
	{
	    float temp = t1;
		t1 = t0;
		t0 = temp;
	}

	float3 tracestart = origin + view * t0 + make_float3(sizevolume / 2 + 0.5f);
	int nsteps = ceil(t1 - t0) * 1;
	//view *= 0.5f;

	float3 posint = make_float3(-1);
	float3 surfacenormal = make_float3(0);
	float accumulated = 0;
	char hittest = 0;

	for (int s = 0; s < nsteps; s++)
	{
	    float3 pos = tracestart + s * view;
		float val;
		if (cubicinterp)
			val = cubicTex3D(t_intensities, pos.x, pos.y, pos.z);
		else
			val = tex3D<tfloat>(t_intensities, pos.x, pos.y, pos.z);

		if (val >= surfacethreshold * 0.95f)
		{
			s--;
			float a = 0, b = 1.5f;
			float R = 0.75f;
			for (int i = 0; i < 10; i++)
			{
			    val = Tex<cubicinterp>(t_intensities, tracestart + (s + R) * view);
				if (val > surfacethreshold)
					b = R;
				else
					a = R;
				R = (a + b) * 0.5f;
			}

			posint = tracestart + (s + R) * view;
			accumulated += R * val;

			// calculate normal (= -gradient)
			surfacenormal.x = Grad<cubicinterp>(t_intensities, posint, make_float3(-0.1f, 0, 0));
			surfacenormal.y = Grad<cubicinterp>(t_intensities, posint, make_float3(0, -0.1f, 0));
			surfacenormal.z = Grad<cubicinterp>(t_intensities, posint, make_float3(0, 0, -0.1f));

			surfacenormal /= sqrt(dotp(surfacenormal, surfacenormal));

			hittest = 1;
		    break;
		}

		accumulated += val;
	}
	
	d_intersectionpoints[idy * dimsimage.x + idx] = posint - make_float3(0.5f);
	d_intersectionnormals[idy * dimsimage.x + idx] = surfacenormal;
	d_hittest[idy * dimsimage.x + idx] = hittest;
	d_accumulated[idy * dimsimage.x + idx] = accumulated;
}

__global__ void RaycastShadingKernel(int2 dimsimage, 
									int sizevolume, 
									cudaTex t_coloring, 
									int sizecoloring, 
									char* d_hittest, 
									float* d_accumulated,
									float3 view,
									float3* d_intersectionpoints, 
									float3* d_intersectionnormals, 
									float3* d_colormap,
									int ncolors,
									float2 colorrange,
									float2 shadingrange,
									float3 intensitycolor,
									float2 intensityrange,
									uchar4* d_bgra)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = idy * dimsimage.x + idx;

	if (idx >= dimsimage.x || idy >= dimsimage.y)
		return;

	char hittest = d_hittest[id];
	float4 outcolor = make_float4(0, 1, 1, 1);

	if (hittest)
	{
		float3 surfacecolor = make_float3(1);

		if (t_coloring > 0)
		{
			float3 intersectionpoint = d_intersectionpoints[id] / sizevolume * sizecoloring;
			float coloring = Tex<false>(t_coloring, intersectionpoint + make_float3(0.5f));
			coloring = (coloring - colorrange.x) / (colorrange.y - colorrange.x);
			coloring = tmax(0, tmin(1, coloring)) * (ncolors - 1);
			float3 colormap0 = d_colormap[tmax(0, (int)coloring)];
			float3 colormap1 = d_colormap[tmax(0, tmin((int)coloring + 1, ncolors - 1))];
			surfacecolor = lerp(colormap0, colormap1, coloring - floor(coloring));
		}

		float shading = abs(dotp(view, d_intersectionnormals[id]));
		shading = shading * (shadingrange.y - shadingrange.x) + shadingrange.x;
		surfacecolor *= shading;

		outcolor = make_float4(1, surfacecolor.x, surfacecolor.y, surfacecolor.z);
	}

	float accumulated = d_accumulated[id];
	accumulated = tmax(0, tmin(1, (accumulated - intensityrange.x) / (intensityrange.y - intensityrange.x)));
	outcolor = lerp(outcolor, make_float4(1, intensitycolor.x, intensitycolor.y, intensitycolor.z), accumulated);

	d_bgra[id] = make_uchar4((uchar)(tmax(0, tmin(1, outcolor.w)) * 255), 
							 (uchar)(tmax(0, tmin(1, outcolor.z)) * 255), 
							 (uchar)(tmax(0, tmin(1, outcolor.y)) * 255), 
							 (uchar)(tmax(0, tmin(1, outcolor.x)) * 255));
}