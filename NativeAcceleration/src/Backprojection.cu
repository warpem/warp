#include "include/Functions.h"
#include "gtom/include/CubicInterp.cuh"
using namespace gtom;

template <bool iscentered, bool sliceonly> __global__ void ProjBackwardKernel(float* d_volume, int3 dimsvolume, cudaTex* dt_image, glm::mat4* d_transforms, bool halfonly);

__declspec(dllexport) void ProjBackwardOne(float* d_volume, int3 dimsvolume, float* d_image, int2 dimsimage, float* h_angles, float* h_offsets, bool outputzerocentered, bool sliceonly, bool halfonly, int batch)
{
    glm::mat4* h_transforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
    for (int b = 0; b < batch; b++)
        h_transforms[b] = Matrix4Translation(tfloat3((tfloat)dimsimage.x / 2.0f, (tfloat)dimsimage.y / 2.0f, 0.0f)) *
                          Matrix4Translation(tfloat3(-h_offsets[b * 2 + 0], -h_offsets[b * 2 + 1], 0.0f)) *
                          Matrix4Euler(((tfloat3*)h_angles)[b]);

    cudaArray_t* ha_image = (cudaArray_t*)malloc(batch * sizeof(cudaArray_t));
    cudaTex* ht_image = (cudaTex*)malloc(batch * sizeof(cudaTex));

    tfloat* d_temp;
    cudaMalloc((void**)&d_temp, Elements2(dimsimage) * batch * sizeof(tfloat));
    cudaMemcpy(d_temp, d_image, Elements2(dimsimage) * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);
    d_CubicBSplinePrefilter2D(d_temp, dimsimage, batch);
    d_BindTextureToArray(d_temp, ha_image, ht_image, dimsimage, cudaFilterModeLinear, false, batch);
    cudaFree(d_temp);

    cudaTex* dt_image = (cudaTex*)CudaMallocFromHostArray(ht_image, batch * sizeof(cudaTex));
    glm::mat4* d_transforms = (glm::mat4*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat4));

    dim3 TpB = dim3(16, 8, 1);
    dim3 grid = dim3((dimsvolume.x + 15) / 16, (dimsvolume.y + 7) / 8, batch);
    if (halfonly)
        grid = dim3((dimsvolume.x / 2 + 1 + 15) / 16, (dimsvolume.y + 7) / 8, batch);

    if (outputzerocentered)
    {
        if (sliceonly)
            ProjBackwardKernel<true, true> << <grid, TpB >> > (d_volume, dimsvolume, dt_image, d_transforms, halfonly);
        else
            ProjBackwardKernel<true, false> << <grid, TpB >> > (d_volume, dimsvolume, dt_image, d_transforms, halfonly);
    }
    else
    {
        if (sliceonly)
            ProjBackwardKernel<false, true> << <grid, TpB >> > (d_volume, dimsvolume, dt_image, d_transforms, halfonly);
        else
            ProjBackwardKernel<false, false> << <grid, TpB >> > (d_volume, dimsvolume, dt_image, d_transforms, halfonly);
    }

    for (int n = 0; n < batch; n++)
    {
        cudaDestroyTextureObject(ht_image[n]);
        cudaFreeArray(ha_image[n]);
    }

    free(ht_image);
    free(ha_image);
    cudaFree(d_transforms);
    cudaFree(dt_image);

    free(h_transforms);
}

template <bool iscentered, bool sliceonly> __global__ void ProjBackwardKernel(float* d_volume, int3 dimsvolume, cudaTex* dt_image, glm::mat4* d_transforms, bool halfonly)
{
    d_volume += blockIdx.z * (halfonly ? ElementsFFT(dimsvolume) : Elements(dimsvolume));

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (halfonly ? dimsvolume.x / 2 + 1 : dimsvolume.x))
        return;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= dimsvolume.y)
        return;

    int posx, posy, posz;
    if (!iscentered)
    {
        posx = idx < dimsvolume.x / 2 + 1 ? idx : idx - dimsvolume.x;
        posy = idy < dimsvolume.y / 2 + 1 ? idy : idy - dimsvolume.y;
    }
    else
    {
        posx = idx - dimsvolume.x / 2;
        posy = idy - dimsvolume.y / 2;
    }

    for (int idz = 0; idz < dimsvolume.z; idz++)
    {
        if (!iscentered)
            posz = idz < dimsvolume.z / 2 + 1 ? idz : idz - dimsvolume.z;
        else
            posz = idz - dimsvolume.z / 2;

        tfloat sum = 0.0f;

        glm::vec4 pos = d_transforms[blockIdx.z] * glm::vec4(posx, posy, posz, 1);

        float sliceweight = sliceonly ? tmax(0, 1.0f - abs(pos.z)) : 1.0f;

        /*for (int y = -5; y <= 5; y++)
        {
            float yy = floor(pos.y) + y;
            float sincy = sinc(pos.y - yy);
            float yy2 = pos.y - yy;
            yy2 *= yy2;
            yy += 0.5f;

            for (int x = -5; x <= 5; x++)
            {
                float xx = floor(pos.x) + x;
                float sincx = sinc(pos.x - xx);
                float xx2 = pos.x - xx;
                xx2 *= xx2;
                float r2 = xx2 + yy2;

                if (r2 > 25)
                    continue;

                float hanning = 1.0f + cos(PI * sqrt(r2) / 5);

                sum += tex2D<tfloat>(dt_image[blockIdx.z], xx + 0.5f, yy) * sincy * sincx * hanning * sliceweight;
            }
        }*/
        sum = cubicTex2D(dt_image[blockIdx.z], pos.x + 0.5f, pos.y + 0.5f) * sliceweight;

        d_volume[(idz * dimsvolume.y + idy) * (halfonly ? dimsvolume.x / 2 + 1 : dimsvolume.x) + idx] = sum;
    }
}