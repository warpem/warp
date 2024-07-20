#include "include/Functions.h"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/CubicInterp.cuh"
using namespace gtom;

__global__ void BoxNet2AugmentKernel(cudaTex t_imagemic,
                                    float* d_inputlabel,
                                    float* d_noise,
                                    int2 dimsinput,
                                    float* d_outputmic,
                                    float* d_outputlabel,
                                    uint dimoutput,
                                    float2* d_offsets,
                                    glm::mat2* d_transforms,
									float offsetmean,
									float offsetscale,
                                    bool channelsfirst);

__global__ void BoxNetMMAugmentPickingKernel(cudaTex t_imagemic,
											float* d_inputlabel,
											float* d_noise,
											int2 dimsinput,
											float* d_outputmic,
											float* d_outputlabel,
											uint dimoutput,
											float2* d_offsets,
											glm::mat2* d_transforms,
											float offsetmean,
											float offsetscale,
											bool channelsfirst);

__global__ void BoxNetMMAugmentDenoisingKernel(cudaTex t_inputodd,
											cudaTex t_inputeven,
											int2 dimsinput,
											float* d_outputodd,
											float* d_outputeven,
											uint dimoutput,
											float2* d_offsets,
											glm::mat2* d_transforms,
											float offsetmean,
											float offsetscale);

void BoxNet2Augment(float* d_inputmic, 
                    float* d_inputlabel, 
                    int2 dimsinput, 
                    float* d_outputmic, 
                    float* d_outputlabel, 
                    int2 dimsoutput, 
                    float2* h_offsets, 
                    float* h_rotations, 
                    float3* h_scales,
				    float offsetmean,
				    float offsetscale,
                    float noisestddev, 
                    int seed,
                    bool channelsfirst,
                    uint batch)
{
    cudaArray_t a_inputmic;
    cudaTex t_inputmic;

    {
        d_BindTextureToArray(d_inputmic, a_inputmic, t_inputmic, toInt2(dimsinput.x, dimsinput.y), cudaFilterModePoint, false);
    }

    glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
    for (int i = 0; i < batch; i++)
        h_transforms[i] = Matrix2Rotation(h_rotations[i]) * Matrix2Rotation(h_scales[i].z) * Matrix2Scale(tfloat2(1.0f / h_scales[i].x, 1.0f / h_scales[i].y)) * Matrix2Rotation(-h_scales[i].z);

    glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
    free(h_transforms);

    float* d_noise = CudaMallocRandomFilled(Elements2(dimsinput), 0, noisestddev, seed);

    float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

    dim3 grid = dim3(tmin(32768, (Elements2(dimsoutput) + 127) / 128), batch, 1);
    BoxNet2AugmentKernel << <grid, 128 >> > (t_inputmic, 
											 d_inputlabel, 
									         d_noise, 
									         dimsinput, 
									         d_outputmic, 
									         d_outputlabel, 
									         dimsoutput.x, 
									         d_offsets, 
									         d_transforms, 
									         offsetmean, 
									         offsetscale, 
									         channelsfirst);

    cudaFree(d_noise);
    cudaFree(d_offsets);
    cudaFree(d_transforms);

    {
        cudaDestroyTextureObject(t_inputmic);
        cudaFreeArray(a_inputmic);
    }
    //d_AddVector(d_outputmic, d_noise, d_outputmic, Elements2(dimsoutput) * batch);


    //d_NormMonolithic(d_outputmic, d_outputmic, Elements2(dimsoutput), T_NORM_MEAN01STD, batch);
}

__global__ void BoxNet2AugmentKernel(cudaTex t_inputmic, 
                                    float* d_inputlabel, 
                                    float* d_noise,
                                    int2 dimsinput, 
                                    float* d_outputmic, 
                                    float* d_outputlabel, 
                                    uint dimoutput, 
                                    float2* d_offsets, 
                                    glm::mat2* d_transforms,
								    float offsetmean,
								    float offsetscale,
                                    bool channelsfirst)
{
    d_outputmic += dimoutput * dimoutput * blockIdx.y;
    d_outputlabel += dimoutput * dimoutput * blockIdx.y * 3;

    int outputcenter = dimoutput / 2;

    glm::mat2 transform = d_transforms[blockIdx.y];
    float2 offset = d_offsets[blockIdx.y];

    for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimoutput * dimoutput; id += gridDim.x * blockDim.x)
    {
        uint idx = id % dimoutput;
        uint idy = id / dimoutput;

        int posx = (int)idx - outputcenter;
        int posy = (int)idy - outputcenter;

        glm::vec2 pos = transform * glm::vec2(posx, posy);

        pos.x += offset.x;
        pos.y += offset.y;

        /*if (pos.x > dimsinput.x - 1)
            pos.x = dimsinput.x * 2 - 2 - pos.x;
        if (pos.y > dimsinput.y - 1)
            pos.y = dimsinput.y * 2 - 2 - pos.y;
        pos.x = abs(pos.x);
        pos.y = abs(pos.y);*/
        
        float val = 0;
        float samples = 0;
        float3 label = make_float3(1, 0, 0);

        if (pos.x > 0 && pos.y > 0 && pos.x < dimsinput.x - 1 && pos.y < dimsinput.y - 1)
        {
            for (int y = -4; y <= 4; y++)
            {
                float yy = floor(pos.y) + y;
                float sincy = sinc(pos.y - yy);
                float yy2 = pos.y - yy;
                yy2 *= yy2;
                yy += 0.5f;

                for (int x = -4; x <= 4; x++)
                {
                    float xx = floor(pos.x) + x;
                    float sincx = sinc(pos.x - xx);
                    float xx2 = pos.x - xx;
                    xx2 *= xx2;
                    float r2 = xx2 + yy2;

                    if (r2 > 16)
                        continue;

                    float hanning = 1.0f + cos(PI * sqrt(r2) / 4);

                    val += tex2D<float>(t_inputmic, xx + 0.5f, yy) * sincy * sincx * hanning;
                    samples += sincy * sincx * hanning;
                }
            }

            int labelcompressed = (int)d_inputlabel[tmin((int)(pos.y + 0.5f), dimsinput.y - 1) * dimsinput.x + tmin((int)(pos.x + 0.5f), dimsinput.x - 1)];
            label = make_float3(labelcompressed == 0 ? 1 : 0,
                                labelcompressed == 1 ? 1 : 0,
                                labelcompressed == 2 ? 1 : 0);

            float noise = d_noise[tmin((int)(pos.y + 0.5f), dimsinput.y - 1) * dimsinput.x + tmin((int)(pos.x + 0.5f), dimsinput.x - 1)];
            val += noise;
        }

        if (samples != 0)
			val /= samples;

        d_outputmic[id] = val * offsetscale + offsetmean;

        if (channelsfirst)
        {
            d_outputlabel[id * 3 + 0] = label.x;
            d_outputlabel[id * 3 + 1] = label.y;
            d_outputlabel[id * 3 + 2] = label.z;
        }
        else
        {
            d_outputlabel[id + dimoutput * dimoutput * 0] = label.x;
            d_outputlabel[id + dimoutput * dimoutput * 1] = label.y;
            d_outputlabel[id + dimoutput * dimoutput * 2] = label.z;
        }
    }
}



void BoxNetMMAugmentPicking(cudaTex t_inputmic,
							float* d_inputlabel,
							int2 dimsinput,
							float* d_outputmic,
							float* d_outputlabel,
							int2 dimsoutput,
							float2* h_offsets,
							float* h_rotations,
							float3* h_scales,
							float offsetmean,
							float offsetscale,
							float noisestddev,
							int seed,
							bool channelsfirst,
							uint batch)
{
	glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
	for (int i = 0; i < batch; i++)
		h_transforms[i] = Matrix2Rotation(h_rotations[i]) * Matrix2Rotation(h_scales[i].z) * Matrix2Scale(tfloat2(1.0f / h_scales[i].x, 1.0f / h_scales[i].y)) * Matrix2Rotation(-h_scales[i].z);

	glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
	free(h_transforms);

	float* d_noise = CudaMallocRandomFilled(Elements2(dimsinput), 0, noisestddev, seed);

	float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

	dim3 grid = dim3(tmin(32768, (Elements2(dimsoutput) + 127) / 128), batch, 1);
	BoxNetMMAugmentPickingKernel << <grid, 128 >> > (t_inputmic,
													d_inputlabel,
													d_noise,
													dimsinput,
													d_outputmic,
													d_outputlabel,
													dimsoutput.x,
													d_offsets,
													d_transforms,
													offsetmean,
													offsetscale,
													channelsfirst);

	cudaFree(d_noise);
	cudaFree(d_offsets);
	cudaFree(d_transforms);

	//d_AddVector(d_outputmic, d_noise, d_outputmic, Elements2(dimsoutput) * batch);


	//d_NormMonolithic(d_outputmic, d_outputmic, Elements2(dimsoutput), T_NORM_MEAN01STD, batch);
}

__global__ void BoxNetMMAugmentPickingKernel(cudaTex t_inputmic,
											float* d_inputlabel,
											float* d_noise,
											int2 dimsinput,
											float* d_outputmic,
											float* d_outputlabel,
											uint dimoutput,
											float2* d_offsets,
											glm::mat2* d_transforms,
											float offsetmean,
											float offsetscale,
											bool channelsfirst)
{
	d_outputmic += dimoutput * dimoutput * blockIdx.y;
	d_outputlabel += dimoutput * dimoutput * blockIdx.y * 3;

	int outputcenter = dimoutput / 2;

	glm::mat2 transform = d_transforms[blockIdx.y];
	float2 offset = d_offsets[blockIdx.y];

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimoutput * dimoutput; id += gridDim.x * blockDim.x)
	{
		uint idx = id % dimoutput;
		uint idy = id / dimoutput;

		int posx = (int)idx - outputcenter;
		int posy = (int)idy - outputcenter;

		glm::vec2 pos = transform * glm::vec2(posx, posy);

		pos.x += offset.x;
		pos.y += offset.y;

		float val = 0;
		float3 label = make_float3(1, 0, 0);

		if (pos.x > 0 && pos.y > 0 && pos.x < dimsinput.x - 1 && pos.y < dimsinput.y - 1)
		{
			val = cubicTex2DSimple<float>(t_inputmic, pos.x + 0.5f, pos.y + 0.5f);

			int labelcompressed = (int)d_inputlabel[tmin((int)(pos.y + 0.5f), dimsinput.y - 1) * dimsinput.x + tmin((int)(pos.x + 0.5f), dimsinput.x - 1)];
			label = make_float3(labelcompressed == 0 ? 1 : 0,
								labelcompressed == 1 ? 1 : 0,
								labelcompressed == 2 ? 1 : 0);

			float noise = d_noise[tmin((int)(pos.y + 0.5f), dimsinput.y - 1) * dimsinput.x + tmin((int)(pos.x + 0.5f), dimsinput.x - 1)];
			val += noise;
		}

		d_outputmic[id] = val * offsetscale + offsetmean;

		if (channelsfirst)
		{
			d_outputlabel[id * 3 + 0] = label.x;
			d_outputlabel[id * 3 + 1] = label.y;
			d_outputlabel[id * 3 + 2] = label.z;
		}
		else
		{
			d_outputlabel[id + dimoutput * dimoutput * 0] = label.x;
			d_outputlabel[id + dimoutput * dimoutput * 1] = label.y;
			d_outputlabel[id + dimoutput * dimoutput * 2] = label.z;
		}
	}
}

void BoxNetMMAugmentDenoising(cudaTex t_inputodd,
							cudaTex t_inputeven,
							int2 dimsinput,
							float* d_outputodd,
							float* d_outputeven,
							int2 dimsoutput,
							float2* h_offsets,
							float* h_rotations,
							float3* h_scales,
							float offsetmean,
							float offsetscale,
							uint batch)
{
	glm::mat2* h_transforms = (glm::mat2*)malloc(batch * sizeof(glm::mat2));
	for (int i = 0; i < batch; i++)
		h_transforms[i] = Matrix2Rotation(h_rotations[i]) * Matrix2Rotation(h_scales[i].z) * Matrix2Scale(tfloat2(1.0f / h_scales[i].x, 1.0f / h_scales[i].y)) * Matrix2Rotation(-h_scales[i].z);

	glm::mat2* d_transforms = (glm::mat2*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat2));
	free(h_transforms);

	float2* d_offsets = (float2*)CudaMallocFromHostArray(h_offsets, batch * sizeof(float2));

	dim3 grid = dim3(tmin(32768, (Elements2(dimsoutput) + 127) / 128), batch, 1);
	BoxNetMMAugmentDenoisingKernel << <grid, 128 >> > (t_inputodd,
														t_inputeven,
														dimsinput,
														d_outputodd,
														d_outputeven,
														dimsoutput.x,
														d_offsets,
														d_transforms,
														offsetmean,
														offsetscale);

	cudaFree(d_offsets);
	cudaFree(d_transforms);
}

__global__ void BoxNetMMAugmentDenoisingKernel(cudaTex t_inputodd,
											cudaTex t_inputeven,
											int2 dimsinput,
											float* d_outputodd,
											float* d_outputeven,
											uint dimoutput,
											float2* d_offsets,
											glm::mat2* d_transforms,
											float offsetmean,
											float offsetscale)
{
	d_outputodd += dimoutput * dimoutput * blockIdx.y;
	d_outputeven += dimoutput * dimoutput * blockIdx.y;

	int outputcenter = dimoutput / 2;

	glm::mat2 transform = d_transforms[blockIdx.y];
	float2 offset = d_offsets[blockIdx.y];

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimoutput * dimoutput; id += gridDim.x * blockDim.x)
	{
		uint idx = id % dimoutput;
		uint idy = id / dimoutput;

		int posx = (int)idx - outputcenter;
		int posy = (int)idy - outputcenter;

		glm::vec2 pos = transform * glm::vec2(posx, posy);

		pos.x += offset.x;
		pos.y += offset.y;

		float valodd = 0;
		float valeven = 0;
		float samples = 0;
		float3 label = make_float3(1, 0, 0);

		if (pos.x > 0 && pos.y > 0 && pos.x < dimsinput.x - 1 && pos.y < dimsinput.y - 1)
		{
			valodd = cubicTex2DSimple<float>(t_inputodd, pos.x + 0.5f, pos.y + 0.5f);
			valeven = cubicTex2DSimple<float>(t_inputeven, pos.x + 0.5f, pos.y + 0.5f);
		}

		d_outputodd[id] = valodd * offsetscale + offsetmean;
		d_outputeven[id] = valeven * offsetscale + offsetmean;
	}
}