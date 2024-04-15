#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/DeviceFunctions.cuh"
#include "gtom/include/FFT.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Relion.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	template<uint TpB> __global__ void __launch_bounds__(TpB) MagAnisotropyCorrectKernel(cudaTex t_image, uint dimimage, tfloat* d_scaled, uint dimscaled, glm::mat2 transform);
	

	void d_MagAnisotropyCorrect(tfloat* d_image, int2 dimsimage, tfloat* d_scaledimage, int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample, uint batch)
	{
		int maxbatch = tmin(128, batch);

		//tfloat* d_imagepadded;
		//cudaMalloc((void**)&d_imagepadded, Elements2(dimsimage) * NextMultipleOf(batch, maxbatch) * sizeof(tfloat));
		//cudaMemcpy(d_imagepadded, d_image, Elements2(dimsimage) * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		//int2 dimssuper = toInt2(dimsimage.x * supersample + 0, dimsimage.y * supersample + 0);
		//tfloat* d_super;
		//cudaMalloc((void**)&d_super, Elements2(dimssuper) * maxbatch * sizeof(tfloat));

		//cufftHandle planforw = d_FFTR2CGetPlan(2, toInt3(dimsimage), maxbatch);
		//cufftHandle planback = d_IFFTC2RGetPlan(2, toInt3(dimssuper), maxbatch);

		for (int b = 0; b < batch; b += maxbatch)
		{
			int curbatch = tmin(maxbatch, (int)batch - b);

			//d_Scale(d_imagepadded + Elements2(dimsimage) * b, d_super, toInt3(dimsimage), toInt3(dimssuper), T_INTERP_FOURIER, &planforw, &planback, maxbatch);
			//d_WriteMRC(d_super, toInt3(dimssuper.x, dimssuper.y, maxbatch), "d_super.mrc");

			cudaArray_t a_image;
			cudaTex t_image;

			{
				d_BindTextureTo3DArray(d_image + Elements2(dimsimage) * b, a_image, t_image, toInt3(dimsimage.x, dimsimage.y, curbatch), cudaFilterModeLinear, false);
			}

			float meanpixel = (majorpixel + minorpixel) * 0.5f;
			majorpixel /= meanpixel;
			minorpixel /= meanpixel;
			glm::mat2 transform = Matrix2Rotation(majorangle) * Matrix2Scale(tfloat2(1.0f / majorpixel, 1.0f / minorpixel)) * Matrix2Rotation(-majorangle);
			
			dim3 grid = dim3(tmin(32768, (Elements2(dimsscaled) + 127) / 128), curbatch, 1);
			MagAnisotropyCorrectKernel<128> << <grid, 128 >> > (t_image, dimsimage.x, d_scaledimage + Elements2(dimsscaled) * b, dimsscaled.x, transform);

			{
				cudaDestroyTextureObject(t_image);
				cudaFreeArray(a_image);
			}
		}

		//cufftDestroy(planforw);
		//cufftDestroy(planback);

		//cudaFree(d_imagepadded);
		//cudaFree(d_super);
	}

	template<uint TpB> __global__ void __launch_bounds__(TpB) MagAnisotropyCorrectKernel(cudaTex t_image, uint dimimage, tfloat* d_scaled, uint dimscaled, glm::mat2 transform)
	{
		d_scaled += dimscaled * dimscaled * blockIdx.y;
		float zcoord = blockIdx.y + 0.5f;
		int imagecenter = dimimage / 2;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimscaled * dimscaled; id += gridDim.x * TpB)
		{
			uint idx = id % dimscaled;
			uint idy = id / dimscaled;

			int posx = (int)idx - dimscaled / 2;
			int posy = (int)idy - dimscaled / 2;

			glm::vec2 pos = transform * glm::vec2(posx, posy);

			pos.x += imagecenter;
			pos.y += imagecenter;
			

			//tfloat val = tex3D<tfloat>(t_image, pos.x + 0.5f, pos.y + 0.5f, zcoord);
			tfloat val = 0;

			for (int y = -8; y <= 8; y++)
			{
				float yy = floor(pos.y) + y;
				float sincy = sinc(pos.y - yy);
				float yy2 = pos.y - yy;
				yy2 *= yy2;
				yy += 0.5f;

				for (int x = -8; x <= 8; x++)
				{
					float xx = floor(pos.x) + x;
					float sincx = sinc(pos.x - xx);
					float xx2 = pos.x - xx;
					xx2 *= xx2;
					float r2 = xx2 + yy2;

					if (r2 > 64)
						continue;

					float hanning = 1.0f + cos(PI * sqrt(r2) / 8);

					val += tex3D<tfloat>(t_image, xx + 0.5f, yy, zcoord) * sincy * sincx * hanning;
				}
			}

			d_scaled[id] = val * 0.5f;
		}
	}
}
