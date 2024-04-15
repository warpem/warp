#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Angles.cuh"
#include "gtom/include/CTF.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/ImageManipulation.cuh"
#include "gtom/include/Masking.cuh"
#include "gtom/include/Relion.cuh"
#include "gtom/include/Transformation.cuh"

namespace gtom
{
	void d_FitMagAnisotropy(tfloat* h_image, int2 dimsimage, float compareradius, float maxdistortion, float distortionstep, float anglestep, float &bestdistortion, float &bestangle)
	{
		int nangles = 2 * PI / anglestep;
		anglestep = 2 * PI / nangles;

		tfloat* d_mask = CudaMallocValueFilled(Elements2(dimsimage) * nangles, (tfloat)1);
		{
			d_SphereMask(d_mask, d_mask, toInt3(dimsimage), &compareradius, 0, NULL, false);

			tfloat* d_maskinner = CudaMallocValueFilled(Elements2(dimsimage), (tfloat)1);
			tfloat innerradius = 0;
			d_SphereMask(d_maskinner, d_maskinner, toInt3(dimsimage), &innerradius, 0, NULL, false);
			d_SubtractVector(d_mask, d_maskinner, d_mask, Elements2(dimsimage));

			CudaMemcpyMulti(d_mask + Elements2(dimsimage), d_mask, Elements2(dimsimage), nangles - 1);

			cudaFree(d_maskinner);
		}
		// Create 2D rotational average
		tfloat* d_rotationalaverage;
		{
			std::vector<tfloat> average1d(dimsimage.x / 2);
			std::vector<int> average1dsamples(dimsimage.x / 2);
			for (int y = 0; y < dimsimage.y; y++)
			{
				int yy = y - dimsimage.y / 2;
				yy *= yy;

				for (int x = 0; x < dimsimage.x; x++)
				{
					int xx = x - dimsimage.x / 2;
					xx *= xx;

					int r = (int)(sqrt((float)xx + yy) + 0.5f);
					if (r >= average1d.size())
						continue;

					average1d[r] += h_image[y * dimsimage.x + x];
					average1dsamples[r]++;
				}
			}

			for (int i = 0; i < average1d.size(); i++)
				average1d[i] /= tmax(1, average1dsamples[i]);

			tfloat* h_rotationalaverage = MallocValueFilled(Elements2(dimsimage), (tfloat)0);

			for (int y = 0; y < dimsimage.y; y++)
			{
				int yy = y - dimsimage.y / 2;
				yy *= yy;

				for (int x = 0; x < dimsimage.x; x++)
				{
					int xx = x - dimsimage.x / 2;
					xx *= xx;

					float r = sqrt((float)xx + yy) + 0.5f;
					float frac = 1 - (r - floor(r));
					h_rotationalaverage[y * dimsimage.x + x] = average1d[tmin((int)r, average1d.size() - 1)] * frac + average1d[tmin((int)r + 1, average1d.size() - 1)] * (1 - frac);
				}
			}

			d_rotationalaverage = (tfloat*)CudaMallocFromHostArray(h_rotationalaverage, Elements2(dimsimage) * sizeof(tfloat));
			free(h_rotationalaverage);

			d_NormMonolithic(d_rotationalaverage, d_rotationalaverage, Elements2(dimsimage), d_mask, T_NORM_MEAN01STD, 1);
			d_MultiplyByVector(d_rotationalaverage, d_mask, d_rotationalaverage, Elements2(dimsimage));
		}
		d_WriteMRC(d_rotationalaverage, toInt3(dimsimage), "d_rotationalaverage.mrc");

		tfloat* d_prerotated = CudaMallocValueFilled(Elements2(dimsimage) * nangles, (tfloat)0);
		{
			int2 dimsimagescaled = dimsimage * 20;
			tfloat* d_image = (tfloat*)CudaMallocFromHostArray(h_image, Elements2(dimsimage) * sizeof(tfloat));
			d_Bandpass(d_image, d_image, toInt3(dimsimage), 0, dimsimage.x / 16, 1);

			tfloat* d_imagescaled = CudaMallocValueFilled(Elements2(dimsimagescaled), (tfloat)0);
			tfloat* d_imagerotated = CudaMallocValueFilled(Elements2(dimsimagescaled), (tfloat)0);

			d_Scale(d_image, d_imagescaled, toInt3(dimsimage), toInt3(dimsimagescaled), T_INTERP_FOURIER);

			for (int i = 0; i < nangles; i++)
			{
				float angle = -i * anglestep;
				d_Rotate2D(d_imagescaled, d_imagerotated, dimsimagescaled, &angle, T_INTERP_LINEAR, true);
				d_Scale(d_imagerotated, d_prerotated + Elements2(dimsimage) * i, toInt3(dimsimagescaled), toInt3(dimsimage), T_INTERP_FOURIER);
			}

			cudaFree(d_imagerotated);
			cudaFree(d_imagescaled);
			cudaFree(d_image);
		}
		d_WriteMRC(d_prerotated, toInt3(dimsimage.x, dimsimage.y, nangles), "d_prerotated.mrc");

		int ndistortions = maxdistortion / distortionstep;
		distortionstep = maxdistortion / tmax(1, ndistortions - 1);

		float bestscore = -1e30;

		tfloat* d_distorted = CudaMallocValueFilled(Elements2(dimsimage) * nangles, (tfloat)0);
		tfloat* d_sums = CudaMallocValueFilled(nangles, (tfloat)0);

		std::vector<tfloat> scores(ndistortions * nangles);

		for (int d = 0; d < ndistortions; d++)
		{
			float distortion = d * distortionstep;

			d_MagAnisotropyCorrect(d_prerotated, dimsimage, d_distorted, dimsimage, 1.0f + distortion / 2, 1.0f - distortion / 2, 0, 8, nangles);
			d_NormMonolithic(d_distorted, d_distorted, Elements2(dimsimage), d_mask, T_NORM_MEAN01STD, nangles);

			d_WriteMRC(d_distorted, toInt3(dimsimage.x, dimsimage.y, nangles), "d_distorted.mrc");

			d_MultiplyByVector(d_distorted, d_rotationalaverage, d_distorted, Elements2(dimsimage), nangles);
			d_SumMonolithic(d_distorted, d_sums, Elements2(dimsimage), nangles);

			tfloat* h_sums = (tfloat*)MallocFromDeviceArray(d_sums, nangles * sizeof(tfloat));

			for (int a = 0; a < nangles; a++)
			{
				scores[d * nangles + a] = h_sums[a];

				if (h_sums[a] > bestscore)
				{
					bestscore = h_sums[a];
					bestangle = a * anglestep;
					bestdistortion = distortion;
				}
			}

			free(h_sums);
		}

		cudaFree(d_sums);
		cudaFree(d_distorted);
		cudaFree(d_prerotated);
		cudaFree(d_rotationalaverage);
		cudaFree(d_mask);
	}
}