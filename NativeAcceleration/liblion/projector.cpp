/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#include "projector.h"
//#define DEBUG


namespace relion
{
	void Projector::initialiseData(int current_size)
	{
		// By default r_max is half ori_size
		if (current_size < 0)
			r_max = ori_size / 2;
		else
			r_max = current_size / 2;

		// Never allow r_max beyond Nyquist...
		r_max = XMIPP_MIN(r_max, ori_size / 2);

		// Set pad_size
		pad_size = 2 * (padding_factor * r_max + 1) + 1;

		// Short side of data array
		if (data.data == NULL)
			switch (ref_dim)
			{
			case 2:
				data.resize(pad_size, pad_size / 2 + 1);
				break;
			case 3:
				data.resize(pad_size, pad_size, pad_size / 2 + 1);
				break;
			default:
				REPORT_ERROR("Projector::resizeData%%ERROR: Dimension of the data array should be 2 or 3");
			}

		data.ndim = 1;
		data.xdim = pad_size / 2 + 1;
		data.ydim = pad_size;
		data.zdim = pad_size;
		data.yxdim = data.ydim * data.xdim;
		data.zyxdim = data.zdim * data.yxdim;
		data.nzyxdim = 1 * data.zyxdim;
		data.nzyxdimAlloc = data.nzyxdim;

		// Set origin in the y.z-center, but on the left side for x.
		data.setXmippOrigin();
		data.xinit = 0;

		memset(data.data, 0, data.nzyxdim * sizeof(Complex));
	}

	void Projector::initZeros(int current_size)
	{
		initialiseData(current_size);
		//data.initZeros();
	}

	long int Projector::getSize()
	{
		// Short side of data array
		switch (ref_dim)
		{
		case 2:
			return pad_size * (pad_size / 2 + 1);
			break;
		case 3:
			return pad_size * pad_size * (pad_size / 2 + 1);
			break;
		default:
			REPORT_ERROR("Projector::resizeData%%ERROR: Dimension of the data array should be 2 or 3");
		}

	}

	// Fill data array with oversampled Fourier transform, and calculate its power spectrum
	void Projector::computeFourierTransformMap(MultidimArray<DOUBLE> &vol_in, float* vol_out, int current_size, int nr_threads, bool do_gridding, bool do_statistics, bool output_centered)
	{

		MultidimArray<DOUBLE> Mpad;
		MultidimArray<Complex > Faux;
		FourierTransformer transformer;
		DOUBLE normfft;

		// Size of padded real-space volume
		int padoridim = padding_factor * ori_size;

		// Initialize data array of the oversampled transform
		ref_dim = vol_in.getDim();

		// Make Mpad
		switch (ref_dim)
		{
		case 2:
			Mpad.initZeros(padoridim, padoridim);
			normfft = (DOUBLE)(padding_factor * padding_factor);
			break;
		case 3:
			Mpad.initZeros(padoridim, padoridim, padoridim);
			if (data_dim == 3)
				normfft = (DOUBLE)(padding_factor * padding_factor * padding_factor);
			else
				normfft = (DOUBLE)(padding_factor * padding_factor * padding_factor * ori_size);
			break;
		default:
			REPORT_ERROR("Projector::computeFourierTransformMap%%ERROR: Dimension of the data array should be 2 or 3");
		}

		//normfft = 1;

		// First do a gridding pre-correction on the real-space map:
		// Divide by the inverse Fourier transform of the interpolator in Fourier-space
		// 10feb11: at least in 2D case, this seems to be the wrong thing to do!!!
		// TODO: check what is best for subtomo!
		if (do_gridding)// && data_dim != 3)
			griddingCorrect(vol_in);

		// Pad translated map with zeros
		vol_in.setXmippOrigin();
		Mpad.setXmippOrigin();

#pragma omp parallel for
		for (long int k = STARTINGZ(vol_in); k <= FINISHINGZ(vol_in); k++)
			for (long int i = STARTINGY(vol_in); i <= FINISHINGY(vol_in); i++)
				for (long int j = STARTINGX(vol_in); j <= FINISHINGX(vol_in); j++)
					A3D_ELEM(Mpad, k, i, j) = A3D_ELEM(vol_in, k, i, j);

		// Translate padded map to put origin of FT in the center
		CenterFFT(Mpad, true);

		// Calculate the oversampled Fourier transform
		transformer.FourierTransform(Mpad, Faux, false);

		//DOUBLE padnorm = 1. / (padding_factor)
		//FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Faux)
		//	DIRECT_MULTIDIM_ELEM(Faux, n) /= size;

		// Free memory: Mpad no longer needed
		Mpad.clear();

		// Resize data array to the right size and initialise to zero
		data.data = (Complex*)vol_out;
		initZeros(current_size);

		int max_r2 = r_max * r_max * padding_factor * padding_factor;
		
		{
			if (output_centered)
			{
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
				{
					int r2 = kp*kp + ip*ip + jp*jp;
					// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
					if (r2 <= max_r2)
					{
						// Set data array
						A3D_ELEM(data, kp, ip, jp) = DIRECT_A3D_ELEM(Faux, k, i, j) * normfft;
					}
				}
			}
			else
			{
				#pragma omp parallel for
				for (long int k = 0; k < ZSIZE(Faux); k++)
				{
					long int kp = (k < XSIZE(Faux)) ? k : k - ZSIZE(Faux);

					for (long int i = 0, ip = 0; i < YSIZE(Faux); i++, ip = (i < XSIZE(Faux)) ? i : i - YSIZE(Faux))
						for (long int j = 0, jp = 0; j < XSIZE(Faux); j++, jp = j)
						{
							int r2 = kp * kp + ip * ip + jp * jp;
							// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
							if (r2 <= max_r2)
							{
								int jj = j;
								int ii = ip < 0 ? YSIZE(data) + ip : ip;
								int kk = kp < 0 ? ZSIZE(data) + kp : kp;
								// Set data array
								DIRECT_A3D_ELEM(data, kk, ii, jj) = DIRECT_A3D_ELEM(Faux, k, i, j) * normfft;
							}
						}
				}
			}
		}

		data.data = NULL;
		transformer.cleanup();
	}

	void Projector::griddingCorrect(MultidimArray<DOUBLE> &vol_in)
	{
		// Correct real-space map by dividing it by the Fourier transform of the interpolator(s)
		vol_in.setXmippOrigin();
#pragma omp parallel for
		for (long int k = STARTINGZ(vol_in); k <= FINISHINGZ(vol_in); k++)
			for (long int i = STARTINGY(vol_in); i <= FINISHINGY(vol_in); i++)
				for (long int j = STARTINGX(vol_in); j <= FINISHINGX(vol_in); j++)
				{
					DOUBLE r = sqrt((DOUBLE)(k*k + i*i + j*j));
					// if r==0: do nothing (i.e. divide by 1)
					if (r > 0.)
					{
						DOUBLE rval = r / (ori_size * padding_factor);
						DOUBLE sinc = sin(PI * rval) / (PI * rval);
						//DOUBLE ftblob = blob_Fourier_val(rval, blob) / blob_Fourier_val(0., blob);
						// Interpolation (goes with "interpolator") to go from arbitrary to fine grid
						if (interpolator == NEAREST_NEIGHBOUR && r_min_nn == 0)
						{
							// NN interpolation is convolution with a rectangular pulse, which FT is a sinc function
							A3D_ELEM(vol_in, k, i, j) /= sinc;
						}
						else if (interpolator == TRILINEAR || (interpolator == NEAREST_NEIGHBOUR && r_min_nn > 0))
						{
							// trilinear interpolation is convolution with a triangular pulse, which FT is a sinc^2 function
							A3D_ELEM(vol_in, k, i, j) /= sinc * sinc;
						}
						else
							REPORT_ERROR("BUG Projector::griddingCorrect: unrecognised interpolator scheme.");
						//#define DEBUG_GRIDDING_CORRECT
		#ifdef DEBUG_GRIDDING_CORRECT
						if (k==0 && i==0 && j > 0)
							std::cerr << " j= " << j << " sinc= " << sinc << std::endl;
		#endif
					}
				}
	}

}
