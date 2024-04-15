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

	void Projector::project(MultidimArray<Complex > &f2d, Matrix2D<DOUBLE> &A, bool inv)
	{
		DOUBLE fx, fy, fz, xp, yp, zp;
		int x0, x1, y0, y1, z0, z1, y, y2, r2;
		bool is_neg_x;
		Complex d000, d001, d010, d011, d100, d101, d110, d111;
		Complex dx00, dx01, dx10, dx11, dxy0, dxy1;
		Matrix2D<DOUBLE> Ainv;

		// f2d should already be in the right size (ori_size,orihalfdim)
		// AND the points outside r_max should already be zero...
		// f2d.initZeros();

		// Use the inverse matrix
		if (inv)
			Ainv = A;
		else
			Ainv = A.transpose();

		// The f2d image may be smaller than r_max, in that case also make sure not to fill the corners!
		int my_r_max = XMIPP_MIN(r_max, XSIZE(f2d) - 1);

		// Go from the 2D slice coordinates to the 3D coordinates
		Ainv *= (DOUBLE)padding_factor;  // take scaling into account directly
		int max_r2 = my_r_max * my_r_max;
		int min_r2_nn = r_min_nn * r_min_nn;

		//#define DEBUG
#ifdef DEBUG
		std::cerr << " XSIZE(f2d)= "<< XSIZE(f2d) << std::endl;
		std::cerr << " YSIZE(f2d)= "<< YSIZE(f2d) << std::endl;
		std::cerr << " XSIZE(data)= "<< XSIZE(data) << std::endl;
		std::cerr << " YSIZE(data)= "<< YSIZE(data) << std::endl;
		std::cerr << " STARTINGX(data)= "<< STARTINGX(data) << std::endl;
		std::cerr << " STARTINGY(data)= "<< STARTINGY(data) << std::endl;
		std::cerr << " STARTINGZ(data)= "<< STARTINGZ(data) << std::endl;
		std::cerr << " max_r= "<< r_max << std::endl;
		std::cerr << " Ainv= " << Ainv << std::endl;
#endif

		for (int i = 0; i < YSIZE(f2d); i++)
		{
			// Dont search beyond square with side max_r
			if (i <= my_r_max)
			{
				y = i;
			}
			else if (i >= YSIZE(f2d) - my_r_max)
			{
				y = i - YSIZE(f2d);
			}
			else
				continue;

			y2 = y * y;
			for (int x = 0; x <= my_r_max; x++)
			{
				// Only include points with radius < max_r (exclude points outside circle in square)
				r2 = x * x + y2;
				if (r2 > max_r2)
					continue;

				// Get logical coordinates in the 3D map
				xp = Ainv(0, 0) * x + Ainv(0, 1) * y;
				yp = Ainv(1, 0) * x + Ainv(1, 1) * y;
				zp = Ainv(2, 0) * x + Ainv(2, 1) * y;

				if (interpolator == TRILINEAR || r2 < min_r2_nn)
				{
					// Only asymmetric half is stored
					if (xp < 0)
					{
						// Get complex conjugated hermitian symmetry pair
						xp = -xp;
						yp = -yp;
						zp = -zp;
						is_neg_x = true;
					}
					else
					{
						is_neg_x = false;
					}

					// Trilinear interpolation (with physical coords)
					// Subtract STARTINGY and STARTINGZ to accelerate access to data (STARTINGX=0)
					// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
					x0 = FLOOR(xp);
					fx = xp - x0;
					x1 = x0 + 1;

					y0 = FLOOR(yp);
					fy = yp - y0;
					y0 -= STARTINGY(data);
					y1 = y0 + 1;

					z0 = FLOOR(zp);
					fz = zp - z0;
					z0 -= STARTINGZ(data);
					z1 = z0 + 1;

					// Matrix access can be accelerated through pre-calculation of z0*xydim etc.
					d000 = DIRECT_A3D_ELEM(data, z0, y0, x0);
					d001 = DIRECT_A3D_ELEM(data, z0, y0, x1);
					d010 = DIRECT_A3D_ELEM(data, z0, y1, x0);
					d011 = DIRECT_A3D_ELEM(data, z0, y1, x1);
					d100 = DIRECT_A3D_ELEM(data, z1, y0, x0);
					d101 = DIRECT_A3D_ELEM(data, z1, y0, x1);
					d110 = DIRECT_A3D_ELEM(data, z1, y1, x0);
					d111 = DIRECT_A3D_ELEM(data, z1, y1, x1);

					// interpolate in x
#ifndef FLOAT_PRECISION
					__m256d __fx = _mm256_set1_pd(fx);
					__m256d __interpx1 = LIN_INTERP_AVX(_mm256_setr_pd(d000.real, d000.imag, d100.real, d100.imag),
						_mm256_setr_pd(d001.real, d001.imag, d101.real, d101.imag),
						__fx);
					__m256d __interpx2 = LIN_INTERP_AVX(_mm256_setr_pd(d010.real, d010.imag, d110.real, d110.imag),
						_mm256_setr_pd(d011.real, d011.imag, d111.real, d111.imag),
						__fx);

					// interpolate in y
					__m256d __fy = _mm256_set1_pd(fy);
					__m256d __interpy = LIN_INTERP_AVX(__interpx1, __interpx2, __fy);
#else
					__m128 __fx = _mm_set1_ps(fx);
					__m128 __interpx1 = LIN_INTERP_AVX(_mm_setr_ps(d000.real, d000.imag, d100.real, d100.imag),
						_mm_setr_ps(d001.real, d001.imag, d101.real, d101.imag),
						__fx);
					__m128 __interpx2 = LIN_INTERP_AVX(_mm_setr_ps(d010.real, d010.imag, d110.real, d110.imag),
						_mm_setr_ps(d011.real, d011.imag, d111.real, d111.imag),
						__fx);

					// interpolate in y
					__m128 __fy = _mm_set1_ps(fy);
					__m128 __interpy = LIN_INTERP_AVX(__interpx1, __interpx2, __fy);
#endif
					Complex* interpy = (Complex*)&__interpy;

					// interpolate in z
					DIRECT_A2D_ELEM(f2d, i, x) = LIN_INTERP(fz, interpy[0], interpy[1]);

					// Take complex conjugated for half with negative x
					if (is_neg_x)
						DIRECT_A2D_ELEM(f2d, i, x) = conj(DIRECT_A2D_ELEM(f2d, i, x));

				} // endif TRILINEAR
				else if (interpolator == NEAREST_NEIGHBOUR)
				{
					x0 = ROUND(xp);
					y0 = ROUND(yp);
					z0 = ROUND(zp);
					if (x0 < 0)
						DIRECT_A2D_ELEM(f2d, i, x) = conj(A3D_ELEM(data, -z0, -y0, -x0));
					else
						DIRECT_A2D_ELEM(f2d, i, x) = A3D_ELEM(data, z0, y0, x0);

				} // endif NEAREST_NEIGHBOUR
				else
					REPORT_ERROR("Unrecognized interpolator in Projector::project");

			} // endif x-loop
		} // endif y-loop


#ifdef DEBUG
		std::cerr << "done with project..." << std::endl;
#endif
	}

	void Projector::rotate2D(MultidimArray<Complex > &f2d, Matrix2D<DOUBLE> &A, bool inv)
	{
		DOUBLE fx, fy, xp, yp;
		int x0, x1, y0, y1, y, y2, r2;
		bool is_neg_x;
		Complex d00, d01, d10, d11, dx0, dx1;
		Matrix2D<DOUBLE> Ainv;

		// f2d should already be in the right size (ori_size,orihalfdim)
		// AND the points outside max_r should already be zero...
		// f2d.initZeros();
		// Use the inverse matrix
		if (inv)
			Ainv = A;
		else
			Ainv = A.transpose();

		// The f2d image may be smaller than r_max, in that case also make sure not to fill the corners!
		int my_r_max = XMIPP_MIN(r_max, XSIZE(f2d) - 1);

		// Go from the 2D slice coordinates to the map coordinates
		Ainv *= (DOUBLE)padding_factor;  // take scaling into account directly
		int max_r2 = my_r_max * my_r_max;
		int min_r2_nn = r_min_nn * r_min_nn;
#ifdef DEBUG
		std::cerr << " XSIZE(f2d)= "<< XSIZE(f2d) << std::endl;
		std::cerr << " YSIZE(f2d)= "<< YSIZE(f2d) << std::endl;
		std::cerr << " XSIZE(data)= "<< XSIZE(data) << std::endl;
		std::cerr << " YSIZE(data)= "<< YSIZE(data) << std::endl;
		std::cerr << " STARTINGX(data)= "<< STARTINGX(data) << std::endl;
		std::cerr << " STARTINGY(data)= "<< STARTINGY(data) << std::endl;
		std::cerr << " STARTINGZ(data)= "<< STARTINGZ(data) << std::endl;
		std::cerr << " max_r= "<< r_max << std::endl;
		std::cerr << " Ainv= " << Ainv << std::endl;
#endif
		for (int i = 0; i < YSIZE(f2d); i++)
		{
			// Don't search beyond square with side max_r
			if (i <= my_r_max)
			{
				y = i;
			}
			else if (i >= YSIZE(f2d) - my_r_max)
			{
				y = i - YSIZE(f2d);
			}
			else
				continue;
			y2 = y * y;
			for (int x = 0; x <= my_r_max; x++)
			{
				// Only include points with radius < max_r (exclude points outside circle in square)
				r2 = x * x + y2;
				if (r2 > max_r2)
					continue;

				// Get logical coordinates in the 3D map
				xp = Ainv(0, 0) * x + Ainv(0, 1) * y;
				yp = Ainv(1, 0) * x + Ainv(1, 1) * y;
				if (interpolator == TRILINEAR || r2 < min_r2_nn)
				{
					// Only asymmetric half is stored
					if (xp < 0)
					{
						// Get complex conjugated hermitian symmetry pair
						xp = -xp;
						yp = -yp;
						is_neg_x = true;
					}
					else
					{
						is_neg_x = false;
					}

					// Trilinear interpolation (with physical coords)
					// Subtract STARTINGY to accelerate access to data (STARTINGX=0)
					// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
					x0 = FLOOR(xp);
					fx = xp - x0;
					x1 = x0 + 1;

					y0 = FLOOR(yp);
					fy = yp - y0;
					y0 -= STARTINGY(data);
					y1 = y0 + 1;

					// Matrix access can be accelerated through pre-calculation of z0*xydim etc.
					d00 = DIRECT_A2D_ELEM(data, y0, x0);
					d01 = DIRECT_A2D_ELEM(data, y0, x1);
					d10 = DIRECT_A2D_ELEM(data, y1, x0);
					d11 = DIRECT_A2D_ELEM(data, y1, x1);

					// Set the interpolated value in the 2D output array
#ifndef FLOAT_PRECISION
					__m256d __interpx = LIN_INTERP_AVX(_mm256_setr_pd(d00.real, d00.imag, d10.real, d10.imag),
						_mm256_setr_pd(d01.real, d01.imag, d11.real, d11.imag),
						_mm256_set1_pd(fx));
#else
					__m128 __interpx = LIN_INTERP_AVX(_mm_setr_ps(d00.real, d00.imag, d10.real, d10.imag),
						_mm_setr_ps(d01.real, d01.imag, d11.real, d11.imag),
						_mm_set1_ps(fx));
#endif

					Complex* interpx = (Complex*)&__interpx;
					DIRECT_A2D_ELEM(f2d, i, x) = LIN_INTERP(fy, interpx[0], interpx[1]);
					// Take complex conjugated for half with negative x
					if (is_neg_x)
						DIRECT_A2D_ELEM(f2d, i, x) = conj(DIRECT_A2D_ELEM(f2d, i, x));
				} // endif TRILINEAR
				else if (interpolator == NEAREST_NEIGHBOUR)
				{
					x0 = ROUND(xp);
					y0 = ROUND(yp);
					if (x0 < 0)
						DIRECT_A2D_ELEM(f2d, i, x) = conj(A2D_ELEM(data, -y0, -x0));
					else
						DIRECT_A2D_ELEM(f2d, i, x) = A2D_ELEM(data, y0, x0);
				} // endif NEAREST_NEIGHBOUR
				else
					REPORT_ERROR("Unrecognized interpolator in Projector::project");
			} // endif x-loop
		} // endif y-loop
	}


	void Projector::rotate3D(MultidimArray<Complex > &f3d, Matrix2D<DOUBLE> &A, bool inv)
	{
		DOUBLE fx, fy, fz, xp, yp, zp;
		int x0, x1, y0, y1, z0, z1, y, z, y2, z2, r2;
		bool is_neg_x;
		Complex d000, d010, d100, d110, d001, d011, d101, d111, dx00, dx10, dxy0, dx01, dx11, dxy1;
		Matrix2D<DOUBLE> Ainv;

		// f3d should already be in the right size (ori_size,orihalfdim)
		// AND the points outside max_r should already be zero...
		// f3d.initZeros();
		// Use the inverse matrix
		if (inv)
			Ainv = A;
		else
			Ainv = A.transpose();

		// The f3d image may be smaller than r_max, in that case also make sure not to fill the corners!
		int my_r_max = XMIPP_MIN(r_max, XSIZE(f3d) - 1);

		// Go from the 3D rotated coordinates to the original map coordinates
		Ainv *= (DOUBLE)padding_factor;  // take scaling into account directly
		int max_r2 = my_r_max * my_r_max;
		int min_r2_nn = r_min_nn * r_min_nn;
#ifdef DEBUG
		std::cerr << " XSIZE(f3d)= "<< XSIZE(f3d) << std::endl;
		std::cerr << " YSIZE(f3d)= "<< YSIZE(f3d) << std::endl;
		std::cerr << " XSIZE(data)= "<< XSIZE(data) << std::endl;
		std::cerr << " YSIZE(data)= "<< YSIZE(data) << std::endl;
		std::cerr << " STARTINGX(data)= "<< STARTINGX(data) << std::endl;
		std::cerr << " STARTINGY(data)= "<< STARTINGY(data) << std::endl;
		std::cerr << " STARTINGZ(data)= "<< STARTINGZ(data) << std::endl;
		std::cerr << " max_r= "<< r_max << std::endl;
		std::cerr << " Ainv= " << Ainv << std::endl;
#endif
		for (int k = 0; k < ZSIZE(f3d); k++)
		{
			// Don't search beyond square with side max_r
			if (k <= my_r_max)
			{
				z = k;
			}
			else if (k >= ZSIZE(f3d) - my_r_max)
			{
				z = k - ZSIZE(f3d);
			}
			else
				continue;
			z2 = z * z;

			for (int i = 0; i < YSIZE(f3d); i++)
			{
				// Don't search beyond square with side max_r
				if (i <= my_r_max)
				{
					y = i;
				}
				else if (i >= YSIZE(f3d) - my_r_max)
				{
					y = i - YSIZE(f3d);
				}
				else
					continue;
				y2 = y * y;

				for (int x = 0; x <= my_r_max; x++)
				{
					// Only include points with radius < max_r (exclude points outside circle in square)
					r2 = x * x + y2 + z2;
					if (r2 > max_r2)
						continue;

					// Get logical coordinates in the 3D map
					xp = Ainv(0, 0) * x + Ainv(0, 1) * y + Ainv(0, 2) * z;
					yp = Ainv(1, 0) * x + Ainv(1, 1) * y + Ainv(1, 2) * z;
					zp = Ainv(2, 0) * x + Ainv(2, 1) * y + Ainv(2, 2) * z;

					if (interpolator == TRILINEAR || r2 < min_r2_nn)
					{
						// Only asymmetric half is stored
						if (xp < 0)
						{
							// Get complex conjugated hermitian symmetry pair
							xp = -xp;
							yp = -yp;
							zp = -zp;
							is_neg_x = true;
						}
						else
						{
							is_neg_x = false;
						}

						// Trilinear interpolation (with physical coords)
						// Subtract STARTINGY to accelerate access to data (STARTINGX=0)
						// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
						x0 = FLOOR(xp);
						fx = xp - x0;
						x1 = x0 + 1;

						y0 = FLOOR(yp);
						fy = yp - y0;
						y0 -= STARTINGY(data);
						y1 = y0 + 1;

						z0 = FLOOR(zp);
						fz = zp - z0;
						z0 -= STARTINGZ(data);
						z1 = z0 + 1;

						// Matrix access can be accelerated through pre-calculation of z0*xydim etc.
						d000 = DIRECT_A3D_ELEM(data, z0, y0, x0);
						d001 = DIRECT_A3D_ELEM(data, z0, y0, x1);
						d010 = DIRECT_A3D_ELEM(data, z0, y1, x0);
						d011 = DIRECT_A3D_ELEM(data, z0, y1, x1);
						d100 = DIRECT_A3D_ELEM(data, z1, y0, x0);
						d101 = DIRECT_A3D_ELEM(data, z1, y0, x1);
						d110 = DIRECT_A3D_ELEM(data, z1, y1, x0);
						d111 = DIRECT_A3D_ELEM(data, z1, y1, x1);

						// Set the interpolated value in the 2D output array
						// interpolate in x
#ifndef FLOAT_PRECISION
						__m256d __fx = _mm256_set1_pd(fx);
						__m256d __interpx1 = LIN_INTERP_AVX(_mm256_setr_pd(d000.real, d000.imag, d100.real, d100.imag),
							_mm256_setr_pd(d001.real, d001.imag, d101.real, d101.imag),
							__fx);
						__m256d __interpx2 = LIN_INTERP_AVX(_mm256_setr_pd(d010.real, d010.imag, d110.real, d110.imag),
							_mm256_setr_pd(d011.real, d011.imag, d111.real, d111.imag),
							__fx);

						// interpolate in y
						__m256d __fy = _mm256_set1_pd(fy);
						__m256d __interpy = LIN_INTERP_AVX(__interpx1, __interpx2, __fy);
#else
						__m128 __fx = _mm_set1_ps(fx);
						__m128 __interpx1 = LIN_INTERP_AVX(_mm_setr_ps(d000.real, d000.imag, d100.real, d100.imag),
							_mm_setr_ps(d001.real, d001.imag, d101.real, d101.imag),
							__fx);
						__m128 __interpx2 = LIN_INTERP_AVX(_mm_setr_ps(d010.real, d010.imag, d110.real, d110.imag),
							_mm_setr_ps(d011.real, d011.imag, d111.real, d111.imag),
							__fx);

						// interpolate in y
						__m128 __fy = _mm_set1_ps(fy);
						__m128 __interpy = LIN_INTERP_AVX(__interpx1, __interpx2, __fy);
#endif

						Complex* interpy = (Complex*)&__interpy;

						//interpolate in z
						DIRECT_A3D_ELEM(f3d, k, i, x) = LIN_INTERP(fz, interpy[0], interpy[1]);

						// Take complex conjugated for half with negative x
						if (is_neg_x)
							DIRECT_A3D_ELEM(f3d, k, i, x) = conj(DIRECT_A3D_ELEM(f3d, k, i, x));

					} // endif TRILINEAR
					else if (interpolator == NEAREST_NEIGHBOUR)
					{
						x0 = ROUND(xp);
						y0 = ROUND(yp);
						z0 = ROUND(zp);

						if (x0 < 0)
							DIRECT_A3D_ELEM(f3d, k, i, x) = conj(A3D_ELEM(data, -z0, -y0, -x0));
						else
							DIRECT_A3D_ELEM(f3d, k, i, x) = A3D_ELEM(data, z0, y0, x0);

					} // endif NEAREST_NEIGHBOUR
					else
						REPORT_ERROR("Unrecognized interpolator in Projector::project");
				} // endif x-loop
			} // endif y-loop
		} // endif z-loop
	}
}