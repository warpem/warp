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
#include "image.h"



namespace relion
{
	// Get size of datatype
	unsigned long  gettypesize(DataType type)
	{
		unsigned long   size;

		switch (type) {
		case UChar: case SChar:  size = sizeof(char); break;
		case UShort: case Short: size = sizeof(short); break;
		case UInt:	 case Int:   size = sizeof(int); break;
		case Float:              size = sizeof(float); break;
		case Double:             size = sizeof(DOUBLE); break;
		case Bool:				  size = sizeof(bool); break;
		default: size = 0;
		}

		return(size);
	}

	int datatypeString2Int(std::string s)
	{
		toLower(s);
		if (!strcmp(s.c_str(), "uchar"))
		{
			return UChar;
		}
		else if (!strcmp(s.c_str(), "ushort"))
		{
			return UShort;
		}
		else if (!strcmp(s.c_str(), "short"))
		{
			return Short;
		}
		else if (!strcmp(s.c_str(), "uint"))
		{
			return UInt;
		}
		else if (!strcmp(s.c_str(), "int"))
		{
			return Int;
		}
		else if (!strcmp(s.c_str(), "float"))
		{
			return Float;
		}
		else REPORT_ERROR("datatypeString2int; unknown datatype");


	}

	// Some image-specific operations
	void normalise(Image<DOUBLE> &I, int bg_radius, DOUBLE white_dust_stddev, DOUBLE black_dust_stddev, bool do_ramp)
	{
		int bg_radius2 = bg_radius * bg_radius;
		DOUBLE avg, stddev;

		if (2 * bg_radius > XSIZE(I()))
			REPORT_ERROR("normalise ERROR: 2*bg_radius is larger than image size!");

		if (white_dust_stddev > 0. || black_dust_stddev > 0.)
		{
			// Calculate initial avg and stddev values
			calculateBackgroundAvgStddev(I, avg, stddev, bg_radius);

			// Remove white and black noise
			if (white_dust_stddev > 0.)
				removeDust(I, true, white_dust_stddev, avg, stddev);
			if (black_dust_stddev > 0.)
				removeDust(I, false, black_dust_stddev, avg, stddev);
		}

		if (do_ramp)
			subtractBackgroundRamp(I, bg_radius);

		// Calculate avg and stddev (also redo if dust was removed!)
		calculateBackgroundAvgStddev(I, avg, stddev, bg_radius);

		if (stddev < 1e-10)
		{
			std::cerr << " WARNING! Stddev of image " << I.name() << " is zero! Skipping normalisation..." << std::endl;
		}
		else
		{
			// Subtract avg and divide by stddev for all pixels
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
				DIRECT_MULTIDIM_ELEM(I(), n) = (DIRECT_MULTIDIM_ELEM(I(), n) - avg) / stddev;
		}
	}

	void calculateBackgroundAvgStddev(Image<DOUBLE> &I, DOUBLE &avg, DOUBLE &stddev, int bg_radius)
	{
		int bg_radius2 = bg_radius * bg_radius;
		DOUBLE n = 0.;
		avg = 0.;
		stddev = 0.;

		// Calculate avg in the background pixels
		FOR_ALL_ELEMENTS_IN_ARRAY3D(I())
		{
			if (k*k + i*i + j*j > bg_radius2)
			{
				avg += A3D_ELEM(I(), k, i, j);
				n += 1.;
			}
		}
		avg /= n;

		// Calculate stddev in the background pixels
		FOR_ALL_ELEMENTS_IN_ARRAY3D(I())
		{
			if (k*k + i*i + j*j > bg_radius2)
			{
				DOUBLE aux = A3D_ELEM(I(), k, i, j) - avg;
				stddev += aux * aux;
			}
		}
		stddev = sqrt(stddev / n);
	}


	void subtractBackgroundRamp(Image<DOUBLE> &I, int bg_radius)
	{

		int bg_radius2 = bg_radius * bg_radius;
		fit_point3D point;
		std::vector<fit_point3D>  allpoints;
		DOUBLE pA, pB, pC;

		if (I().getDim() == 3)
			REPORT_ERROR("ERROR %% calculateBackgroundRamp is not implemented for 3D data!");

		FOR_ALL_ELEMENTS_IN_ARRAY2D(I())
		{
			if (i*i + j*j > bg_radius2)
			{
				point.x = j;
				point.y = i;
				point.z = A2D_ELEM(I(), i, j);
				point.w = 1.;
				allpoints.push_back(point);
			}
		}

		fitLeastSquaresPlane(allpoints, pA, pB, pC);

		// Substract the plane from the image
		FOR_ALL_ELEMENTS_IN_ARRAY2D(I())
		{
			A2D_ELEM(I(), i, j) -= pA * j + pB * i + pC;
		}

	}


	void removeDust(Image<DOUBLE> &I, bool is_white, DOUBLE thresh, DOUBLE avg, DOUBLE stddev)
	{
		FOR_ALL_ELEMENTS_IN_ARRAY3D(I())
		{
			DOUBLE aux = A3D_ELEM(I(), k, i, j);
			if (is_white && aux - avg > thresh * stddev)
				A3D_ELEM(I(), k, i, j) = rnd_gaus(avg, stddev);
			else if (!is_white && aux - avg < -thresh * stddev)
				A3D_ELEM(I(), k, i, j) = rnd_gaus(avg, stddev);
		}
	}

	void invert_contrast(Image<DOUBLE> &I)
	{
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(I())
		{
			DIRECT_MULTIDIM_ELEM(I(), n) *= -1;
		}
	}

	void rescale(Image<DOUBLE> &I, int mysize)
	{
		int olddim = XSIZE(I());

		resizeMap(I(), mysize);
	}

	void rewindow(Image<DOUBLE> &I, int mysize)
	{
		// Check 2D or 3D dimensionality
		if (I().getDim() == 2)
		{
			I().window(FIRST_XMIPP_INDEX(mysize), FIRST_XMIPP_INDEX(mysize),
				LAST_XMIPP_INDEX(mysize), LAST_XMIPP_INDEX(mysize));
		}
		else if (I().getDim() == 3)
		{
			I().window(FIRST_XMIPP_INDEX(mysize), FIRST_XMIPP_INDEX(mysize), FIRST_XMIPP_INDEX(mysize),
				LAST_XMIPP_INDEX(mysize), LAST_XMIPP_INDEX(mysize), LAST_XMIPP_INDEX(mysize));
		}

	}
}