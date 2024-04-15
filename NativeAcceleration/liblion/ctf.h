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
/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#ifndef _CTF_HH
#define _CTF_HH

#include "multidim_array.h"
#include <map>


namespace relion
{
	class CTF
	{
	protected:

		// Different constants
		DOUBLE K1;
		DOUBLE K2;
		DOUBLE K3;
		DOUBLE K4;

		// Azimuthal angle in radians
		DOUBLE rad_azimuth;

		// defocus_average = (defocus_u + defocus_v)/2
		DOUBLE defocus_average;

		// defocus_deviation = (defocus_u - defocus_v)/2
		DOUBLE defocus_deviation;

		// Phase shift in radians
		DOUBLE rad_phaseshift;

	public:

		/// Accelerating Voltage (in KiloVolts)
		DOUBLE kV;

		/// Defocus in U (in Angstroms). Positive values are underfocused
		DOUBLE DeltafU;

		/// Defocus in V (in Angstroms). Postive values are underfocused
		DOUBLE DeltafV;

		/// Azimuthal angle (between X and U) in degrees
		DOUBLE azimuthal_angle;

		// Electron wavelength (Amstrongs)
		DOUBLE lambda;

		// Radius of the aperture (in micras)
		// DOUBLE aperture;
		/// Spherical aberration (in milimeters). Typical value 5.6
		DOUBLE Cs;

		/// Chromatic aberration (in milimeters). Typical value 2
		DOUBLE Ca;

		/** Mean energy loss (eV) due to interaction with sample.
			Typical value 1*/
		DOUBLE espr;

		/// Objective lens stability (deltaI/I) (ppm). Typical value 1
		DOUBLE ispr;

		/// Convergence cone semiangle (in mrad). Typical value 0.5
		DOUBLE alpha;

		/// Longitudinal mechanical displacement (Angstrom). Typical value 100
		DOUBLE DeltaF;

		/// Transversal mechanical displacement (Angstrom). Typical value 3
		DOUBLE DeltaR;

		/// Amplitude contrast. Typical values 0.07 for cryo, 0.2 for negative stain
		DOUBLE Q0;

		// B-factor fall-off
		DOUBLE Bfac;

		// Phase shift. 0 when not using a phase plate, from 0 to pi otherwise.
		DOUBLE PhaseShift;

		// Overall scale-factor of CTF
		DOUBLE scale;

		/** Empty constructor. */
		CTF() { clear(); }

		/** Just set all values explicitly */
		void setValues(DOUBLE _defU, DOUBLE _defV, DOUBLE _defAng,
			DOUBLE _voltage, DOUBLE _Cs, DOUBLE _Q0, DOUBLE _Bfac, DOUBLE _PhaseShift, DOUBLE _scale);

		/** Write to output. */
		void write(std::ostream &out);

		/// Clear.
		void clear();

		/// Set up the CTF object, read parameters from MetaDataTables with micrograph and particle information
		/// If no MDmic is provided or it does not contain certain parameters, these parameters are tried to be read from MDimg
		void initialise();

		/// Compute CTF at (U,V). Continuous frequencies
		inline DOUBLE getCTF(DOUBLE X, DOUBLE Y,
			bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true) const
		{
			DOUBLE u2 = X * X + Y * Y;
			DOUBLE u = sqrt(u2);
			DOUBLE u4 = u2 * u2;
			// if (u2>=ua2) return 0;
			DOUBLE deltaf = getDeltaF(X, Y);
			//DOUBLE phaseshiftweight = u < (1.0 / 50.0) ? 0.0 : 1.0;
			DOUBLE argument = K1 * deltaf * u2 + K2 * u4 - rad_phaseshift;// * phaseshiftweight;
			DOUBLE retval;
			if (do_intact_until_first_peak && ABS(argument) < PI / 2.)
			{
				retval = 1.;
			}
			/*else if (u < 25.0 / 256.0 / 1.0605)
			{
			retval = 1.;
			}*/
			else
			{
				retval = -(K3*sin(argument) - Q0*cos(argument)); // Q0 should be positive
			}
			if (do_damping)
			{
				DOUBLE E = exp(K4 * u2); // B-factor decay (K4 = -Bfac/4);
				retval *= E;
			}
			if (do_abs)
			{
				retval = ABS(retval);
			}
			else if (do_only_flip_phases)
			{
				retval = (retval < 0.) ? -1. : 1.;
			}
			return scale * retval;
		}

		/// Compute Deltaf at a given direction
		inline DOUBLE getDeltaF(DOUBLE X, DOUBLE Y) const
		{
			if (ABS(X) < XMIPP_EQUAL_ACCURACY &&
				ABS(Y) < XMIPP_EQUAL_ACCURACY)
				return 0;

			DOUBLE ellipsoid_ang = atan2(Y, X) - rad_azimuth;
			/*
			* For a derivation of this formulae confer
			* Principles of Electron Optics page 1380
			* in particular term defocus and twofold axial astigmatism
			* take into account that a1 and a2 are the coefficient
			* of the zernike polynomials difference of defocus at 0
			* and at 45 degrees. In this case a2=0
			*/
			DOUBLE cos_ellipsoid_ang_2 = cos(2 * ellipsoid_ang);
			return (defocus_average + defocus_deviation*cos_ellipsoid_ang_2);

		}

		/// Generate (Fourier-space, i.e. FFTW format) image with all CTF values.
		/// The dimensions of the result array should have been set correctly already
		void getFftwImage(MultidimArray < DOUBLE > &result, int orixdim, int oriydim, DOUBLE angpix,
			bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);

		/// Generate a centered image (with hermitian symmetry)
		void getCenteredImage(MultidimArray < DOUBLE > &result, DOUBLE angpix,
			bool do_abs = false, bool do_only_flip_phases = false, bool do_intact_until_first_peak = false, bool do_damping = true);


	};
}
//@}
#endif
