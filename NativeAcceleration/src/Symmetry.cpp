#include "include/Functions.h"
#include "liblion/liblion.h"
using namespace gtom;


__declspec(dllexport) int __stdcall SymmetryGetNumberOfMatrices(char* c_symmetry)
{
	relion::FileName fn_symmetry(c_symmetry);
	relion::SymList SL;
	SL.read_sym_file(fn_symmetry);

	return SL.SymsNo() + 1;
}

__declspec(dllexport) void __stdcall SymmetryGetMatrices(char* c_symmetry, float* h_matrices)
{
	relion::FileName fn_symmetry(c_symmetry);
	relion::SymList SL;
	SL.read_sym_file(fn_symmetry);

	relion::Matrix2D<DOUBLE> L(4, 4), R(4, 4);
	R(0, 0) = 1;
	R(1, 1) = 1;
	R(2, 2) = 1;

	for (int isym = -1; isym < SL.SymsNo(); isym++)
	{
		if (isym >= 0)
			SL.get_matrices(isym, L, R);

		for (int i = 0; i < 9; i++)
			h_matrices[(isym + 1) * 9 + i] = R(i % 3, i / 3);
	}
}

__declspec(dllexport) void __stdcall SymmetrizeFT(float2* d_data, int3 dims, char* c_symmetry)
{
    relion::FileName fn_symmetry(c_symmetry);
    relion::SymList SL;
    SL.read_sym_file(fn_symmetry);

    if (SL.SymsNo() > 0)
    {
        int3 dimsft = make_int3(ElementsFFT1(dims.x), dims.y, dims.z);
        int rmax2 = dims.x * dims.x / 4 - 1;

        relion::MultidimArray<relion::Complex > my_data;
        my_data.initZeros(dimsft.z, dimsft.y, dimsft.x);
        cudaMemcpy(my_data.data, d_data, ElementsFFT(dims) * sizeof(float2), cudaMemcpyDeviceToHost);

        relion::Matrix2D<DOUBLE> L(4, 4), R(4, 4); // A matrix from the list
        relion::MultidimArray<relion::Complex> sum_data;
        DOUBLE x, y, z, fx, fy, fz, xp, yp, zp, r2;
        bool is_neg_x;
        int x0, x1, y0, y1, z0, z1;
        relion::Complex d000, d001, d010, d011, d100, d101, d110, d111;
        relion::Complex dx00, dx01, dx10, dx11, dxy0, dxy1, dxyz;

        // First symmetry operator (not stored in SL) is the identity matrix
        sum_data.initZeros(dimsft.z, dimsft.y, dimsft.x);
        // Loop over all other symmetry operators
        for (int isym = -1; isym < SL.SymsNo(); isym++)
        {
            if (isym >= 0)
                SL.get_matrices(isym, L, R);

            // Loop over all points in the output (i.e. rotated, or summed) array
            FOR_ALL_ELEMENTS_IN_ARRAY3D(sum_data)
            {
                x = (DOUBLE)j; // STARTINGX(sum_weight) is zero!
                y = i < sum_data.ydim / 2 + 1 ? i : i - sum_data.ydim;
                z = k < sum_data.zdim / 2 + 1 ? k : k - sum_data.zdim;

                r2 = x*x + y*y + z*z;

                if (r2 < rmax2)
                {
                    if (isym >= 0)
                    {
                        // coords_output(x,y) = A * coords_input (xp,yp)
                        xp = x * R(0, 0) + y * R(0, 1) + z * R(0, 2);
                        yp = x * R(1, 0) + y * R(1, 1) + z * R(1, 2);
                        zp = x * R(2, 0) + y * R(2, 1) + z * R(2, 2);
                    }
                    else
                    {
                        xp = x;
                        yp = y;
                        zp = z;
                    }

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

                    // Trilinear interpolation
                    x0 = FLOOR(xp);
                    fx = xp - x0;
                    x1 = x0 + 1;

                    y0 = FLOOR(yp);
                    fy = yp - y0;
                    y1 = y0 + 1;
                    y0 = y0 >= 0 ? y0 : y0 + sum_data.ydim;
                    y1 = y1 >= 0 ? y1 : y1 + sum_data.ydim;

                    z0 = FLOOR(zp);
                    fz = zp - z0;
                    z1 = z0 + 1;
                    z0 = z0 >= 0 ? z0 : z0 + sum_data.zdim;
                    z1 = z1 >= 0 ? z1 : z1 + sum_data.zdim;


                    d000 = DIRECT_A3D_ELEM(my_data, z0, y0, x0);
                    d001 = DIRECT_A3D_ELEM(my_data, z0, y0, x1);
                    d010 = DIRECT_A3D_ELEM(my_data, z0, y1, x0);
                    d011 = DIRECT_A3D_ELEM(my_data, z0, y1, x1);
                    d100 = DIRECT_A3D_ELEM(my_data, z1, y0, x0);
                    d101 = DIRECT_A3D_ELEM(my_data, z1, y0, x1);
                    d110 = DIRECT_A3D_ELEM(my_data, z1, y1, x0);
                    d111 = DIRECT_A3D_ELEM(my_data, z1, y1, x1);

                    dx00 = LIN_INTERP(fx, d000, d001);
                    dx01 = LIN_INTERP(fx, d100, d101);
                    dx10 = LIN_INTERP(fx, d010, d011);
                    dx11 = LIN_INTERP(fx, d110, d111);
                    dxy0 = LIN_INTERP(fy, dx00, dx10);
                    dxy1 = LIN_INTERP(fy, dx01, dx11);

                    dxyz = LIN_INTERP(fz, dxy0, dxy1);

                    // Take complex conjugated for half with negative x
                    if (is_neg_x)
                        A3D_ELEM(sum_data, k, i, j) += relion::conj(dxyz);
                    else
                        A3D_ELEM(sum_data, k, i, j) += dxyz;
                } // end if r2 < my_rmax2

            } // end loop over all elements of sum_weight

        } // end loop over symmetry operators

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sum_data)
        {
            DIRECT_MULTIDIM_ELEM(sum_data, n) = DIRECT_MULTIDIM_ELEM(sum_data, n) / (DOUBLE)(SL.SymsNo() + 1);
        }

        cudaMemcpy(d_data, sum_data.data, ElementsFFT(dims) * sizeof(float2), cudaMemcpyHostToDevice);
    }
}