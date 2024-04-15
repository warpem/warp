#include "gtom/include/Prerequisites.cuh"

namespace gtom
{
	/*int pow(int base, int exponent)
	{
	int result = base;
	for (int i = 0; i < exponent - 1; i++)
	result *= base;
	return result;
	}*/

	// (c) Abramowitz and Stegun, Handbook of Mathematical Functions 
	tfloat gaussrand()
	{
		static tfloat U, V;
		static int phase = 0;
		tfloat Z;

		if (phase == 0) {
			U = (rand() + 1.) / (RAND_MAX + 2.);
			V = rand() / (RAND_MAX + 1.);
			Z = sqrt(-2 * log(U)) * sin(2 * PI * V);
		}
		else
			Z = sqrt(-2 * log(U)) * cos(2 * PI * V);

		phase = 1 - phase;

		return Z;
	}

	void linearfit(tfloat* h_x, tfloat* h_y, uint n, tfloat &a, tfloat &b)
	{
		tfloat sumx = 0, sumy = 0, sumxy = 0, sumx2 = 0;

		for (uint i = 0; i < n; i++)
		{
			sumx += h_x[i];
			sumx2 += h_x[i] * h_x[i];
			sumy += h_y[i];
			sumxy += h_x[i] * h_y[i];
		}

		a = (sumx2 * sumy - sumx * sumxy) / ((tfloat)n * sumx2 - sumx * sumx);
		b = ((tfloat)n * sumxy - sumx * sumy) / ((tfloat)n * sumx2 - sumx * sumx);
	}

	tfloat linearinterpolate(tfloat2 p0, tfloat2 p1, tfloat interpolantx)
	{
		tfloat h_x[2] = { p0.x, p1.x };
		tfloat h_y[2] = { p0.y, p1.y };

		tfloat a = 0, b = 0;
		linearfit(h_x, h_y, 2, a, b);

		return a + interpolantx * b;
	}
}