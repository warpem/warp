// Kaiser-Bessel functions taken from RELION's numerical_recipes.cpp,
// which, in turn, is taken from XMIPP,
// which, according to XMIPP, is taken from Gabor's 1984 paper

#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Helper.cuh"

namespace gtom
{
#define NRSIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))


	double chebev(double a, double b, double c[], int m, double x)
	{
		double d = 0.0, dd = 0.0, sv, y, y2;
		int j;

		if ((x - a)*(x - b) > 0.0)
			throw;
		y2 = 2.0 * (y = (2.0 * x - a - b) / (b - a));
		for (j = m - 1; j >= 1; j--)
		{
			sv = d;
			d = y2 * d - dd + c[j];
			dd = sv;
		}
		return y*d - dd + 0.5*c[0];
	}


#define NUSE1 5
#define NUSE2 5

	void beschb(double x, double *gam1, double *gam2, double *gampl, double *gammi)
	{
		double xx;
		static double c1[] =
		{
			-1.142022680371172e0, 6.516511267076e-3,
			3.08709017308e-4, -3.470626964e-6, 6.943764e-9,
			3.6780e-11, -1.36e-13
		};
		static double c2[] =
		{
			1.843740587300906e0, -0.076852840844786e0,
			1.271927136655e-3, -4.971736704e-6, -3.3126120e-8,
			2.42310e-10, -1.70e-13, -1.0e-15
		};

		xx = 8.0 * x * x - 1.0;
		*gam1 = chebev(-1.0, 1.0, c1, NUSE1, xx);
		*gam2 = chebev(-1.0, 1.0, c2, NUSE2, xx);
		*gampl = *gam2 - x * (*gam1);
		*gammi = *gam2 + x * (*gam1);
	}
#undef NUSE1
#undef NUSE2


#define EPS 1.0e-16
#define FPMIN 1.0e-30
#define MAXIT 10000
#define XMIN 2.0
	void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp)
	{
		int i, isign, l, nl;
		double a, b, br, bi, c, cr, ci, d, del, del1, den, di, dlr, dli, dr, e, f, fact, fact2,
			fact3, ff, gam, gam1, gam2, gammi, gampl, h, p, pimu, pimu2, q, r, rjl,
			rjl1, rjmu, rjp1, rjpl, rjtemp, ry1, rymu, rymup, rytemp, sum, sum1,
			temp, w, x2, xi, xi2, xmu, xmu2;

		if (x <= 0.0 || xnu < 0.0)
			throw;
		nl = (x < XMIN ? (int)(xnu + 0.5) : tmax(0, (int)(xnu - x + 1.5)));
		xmu = xnu - nl;
		xmu2 = xmu * xmu;
		xi = 1.0 / x;
		xi2 = 2.0 * xi;
		w = xi2 / PI;
		isign = 1;
		h = xnu * xi;
		if (h < FPMIN)
			h = FPMIN;
		b = xi2 * xnu;
		d = 0.0;
		c = h;
		for (i = 1; i <= MAXIT; i++)
		{
			b += xi2;
			d = b - d;
			if (fabs(d) < FPMIN)
				d = FPMIN;
			c = b - 1.0 / c;
			if (fabs(c) < FPMIN)
				c = FPMIN;
			d = 1.0 / d;
			del = c * d;
			h = del * h;
			if (d < 0.0)
				isign = -isign;
			if (fabs(del - 1.0) < EPS)
				break;
		}
		if (i > MAXIT)
			throw;
		rjl = isign * FPMIN;
		rjpl = h * rjl;
		rjl1 = rjl;
		rjp1 = rjpl;
		fact = xnu * xi;
		for (l = nl; l >= 1; l--)
		{
			rjtemp = fact * rjl + rjpl;
			fact -= xi;
			rjpl = fact * rjtemp - rjl;
			rjl = rjtemp;
		}
		if (rjl == 0.0)
			rjl = EPS;
		f = rjpl / rjl;
		if (x < XMIN)
		{
			x2 = 0.5 * x;
			pimu = PI * xmu;
			fact = (fabs(pimu) < EPS ? 1.0 : pimu / sin(pimu));
			d = -log(x2);
			e = xmu * d;
			fact2 = (fabs(e) < EPS ? 1.0 : sinh(e) / e);
			beschb(xmu, &gam1, &gam2, &gampl, &gammi);
			ff = 2.0 / PI * fact * (gam1 * cosh(e) + gam2 * fact2 * d);
			e = exp(e);
			p = e / (gampl * PI);
			q = 1.0 / (e * PI * gammi);
			pimu2 = 0.5 * pimu;
			fact3 = (fabs(pimu2) < EPS ? 1.0 : sin(pimu2) / pimu2);
			r = PI * pimu2 * fact3 * fact3;
			c = 1.0;
			d = -x2 * x2;
			sum = ff + r * q;
			sum1 = p;
			for (i = 1; i <= MAXIT; i++)
			{
				ff = (i * ff + p + q) / (i * i - xmu2);
				c *= (d / i);
				p /= (i - xmu);
				q /= (i + xmu);
				del = c * (ff + r * q);
				sum += del;
				del1 = c * p - i * del;
				sum1 += del1;
				if (fabs(del) < (1.0 + fabs(sum))*EPS)
					break;
			}
			if (i > MAXIT)
				throw;
			rymu = -sum;
			ry1 = -sum1 * xi2;
			rymup = xmu * xi * rymu - ry1;
			rjmu = w / (rymup - f * rymu);
		}
		else
		{
			a = 0.25 - xmu2;
			p = -0.5 * xi;
			q = 1.0;
			br = 2.0 * x;
			bi = 2.0;
			fact = a * xi / (p * p + q * q);
			cr = br + q * fact;
			ci = bi + p * fact;
			den = br * br + bi * bi;
			dr = br / den;
			di = -bi / den;
			dlr = cr * dr - ci * di;
			dli = cr * di + ci * dr;
			temp = p * dlr - q * dli;
			q = p * dli + q * dlr;
			p = temp;
			for (i = 2; i <= MAXIT; i++)
			{
				a += 2 * (i - 1);
				bi += 2.0;
				dr = a * dr + br;
				di = a * di + bi;
				if (fabs(dr) + fabs(di) < FPMIN)
					dr = FPMIN;
				fact = a / (cr * cr + ci * ci);
				cr = br + cr * fact;
				ci = bi - ci * fact;
				if (fabs(cr) + fabs(ci) < FPMIN)
					cr = FPMIN;
				den = dr * dr + di * di;
				dr /= den;
				di /= -den;
				dlr = cr * dr - ci * di;
				dli = cr * di + ci * dr;
				temp = p * dlr - q * dli;
				q = p * dli + q * dlr;
				p = temp;
				if (fabs(dlr - 1.0) + fabs(dli) < EPS)
					break;
			}
			if (i > MAXIT)
				throw;
			gam = (p - f) / q;
			rjmu = sqrt(w / ((p - f) * gam + q));
			rjmu = NRSIGN(rjmu, rjl);
			rymu = rjmu * gam;
			rymup = rymu * (p + q / gam);
			ry1 = xmu * xi * rymu - rymup;
		}
		fact = rjmu / rjl;
		*rj = rjl1 * fact;
		*rjp = rjp1 * fact;
		for (i = 1; i <= nl; i++)
		{
			rytemp = (xmu + i) * xi2 * ry1 - rymu;
			rymu = ry1;
			ry1 = rytemp;
		}
		*ry = rymu;
		*ryp = xnu * xi * rymu - ry1;
	}
#undef EPS
#undef FPMIN
#undef MAXIT
#undef XMIN

	double bessj0(double x)
	{
		double ax, z;
		double xx, y, ans, ans1, ans2;

		if ((ax = fabs(x)) < 8.0)
		{
			y = x * x;
			ans1 = 57568490574.0 + y * (-13362590354.0 +
				y * (651619640.7
				+ y * (-11214424.18 +
				y * (77392.33017 +
				y * (-184.9052456)))));
			ans2 = 57568490411.0 + y * (1029532985.0 +
				y * (9494680.718
				+ y * (59272.64853 +
				y * (267.8532712 +
				y * 1.0))));
			ans = ans1 / ans2;
		}
		else
		{
			z = 8.0 / ax;
			y = z * z;
			xx = ax - 0.785398164;
			ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
				+ y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
			ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
				+ y * (-0.6911147651e-5 + y * (0.7621095161e-6
				- y * 0.934935152e-7)));
			ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
		}
		return ans;
	}

	double bessi0(double x)
	{
		double y, ax, ans;
		if ((ax = fabs(x)) < 3.75)
		{
			y = x / 3.75;
			y *= y;
			ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
				+ y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
		}
		else
		{
			y = 3.75 / ax;
			ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
				+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
				+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
				+ y * 0.392377e-2))))))));
		}
		return ans;
	}

	double bessi1(double x)
	{
		double ax, ans;
		double y;
		if ((ax = fabs(x)) < 3.75)
		{
			y = x / 3.75;
			y *= y;
			ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
				+ y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
		}
		else
		{
			y = 3.75 / ax;
			ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
				- y * 0.420059e-2));
			ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
				+ y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
			ans *= (exp(ax) / sqrt(ax));
		}
		return x < 0.0 ? -ans : ans;
	}

	double bessi0_5(double x)
	{
		return (x == 0) ? 0 : sqrt(2 / (PI*x))*sinh(x);
	}

	double bessi1_5(double x)
	{
		return (x == 0) ? 0 : sqrt(2 / (PI*x))*(cosh(x) - sinh(x) / x);
	}

	double bessi2(double x)
	{
		return (x == 0) ? 0 : bessi0(x) - ((2 * 1) / x) * bessi1(x);
	}

	double bessi2_5(double x)
	{
		return (x == 0) ? 0 : bessi0_5(x) - ((2 * 1.5) / x) * bessi1_5(x);
	}

	double bessi3(double x)
	{
		return (x == 0) ? 0 : bessi1(x) - ((2 * 2) / x) * bessi2(x);
	}

	double bessi3_5(double x)
	{
		return (x == 0) ? 0 : bessi1_5(x) - ((2 * 2.5) / x) * bessi2_5(x);
	}

	double bessi4(double x)
	{
		return (x == 0) ? 0 : bessi2(x) - ((2 * 3) / x) * bessi3(x);
	}

	double bessj1_5(double x)
	{
		double rj, ry, rjp, ryp;
		bessjy(x, 1.5, &rj, &ry, &rjp, &ryp);
		return rj;
	}

	double bessj3_5(double x)
	{
		double rj, ry, rjp, ryp;
		bessjy(x, 3.5, &rj, &ry, &rjp, &ryp);
		return rj;
	}

	double kaiser_Fourier_value(double w, double a, double alpha, int m)
	{
		double sigma = sqrt(abs(alpha * alpha - (PI2 * a * w) * (PI2 * a * w)));
		if (m == 2)
		{
			if (PI2 * a * w > alpha)
				return  pow(PI2, 1.5) * pow(a, 3.) * pow(alpha, 2.) * bessj3_5(sigma) / (bessi0(alpha) * pow(sigma, 3.5));
			else
				return  pow(PI2, 1.5) * pow(a, 3.) * pow(alpha, 2.) * bessi3_5(sigma) / (bessi0(alpha) * pow(sigma, 3.5));
		}
		else if (m == 0)
		{
			if (PI2 * a * w > alpha)
				return  pow(PI2, 1.5) * pow(a, 3.) * bessj1_5(sigma) / (bessi0(alpha)*pow(sigma, 1.5));
			else
				return  pow(PI2, 1.5) * pow(a, 3.) * bessi1_5(sigma) / (bessi0(alpha)*pow(sigma, 1.5));
		}
		else
			throw;
	}
}