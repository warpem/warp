// C# port of Jasenko Zivanov's Zernike.cpp from RELION 3.1
// https://github.com/3dem/relion
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Warp.Tools
{
	public static class Zernike
	{
		static double[][][] R_coeffs = new double[0][][];

		static public double Z(int m, int n, double rho, double phi)
		{
			if (m >= 0)
			{
				return R(m, n, rho) * Math.Cos(m * phi);
			}
			else
			{
				return R(-m, n, rho) * Math.Sin(-m * phi);
			}
		}

		static public double Zcart(int m, int n, double x, double y)
		{
			if (x == 0 && y == 0)
			{
				return Z(m, n, Math.Sqrt(x * x + y * y), 0.0);
			}
			else
			{
				return Z(m, n, Math.Sqrt(x * x + y * y), Math.Atan2(y, x));
			}
		}

		static public double R(int m, int n, double rho)
		{
			if (m > n)
				throw new Exception($"Zernike::R: illegal argument: m = {m}, n = {n}.");

			if ((n - m) % 2 == 1) return 0.0;

			if (R_coeffs.Length <= n)
				PrepCoeffs(n);

			double Result = 0.0;

			for (int k = 0; k <= (n - m) / 2; k++)
				Result += R_coeffs[n][m][k] * Math.Pow(rho, n - 2 * k);

			return Result;
		}

		static public void EvenIndexToMN(int i, out int m, out int n)
		{
			int k = (int)Math.Sqrt((double)i);

			m = 2 * (i - k * k - k);
			n = 2 * k;
		}

		static public int NumberOfEvenCoeffs(int n_max)
		{
			int l = n_max / 2;
			return l * l + 2 * l + 1;
		}

		static public void OddIndexToMN(int i, out int m, out int n)
		{
			int k = (int)((Math.Sqrt(1 + 4 * i) - 1.0) / 2.0);
			int i0 = k * k + k;

			n = 2 * k + 1;
			m = 2 * (i - i0) - n;
		}

		static public int NumberOfOddCoeffs(int n_max)
		{
			int l = (n_max - 1) / 2 + 1;
			return l * l + l;
		}

		static double Factorial(int k)
		{
			double Result = 1.0;

			for (int i = 2; i <= k; i++)
				Result *= (double)i;

			return Result;
		}

		static void PrepCoeffs(int n)
		{
			double[][][] newCoeffs = new double[n + 1][][];

			for (int nn = 0; nn < R_coeffs.Length; nn++)
				newCoeffs[nn] = R_coeffs[nn];

			for (int nn = R_coeffs.Length; nn <= n; nn++)
			{
				newCoeffs[nn] = new double[nn + 1][];

				for (int m = 0; m <= nn; m++)
				{
					if ((nn - m) % 2 == 1) continue;

					newCoeffs[nn][m] = new double[(nn - m) / 2 + 1];

					for (int k = 0; k <= (nn - m) / 2; k++)
					{
						newCoeffs[nn][m][k] = (1 - 2 * (k % 2)) * Factorial(nn - k) /
											  (Factorial(k) * Factorial((nn + m) / 2 - k) * Factorial((nn - m) / 2 - k));
					}
				}
			}

			R_coeffs = newCoeffs;
		}
	}
}
