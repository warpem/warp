#ifndef PCHIP
#define PCHIP

#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/CubicInterp.cuh"

namespace gtom
{
	Cubic1D::Cubic1D(std::vector<tfloat2> data)
	{
		Data = data;
		for (int i = 0; i < data.size(); i++)
			Breaks.push_back(data[i].x);
		Coefficients.resize(data.size() - 1);

		std::vector<tfloat> h = Diff(Breaks);
		std::vector<tfloat> del(data.size() - 1);
		for (int i = 0; i < data.size() - 1; i++)
			del.push_back((data[i + 1].y - data[i].y) / h[i]);

		std::vector<tfloat> slopes = GetPCHIPSlopes(data, del);

		std::vector<tfloat> dzzdx = std::vector<tfloat>(del.size());
		for (int i = 0; i < dzzdx.size(); i++)
			dzzdx[i] = (del[i] - slopes[i]) / h[i];

		std::vector<tfloat> dzdxdx = std::vector<tfloat>(del.size());
		for (int i = 0; i < dzdxdx.size(); i++)
			dzdxdx[i] = (slopes[i + 1] - del[i]) / h[i];

		for (int i = 0; i < Coefficients.size(); i++)
			Coefficients[i] = tfloat4((dzdxdx[i] - dzzdx[i]) / h[i],
										2.0f * dzzdx[i] - dzdxdx[i],
										slopes[i],
										data[i].y);
	}

	std::vector<tfloat> Cubic1D::Interp(std::vector<tfloat> x)
	{
		std::vector<tfloat> y = std::vector<tfloat>(x.size());

		std::vector<tfloat> b = Breaks;
		std::vector<tfloat4> c = Coefficients;

		std::vector<int> indices = std::vector<int>(x.size());
		for (int i = 0; i < x.size(); i++)
		{
			if (x[i] < b[1])
				indices[i] = 0;
			else if (x[i] >= b[b.size() - 2])
				indices[i] = b.size() - 2;
			else
				for (int j = 2; j < b.size() - 1; j++)
					if (x[i] < b[j])
					{
						indices[i] = j - 1;
						break;
					}
		}

		std::vector<tfloat> xs = std::vector<tfloat>(x.size());
		for (int i = 0; i < xs.size(); i++)
			xs[i] = x[i] - b[indices[i]];

		for (int i = 0; i < x.size(); i++)
		{
			int index = indices[i];
			float v = c[index].x;
			v = xs[i] * v + c[index].y;
			v = xs[i] * v + c[index].z;
			v = xs[i] * v + c[index].w;

			y[i] = v;
		}

		return y;
	}

	std::vector<tfloat> Cubic1D::GetPCHIPSlopes(std::vector<tfloat2> data, std::vector<tfloat> del)
	{
		if (data.size() == 2)
			return std::vector<tfloat> { del[0], del[0] };   // Do only linear

		std::vector<tfloat> d = std::vector<tfloat>(data.size());
		std::vector<tfloat> h = Diff(Breaks);
		for (int k = 0; k < del.size() - 1; k++)
		{
			if (del[k] * del[k + 1] <= 0)
				continue;

			float hs = h[k] + h[k + 1];
			float w1 = (h[k] + hs) / (3 * hs);
			float w2 = (hs + h[k + 1]) / (3 * hs);
			float dmax = tmax(abs(del[k]), abs(del[k + 1]));
			float dmin = tmin(abs(del[k]), abs(del[k + 1]));
			d[k + 1] = dmin / (w1 * (del[k] / dmax) + w2 * (del[k + 1] / dmax));
		}

		d[0] = ((2 * h[0] + h[1]) * del[0] - h[0] * del[1]) / (h[0] + h[1]);
		if (sgn(d[0]) != sgn(del[0]))
			d[0] = 0;
		else if (sgn(del[0]) != sgn(del[1]) && abs(d[0]) > abs(3 * del[0]))
			d[0] = 3 * del[0];

		int n = d.size() - 1;
		d[n] = ((2 * h[n - 1] + h[n - 2]) * del[n - 1] - h[n - 1] * del[n - 2]) / (h[n - 1] + h[n - 2]);
		if (sgn(d[n]) != sgn(del[n - 1]))
			d[n] = 0;
		else if (sgn(del[n - 1]) != sgn(del[n - 2]) && abs(d[n]) > abs(3 * del[n - 1]))
			d[n] = 3 * del[n - 1];

		return d;
	}

	std::vector<tfloat> Cubic1D::Diff(std::vector<tfloat> series)
	{
		std::vector<tfloat> result(series.size() - 1);

		for (int i = 0; i < series.size() - 1; i++)
			result.push_back(series[i + 1] - series[i]);

		return result;
	}
}

#endif