#include "gtom/include/Prerequisites.cuh"
#include "gtom/include/Generics.cuh"
#include "gtom/include/Helper.cuh"
#include "gtom/include/Masking.cuh"


namespace gtom
{
	//////////////////////////////////////
	//Equivalent of TOM's tom_dev method//
	//////////////////////////////////////

	template <class Tmask> void d_Dev(tfloat* d_input, imgstats5* d_output, size_t elements, Tmask* d_mask, int batch)
	{
		size_t denseelements = elements;
		tfloat* d_denseinput = d_input;
		if (d_mask != NULL)
		{
			size_t* d_mapforward = NULL;
			d_MaskSparseToDense(d_mask, &d_mapforward, NULL, denseelements, elements);
			if (denseelements == 0)
				throw;

			tfloat* d_remapped;
			cudaMalloc((void**)&d_remapped, denseelements * batch * sizeof(tfloat));
			d_Remap(d_input, d_mapforward, d_remapped, denseelements, elements, (tfloat)0, batch);
			d_denseinput = d_remapped;
		}

		tfloat* d_mins;
		cudaMalloc((void**)&d_mins, batch * sizeof(tfloat));
		tfloat* d_maxs;
		cudaMalloc((void**)&d_maxs, batch * sizeof(tfloat));
		tfloat* d_means;
		cudaMalloc((void**)&d_means, batch * sizeof(tfloat));
		tfloat* d_meancentered;
		cudaMalloc((void**)&d_meancentered, denseelements * batch * sizeof(tfloat));
		tfloat* d_vars;
		cudaMalloc((void**)&d_vars, batch * sizeof(tfloat));
		tfloat* d_devs;
		cudaMalloc((void**)&d_devs, batch * sizeof(tfloat));

		d_SumMinMax(d_denseinput, d_means, d_mins, d_maxs, denseelements, batch);
		d_MultiplyByScalar(d_means, d_means, batch, (tfloat)1 / (tfloat)denseelements);

		d_SquaredDistanceFromScalar(d_denseinput, d_means, d_meancentered, denseelements, batch);

		d_Sum(d_meancentered, d_vars, denseelements, batch);
		d_MultiplyByScalar(d_vars, d_vars, batch, (tfloat)1 / (tfloat)denseelements);

		d_Sqrt(d_vars, d_devs, batch);

		tfloat** h_fields = (tfloat**)malloc(5 * sizeof(tfloat*));
		h_fields[0] = d_means;
		h_fields[1] = d_mins;
		h_fields[2] = d_maxs;
		h_fields[3] = d_devs;
		h_fields[4] = d_vars;
		tfloat** d_fields = (tfloat**)CudaMallocFromHostArray(h_fields, 5 * sizeof(tfloat*));

		d_JoinInterleaved<tfloat, 5>(d_fields, (tfloat*)d_output, batch);

		if (d_denseinput != d_input)
			cudaFree(d_denseinput);
		cudaFree(d_fields);
		cudaFree(d_means);
		cudaFree(d_mins);
		cudaFree(d_maxs);
		cudaFree(d_meancentered);
		cudaFree(d_vars);
		cudaFree(d_devs);
	}
	template void d_Dev<tfloat>(tfloat* d_input, imgstats5* d_output, size_t elements, tfloat* d_mask, int batch);
	template void d_Dev<int>(tfloat* d_input, imgstats5* d_output, size_t elements, int* d_mask, int batch);
	template void d_Dev<char>(tfloat* d_input, imgstats5* d_output, size_t elements, char* d_mask, int batch);
	template void d_Dev<bool>(tfloat* d_input, imgstats5* d_output, size_t elements, bool* d_mask, int batch);
}