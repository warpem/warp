#include "cufft.h"
#include "Prerequisites.cuh"

#ifndef CORRELATION_CUH
#define CORRELATION_CUH

namespace gtom
{
	///////////////
	//Correlation//
	///////////////

	//CCF.cu:

	/**
	* \brief Computes the correlation surface by folding two maps in Fourier space; an FFTShift operation is applied afterwards, so that center = no translation
	* \param[in] d_input1	First input map
	* \param[in] d_input2	Second input map
	* \param[in] d_output	Array that will contain the correlation data
	* \param[in] dims	Array dimensions
	* \param[in] normalized	Indicates if the input maps are already normalized, i. e. if this step can be skipped
	* \param[in] d_mask	Optional mask used during normalization, if normalized = false
	* \param[in] batch	Number of map pairs
	*/
	template<class T> void d_CCF(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);

	/**
	* \brief Computes the correlation surface by folding two maps in Fourier space; FFTShift operation is not applied afterwards, i. e. first element in array = no translation
	* \param[in] d_input1	First input map
	* \param[in] d_input2	Second input map
	* \param[in] d_output	Array that will contain the correlation data
	* \param[in] dims	Array dimensions
	* \param[in] normalized	Indicates if the input maps are already normalized, i. e. if this step can be skipped
	* \param[in] d_mask	Optional mask used during normalization, if normalized = false
	* \param[in] batch	Number of map pairs
	*/
	template<class T> void d_CCFUnshifted(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);

	//Peak.cu:

	/**
	* \brief Specifies how the position of a peak should be determined
	*/
	enum T_PEAK_MODE
	{
		/**Only integer values; fastest*/
		T_PEAK_INTEGER = 1,
		/**Subpixel precision, but with x, y and z determined by scaling a row/column independently in each dimension; moderately fast*/
		T_PEAK_SUBCOARSE = 2,
		/**Subpixel precision, with a portion around the peak extracted and up-scaled in Fourier space; slow, but highest precision*/
		T_PEAK_SUBFINE = 3
	};

	/**
	* \brief Locates the position of the maximum value in a map with the specified precision
	* \param[in] d_input	Array with input data
	* \param[in] d_positions	Array that will contain the peak position for each map in batch
	* \param[in] d_values	Array that will contain the peak values for each map in batch
	* \param[in] dims	Array dimensions
	* \param[in] mode	Desired positional precision
	* \param[in] planforw	Optional pre-cooked forward FFT plan; can be made with d_PeakMakePlans
	* \param[in] planback	Optional pre-cooked reverse FFT plan; can be made with d_PeakMakePlans
	* \param[in] batch	Number of maps
	*/
	void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, cufftHandle* planforw = (cufftHandle*)NULL, cufftHandle* planback = (cufftHandle*)NULL, int batch = 1);

	void d_PeakOne2D(tfloat* d_input, float3* d_positions, tfloat* d_values, int2 dims, int2 dimsregion, bool subtractcenter, int batch = 1);


	//LocalPeaks.cu:

	/**
	* \brief Detects multiple local peaks in a map
	* \param[in] d_input	Array with input data
	* \param[in] h_peaks	Pointer that will contain a host array with peak positions
	* \param[in] h_peaksnum	Host array that will contain the number of peaks in each map
	* \param[in] localextent	Distance that a peak has to be apart from a higher/equal peak to be detected
	* \param[in] threshold	Minimum value for peaks to be considered
	* \param[in] batch	Number of maps
	*/
	void d_LocalPeaks(tfloat* d_input, int3** h_peaks, int* h_peaksnum, int3 dims, int localextent, tfloat threshold, int batch = 1);

	void d_SubpixelMax(tfloat* d_input, tfloat* d_output, int3 dims, int subpixsteps);

	//Realspace.cu:

	void d_CorrelateRealspace(tfloat* d_image1, tfloat* d_image2, int3 dims, tfloat* d_mask, tfloat* d_corr, uint batch);

	//SimilarityMatrix.cu:

	void d_RotationSeries(tfloat* d_image, tfloat* d_series, int2 dimsimage, int anglesteps);
	void d_SimilarityMatrixRow(tfloat* d_images, tcomplex* d_imagesft, int2 dimsimage, int nimages, int anglesteps, int target, tfloat* d_similarity);
	void d_LineSimilarityMatrixRow(tcomplex* d_linesft, int2 dimsimage, int nimages, int linewidth, int anglesteps, int target, tfloat* d_similarity);

	// SubTomograms.cu:

	void d_PickSubTomograms(cudaTex t_projectordataRe,
							cudaTex t_projectordataIm,
							tfloat projectoroversample,
							int3 dimsprojector,
							tcomplex* d_experimentalft,
							tfloat* d_ctf,
							int3 dimsvolume,
							uint nvolumes,
							tfloat3* h_angles,
							uint nangles,
							tfloat maskradius,
							tfloat* d_bestcorrelation,
							float* d_bestangle,
							float* h_progressfraction = NULL);

	void d_PickSubTomogramsDiff2(cudaTex t_projectordataRe,
								cudaTex t_projectordataIm,
								tfloat projectoroversample,
								int3 dimsprojector,
								tcomplex* d_experimentalft,
								tfloat* d_ctf,
								int3 dimsvolume,
								uint nvolumes,
								int3 dimsrelevant,
								tfloat3* h_angles,
								uint nangles,
								tfloat* d_bestdiff2,
								float* d_bestangle);

	//Picker.cu:

	struct Peak
	{
		tfloat3 position;
		uint ref;
		tfloat3 angles;
		tfloat fom;
		tfloat relativefom;
	};

	class Picker
	{
	public:
		int3 dimsref, dimsimage;
		uint ndims;
		
		cufftHandle planforw, planback, planrefback;

		tcomplex* d_imageft, *d_image2ft;
		tfloat* d_ctf;

		cudaArray_t a_maskRe, a_maskIm;
		cudaTex t_maskRe, t_maskIm;
		tcomplex* d_maskft;

		tfloat* d_imagesum1, *d_imagesum2;

		cudaArray_t a_refRe, a_refIm;
		cudaTex t_refRe, t_refIm;
		tcomplex* d_refft;
		tfloat* d_refpadded;

		Picker();
		~Picker();

		void Initialize(tfloat* _d_ref, int3 _dimsref, tfloat* _d_refmask, int3 _dimsimage);
		void SetImage(tfloat* _d_image, tfloat* _d_ctf);
		void PerformCorrelation(tfloat anglestep, tfloat* d_bestccf, tfloat3* d_bestangle);
	};

	class ProbabilisticPicker
	{
	private:
		void CalcSolventStatistics(tcomplex* d_imageft, tcomplex* d_image2ft, tcomplex* d_solventmaskft, tfloat solventsamples, tfloat* d_solventmean, tfloat* d_solventstd);

	public:
		int3 dimsref, dimsimage, dimslowpass;
		uint ndims;
		uint nrefs;
		bool ismaskcircular;
		bool doctf;

		tfloat* h_masksum, *h_invmasksum;

		tfloat* h_refsRe, *h_refsIm;

		tfloat* h_masks;
		tcomplex* h_invmasksft;

		cufftHandle planforw, planback;

		tcomplex* d_imageft, *d_image2ft;
		tfloat* d_ctf;

		tfloat* d_maskpadded, *d_maskcropped, *d_invmask;
		tcomplex* d_maskft, *d_invmaskft;

		tfloat* d_solventmean, *d_solventstd;

		tcomplex* d_refft, *d_reflowft;
		tfloat* d_refcropped;

		tfloat* d_buffer1, *d_buffer2;

		ProbabilisticPicker();
		~ProbabilisticPicker();
		
		void Initialize(tfloat* _d_refs, int3 _dimsref, uint _nrefs, tfloat* _d_refmasks, bool _ismaskcircular, bool _doctf, int3 _dimsimage, uint _lowpassfreq);
		std::vector<Peak> PickImage(tfloat* d_image, tfloat* d_ctf, tfloat anglestep, tfloat* d_out_bestccf = NULL, tfloat* d_out_bestpsi = NULL);
		void SetImage(tfloat* _d_image, tfloat* _d_ctf);
		void PerformCorrelation(uint n, tfloat anglestep, tfloat* d_bestccf, tfloat3* d_bestangle, int* d_bestref);
	};

	class TomoPicker
	{
	private:
		void CalcSolventStatistics(tfloat solventsamples);

	public:
		int3 dimsref;
		int3 dimsrefpadded;
		int2 dimsimage;
		uint nimages;
		bool ismaskcircular;

		tfloat* d_masksum;

		tfloat* d_refRe, *d_refIm;
		cudaArray_t a_refRe, a_refIm;
		cudaTex t_refRe, t_refIm;
		tcomplex* d_refrotated;

		tfloat* d_maskRe, *d_maskIm;
		cudaArray_t a_maskRe, a_maskIm;
		cudaTex t_maskRe, t_maskIm;
		tcomplex* d_maskrotated;

		tcomplex* d_ref2dft, *d_mask2dft;
		tfloat* d_ref2d, *d_mask2d;
		tfloat* d_ref2dcropped, *d_mask2dcropped;

		cufftHandle planimageforw, planimageback;
		cufftHandle planrefback;

		tcomplex* d_imageft, *d_image2ft;
		tcomplex* d_maskpaddedft, *d_maskpaddedcorr;
		tfloat* d_maskpadded;

		tfloat* d_image;
		tfloat* d_imagesum1, *d_imagesum2;
		tfloat* d_imagecorr;

		tfloat3* h_imageangles;
		tfloat* d_imageweights;

		TomoPicker();
		~TomoPicker();

		void Initialize(tfloat* _d_ref, int3 _dimsref, tfloat* _d_refmask, bool _ismaskcircular, int2 _dimsimage, uint _nimages);
		void SetImage(tfloat* _d_image, tfloat3* _h_imageangles, tfloat* _h_imageweights);
		void PerformCorrelation(tfloat* d_corr, tfloat3* d_corrangle, int3 dimscorr, tfloat anglestep);
	};
}
#endif