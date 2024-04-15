#include "Prerequisites.cuh"

#ifndef BINARY_CUH
#define BINARY_CUH

namespace gtom
{
	///////////////////////
	//Binary manipulation//
	///////////////////////

	/**
	* \brief Performs a dilation operation on a binary image/volume
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the dilated data
	* \param[in] dims	Array dimensions
	* \param[in] batch	Number of arrays
	*/
	template <class T> void d_Dilate(T* d_input, T* d_output, int3 dims, int batch = 1);

	/**
	* \brief Calculates the distance between an unmasked voxel and the closest masked voxel using a BFS. May be not exact.
	* \param[in] d_input	Array with binary input data
	* \param[in] d_output	Array that will contain the distance values
	* \param[in] dims	Array dimensions
	* \param[in] maxiterations	Number of iterations to run the algorithm, usually the expected maximum distance for that volume
	*/
	void d_DistanceMap(tfloat* d_input, tfloat* d_output, int3 dims, int maxiterations);

	/**
	* \brief Calculates the distance between an unmasked voxel and the closest masked voxel using a brute-force check around each voxel
	* \param[in] d_input	Array with binary input data
	* \param[in] d_output	Array that will contain the distance values
	* \param[in] dims	Array dimensions
	* \param[in] maxdistance	Maximum distance to search within, since complexity scales with n^3
	*/
	void d_DistanceMapExact(tfloat* d_input, tfloat* d_output, int3 dims, int maxdistance);

	/**
	* \brief Converts floating point data to binary by applying a threshold; value >= threshold is set to 1, otherwise 0; binary data type can be char or int
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the binarized data
	* \param[in] elements	Number of elements in array
	* \param[in] threshold	Threshold to be applied
	* \param[in] batch	Number of arrays
	*/
	template <class T> void d_Binarize(tfloat* d_input, T* d_output, size_t elements, tfloat threshold, int batch = 1);
}
#endif