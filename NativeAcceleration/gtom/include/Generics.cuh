#include "Prerequisites.cuh"

#ifndef GENERICS_CUH
#define GENERICS_CUH

namespace gtom
{
	////////////
	//Generics//
	////////////

	//Arithmetics.cu:

	/**
	 * \brief Multiplies every input element by the same scalar.
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] multiplicator	Multiplicator used for every operation
	 */
	template <class T> void d_MultiplyByScalar(T* d_input, T* d_output, size_t elements, T multiplicator);

	/**
	 * \brief Multiplies every element of the nth vector by the nth scalar
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	template <class T> void d_MultiplyByScalar(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of two vectors
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	template <class T> void d_MultiplyByVector(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Multiplies every input element by the same non-complex scalar.
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] multiplicator	Multiplicator used for every operation
	 */
	void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator);

	/**
	 * \brief Multiplies every element of the nth vector by the nth non-complex scalar
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByScalar(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of a complex vector by a non-complex vector
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByVector(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Multiplies every input element by the same complex scalar.
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] multiplicator	Multiplicator used for every operation
	 */
	void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);

	/**
	 * \brief Multiplies every element of the nth vector by the nth complex scalar
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of two complex vectors
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);
	
	/**
	* \brief Performs element-wise division of complex numbers by scalars
	* \param[in] d_input	Array with numbers to be divided
	* \param[in] d_multiplicators	Array with divisors for the corresponding elements in d_input
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements per vector
	* \param[in] batch	Number of vectors to be divided
	*/
	void d_ComplexDivideByVector(tcomplex* d_input, tfloat* d_divisors, tcomplex* d_output, size_t elements, int batch = 1);

	/**
	* \brief Performs element-wise division of complex numbers by scalars, setting an output element to 0 if a division by 0 occurs
	* \param[in] d_input	Array with numbers to be divided
	* \param[in] d_multiplicators	Array with divisors for the corresponding elements in d_input
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements per vector
	* \param[in] batch	Number of vectors to be divided
	*/
	void d_ComplexDivideSafeByVector(tcomplex* d_input, tfloat* d_divisors, tcomplex* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Multiplies every input element by the conjugate of the same complex scalar.
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] multiplicator	Multiplicator used for every operation
	 */
	void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);

	/**
	 * \brief Multiplies every element of the nth vector by the conjugate of the nth complex scalar
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of a complex vector by the conjugate of another complex vector
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be multiplied
	 */
	void d_ComplexMultiplyByConjVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Divides every input element by the same scalar.
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] divisor	Divisor used for every operation
	 */
	template <class T> void d_DivideByScalar(T* d_input, T* d_output, size_t elements, T divisor);

	/**
	 * \brief Divides every element of the nth vector by the nth scalar
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_divisors	Array with scalar divisors for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be divided
	 */
	template <class T> void d_DivideByScalar(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of two vectors
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_divisors	Array with divisors for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be divided
	 */
	template <class T> void d_DivideByVector(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Performs element-wise multiplication of two vectors; if division by 0 occurs, 0 is written to the result instead of NaN
	 * \param[in] d_input	Array with numbers to be multiplied
	 * \param[in] d_divisors	Array with divisors for the corresponding elements in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be divided
	 */
	template <class T> void d_DivideSafeByVector(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Adds the same scalar to every input element.
	 * \param[in] d_input	Array with numbers to be incremented
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] summand	Summand used for every operation
	 */
	template <class T> void d_AddScalar(T* d_input, T* d_output, size_t elements, T summand);

	/**
	 * \brief Adds the nth summand to all elements of the nth input vector
	 * \param[in] d_input	Array with numbers to be incremented
	 * \param[in] d_summands	Array with scalar summands for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be incremented
	 */
	template <class T> void d_AddScalar(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Adds a vector to all input vectors
	 * \param[in] d_input	Array with numbers to be incremented
	 * \param[in] d_summands	Array with vector summand
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be incremented
	 */
	template <class T> void d_AddVector(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Subtracts the same scalar from every input element.
	 * \param[in] d_input	Array with numbers to be decremented
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] subtrahend	Subtrahend used for every operation
	 */
	template <class T> void d_SubtractScalar(T* d_input, T* d_output, size_t elements, T subtrahend);

	/**
	 * \brief Subtracts the nth subtrahend from all elements of the nth input vector
	 * \param[in] d_input	Array with numbers to be decremented
	 * \param[in] d_subtrahends	Array with scalar subtrahends for the corresponding vectors in d_input
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be decremented
	 */
	template <class T> void d_SubtractScalar(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Subtracts a vector from all input vectors
	 * \param[in] d_input	Array with numbers to be decremented
	 * \param[in] d_subtrahends	Array with vector subtrahend
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements per vector
	 * \param[in] batch	Number of vectors to be decremented
	 */
	template <class T> void d_SubtractVector(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);


	/**
	 * \brief Computes the square root of every input element
	 * \param[in] d_input	Array with numbers to be raised to ^1/2
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	template <class T> void d_Sqrt(T* d_input, T* d_output, size_t elements);

	/**
	 * \brief Computes the square of every input element
	 * \param[in] d_input	Array with numbers to be raised to ^2
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	template <class T> void d_Square(T* d_input, T* d_output, size_t elements);

	/**
	 * \brief Raises every input element to the same power
	 * \param[in] d_input	Array with numbers to be raised
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] exponent	Power to raise every element to
	 */
	template <class T> void d_Pow(T* d_input, T* d_output, size_t elements, T exponent);

	/**
	 * \brief Computes the absolute value (magnitude) of every input element
	 * \param[in] d_input	Array with numbers to be made absolute
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	template <class T> void d_Abs(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes the absolute value (magnitude) of every complex input element
	* \param[in] d_input	Array with complex numbers
	* \param[in] d_output	Array that will contain the result; d_output == d_input is not valid
	* \param[in] elements	Number of elements
	*/
	void d_Abs(tcomplex* d_input, tfloat* d_output, size_t elements);

	/**
	 * \brief Computes the inverse (1/value) of every input element
	 * \param[in] d_input	Array with numbers to be inversed
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	template <class T> void d_Inv(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes the logarithm of every input element
	* \param[in] d_input	Array with positive numbers
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_Log(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes the exponent function of every input element
	* \param[in] d_input	Array with input values
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_Exp(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes (1 - x) for every input element x
	* \param[in] d_input	Array with input values
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_OneMinus(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes sign(x) for every input element x
	* \param[in] d_input	Array with input values
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_Sign(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes cos(x) for every input element x
	* \param[in] d_input	Array with input values
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_Cos(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes sin(x) for every input element x
	* \param[in] d_input	Array with input values
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_Sin(T* d_input, T* d_output, size_t elements);

	/**
	* \brief Computes a * b + c for every input
	* \param[in] d_mult1	Array with a
	* \param[in] d_mult2	Array with b
	* \param[in] d_summand	Array with c
	* \param[in] d_output	Array that will contain the result; d_output == d_mult1/d_mult2/d_summand is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_MultiplyAdd(T* d_mult1, T* d_mult2, T* d_summand, T* d_output, size_t elements);

	/**
	 * \brief Transforms every complex input element from polar to cartesian form
	 * \param[in] d_input	Array with numbers in polar form
	 * \param[in] d_cart	Array that will contain the cartesian form; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	void d_ComplexPolarToCart(tcomplex* d_polar, tcomplex* d_cart, size_t elements);

	/**
	 * \brief Transforms every complex input element from cartesian to polar form
	 * \param[in] d_input	Array with numbers in cartesian form
	 * \param[in] d_cart	Array that will contain the polar form; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	void d_ComplexCartToPolar(tcomplex* d_cart, tcomplex* d_polar, size_t elements);

	/**
	* \brief Normalizes the length of every complex input element to 1, essentially reducing it to its phase information
	* \param[in] d_input	Array with unnormalized numbers
	* \param[in] d_output	Array that will contain the normalized numbers; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	void d_ComplexNormalize(tcomplex* d_input, tcomplex* d_output, size_t elements);


	/**
	 * \brief For each pair n of input elements, max(input1[n], input2[n]) is written to output
	 * \param[in] d_input1	Array with the first numbers in each pair
	 * \param[in] d_input2	Array with the second numbers in each pair
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 */
	template <class T> void d_MaxOp(T* d_input1, T* d_input2, T* d_output, size_t elements);


	/**
	* \brief For each input element, max(input1[n], input2) is written to output
	* \param[in] d_input1	Array with the first numbers in each pair
	* \param[in] input2		Scalar number to compare with
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_MaxOp(T* d_input1, T input2, T* d_output, size_t elements);

	/**
	* \brief For each pair n of input elements, min(input1[n], input2[n]) is written to output
	* \param[in] d_input1	Array with the first numbers in each pair
	* \param[in] d_input2	Array with the second numbers in each pair
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_MinOp(T* d_input1, T* d_input2, T* d_output, size_t elements);

	/**
	* \brief For each input element, min(input1[n], input2) is written to output
	* \param[in] d_input1	Array with the first numbers in each pair
	* \param[in] d_input2	Scalar number to compare with
	* \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	* \param[in] elements	Number of elements
	*/
	template <class T> void d_MinOp(T* d_input1, T input2, T* d_output, size_t elements);


	/**
	 * \brief Computes the smallest power of 2 that is >= x
	 * \param[in] x	Lower limit for the power of 2
	 * \returns Next power of 2
	 */
	size_t NextPow2(size_t x);

	/**
	 * \brief Determines if a number is a power of 2
	 * \param[in] x	Number in question
	 * \returns True if x is power of 2
	 */
	bool IsPow2(size_t x);

	//Boolean.cu:

	/**
	* \brief Calculates the OR conjunction of all elements
	* \param[in] d_input	Array with input elements to be combined
	* \param[in] d_output	Array that will contain the results
	* \param[in] n	Number of elements
	* \param[in] batch	Number of groups of elements
	*/
	template <class T> void d_Or(T* d_input, T* d_output, uint n, uint batch = 1);

	/**
	* \brief Calculates the AND conjunction of all elements
	* \param[in] d_input	Array with input elements to be combined
	* \param[in] d_output	Array that will contain the results
	* \param[in] n	Number of elements
	* \param[in] batch	Number of groups of elements
	*/
	template <class T> void d_And(T* d_input, T* d_output, uint n, uint batch = 1);

	//CompositeArithmetics.cu:

	/**
	 * \brief Computes (n - a)^2 for every input element n and the fixed scalar a
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] scalar	The scalar substracted from every element before squaring
	 */
	template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_output, size_t elements, T scalar);

	/**
	 * \brief Computes (n - a)^2 for every element n in input vector and the per-vector fixed scalar a
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_scalars	The scalar substracted from every vector element before squaring
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_scalars, T* d_output, size_t elements, int batch = 1);

	/**
	 * \brief Computes (n - a).^2 for every input vector n and fixed vector a (.^2 = element-wise square)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_vector	The vector substracted from every input vector before squaring
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
	 * \param[in] elements	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_SquaredDistanceFromVector(T* d_input, T* d_vector, T* d_output, size_t elements, int batch = 1);

	//Histogram.cu:

	/**
	 * \brief Bins input elements fitting [minval; maxval] into n bins and outputs bin sizes; bin centers are (n + 0.5) * (maxval - minval) / nbins + minval
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_histogram	Array that will contain the histogram; d_histogram == d_input is not valid
	 * \param[in] elements	Number of elements
	 * \param[in] nbins	Number of bins
	 * \param[in] minval	Elements >= minval are considered
	 * \param[in] maxval	Elements <= maxval are considered
	 * \param[in] batch	Number of input populations/histograms
	 */
	template<class T> void d_Histogram(T* d_input, uint* d_histogram, size_t elements, int nbins, T minval, T maxval, int batch = 1);

	//IndexOf.cu:

	/**
	 * \brief Finds the position of the first occurrence of a value; if no integer positions match, linear interpolation is performed
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the position; 0 if all values are smaller, elements if all values are larger, -1 if NaN is encountered
	 * \param[in] elements	Number of elements
	 * \param[in] value	Value to be found
	 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR is supported
	 * \param[in] batch	Number of input vectors/output positions
	 */
	void d_FirstIndexOf(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value, T_INTERP_MODE mode, int batch = 1);

	/**
	* \brief Finds the position of the first intersection between FSC curve and the 1/2 bit function (van Heel 2005), which depends on the fraction occupied by actual structure
	* \param[in] d_input	Array with input numbers
	* \param[in] d_output	Array that will contain the position; 0 if all values are smaller, elements if all values are larger, -1 if NaN is encountered
	* \param[in] elements	Number of elements
	* \param[in] d_structurefraction	Array with individual structure fractions for all members of the current batch
	* \param[in] batch	Number of input vectors/output positions
	*/
	void d_IntersectHalfBitFSC(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_structurefraction, int batch = 1);

	/**
	 * \brief Finds the position of the first (local) minimum
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the position; -1 if no minimum is found
	 * \param[in] elements	Number of elements
	 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR is supported (thus useless)
	 * \param[in] batch	Number of input vectors/output positions
	 */
	void d_FirstMinimum(tfloat* d_input, tfloat* d_output, size_t elements, T_INTERP_MODE mode, int batch = 1);

	/**
	 * \brief Evaluates the Boolean n > value for every input element n; outputs 1 if true, 0 if false
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
	 * \param[in] elements	Number of elements
	 * \param[in] value	The value to be compared to input
	 */
	template<class T> void d_BiggerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value);

	/**
	 * \brief Evaluates the Boolean n < value for every input element n; outputs 1 if true, 0 if false
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
	 * \param[in] elements	Number of elements
	 * \param[in] value	The value to be compared to input
	 */
	template<class T> void d_SmallerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value);

	/**
	 * \brief Evaluates the Boolean (n >= minval && n < maxval) for every input element n; outputs 1 if true, 0 if false
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
	 * \param[in] elements	Number of elements
	 * \param[in] minval	The lower bound for the comparison
	 * \param[in] maxval	The upper bound for the comparison
	 */
	template<class T> void d_IsBetween(tfloat* d_input, T* d_output, size_t elements, tfloat minval, tfloat maxval);

	//MakeAtlas.cu:

	/**
	 * \brief Composes small images into one bigger, square image to reduce overhead from texture binding
	 * \param[in] d_input	Array with image patches
	 * \param[in] inputdims	X and Y are respective patch dimensions, Z is number of patches
	 * \param[in] outputdims	Will contain the atlas dimensions
	 * \param[in] primitivesperdim	X and Y will contain number of patches per atlas row and column, respectively
	 * \param[in] h_primitivecoords	Host array that will contain the coordinates of the upper left corners of patches in atlas
	 * \returns Atlas array; make sure to dispose manually
	 */
	template <class T> T* d_MakeAtlas(T* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords);

	//Sum.cu:

	/**
	 * \brief Performs vector reduction by summing up the elements; use this version for few, but large vectors
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the sum; d_output == d_input is not valid
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors to be reduced
	 */
	template <class T> void d_Sum(T *d_input, T *d_output, size_t n, int batch = 1);

	/**
	* \brief Performs vector reduction by summing up the elements; use this version for many, but small vectors (one CUDA block is used per vector)
	* \param[in] d_input	Array with input numbers
	* \param[in] d_output	Array that will contain the sum; d_output == d_input is not valid
	* \param[in] n	Number of elements
	* \param[in] batch	Number of vectors to be reduced
	*/
	template <class T> void d_SumMonolithic(T* d_input, T* d_output, int n, int batch);

	/**
	 * \brief Performs vector reduction by summing up the elements; use this version for many, but small vectors (one CUDA block is used per vector)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the sum; d_output == d_input is not valid
	 * \param[in] d_mask	Mask to be applied to d_input; pass NULL if not needed
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors to be reduced
	 */
	template <class T> void d_SumMonolithic(T* d_input, T* d_output, tfloat* d_mask, int n, int batch);

	//MinMax.cu:

	/**
	 * \brief Finds the smallest element and its position in a vector; use this version for few, but large vectors
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the minimum value and its position
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_Min(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);

	/**
	 * \brief Finds the smallest element in a vector; use this version for few, but large vectors
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the minimum value
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_Min(T *d_input, T *d_output, size_t n, int batch = 1);

	/**
	 * \brief Finds the largest element and its position in a vector; use this version for few, but large vectors
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the maximum value and its position
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_Max(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);

	/**
	 * \brief Finds the largest element in a vector; use this version for few, but large vectors
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the maximum value
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_Max(T *d_input, T *d_output, size_t n, int batch = 1);

	//MinMaxMonolithic.cu:

	/**
	 * \brief Finds the smallest element and its position in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the minimum value and its position
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_MinMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);

	/**
	 * \brief Finds the smallest element in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the minimum value
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_MinMonolithic(T* d_input, T* d_output, int n, int batch);

	/**
	 * \brief Finds the largest element and its position in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the maximum value and its position
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_MaxMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);

	/**
	 * \brief Finds the largest element in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_output	Array that will contain the maximum value
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors
	 */
	template <class T> void d_MaxMonolithic(T* d_input, T* d_output, int n, int batch);

	//SumMinMax.cu:

	/**
	 * \brief Performs vector reduction by summing up the elements while also finding minimum and maximum values
	 * \param[in] d_input	Array with input numbers
	 * \param[in] d_sum	Array that will contain the sum; d_sum == d_input is not valid
	 * \param[in] d_min	Array that will contain the minimum value; d_min == d_input is not valid
	 * \param[in] d_max	Array that will contain the maximum value; d_max == d_input is not valid
	 * \param[in] n	Number of elements
	 * \param[in] batch	Number of vectors to be reduced
	 */
	template <class T> void d_SumMinMax(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int batch = 1);

	//Dev.cu:

	/**
	 * \brief Calculates the image metrics represented by a imgstats5 structure
	 * \param[in] d_input	Array with input data
	 * \param[in] d_output	Array that will contain the calculated metrics
	 * \param[in] elements	Number of elements
	 * \param[in] d_mask	Binary mask to restrict analysis to certain areas; optional: pass NULL to consider entire image
	 * \param[in] batch	Number of images to be analyzed
	 */
	template <class Tmask> void d_Dev(tfloat* d_input, imgstats5* d_output, size_t elements, Tmask* d_mask, int batch = 1);

	//Extraction.cu:

	/**
	 * \brief Extracts a rectangular portion from one or multiple images/volumes at a fixed location
	 * \param[in] d_input	Array with input data
	 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
	 * \param[in] sourcedims	Dimensions of original image/volume
	 * \param[in] regiondims	Dimensions of extracted portion
	 * \param[in] regioncenter	Coordinates of the center point of the extracted portion; center is defined as dims / 2
	 * \param[in] batch	Number of images to be processed
	 */
	template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch = 1);

	/**
	* \brief Extracts a rectangular portion from one or multiple images/volumes at variable locations
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
	* \param[in] sourcedims	Dimensions of original image/volume
	* \param[in] regiondims	Dimensions of extracted portion
	* \param[in] d_regionorigins	Coordinates of the left upper corner of the extracted portion
	* \param[in] batch	Number of images to be processed
	*/
	template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch = 1);

	/**
	 * \brief Extracts rectangular portions from the same image/volume at different positions
	 * \param[in] d_input	Array with input data
	 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
	 * \param[in] sourcedims	Dimensions of original image/volume
	 * \param[in] regiondims	Dimensions of extracted portion
	 * \param[in] d_regionorigins	Coordinates of the upper left corner of the extracted portion
	 * \param[in] batch	Number of images to be extracted
	 */
	template <class T> void d_ExtractMany(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, bool zeropad, int batch);

	/**
	 * \brief Extracts rectangular portions from a set of images/volumes stored in separate memory locations at different positions
	 * \param[in] d_input	Array with pointers to input data
	 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
	 * \param[in] sourcedims	Dimensions of original image/volume
	 * \param[in] regiondims	Dimensions of extracted portion
	 * \param[in] d_regionorigins	Coordinates of the upper left corner of the extracted portion
	 * \param[in] nsources	Number of independently allocated source arrays, the selection in kernel is then ibatch % nsources
	 * \param[in] batch	Number of images to be extracted
	 */
	template <class T> void d_ExtractManyMultisource(T** d_inputs, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int nsources, int batch);

	/**
	 * \brief Extracts rectangular portions from the same image/volume at different positions, sampling it in a transformed reference frame
	 * \param[in] d_input	Array with input data
	 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
	 * \param[in] sourcedims	Dimensions of original image/volume
	 * \param[in] regiondims	Dimensions of extracted portion
	 * \param[in] h_scale	Host array containing the scale factors
	 * \param[in] h_rotation	Host array containing the rotation angles in radians
	 * \param[in] h_translation	Host array containing the shift vectors; no shift = extracted center is at source's upper left corner
	 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR and T_INTERP_CUBIC are supported
	 * \param[in] batch	Number of images to be extracted
	 */
	void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int2 sourcedims, int2 regiondims, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch = 1);

	//Insertion.cu:

	void d_InsertAdditive(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);

	//LocalStd.cu:

	/**
	* \brief Computes the local standard deviation within a sphere centered around each pixel/voxel
	* \param[in] d_map	Array with input data
	* \param[in] dimsmap	Map dimensions
	* \param[in] localradius	Sphere radius
	* \param[in] d_std	Array that will contain the local standard deviation
	* \param[in] d_mean	Optional array that will contain the local mean
	*/
	void d_LocalStd(tfloat* d_map, int3 dimsmap, tfloat* d_fouriermask, tfloat localradius, tfloat* d_std, tfloat* d_mean = NULL, cufftHandle planforw = 0, cufftHandle planback = 0);

	//Padding.cu:

	/**
	* \brief Specifies how the pad region should be filled
	*/
	enum T_PAD_MODE
	{
		/**Pad with fixed value*/
		T_PAD_VALUE = 1,
		/**Pad by repeating mirrored data*/
		T_PAD_MIRROR = 2,
		/**Pad by repeating data*/
		T_PAD_TILE = 3,
		/**Pad by clamping coordinates**/
		T_PAD_CLAMP = 4,
		/**Leave memory contents unchanged within pad*/
		T_PAD_NOTHING = 5
	};

	/**
	* \brief Pads a 1/2/3 dimensional array to obtain the desired size, while keeping the centers of old and new array aligned. Can also be used to extract the central portion if new size is smaller than old.
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the padded data
	* \param[in] inputdims	Dimensions of original array
	* \param[in] outputdims	Dimensions of the new array
	* \param[in] mode	Specifies how to fill the pad region
	* \param[in] value	If mode = T_PAD_VALUE, this is the fixed value used
	* \param[in] batch	Number of arrays to be padded
	*/
	template <class T> void d_Pad(T* d_input, T* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, T value, int batch = 1);

	template <class T> void d_PadClampSoft(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int softdist, int batch = 1);

	//Polynomials.cu:

	/**
	* \brief Computes the values of a 1D polynomial with degree < 1024 at coordinates given in d_x.
	* \param[in] d_x	Array with input coordinates
	* \param[in] d_output	Array that will contain the polynomial values
	* \param[in] npoints	Number of coordinates in d_x
	* \param[in] d_factors	Array with the polynomial factors, first element is 0th order
	* \param[in] degree	The polynomial's degree
	* \param[in] batch	Number of coordinate and factor sets to compute values for
	*/
	void d_Polynomial1D(tfloat* d_x, tfloat* d_output, int npoints, tfloat* d_factors, int degree, int batch);

	//Reductions.cu:

	/**
	* \brief Sum over multiple vectors
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the reduced data
	* \param[in] vectorlength	Vector length
	* \param[in] nvectors	Number of vectors to sum over
	* \param[in] batch	Number of vector sets to be reduced independently
	*/
	template<class T> void d_ReduceAdd(T* d_input, T* d_output, int vectorlength, int nvectors, int batch = 1);

	/**
	* \brief Average over multiple vectors
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the reduced data
	* \param[in] vectorlength	Vector length
	* \param[in] nvectors	Number of vectors to average over
	* \param[in] batch	Number of vector sets to be reduced independently
	*/
	template<class T> void d_ReduceMean(T* d_input, T* d_output, int vectorlength, int nvectors, int batch = 1);

	/**
	* \brief Average over multiple vectors with individual weights for all values
	* \param[in] d_input	Array with input data
	* \param[in] d_inputweights	Array with input data weights
	* \param[in] d_output	Array that will contain the reduced data
	* \param[in] vectorlength	Vector length
	* \param[in] nvectors	Number of vectors to average over
	* \param[in] batch	Number of vector sets to be reduced independently
	*/
	template<class T> void d_ReduceMeanWeighted(T* d_input, tfloat* d_inputweights, T* d_output, int vectorlength, int nvectors, int batch = 1);

	/**
	* \brief Or conjunction over multiple vectors
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the reduced data
	* \param[in] vectorlength	Vector length
	* \param[in] nvectors	Number of vectors to average over
	* \param[in] batch	Number of vector sets to be reduced independently
	*/
	template<class T> void d_ReduceOr(T* d_input, T* d_output, uint vectorlength, uint nvectors, uint batch = 1);

	/**
	* \brief Max over multiple vectors
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the reduced data
	* \param[in] vectorlength	Vector length
	* \param[in] nvectors	Number of vectors to max over
	* \param[in] batch	Number of vector sets to be reduced independently
	*/
	template<class T> void d_ReduceMax(T* d_input, T* d_output, int vectorlength, int nvectors, int batch = 1);
}
#endif