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

#ifndef TABFUNCS_H_
#define TABFUNCS_H_

#include "multidim_array.h"
#include "funcs.h"


namespace relion
{
	// Class to tabulate some functions
	class TabFunction
	{

	protected:
		MultidimArray<DOUBLE> tabulatedValues;
		DOUBLE  sampling;
	public:
		// Empty constructor
		TabFunction() {}

		// Destructor
		virtual ~TabFunction()
		{
			tabulatedValues.clear();
		}

		/** Copy constructor
		 *
		 * The created TabFunction is a perfect copy of the input array but with a
		 * different memory assignment.
		 */
		TabFunction(const TabFunction& op)
		{
			tabulatedValues.clear();
			*this = op;
		}

		/** Assignment.
		 *
		 * You can build as complex assignment expressions as you like. Multiple
		 * assignment is allowed.
		 */
		TabFunction& operator=(const TabFunction& op)
		{
			if (&op != this)
			{
				// Projector stuff (is this necessary in C++?)
				tabulatedValues = op.tabulatedValues;
				sampling = op.sampling;
			}
			return *this;
		}


	};

	class TabSine : public TabFunction
	{
	public:
		// Empty constructor
		TabSine() {}

		// Constructor (with parameters)
		void initialise(const int _nr_elem = 5000);

		//Pre-calculate table values
		void fillTable(const int _nr_elem = 5000);

		// Value access
		DOUBLE operator()(DOUBLE val) const;

	};

	class TabCosine : public TabFunction
	{
	public:
		// Empty constructor
		TabCosine() {}

		void initialise(const int _nr_elem = 5000);

		//Pre-calculate table values
		void fillTable(const int _nr_elem = 5000);

		// Value access
		DOUBLE operator()(DOUBLE val) const;

	};

	class TabBlob : public TabFunction
	{

	private:
		DOUBLE radius;
		DOUBLE alpha;
		int order;

	public:
		// Empty constructor
		TabBlob() {}

		// Constructor (with parameters)
		void initialise(DOUBLE _radius, DOUBLE _alpha, int _order, const int _nr_elem = 10000);

		//Pre-calculate table values
		void fillTable(const int _nr_elem = 5000);

		// Value access
		DOUBLE operator()(DOUBLE val) const;

	};

	class TabFtBlob : public TabFunction
	{

	private:
		DOUBLE radius;
		DOUBLE alpha;
		int order;

	public:
		// Empty constructor
		TabFtBlob() {}

		// Constructor (with parameters)
		void initialise(DOUBLE _radius, DOUBLE _alpha, int _order, const int _nr_elem = 10000);

		//Pre-calculate table values
		void fillTable(const int _nr_elem = 5000);

		// Value access
		DOUBLE operator()(DOUBLE val) const;

	};

}
#endif /* TABFUNCS_H_ */
