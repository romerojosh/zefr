#ifndef polynomials_hpp
#define polynomials_hpp

#include <vector>

//! Evaluates the Lagrange function corresponding to the specified mode on xiGrid at location xi.
/*!
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Lagrange function at xi.
 */

double Lagrange_d0(std::vector<double> xiGrid, unsigned int mode, double xi);

//! Evaluates the first derivative of the Lagrange function corresponding to the specified mode 
//  on xiGrid at location xi.
/*!
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of first derivative of the Lagrange function at xi.
 */

double Lagrange_d1(std::vector<double> xiGrid, unsigned int mode, double xi);

#endif /* polynomials_hpp */
