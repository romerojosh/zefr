#ifndef polynomials_hpp
#define polynomials_hpp

#include <vector>
#include "mdvector.hpp"

//! Evaluates the Lagrange function corresponding to the specified mode on xiGrid at location xi.
/*!
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Lagrange function at xi.
 */

double Lagrange(std::vector<double> xiGrid, unsigned int mode, double xi);

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

//! Evaluates the Legendre polynomial of degree P at location xi.
/*!
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Legendre polynomial at xi.
 */
double Legendre(unsigned int P, double xi);

//! Evaluates the first derivative of the Legendre polynomial of degree P at location xi.
/*!
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of derivative of Legendre polynomial at xi.
 */
double Legendre_d1(unsigned int P, double xi);

//! Evaluates the 2D Legendre modal basis at loc.
/*!
 * \param in_mode  Mode to evaluate
 * \param loc      mdvector of size giving [xi,eta] of the point
 * \param order    order of the polynomial basis (overall)
 * \return 	   Value of mode in_mode at loc.
 */
double Legendre2D(unsigned int in_mode, mdvector<double> loc, unsigned int order);


#endif /* polynomials_hpp */
