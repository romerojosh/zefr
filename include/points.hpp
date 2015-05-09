#ifndef points_hpp
#define points_hpp

#include <vector>

//! Returns a vector containing the P-th order Legendre polynomial in ascending order 
/*!
 * \param P Order of the Legendre polynomial
 * 
 * \return Vector containing zeros in ascending order
 */

std::vector<double> Gauss_Legendre_pts(unsigned int P); 

//! Returns a vector containing weights for Gauss-Legendre quadrature
/*!
 * \param n Number of quadrature points in 1D
 * 
 * \return Vector of weights, ordered consistently with point order
 */

std::vector<double> Gauss_Legendre_weights(unsigned int n); 


//! Returns a vector containing the shape point locations along an edge of a tensor product element 
/*!
 * \param P Order of the edge
 * 
 * \return Vector containing shape point locations in ascending order
 */

std::vector<double> Shape_pts(unsigned int P); 

#endif /* points_hpp */
