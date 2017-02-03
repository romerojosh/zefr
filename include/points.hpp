/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

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
std::vector<double> DFRsp_pts(unsigned int P, double z1);

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

mdvector<double> WS_Tri_pts(unsigned int P);
mdvector<double> WS_Tri_weights(unsigned int P);
mdvector<double> RW_Tri_pts(unsigned int P);
mdvector<double> symm_pts_tri(std::vector<double> a3, std::vector<double> a21, std::vector<double> a111, std::vector<double> b111);

#endif /* points_hpp */
