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

#ifndef polynomials_hpp
#define polynomials_hpp

#include <vector>

/*! Evaluates the Lagrange function corresponding to the specified mode on xiGrid at location xi.
 *
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Lagrange function at xi.
 */
static inline
double Lagrange(const std::vector<double> &xiGrid, unsigned int mode, double xi)
{
  double val = 1.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  assert(mode < npts);

  for (unsigned int i = 0; i < mode; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  for (unsigned int i = mode + 1; i < npts; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

/*! Because Jacob is too lazy to convert all his copied code from Flurry */
static inline
double Lagrange(const std::vector<double> &xiGrid, double xi, unsigned int mode)
{
  return Lagrange(xiGrid, mode, xi);
}

/*! Evaluates the first derivative of the Lagrange function corresponding to the specified mode
 *  on xiGrid at location xi.
 *
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid[mode]
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 *
 * \return Value of first derivative of the Lagrange function at xi.
 */
static inline
double Lagrange_d1(const std::vector<double>& xiGrid, unsigned int mode, double xi)
{
  double val = 0.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  assert(mode < npts);

  /* Compute normalization constant */
  double den = 1.0;
  for (unsigned int i = 0; i < mode; i++)
    den *= (xiGrid[mode] - xiGrid[i]);

  for (unsigned int i = mode+1; i < npts; i++)
    den *= (xiGrid[mode] - xiGrid[i]);

  /* Compute sum of products */
  for (unsigned int j = 0; j < npts; j++)
  {
    if (j == mode)
      continue;

    double term = 1.0;
    for (unsigned int i = 0; i < npts; i++)
      if (i != mode and i != j)
        term *= (xi - xiGrid[i]);

    val += term;
  }

  return val/den;
}

/*! Because Jacob is too lazy to convert all his copied code from Flurry */
static inline
double dLagrange(const std::vector<double>& xiGrid, double xi, unsigned int mode)
{
  return Lagrange_d1(xiGrid, mode, xi);
}

/*! Evaluates the Legendre polynomial of degree P at location xi.
 *
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of Legendre polynomial at xi.
 */
double Legendre(unsigned int P, double xi);

/*! Evaluates the first derivative of the Legendre polynomial of degree P at location xi.
 *
 * \param P  Order of the Legendre polynomial
 * \param xi  Point of evaluation in domain [-1,1].
 * 
 * \return Value of derivative of Legendre polynomial at xi.
 */
double Legendre_d1(unsigned int P, double xi);

//! Multi-dimensional Legendre polynomial
double LegendreND(unsigned int in_mode, const std::vector<double>& loc, unsigned int order, unsigned int nDims);

#endif /* polynomials_hpp */
