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

#include <cassert>
#include <vector>

#include "polynomials.hpp"


double Lagrange(const std::vector<double> &xiGrid, double xi, unsigned int mode)
{
  return Lagrange(xiGrid, mode, xi);
}

double Lagrange(const std::vector<double> &xiGrid, unsigned int mode, double xi)
{
  double val = 1.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  assert(mode < npts);

  for (unsigned int i = 0; i < npts; i++)
    if (i != mode)
      val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

double dLagrange(const std::vector<double>& xiGrid, double xi, unsigned int mode)
{
  return Lagrange_d1(xiGrid, mode, xi);
}

double Lagrange_d1(const std::vector<double>& xiGrid, unsigned int mode, double xi)
{
  double val = 0.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  assert(mode < npts);

  /* Compute normalization constant */
  double den = 1.0;
  for (unsigned int i = 0; i < npts; i++)
    if (i != mode)
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

double Legendre(unsigned int P, double xi)
{
  if (P == 0)
    return 1.0;
  else if (P == 1)
    return xi;

	return ((2.0 * P - 1.0) / P) *xi *Legendre(P-1, xi) - ((P - 1.0) / P) *Legendre(P-2, xi);
}

double Legendre_d1(unsigned int P, double xi)
{
	if (P == 0)
    return 0.0;
  else if (P == 1)
    return 1.0;
	
  return ((2.0 * P - 1.0) / P) * (Legendre(P-1, xi) + xi *Legendre_d1(P-1, xi)) - ((P - 1.0) / P) *Legendre_d1(P-2, xi);
}
