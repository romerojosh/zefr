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
#include <cmath>
#include <vector>

#include "macros.hpp"
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

// Evaluate 2D legendre basis
double LegendreND(unsigned int in_mode, const std::vector<double> &loc, unsigned int order, unsigned int nDims)
{
  double leg_basis;
  if (nDims == 2)
  {
    unsigned int n_dof = (order+1)*(order+1);

    if (in_mode < n_dof)
    {
      double normCi, normCj;
      unsigned int mode = 0;
      #pragma omp parallel for
      for (unsigned int k = 0; k <= 2*order; k++)
      {
        for (unsigned int j = 0; j < k+1; j++)
        {
          unsigned int i = k-j;
          if (i <= order && j <= order) // Order would be (0,2) (1,1) (2,0) ... any hierarchical ordering is fine
          {
            if (mode == in_mode) // found the correct mode
            {
              normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
              normCj = std::sqrt(2.0 / (2.0 * j + 1.0));
              leg_basis = Legendre(i,loc[0])*Legendre(j,loc[1])/(normCi*normCj);
            }
            mode++;
          }
        }
      }
    }
    else
      ThrowException("ERROR: Invalid mode when evaluating Legendre basis ....");
  }
  else if (nDims == 3)
  {
    unsigned int n_dof = (order+1)*(order+1)*(order+1);

    if (in_mode < n_dof)
    {
      double normCi, normCj,normCk;
      unsigned int mode = 0;
      #pragma omp parallel for
      for (unsigned int l=0; l <= 3*order; l++)
      {
        for (unsigned int k = 0; k <= l; k++)
        {
          for (unsigned int j = 0; j <= l-k; j++)
          {
            unsigned int i = l-k-j;
            if (i <= order && j <= order && k <= order)
            {
              if (mode == in_mode) // found the correct mode
              {
                normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
                normCj = std::sqrt(2.0 / (2.0 * j + 1.0));
                normCk = std::sqrt(2.0 / (2.0 * k + 1.0));
                leg_basis = Legendre(i,loc[0])*Legendre(j,loc[1])*Legendre(k,loc[2])/(normCi*normCj*normCk);
              }
              mode++;
            }
          }
        }
      }
    }
    else
      ThrowException("ERROR: Invalid mode when evaluating Legendre basis ....");
  }
  else
    ThrowException("ERROR: Legendre basis not implemented for higher than 3 dimensions");

  return leg_basis;
}
