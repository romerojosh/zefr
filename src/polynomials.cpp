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
#include <iostream>
#include <vector>

#include "macros.hpp"
#include "polynomials.hpp"


double Lagrange(std::vector<double> xiGrid, double xi, unsigned int mode)
{
  return Lagrange(xiGrid, mode, xi);
}

double Lagrange(std::vector<double> xiGrid, unsigned int mode, double xi)
{
  double val = 1.0;
  unsigned int npts = (unsigned int) xiGrid.size();

  assert(mode < npts);

  for (unsigned int i = 0; i < npts; i++)
    if (i != mode)
      val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

double dLagrange(std::vector<double> xiGrid, double xi, unsigned int mode)
{
  return Lagrange_d1(xiGrid, mode, xi);
}

double Lagrange_d1(std::vector<double> xiGrid, unsigned int mode, double xi)
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

/* Jacobi polynomial function from HiFiLES */
double Jacobi(double xi, double a, double b, unsigned int mode)
{
  double val; 

  if(mode == 0)
  {
    double d0, d1, d2;

    d0 = std::pow(2.0,(-a-b-1));
    d1 = std::tgamma(a+b+2);
    d2 = std::tgamma(a+1)*std::tgamma(b+1);

    val = std::sqrt(d0*(d1/d2));
  }
  else if(mode == 1)
  {
    double d0, d1, d2, d3, d4, d5;

    d0 = std::pow(2.0,(-a-b-1));
    d1 = std::tgamma(a+b+2);
    d2 = std::tgamma(a+1)*std::tgamma(b+1);
    d3 = a+b+3;
    d4 = (a+1)*(b+1);
    d5 = (xi*(a+b+2)+(a-b));

    val = 0.5*std::sqrt(d0*(d1/d2))*std::sqrt(d3/d4)*d5;
  }
  else
  {
    double d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14;

    d0 = mode*(mode+a+b)*(mode+a)*(mode+b);
    d1 = ((2*mode)+a+b-1)*((2*mode)+a+b+1);
    d3 = (2*mode)+a+b;

    d4 = (mode-1)*((mode-1)+a+b)*((mode-1)+a)*((mode-1)+b);
    d5 = ((2*(mode-1))+a+b-1)*((2*(mode-1))+a+b+1);
    d6 = (2*(mode-1))+a+b;

    d7 = -((a*a)-(b*b));
    d8 = ((2*(mode-1))+a+b)*((2*(mode-1))+a+b+2);

    d9 = (2.0/d3)*std::sqrt(d0/d1);
    d10 = (2.0/d6)*std::sqrt(d4/d5);
    d11 = d7/d8;

    d12 = xi*Jacobi(xi,a,b,mode-1);
    d13 = d10*Jacobi(xi,a,b,mode-2);
          d14 = d11*Jacobi(xi,a,b,mode-1);

    val = (1.0/d9)*(d12-d13-d14);
  }
 
  return val;
}

double dJacobi(double xi, double a, double b, unsigned int mode)
{
  double val;
  if (mode == 0)
    val = 0.0;
  else
    val = std::sqrt(mode * (mode + a + b + 1)) * Jacobi(xi, a + 1, b + 1, mode - 1);

  return val;
}

double Dubiner2D(unsigned int P, double xi, double eta, unsigned int mode)
{
  double val;
  int nModes = (P + 1) * (P + 2) / 2;
  if (mode > nModes) 
    ThrowException("ERROR: mode value is too high for given P!")

  double ab[2];
  ab[0] = (eta == 1.0) ? (-1) : ((2 * (1 + xi) / (1 - eta)) - 1);
  ab[1] = eta;

  unsigned int m = 0;
  for (unsigned int k = 0; k <= P; k++)
  {
    for (unsigned int j = 0; j <= k; j++)
    {
      unsigned int i  = k - j;

      if (m == mode)
      {
        double j0 = Jacobi(ab[0], 0, 0, i);
        double j1 = Jacobi(ab[1], 2*i + 1, 0, j);
        val =  std::sqrt(2) * j0 * j1 * std::pow(1 - ab[1], i);
      }

      m++;
    }
  } 

  return val;
}
