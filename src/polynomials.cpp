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
  if (mode >= nModes) 
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

double dDubiner2D(unsigned int P, double xi, double eta, double dim, unsigned int mode)
{
  double val;
  int nModes = (P + 1) * (P + 2) / 2;
  if (mode >= nModes) 
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
        if (dim == 0)
        {
          double j0 = dJacobi(ab[0], 0, 0, i);
          double j1 = Jacobi(ab[1], 2*i + 1, 0, j);
          if (i == 0)
            val = 0.0;
          else
            val =  2.0 * std::sqrt(2) * j0 * j1 * std::pow(1 - ab[1], i-1);
        }
        else if (dim == 1)
        {
          double j0 = dJacobi(ab[0], 0, 0, i);
          double j1 = Jacobi(ab[1], 2*i + 1, 0, j);
          double j2 = Jacobi(ab[0], 0, 0, i);
          double j3 = dJacobi(ab[1], 2*i + 1, 0, j) * std::pow(1 - ab[1], i);
          double j4 = Jacobi(ab[1], 2*i + 1, 0, j) * i * std::pow(1 - ab[1], i-1);

          if (i == 0)
            val = std::sqrt(2) * j2 * j3;
          else
            val = std::sqrt(2) * (j0 * j1 * std::pow(1 - ab[1], i-1) * (1 + ab[0]) + j2 * (j3 - j4)); 
        }
      }

      m++;
    }
  } 

  return val;
}

double RTMonomial2D(unsigned int P, double xi, double eta, unsigned int dim, unsigned int mode)
{
  double val;
  unsigned int nP2Modes = (P+1)*(P+2); // number of regular monomial modes

  if (mode < nP2Modes)
  {
    /* Regular monomial mode */
    unsigned int idx = mode/2;
    unsigned int slot = mode%2; //which dimension mode is nonzero

    unsigned int n = 0;
    for (unsigned int j = 0; j <= P; j++)
    {
      for (unsigned int i = 0; i <= P; i++)
      {
        if (i+j <= P)
        {
          if (n == idx)
          {
            val =  (dim == slot) ? std::pow(xi, i) * std::pow(eta, j) : 0.0;
            return val;
          }
          n++;
        }
      }
    }
  }
  else
  {
    /* RT mode */
    unsigned int idx = mode - nP2Modes;

    unsigned int n = 0;
    for (unsigned int j = 0; j <= P; j++)
    {
      for (unsigned int i = 0; i <= P; i++)
      {
        if (i+j == P)
        {
          if (n == idx)
          {
            val =  (dim == 0) ? std::pow(xi, i+1) * std::pow(eta, j) : std::pow(xi, i) * std::pow(eta, j+1);
            return val;
          }
          n++;
        }
      }
    }
  }

  return 0.0;
}

double divRTMonomial2D(unsigned int P, double xi, double eta, unsigned int mode)
{
  double val;
  unsigned int nP2Modes = (P+1)*(P+2); // number of regular monomial modes

  if (mode < nP2Modes)
  {
    /* Regular monomial mode */
    unsigned int idx = mode/2;
    unsigned int slot = mode%2; //which dimension mode is nonzero

    unsigned int n = 0;
    for (unsigned int j = 0; j <= P; j++)
    {
      for (unsigned int i = 0; i <= P; i++)
      {
        if (i+j <= P)
        {
          if (n == idx)
          {
            val =  (slot == 0) ? i * std::pow(xi, i-1) * std::pow(eta, j) : std::pow(xi, i) * j * std::pow(eta, j-1);
            return val;
          }
          n++;
        }
      }
    }
  }
  else
  {
    /* RT mode */
    unsigned int idx = mode - nP2Modes;

    unsigned int n = 0;
    for (unsigned int j = 0; j <= P; j++)
    {
      for (unsigned int i = 0; i <= P; i++)
      {
        if (i+j == P)
        {
          if (n == idx)
          {
            val =  (i+1)*std::pow(xi, i) * std::pow(eta, j) + std::pow(xi, i) * (j+1) * std::pow(eta, j);
            return val;
          }
          n++;
        }
      }
    }
  }

  return 0.0;
}

double Dubiner3D(unsigned int P, double xi, double eta, double zeta, unsigned int mode)
{
  double val;
  int nModes = (P + 1) * (P + 2) * (P + 3) / 6;
  if (mode >= nModes) 
    ThrowException("ERROR: mode value is too high for given P!")

  double abc[3];
  abc[0] = (eta + zeta == 0) ? (-1) : -2.0 * (1.0 + xi) / (eta + zeta) - 1.0;
  abc[1] = (zeta == 1) ? (-1) : 2.0 * (1.0 + eta) / (1.0 - zeta) - 1.0;
  abc[2] = zeta;

  unsigned int m = 0;
  for (unsigned int l = 0; l <= P; l++)
  {
    for (unsigned int n = 0; n <= l; n++)
    {
      for (unsigned int k = 0; k <= n; k++)
      {
        unsigned int j  = n - k;
        unsigned int i  = l - j - k;

        if (m == mode)
        {
          double j0 = Jacobi(abc[0], 0, 0, i);
          double j1 = Jacobi(abc[1], 2*i + 1, 0, j);
          double j2 = Jacobi(abc[2], 2*i + 2*j + 2, 0, k);
          val =  2.0 * std::sqrt(2) * j0 * j1 * j2 * std::pow(1 - abc[1], i) * std::pow(1 - abc[2], i+j);
        }

        m++;
      }
    }
  } 

  return val;
}

double dDubiner3D(unsigned int P, double xi, double eta, double zeta, double dim, unsigned int mode)
{
  int nModes = (P + 1) * (P + 2) * (P + 3) / 6;
  if (mode >= nModes) 
    ThrowException("ERROR: mode value is too high for given P!")

  double abc[3];
  abc[0] = (eta + zeta == 0) ? (-1) : -2.0 * (1.0 + xi) / (eta + zeta) - 1.0;
  abc[1] = (zeta == 1) ? (-1) : 2.0 * (1.0 + eta) / (1.0 - zeta) - 1.0;
  abc[2] = zeta;

  unsigned int m = 0;
  for (unsigned int l = 0; l <= P; l++)
  {
    for (unsigned int n = 0; n <= l; n++)
    {
      for (unsigned int k = 0; k <= n; k++)
      {
        unsigned int j  = n - k;
        unsigned int i  = l - j - k;

        if (m == mode)
        {
          double j0 = Jacobi(abc[0], 0, 0, i);
          double j1 = Jacobi(abc[1], 2*i + 1, 0, j);
          double j2 = Jacobi(abc[2], 2*i + 2*j + 2, 0, k);
          double dj0 = dJacobi(abc[0], 0, 0, i);
          double dj1 = dJacobi(abc[1], 2*i + 1, 0, j);
          double dj2 = dJacobi(abc[2], 2*i + 2*j + 2, 0, k);

          double dxi = dj0 * j1 * j2;
          if (i > 0)
            dxi *= std::pow(0.5 * (1 - abc[1]), i - 1);
          if (i+j > 0)
            dxi *= std::pow(0.5 * (1 - abc[2]), i + j - 1);

          if (dim == 0)
            return(dxi * std::pow(2, 2*i + j + 1.5));

          double deta = (0.5 * (1 + abc[0])) * dxi;
          double tmp = dj1 * std::pow(0.5 * (1-abc[1]), i);

          if (i > 0)
            tmp -= 0.5 * i * j1 * std::pow(0.5 * (1-abc[1]), i - 1);
          if (i+j > 0) 
            tmp *= std::pow(0.5 * (1 - abc[2]), i + j - 1);

          tmp *= j0 * j2;
          deta += tmp;

          if (dim == 1)
            return(deta * std::pow(2, 2*i + j + 1.5));

          double dzeta = 0.5 * (1 + abc[0]) * dxi + 0.5 * (1 + abc[1]) * tmp;
          tmp = dj2 * std::pow(0.5 * (1 - abc[2]), i + j);

          if (i+j > 0)
            tmp -= 0.5 * (i+j) * (j2 * std::pow(0.5 * (1 - abc[2]), i + j - 1));

          tmp *= j0 * j1 * std::pow(0.5 * (1 - abc[1]), i);
          dzeta += tmp;

          if (dim == 2)
            return (dzeta * std::pow(2, 2*i + j + 1.5));

        }

        m++;
      }
    }
  } 
}

double RTMonomial3D(unsigned int P, double xi, double eta, double zeta, unsigned int dim, unsigned int mode)
{
  double val;
  unsigned int nP3Modes = (P+1)*(P+2)*(P+3) / 2; // number of regular monomial modes 

  if (mode < nP3Modes)
  {
    /* Regular monomial mode */
    unsigned int idx = mode / 3;
    unsigned int slot = mode % 3; //which dimension mode is nonzero

    unsigned int n = 0;
    for (unsigned int k = 0; k <= P; k++)
    {
      for (unsigned int j = 0; j <= P; j++)
      {
        for (unsigned int i = 0; i <= P; i++)
        {
          if (i+j+k <= P)
          {
            if (n == idx)
            {
              val =  (dim == slot) ? std::pow(xi, i) * std::pow(eta, j) * std::pow(zeta, k): 0.0;
              return val;
            }
            n++;
          }
        }
      }
    }
  }
  else
  {
    /* RT mode */
    unsigned int idx = mode - nP3Modes;

    unsigned int n = 0;
    for (unsigned int k = 0; k <= P; k++)
    {
      for (unsigned int j = 0; j <= P; j++)
      {
        for (unsigned int i = 0; i <= P; i++)
        {
          if (i+j+k == P)
          {
            if (n == idx)
            {
              if (dim == 0)
                val = std::pow(xi, i+1) * std::pow(eta, j) * std::pow(zeta, k);
              else if (dim == 1)
                val = std::pow(xi, i) * std::pow(eta, j+1) * std::pow(zeta, k);
              else if (dim == 2)
                val = std::pow(xi, i) * std::pow(eta, j) * std::pow(zeta, k+1);

              return val;
            }
            n++;
          }
        }
      }
    }
  }

  return 0.0;
}

double divRTMonomial3D(unsigned int P, double xi, double eta, double zeta, unsigned int mode)
{
  double val;
  unsigned int nP3Modes = (P+1)*(P+2)*(P+3)/2; // number of regular monomial modes

  if (mode < nP3Modes)
  {
    /* Regular monomial mode */
    unsigned int idx = mode / 3;
    unsigned int slot = mode % 3; //which dimension mode is nonzero

    unsigned int n = 0;
    for (unsigned int k = 0; k <= P; k++)
    {
      for (unsigned int j = 0; j <= P; j++)
      {
        for (unsigned int i = 0; i <= P; i++)
        {
          if (i+j+k <= P)
          {
            if (n == idx)
            {
              if (slot == 0)
                val = i * std::pow(xi, i-1) * std::pow(eta, j) * std::pow(zeta, k);
              else if (slot == 1)
                val = std::pow(xi, i) * j * std::pow(eta, j-1) * std::pow(zeta, k);
              else if (slot == 2)
                val = std::pow(xi, i) * std::pow(eta, j) * k * std::pow(zeta, k-1);

              return val;
            }
            n++;
          }
        }
      }
    }
  }
  else
  {
    /* RT mode */
    unsigned int idx = mode - nP3Modes;

    unsigned int n = 0;
    for (unsigned int k = 0; k <= P; k++)
    {
      for (unsigned int j = 0; j <= P; j++)
      {
        for (unsigned int i = 0; i <= P; i++)
        {
          if (i+j+k == P)
          {
            if (n == idx)
            {
              val =  (i+1) * std::pow(xi, i) * std::pow(eta, j) * std::pow(zeta, k) + 
                     std::pow(xi, i) * (j+1) * std::pow(eta, j) * std::pow(zeta, k) +
                     std::pow(xi, i) * std::pow(eta, j) * (k+1) * std::pow(zeta, k);

              return val;
            }
            n++;
          }
        }
      }
    }
  }

  return 0.0;
}

double Legendre2D(unsigned int P, double xi, double eta, unsigned int mode)
{
  double val;
  int nModes = (P + 1) * (P + 1);
  if (mode >= nModes) 
    ThrowException("ERROR: mode value is too high for given P!")

  double normCi, normCj;
  unsigned int m = 0;

  for (unsigned int k = 0; k <= 2*P; k++)
  {
    for (unsigned int j = 0; j < k+1; j++)
    {
      unsigned int i = k-j;
      if (i <= P && j <= P) // Order would be (0,2) (1,1) (2,0) ... any hierarchical ordering is fine
      {
        if (m == mode) // found the correct mode
        {
          normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
          normCj = std::sqrt(2.0 / (2.0 * j + 1.0));
          val = Legendre(i, xi) * Legendre(j, eta) / (normCi*normCj);
        }
        m++;
      }
    }
  }

  return val;
}

double Legendre3D(unsigned int P, double xi, double eta, double mu, unsigned int mode)
{
  double val;
  int nModes = (P + 1) * (P + 1) * (P + 1);
  if (mode >= nModes) 
    ThrowException("ERROR: mode value is too high for given P!")


  double normCi, normCj, normCk;
  unsigned int m = 0;
  for (unsigned int l=0; l <= 3*P; l++)
  {
    for (unsigned int k = 0; k <= l; k++)
    {
      for (unsigned int j = 0; j <= l-k; j++)
      {
        unsigned int i = l-k-j;
        if (i <= P && j <= P && k <= P)
        {
          if (m == mode) // found the correct mode
          {
            normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
            normCj = std::sqrt(2.0 / (2.0 * j + 1.0));
            normCk = std::sqrt(2.0 / (2.0 * k + 1.0));
            val = Legendre(i, xi) * Legendre(j, eta) * Legendre(k, mu) / (normCi*normCj*normCk);
          }
          m++;
        }
      }
    }
  }

  return val;
}
