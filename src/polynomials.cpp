#include <cassert>
#include <vector>
#include <cmath>

#include "polynomials.hpp"
#include "mdvector.hpp"


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

// Evaluate 2D legendre basis
double LegendreND(unsigned int in_mode, mdvector<double> loc, unsigned int order, unsigned int nDims)
{
  double leg_basis;
  if(nDims == 2)
  {
    unsigned int n_dof=(order+1)*(order+1);

    if(in_mode<n_dof)
    {
      unsigned int i,j,k;
      unsigned int mode;
      double normCi, normCj;
      mode = 0;
      #pragma omp parallel for
      for(k=0;k<=2*order;k++)
      {
        for(j=0;j<k+1;j++)
        {
          i = k-j;
          if(i<=order && j<=order) // Order would be (0,2) (1,1) (2,0) ... any hierarchical ordering is fine
          {
            if(mode==in_mode) // found the correct mode 
            {
  	          normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
              normCj = std::sqrt(2.0 / (2.0 * j + 1.0)); 
              leg_basis = Legendre(i,loc(0))*Legendre(j,loc(1))/(normCi*normCj);
            }
            mode++;
          }
        }
      }
    }
    else
    {
      cout << "ERROR: Invalid mode when evaluating Legendre basis ...." << endl;
    }
  }
  else if(nDims == 3)
  {
    unsigned int n_dof=(order+1)*(order+1)*(order+1);

    if(in_mode<n_dof)
    {
      unsigned int i,j,k,l;
      unsigned int mode;
      double normCi, normCj,normCk;
      mode = 0;
      #pragma omp parallel for
      for(l=0;l<=3*order;l++)
      {
        for(k=0;k<=l;k++)
        {
          for(j=0;j<=l-k;j++)
          {
            i = l-k-j;
            if(i<=order && j<=order && k <=order)
            {
              if(mode==in_mode) // found the correct mode
              {
                normCi = std::sqrt(2.0 / (2.0 * i + 1.0));
                normCj = std::sqrt(2.0 / (2.0 * j + 1.0));
                normCk = std::sqrt(2.0 / (2.0 * k + 1.0)); 
                leg_basis = Legendre(i,loc(0))*Legendre(j,loc(1))*Legendre(k,loc(2))/(normCi*normCj*normCk);
              }
              mode++;
            }
          }
        }
      }
    }
    else
    {
      cout << "ERROR: Invalid mode when evaluating Legendre basis ...." << endl;
    }    
  }
  else
  {
    cout << "ERROR: Legendre basis not implemented for higher than 3 dimensions" << endl;
  }
  
  return leg_basis;
}
