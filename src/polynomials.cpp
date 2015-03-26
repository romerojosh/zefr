#include <cassert>
#include <vector>

#include "polynomials.hpp"


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
