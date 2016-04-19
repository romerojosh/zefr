/* NOTE: This file is directly included into solver_kernels.cu. */

#include "input.hpp"

__device__
double compute_source_term_dev(double x, double y, double z, double t, unsigned int var, unsigned int nDims, unsigned int equation)
{
  double val = 0.;
  if (equation == AdvDiff)
  {
    if (nDims == 2)
    {
      val =  -M_PI * (std::cos(M_PI * x) + M_PI * std::sin(M_PI * x) + 
             std::cos(M_PI * y) + M_PI * std::sin(M_PI * y));

      if(x*x + y*y <= 0.09)
        val = 1;
    }
    else
    {
      val =  -M_PI * (std::cos(M_PI * x) + M_PI * std::sin(M_PI * x) + 
             std::cos(M_PI * y) + M_PI * std::sin(M_PI * y) + 
             std::cos(M_PI * z) + M_PI * std::sin(M_PI * z));
    }
  }
  else
  {
    // NOT DEFINED. Cannot throw exception.
  }

  return val;
}

__device__
double get_cfl_limit_dev(int order)
{
 switch(order)
  {
    case 0:
      return 1.392;

    case 1:
      return 0.4642; 

    case 2:
      return 0.2351;

    case 3:
      return 0.1453;

    case 4:
      return 0.1000;

    case 5:
      return 0.0736;

    default:
      return 0.0;
  }

}


