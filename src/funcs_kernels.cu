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

/* NOTE: This file is directly included into solver_kernels.cu. */

#include "input.hpp"

__device__
double compute_source_term_dev(double x, double y, double z, double t, unsigned int var, unsigned int nDims, unsigned int equation)
{
  double val = 0.;
  if (equation == AdvDiff || equation == Burgers)
  {
    if (nDims == 2)
    {
      val =  -M_PI * (std::cos(M_PI * x) + M_PI * std::sin(M_PI * x) + 
             std::cos(M_PI * y) + M_PI * std::sin(M_PI * y));
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
double get_cfl_limit_adv_dev(int order)
{
  /* Upwinded */
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
      return 0.07363;
      
    case 6:
      return 0.05678;
    
    case 7:
      return 0.04530;
    
    case 8:
      return 0.03709;
    
    case 9:
      return 0.03101;
      
    case 10:
      return 0.02635;

    default:
      return 0.0;
  }
}

__device__
double get_cfl_limit_diff_dev(int order, double beta)
{
  /* Centered */
  if (beta == 0)
  {
    switch(order)
    {
      case 0:
        return 2.785;

      case 1:
        return 0.1740;

      case 2:
        return 0.04264;

      case 3:
        return 0.01580;

      case 4:
        return 0.007193;

      case 5:
        return 0.003730;

      case 6:
        return 0.002120;

      case 7:
        return 0.001292;

      case 8:
        return 0.0008314;

      case 9:
        return 0.0005586;

      case 10:
        return 0.0003890;

      default:
        return 0.0;
    }
  }

  /* Upwinded */
  else
  {
    switch(order)
    {
      case 0:
        return 0.6963;

      case 1:
        return 0.07736; 

      case 2:
        return 0.01878;

      case 3:
        return 0.006345;

      case 4:
        return 0.002664;

      case 5:
        return 0.001299;

      case 6:
        return 0.0007060;

      case 7:
        return 0.0004153;

      case 8:
        return 0.0002599;

      case 9:
        return 0.0001708;

      case 10:
        return 0.0001168;

      default:
        return 0.0;
    }
  }
}
