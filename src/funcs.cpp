#include <cmath>

#include "input.hpp"

double compute_U_true(double x, double y, double t, unsigned int var, const InputStruct *input)
{
   
  double val;

  if (input->equation == "AdvDiff")
    val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
           std::sin(M_PI * (x - input->AdvDiff_Ax * t))* 
           std::sin(M_PI * (y - input->AdvDiff_Ay * t));
  else
    ThrowException("Under construction!");

  return val;
}
