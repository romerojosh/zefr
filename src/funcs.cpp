#include <cmath>

#include "input.hpp"

double compute_U_true(double x, double y, double t, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == "AdvDiff")
  {
    val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
           std::sin(M_PI * (x - input->AdvDiff_A[0] * t))* 
           std::sin(M_PI * (y - input->AdvDiff_A[1] * t));
  }
  else if (input->equation == "EulerNS")
  {
    double G = 5.0;
    double R = 1.;

    double f = (1.0 - x*x - y*y)/R;

    double rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
              M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
    double Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f); 
    double Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
    double P = std::pow(rho, input->gamma);

    switch (var)
    {
      case 0:
        val = rho; break;
      case 1:
        val = rho * Vx; break;
      case 2:
        val = rho * Vy; break;
      case 3:
        val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}
