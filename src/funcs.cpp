#include <cmath>

#include "input.hpp"

double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))* 
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t));
    }
    else if (input->nDims == 3)
    {
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))* 
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t))*
             std::sin(M_PI * (z - input->AdvDiff_A(2) * t));
    }
  }
  else if (input->equation == EulerNS)
  {
    if (!input->viscous || input->viscous)
    {
      double G = 5.0;
      double R = 1.;

      double f = (1.0 - x*x - y*y)/R;

      double rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
                M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
      double Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f); 
      double Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
      double P = std::pow(rho, input->gamma);

      if (input->nDims == 2)
      {
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
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * Vx; break;
          case 2:
            val = rho * Vy; break;
          case 3:
            val = 0; break;
          case 4:
            val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
        }
      }
    }
    else
    {
      val = 1.0246950765959597 * y;
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}

double compute_dU_true(double x, double y, double z, double t, unsigned int var, 
    unsigned int dim, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      if (dim == 0)
      {
        val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
               std::sin(M_PI * (y - input->AdvDiff_A(1) * t)) *
               M_PI * std::cos(M_PI * (x - input->AdvDiff_A(0) * t));

      }
      else
      {
        val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
               std::sin(M_PI * (x - input->AdvDiff_A(0) * t)) *
               M_PI * std::cos(M_PI * (y - input->AdvDiff_A(1) * t));
      }
    }
    else if (input->nDims == 3)
    {
      ThrowException("Under construction!");
    }
  }
  else if (input->equation == EulerNS)
  {
    if (!input->viscous)
    {
      double G = 5.0;
      double R = 1.;

      double f = (1.0 - x*x - y*y)/R;

      double rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
                M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
      double Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f); 
      double Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
      double P = std::pow(rho, input->gamma);

      if (input->nDims == 2)
      {
        ThrowException("Under construction!");
      }
      else
      {
        ThrowException("Under construction!");
      }
    }
    else
    {
      if (dim == 0)
      {
        val = 0.0;
      }
      else
      {
        val = 1.0246950765959597;
      }
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}

double get_cfl_limit(int order)
{
  switch(order)
  {
    case 0:
      return 1.393;

    case 1:
      return 0.464; 

    case 2:
      return 0.235;

    case 3:
      return 0.139;

    case 4:
      return 0.100;

    case 5:
      return 0.068;

    default:
      ThrowException("CFL limit no available for this order!");
  }
}
