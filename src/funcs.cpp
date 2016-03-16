/*!
 * \file funcs.cpp
 * \author J. Romero, Stanford University
 * \brief This file contains several miscellaneous functions used in the code.
 */

#include <cmath>

#include "cblas.h"

#ifdef _OMP
#include "omp.h"
#endif

#include "input.hpp"
#include "funcs.hpp"

double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      /*
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))* 
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t));
             */
      val =  std::sin(2 * M_PI * x) + std::sin(2 * M_PI * y);
    }
    else if (input->nDims == 3)
    {
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))* 
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t))*
             std::sin(M_PI * (z - input->AdvDiff_A(2) * t));
      //val =  std::sin(2 * M_PI * x) + std::sin(2 * M_PI * y) + std::sin(2 * M_PI * z);
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
      val = 0.0; // Just a placeholder value for now.
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


double compute_source_term(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
  double val = 0.;
  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
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
    ThrowException("No source defined for EulerNS!");
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

mdvector<double> get_alpha_opt(int order, int nStages, double CFLfac)
{
  mdvector<double> rk_alpha({nStages});
  rk_alpha(nStages - 1) = 1.0;

  switch(order)
  {
    case 0:
      switch(nStages)
      {
        case 4:
          // r = 0.5 
          //rk_alpha(0) = 0.301; rk_alpha(1) = 0.0; rk_alpha(2) = 0.294; break; //trapz(rho)
          //rk_alpha(0) = 1.0; rk_alpha(1) = 0.212; rk_alpha(2) = 0.238; break; //trapz(rho*k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.0490; rk_alpha(2) = 0.446; break; //trapz(rho/k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.236; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)

          rk_alpha(0) = 1./4.; rk_alpha(1) = 1./3.; rk_alpha(2) = 1./2.; break; //standard
          //rk_alpha(0) = 0.2266; rk_alpha(1) = 0.2352; rk_alpha(2) = 0.6596; break; //sum(rho<0.6)

        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;

    case 1:
      switch(nStages)
      {
        case 4:
          // r = 0.5 
          rk_alpha(0) = 0.0; rk_alpha(1) = 0.248; rk_alpha(2) = 0.680; break; //trapz(rho)
          //rk_alpha(0) = 0.00258; rk_alpha(1) = 0.172; rk_alpha(2) = 0.5; break; //trapz(rho*k)
          //rk_alpha(0) = 0.164; rk_alpha(1) = 0.393; rk_alpha(2) = 1.0; break; //trapz(rho/k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.288; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)
          
          //rk_alpha(0) = 0.2070; rk_alpha(1) = 0.4763; rk_alpha(2) = 0.9695; break; //sum(rho < 0.6)

        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;

    case 2:
      switch(nStages)
      {
        case 3:
          // r = 0.5
          rk_alpha(0) = 0.346; rk_alpha(1) = 0.866; break; //trapz(rho)
        case 4:
          // r = 0.5 
          rk_alpha(0) = 0.153; rk_alpha(1) = 0.442; rk_alpha(2) = 0.931; break; //trapz(rho)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.258; rk_alpha(2) = 0.611; break; //trapz(rho*k)
          //rk_alpha(0) = 0.0776; rk_alpha(1) = 0.389; rk_alpha(2) = 1.0; break; //trapz(rho/k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.336; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)
          
          //rk_alpha(0) = 0.0181; rk_alpha(1) = 0.469; rk_alpha(2) = 1.0; break; //trapz(rho)
          //rk_alpha(0) = 0.1894; rk_alpha(1) = 0.5262; rk_alpha(2) = 1.0; break; //sum(rho < 0.6)
          //rk_alpha(0) = 0.001; rk_alpha(1) = 0.3870; rk_alpha(2) = 0.975; break; //sum(rho < 0.7)
          // r = 0.8
          //
          //
          //rk_alpha(0) = 0.184; rk_alpha(1) = 0.421; rk_alpha(2) = 0.776; break; //trapz(rho)
          //
          
        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;

    case 3:
      switch(nStages)
      {
        case 4:
          // r = 0.5 
          rk_alpha(0) = 0.158; rk_alpha(1) = 0.477; rk_alpha(2) = 1.0; break; //trapz(rho)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.303; rk_alpha(2) = 0.694; break; //trapz(rho*k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.368; rk_alpha(2) = 1.0; break; //trapz(rho/k)
          //rk_alpha(0) = 0.0392; rk_alpha(1) = 0.389; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)

        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;

    case 4:
      switch(nStages)
      {
        case 4:
          // r = 0.5 
          rk_alpha(0) = 0.182; rk_alpha(1) = 0.497; rk_alpha(2) = 1.0; break; //trapz(rho)
          //rk_alpha(0) = 0.0626; rk_alpha(1) = 0.361; rk_alpha(2) = 0.763; break; //trapz(rho*k)
          //rk_alpha(0) = 0.0801; rk_alpha(1) = 0.411; rk_alpha(2) = 1.0; break; //trapz(rho/k)
          //rk_alpha(0) = 0.176; rk_alpha(1) = 0.475; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)

        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;

    case 5:
      switch(nStages)
      {
        case 4:
          // r = 0.5 
          rk_alpha(0) = 0.123; rk_alpha(1) = 0.461; rk_alpha(2) = 1.0; break; //trapz(rho)
          //rk_alpha(0) = 0.0365; rk_alpha(1) = 0.372; rk_alpha(2) = 0.815; break; //trapz(rho*k)
          //rk_alpha(0) = 0.0; rk_alpha(1) = 0.376; rk_alpha(2) = 1.0; break; //trapz(rho/k)
          //rk_alpha(0) = 0.258; rk_alpha(1) = 0.446; rk_alpha(2) = 1.0; break; //trapz(rho/k^2)

        default:
          ThrowException("Optimized alpha not computed for this case!");
      } break;



    default:
      ThrowException("Optimized alpha not computed for this case!");
  }

  return rk_alpha;

}

#ifdef _OMP
void omp_blocked_dgemm(CBLAS_ORDER mode, CBLAS_TRANSPOSE transA, 
    CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, double* A, int lda, 
    double* B, int ldb, double beta, double* C, int ldc)
{
#pragma omp parallel
    {
      int nThreads = omp_get_num_threads();
      int thread_idx = omp_get_thread_num();

      int block_size = N / nThreads;
      int start_idx = block_size * thread_idx;

      if (thread_idx == nThreads-1)
        block_size += N % (block_size);

      cblas_dgemm(mode, transA, transB, M, block_size, K, alpha, A, lda, 
          B + ldb * start_idx, ldb, beta, C + ldc * start_idx, ldc);
  }
}
#endif

