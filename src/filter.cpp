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

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

extern "C" {
#include "cblas.h"
}
#include "funcs.hpp"
#include "quads.hpp"
#include "polynomials.hpp"
#include "points.hpp"
#include "filter.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#include "filter_kernels.h"
#endif


void Filter::setup(InputStruct *input, FRSolver &solver)
{
  /* Assign local pointers */
  this->input = input;
  this->solver = &solver;

  /* Setup required matrices and allocate data */
  for (auto e : solver.elesObjs)
  {
    e->setup_filter();
    U_ini[e->etype] = e->U_spts;
    U_filt[e->etype] = e->U_spts;
    sensor[e->etype].assign({e->nEles});
    sensor_bool[e->etype].assign({e->nEles});
    int nSptsKS = e->nSpts1D * e->nSpts1D;
    if (e->nDims == 3) nSptsKS *= e->nSpts1D;
    KS[e->etype].assign({e->nDims * nSptsKS, e->nVars, e->nEles});

#ifdef _GPU
    U_ini_d[e->etype] = U_ini[e->etype];
    U_filt_d[e->etype] = U_filt[e->etype];
    sensor_d[e->etype] = sensor[e->etype];
    sensor_bool_d[e->etype] = sensor_bool[e->etype];
    KS_d[e->etype] = KS[e->etype];
#endif
  }

  setup_threshold();

}


void Filter::setup_threshold()
{
  // Local arrays
  mdvector<double> u_canon, KS_canon, KS_step, KS_ramp;

  for (auto e : solver->elesObjs)
  {
    u_canon.assign({e->nSpts1D, 2});
    KS_canon.assign({e->nSpts1D, 2});
    KS_step.assign({e->nSpts1D});
    KS_ramp.assign({e->nSpts1D});
    
    // Centered step in parent domain
    for (unsigned int spt = 0; spt < e->nSpts1D; spt++)
      u_canon(spt, 0) = step(e->loc_spts_1D[spt]);
      
    // Ramp in parent domain
    for (unsigned int spt = 0; spt < e->nSpts1D; spt++)
      u_canon(spt, 1) = step(e->loc_spts_1D[spt]) * e->loc_spts_1D[spt];
    
    // Evaluate filtered kernel
    auto &A = e->oppS_1D(0, 0);
    auto &B = u_canon(0, 0);
    auto &C = KS_canon(0, 0);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
      e->nSpts1D, 2, e->nSpts1D, 1.0, &A, e->nSpts1D, &B, 2, 0.0, &C, 2);
    
    // Apply non-linear enhancement
    double epsilon = std::log(e->order)/e->order;
    double Q = input->nonlin_exp;
    for (unsigned int spt = 0; spt < e->nSpts1D; spt++)
    {
      KS_step(spt) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS_canon(spt, 0)), Q);
      KS_ramp(spt) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS_canon(spt, 1)), Q);
    }
    
    // Calculate threshold
    threshJ[e->etype] = (1.0 - input->sen_Jfac) * KS_ramp.max_val() + input->sen_Jfac * KS_step.max_val();
      
    // Print results
    if (input->rank == 0) 
    {
      std::cout << " Sensor threshold = " << threshJ[e->etype] << std::endl;
    }
  }
}


void Filter::apply_sensor()
{
#ifdef _CPU
  for (auto e : solver->elesObjs)
  {
    //Initialize sensor
    sensor[e->etype].fill(0.0); 
    
    // Copy data to local structure
    U_ini[e->etype] = e->U_spts;
    
    // Normalize data
    double normalTol = 0.1;
    if (input->sen_norm)
    {
      for (unsigned int var = 0; var < e->nVars; var++)
      {
        // Find element maximum and minimum
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          double uMax = U_ini[e->etype](0, var, ele), uMin = U_ini[e->etype](0, var, ele);

          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            uMax = std::max(uMax, U_ini[e->etype](spt, var, ele));
            uMin = std::min(uMin, U_ini[e->etype](spt,var , ele));
          }
          
          if (uMax - uMin > normalTol)
          {
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
              U_ini[e->etype](spt, var, ele) = (U_ini[e->etype](spt, var, ele) - uMin) / (uMax - uMin);
          }
        }
      }
    }
    
    // Calculate KS
    int nSptsKS = e->nSpts1D * e->nSpts1D;
    if (e->nDims == 3) nSptsKS *= e->nSpts1D;

    auto &A = e->oppS(0,0);
    auto &B = U_ini[e->etype](0, 0, 0);
    auto &C = KS[e->etype](0, 0, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      e->nDims * nSptsKS, e->nEles * e->nVars, e->nSpts, 1.0, &A, e->nSpts,
      &B, e->nEles * e->nVars, 0.0, &C, e->nEles * e->nVars);

    // Apply non-linear enhancement and store sensor values
    double Q = input->nonlin_exp;
    double epsilon = log(e->order)/e->order;
    for (unsigned int var = 0; var < e->nVars; var++)
    {

      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double sen = 0.0;
        for (unsigned int row = 0; row < e->nDims*nSptsKS; row++)
        {
          KS[e->etype](row, var, ele) = pow(1.0/epsilon, Q/2.0) * pow(abs(KS[e->etype](row, var, ele)), Q);
          sen = std::max(sen, KS[e->etype](row, var, ele));
        }
        sensor[e->etype](ele) = std::max(sensor[e->etype](ele), sen);
        sensor_bool[e->etype](ele) = sensor[e->etype](ele) > threshJ[e->etype];
      }
    }
  }
#endif

#ifdef _GPU  
  for (auto e : solver->elesObjs)
  {
    // Copy data to local structure
    device_copy(U_ini_d[e->etype], e->U_spts_d, U_ini_d[e->etype].max_size());
    
    // Normalize data
    double normalTol = 0.1;
    if (input->sen_norm)
    {
      normalize_data_wrapper(U_ini_d[e->etype], normalTol, e->nSpts, e->nEles, e->nVars);
    }
    
    // Calculate KS
    int nSptsKS = e->nSpts1D * e->nSpts1D;
    if (e->nDims == 3) nSptsKS *= e->nSpts1D;

    auto *A = e->oppS_d.data();
    auto *B = U_ini_d[e->etype].data();
    auto *C = KS_d[e->etype].data();

    cublasDGEMM_wrapper(e->nElesPad * e->nVars, e->nDims * nSptsKS, e->nSpts, 1.0, B, e->nElesPad * e->nVars,
      A, e->nSpts, 0.0, C, e->nElesPad * e->nVars);
    
    // Apply non-linear enhancement and store sensor values
    double epsilon = log(e->order)/e->order;
    compute_sensor_wrapper(KS_d[e->etype], sensor_d[e->etype], threshJ[e->etype], e->nSpts,
      e->nEles, e->nVars, e->nDims, nSptsKS, input->nonlin_exp, epsilon);
  }

#endif
}

void Filter::apply_expfilter()
{
#ifdef _CPU
  for (auto e : solver->elesObjs)
  {
    auto &A = e->oppF(0,0);
    auto &B = e->U_spts(0, 0, 0);
    auto &C = U_filt[e->etype](0, 0, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nSpts, e->nEles * e->nVars,
      e->nSpts, 1.0, &A, e->nSpts, &B, e->nEles * e->nVars, 0.0, &C, e->nEles * e->nVars);

    // Copy back to e->U_Spts only when sensor is greater than threshold
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      // Check for sensor value
      if (sensor[e->etype](ele) < threshJ[e->etype]) continue;

      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
          e->U_spts(spt, var, ele) = U_filt[e->etype](spt, var, ele);
    }
  }

#endif

#ifdef _GPU
  for (auto e : solver->elesObjs)
  {
    auto *A = e->oppF_d.data();
    auto *B = e->U_spts_d.data();
    auto *C = U_filt_d[e->etype].data();

    cublasDGEMM_wrapper(e->nElesPad * e->nVars, e->nSpts,
      e->nSpts, 1.0, B, e->nElesPad * e->nVars, A, e->nSpts, 0.0, C, e->nElesPad * e->nVars);

    // Copy back to e->U_Spts only when sensor is greater than threshold
    copy_filtered_solution_wrapper(U_filt_d[e->etype], e->U_spts_d, sensor_d[e->etype], threshJ[e->etype], e->nSpts, e->nEles, e->nVars);
  }
#endif
}

double calc_expfilter_coeffs(unsigned int P, unsigned int nModes, double alpha, double s, unsigned int nDims, int mode)
{
  // Evaluate exponential filter
  double sigma, eta;
  sigma = 1;

  if (nDims == 2)
  {
    // Threshold no. of modes below which there is no effect of filter
    double eta_c = 2.0 / nModes;

    if (mode >= nModes)
      ThrowException("ERROR: Invalid mode when evaluating exponential filter ....");

    int m = 0;
    for (int k = 0; k <= 2*P; k++)
    {
      for (int j = 0; j <= k; j++)
      {
        int i = k-j;
        if (i <= P && j <= P)
        {
          if (m == mode) // found the correct mode
          {
            eta = (double)(i+j) / nModes; //TODO: The divisor is not consistent with Abhishek's thesis here.

            sigma = exp(-alpha*pow(eta,s));

            //if(eta <= eta_c)
            //sigma = 1;
            //else
            //sigma = exp(-alpha*pow( (eta - eta_c)/(1 - eta_c), s ));
          }
          m++;
        }
      }
    }
  }

  else if (nDims == 3)
  {
    // Threshold no. of modes below which there is no effect of filter
    double eta_c = 3.0 / nModes;

    if(mode >= nModes)
      ThrowException("ERROR: Invalid mode when evaluating exponential filter ....");

    unsigned int m = 0;

    for (unsigned int l = 0; l <= 3*P; l++)
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
              eta = (double)(i+j+k) / nModes;

              sigma = exp(-alpha*pow(eta,s));

              //if(eta <= eta_c)
              //sigma = 1;
              //else
              //sigma = exp(-alpha*pow( (eta - eta_c)/(1 - eta_c), s ));
            }
            m++;
          }
        }
      }
    }
  }

  return sigma;
}
