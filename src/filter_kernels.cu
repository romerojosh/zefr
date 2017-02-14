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

#include "mdvector_gpu.h"

__global__
void normalize_data(mdvector_gpu<double> U_spts, double normalTol, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  for (unsigned int var = 0; var < nVars; var++)
  {
    // Find element maximum and minimum
    double uMax = U_spts(0, ele, var), uMin = U_spts(0, ele, var);

    for (unsigned int spt = 1; spt < nSpts; spt++)
    {
      uMax = max(uMax, U_spts(spt, ele, var));
      uMin = min(uMin, U_spts(spt, ele, var));
    }
    
    if (uMax - uMin > normalTol)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
        U_spts(spt,ele,var) = (U_spts(spt,ele,var) - uMin) / (uMax - uMin);
    }
  }
}

void normalize_data_wrapper(mdvector_gpu<double>& U_spts, double normalTol, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/threads;

  normalize_data<<<blocks, threads>>>(U_spts, normalTol, nSpts, nEles, nVars);
}

__global__
void compute_sensor(mdvector_gpu<double> KS, mdvector_gpu<double> sensor,
    double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int nSptsKS, double Q, double epsilon)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  double max_sen = 0.0;

  for (unsigned int var = 0; var < nVars; var++)
  {
    double sen = 0.0;
    for (unsigned int row = 0; row < nDims * nSptsKS; row++)
    {
      KS(row, ele, var) = pow(1.0/epsilon , Q/2.0) * pow(abs(KS(row, ele, var)), Q);
      sen = max(sen, KS(row, ele, var));
    }
    max_sen = max(max_sen, sen);
  }

  sensor(ele) = max_sen;
}

void compute_sensor_wrapper(mdvector_gpu<double>& KS, mdvector_gpu<double>& sensor,
    double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int nSptsKS, double Q, double epsilon)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  compute_sensor<<<blocks, threads>>>(KS, sensor, threshJ, nSpts, nEles, nVars, nDims, nSptsKS, Q, epsilon);

}

__global__
void copy_filtered_solution(mdvector_gpu<double> U_filt, mdvector_gpu<double> U_spts,
    mdvector_gpu<double> sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  // Check for sensor value
  if (sensor(ele) < threshJ)
    return; 

  for (unsigned int var = 0; var < nVars; var++)
    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, var) = U_filt(spt, ele, var);

}

void copy_filtered_solution_wrapper(mdvector_gpu<double>& U_filt, mdvector_gpu<double>& U_spts,
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  copy_filtered_solution<<<blocks, threads>>>(U_filt, U_spts, sensor, threshJ, nSpts, nEles, nVars);
}

