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

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

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
void compute_max_sensor(mdvector_gpu<double> KS, mdvector_gpu<double> sensor,
    mdvector_gpu<unsigned int> sensor_bool, double threshJ, unsigned int order, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars, unsigned int nDims, double Q)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  double max_sen = 0.0;

  for (unsigned int var = 0; var < nVars; var++)
  {
    double sen = 0.0;
    for (unsigned int row = 0; row < nDims * nSpts; row++)
    {
      KS(row, ele, var) = pow((double)order+1,Q/2) * pow(abs(KS(row, ele, var)), Q);
      //KS(row, ele, var) = order * (KS(row, ele, var) * KS(row, ele, var)); // Kartikey's version
      sen = max(sen, KS(row, ele, var));
    }
    max_sen = max(max_sen, sen);
  }

  sensor(ele) = max_sen;

  sensor_bool(ele) = max_sen > threshJ;
}

void compute_max_sensor_wrapper(mdvector_gpu<double>& KS, mdvector_gpu<double>& sensor,
    unsigned int order, double& max_sensor, mdvector_gpu<unsigned int>& sensor_bool, double threshJ,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, double Q)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  compute_max_sensor<<<blocks, threads>>>(KS, sensor, sensor_bool, threshJ, order, nSpts, nEles, nVars, nDims, Q);


  /* Get max sensor value using thrust */
  thrust::device_ptr<double> s_ptr = thrust::device_pointer_cast(sensor.data());
  thrust::device_ptr<double> max_ptr = thrust::max_element(s_ptr, s_ptr + nEles);
  max_sensor = max_ptr[0];
}

__global__
void copy_filtered_solution_ka(mdvector_gpu<double> U_spts_filt, mdvector_gpu<double> U_spts,
    mdvector_gpu<double> sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  // Check for sensor value
  if (sensor(ele) < threshJ) return; //TODO: This causes divergence. Need to address.

  for (unsigned int var = 0; var < nVars; var++)
    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, var) = U_spts_filt(spt, ele, var);
}

void copy_filtered_solution_wrapper_ka(mdvector_gpu<double>& U_spts_filt, mdvector_gpu<double>& U_spts,
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/threads;

  copy_filtered_solution_ka<<<blocks, threads>>>(U_spts_filt, U_spts, sensor, threshJ, nSpts, nEles, nVars);
}

__global__
void copy_filtered_solution(mdvector_gpu<double> U_spts_limited, mdvector_gpu<double> U_spts,
    mdvector_gpu<double> sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, int type)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  // Check for sensor value
  if ( (type ==1 && sensor(ele) < threshJ) || (type == 2 && sensor(ele) > threshJ) )
    return; //TODO: This causes divergence. Need to address.

  for (unsigned int var = 0; var < nVars; var++)
    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, var) = U_spts_limited(spt, ele, var);

}

void copy_filtered_solution_wrapper(mdvector_gpu<double>& U_spts_limited, mdvector_gpu<double>& U_spts,
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, int type)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  copy_filtered_solution<<<blocks, threads>>>(U_spts_limited, U_spts, sensor, threshJ, nSpts, nEles, nVars, type);
}

__device__
double max_3(double a, double b, double c)
{
  double m = a;
  (m < b) && (m = b);
  (m < c) && (m = c);
  return m;
}

__global__
void limiter_gpu(uint nEles, uint nFaces, uint nVars, double threshJ,
    mdvector_gpu<int> ele_adj,  mdvector_gpu<double> sensor, mdvector_gpu<double> Umodal)
{
  const uint ele = blockDim.x * blockIdx.x + threadIdx.x;

  if(ele >= nEles) return;

  // Check for sensor value
  if (sensor(ele) < threshJ) return;    //TODO: This causes divergence. Need to address.

  for(uint var = 0; var < nVars; var++)
  {
    double Ucurr = 0.5*Umodal(0, ele, var);
    double Umin = Ucurr;
    double Umax = Ucurr;
    for(uint face = 0; face < nFaces; face++)
    {
      int ele2 = ele_adj(face, ele);
      if(ele2 != -1)
      {
        Umax = 0.5*Umodal(0, ele2, var) > Umax ? 0.5*Umodal(0, ele2, var) : Umax;
        Umin = 0.5*Umodal(0, ele2, var) < Umin ? 0.5*Umodal(0, ele2, var) : Umin;
      }
    }

    double Ubound = fmin(Ucurr - Umin, Umax - Ucurr);
    double x = Umodal(1, ele, var) + Umodal(2, ele, var);
    double y = Umodal(1, ele, var) - Umodal(2, ele, var);

    double dx = max_3(-Ubound - x, 0, x - Ubound);
    double dy = max_3(-Ubound - x, 0, x - Ubound);
    x = x + dx;
    y = y + dy;
    Umodal(1, ele, var) = 0.5*(x + y);
    Umodal(2, ele, var) = 0.5*(x - y);
  }
}

void limiter_wrapper(uint nEles, uint nFaces, uint nVars, double threshJ, mdvector_gpu<int> ele_adj,
  mdvector_gpu<double> sensor, mdvector_gpu<double>& Umodal)
{
  uint numThreads = 256;
  uint numBlocks = (nEles + numThreads - 1)/numThreads;

  limiter_gpu<<<numBlocks, numThreads>>>(nEles, nFaces, nVars, threshJ, ele_adj, sensor, Umodal);
}
