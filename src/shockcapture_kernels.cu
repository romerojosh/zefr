#include "mdvector_gpu.h"
#include "shockcapture_kernels.h"

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

    for (unsigned int spt = 0; spt < nSpts; spt++)
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
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  normalize_data<<<blocks, threads>>>(U_spts, normalTol, nSpts, nEles, nVars);
}

__global__
void compute_max_sensor(mdvector_gpu<double> KS, mdvector_gpu<double> sensor, 
    mdvector_gpu<unsigned int> sensor_bool, double threshJ, unsigned int order, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, double Q)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  double max_sen = 0.0;

  for (unsigned int var = 0; var < nVars; var++)
  {
    double sen = 0.0;
    for (unsigned int row = 0; row < 2 * nSpts; row++)
    {
      KS(row, ele, var) = pow(order+1,Q/2) * pow(abs(KS(row, ele, var)), Q);
      //KS(row, ele, var) = order * (KS(row, ele, var) * KS(row, ele, var));
      sen = max(sen, KS(row, ele, var));
    }
    max_sen = max(max_sen, sen);
  }

  sensor(ele) = max_sen;

  sensor_bool(ele) = max_sen > threshJ;
}

void compute_max_sensor_wrapper(mdvector_gpu<double>& KS, mdvector_gpu<double>& sensor, 
    unsigned int order, double& max_sensor, mdvector_gpu<unsigned int>& sensor_bool, double threshJ, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, double Q)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  compute_max_sensor<<<blocks, threads>>>(KS, sensor, sensor_bool, threshJ, order, nSpts, nEles, nVars, Q);


  /* Get max sensor value using thrust */
  thrust::device_ptr<double> s_ptr = thrust::device_pointer_cast(sensor.data());
  thrust::device_ptr<double> max_ptr = thrust::max_element(s_ptr, s_ptr + nEles);
  max_sensor = max_ptr[0];

}

__global__
void copy_filtered_solution(mdvector_gpu<double> U_spts_limited, mdvector_gpu<double> U_spts, 
    mdvector_gpu<double> sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  // Check for sensor value
  if (sensor(ele) < threshJ) return; //TODO: This causes divergence. Need to address.

  for (unsigned int var = 0; var < nVars; var++)
    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, var) = U_spts_limited(spt, ele, var);

}

void copy_filtered_solution_wrapper(mdvector_gpu<double>& U_spts_limited, mdvector_gpu<double>& U_spts, 
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  copy_filtered_solution<<<blocks, threads>>>(U_spts_limited, U_spts, sensor, threshJ, nSpts, nEles, nVars);
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
void limiter_gpu(uint nEles, uint nFaces, uint nVars, double threshJ, mdvector_gpu<int> ele_adj, 
    mdvector_gpu<double> sensor, mdvector_gpu<double> Umodal)
{
  const uint ele = blockDim.x * blockIdx.x + threadIdx.x; 

  if(ele >= nEles) return;

  // Check for sensor value
  if (sensor(ele) < threshJ) return;    //TODO: This causes divergence. Need to address.

  int nbr;
  double Ucurr, Umin, Umax;
  for(uint var = 0; var < nVars; var++)
  {
    Ucurr = 0.5*Umodal(0, ele, var);
    Umin = Ucurr; Umax = Ucurr;
    for(uint face = 0; face < nFaces; face++)
    {
      nbr = ele_adj(face, ele);
      if(nbr != -1)
      {
        Umax = 0.5*Umodal(0, nbr, var) > Umax ? 0.5*Umodal(0, nbr, var) : Umax;
        Umin = 0.5*Umodal(0, nbr, var) < Umin ? 0.5*Umodal(0, nbr, var) : Umin;
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

// void compute_primitive_gpu(uint nSpts, uint nEles, uint nVars, mdvector_gpu<double> U_spts, mdvector_gpu<double> U_prim)
// {
//   const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;
//   if (ele >= nEles) return;

//   for (unsigned int spt = 0; spt < nSpts; spt++)
//   {
    
//   }

// }

// void compute_primitive_wrapper(uint nSpts, uint nEles, uint nVars, mdvector_gpu<double>& U_spts, mdvector_gpu<double>& U_prim)
// {
//   uint numThreads = 256;
//   uint numBlocks = (nEles + numThreads - 1)/numThreads;

//   compute_primitive_gpu<<<numBlocks, numThreads>>>(nSpts, nEles, nVars, U_spts, U_prim);  
// }

