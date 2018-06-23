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
    double uMax = U_spts(0, var, ele), uMin = U_spts(0, var, ele);

    for (unsigned int spt = 1; spt < nSpts; spt++)
    {
      uMax = max(uMax, U_spts(spt, var, ele));
      uMin = min(uMin, U_spts(spt, var, ele));
    }
    
    if (uMax - uMin > normalTol)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
        U_spts(spt, var, ele) = (U_spts(spt, var, ele) - uMin) / (uMax - uMin);
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
      KS(row, var, ele) = pow(1.0/epsilon , Q/2.0) * pow(abs(KS(row, var, ele)), Q);
      sen = max(sen, KS(row, var, ele));
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
      U_spts(spt, var, ele) = U_filt(spt, var, ele);

}

void copy_filtered_solution_wrapper(mdvector_gpu<double>& U_filt, mdvector_gpu<double>& U_spts,
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  copy_filtered_solution<<<blocks, threads>>>(U_filt, U_spts, sensor, threshJ, nSpts, nEles, nVars);
}

