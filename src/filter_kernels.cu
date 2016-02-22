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

void normalize_data_wrapper(mdvector_gpu<double> U_spts, double normalTol, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  normalize_data<<<blocks, threads>>>(U_spts, normalTol, nSpts, nEles, nVars);
}

__global__
void compute_max_sensor(mdvector_gpu<double> KS, mdvector_gpu<double> sensor, 
    unsigned int order, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles) return;

  double max_sen = 0.0;

  for (unsigned int var = 0; var < nVars; var++)
  {
    double sen = 0.0;
    for (unsigned int row = 0; row < 2 * nSpts; row++)
    {
      KS(row, ele, var) = order * (KS(row, ele, var) * KS(row, ele, var));
      sen = max(sen, KS(row, ele, var));
    }
    max_sen = max(max_sen, sen);
  }

  sensor(ele) = max_sen;


}

void compute_max_sensor_wrapper(mdvector_gpu<double> KS, mdvector_gpu<double> sensor, 
    unsigned int order, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1)/threads;

  compute_max_sensor<<<blocks, threads>>>(KS, sensor, order, nSpts, nEles, nVars);
}
