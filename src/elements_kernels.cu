#include "elements_kernels.h"
#include "mdvector_gpu.h"

__global__
void compute_Fconv_spts_2D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, 
    unsigned int nSpts, unsigned int nEles, double gamma)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Compute some primitive variables */
  double momF = (U(spt, ele, 1) * U(spt,ele,1) + U(spt, ele, 2) * 
      U(spt, ele,2)) / U(spt, ele, 0);
  double P = (gamma - 1.0) * (U(spt, ele, 3) - 0.5 * momF);
  double H = (U(spt, ele, 3) + P) / U(spt, ele, 0);


  F(spt, ele, 0, 0) = U(spt, ele, 1);
  F(spt, ele, 1, 0) = U(spt, ele, 1) * U(spt, ele, 1) / U(spt, ele, 0) + P;
  F(spt, ele, 2, 0) = U(spt, ele, 1) * U(spt, ele, 2) / U(spt, ele, 0);
  F(spt, ele, 3, 0) = U(spt, ele, 1) * H;

  F(spt, ele, 0, 1) = U(spt, ele, 2);
  F(spt, ele, 1, 1) = U(spt, ele, 1) * U(spt, ele, 2) / U(spt, ele, 0);
  F(spt, ele, 2, 1) = U(spt, ele, 2) * U(spt, ele, 2) / U(spt, ele, 0) + P;
  F(spt, ele, 3, 1) = U(spt, ele, 2) * H;
 
}

void compute_Fconv_spts_2D_EulerNS_wrapper(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts, unsigned int nEles,
    double gamma)
{
  dim3 threads(32,32);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  compute_Fconv_spts_2D_EulerNS<<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles, gamma);
}

__global__
void transform_flux_quad(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int var = blockDim.z * blockIdx.z + threadIdx.z;

  if (spt >= nSpts || ele >= nEles || var >= nVars)
    return;

  double Ftemp = F_spts(spt, ele, var, 0);

  F_spts(spt, ele, var, 0) = F_spts(spt, ele, var, 0) * jaco_spts(1, 1, spt, ele) -
                           F_spts(spt, ele, var, 1) * jaco_spts(0, 1, spt, ele);
  F_spts(spt, ele, var, 1) = F_spts(spt, ele, var, 1) * jaco_spts(0, 0, spt, ele) -
                           Ftemp * jaco_spts(1, 0, spt, ele);

}

void transform_flux_quad_wrapper(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars)
{
  dim3 threads(16, 16, 4);
  dim3 blocks((nSpts + threads.x - 1) / threads.x, (nEles + threads.y - 1) / 
      threads.y, (nVars + threads.z - 1) / threads.z);

  transform_flux_quad<<<threads, blocks>>>(F_spts, jaco_spts, nSpts, nEles, nVars);
}
