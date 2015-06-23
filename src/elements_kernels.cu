#include "elements_kernels.h"
#include "mdvector_gpu.h"

__global__
void compute_Fconv_spts_2D_EulerNS(mdvector_gpu<double> F_spts, mdvector_gpu<double> U_spts, unsigned int nSpts, 
    unsigned int nEles, double gamma)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Compute some primitive variables */
  double momF = (U_spts(spt, ele, 1) * U_spts(spt,ele,1) + U_spts(spt, ele, 2) * 
      U_spts(spt, ele,2)) / U_spts(spt, ele, 0);
  double P = (gamma - 1.0) * (U_spts(spt, ele, 3) - 0.5 * momF);
  double H = (U_spts(spt, ele, 3) + P) / U_spts(spt, ele, 0);


  F_spts(spt, ele, 0, 0) = U_spts(spt, ele, 1);
  F_spts(spt, ele, 1, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0) + P;
  F_spts(spt, ele, 2, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
  F_spts(spt, ele, 3, 0) = U_spts(spt, ele, 1) * H;

  F_spts(spt, ele, 0, 1) = U_spts(spt, ele, 2);
  F_spts(spt, ele, 1, 1) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
  F_spts(spt, ele, 2, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0) + P;
  F_spts(spt, ele, 3, 1) = U_spts(spt, ele, 2) * H;
 
}

void compute_Fconv_spts_2D_EulerNS_wrapper(mdvector_gpu<double> F_spts, mdvector_gpu<double> U_spts, unsigned int nSpts, 
    unsigned int nEles, double gamma)
{
  dim3 threads(32,32);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/threads.y);

  compute_Fconv_spts_2D_EulerNS<<<blocks, threads>>>(F_spts, U_spts, nSpts, nEles, gamma);
}
