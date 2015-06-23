#include "faces_kernels.h"
#include "mdvector_gpu.h"

__global__
void compute_Fconv_fpts_2D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, mdvector_gpu<double> P, 
    unsigned int nFpts, double gamma)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;

  if (fpt >= nFpts)
    return;

   for (unsigned int slot = 0; slot < 2; slot ++)
   {
     /* Compute some primitive variables (keep pressure)*/
     double momF = (U(slot, 1, fpt) * U(slot, 1, fpt) + U(slot, 2, fpt) * 
         U(slot, 2, fpt)) / U(slot, 0, fpt);

     P(slot, fpt) = (gamma - 1.0) * (U(slot, 3, fpt) - 0.5 * momF);
     double H = (U(slot, 3, fpt) + P(slot,fpt)) / U(slot, 0, fpt);

     F(slot, 0, 0, fpt) = U(slot, 1, fpt);
     F(slot, 1, 0, fpt) = U(slot, 1, fpt) * U(slot, 1, fpt) / U(slot, 0, fpt) + P(slot, fpt);
     F(slot, 2, 0, fpt) = U(slot, 1, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt);
     F(slot, 3, 0, fpt) = U(slot, 1, fpt) * H;

     F(slot, 0, 1, fpt) = U(slot, 2, fpt);
     F(slot, 1, 1, fpt) = U(slot, 1, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt);
     F(slot, 2, 1, fpt) = U(slot, 2, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt) + P(slot, fpt);
     F(slot, 3, 1, fpt) = U(slot, 2, fpt) * H;
   }
}

void compute_Fconv_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> F_gfpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> P_gfpts, 
    unsigned int nFpts, double gamma)
{
  unsigned int threads = 192;
  unsigned int blocks = (nFpts + threads - 1)/threads;

  compute_Fconv_fpts_2D_EulerNS<<<blocks, threads>>>(F_gfpts, U_gfpts, P_gfpts, nFpts, gamma);
}

