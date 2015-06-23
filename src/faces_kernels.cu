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
     double momF = (U(fpt, 1, slot) * U(fpt, 1, slot) + U(fpt, 2, slot) * 
         U(fpt, 2, slot)) / U(fpt, 0, slot);

     P(fpt, slot) = (gamma - 1.0) * (U(fpt, 3, slot) - 0.5 * momF);
     double H = (U(fpt, 3, slot) + P(fpt, slot)) / U(fpt, 0, slot);

     F(fpt, 0, 0, slot) = U(fpt, 1, slot);
     F(fpt, 1, 0, slot) = U(fpt, 1, slot) * U(fpt, 1, slot) / U(fpt, 0, slot) + P(fpt, slot);
     F(fpt, 2, 0, slot) = U(fpt, 1, slot) * U(fpt, 2, slot) / U(fpt, 0, slot);
     F(fpt, 3, 0, slot) = U(fpt, 1, slot) * H;

     F(fpt, 0, 1, slot) = U(fpt, 2, slot);
     F(fpt, 1, 1, slot) = U(fpt, 1, slot) * U(fpt, 2, slot) / U(fpt, 0, slot);
     F(fpt, 2, 1, slot) = U(fpt, 2, slot) * U(fpt, 2, slot) / U(fpt, 0, slot) + P(fpt, slot);
     F(fpt, 3, 1, slot) = U(fpt, 2, slot) * H;
   }
}

void compute_Fconv_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> F_gfpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> P_gfpts, 
    unsigned int nFpts, double gamma)
{
  unsigned int threads = 192;
  unsigned int blocks = (nFpts + threads - 1)/threads;

  compute_Fconv_fpts_2D_EulerNS<<<blocks, threads>>>(F_gfpts, U_gfpts, P_gfpts, nFpts, gamma);
}

