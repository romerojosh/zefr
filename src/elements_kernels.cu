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
void compute_Fvisc_spts_2D_EulerNS(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, mdvector_gpu<double> dU_spts, 
    unsigned int nSpts, unsigned int nEles, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Setting variables for convenience */
  /* States */
  double rho = U_spts(spt, ele, 0);
  double momx = U_spts(spt, ele, 1);
  double momy = U_spts(spt, ele, 2);
  double e = U_spts(spt, ele, 3);

  double u = momx / rho;
  double v = momy / rho;
  double e_int = e / rho - 0.5 * (u*u + v*v);

  /* Gradients */
  double rho_dx = dU_spts(spt, ele, 0, 0);
  double momx_dx = dU_spts(spt, ele, 1, 0);
  double momy_dx = dU_spts(spt, ele, 2, 0);
  double e_dx = dU_spts(spt, ele, 3, 0);
  
  double rho_dy = dU_spts(spt, ele, 0, 1);
  double momx_dy = dU_spts(spt, ele, 1, 1);
  double momy_dy = dU_spts(spt, ele, 2, 1);
  double e_dy = dU_spts(spt, ele, 3, 1);

  /* Set viscosity */
  double mu;
  if (fix_vis)
  {
    mu = mu_in;
  }
  else
  {
    double rt_ratio = (gamma - 1.0) * e_int / (rt);
    mu = mu_in * std::pow(rt_ratio,1.5) * (1. + c_sth) / (rt_ratio + c_sth);
  }

  double du_dx = (momx_dx - rho_dx * u) / rho;
  double du_dy = (momx_dy - rho_dy * u) / rho;

  double dv_dx = (momy_dx - rho_dx * v) / rho;
  double dv_dy = (momy_dy - rho_dy * v) / rho;

  double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
  double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

  double de_dx = (e_dx - dke_dx - rho_dx * e_int) / rho;
  double de_dy = (e_dy - dke_dy - rho_dy * e_int) / rho;

  double diag = (du_dx + dv_dy) / 3.0;

  double tauxx = 2.0 * mu * (du_dx - diag);
  double tauxy = mu * (du_dy + dv_dx);
  double tauyy = 2.0 * mu * (dv_dy - diag);

  /* Set viscous flux values */
  F_spts(spt, ele, 1, 0) -= tauxx;
  F_spts(spt, ele, 2, 0) -= tauxy;
  F_spts(spt, ele, 3, 0) -= (u * tauxx + v * tauxy + (mu / prandtl) *
      gamma * de_dx);

  F_spts(spt, ele, 1, 1) -= tauxy;
  F_spts(spt, ele, 2, 1) -= tauyy;
  F_spts(spt, ele, 3, 1) -= (u * tauxy + v * tauyy + (mu / prandtl) *
          gamma * de_dy);

}

void compute_Fvisc_spts_2D_EulerNS_wrapper(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, mdvector_gpu<double> dU_spts, 
    unsigned int nSpts, unsigned int nEles, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  dim3 threads(32,32);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  compute_Fvisc_spts_2D_EulerNS<<<threads, blocks>>>(F_spts, 
    U_spts, dU_spts, nSpts, nEles, gamma, prandtl, mu_in, c_sth, rt, fix_vis);

}

__global__
void transform_dU_quad(mdvector_gpu<double> dU_spts, 
    mdvector_gpu<double> jaco_spts, mdvector_gpu<double> jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int var = blockDim.z * blockIdx.z + threadIdx.z;

  if (spt >= nSpts || ele >= nEles || var >= nVars)
    return;

  double dUtemp = dU_spts(spt, ele, var, 0);

  dU_spts(spt, ele, var, 0) = dU_spts(spt, ele, var, 0) * jaco_spts(1, 1, spt, ele) - 
                            dU_spts(spt, ele, var, 1) * jaco_spts(1, 0, spt, ele); 

  dU_spts(spt, ele, var, 1) = dU_spts(spt, ele, var, 1) * jaco_spts(0, 0, spt, ele) -
                            dUtemp * jaco_spts(0, 1, spt, ele);

  dU_spts(spt, ele, var, 0) /= jaco_det_spts(spt, ele);
  dU_spts(spt, ele, var, 1) /= jaco_det_spts(spt, ele);

}

void transform_dU_quad_wrapper(mdvector_gpu<double> dU_spts, 
    mdvector_gpu<double> jaco_spts, mdvector_gpu<double> jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  dim3 threads(16, 16, 4);
  dim3 blocks((nSpts + threads.x - 1) / threads.x, (nEles + threads.y - 1) / 
      threads.y, (nVars + threads.z - 1) / threads.z);

  transform_dU_quad<<<threads, blocks>>>(dU_spts, jaco_spts, jaco_det_spts,
      nSpts, nEles, nVars);
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
