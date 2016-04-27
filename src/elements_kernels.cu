#include "elements_kernels.h"
#include "input.hpp"
#include "mdvector_gpu.h"

template <unsigned int nDims>
__global__
void compute_Fconv_spts_AdvDiff(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts, unsigned int nEles, 
    mdvector_gpu<double> AdvDiff_A)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    F_spts(spt, ele, 0, dim) = AdvDiff_A(dim) * U_spts(spt, ele, 0);
  }
}

void compute_Fconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, mdvector_gpu<double> &AdvDiff_A)
{

  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_Fconv_spts_AdvDiff<2><<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles, AdvDiff_A);
  }
  else
  {
    compute_Fconv_spts_AdvDiff<3><<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles, AdvDiff_A);
  }
}

template <unsigned int nDims>
__global__
void compute_Fconv_spts_Burgers(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts, unsigned int nEles)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    F_spts(spt, ele, 0, dim) = 0.5 * U_spts(spt, ele, 0) * U_spts(spt, ele, 0);
  }
}

void compute_Fconv_spts_Burgers_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims)
{

  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_Fconv_spts_Burgers<2><<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles);
  }
  else
  {
    compute_Fconv_spts_Burgers<3><<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles);
  }
}

__global__
void compute_Fconv_spts_2D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, 
    unsigned int nSpts, unsigned int nEles, double gamma)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get state variables */
  double rho = U(spt, ele, 0);
  double momx = U(spt, ele, 1);
  double momy = U(spt, ele, 2);
  double ene = U(spt, ele, 3);

  /* Compute some primitive variables */
  double momF = (momx * momx + momy * momy) / rho;
  double P = (gamma - 1.0) * (ene - 0.5 * momF);
  double H = (ene + P) / rho;


  F(spt, ele, 0, 0) = momx;
  F(spt, ele, 1, 0) = momx * momx / rho + P;
  F(spt, ele, 2, 0) = momx * momy / rho;
  F(spt, ele, 3, 0) = momx * H;

  F(spt, ele, 0, 1) = momy;
  F(spt, ele, 1, 1) = momy * momx / rho;
  F(spt, ele, 2, 1) = momy * momy / rho + P;
  F(spt, ele, 3, 1) = momy * H;
 
}

__global__
void compute_Fconv_spts_3D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, 
    unsigned int nSpts, unsigned int nEles, double gamma)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get state variables */
  double rho = U(spt, ele, 0);
  double momx = U(spt, ele, 1);
  double momy = U(spt, ele, 2);
  double momz = U(spt, ele, 3);
  double ene = U(spt, ele, 4);

  /* Compute some primitive variables */
  double momF = (momx * momx + momy * momy + momz * momz) / rho;
  double P = (gamma - 1.0) * (ene - 0.5 * momF);
  double H = (ene + P) / rho;


  F(spt, ele, 0, 0) = momx;
  F(spt, ele, 1, 0) = momx * momx / rho + P;
  F(spt, ele, 2, 0) = momx * momy / rho;
  F(spt, ele, 3, 0) = momx * momz / rho;
  F(spt, ele, 4, 0) = momx * H;

  F(spt, ele, 0, 1) = momy;
  F(spt, ele, 1, 1) = momy * momx / rho;
  F(spt, ele, 2, 1) = momy * momy / rho + P;
  F(spt, ele, 3, 1) = momy * momz / rho;
  F(spt, ele, 4, 1) = momy * H;
 
  F(spt, ele, 0, 2) = momz;
  F(spt, ele, 1, 2) = momz * momx / rho;
  F(spt, ele, 2, 2) = momz * momy / rho;
  F(spt, ele, 3, 2) = momz * momz / rho + P;
  F(spt, ele, 4, 2) = momz * H;
}

void compute_Fconv_spts_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_Fconv_spts_2D_EulerNS<<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles, gamma);
  }
  else
  {
    compute_Fconv_spts_3D_EulerNS<<<blocks, threads>>>(F_spts, U_spts, nSpts, 
      nEles, gamma);
  }
}

template <unsigned int nDims>
__global__
void compute_Fvisc_spts_AdvDiff(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> dU_spts, unsigned int nSpts, unsigned int nEles, 
    double AdvDiff_D)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Can just add viscous flux to existing convective flux */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    F_spts(spt, ele, 0, dim) -= AdvDiff_D * dU_spts(spt, ele, 0, dim);
  }
}

void compute_Fvisc_spts_AdvDiff_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &dU_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, double AdvDiff_D)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_Fvisc_spts_AdvDiff<2><<<blocks, threads>>>(F_spts, dU_spts, nSpts, 
      nEles, AdvDiff_D);
  }
  else
  {
    compute_Fvisc_spts_AdvDiff<3><<<blocks, threads>>>(F_spts, dU_spts, nSpts, 
      nEles, AdvDiff_D);
  }

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
    mu = mu_in * pow(rt_ratio,1.5) * (1. + c_sth) / (rt_ratio + c_sth);
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

  __global__
void compute_Fvisc_spts_3D_EulerNS(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, mdvector_gpu<double> dU_spts, 
    unsigned int nSpts, unsigned int nEles, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* States */
  double rho = U_spts(spt, ele, 0);
  double momx = U_spts(spt, ele, 1);
  double momy = U_spts(spt, ele, 2);
  double momz = U_spts(spt, ele, 3);
  double e = U_spts(spt, ele, 4);

  double u = momx / rho;
  double v = momy / rho;
  double w = momz / rho;
  double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

  /* Gradients */
  double rho_dx = dU_spts(spt, ele, 0, 0);
  double momx_dx = dU_spts(spt, ele, 1, 0);
  double momy_dx = dU_spts(spt, ele, 2, 0);
  double momz_dx = dU_spts(spt, ele, 3, 0);
  double e_dx = dU_spts(spt, ele, 4, 0);
  
  double rho_dy = dU_spts(spt, ele, 0, 1);
  double momx_dy = dU_spts(spt, ele, 1, 1);
  double momy_dy = dU_spts(spt, ele, 2, 1);
  double momz_dy = dU_spts(spt, ele, 3, 1);
  double e_dy = dU_spts(spt, ele, 4, 1);

  double rho_dz = dU_spts(spt, ele, 0, 2);
  double momx_dz = dU_spts(spt, ele, 1, 2);
  double momy_dz = dU_spts(spt, ele, 2, 2);
  double momz_dz = dU_spts(spt, ele, 3, 2);
  double e_dz = dU_spts(spt, ele, 4, 2);

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
  double du_dz = (momx_dz - rho_dz * u) / rho;

  double dv_dx = (momy_dx - rho_dx * v) / rho;
  double dv_dy = (momy_dy - rho_dy * v) / rho;
  double dv_dz = (momy_dz - rho_dz * v) / rho;

  double dw_dx = (momz_dx - rho_dx * w) / rho;
  double dw_dy = (momz_dy - rho_dy * w) / rho;
  double dw_dz = (momz_dz - rho_dz * w) / rho;

  double dke_dx = 0.5 * (u*u + v*v + w*w) * rho_dx + rho * (u * du_dx + v * dv_dx + w * dw_dx);
  double dke_dy = 0.5 * (u*u + v*v + w*w) * rho_dy + rho * (u * du_dy + v * dv_dy + w * dw_dy);
  double dke_dz = 0.5 * (u*u + v*v + w*w) * rho_dz + rho * (u * du_dz + v * dv_dz + w * dw_dz);

  double de_dx = (e_dx - dke_dx - rho_dx * e_int) / rho;
  double de_dy = (e_dy - dke_dy - rho_dy * e_int) / rho;
  double de_dz = (e_dz - dke_dz - rho_dz * e_int) / rho;

  double diag = (du_dx + dv_dy + dw_dz) / 3.0;

  double tauxx = 2.0 * mu * (du_dx - diag);
  double tauyy = 2.0 * mu * (dv_dy - diag);
  double tauzz = 2.0 * mu * (dw_dz - diag);
  double tauxy = mu * (du_dy + dv_dx);
  double tauxz = mu * (du_dz + dw_dx);
  double tauyz = mu * (dv_dz + dw_dy);

  /* Set viscous flux values */
  F_spts(spt, ele, 1, 0) -= tauxx;
  F_spts(spt, ele, 2, 0) -= tauxy;
  F_spts(spt, ele, 3, 0) -= tauxz;
  F_spts(spt, ele, 4, 0) -= (u * tauxx + v * tauxy + w * tauxz + (mu / prandtl) *
      gamma * de_dx);

  F_spts(spt, ele, 1, 1) -= tauxy;
  F_spts(spt, ele, 2, 1) -= tauyy;
  F_spts(spt, ele, 3, 1) -= tauyz;
  F_spts(spt, ele, 4, 1) -= (u * tauxy + v * tauyy + w * tauyz + (mu / prandtl) *
      gamma * de_dy);

  F_spts(spt, ele, 1, 2) -= tauxz;
  F_spts(spt, ele, 2, 2) -= tauyz;
  F_spts(spt, ele, 3, 2) -= tauzz;
  F_spts(spt, ele, 4, 2) -= (u * tauxz + v * tauyz + w * tauzz + (mu / prandtl) *
      gamma * de_dz);

}


void compute_Fvisc_spts_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  dim3 threads(16, 12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_Fvisc_spts_2D_EulerNS<<<blocks, threads>>>(F_spts, 
      U_spts, dU_spts, nSpts, nEles, gamma, prandtl, mu_in, c_sth, rt, fix_vis);
  }
  else if (nDims == 3)
  {
    compute_Fvisc_spts_3D_EulerNS<<<blocks, threads>>>(F_spts, 
      U_spts, dU_spts, nSpts, nEles, gamma, prandtl, mu_in, c_sth, rt, fix_vis);
  }

}

template <unsigned int nDims>
__global__
void compute_dFdUconv_spts_AdvDiff(mdvector_gpu<double> dFdU_spts, 
    unsigned int nSpts, unsigned int nEles, mdvector_gpu<double> AdvDiff_A)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dFdU_spts(spt, ele, 0, 0, dim) = AdvDiff_A(dim);
  }
}

void compute_dFdUconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &dFdU_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, mdvector_gpu<double> &AdvDiff_A)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_dFdUconv_spts_AdvDiff<2><<<blocks, threads>>>(dFdU_spts, nSpts, nEles, AdvDiff_A);
  }
  else
  {
    compute_dFdUconv_spts_AdvDiff<3><<<blocks, threads>>>(dFdU_spts, nSpts, nEles, AdvDiff_A);
  }
}

template <unsigned int nDims>
__global__
void compute_dFdUconv_spts_Burgers(mdvector_gpu<double> dFdU_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts, unsigned int nEles)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dFdU_spts(spt, ele, 0, 0, dim) = U_spts(spt, ele, 0);
  }
}

void compute_dFdUconv_spts_Burgers_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_dFdUconv_spts_Burgers<2><<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles);
  }
  else
  {
    compute_dFdUconv_spts_Burgers<3><<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles);
  }
}

__global__
void compute_dFdUconv_spts_2D_EulerNS(mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> U_spts, 
    unsigned int nSpts, unsigned int nEles, double gam)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Primitive Variables */
  double rho = U_spts(spt, ele, 0);
  double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
  double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
  double e = U_spts(spt, ele, 3);

  /* Set convective dFdU values in the x-direction */
  dFdU_spts(spt, ele, 0, 0, 0) = 0;
  dFdU_spts(spt, ele, 1, 0, 0) = 0.5 * ((gam-3.0) * u*u + (gam-1.0) * v*v);
  dFdU_spts(spt, ele, 2, 0, 0) = -u * v;
  dFdU_spts(spt, ele, 3, 0, 0) = -gam * e * u / rho + (gam-1.0) * u * (u*u + v*v);

  dFdU_spts(spt, ele, 0, 1, 0) = 1;
  dFdU_spts(spt, ele, 1, 1, 0) = (3.0-gam) * u;
  dFdU_spts(spt, ele, 2, 1, 0) = v;
  dFdU_spts(spt, ele, 3, 1, 0) = gam * e / rho + 0.5 * (1.0-gam) * (3.0*u*u + v*v);

  dFdU_spts(spt, ele, 0, 2, 0) = 0;
  dFdU_spts(spt, ele, 1, 2, 0) = (1.0-gam) * v;
  dFdU_spts(spt, ele, 2, 2, 0) = u;
  dFdU_spts(spt, ele, 3, 2, 0) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 3, 0) = 0;
  dFdU_spts(spt, ele, 1, 3, 0) = (gam-1.0);
  dFdU_spts(spt, ele, 2, 3, 0) = 0;
  dFdU_spts(spt, ele, 3, 3, 0) = gam * u;

  /* Set convective dFdU values in the y-direction */
  dFdU_spts(spt, ele, 0, 0, 1) = 0;
  dFdU_spts(spt, ele, 1, 0, 1) = -u * v;
  dFdU_spts(spt, ele, 2, 0, 1) = 0.5 * ((gam-1.0) * u*u + (gam-3.0) * v*v);
  dFdU_spts(spt, ele, 3, 0, 1) = -gam * e * v / rho + (gam-1.0) * v * (u*u + v*v);

  dFdU_spts(spt, ele, 0, 1, 1) = 0;
  dFdU_spts(spt, ele, 1, 1, 1) = v;
  dFdU_spts(spt, ele, 2, 1, 1) = (1.0-gam) * u;
  dFdU_spts(spt, ele, 3, 1, 1) = (1.0-gam) * u * v;

  dFdU_spts(spt, ele, 0, 2, 1) = 1;
  dFdU_spts(spt, ele, 1, 2, 1) = u;
  dFdU_spts(spt, ele, 2, 2, 1) = (3.0-gam) * v;
  dFdU_spts(spt, ele, 3, 2, 1) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + 3.0*v*v);

  dFdU_spts(spt, ele, 0, 3, 1) = 0;
  dFdU_spts(spt, ele, 1, 3, 1) = 0;
  dFdU_spts(spt, ele, 2, 3, 1) = (gam-1.0);
  dFdU_spts(spt, ele, 3, 3, 1) = gam * v;
}

void compute_dFdUconv_spts_EulerNS_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1) / 
      threads.y);

  if (nDims == 2)
  {
    compute_dFdUconv_spts_2D_EulerNS<<<blocks, threads>>>(dFdU_spts, U_spts, nSpts, 
      nEles, gamma);
  }
  else
  {
    ThrowException("compute_dFdUconv for 3D EulerNS not implemented yet!");
  }
}

__global__
void add_scaled_oppD(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD, 
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims)
{
  const unsigned int ni = blockIdx.x;
  const unsigned int nj = blockIdx.y % nVars;
  const unsigned int ele = blockIdx.y / nVars;
  const unsigned int i = threadIdx.x;
  const unsigned int j = threadIdx.y;

  if (ele >= nEles || ni >= nVars || nj >= nVars)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    LHS(i, ni, j, nj, ele) += oppD(i, j, dim) * C(j, ele, ni, nj, dim);
  }
}

void add_scaled_oppD_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims)
{
  dim3 threads(nSpts, nSpts);
  dim3 blocks((nSpts * nVars + threads.x - 1) / threads.x, 
      (nSpts * nVars * nEles + threads.y - 1) / threads.y);

  add_scaled_oppD<<<blocks, threads>>>(LHS, oppD, C, nSpts, nVars, nEles, nDims);
}

__global__
void add_scaled_oppDiv(mdvector_gpu<double> LHS_tempSF, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles)
{
  const unsigned int ni = blockIdx.x;
  const unsigned int nj = blockIdx.y % nVars;
  const unsigned int ele = blockIdx.y / nVars;
  const unsigned int i = threadIdx.x;
  const unsigned int j = threadIdx.y;

  if (ele >= nEles || ni >= nVars || nj >= nVars)
    return;

  LHS_tempSF(i, ni, j, nj, ele) = oppDiv_fpts(i, j) * C(j, ele, ni, nj, 0);
}

void add_scaled_oppDiv_wrapper(mdvector_gpu<double> &LHS_tempSF, mdvector_gpu<double> &oppDiv_fpts, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles)
{
  dim3 threads(nSpts, nFpts);
  dim3 blocks((nSpts * nVars + threads.x - 1) / threads.x, 
      (nFpts * nVars * nEles + threads.y - 1) / threads.y);

  add_scaled_oppDiv<<<blocks, threads>>>(LHS_tempSF, oppDiv_fpts, C, nSpts, nFpts, nVars, nEles);
}

__global__
void finalize_LHS(mdvector_gpu<double> LHS, mdvector_gpu<double> dt, 
    mdvector_gpu<double> jaco_det_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int dt_type)
{
  const unsigned int ni = blockIdx.x;
  const unsigned int nj = blockIdx.y % nVars;
  const unsigned int ele = blockIdx.y / nVars;
  const unsigned int i = threadIdx.x;
  const unsigned int j = threadIdx.y;

  if (ele >= nEles || ni >= nVars || nj >= nVars)
    return;

  double add_one = (double) (i == j && nj == ni);

  if (dt_type != 2)
  {
    LHS(i, ni, j, nj, ele) = dt(0) * LHS(i, ni, j, nj, ele) / jaco_det_spts(i, ele) + add_one;
  }
  else
  {
    LHS(i, ni, j, nj, ele) = dt(ele) * LHS(i, ni, j, nj, ele) / jaco_det_spts(i, ele) + add_one;
  }
}

void finalize_LHS_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int dt_type)
{
  dim3 threads(nSpts, nSpts);
  dim3 blocks((nSpts * nVars + threads.x - 1) / threads.x, 
      (nSpts * nVars * nEles + threads.y - 1) / threads.y);

  finalize_LHS<<<blocks, threads>>>(LHS, dt, jaco_det_spts, nSpts, nVars, nEles, dt_type);
}


template <unsigned int nVars>
__global__
void transform_dU_quad(mdvector_gpu<double> dU_spts, 
    mdvector_gpu<double> jaco_spts, mdvector_gpu<double> jaco_det_spts,
    unsigned int nSpts, unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  double jaco[2][2];
  jaco[0][0] = jaco_spts(0, 0, spt, ele);
  jaco[0][1] = jaco_spts(0, 1, spt, ele);
  jaco[1][0] = jaco_spts(1, 0, spt, ele);
  jaco[1][1] = jaco_spts(1, 1, spt, ele);
  double jaco_det = jaco_det_spts(spt,ele);

  for (unsigned int var = 0; var < nVars; var++)
  {
    double dUtemp = dU_spts(spt, ele, var, 0);

    dU_spts(spt, ele, var, 0) = (dU_spts(spt, ele, var, 0) * jaco[1][1] - 
                                dU_spts(spt, ele, var, 1) * jaco[1][0]) /
                                jaco_det; 

    dU_spts(spt, ele, var, 1) = (dU_spts(spt, ele, var, 1) * jaco[0][0] -
                                 dUtemp * jaco[0][1]) / jaco_det;

  }

}

void transform_dU_quad_wrapper(mdvector_gpu<double> &dU_spts, 
    mdvector_gpu<double> &jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_dU_quad<1><<<blocks, threads>>>(dU_spts, jaco_spts, jaco_det_spts,
        nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_dU_quad<4><<<blocks, threads>>>(dU_spts, jaco_spts, jaco_det_spts,
        nSpts, nEles);
  }
}

template <unsigned int nVars>
__global__
void transform_dU_hexa(mdvector_gpu<double> dU_spts, 
    mdvector_gpu<double> inv_jaco_spts, mdvector_gpu<double> jaco_det_spts,
    unsigned int nSpts, unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  double inv_jaco[3][3];
  inv_jaco[0][0] = inv_jaco_spts(0, 0, spt, ele);
  inv_jaco[0][1] = inv_jaco_spts(0, 1, spt, ele);
  inv_jaco[0][2] = inv_jaco_spts(0, 2, spt, ele);
  inv_jaco[1][0] = inv_jaco_spts(1, 0, spt, ele);
  inv_jaco[1][1] = inv_jaco_spts(1, 1, spt, ele);
  inv_jaco[1][2] = inv_jaco_spts(1, 2, spt, ele);
  inv_jaco[2][0] = inv_jaco_spts(2, 0, spt, ele);
  inv_jaco[2][1] = inv_jaco_spts(2, 1, spt, ele);
  inv_jaco[2][2] = inv_jaco_spts(2, 2, spt, ele);
  double jaco_det = jaco_det_spts(spt,ele);

  for (unsigned int n = 0; n < nVars; n++)
  {
    double dUtemp0 = dU_spts(spt, ele, n, 0);
    double dUtemp1 = dU_spts(spt, ele, n, 1);

    dU_spts(spt, ele, n, 0) = dU_spts(spt, ele, n, 0) * inv_jaco[0][0] + 
                              dU_spts(spt, ele, n, 1) * inv_jaco[1][0] +  
                              dU_spts(spt, ele, n, 2) * inv_jaco[2][0];  

    dU_spts(spt, ele, n, 1) = dUtemp0 * inv_jaco[0][1] + 
                              dU_spts(spt, ele, n, 1) * inv_jaco[1][1] +  
                              dU_spts(spt, ele, n, 2) * inv_jaco[2][1];  
                              
    dU_spts(spt, ele, n, 2) = dUtemp0 * inv_jaco[0][2] + 
                              dUtemp1 * inv_jaco[1][2] +  
                              dU_spts(spt, ele, n, 2) * inv_jaco[2][2];  

    dU_spts(spt, ele, n, 0) /= jaco_det;
    dU_spts(spt, ele, n, 1) /= jaco_det;
    dU_spts(spt, ele, n, 2) /= jaco_det;
  }

}

void transform_dU_hexa_wrapper(mdvector_gpu<double> &dU_spts, 
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_dU_hexa<1><<<blocks, threads>>>(dU_spts, inv_jaco_spts, jaco_det_spts,
        nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_dU_hexa<5><<<blocks, threads>>>(dU_spts, inv_jaco_spts, jaco_det_spts,
        nSpts, nEles);
  }
}

template <unsigned int nVars>
__global__
void transform_flux_quad(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get metric terms */
  double jaco[2][2];
  jaco[0][0] = jaco_spts(0, 0, spt, ele);
  jaco[0][1] = jaco_spts(0, 1, spt, ele);
  jaco[1][0] = jaco_spts(1, 0, spt, ele);
  jaco[1][1] = jaco_spts(1, 1, spt, ele);

  for (unsigned int var = 0; var < nVars; var ++)
  {
    double Ftemp = F_spts(spt, ele, var, 0);

    F_spts(spt, ele, var, 0) = F_spts(spt, ele, var, 0) * jaco[1][1] -
                             F_spts(spt, ele, var, 1) * jaco[0][1];
    F_spts(spt, ele, var, 1) = F_spts(spt, ele, var, 1) * jaco[0][0] -
                             Ftemp * jaco[1][0];
  }

}

void transform_flux_quad_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_flux_quad<1><<<blocks, threads>>>(F_spts, jaco_spts, nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_flux_quad<4><<<blocks, threads>>>(F_spts, jaco_spts, nSpts, nEles);
  }
}

template <unsigned int nVars>
__global__
void transform_flux_hexa(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get metric terms */
  double inv_jaco[3][3];
  inv_jaco[0][0] = inv_jaco_spts(0, 0, spt, ele);
  inv_jaco[0][1] = inv_jaco_spts(0, 1, spt, ele);
  inv_jaco[0][2] = inv_jaco_spts(0, 2, spt, ele);
  inv_jaco[1][0] = inv_jaco_spts(1, 0, spt, ele);
  inv_jaco[1][1] = inv_jaco_spts(1, 1, spt, ele);
  inv_jaco[1][2] = inv_jaco_spts(1, 2, spt, ele);
  inv_jaco[2][0] = inv_jaco_spts(2, 0, spt, ele);
  inv_jaco[2][1] = inv_jaco_spts(2, 1, spt, ele);
  inv_jaco[2][2] = inv_jaco_spts(2, 2, spt, ele);

  for (unsigned int n = 0; n < nVars; n++)
  {
    double Ftemp0 = F_spts(spt, ele, n, 0);
    double Ftemp1 = F_spts(spt, ele, n, 1);

    F_spts(spt, ele, n, 0) = F_spts(spt, ele, n, 0) * inv_jaco[0][0] +
                             F_spts(spt, ele, n, 1) * inv_jaco[0][1] +
                             F_spts(spt, ele, n, 2) * inv_jaco[0][2];

    F_spts(spt, ele, n, 1) = Ftemp0 * inv_jaco[1][0] +
                             F_spts(spt, ele, n, 1) * inv_jaco[1][1] +
                             F_spts(spt, ele, n, 2) * inv_jaco[1][2];  
                              
    F_spts(spt, ele, n, 2) = Ftemp0 * inv_jaco[2][0]+ 
                             Ftemp1 * inv_jaco[2][1] +  
                             F_spts(spt, ele, n, 2) * inv_jaco[2][2]; 

  }

}

void transform_flux_hexa_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_flux_hexa<1><<<blocks, threads>>>(F_spts, inv_jaco_spts, nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_flux_hexa<5><<<blocks, threads>>>(F_spts, inv_jaco_spts, nSpts, nEles);
  }
}

template <unsigned int nVars>
__global__
void transform_dFdU_quad(mdvector_gpu<double> dFdU_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  /* Get metric terms */
  double jaco[2][2];
  jaco[0][0] = jaco_spts(0, 0, spt, ele);
  jaco[0][1] = jaco_spts(0, 1, spt, ele);
  jaco[1][0] = jaco_spts(1, 0, spt, ele);
  jaco[1][1] = jaco_spts(1, 1, spt, ele);

  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      double dFdUtemp = dFdU_spts(spt, ele, ni, nj, 0);

      dFdU_spts(spt, ele, ni, nj, 0) = dFdU_spts(spt, ele, ni, nj, 0) * jaco[1][1] -
                                       dFdU_spts(spt, ele, ni, nj, 1) * jaco[0][1];
      dFdU_spts(spt, ele, ni, nj, 1) = dFdU_spts(spt, ele, ni, nj, 1) * jaco[0][0] -
                                       dFdUtemp * jaco[1][0];
    }
  }
}

void transform_dFdU_quad_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    transform_dFdU_quad<1><<<blocks, threads>>>(dFdU_spts, jaco_spts, nSpts, nEles);
  }
  else if (equation == EulerNS)
  {
    transform_dFdU_quad<4><<<blocks, threads>>>(dFdU_spts, jaco_spts, nSpts, nEles);
  }
}

__global__
void compute_Uavg(mdvector_gpu<double> U_spts, 
    mdvector_gpu<double> Uavg, mdvector_gpu<double> jaco_det_spts, 
    mdvector_gpu<double> weights_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, int order)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  /* Compute average solution using quadrature */
  for (unsigned int n = 0; n < nVars; n++)
  {
    double sum = 0.0;
    double vol = 0.0;

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get quadrature weight */
      unsigned int i = spt % (order + 1);
      unsigned int j = spt / (order + 1);
      double weight = weights_spts(i) * weights_spts(j);

      sum += weight * jaco_det_spts(spt, ele) * U_spts(spt, ele, n);
      vol += weight * jaco_det_spts(spt, ele);
    }

    Uavg(ele, n) = sum / vol; 

  }

}

void compute_Uavg_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &Uavg, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &weights_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, int order)
{
  unsigned int threads= 192;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  compute_Uavg<<<blocks, threads>>>(U_spts, Uavg, jaco_det_spts, weights_spts, nSpts, nEles, nVars, order);
}

__global__
void poly_squeeze(mdvector_gpu<double> U_spts, 
    mdvector_gpu<double> U_fpts, mdvector_gpu<double> Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  double V[3]; 

  /* For each element, check for negative density at solution and flux points */
  double tol = 1e-10;

  bool negRho = false;
  double minRho = U_spts(0, ele, 0);

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    if (U_spts(spt, ele, 0) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_spts(spt, ele, 0));
    }
  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    if (U_fpts(fpt, ele, 0) < 0)
    {
      negRho = true;
      minRho = min(minRho, U_fpts(fpt, ele, 0));
    }
  }

  /* If negative density found, squeeze density */
  if (negRho)
  {
    double theta = (Uavg(ele, 0) - tol) / (Uavg(ele , 0) - minRho); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
      U_spts(spt, ele, 0) = theta * U_spts(spt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);

    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      U_fpts(fpt, ele, 0) = theta * U_fpts(fpt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);
    
  }

  /* For each element, check for entropy loss */
  double minTau = 1.0;

  /* Get minimum tau value */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double rho = U_spts(spt, ele, 0);
    double momF = (U_spts(spt, ele, 1) * U_spts(spt,ele,1) + U_spts(spt, ele, 2) * 
        U_spts(spt, ele,2)) / U_spts(spt, ele, 0);
    double P = (gamma - 1.0) * (U_spts(spt, ele, 3) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }
  
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    double rho = U_fpts(fpt, ele, 0);
    double momF = (U_fpts(fpt, ele, 1) * U_fpts(fpt,ele,1) + U_spts(fpt, ele, 2) * 
        U_fpts(fpt, ele,2)) / U_fpts(fpt, ele, 0);
    double P = (gamma - 1.0) * (U_fpts(fpt, ele, 3) - 0.5 * momF);

    double tau = P - exps0 * pow(rho, gamma);
    minTau = min(minTau, tau);

  }

  /* If minTau is negative, squeeze solution */
  if (minTau < 0)
  {
    double rho = Uavg(ele, 0);
    double Vsq = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      V[dim] = Uavg(ele, dim+1) / rho;
      Vsq += V[dim] * V[dim];
    }

    double e = Uavg(ele, 3);
    double P = (gamma - 1.0) * (e - 0.5 * rho * Vsq);

    double eps = minTau / (minTau - P + exps0 * pow(rho, gamma));

//      if (P < input->exps0 * std::pow(rho, input->gamma))
//        std::cout << "Constraint violated. Lower CFL?" << std::endl;

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        U_spts(spt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_spts(spt, ele, n);
      }
    }

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        U_fpts(fpt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_fpts(fpt, ele, n);
      }
    }
  }

}

void poly_squeeze_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims)
{
  unsigned int threads= 192;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  poly_squeeze<<<blocks, threads>>>(U_spts, U_fpts, Uavg, gamma, exps0, nSpts, nFpts,
      nEles, nVars, nDims);
}
