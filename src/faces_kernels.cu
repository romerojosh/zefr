#include "input.hpp"
#include "faces_kernels.h"
#include "mdvector_gpu.h"

template <unsigned int nDims>
__global__
void compute_Fconv_fpts_AdvDiff(mdvector_gpu<double> F, mdvector_gpu<double> U, 
    unsigned int nFpts, mdvector_gpu<double> AdvDiff_A, 
    unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
      F(fpt, 0, dim, 0) = AdvDiff_A(dim) * U(fpt, 0, 0);

      F(fpt, 0, dim, 1) = AdvDiff_A(dim) * U(fpt, 0, 1);
  }

}

void compute_Fconv_fpts_AdvDiff_wrapper(mdvector_gpu<double> &F, 
    mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nDims, 
    mdvector_gpu<double> &AdvDiff_A, unsigned int startFpt,
    unsigned int endFpt)
{
  unsigned int threads = 192;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (nDims == 2)
    compute_Fconv_fpts_AdvDiff<2><<<blocks, threads>>>(F, U, nFpts, AdvDiff_A, 
        startFpt, endFpt);
  else 
    compute_Fconv_fpts_AdvDiff<3><<<blocks, threads>>>(F, U, nFpts, AdvDiff_A,
        startFpt, endFpt);

}

template <unsigned int nDims>
__global__
void compute_Fconv_fpts_Burgers(mdvector_gpu<double> F, mdvector_gpu<double> U, 
    unsigned int nFpts, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
      F(fpt, 0, dim, 0) = 0.5 * U(fpt, 0, 0) * U(fpt, 0, 0);

      F(fpt, 0, dim, 1) = 0.5 * U(fpt, 0, 1) * U(fpt, 0, 1);
  }

}

void compute_Fconv_fpts_Burgers_wrapper(mdvector_gpu<double> &F, 
    mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nDims, 
    unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 192;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (nDims == 2)
    compute_Fconv_fpts_Burgers<2><<<blocks, threads>>>(F, U, nFpts, startFpt, endFpt);
  else 
    compute_Fconv_fpts_Burgers<3><<<blocks, threads>>>(F, U, nFpts, startFpt, endFpt);

}

__global__
void compute_Fconv_fpts_2D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, mdvector_gpu<double> P, 
    unsigned int nFpts, double gamma, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

   for (unsigned int slot = 0; slot < 2; slot ++)
   {
     /* Get states */
     double rho = U(fpt, 0, slot);
     double momx = U(fpt, 1, slot);
     double momy = U(fpt, 2, slot);
     double ene = U(fpt, 3, slot);

     /* Compute some primitive variables (keep pressure)*/
     double momF = (momx * momx + momy * momy) / rho;

     double P_d = (gamma - 1.0) * (ene - 0.5 * momF);
     P(fpt, slot) = P_d;

     double H = (ene + P_d) / rho;

     F(fpt, 0, 0, slot) = momx;
     F(fpt, 1, 0, slot) = momx * momx / rho + P_d;
     F(fpt, 2, 0, slot) = momx * momy / rho;
     F(fpt, 3, 0, slot) = momx * H;

     F(fpt, 0, 1, slot) = momy;
     F(fpt, 1, 1, slot) = momy * momx / rho;
     F(fpt, 2, 1, slot) = momy * momy / rho + P_d;
     F(fpt, 3, 1, slot) = momy * H;
   }
}

__global__
void compute_Fconv_fpts_3D_EulerNS(mdvector_gpu<double> F, mdvector_gpu<double> U, mdvector_gpu<double> P, 
    unsigned int nFpts, double gamma, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

   for (unsigned int slot = 0; slot < 2; slot ++)
   {
     /* Get states */
     double rho = U(fpt, 0, slot);
     double momx = U(fpt, 1, slot);
     double momy = U(fpt, 2, slot);
     double momz = U(fpt, 3, slot);
     double ene = U(fpt, 4, slot);

     /* Compute some primitive variables (keep pressure)*/
     double momF = (momx * momx + momy * momy + momz * momz) / rho;

     double P_d = (gamma - 1.0) * (ene - 0.5 * momF);
     P(fpt, slot) = P_d;

     double H = (ene + P_d) / rho;

     F(fpt, 0, 0, slot) = momx;
     F(fpt, 1, 0, slot) = momx * momx / rho + P_d;
     F(fpt, 2, 0, slot) = momx * momy / rho;
     F(fpt, 3, 0, slot) = momx * momz / rho;
     F(fpt, 4, 0, slot) = momx * H;

     F(fpt, 0, 1, slot) = momy;
     F(fpt, 1, 1, slot) = momy * momx / rho;
     F(fpt, 2, 1, slot) = momy * momy / rho + P_d;
     F(fpt, 3, 1, slot) = momy * momz / rho;
     F(fpt, 4, 1, slot) = momy * H;

     F(fpt, 0, 2, slot) = momz;
     F(fpt, 1, 2, slot) = momz * momx / rho;
     F(fpt, 2, 2, slot) = momz * momy / rho;
     F(fpt, 3, 2, slot) = momz * momz / rho + P_d;
     F(fpt, 4, 2, slot) = momz * H;
   }
}

void compute_Fconv_fpts_EulerNS_wrapper(mdvector_gpu<double> &F_gfpts, 
    mdvector_gpu<double> &U_gfpts, mdvector_gpu<double> &P_gfpts, 
    unsigned int nFpts, unsigned int nDims, double gamma,
    unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 192;
  //unsigned int blocks = (nFpts + threads - 1)/threads;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (nDims == 2)
  {
    compute_Fconv_fpts_2D_EulerNS<<<blocks, threads>>>(F_gfpts, U_gfpts, P_gfpts, nFpts, gamma,
        startFpt, endFpt);
  }
  else 
  {
    compute_Fconv_fpts_3D_EulerNS<<<blocks, threads>>>(F_gfpts, U_gfpts, P_gfpts, nFpts, gamma, 
        startFpt, endFpt);
  }
}

template <unsigned int nDims>
__global__
void compute_Fvisc_fpts_AdvDiff(mdvector_gpu<double> Fvisc, mdvector_gpu<double> dU, 
    unsigned int nFpts, double AdvDiff_D, unsigned int startFpt,
    unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
      Fvisc(fpt, 0, dim, 0) = -AdvDiff_D * dU(fpt, 0, dim, 0);

      Fvisc(fpt, 0, dim, 1) = -AdvDiff_D * dU(fpt, 0, dim, 1);
  }
}

void compute_Fvisc_fpts_AdvDiff_wrapper(mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &dU, unsigned int nFpts, unsigned int nDims, 
    double AdvDiff_D, unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 192;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (nDims == 2)
    compute_Fvisc_fpts_AdvDiff<2><<<blocks, threads>>>(Fvisc, dU, nFpts, AdvDiff_D,
        startFpt, endFpt);
  else 
    compute_Fvisc_fpts_AdvDiff<3><<<blocks, threads>>>(Fvisc, dU, nFpts, AdvDiff_D,
        startFpt, endFpt);
}


__global__
void compute_Fvisc_fpts_2D_EulerNS(mdvector_gpu<double> Fvisc, mdvector_gpu<double> U, 
    mdvector_gpu<double> dU, unsigned int nFpts, double gamma, double prandtl, 
    double mu_in, double c_sth, double rt, bool fix_vis, unsigned int startFpt,
    unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  for (unsigned int slot = 0; slot < 2; slot++)
  {
    /* Setting variables for convenience */
    /* States */
    double rho = U(fpt, 0, slot);
    double momx = U(fpt, 1, slot);
    double momy = U(fpt, 2, slot);
    double e = U(fpt, 3, slot);

    double u = momx / rho;
    double v = momy / rho;
    double e_int = e / rho - 0.5 * (u*u + v*v);

    /* Gradients */
    double rho_dx = dU(fpt, 0, 0, slot);
    double momx_dx = dU(fpt, 1, 0, slot);
    double momy_dx = dU(fpt, 2, 0, slot);
    double e_dx = dU(fpt, 3, 0, slot);
    
    double rho_dy = dU(fpt, 0, 1, slot);
    double momx_dy = dU(fpt, 1, 1, slot);
    double momy_dy = dU(fpt, 2, 1, slot);
    double e_dy = dU(fpt, 3, 1, slot);

    /* Set viscosity */
    double mu;
    if (fix_vis)
    {
      mu = mu_in;
    }
    /* If desired, use Sutherland's law */
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
    Fvisc(fpt, 0, 0, slot) = 0.0;
    Fvisc(fpt, 1, 0, slot) = -tauxx;
    Fvisc(fpt, 2, 0, slot) = -tauxy;
    Fvisc(fpt, 3, 0, slot) = -(u * tauxx + v * tauxy + (mu / prandtl) *
        gamma * de_dx);

    Fvisc(fpt, 0, 1, slot) = 0.0;
    Fvisc(fpt, 1, 1, slot) = -tauxy;
    Fvisc(fpt, 2, 1, slot) = -tauyy;
    Fvisc(fpt, 3, 1, slot) = -(u * tauxy + v * tauyy + (mu / prandtl) *
        gamma * de_dy);
  }

}

__global__
void compute_Fvisc_fpts_3D_EulerNS(mdvector_gpu<double> Fvisc, mdvector_gpu<double> U, 
    mdvector_gpu<double> dU, unsigned int nFpts, double gamma, double prandtl, 
    double mu_in, double c_sth, double rt, bool fix_vis, unsigned int startFpt,
    unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  for (unsigned int slot = 0; slot < 2; slot++)
  {
    /* States */
    double rho = U(fpt, 0, slot);
    double momx = U(fpt, 1, slot);
    double momy = U(fpt, 2, slot);
    double momz = U(fpt, 3, slot);
    double e = U(fpt, 4, slot);

    double u = momx / rho;
    double v = momy / rho;
    double w = momz / rho;
    double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

    /* Gradients */
    double rho_dx = dU(fpt, 0, 0, slot);
    double momx_dx = dU(fpt, 1, 0, slot);
    double momy_dx = dU(fpt, 2, 0, slot);
    double momz_dx = dU(fpt, 3, 0, slot);
    double e_dx = dU(fpt, 4, 0, slot);

    double rho_dy = dU(fpt, 0, 1, slot);
    double momx_dy = dU(fpt, 1, 1, slot);
    double momy_dy = dU(fpt, 2, 1, slot);
    double momz_dy = dU(fpt, 3, 1, slot);
    double e_dy = dU(fpt, 4, 1, slot);

    double rho_dz = dU(fpt, 0, 2, slot);
    double momx_dz = dU(fpt, 1, 2, slot);
    double momy_dz = dU(fpt, 2, 2, slot);
    double momz_dz = dU(fpt, 3, 2, slot);
    double e_dz = dU(fpt, 4, 2, slot);

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
    Fvisc(fpt, 0, 0, slot) = 0;
    Fvisc(fpt, 1, 0, slot) = -tauxx;
    Fvisc(fpt, 2, 0, slot) = -tauxy;
    Fvisc(fpt, 3, 0, slot) = -tauxz;
    Fvisc(fpt, 4, 0, slot) = -(u * tauxx + v * tauxy + w * tauxz + (mu / prandtl) *
      gamma * de_dx);

    Fvisc(fpt, 0, 1, slot) = 0;
    Fvisc(fpt, 1, 1, slot) = -tauxy;
    Fvisc(fpt, 2, 1, slot) = -tauyy;
    Fvisc(fpt, 3, 1, slot) = -tauyz;
    Fvisc(fpt, 4, 1, slot) = -(u * tauxy + v * tauyy + w * tauyz + (mu / prandtl) *
      gamma * de_dy);

    Fvisc(fpt, 0, 2, slot) = 0;
    Fvisc(fpt, 1, 2, slot) = -tauxz;
    Fvisc(fpt, 2, 2, slot) = -tauyz;
    Fvisc(fpt, 3, 2, slot) = -tauzz;
    Fvisc(fpt, 4, 2, slot) = -(u * tauxz + v * tauyz + w * tauzz + (mu / prandtl) *
      gamma * de_dz);
  }

}

void compute_Fvisc_fpts_EulerNS_wrapper(mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &U, mdvector_gpu<double> &dU, unsigned int nFpts, unsigned int nDims, 
    double gamma, double prandtl, double mu_in, double c_sth, double rt, bool fix_vis,
    unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 192;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (nDims == 2)
  {
    compute_Fvisc_fpts_2D_EulerNS<<<blocks, threads>>>(Fvisc, U, dU, nFpts, gamma, 
        prandtl, mu_in, c_sth, rt, fix_vis, startFpt, endFpt);
  }
  else
  {
    compute_Fvisc_fpts_3D_EulerNS<<<blocks, threads>>>(Fvisc, U, dU, nFpts, gamma, 
        prandtl, mu_in, c_sth, rt, fix_vis, startFpt, endFpt);
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void apply_bcs(mdvector_gpu<double> U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, double rho_fs, 
    mdvector_gpu<double> V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, mdvector_gpu<double> V_wall, mdvector_gpu<double> norm_fs, 
    mdvector_gpu<double> norm, mdvector_gpu<unsigned int> gfpt2bnd, 
    mdvector_gpu<unsigned int> per_fpt_list, mdvector_gpu<int> LDG_bias, mdvector_gpu<int> bc_bias)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  switch(bnd_id)
  {
    case 1:/* Periodic */
    {
      unsigned int per_fpt = per_fpt_list(fpt - nGfpts_int);

      for (unsigned int n = 0; n < nVars; n++)
      {
        U(fpt, n, 1) = U(per_fpt, n, 0);
      }
      break;
    }
  
    case 2: /* Farfield and Supersonic Inlet */
    {
      if (equation == AdvDiff || equation == Burgers)
      {
        /* Set boundaries to zero */
        U(fpt, 0, 1) = 0;
      }
      else
      {
        /* Set boundaries to freestream values */
        U(fpt, 0, 1) = rho_fs;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(fpt, dim+1, 1) = rho_fs * V_fs(dim);
          Vsq += V_fs(dim) * V_fs(dim);
        }

        U(fpt, nDims + 1, 1) = P_fs/(gamma-1.0) + 0.5*rho_fs * Vsq; 
      }

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;
      bc_bias(fpt) = 1;

      break;
    }

    case 3: /* Supersonic Outlet */
    {
      /* Extrapolate boundary values from interior */
      for (unsigned int n = 0; n < nVars; n++)
        U(fpt, n, 1) = U(fpt, n, 0);

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;

      break;
    }

    case 4: /* Subsonic Inlet */
    {
      double VL[3]; double VR[3];
      /*
      if (!input->viscous)
        ThrowException("Subsonic inlet only for viscous flows currently!");
      */

      /* Get states for convenience */
      double rhoL = U(fpt, 0, 0);

      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VL[dim] = U(fpt, dim+1, 0) / rhoL;
        Vsq += VL[dim] * VL[dim];
      }

      double eL = U(fpt, nDims + 1 ,0);
      double PL = (gamma - 1.0) * (eL - 0.5 * rhoL * Vsq);


      /* Compute left normal velocity and dot product of normal*/
      double VnL = 0.0;
      double alpha = 0.0;

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += VL[dim] * norm(fpt, dim, 0);
        alpha += norm_fs(dim) * norm(fpt, dim, 0);
      }

      /* Compute speed of sound */
      double cL = std::sqrt(gamma * PL / rhoL);

      /* Extrapolate Riemann invariant */
      double R_plus  = VnL + 2.0 * cL / (gamma - 1.0);

      /* Specify total enthalpy */
      double H_tot = gamma * R_ref / (gamma - 1.0) * T_tot_fs;

      /* Compute total speed of sound squared */
      double c_tot_sq = (gamma - 1.0) * (H_tot - (eL + PL) / rhoL + 0.5 * Vsq) + cL * cL;

      /* Coefficients of Quadratic equation */
      double aa = 1.0 + 0.5 * (gamma - 1.0) * alpha * alpha;
      double bb = -(gamma - 1.0) * alpha * R_plus;
      double cc = 0.5 * (gamma - 1.0) * R_plus * R_plus - 2.0 * c_tot_sq / (gamma - 1.0);

      /* Solve quadratic for right velocity */
      double dd = bb * bb  - 4.0 * aa * cc;
      dd = std::sqrt(max(dd, 0.0));  // Max to keep from producing NaN
      double VR_mag = (dd - bb) / (2.0 * aa);
      VR_mag = max(VR_mag, 0.0);
      double VR_mag_sq = VR_mag * VR_mag;

      /* Compute right speed of sound and Mach */
      /* Note: Need to verify what is going on here. */
      double cR_sq = c_tot_sq - 0.5 * (gamma - 1.0) * VR_mag_sq;
      double Mach_sq = VR_mag_sq / cR_sq;
      Mach_sq = min(Mach_sq, 1.0); // Clamp to Mach = 1
      VR_mag_sq = Mach_sq * cR_sq;
      VR_mag = std::sqrt(VR_mag_sq);
      cR_sq = c_tot_sq - 0.5 * (gamma - 1.0) * VR_mag_sq;

      /* Compute right states */

      double TR = cR_sq / (gamma * R_ref);
      double PR = P_tot_fs * std::pow(TR / T_tot_fs, gamma/ (gamma - 1.0));

      U(fpt, 0, 1) = PR / (R_ref * TR);

      Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VR[dim] = VR_mag * norm_fs(dim);
        U(fpt, dim+1, 1) = U(fpt, 0, 1) * VR[dim];
        Vsq += VR[dim] * VR[dim];
      }

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;

      break;
    }

    case 5: /* Subsonic Outlet */
    { 
      /* Extrapolate Density */
      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Extrapolate Momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(fpt, dim+1, 1) =  U(fpt, dim+1, 0);
      }

      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      /* Fix pressure */
      U(fpt, nDims + 1, 1) = P_fs/(gamma-1.0) + 0.5 * momF; 

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;

      break;

      /*
      if (!input->viscous)
        ThrowException("Subsonic outlet only for viscous flows currently!");
      */

      double VL[3]; double VR[3];

      /* Get states for convenience */
      double rhoL = U(fpt, 0, 0);

      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VL[dim] = U(fpt, dim+1, 0) / rhoL;
        Vsq += VL[dim] * VL[dim];
      }

      double eL = U(fpt, nDims + 1, 0);
      double PL = (gamma - 1.0) * (eL - 0.5 * rhoL * Vsq);

      /* Compute left normal velocity */
      double VnL = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += VL[dim] * norm(fpt, dim, 0);
      }

      /* Compute speed of sound */
      double cL = std::sqrt(gamma * PL / rhoL);

      /* Extrapolate Riemann invariant */
      double R_plus  = VnL + 2.0 * cL / (gamma - 1.0);

      /* Extrapolate entropy */
      double s = PL / std::pow(rhoL, gamma);

      /* Fix pressure */
      double PR = P_fs;

      U(fpt, 0, 1) = std::pow(PR / s, 1.0 / gamma);

      /* Compute right speed of sound and velocity magnitude */
      double cR = std::sqrt(gamma * PR/ U(fpt, 0, 1));

      double VnR = R_plus - 2.0 * cR / (gamma - 1.0);

      Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VR[dim] = VL[dim] + (VnR - VnL) * norm(fpt, dim, 0);
        U(fpt, dim + 1, 1) = U(fpt, 0, 1) * VR[dim];
        Vsq += VR[dim] * VR[dim];
      }

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;

      break;
    }

    case 6: /* Characteristic (from HiFiLES) */
    {
      /* Compute wall normal velocities */
      double VnL = 0.0; double VnR = 0.0;

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += U(fpt, dim+1, 0) / U(fpt, 0, 0) * norm(fpt, dim, 0);
        VnR += V_fs(dim) * norm(fpt, dim, 0);
      }
    

      /* Compute pressure. TODO: Compute pressure once!*/
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
      double PR = P_fs;

      /* Compute Riemann Invariants */
      double Rp = VnL + 2.0 / (gamma - 1) * std::sqrt(gamma * PL / 
          U(fpt, 0,0));
      double Rn = VnR - 2.0 / (gamma - 1) * std::sqrt(gamma * PR / 
          rho_fs);

      double cstar = 0.25 * (gamma - 1) * (Rp - Rn);
      double ustarn = 0.5 * (Rp + Rn);

      if (VnL < 0.0) /* Case 1: Inflow */
      {
        double s_inv = std::pow(rho_fs, gamma) / PR;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          Vsq += V_fs(dim) * V_fs(dim);

        double H_fs = gamma / (gamma - 1.0) * PR / rho_fs +
            0.5 * Vsq;

        double rhoR = std::pow(1.0 / gamma * (s_inv * cstar * cstar), 1.0/ 
            (gamma - 1.0));

        U(fpt, 0, 1) = rhoR;
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim+1, 1) = rhoR * (ustarn * norm(fpt, dim, 0) + V_fs(dim) - VnR * 
            norm(fpt, dim, 0));

        PR = rhoR / gamma * cstar * cstar;
        U(fpt, nDims + 1, 1) = rhoR * H_fs - PR;
        
      }
      else  /* Case 2: Outflow */
      {
        double rhoL = U(fpt, 0, 0);
        double s_inv = std::pow(rhoL, gamma) / PL;

        double rhoR = std::pow(1.0 / gamma * (s_inv * cstar * cstar), 1.0/ 
            (gamma - 1.0));

        U(fpt, 0, 1) = rhoR;

        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim + 1, 1) = rhoR * (ustarn * norm(fpt, dim, 0) +(U(fpt, dim + 1, 0) / 
                U(fpt, 0, 0) - VnL * norm(fpt, dim, 0)));

        double PR = rhoR / gamma * cstar * cstar;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          Vsq += U(fpt, dim+1, 1) * U(fpt, dim+1, 1) / (rhoR * rhoR) ;
        
        U(fpt, nDims + 1, 1) = PR / (gamma - 1.0) + 0.5 * rhoR * Vsq; 
      }

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;
      bc_bias(fpt) = 1;

      break;

    }
    case 7:
    case 8: /* Slip Wall */
    {
      double momN = 0.0;

      /* Compute wall normal momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
        momN += U(fpt, dim+1, 0) * norm(fpt, dim, 0);

      U(fpt, 0, 1) = U(fpt, 0, 0);

      for (unsigned int dim = 0; dim < nDims; dim++)
        /* Set boundary state to cancelled normal velocity (strong)*/
        //U(fpt, dim+1, 1) = U(fpt, dim+1, 0) - momN * norm(fpt, dim, 0);
        /* Set boundary state to reflect normal velocity */
        U(fpt, dim+1, 1) = U(fpt, dim+1, 0) - 2.0 * momN * norm(fpt, dim, 0);

      //U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0) - 0.5 * (momN * momN) / U(fpt, 0, 0);
      U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0);

      /* Set LDG bias */
      //LDG_bias(fpt) = -1;
      LDG_bias(fpt) = 0;
      bc_bias(fpt) = 1;

      break;
    }

    case 9: /* No-slip Wall (isothermal) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
      double PR = PL;
      //double TR = T_wall;
      double TR = 1; // T_wall = T_fs (hardcoded for couette flow)
      
      U(fpt, 0, 1) = PR / (R_ref * TR);

      /* Set velocity to zero */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(fpt, dim+1, 1) = 0.0;

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0);

      /* Set LDG bias */
      LDG_bias(fpt) = -1;

      break;
    }

    case 10: /* No-slip Wall (isothermal and moving) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);

      double PR = PL;
      double TR = T_wall;
      
      U(fpt, 0, 1) = PR / (R_ref * TR);

      /* Set velocity to wall velocity */
      double V_wall_sq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(fpt, dim+1, 1) = U(fpt, 0 , 1) * V_wall(dim);
        V_wall_sq += V_wall(dim) * V_wall(dim);
      }

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0 , 1) * V_wall_sq;

      /* Set LDG bias */
      LDG_bias(fpt) = -1;

      break;
    }

    case 11: /* No-slip Wall (adiabatic) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      /* Extrapolate density */
      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Extrapolate pressure */
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
      double PR = PL; 

      /* Set velocity to zero */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(fpt, dim+1, 1) = 0.0;
        //U(fpt, dim+1, 1) = -U(fpt, dim+1, 0);

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0);
      //U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 12: /* No-slip Wall (adiabatic and moving) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      /* Extrapolate density */
      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Extrapolate pressure */
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
      }

      momF /= U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
      double PR = PL; 

      /* Set velocity to wall velocity */
      double V_wall_sq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(fpt, dim+1, 1) = U(fpt, 0 , 1) * V_wall(dim);
        V_wall_sq += V_wall(dim) * V_wall(dim);
      }

      U(fpt, nDims + 1, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * V_wall_sq;

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }
  }

}

void apply_bcs_wrapper(mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, double rho_fs, 
    mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, mdvector_gpu<double> &V_wall, mdvector_gpu<double> &norm_fs, 
    mdvector_gpu<double> &norm, mdvector_gpu<unsigned int> &gfpt2bnd, 
    mdvector_gpu<unsigned int> &per_fpt_list, mdvector_gpu<int> &LDG_bias, mdvector_gpu<int> &bc_bias, unsigned int equation)
{
  unsigned int threads = 192;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (blocks != 0)
  {
    if (equation == AdvDiff)
    {
      if (nDims == 2)
        apply_bcs<1, 2, AdvDiff><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
      else
        apply_bcs<1, 3, AdvDiff><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
    }
    else if (equation == Burgers)
    {
      if (nDims == 2)
        apply_bcs<1, 2, Burgers><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
      else
        apply_bcs<1, 3, Burgers><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
    }
    else if (equation == EulerNS)
    {
      if (nDims == 2)
        apply_bcs<4, 2, EulerNS><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
      else
        apply_bcs<5, 3, EulerNS><<<blocks, threads>>>(U, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
            gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias, bc_bias); 
    }
  }
}

template<unsigned int nVars, unsigned int nDims>
__global__
void apply_bcs_dU(mdvector_gpu<double> dU, mdvector_gpu<double> U, mdvector_gpu<double> norm_gfpt,
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  if (bnd_id == 1) /* Periodic */
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
	  unsigned int per_fpt = per_fpt_list(fpt - nGfpts_int);
          dU(fpt, n, dim, 1) = dU(per_fpt, n, dim, 0);
      }
    }
  }
  else if(bnd_id == 11 || bnd_id == 12) /* Adibatic Wall */
  {
    double norm[nDims];

    for (unsigned int dim = 0; dim < nDims; dim++)
      norm[dim] = norm_gfpt(fpt, dim, 0);

    /* Extrapolate density gradient */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      dU(fpt, 0, dim, 1) = dU(fpt, 0, dim, 0);
    }

    if (nDims == 2)
    {
      /* Compute energy gradient */
      /* Get right states and velocity gradients*/
      double rho = U(fpt, 0, 0);
      double momx = U(fpt, 1, 0);
      double momy = U(fpt, 2, 0);
      double E = U(fpt, 3, 0);

      double u = momx / rho;
      double v = momy / rho;
      //double e_int = e / rho - 0.5 * (u*u + v*v);

      double rho_dx = dU(fpt, 0, 0, 0);
      double momx_dx = dU(fpt, 1, 0, 0);
      double momy_dx = dU(fpt, 2, 0, 0);
      double E_dx = dU(fpt, 3, 0, 0);
      
      double rho_dy = dU(fpt, 0, 1, 0);
      double momx_dy = dU(fpt, 1, 1, 0);
      double momy_dy = dU(fpt, 2, 1, 0);
      double E_dy = dU(fpt, 3, 1, 0);

      double du_dx = (momx_dx - rho_dx * u) / rho;
      double du_dy = (momx_dy - rho_dy * u) / rho;

      double dv_dx = (momy_dx - rho_dx * v) / rho;
      double dv_dy = (momy_dy - rho_dy * v) / rho;

      /* Option 1: Extrapolate momentum gradients */
      dU(fpt, 1, 0, 1) = dU(fpt, 1, 0, 0);
      dU(fpt, 1, 1, 1) = dU(fpt, 1, 1, 0);
      dU(fpt, 2, 0, 1) = dU(fpt, 2, 0, 0);
      dU(fpt, 2, 1, 1) = dU(fpt, 2, 1, 0);

      /* Option 2: Enforce constraint on tangential velocity gradient */
      //double du_dn = du_dx * norm[0] + du_dy * norm[1];
      //double dv_dn = dv_dx * norm[0] + dv_dy * norm[1];

      //dU(fpt, 1, 0, 1) = rho * du_dn * norm[0];
      //dU(fpt, 1, 1, 1) = rho * du_dn * norm[1];
      //dU(fpt, 2, 0, 1) = rho * dv_dn * norm[0];
      //dU(fpt, 2, 1, 1) =  rho * dv_dn * norm[1];

     // double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
     // double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

      /* Compute temperature gradient (actually C_v * rho * dT) */
      double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx);
      double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy);

      /* Compute wall normal temperature gradient */
      double dT_dn = dT_dx * norm[0] + dT_dy * norm[1];

      /* Option 1: Simply remove contribution of dT from total energy gradient */
      dU(fpt, 3, 0, 1) = E_dx - dT_dn * norm[0]; 
      dU(fpt, 3, 1, 1) = E_dy - dT_dn * norm[1]; 

      /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
      //dU(fpt, 3, 0, 1) = (dT_dx - dT_dn * norm[0]) + rho_dx * U(fpt, 3, 1) / rho; 
      //dU(fpt, 3, 1, 1) = (dT_dy - dT_dn * norm[1]) + rho_dy * U(fpt, 3, 1) / rho; 
    }
    else
    {
      /* Compute energy gradient */
      /* Get right states and velocity gradients*/
      double rho = U(fpt, 0, 0);
      double momx = U(fpt, 1, 0);
      double momy = U(fpt, 2, 0);
      double momz = U(fpt, 3, 0);
      double E = U(fpt, 4, 0);

      double u = momx / rho;
      double v = momy / rho;
      double w = momz / rho;

      /* Gradients */
      double rho_dx = dU(fpt, 0, 0, 0);
      double momx_dx = dU(fpt, 1, 0, 0);
      double momy_dx = dU(fpt, 2, 0, 0);
      double momz_dx = dU(fpt, 3, 0, 0);
      double E_dx = dU(fpt, 4, 0, 0);

      double rho_dy = dU(fpt, 0, 1, 0);
      double momx_dy = dU(fpt, 1, 1, 0);
      double momy_dy = dU(fpt, 2, 1, 0);
      double momz_dy = dU(fpt, 3, 1, 0);
      double E_dy = dU(fpt, 4, 1, 0);

      double rho_dz = dU(fpt, 0, 2, 0);
      double momx_dz = dU(fpt, 1, 2, 0);
      double momy_dz = dU(fpt, 2, 2, 0);
      double momz_dz = dU(fpt, 3, 2, 0);
      double E_dz = dU(fpt, 4, 2, 0);

      double du_dx = (momx_dx - rho_dx * u) / rho;
      double du_dy = (momx_dy - rho_dy * u) / rho;
      double du_dz = (momx_dz - rho_dz * u) / rho;

      double dv_dx = (momy_dx - rho_dx * v) / rho;
      double dv_dy = (momy_dy - rho_dy * v) / rho;
      double dv_dz = (momy_dz - rho_dz * v) / rho;

      double dw_dx = (momz_dx - rho_dx * w) / rho;
      double dw_dy = (momz_dy - rho_dy * w) / rho;
      double dw_dz = (momz_dz - rho_dz * w) / rho;

      /* Option 1: Extrapolate momentum gradients */
      dU(fpt, 1, 0, 1) = dU(fpt, 1, 0, 0);
      dU(fpt, 1, 1, 1) = dU(fpt, 1, 1, 0);
      dU(fpt, 1, 2, 1) = dU(fpt, 1, 2, 0);

      dU(fpt, 2, 0, 1) = dU(fpt, 2, 0, 0);
      dU(fpt, 2, 1, 1) = dU(fpt, 2, 1, 0);
      dU(fpt, 2, 2, 1) = dU(fpt, 2, 2, 0);

      dU(fpt, 3, 0, 1) = dU(fpt, 3, 0, 0);
      dU(fpt, 3, 1, 1) = dU(fpt, 3, 1, 0);
      dU(fpt, 3, 2, 1) = dU(fpt, 3, 2, 0);

      /* Option 2: Enforce constraint on tangential velocity gradient */
      //double du_dn = du_dx * norm[0] + du_dy * norm[1] + du_dz * norm[2];
      //double dv_dn = dv_dx * norm[0] + dv_dy * norm[1] + dv_dz * norm[2];
      //double dw_dn = dw_dx * norm[0] + dw_dy * norm[1] + dw_dz * norm[2];

      //dU(fpt, 1, 0, 1) = rho * du_dn * norm[0];
      //dU(fpt, 1, 1, 1) = rho * du_dn * norm[1];
      //dU(fpt, 1, 2, 1) = rho * du_dn * norm[2];
      //dU(fpt, 2, 0, 1) = rho * dv_dn * norm[0];
      //dU(fpt, 2, 1, 1) =  rho * dv_dn * norm[1];
      //dU(fpt, 2, 2, 1) =  rho * dv_dn * norm[2];
      //dU(fpt, 3, 0, 1) = rho * dw_dn * norm[0];
      //dU(fpt, 3, 1, 1) =  rho * dw_dn * norm[1];
      //dU(fpt, 3, 2, 1) =  rho * dw_dn * norm[2];

     // double dke_dx = 0.5 * (u*u + v*v + w*w) * rho_dx + rho * (u * du_dx + v * dv_dx + w * dw_dx);
     // double dke_dy = 0.5 * (u*u + v*v + w*w) * rho_dy + rho * (u * du_dy + v * dv_dy + w * dw_dy);
     // double dke_dz = 0.5 * (u*u + v*v + w*w) * rho_dz + rho * (u * du_dz + v * dv_dz + w * dw_dz);

      /* Compute temperature gradient (actually C_v * rho * dT) */
      double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx + w * dw_dx);
      double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy + w * dw_dy);
      double dT_dz = E_dz - rho_dz * E/rho - rho * (u * du_dz + v * dv_dz + w * dw_dz);

      /* Compute wall normal temperature gradient */
      double dT_dn = dT_dx * norm[0] + dT_dy * norm[1] + dT_dz * norm[2];

      /* Option 1: Simply remove contribution of dT from total energy gradient */
      dU(fpt, 4, 0, 1) = E_dx - dT_dn * norm[0]; 
      dU(fpt, 4, 1, 1) = E_dy - dT_dn * norm[1]; 
      dU(fpt, 4, 2, 1) = E_dz - dT_dn * norm[2]; 

      /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
      //dU(fpt, 4, 0, 1) = (dT_dx - dT_dn * norm[0]) + rho_dx * U(fpt, 4, 1) / rho; 
      //dU(fpt, 4, 1, 1) = (dT_dy - dT_dn * norm[1]) + rho_dy * U(fpt, 4, 1) / rho; 
      //dU(fpt, 4, 2, 1) = (dT_dz - dT_dn * norm[2]) + rho_dz * U(fpt, 4, 1) / rho; 

    }

  }
  else
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        dU(fpt, n, dim, 1) = dU(fpt, n, dim , 0);
      }
    }
  }

}


void apply_bcs_dU_wrapper(mdvector_gpu<double> &dU, mdvector_gpu<double> &U, mdvector_gpu<double> &norm, 
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, 
    unsigned int nDims, mdvector_gpu<unsigned int> &gfpt2bnd, mdvector_gpu<unsigned int> &per_fpt_list)
{
  unsigned int threads = 192;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (blocks != 0)
  {
    if (nDims == 2)
      apply_bcs_dU<4, 2><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd, per_fpt_list);
    else
      apply_bcs_dU<5, 3><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd, per_fpt_list);
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void rusanov_flux(mdvector_gpu<double> U, mdvector_gpu<double> Fconv, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> AdvDiff_A, 
    mdvector_gpu<double> norm_gfpts, mdvector_gpu<int> outnorm_gfpts, 
    mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<int> LDG_bias, mdvector_gpu<int> bc_bias,
    double gamma, double rus_k, unsigned int nFpts, unsigned int startFpt,
    unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  /* Apply central flux at boundaries */
  double k = rus_k;
  if (bc_bias(fpt))
  {
    k = 1.0;
  }

  double FL[nVars]; double FR[nVars];
  double WL[nVars]; double WR[nVars];
  double norm[nDims]; double outnorm[2];

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(fpt, dim, 0);
  }

  outnorm[0] = outnorm_gfpts(fpt, 0);
  outnorm[1] = outnorm_gfpts(fpt, 1);


  /* Initialize FL, FR */
  for (unsigned int n = 0; n < nVars; n++)
  {
    FL[n] = 0.0; FR[n] = 0.0;
  }

  /* Get interface-normal flux components  (from L to R) */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      FL[n] += Fconv(fpt, n, dim, 0) * norm[dim];
      FR[n] += Fconv(fpt, n, dim, 1) * norm[dim];
    }
  }

  /* If on boundary, set common to right state flux */
  if (LDG_bias(fpt) != 0)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(fpt, n, 0) = FR[n] * outnorm[0];
      Fcomm(fpt, n, 1) = FR[n] * -outnorm[1];
    }

    return;
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
  }

  /* Get numerical wavespeed */
  double waveSp = 0.;
  if (equation == AdvDiff) 
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      waveSp += AdvDiff_A(dim) * norm[dim];
    }

    waveSp_gfpts(fpt) = waveSp;

    waveSp = std::abs(waveSp);
  }
  else if (equation == Burgers) 
  {
    double AnL = 0;
    double AnR = 0;

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      AnL += WL[0] * norm[dim];
      AnR += WR[0] * norm[dim];
    }

    waveSp = max(std::abs(AnL), std::abs(AnR));

    // NOTE: Can I just store absolute of waveSp?
    waveSp_gfpts(fpt) = waveSp;
    waveSp = std::abs(waveSp);
  }
  else if (equation == EulerNS)
  {
    /* Compute speed of sound */
    double aL = std::sqrt(std::abs(gamma * P(fpt, 0) / WL[0]));
    double aR = std::sqrt(std::abs(gamma * P(fpt, 1) / WR[0]));

    /* Compute normal velocities */
    double VnL = 0.0; double VnR = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      VnL += WL[dim+1]/WL[0] * norm[dim];
      VnR += WR[dim+1]/WR[0] * norm[dim];
    }

    waveSp = max(std::abs(VnL) + aL, std::abs(VnR) + aR);

    // NOTE: Can I just store absolute of waveSp?
    waveSp_gfpts(fpt) = waveSp;
    waveSp = std::abs(waveSp);
  }

  /* Compute common normal flux */
  for (unsigned int n = 0; n < nVars; n++)
  {
    double F = 0.5 * (FR[n]+FL[n]) - 0.5 * waveSp * (1.0-k) * 
        (WR[n]-WL[n]);
    Fcomm(fpt, n, 0) = F * outnorm[0];
    Fcomm(fpt, n, 1) = F * -outnorm[1];
  }

}

void rusanov_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fconv, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, 
    mdvector_gpu<double> &norm, mdvector_gpu<int> &outnorm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<int> &LDG_bias, mdvector_gpu<int> &bc_bias, double gamma, double rus_k, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 256;
  //unsigned int blocks = (nFpts + threads - 1)/threads;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;
  //int threads; int minBlocks; int blocks;

  //cudaOccupancyMaxPotentialBlockSize(&minBlocks, &threads, (const void*)rusanov_flux, 0, nFpts);

  //blocks = (nFpts + threads - 1) / threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      rusanov_flux<1, 2, AdvDiff><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);
    else
      rusanov_flux<1, 3, AdvDiff><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);
  }
  else if (equation == Burgers)
  {
    if (nDims == 2)
      rusanov_flux<1, 2, Burgers><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);
    else
      rusanov_flux<1, 3, Burgers><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      rusanov_flux<4, 2, EulerNS><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);
    else
      rusanov_flux<5, 3, EulerNS><<<blocks, threads>>>(U, Fconv, Fcomm, P, AdvDiff_A, norm, outnorm, 
          waveSp, LDG_bias, bc_bias, gamma, rus_k, nFpts, startFpt, endFpt);

  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void roe_flux(mdvector_gpu<double> U, mdvector_gpu<double> Fconv, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> norm_gfpts, 
    mdvector_gpu<int> outnorm_gfpts, mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<int> bc_bias,
    double gamma, unsigned int nFpts, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  /* Apply central flux at boundaries */
  double k = 0;
  if (bc_bias(fpt))
  {
    k = 1.0;
  }

  double FL[nVars]; double FR[nVars]; 
  double F[nVars]; double dW[nVars];
  double norm[nDims]; double outnorm[2];

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(fpt, dim, 0);
  }

  outnorm[0] = outnorm_gfpts(fpt, 0);
  outnorm[1] = outnorm_gfpts(fpt, 1);


  /* Initialize FL, FR */
  for (unsigned int n = 0; n < nVars; n++)
  {
    FL[n] = 0.0; FR[n] = 0.0;
  }

  /* Get interface-normal flux components  (from L to R) */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      FL[n] += Fconv(fpt, n, dim, 0) * norm[dim];
      FR[n] += Fconv(fpt, n, dim, 1) * norm[dim];
    }
  }

  /* Get difference in state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    dW[n] = U(fpt, n, 1) - U(fpt, n, 0);
  }

  /* Get numerical wavespeed */
  if (equation == EulerNS)
  {
    /* Primitive Variables */
    double gam = gamma;
    double rhoL = U(fpt, 0, 0);
    double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
    double vL = U(fpt, 2, 0) / U(fpt, 0, 0);
    double pL = (gam-1.0) * (U(fpt, 3, 0) - 0.5 * rhoL * (uL*uL + vL*vL));
    double hL = (U(fpt, 3, 0) + pL) / rhoL;

    double rhoR = U(fpt, 0, 1);
    double uR = U(fpt, 1, 1) / U(fpt, 0, 1);
    double vR = U(fpt, 2, 1) / U(fpt, 0, 1);
    double pR = (gam-1.0) * (U(fpt, 3, 1) - 0.5 * rhoR * (uR*uR + vR*vR));
    double hR = (U(fpt, 3, 0) + pL) / rhoL;

    /* Compute averaged values */
    double sq_rho = std::sqrt(rhoR / rhoL);
    double rrho = 1.0 / (sq_rho + 1.0);
    double um = rrho * (uL + sq_rho * uR);
    double vm = rrho * (vL + sq_rho * vR);
    double hm = rrho * (hL + sq_rho * hR);

    double Vmsq = 0.5 * (um*um + vm*vm);
    double am = std::sqrt((gam-1.0) * (hm - Vmsq));
    double Vnm = um * norm[0] + vm * norm[1];

    /* Compute Wavespeeds */
    double lambda0 = std::abs(Vnm);
    double lambdaP = std::abs(Vnm + am);
    double lambdaM = std::abs(Vnm - am);

    /* Entropy fix */
    double eps = 0.5 * (std::abs(FL[0] / rhoL - FR[0] / rhoR) + std::abs(std::sqrt(gam*pL/rhoL) - std::sqrt(gam*pR/rhoR)));
    if (lambda0 < 2.0 * eps)
      lambda0 = 0.25 * lambda0*lambda0 / eps + eps;
    if (lambdaP < 2.0 * eps)
      lambdaP = 0.25 * lambdaP*lambdaP / eps + eps;
    if (lambdaM < 2.0 * eps)
      lambdaM = 0.25 * lambdaM*lambdaM / eps + eps;

    /* Matrix terms */
    double a2 = 0.5 * (lambdaP + lambdaM) - lambda0;
    double a3 = 0.5 * (lambdaP - lambdaM) / am;
    double a1 = a2 * (gam-1.0) / (am*am);
    double a4 = a3 * (gam-1.0);
    double a5 = Vmsq * dW[0] - um * dW[1] - vm * dW[2] + dW[3];
    double a6 = Vnm * dW[0] - norm[0] * dW[1] - norm[1] * dW[2];
    double aL1 = a1 * a5 - a3 * a6;
    double bL1 = a4 * a5 - a2 * a6;

    F[0] = 0.5 * (FR[0] + FL[0]) - (1.0-k) * (lambda0 * dW[0] + aL1);
    F[1] = 0.5 * (FR[1] + FL[1]) - (1.0-k) * (lambda0 * dW[1] + aL1 * um + bL1 * norm[0]);
    F[2] = 0.5 * (FR[2] + FL[2]) - (1.0-k) * (lambda0 * dW[2] + aL1 * vm + bL1 * norm[1]);
    F[3] = 0.5 * (FR[3] + FL[3]) - (1.0-k) * (lambda0 * dW[3] + aL1 * hm + bL1 * Vnm);

    waveSp_gfpts(fpt) = max(max(lambda0, lambdaP), lambdaM);
  }

  /* Correct for positive parent space sign convention */
  for (unsigned int n = 0; n < nVars; n++)
  {
    Fcomm(fpt, n, 0) = F[n] * outnorm[0];
    Fcomm(fpt, n, 1) = F[n] * -outnorm[1];
  }
}

void roe_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fconv, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &norm, mdvector_gpu<int> &outnorm, 
    mdvector_gpu<double> &waveSp, mdvector_gpu<int> &bc_bias, double gamma, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 256;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == EulerNS)
  {
    if (nDims == 2)
      roe_flux<4, 2, EulerNS><<<blocks, threads>>>(U, Fconv, Fcomm, norm, outnorm, 
          waveSp, bc_bias, gamma, nFpts, startFpt, endFpt);
    else
      ThrowException("Roe flux only implemented for 2D!");
  }
  else
  {
    ThrowException("Roe flux not implemented for this equation type!");
  }
}

template <unsigned int nVars, unsigned int nDims>
__global__
void LDG_flux(mdvector_gpu<double> U, mdvector_gpu<double> Fvisc, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> Fcomm_no, mdvector_gpu<double> norm_gfpts, 
    mdvector_gpu<int> outnorm_gfpts, mdvector_gpu<int> LDG_bias, double beta, double tau, 
    unsigned int nFpts, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  double FL[nVars]; double FR[nVars];
  double WL[nVars]; double WR[nVars];
  double Fcomm_temp[nVars][nDims];
  double norm[nDims];
  double outnorm[2];
   
  /* Zero out temporary array */
  for (unsigned int n = 0; n < nVars; n++)
    for (unsigned int dim = 0; dim < nDims; dim++)
      Fcomm_temp[n][dim] = 0.0;

  /* Initialize FL, FR */
  for (unsigned int n = 0; n < nVars; n++)
  {
    FL[n] = 0.0; FR[n] = 0.0;
  }

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(fpt, dim, 0);
  }

  outnorm[0] = outnorm_gfpts(fpt, 0);
  outnorm[1] = outnorm_gfpts(fpt, 1);

  /* Setting sign of beta (from HiFiLES) */
  if (nDims == 2)
  {
    if (norm[0] + norm[1] < 0.0)
      beta = -beta;
  }
  else if (nDims == 3)
  {
    if (norm[0] + norm[1] + sqrt(2.) * norm[2] < 0.0)
      beta = -beta;
  }


  /* Get interface-normal flux components  (from L to R)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      FL[n] += Fvisc(fpt, n, dim, 0) * norm[dim];
      FR[n] += Fvisc(fpt, n, dim, 1) * norm[dim];
    }
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
  }

  /* Compute common normal viscous flux and accumulate */
  /* If interior, use central */
  if (LDG_bias(fpt) == 0)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm_temp[n][dim] += 0.5*(Fvisc(fpt, n, dim, 0) + Fvisc(fpt, n, dim, 1)) + 
          tau * norm[dim] * (WL[n] - WR[n]) + beta * norm[dim] * (FL[n] - FR[n]);
      }
    }
  }
  /* If boundary, use right state only */
  else
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm_temp[n][dim] += Fvisc(fpt, n, dim, 1) + tau * norm[dim] * (WL[n] - WR[n]);
      }
    }
  }

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      double F = Fcomm_temp[n][dim] * norm[dim];
      Fcomm(fpt, n, 0) += F * outnorm[0];
      Fcomm(fpt, n, 1) += F * -outnorm[1];
    }
  }

}

void LDG_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &Fcomm_temp, mdvector_gpu<double> &norm, 
    mdvector_gpu<int> &outnorm, mdvector_gpu<int> &LDG_bias, double beta, double tau, 
    unsigned int nFpts, unsigned int nVars, unsigned int nDims, unsigned int equation,
    unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 256;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    if (nDims == 2)
      LDG_flux<1, 2><<<blocks, threads>>>(U, Fvisc, Fcomm, Fcomm_temp, norm, outnorm, LDG_bias, beta, tau, 
          nFpts, startFpt, endFpt);
    else
      LDG_flux<1, 3><<<blocks, threads>>>(U, Fvisc, Fcomm, Fcomm_temp, norm, outnorm, LDG_bias, beta, tau, 
          nFpts, startFpt, endFpt);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      LDG_flux<4, 2><<<blocks, threads>>>(U, Fvisc, Fcomm, Fcomm_temp, norm, outnorm, LDG_bias, beta, tau, 
          nFpts, startFpt, endFpt);
    else
      LDG_flux<5, 3><<<blocks, threads>>>(U, Fvisc, Fcomm, Fcomm_temp, norm, outnorm, LDG_bias, beta, tau, 
          nFpts, startFpt, endFpt);
  }
}

template <unsigned int nDims>
__global__
void compute_common_U_LDG(mdvector_gpu<double> U, mdvector_gpu<double> Ucomm, 
    mdvector_gpu<double> norm, double beta, unsigned int nFpts, unsigned int nVars,
    mdvector_gpu<int> LDG_bias, unsigned int startFpt, unsigned int endFpt)
{
    const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;
    const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

    if (fpt >= endFpt || var >= nVars)
      return;

    /* Setting sign of beta (from HiFiLES) */
    if (nDims == 2)
    {
      if (norm(fpt, 0, 0) + norm(fpt, 1, 0) < 0.0)
        beta = -beta;
    }
    else if (nDims == 3)
    {
      if (norm(fpt, 0,0) + norm(fpt, 1, 0) + sqrt(2.) * norm(fpt, 2, 0) < 0.0)
        beta = -beta;
    }

    double UL = U(fpt, var, 0); double UR = U(fpt, var, 1);

    if (LDG_bias(fpt) == 0)
    {
      double UC = 0.5*(UL + UR) - beta*(UL - UR);
      Ucomm(fpt, var, 0) = UC;
      Ucomm(fpt, var, 1) = UC;
    }
    /* If on boundary, don't use beta (this is from HiFILES. Need to check) */
    else
    {
      Ucomm(fpt, var, 0) = UR;
      Ucomm(fpt, var, 1) = UR;
      //Ucomm(fpt, var, 0) = 0.5*(UL + UR);
      //Ucomm(fpt, var, 1) = 0.5*(UL + UR);
    }
}

void compute_common_U_LDG_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, mdvector_gpu<int> &LDG_bias, unsigned int startFpt,
    unsigned int endFpt)
{
  dim3 threads(32,4);
  dim3 blocks(((endFpt - startFpt + 1) + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (nDims == 2)
    compute_common_U_LDG<2><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts, nVars,
        LDG_bias, startFpt, endFpt);
  else
    compute_common_U_LDG<3><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts, nVars,
        LDG_bias, startFpt, endFpt);

}
__global__
void transform_flux_faces(mdvector_gpu<double> Fcomm, mdvector_gpu<double> dA, 
    unsigned int nFpts, unsigned int nVars)
{
    const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

    if (fpt >= nFpts || var >= nVars)
      return;

    Fcomm(fpt, var, 0) *= dA(fpt);
    Fcomm(fpt, var, 1) *= dA(fpt);
}

void transform_flux_faces_wrapper(mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &dA, 
    unsigned int nFpts, unsigned int nVars)
{
  dim3 threads(32,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  transform_flux_faces<<<blocks,threads>>>(Fcomm, dA, nFpts, nVars);

}

#ifdef _MPI
__global__
void pack_U(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> U, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  U_sbuffs(i, var) = U(fpts(i), var, 0);
}

void pack_U_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  pack_U<<<blocks,threads>>>(U_sbuffs, fpts, U, nVars, fpts.size());
}

__global__
void unpack_U(mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> U, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  U(fpts(i), var, 1) = U_rbuffs(i, var);
}

void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  unpack_U<<<blocks,threads>>>(U_rbuffs, fpts, U, nVars, fpts.size());
}

template<unsigned int nDims>
__global__
void pack_dU(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> dU, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    U_sbuffs(i, var, dim) = dU(fpts(i), var, dim, 0);
  }
}

void pack_dU_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (nDims == 2)
    pack_dU<2><<<blocks,threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
  else
    pack_dU<3><<<blocks,threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
}

template<unsigned int nDims>
__global__
void unpack_dU(mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> dU, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dU(fpts(i), var, dim, 1) = U_rbuffs(i, var, dim);
  }
}

void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (nDims == 2)
    unpack_dU<2><<<blocks,threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
  else 
    unpack_dU<3><<<blocks,threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
}

#endif
