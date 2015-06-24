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

__global__
void compute_Fvisc_fpts_2D_EulerNS(mdvector_gpu<double> Fvisc, 
    mdvector_gpu<double> U, mdvector_gpu<double> dU, unsigned int nFpts, double gamma, 
        double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;

  if (fpt >= nFpts)
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

void compute_Fvisc_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> Fvisc, 
    mdvector_gpu<double> U, mdvector_gpu<double> dU, unsigned int nFpts, double gamma, 
        double prandtl, double mu_in, double c_sth, double rt, bool fix_vis)
{
  unsigned int threads = 192;
  unsigned int blocks = (nFpts + threads - 1)/threads;

  compute_Fvisc_fpts_2D_EulerNS<<<threads, blocks>>>(Fvisc, U, dU, nFpts, gamma, 
      prandtl, mu_in, c_sth, rt, fix_vis);
}
__global__
void apply_bcs(mdvector_gpu<double> U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nVars, unsigned int nDims, double rho_fs, mdvector_gpu<double> V_fs, 
    double P_fs, double gamma, double R_ref, double T_tot_fs, double P_tot_fs, double T_wall, 
    mdvector_gpu<double> V_wall, mdvector_gpu<double> norm_fs, mdvector_gpu<double> norm, 
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list,
    mdvector_gpu<int> LDG_bias)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nFpts)
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
      /* Set boundaries to freestream values */
      U(fpt, 0, 1) = rho_fs;

      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(fpt, dim+1, 1) = rho_fs * V_fs(dim);
        Vsq += V_fs(dim) * V_fs(dim);
      }

      U(fpt, 3, 1) = P_fs/(gamma-1.0) + 0.5*rho_fs * Vsq; 

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 3: /* Supersonic Outlet */
    {
      /* Extrapolate boundary values from interior */
      for (unsigned int n = 0; n < nVars; n++)
        U(fpt, n, 1) = U(fpt, n, 0);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

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

      double eL = U(fpt, 3 ,0);
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

      U(fpt, 3, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 5: /* Subsonic Outlet */
    {
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

      double eL = U(fpt, 3 ,0);
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
        U(fpt, dim+1, 1) = U(fpt, 0, 1) * VR[dim];
        Vsq += VR[dim] * VR[dim];
      }

      U(fpt, 3, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

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
      double momF = (U(fpt, 1, 0) * U(fpt, 1, 0) + U(fpt, 2, 0) * 
          U(fpt, 2, 0)) / U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, 3, 0) - 0.5 * momF);
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
        U(fpt, 3, 1) = rhoR * H_fs - PR;
        
      }
      else  /* Case 2: Outflow */
      {
        double rhoL = U(fpt, 0, 0);
        double s_inv = std::pow(rhoL, gamma) / PL;

        double rhoR = std::pow(1.0 / gamma * (s_inv * cstar * cstar), 1.0/ 
            (gamma - 1.0));

        U(fpt, 0, 1) = rhoR;
        U(fpt, 1, 1) = rhoR * (ustarn * norm(fpt, 0, 0) +(U(fpt, 1, 0) / 
              U(fpt, 0, 0) - VnL * norm(fpt, 0, 0)));
        U(fpt, 2, 1) = rhoR * (ustarn * norm(fpt, 1, 0) +(U(fpt, 2, 0) / 
              U(fpt, 0, 0) - VnL * norm(fpt, 1, 0)));
        double PR = rhoR / gamma * cstar * cstar;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          Vsq += U(fpt, dim+1, 1) * U(fpt, dim+1, 1) / (rhoL * rhoL) ;
        
        U(fpt, 3, 1) = PR / (gamma - 1.0) + 0.5 * rhoR * Vsq; 
      }

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;

    }
    case 7: /* Slip Wall */
    {
      double momN = 0.0;

      /* Compute wall normal momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
        momN += U(fpt, dim+1, 0) * norm(fpt, dim, 0);

      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Set boundary state to cancel normal velocity */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(fpt, dim+1, 1) = U(fpt, dim+1, 0) - momN * norm(fpt, dim, 0);

      U(fpt, 3, 1) = U(fpt, 3, 0) - 0.5 * (momN * momN) / U(fpt, 0, 0);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 8: /* No-slip Wall (isothermal) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      double momF = (U(fpt, 1, 0) * U(fpt, 1, 0) + U(fpt, 2, 0) * 
          U(fpt, 2, 0)) / U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, 3, 0) - 0.5 * momF);

      double PR = PL;
      double TR = T_wall;
      
      U(fpt, 0, 1) = PR / (R_ref * TR);

      /* Set velocity to zero */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(fpt, dim+1, 1) = 0.0;

      U(fpt, 3, 1) = PR / (gamma - 1.0);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 9: /* No-slip Wall (isothermal and moving) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      double momF = (U(fpt, 1, 0) * U(fpt, 1, 0) + U(fpt, 2, 0) * 
          U(fpt, 2, 0)) / U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, 3, 0) - 0.5 * momF);

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

      U(fpt, 3, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0 , 1) * V_wall_sq;

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case 10: /* No-slip Wall (adiabatic) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      /* Extrapolate density */
      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Extrapolate pressure */
      double momF = (U(fpt, 1, 0) * U(fpt, 1, 0) + U(fpt, 2, 0) * 
          U(fpt, 2, 0)) / U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, 3, 0) - 0.5 * momF);
      double PR = PL; 

      /* Set velocity to zero */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(fpt, dim+1, 1) = 0.0;

      U(fpt, 3, 1) = PR / (gamma - 1.0);

      break;
    }

    case 11: /* No-slip Wall (adiabatic and moving) */
    {
      /*
      if (!input->viscous)
        ThrowException("No slip wall boundary only for viscous flows!");
      */

      /* Extrapolate density */
      U(fpt, 0, 1) = U(fpt, 0, 0);

      /* Extrapolate pressure */
      double momF = (U(fpt, 1, 0) * U(fpt, 1, 0) + U(fpt, 2, 0) * 
          U(fpt, 2, 0)) / U(fpt, 0, 0);

      double PL = (gamma - 1.0) * (U(fpt, 3, 0) - 0.5 * momF);
      double PR = PL; 

      /* Set velocity to wall velocity */
      double V_wall_sq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(fpt, dim+1, 1) = U(fpt, 0 , 1) * V_wall(dim);
        V_wall_sq += V_wall(dim) * V_wall(dim);
      }

      U(fpt, 3, 1) = PR / (gamma - 1.0) + 0.5 * U(fpt, 0, 1) * V_wall_sq;

      break;
    }
  }

}

void apply_bcs_wrapper(mdvector_gpu<double> U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nVars, unsigned int nDims, double rho_fs, mdvector_gpu<double> V_fs, 
    double P_fs, double gamma, double R_ref, double T_tot_fs, double P_tot_fs, double T_wall, 
    mdvector_gpu<double> V_wall, mdvector_gpu<double> norm_fs, mdvector_gpu<double> norm, 
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list,
    mdvector_gpu<int> LDG_bias)
{
  unsigned int threads = 192;
  unsigned int blocks = ((nFpts - nGfpts_int) + threads - 1)/threads;

  apply_bcs<<<threads, blocks>>>(U, nFpts, nGfpts_int, nVars, nDims, rho_fs, V_fs, P_fs, gamma, R_ref, 
      T_tot_fs, P_tot_fs, T_wall, V_wall, norm_fs, norm, gfpt2bnd, per_fpt_list, LDG_bias); 
}

__global__
void apply_bcs_dU(mdvector_gpu<double> dU, mdvector_gpu<double> U, unsigned int nFpts, 
    unsigned int nGfpts_int, unsigned int nVars, unsigned int nDims,
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nFpts)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  if (bnd_id == 1) /* Periodic */
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
          //unsigned int per_fpt = per_fpt_pairs[fpt];
          unsigned int per_fpt = per_fpt_list(fpt);
          dU(fpt, n, dim, 1) = dU(per_fpt, n, dim, 0);
      }
    }
  }
  else if(bnd_id == 10 || bnd_id == 11) /* Adibatic Wall */
  {
    /* Extrapolate gradients except for energy */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars - 1; n++)
      {
          dU(fpt, n, dim, 1) = dU(fpt, n, dim, 0);
      }
    }

    /* Compute energy gradient */
    /* Get right states and velocity gradients*/
    double rho = U(fpt, 0, 1);
    double momx = U(fpt, 1, 1);
    double momy = U(fpt, 2, 1);
    double e = U(fpt, 3, 1);

    double u = momx / rho;
    double v = momy / rho;
    double e_int = e / rho - 0.5 * (u*u + v*v);

    double rho_dx = dU(fpt, 0, 0, 1);
    double momx_dx = dU(fpt, 1, 0, 1);
    double momy_dx = dU(fpt, 2, 0, 1);
    
    double rho_dy = dU(fpt, 0, 1, 1);
    double momx_dy = dU(fpt, 1, 1, 1);
    double momy_dy = dU(fpt, 2, 1, 1);

    double du_dx = (momx_dx - rho_dx * u) / rho;
    double du_dy = (momx_dy - rho_dy * u) / rho;

    double dv_dx = (momy_dx - rho_dx * v) / rho;
    double dv_dy = (momy_dy - rho_dy * v) / rho;

    double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
    double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

    dU(fpt, 3, 0, 1) = (dke_dx + rho_dx * e_int);
    dU(fpt, 3, 1, 1) = (dke_dy + rho_dy * e_int);

    }

}


void apply_bcs_dU_wrapper(mdvector_gpu<double> dU, mdvector_gpu<double> U, unsigned int nFpts, 
    unsigned int nGfpts_int, unsigned int nVars, unsigned int nDims,
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list)
{
  unsigned int threads = 192;
  unsigned int blocks = ((nFpts - nGfpts_int) + threads - 1)/threads;

  apply_bcs_dU<<<threads, blocks>>>(dU, U, nFpts, nGfpts_int, nVars, nDims, 
      gfpt2bnd, per_fpt_list);
}

__global__
void rusanov_flux(mdvector_gpu<double> U, mdvector_gpu<double> Fconv, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> norm,
    mdvector_gpu<int> outnorm, mdvector_gpu<double> waveSp, double gamma, double rus_k,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;

  if (fpt >= nFpts)
    return;

  /* Currently hardcoded for Euler. Need to pass nVars as template arg. */
  double FL[4]; double FR[4];
  double WL[4]; double WR[4];

  /* Initialize FL, FR */
  for (unsigned int i = 0; i < 4; i++)
  {
    FL[i] = 0.0; FR[i] = 0.0;
  }

  /* Get interface-normal flux components  (from L to R)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      FL[n] += Fconv(fpt, n, dim, 0) * norm(fpt, dim, 0);
      FR[n] += Fconv(fpt, n, dim, 1) * norm(fpt, dim, 0);
    }
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
  }

  /* Get numerical wavespeed */
  /*
  if (input->equation == "AdvDiff")
  {
    waveSp[fpt] = 0.0;

    for (unsigned int dim = 0; dim < nDims; dim++)
      waveSp[fpt] += input->AdvDiff_A(dim) * norm(fpt, dim, 0);
  }
  else if (input->equation == "EulerNS")
  {
  */
  /* Compute speed of sound */
  double aL = std::sqrt(std::abs(gamma * P(fpt, 0) / WL[0]));
  double aR = std::sqrt(std::abs(gamma * P(fpt, 1) / WR[0]));

  /* Compute normal velocities */
  double VnL = 0.0; double VnR = 0.0;
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    VnL += WL[dim+1]/WL[0] * norm(fpt, dim, 0);
    VnR += WR[dim+1]/WR[0] * norm(fpt, dim, 0);
  }

  waveSp(fpt) = max(std::abs(VnL) + aL, std::abs(VnR) + aR);
  //}

  /* Compute common normal flux */
  for (unsigned int n = 0; n < nVars; n++)
  {
    Fcomm(fpt, n, 0) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp(fpt))*(1.0-rus_k) * (WR[n]-WL[n]);
    Fcomm(fpt, n, 1) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp(fpt))*(1.0-rus_k) * (WR[n]-WL[n]);

    /* Correct for positive parent space sign convention */
    Fcomm(fpt, n, 0) *= outnorm(fpt, 0);
    Fcomm(fpt, n, 1) *= -outnorm(fpt, 1);
  }

}

void rusanov_flux_wrapper(mdvector_gpu<double> U, mdvector_gpu<double> Fconv, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> norm,
    mdvector_gpu<int> outnorm, mdvector_gpu<double> waveSp, double gamma, double rus_k,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims)
{
  unsigned int threads = 192;
  unsigned int blocks = (nFpts + threads - 1)/threads;

  rusanov_flux<<<threads, blocks>>>(U, Fconv, Fcomm, P, norm, outnorm, waveSp, gamma, rus_k, 
      nFpts, nVars, nDims);
}

__global__
void LDG_flux(mdvector_gpu<double> U, mdvector_gpu<double> Fvisc, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> Fcomm_temp, mdvector_gpu<double> norm, 
    mdvector_gpu<int> outnorm, mdvector_gpu<int> LDG_bias, double beta, double tau, 
    unsigned int nFpts, unsigned int nVars, unsigned int nDims)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;

  if (fpt >= nFpts)
    return;

  /* Hardcoded for EulerNS. Will need to add nVar template */
  double FL[4]; double FR[4];
  double WL[4]; double WR[4];
   
  /* Zero out temporary array */
  for (unsigned int n = 0; n < nVars; n++)
    for (unsigned int dim = 0; dim < nDims; dim++)
      Fcomm_temp(fpt, n, dim) = 0.0;

  /* Initialize FL, FR */
  for (unsigned int n = 0; n < 4; n++)
  {
    FL[n] = 0.0; FR[n] = 0.0;
  }


  /* Setting sign of beta (from HiFiLES) */
  if (norm(fpt, 0, 0) + norm(fpt, 1, 0) < 0.0)
    beta = -beta;


  /* Get interface-normal flux components  (from L to R)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      FL[n] += Fvisc(fpt, n, dim, 0) * norm(fpt, dim, 0);
      FR[n] += Fvisc(fpt, n, dim, 1) * norm(fpt, dim, 0);
    }
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
  }

  /* Compute common normal viscous flux and accumulate */
  if (!LDG_bias(fpt))
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm_temp(fpt, n, 0) += 0.5*(Fvisc(fpt, n, 0, 0) + Fvisc(fpt, n, 0, 1)) + tau * norm(fpt, 0, 0)* (WL[n]
          - WR[n]) + beta * norm(fpt, 0, 0)* (FL[n] - FR[n]);
      Fcomm_temp(fpt, n, 1) += 0.5*(Fvisc(fpt, n, 1, 0) + Fvisc(fpt, n, 1, 1)) + tau * norm(fpt, 1, 0)* (WL[n]
          - WR[n]) + beta * norm(fpt, 1, 0)* (FL[n] - FR[n]);
    }
  }
  /* Else, only use left flux state */
  else
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm_temp(fpt, n, 0) += Fvisc(fpt, n, 0, 0) + tau * norm(fpt, 0, 0)* (WL[n] - WR[n]);
      Fcomm_temp(fpt, n, 1) += Fvisc(fpt, n, 1, 0) + tau * norm(fpt, 1, 0)* (WL[n] - WR[n]);
    }
  }

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(fpt, n, 0) += (Fcomm_temp(fpt, n, dim) * norm(fpt, dim, 0)) * outnorm(fpt, 0);
      Fcomm(fpt, n, 1) += (Fcomm_temp(fpt, n, dim) * norm(fpt, dim, 0)) * -outnorm(fpt, 1);
    }
  }

}

void LDG_flux_wrapper(mdvector_gpu<double> U, mdvector_gpu<double> Fvisc, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> Fcomm_temp, mdvector_gpu<double> norm, 
    mdvector_gpu<int> outnorm, mdvector_gpu<int> LDG_bias, double beta, double tau, 
    unsigned int nFpts, unsigned int nVars, unsigned int nDims)
{
  unsigned int threads = 192;
  unsigned int blocks = (nFpts + threads - 1)/threads;

  LDG_flux<<<threads,blocks>>>(U, Fvisc, Fcomm, Fcomm_temp, norm, outnorm, LDG_bias, beta, tau, 
      nFpts, nVars, nDims);
}
__global__
void compute_common_U_LDG(mdvector_gpu<double> U, mdvector_gpu<double> Ucomm, 
    mdvector_gpu<double> norm, double beta, unsigned int nFpts, unsigned int nVars)
{
    const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

    if (fpt >= nFpts || var >= nVars)
      return;

    /* Setting sign of beta (from HiFiLES) */
    if (norm(fpt, 0, 0) + norm(fpt, 1, 0) < 0.0)
      beta = -beta;

    double UL = U(fpt, var, 0); double UR = U(fpt, var, 1);

    Ucomm(fpt, var, 0) = 0.5*(UL + UR) - beta*(UL - UR);
    Ucomm(fpt, var, 1) = 0.5*(UL + UR) - beta*(UL - UR);

}

void compute_common_U_LDG_wrapper(mdvector_gpu<double> U, mdvector_gpu<double> Ucomm, 
    mdvector_gpu<double> norm, double beta, unsigned int nFpts, unsigned int nVars)
{
  dim3 threads(32,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  compute_common_U_LDG<<<threads, blocks>>>(U, Ucomm, norm, beta, nFpts, nVars);

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

void transform_flux_faces_wrapper(mdvector_gpu<double> Fcomm, mdvector_gpu<double> dA, 
    unsigned int nFpts, unsigned int nVars)
{
  dim3 threads(32,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  transform_flux_faces<<<threads, blocks>>>(Fcomm, dA, nFpts, nVars);

}
