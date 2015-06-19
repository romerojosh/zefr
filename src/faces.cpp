#include <cmath>

#include <omp.h>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"


Faces::Faces(GeoStruct *geo, const InputStruct *input)
{
  this->input = input;
  this->geo = geo;
  nFpts = geo->nGfpts;
}

void Faces::setup(unsigned int nDims, unsigned int nVars)
{
  this->nVars = nVars;
  this->nDims = nDims;

  /* Allocate memory for solution structures */
  U.assign({2, nVars, nFpts});
  dU.assign({2, nVars, nDims, nFpts});
  Fconv.assign({2, nVars, nDims, nFpts});
  Fvisc.assign({2, nVars, nDims, nFpts});
  Fcomm.assign({2, nVars, nFpts});
  Ucomm.assign({2, nVars, nFpts});

  /* If running Euler/NS, allocate memory for pressure */
  if (input->equation == "EulerNS")
    P.assign({2,nFpts});

  waveSp.assign(nFpts,0.0);

  /* Allocate memory for geometry structures */
  norm.assign({2, nDims, nFpts});
  outnorm.assign({2, nFpts});
  dA.assign(nFpts,0.0);
  jaco.assign({2, nDims, nDims , nFpts});
}

void Faces::apply_bcs()
{
  /* Create some useful variables outside loop */
  std::array<double, 3> VL, VR;

  /* Loop over boundary flux points */
#pragma omp parallel for
  for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
  {
    unsigned int bnd_id = geo->gfpt2bnd[fpt - geo->nGfpts_int];

    /* Apply specified boundary condition */
    switch(bnd_id)
    {
      case 1:/* Periodic */
      {
        unsigned int per_fpt = geo->per_fpt_pairs[fpt];

        for (unsigned int n = 0; n < nVars; n++)
        {
          U(1, n, fpt) = U(0, n, per_fpt);
        }
        break;
      }
    
      case 2: /* Farfield and Supersonic Inlet */
      {
        /* Set boundaries to freestream values */
        U(1, 0, fpt) = input->rho_fs;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(1, dim+1, fpt) = input->rho_fs * input->V_fs[dim];
          Vsq += input->V_fs[dim] * input->V_fs[dim];
        }

        U(1, 3, fpt) = input->P_fs/(input->gamma-1.0) + 0.5*input->rho_fs * Vsq; 
        break;
      }

      case 3: /* Supersonic Outlet */
      {
        /* Extrapolate boundary values from interior */
        for (unsigned int n = 0; n < nVars; n++)
          U(1, n, fpt) = U(0, n, fpt);
        break;
      }

      case 4: /* Subsonic Inlet */
      {
        if (!input->viscous)
          ThrowException("Subsonic inlet only for viscous flows currently!");

        /* Get states for convenience */
        double rhoL = U(0, 0, fpt);

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VL[dim] = U(0, dim+1, fpt) / rhoL;
          Vsq += VL[dim] * VL[dim];
        }

        double eL = U(0, 3 ,fpt);
        double PL = (input->gamma - 1.0) * (eL - 0.5 * Vsq);


        /* Compute left normal velocity and dot product of normal*/
        double VnL = 0.0;
        double alpha = 0.0;

        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += VL[dim] * norm(0, dim, fpt);
          alpha += input->norm_fs[dim] * norm(0, dim, fpt);
        }

        /* Compute speed of sound */
        double cL = std::sqrt(input->gamma * PL / rhoL);

        /* Extrapolate Riemann invariant */
        double R_plus  = VnL + 2.0 * cL / (input->gamma - 1.0);

        /* Specify total enthalpy */
        double H_tot = input->gamma * input->R_ref / (input->gamma - 1.0) * input->T_tot_fs;

        /* Compute total speed of sound squared */
        double c_tot_sq = (input->gamma - 1.0) * (H_tot - (eL + PL) / rhoL + 0.5 * Vsq) + cL * cL;

        /* Coefficients of Quadratic equation */
        double aa = 1.0 + 0.5 * (input->gamma - 1.0) * alpha * alpha;
        double bb = -(input->gamma - 1.0) * alpha * R_plus;
        double cc = 0.5 * (input->gamma - 1.0) * R_plus * R_plus - 2.0 * c_tot_sq / (input->gamma - 1.0);

        /* Solve quadratic for right velocity */
        double dd = bb * bb  - 4.0 * aa * cc;
        dd = std::sqrt(std::max(dd, 0.0));  // Max to keep from producing NaN
        double VR_mag = (dd - bb) / (2.0 * aa);
        VR_mag = std::max(VR_mag, 0.0);
        double VR_mag_sq = VR_mag * VR_mag;

        /* Compute right speed of sound and Mach */
        /* Note: Need to verify what is going on here. */
        double cR_sq = c_tot_sq - 0.5 * (input->gamma - 1.0) * VR_mag_sq;
        double Mach_sq = VR_mag_sq / cR_sq;
        Mach_sq = std::min(Mach_sq, 1.0); // Clamp to Mach = 1
        VR_mag_sq = Mach_sq * cR_sq;
        VR_mag = std::sqrt(VR_mag_sq);
        cR_sq = c_tot_sq - 0.5 * (input->gamma - 1.0) * VR_mag_sq;

        /* Compute right states */

        double TR = cR_sq / (input->gamma * input->R_ref);
        double PR = input->P_tot_fs * std::pow(TR / input->T_tot_fs, input->gamma/ (input->gamma - 1.0));

        U(1, 0, fpt) = PR / (input->R_ref * TR);

        Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VR[dim] = VR_mag * input->norm_fs[dim];
          U(1, dim+1, fpt) = U(1, 0, fpt) * VR[dim];
          Vsq += VR[dim] * VR[dim];
        }

        U(1, 4, fpt) = PR / (input->gamma - 1.0) + 0.5 * U(1, 0, fpt) * Vsq;

        break;
      }

      case 5: /* Subsonic Outlet */
      {
        if (!input->viscous)
          ThrowException("Subsonic outlet only for viscous flows currently!");

        /* Get states for convenience */
        double rhoL = U(0, 0, fpt);

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VL[dim] = U(0, dim+1, fpt) / rhoL;
          Vsq += VL[dim] * VL[dim];
        }

        double eL = U(0, 3 ,fpt);
        double PL = (input->gamma - 1.0) * (eL - 0.5 * Vsq);

        /* Compute left normal velocity */
        double VnL = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += VL[dim] * norm(0, dim, fpt);
        }

        /* Compute speed of sound */
        double cL = std::sqrt(input->gamma * PL / rhoL);

        /* Extrapolate Riemann invariant */
        double R_plus  = VnL + 2.0 * cL / (input->gamma - 1.0);

        /* Extrapolate entropy */
        double s = PL / std::pow(rhoL, input->gamma);

        /* Fix pressure */
        double PR = input->P_fs;

        U(1, 0, fpt) = std::pow(PR / s, 1.0 / input->gamma);

        /* Compute right speed of sound and velocity magnitude */
        double cR = std::sqrt(input->gamma * PR/ U(0, 0, fpt));

        double VnR = R_plus - 2.0 * cR / (input->gamma - 1.0);

        Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VR[dim] = VL[dim] - (VnR - VnL) * norm(0, dim+1, fpt);
          U(1, dim+1, fpt) = U(1, 0, fpt) * VR[dim];
          Vsq += VR[dim] * VR[dim];
        }

        U(1, 4, fpt) = PR / (input->gamma - 1.0) + 0.5 * U(1, 0, fpt) * Vsq;

        break;
      }

      case 6: /* Characteristic (from HiFiLES) */
      {
        /* Compute wall normal velocities */
        double VnL = 0.0; double VnR = 0.0;

        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += U(0, dim+1, fpt) / U(0, 0, fpt) * norm(0, dim, fpt);
          VnR += input->V_fs[dim] * norm(0,dim,fpt);
        }
      

        /* Compute pressure. TODO: Compute pressure once!*/
        double momF = (U(0, 1, fpt) * U(0, 1, fpt) + U(0, 2, fpt) * 
            U(0, 2, fpt)) / U(0, 0, fpt);

        double PL = (input->gamma - 1.0) * (U(0, 3, fpt) - 0.5 * momF);
        double PR = input->P_fs;

        /* Compute Riemann Invariants */
        double Rp = VnL + 2.0 / (input->gamma - 1) * std::sqrt(input->gamma * PL / 
            U(0,0,fpt));
        double Rn = VnR - 2.0 / (input->gamma - 1) * std::sqrt(input->gamma * PR / 
            input->rho_fs);

        double cstar = 0.25 * (input->gamma - 1) * (Rp - Rn);
        double ustarn = 0.5 * (Rp + Rn);

        if (VnL < 0.0) /* Case 1: Inflow */
        {
          double s_inv = std::pow(input->rho_fs, input->gamma) / PR;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
            Vsq += input->V_fs[dim] * input->V_fs[dim];

          double H_fs = input->gamma / (input->gamma - 1.0) * PR / input->rho_fs +
              0.5 * Vsq;

          double rhoR = std::pow(1.0 / input->gamma * (s_inv * cstar * cstar), 1.0/ 
              (input-> gamma - 1.0));

          U(1, 0, fpt) = rhoR;
          for (unsigned int dim = 0; dim < nDims; dim++)
            U(1, dim+1, fpt) = rhoR * (ustarn * norm(0, dim, fpt) + input->V_fs[dim] - VnR * 
              norm(0, dim, fpt));

          PR = rhoR / input->gamma * cstar * cstar;
          U(1, 3, fpt) = rhoR * H_fs - PR;
          
        }
        else  /* Case 2: Outflow */
        {
          double rhoL = U(0, 0, fpt);
          double s_inv = std::pow(rhoL, input->gamma) / PL;

          double rhoR = std::pow(1.0 / input->gamma * (s_inv * cstar * cstar), 1.0/ 
              (input-> gamma - 1.0));

          U(1, 0, fpt) = rhoR;
          U(1, 1, fpt) = rhoR * (ustarn * norm(0, 0, fpt) +(U(0, 1, fpt) / 
                U(0, 0, fpt) - VnL * norm(0, 0, fpt)));
          U(1, 2, fpt) = rhoR * (ustarn * norm(0, 1, fpt) +(U(0, 2, fpt) / 
                U(0, 0, fpt) - VnL * norm(0, 1, fpt)));
          double PR = rhoR / input->gamma * cstar * cstar;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
            Vsq += U(1, dim+1, fpt) * U(1, dim+1, fpt) / (rhoL * rhoL) ;
          
          U(1, 3, fpt) = PR / (input->gamma - 1.0) + 0.5 * rhoR * Vsq; 
        }
 
        break;

      }
      case 7: /* Slip Wall */
      {
        double momN = 0.0;

        /* Compute wall normal momentum */
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN += U(0, dim+1, fpt) * norm(0, dim, fpt);

        U(1, 0, fpt) = U(0, 0, fpt);

        /* Set boundary state to cancel normal velocity */
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(1, dim+1, fpt) = U(0, dim+1, fpt) - momN * norm(0, dim, fpt);

        U(1, 3, fpt) = U(0, 3, fpt) - 0.5 * (momN * momN) / U(0, 0, fpt);
        break;
      }

      case 8: /* No-slip Wall (isothermal) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double momF = (U(0, 1, fpt) * U(0, 1, fpt) + U(0, 2, fpt) * 
            U(0, 2, fpt)) / U(0, 0, fpt);

        double PL = (input->gamma - 1.0) * (U(0, 3, fpt) - 0.5 * momF);

        double PR = PL;
        double TR = input->T_wall;
        
        U(1, 0, fpt) = PR / (input->R_ref * TR);

        for (unsigned int dim = 0; dim < nDims; dim++)
          U(1, dim+1, fpt) = 0.0;

        U(1, 3, fpt) = PR / (input->gamma - 1);

        break;
      }

      case 9: /* No-slip Wall (isothermal and moving) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double momF = (U(0, 1, fpt) * U(0, 1, fpt) + U(0, 2, fpt) * 
            U(0, 2, fpt)) / U(0, 0, fpt);

        double PL = (input->gamma - 1.0) * (U(0, 3, fpt) - 0.5 * momF);

        double PR = PL;
        double TR = input->T_wall;
        
        U(1, 0, fpt) = PR / (input->R_ref * TR);

        double V_wall_sq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(1, dim+1, fpt) = U(1, 0 , fpt) * input->V_wall[dim];
          V_wall_sq += input->V_wall[dim] * input->V_wall[dim];
        }

        U(1, 3, fpt) = PR / (input->gamma - 1.0) + 0.5 * U(1, 0 , fpt) * V_wall_sq;

        break;
      }

      case 10: /* No-slip Wall (adiabatic) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("Under construction !");
        break;
      }

      case 11: /* No-slip Wall (adiabatic and moving) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("Under construction !");
        break;
      }
    }
  } 
}

void Faces::apply_bcs_dU()
{
  /* Apply boundaries to solution derivative */
#pragma omp parallel for
  for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
  {
    unsigned int bnd_id = geo->gfpt2bnd[fpt - geo->nGfpts_int];

    /* Apply specified boundary condition */
    if (bnd_id == 1) /* Periodic */
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
            unsigned int per_fpt = geo->per_fpt_pairs[fpt];
            dU(1, n, dim, fpt) = dU(0, n, dim, per_fpt);
        }

      }
    }
    else if(bnd_id == 10 || bnd_id == 11) /* Adibatic Wall */
    {
        ThrowException("Under construction !");
    }
  }
}


void Faces::compute_Fconv()
{  
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(3)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fconv(0, n, dim, fpt) = input->AdvDiff_A[dim] * U(0, n, fpt);

          Fconv(1, n, dim, fpt) = input->AdvDiff_A[dim] * U(1, n, fpt);

        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        for (unsigned int slot = 0; slot < 2; slot ++)
        {
          /* Compute some primitive variables (keep pressure)*/
          double momF = (U(slot, 1, fpt) * U(slot, 1, fpt) + U(slot, 2, fpt) * 
              U(slot, 2, fpt)) / U(slot, 0, fpt);

          P(slot, fpt) = (input->gamma - 1.0) * (U(slot, 3, fpt) - 0.5 * momF);
          double H = (U(slot, 3, fpt) + P(slot,fpt)) / U(slot, 0, fpt);

          Fconv(slot, 0, 0, fpt) = U(slot, 1, fpt);
          Fconv(slot, 1, 0, fpt) = U(slot, 1, fpt) * U(slot, 1, fpt) / U(slot, 0, fpt) + P(slot, fpt);
          Fconv(slot, 2, 0, fpt) = U(slot, 1, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt);
          Fconv(slot, 3, 0, fpt) = U(slot, 1, fpt) * H;

          Fconv(slot, 0, 1, fpt) = U(slot, 2, fpt);
          Fconv(slot, 1, 1, fpt) = U(slot, 1, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt);
          Fconv(slot, 2, 1, fpt) = U(slot, 2, fpt) * U(slot, 2, fpt) / U(slot, 0, fpt) + P(slot, fpt);
          Fconv(slot, 3, 1, fpt) = U(slot, 2, fpt) * H;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("3D Euler not implemented yet!");
    }
  }

}

void Faces::compute_Fvisc()
{  
  if (input->equation == "AdvDiff")
  {
#pragma omp parallel for collapse(3)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fvisc(0, n, dim, fpt) = -input->AdvDiff_D * dU(0, n, dim, fpt);

          Fvisc(1, n, dim, fpt) = -input->AdvDiff_D * dU(1, n, dim, fpt);

        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int slot = 0; slot < 2; slot++)
      {
        /* Setting variables for convenience */
        /* States */
        double rho = U(slot, 0, fpt);
        double momx = U(slot, 1, fpt);
        double momy = U(slot, 2, fpt);
        double e = U(slot, 3, fpt);

        double u = momx / rho;
        double v = momy / rho;
        double e_int = e / rho - 0.5 * (u*u + v*v);

        /* Gradients */
        double rho_dx = dU(slot, 0, 0, fpt);
        double momx_dx = dU(slot, 1, 0, fpt);
        double momy_dx = dU(slot, 2, 0, fpt);
        double e_dx = dU(slot, 3, 0, fpt);
        
        double rho_dy = dU(slot, 0, 1, fpt);
        double momx_dy = dU(slot, 1, 1, fpt);
        double momy_dy = dU(slot, 2, 1, fpt);
        double e_dy = dU(slot, 3, 1, fpt);

        /* Set viscosity */
        double mu;
        if (input->fix_vis)
        {
          mu = input->mu;
        }
        /* If desired, use Sutherland's law */
        else
        {
          double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
          mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + input->c_sth);
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
        Fvisc(slot, 0, 0, fpt) = 0.0;
        Fvisc(slot, 1, 0, fpt) = -tauxx;
        Fvisc(slot, 2, 0, fpt) = -tauxy;
        Fvisc(slot, 3, 0, fpt) = -(u * tauxx + v * tauxy + (mu / input->prandtl) *
            input-> gamma * de_dx);

        Fvisc(slot, 0, 1, fpt) = 0.0;
        Fvisc(slot, 1, 1, fpt) = -tauxy;
        Fvisc(slot, 2, 1, fpt) = -tauyy;
        Fvisc(slot, 3, 1, fpt) = -(u * tauxy + v * tauyy + (mu / input->prandtl) *
            input->gamma * de_dy);
      }
    }
  }
}

void Faces::compute_common_F()
{
  if (input->fconv_type == "Rusanov")
    rusanov_flux();
  else
    ThrowException("Numerical convective flux type not recognized!");

  if (input->viscous)
  {
    if (input->fvisc_type == "LDG")
      LDG_flux();
    else if (input->fvisc_type == "Central")
      central_flux();
    else
      ThrowException("Numerical viscous flux type not recognized!");
  }

  transform_flux();
}

void Faces::compute_common_U()
{
  
  double beta = input->ldg_b;

  /* Compute common solution */
  if (input->fvisc_type == "LDG")
  {
#pragma omp parallel for 
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Setting sign of beta (from HiFiLES) */
      if (norm(0,0, fpt) + norm(0,1,fpt) < 0.0)
        beta = -beta;

      /* Get left and right state variables */
      // TODO: Verify that this is the correct formula. Seem different than papers...
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(0, n, fpt); double UR = U(1, n, fpt);

         Ucomm(0, n, fpt) = 0.5*(UL + UR) - beta*(UL - UR);
         Ucomm(1, n, fpt) = 0.5*(UL + UR) - beta*(UL - UR);

      }

    }
  }

  // TODO: Can potentially remove central treatment since LDG recovers.
  else if (input->fvisc_type == "Central")
  {
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      /* Get left and right state variables */
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(0, n, fpt); double UR = U(1, n, fpt);

        Ucomm(0, n, fpt) = 0.5*(UL + UR);
        Ucomm(1, n, fpt) = 0.5*(UL + UR);
      }

    }
  }
  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }

}

void Faces::rusanov_flux()
{

  double k = input->rus_k;

  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);

#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fconv(0, n, dim, fpt) * norm(0, dim, fpt);
        FR[n] += Fconv(1, n, dim, fpt) * norm(0, dim, fpt);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, n, fpt); WR[n] = U(1, n, fpt);
    }

    /* Get numerical wavespeed */
    if (input->equation == "AdvDiff")
    {
      waveSp[fpt] = 0.0;

      for (unsigned int dim = 0; dim < nDims; dim++)
        waveSp[fpt] += input->AdvDiff_A[dim] * norm(0, dim, fpt);
    }
    else if (input->equation == "EulerNS")
    {
      /* Compute speed of sound */
      double aL = std::sqrt(std::abs(input->gamma * P(0,fpt) / WL[0]));
      double aR = std::sqrt(std::abs(input->gamma * P(1,fpt) / WR[0]));

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += WL[dim+1]/WL[0] * norm(0,dim,fpt);
        VnR += WR[dim+1]/WR[0] * norm(0,dim,fpt);
      }

      waveSp[fpt] = std::max(std::abs(VnL) + aL, std::abs(VnR) + aR);
    }

    /* Compute common normal flux */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, n, fpt) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp[fpt])*(1.0-k) * (WR[n]-WL[n]);
      Fcomm(1, n, fpt) = 0.5 * (FR[n]+FL[n]) - 0.5 * std::abs(waveSp[fpt])*(1.0-k) * (WR[n]-WL[n]);

      /* Correct for positive parent space sign convention */
      Fcomm(0, n, fpt) *= outnorm(0, fpt);
      Fcomm(1, n, fpt) *= -outnorm(1, fpt);
    }
  }
}

void Faces::transform_flux()
{
#pragma omp parallel for collapse(2)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, n, fpt) *= dA[fpt];
      Fcomm(1, n, fpt) *= dA[fpt];
    }
  }
}

void Faces::LDG_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
  double tau = input->ldg_tau;
  double beta = input->ldg_b;

  mdvector<double> Fcomm_temp({nVars,nDims});

#pragma omp parallel for firstprivate(FL, FR, WL, WR, Fcomm_temp, tau, beta)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {

    /* Setting sign of beta (from HiFiLES) */
    if (norm(0,0, fpt) + norm(0,1,fpt) < 0.0)
      beta = -beta;

    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(0, n, dim, fpt) * norm(0, dim, fpt);
        FR[n] += Fvisc(1, n, dim, fpt) * norm(0, dim, fpt);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, n, fpt); WR[n] = U(1, n, fpt);
    }

    /* Compute common normal viscous flux and accumulate */
    /* If fpt is on interior face, use normal stencil */
    if (fpt < geo->nGfpts_int)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm_temp(n,0) += 0.5*(Fvisc(0, n, 0, fpt) + Fvisc(1, n, 0, fpt)) + tau * norm(0, 0, fpt)* (WL[n]
            - WR[n]) + beta * norm(0, 0, fpt)* (FL[n] - FR[n]);
        Fcomm_temp(n,1) += 0.5*(Fvisc(0, n, 1, fpt) + Fvisc(1, n, 1, fpt)) + tau * norm(0, 1, fpt)* (WL[n]
            - WR[n]) + beta * norm(0, 1, fpt)* (FL[n] - FR[n]);
      }
    }
    /* Else, only use left flux state (unless periodic) */
    else
    {
      unsigned int bnd_id = geo->gfpt2bnd[fpt - geo->nGfpts_int];
      if (bnd_id != 1)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fcomm_temp(n,0) += Fvisc(0, n, 0, fpt) + tau * norm(0, 0, fpt)* (WL[n] - WR[n]);
          Fcomm_temp(n,1) += Fvisc(0, n, 1, fpt) + tau * norm(0, 1, fpt)* (WL[n] - WR[n]);
        }
      }
      /* If periodic, treat like interior face */
      else
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fcomm_temp(n,0) += 0.5*(Fvisc(0, n, 0, fpt) + Fvisc(1, n, 0, fpt)) + tau * norm(0, 0, fpt)* (WL[n]
              - WR[n]) + beta * norm(0, 0, fpt)* (FL[n] - FR[n]);
          Fcomm_temp(n,1) += 0.5*(Fvisc(0, n, 1, fpt) + Fvisc(1, n, 1, fpt)) + tau * norm(0, 1, fpt)* (WL[n]
              - WR[n]) + beta * norm(0, 1, fpt)* (FL[n] - FR[n]);
        }
      }
    }

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm(0, n, fpt) += (Fcomm_temp(n, dim) * norm(0, dim, fpt)) * outnorm(0,fpt);
        Fcomm(1, n, fpt) += (Fcomm_temp(n, dim) * norm(0, dim, fpt)) * -outnorm(1,fpt);
      }
    }

    Fcomm_temp.fill(0);
  }
}

void Faces::central_flux()
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fvisc(0, n, dim, fpt) * norm(0, dim, fpt);
        FR[n] += Fvisc(1, n, dim, fpt) * norm(0, dim, fpt);
      }
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(0, n, fpt); WR[n] = U(1, n, fpt);
    }

    /* Compute common normal viscous flux and accumulate */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, n, fpt) += (0.5 * (FL[n]+FR[n])) * outnorm(0,fpt); 
      Fcomm(1, n, fpt) += (0.5 * (FL[n]+FR[n])) * -outnorm(1,fpt); 
    }
  }
}



