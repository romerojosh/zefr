#include <cmath>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "faces_kernels.h"
#include "solver_kernels.h"
#endif

Faces::Faces(GeoStruct *geo, InputStruct *input)
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
  U.assign({nFpts, nVars, 2});
  dU.assign({nFpts, nVars, nDims, 2});
  Fconv.assign({nFpts, nVars, nDims, 2});
  Fvisc.assign({nFpts, nVars, nDims, 2});
  Fcomm.assign({nFpts, nVars, 2});

  /* If viscous, allocate arrays used for LDG flux */
  //if(input->viscous)
  //{
    Fcomm_temp.assign({nFpts, nVars, nDims});
    LDG_bias.assign({nFpts}, 0);
  //}

  Ucomm.assign({nFpts, nVars, 2});

  /* If running Euler/NS, allocate memory for pressure */
  if (input->equation == EulerNS)
    P.assign({nFpts, 2});

  waveSp.assign({nFpts}, 0.0);

  /* Allocate memory for geometry structures */
  norm.assign({nFpts, nDims, 2});
  dA.assign({nFpts},0.0);
  jaco.assign({nFpts, nDims, nDims , 2});

#ifdef _MPI
  /* Allocate memory for send/receive buffers */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int pairedRank = entry.first;
    const auto &fpts = entry.second;

    U_sbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars, nDims}, 0.0);
    U_rbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars, nDims}, 0.0);
  }

  sreqs.resize(geo->fpt_buffer_map.size());
  rreqs.resize(geo->fpt_buffer_map.size());
#endif

}

void Faces::apply_bcs()
{
#ifdef _CPU
  /* Create some useful variables outside loop */
  std::array<double, 3> VL, VR;

  /* Loop over boundary flux points */
#pragma omp parallel for private(VL,VR)
  //for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    switch(bnd_id)
    {
      case 1:/* Periodic */
      {
        unsigned int per_fpt = geo->per_fpt_list(fpt - geo->nGfpts_int);

        for (unsigned int n = 0; n < nVars; n++)
        {
          U(fpt, n, 1) = U(per_fpt, n, 0);
        }
        break;
      }
    
      case 2: /* Farfield and Supersonic Inlet */
      {
        if (input->equation == AdvDiff)
        {
          /* Set boundaries to zero */
          U(fpt, 0, 1) = 0;
        }
        else
        {
          /* Set boundaries to freestream values */
          U(fpt, 0, 1) = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            U(fpt, dim+1, 1) = input->rho_fs * input->V_fs(dim);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          U(fpt, nDims + 1, 1) = input->P_fs/(input->gamma-1.0) + 0.5*input->rho_fs * Vsq; 
        }

        /* Set LDG bias */
        //LDG_bias(fpt) = -1;
        LDG_bias(fpt) = 0;

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
        if (!input->viscous)
          ThrowException("Subsonic inlet only for viscous flows currently!");

        /* Get states for convenience */
        double rhoL = U(fpt, 0, 0);

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VL[dim] = U(fpt, dim+1, 0) / rhoL;
          Vsq += VL[dim] * VL[dim];
        }

        double eL = U(fpt, nDims + 1 ,0);
        double PL = (input->gamma - 1.0) * (eL - 0.5 * rhoL * Vsq);


        /* Compute left normal velocity and dot product of normal*/
        double VnL = 0.0;
        double alpha = 0.0;

        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += VL[dim] * norm(fpt, dim, 0);
          alpha += input->norm_fs(dim) * norm(fpt, dim, 0);
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

        U(fpt, 0, 1) = PR / (input->R_ref * TR);

        Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VR[dim] = VR_mag * input->norm_fs(dim);
          U(fpt, dim+1, 1) = U(fpt, 0, 1) * VR[dim];
          Vsq += VR[dim] * VR[dim];
        }

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

        /* Set LDG bias */
        //LDG_bias(fpt) = -1;
        LDG_bias(fpt) = 0;

        break;
      }

      case 5: /* Subsonic Outlet */
      {
        if (!input->viscous)
          ThrowException("Subsonic outlet only for viscous flows currently!");

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
        U(fpt, nDims + 1, 1) = input->P_fs/(input->gamma-1.0) + 0.5 * momF; 

        /* Set LDG bias */
        //LDG_bias(fpt) = -1;
        LDG_bias(fpt) = 0;

        break;


        /* Get states for convenience */
        double rhoL = U(fpt, 0, 0);

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VL[dim] = U(fpt, dim+1, 0) / rhoL;
          Vsq += VL[dim] * VL[dim];
        }

        double eL = U(fpt, nDims + 1, 0);
        double PL = (input->gamma - 1.0) * (eL - 0.5 * rhoL * Vsq);

        /* Compute left normal velocity */
        double VnL = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += VL[dim] * norm(fpt, dim, 0);
        }

        /* Compute speed of sound */
        double cL = std::sqrt(input->gamma * PL / rhoL);

        /* Extrapolate Riemann invariant */
        double R_plus  = VnL + 2.0 * cL / (input->gamma - 1.0);

        /* Extrapolate entropy */
        double s = PL / std::pow(rhoL, input->gamma);

        /* Fix pressure */
        double PR = input->P_fs;

        U(fpt, 0, 1) = std::pow(PR / s, 1.0 / input->gamma);

        /* Compute right speed of sound and velocity magnitude */
        double cR = std::sqrt(input->gamma * PR/ U(fpt, 0, 1));

        double VnR = R_plus - 2.0 * cR / (input->gamma - 1.0);

        Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VR[dim] = VL[dim] + (VnR - VnL) * norm(fpt, dim, 0);
          U(fpt, dim+1, 1) = U(fpt, 0, 1) * VR[dim];
          Vsq += VR[dim] * VR[dim];
        }

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0) + 0.5 * U(fpt, 0, 1) * Vsq;

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
          VnR += input->V_fs(dim) * norm(fpt, dim, 0);
        }
      

        /* Compute pressure. TODO: Compute pressure once!*/
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
        }

        momF /= U(fpt, 0, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
        double PR = input->P_fs;

        /* Compute Riemann Invariants */
        double Rp = VnL + 2.0 / (input->gamma - 1) * std::sqrt(input->gamma * PL / 
            U(fpt, 0,0));
        double Rn = VnR - 2.0 / (input->gamma - 1) * std::sqrt(input->gamma * PR / 
            input->rho_fs);

        double cstar = 0.25 * (input->gamma - 1) * (Rp - Rn);
        double ustarn = 0.5 * (Rp + Rn);

        if (VnL < 0.0) /* Case 1: Inflow */
        {
          double s_inv = std::pow(input->rho_fs, input->gamma) / PR;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
            Vsq += input->V_fs(dim) * input->V_fs(dim);

          double H_fs = input->gamma / (input->gamma - 1.0) * PR / input->rho_fs +
              0.5 * Vsq;

          double rhoR = std::pow(1.0 / input->gamma * (s_inv * cstar * cstar), 1.0/ 
              (input-> gamma - 1.0));

          U(fpt, 0, 1) = rhoR;
          for (unsigned int dim = 0; dim < nDims; dim++)
            U(fpt, dim + 1, 1) = rhoR * (ustarn * norm(fpt, dim, 0) + input->V_fs(dim) - VnR * 
              norm(fpt, dim, 0));

          PR = rhoR / input->gamma * cstar * cstar;
          U(fpt, nDims + 1, 1) = rhoR * H_fs - PR;
          
        }
        else  /* Case 2: Outflow */
        {
          double rhoL = U(fpt, 0, 0);
          double s_inv = std::pow(rhoL, input->gamma) / PL;

          double rhoR = std::pow(1.0 / input->gamma * (s_inv * cstar * cstar), 1.0/ 
              (input-> gamma - 1.0));

          U(fpt, 0, 1) = rhoR;

          for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim + 1, 1) = rhoR * (ustarn * norm(fpt, dim, 0) +(U(fpt, dim + 1, 0) / 
                U(fpt, 0, 0) - VnL * norm(fpt, dim, 0)));

          double PR = rhoR / input->gamma * cstar * cstar;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
            Vsq += U(fpt, dim+1, 1) * U(fpt, dim+1, 1) / (rhoR * rhoR) ;
          
          U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0) + 0.5 * rhoR * Vsq; 
        }

        /* Set LDG bias */
        //LDG_bias(fpt) = -1;
        LDG_bias(fpt) = 0;

 
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

        /* Set energy */
        //U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0) - 0.5 * (momN * momN) / U(fpt, 0, 0);
        U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0);

        /* Set LDG bias */
        //LDG_bias(fpt) = -1;
        LDG_bias(fpt) = 0;

        break;
      }

      case 9: /* No-slip Wall (isothermal) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
        }

        momF /= U(fpt, 0, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1 , 0) - 0.5 * momF);

        double PR = PL;
        double TR = input->T_wall;
        
        U(fpt, 0, 1) = PR / (input->R_ref * TR);

        /* Set velocity to zero */
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim+1, 1) = 0.0;

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0);

        /* Set LDG bias */
        LDG_bias(fpt) = -1;


        break;
      }

      case 10: /* No-slip Wall (isothermal and moving) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
        }

        momF /= U(fpt, 0, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);

        double PR = PL;
        double TR = input->T_wall;
        
        U(fpt, 0, 1) = PR / (input->R_ref * TR);

        /* Set velocity to wall velocity */
        double V_wall_sq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(fpt, dim+1, 1) = U(fpt, 0 , 1) * input->V_wall(dim);
          V_wall_sq += input->V_wall(dim) * input->V_wall(dim);
        }

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0) + 0.5 * U(fpt, 0 , 1) * V_wall_sq;

        /* Set LDG bias */
        LDG_bias(fpt) = -1;

        break;
      }

      case 11: /* No-slip Wall (adiabatic) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        /* Extrapolate density */
        U(fpt, 0, 1) = U(fpt, 0, 0);

        /* Extrapolate pressure */
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
        }

        momF /= U(fpt, 0, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
        double PR = PL; 

        /* Set right state (common) velocity to zero */
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim+1, 1) = 0.0;
          //U(fpt, dim+1, 1) = -U(fpt, dim+1, 0);

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0);
        //U(fpt, 3, 1) = U(fpt, 3, 0);

        /* Set LDG bias */
        LDG_bias(fpt) = 1;

        break;
      }

      case 12: /* No-slip Wall (adiabatic and moving) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        /* Extrapolate density */
        U(fpt, 0, 1) = U(fpt, 0, 0);

        /* Extrapolate pressure */
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);
        }

        momF /= U(fpt, 0, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1, 0) - 0.5 * momF);
        double PR = PL; 

        /* Set velocity to wall velocity */
        double V_wall_sq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(fpt, dim+1, 1) = U(fpt, 0 , 1) * input->V_wall(dim);
          V_wall_sq += input->V_wall(dim) * input->V_wall(dim);
        }

        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1.0) + 0.5 * U(fpt, 0, 1) * V_wall_sq;

        /* Set LDG bias */
        LDG_bias(fpt) = 1;

        break;
      }
    
    } 
  }
#endif

#ifdef _GPU
  apply_bcs_wrapper(U_d, nFpts, geo->nGfpts_int, geo->nGfpts_bnd, nVars, nDims, input->rho_fs, input->V_fs_d, 
      input->P_fs, input->gamma, input->R_ref, input->T_tot_fs, input->P_tot_fs, input->T_wall, input->V_wall_d, 
      input->norm_fs_d, norm_d, geo->gfpt2bnd_d, geo->per_fpt_list_d, LDG_bias_d, input->equation);

  check_error();

  //U = U_d;
  //LDG_bias = LDG_bias_d;
#endif

}

void Faces::apply_bcs_dU()
{
#ifdef _CPU
  /* Apply boundaries to solution derivative */
#pragma omp parallel for 
  //for (unsigned int fpt = geo->nGfpts_int; fpt < nFpts; fpt++)
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    if (bnd_id == 1) /* Periodic */
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
            unsigned int per_fpt = geo->per_fpt_list(fpt - geo->nGfpts_int);
            dU(fpt, n, dim, 1) = dU(per_fpt, n, dim, 0);
        }
      }
    }
    else if(bnd_id == 11 || bnd_id == 12) /* Adibatic Wall */
    {
      /* Extrapolate density gradient */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dU(fpt, 0, dim, 1) = dU(fpt, 0, dim, 0);
      }

      if (nDims == 2)
      {
        /* Compute energy gradient */
        /* Get left states and velocity gradients*/
        double rho = U(fpt, 0, 0);
        double momx = U(fpt, 1, 0);
        double momy = U(fpt, 2, 0);
        double E = U(fpt, 3, 0);

        double u = momx / rho;
        double v = momy / rho;


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
        //double du_dn = du_dx * norm(fpt, 0, 0) + du_dy * norm(fpt, 1, 0);
        //double dv_dn = dv_dx * norm(fpt, 0, 0) + dv_dy * norm(fpt, 1, 0);

        //dU(fpt, 1, 0, 1) = rho * du_dn * norm(fpt, 0, 0);
        //dU(fpt, 1, 1, 1) = rho * du_dn * norm(fpt, 1, 0);
        //dU(fpt, 2, 0, 1) = rho * dv_dn * norm(fpt, 0, 0);
        //dU(fpt, 2, 1, 1) =  rho * dv_dn * norm(fpt, 1, 0);

        // double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
        // double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

        /* Compute temperature gradient (actually C_v * rho * dT) */
        double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx);
        double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy);

        /* Compute wall normal temperature gradient */
        double dT_dn = dT_dx * norm(fpt, 0, 0) + dT_dy * norm(fpt, 1, 0);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(fpt, 3, 0, 1) = E_dx - dT_dn * norm(fpt, 0, 0);
        dU(fpt, 3, 1, 1) = E_dy - dT_dn * norm(fpt, 1, 0);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 3, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0, 0)) + rho_dx * U(fpt, 3, 1) / rho; 
        //dU(fpt, 3, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1, 0)) + rho_dy * U(fpt, 3, 1) / rho; 
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
        //double du_dn = du_dx * norm(fpt, 0, 0) + du_dy * norm(fpt, 1, 0) + du_dz * norm(fpt, 2, 0);
        //double dv_dn = dv_dx * norm(fpt, 0, 0) + dv_dy * norm(fpt, 1, 0) + dv_dz * norm(fpt, 2, 0);
        //double dw_dn = dw_dx * norm(fpt, 0, 0) + dw_dy * norm(fpt, 1, 0) + dw_dz * norm(fpt, 2, 0);

        //dU(fpt, 1, 0, 1) = rho * du_dn * norm(fpt, 0, 0);
        //dU(fpt, 1, 1, 1) = rho * du_dn * norm(fpt, 1, 0);
        //dU(fpt, 1, 2, 1) = rho * du_dn * norm(fpt, 2, 0);
        //dU(fpt, 2, 0, 1) = rho * dv_dn * norm(fpt, 0, 0);
        //dU(fpt, 2, 1, 1) = rho * dv_dn * norm(fpt, 1, 0);
        //dU(fpt, 2, 2, 1) = rho * dv_dn * norm(fpt, 2, 0);
        //dU(fpt, 3, 0, 1) = rho * dw_dn * norm(fpt, 0, 0);
        //dU(fpt, 3, 1, 1) = rho * dw_dn * norm(fpt, 1, 0);
        //dU(fpt, 3, 2, 1) = rho * dw_dn * norm(fpt, 2, 0);

       // double dke_dx = 0.5 * (u*u + v*v + w*w) * rho_dx + rho * (u * du_dx + v * dv_dx + w * dw_dx);
       // double dke_dy = 0.5 * (u*u + v*v + w*w) * rho_dy + rho * (u * du_dy + v * dv_dy + w * dw_dy);
       // double dke_dz = 0.5 * (u*u + v*v + w*w) * rho_dz + rho * (u * du_dz + v * dv_dz + w * dw_dz);

        /* Compute temperature gradient (actually C_v * rho * dT) */
        double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx + w * dw_dx);
        double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy + w * dw_dy);
        double dT_dz = E_dz - rho_dz * E/rho - rho * (u * du_dz + v * dv_dz + w * dw_dz);

        /* Compute wall normal temperature gradient */
        double dT_dn = dT_dx * norm(fpt, 0, 0) + dT_dy * norm(fpt, 1, 0) + dT_dz * norm(fpt, 2, 0);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(fpt, 4, 0, 1) = E_dx - dT_dn * norm(fpt, 0, 0);
        dU(fpt, 4, 1, 1) = E_dy - dT_dn * norm(fpt, 1, 0);
        dU(fpt, 4, 2, 1) = E_dz - dT_dn * norm(fpt, 2, 0);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 4, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0, 0)) + rho_dx * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1, 0)) + rho_dy * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 2, 1) = (dT_dz - dT_dn * norm(fpt, 2, 0)) + rho_dz * U(fpt, 4, 1) / rho; 
      }

    }
    else /* Otherwise, right state gradient equals left state gradient */
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
            dU(fpt, n, dim, 1) = dU(fpt, n, dim, 0);
        }
      }
    }
  }
#endif

#ifdef _GPU
  apply_bcs_dU_wrapper(dU_d, U_d, norm_d, nFpts, geo->nGfpts_int, geo->nGfpts_bnd, nVars, 
      nDims, geo->gfpt2bnd_d, geo->per_fpt_list_d);

  check_error();
  
  //dU = dU_d;
#endif
}


void Faces::compute_Fconv(unsigned int startFpt, unsigned int endFpt)
{  
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        Fconv(fpt, 0, dim, 0) = input->AdvDiff_A(dim) * U(fpt, 0, 0);
        Fconv(fpt, 0, dim, 1) = input->AdvDiff_A(dim) * U(fpt, 0, 1);
      }
    }
#endif

#ifdef _GPU
    compute_Fconv_fpts_AdvDiff_wrapper(Fconv_d, U_d, nFpts, nDims, input->AdvDiff_A_d,
        startFpt, endFpt);
    check_error();
#endif
  }
  else if (input->equation == EulerNS)
  {
#ifdef _CPU
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      //for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
      {
        for (unsigned int slot = 0; slot < 2; slot ++)
        {
          /* Compute some primitive variables (keep pressure)*/
          double momF = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim ++)
          {
            momF += U(fpt, dim + 1, slot) * U(fpt, dim + 1, slot);
          }

          momF /= U(fpt, 0, slot);

          P(fpt, slot) = (input->gamma - 1.0) * (U(fpt, 3, slot) - 0.5 * momF);
          double H = (U(fpt, 3, slot) + P(fpt, slot)) / U(fpt, 0, slot);

          Fconv(fpt, 0, 0, slot) = U(fpt, 1, slot);
          Fconv(fpt, 1, 0, slot) = U(fpt, 1, slot) * U(fpt, 1, slot) / U(fpt, 0, slot) + P(fpt, slot);
          Fconv(fpt, 2, 0, slot) = U(fpt, 1, slot) * U(fpt, 2, slot) / U(fpt, 0, slot);
          Fconv(fpt, 3, 0, slot) = U(fpt, 1, slot) * H;

          Fconv(fpt, 0, 1, slot) = U(fpt, 2, slot);
          Fconv(fpt, 1, 1, slot) = U(fpt, 2, slot) * U(fpt, 1, slot) / U(fpt, 0, slot);
          Fconv(fpt, 2, 1, slot) = U(fpt, 2, slot) * U(fpt, 2, slot) / U(fpt, 0, slot) + P(fpt, slot);
          Fconv(fpt, 3, 1, slot) = U(fpt, 2, slot) * H;
        }
      }
    }
    else if (nDims == 3)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
      {
        for (unsigned int slot = 0; slot < 2; slot ++)
        {
          /* Compute some primitive variables (keep pressure)*/
          double momF = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim ++)
          {
            momF += U(fpt, dim + 1, slot) * U(fpt, dim + 1, slot);
          }

          momF /= U(fpt, 0, slot);

          P(fpt, slot) = (input->gamma - 1.0) * (U(fpt, 4, slot) - 0.5 * momF);
          double H = (U(fpt, 4, slot) + P(fpt, slot)) / U(fpt, 0, slot);

          Fconv(fpt, 0, 0, slot) = U(fpt, 1, slot);
          Fconv(fpt, 1, 0, slot) = U(fpt, 1, slot) * U(fpt, 1, slot) / U(fpt, 0, slot) + P(fpt, slot);
          Fconv(fpt, 2, 0, slot) = U(fpt, 1, slot) * U(fpt, 2, slot) / U(fpt, 0, slot);
          Fconv(fpt, 3, 0, slot) = U(fpt, 1, slot) * U(fpt, 3, slot) / U(fpt, 0, slot);
          Fconv(fpt, 4, 0, slot) = U(fpt, 1, slot) * H;

          Fconv(fpt, 0, 1, slot) = U(fpt, 2, slot);
          Fconv(fpt, 1, 1, slot) = U(fpt, 2, slot) * U(fpt, 1, slot) / U(fpt, 0, slot);
          Fconv(fpt, 2, 1, slot) = U(fpt, 2, slot) * U(fpt, 2, slot) / U(fpt, 0, slot) + P(fpt, slot);
          Fconv(fpt, 3, 1, slot) = U(fpt, 2, slot) * U(fpt, 3, slot) / U(fpt, 0, slot);
          Fconv(fpt, 4, 1, slot) = U(fpt, 2, slot) * H;

          Fconv(fpt, 0, 2, slot) = U(fpt, 3, slot);
          Fconv(fpt, 1, 2, slot) = U(fpt, 3, slot) * U(fpt, 1, slot) / U(fpt, 0, slot);
          Fconv(fpt, 2, 2, slot) = U(fpt, 3, slot) * U(fpt, 2, slot) / U(fpt, 0, slot);
          Fconv(fpt, 3, 2, slot) = U(fpt, 3, slot) * U(fpt, 3, slot) / U(fpt, 0, slot) + P(fpt, slot);
          Fconv(fpt, 4, 2, slot) = U(fpt, 3, slot) * H;
        }
      }
    }
#endif

#ifdef _GPU
      compute_Fconv_fpts_EulerNS_wrapper(Fconv_d, U_d, P_d, nFpts, nDims, input->gamma, 
          startFpt, endFpt);
      check_error();
#endif
  }

}

void Faces::compute_Fvisc(unsigned int startFpt, unsigned int endFpt)
{  
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fvisc(fpt, n, dim, 0) = -input->AdvDiff_D * dU(fpt, n, dim, 0);

          Fvisc(fpt, n, dim, 1) = -input->AdvDiff_D * dU(fpt, n, dim, 1);

        }
      }
    }
#endif

#ifdef _GPU
    compute_Fvisc_fpts_AdvDiff_wrapper(Fvisc_d, dU_d, nFpts, nDims, input->AdvDiff_D,
        startFpt, endFpt);
    check_error();
#endif

  }
  else if (input->equation == EulerNS)
  {
#ifdef _CPU
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
      {
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
          Fvisc(fpt, 0, 0, slot) = 0.0;
          Fvisc(fpt, 1, 0, slot) = -tauxx;
          Fvisc(fpt, 2, 0, slot) = -tauxy;
          Fvisc(fpt, 3, 0, slot) = -(u * tauxx + v * tauxy + (mu / input->prandtl) *
              input-> gamma * de_dx);

          Fvisc(fpt, 0, 1, slot) = 0.0;
          Fvisc(fpt, 1, 1, slot) = -tauxy;
          Fvisc(fpt, 2, 1, slot) = -tauyy;
          Fvisc(fpt, 3, 1, slot) = -(u * tauxy + v * tauyy + (mu / input->prandtl) *
              input->gamma * de_dy);
        }
      }
    }
    else if (nDims == 3)
    {
      for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
      {
        for (unsigned int slot = 0; slot < 2; slot++)
        {
          /* Setting variables for convenience */
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
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + input->c_sth);
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
          Fvisc(fpt, 4, 0, slot) = -(u * tauxx + v * tauxy + w * tauxz + (mu / input->prandtl) *
              input-> gamma * de_dx);

          Fvisc(fpt, 0, 1, slot) = 0;
          Fvisc(fpt, 1, 1, slot) = -tauxy;
          Fvisc(fpt, 2, 1, slot) = -tauyy;
          Fvisc(fpt, 3, 1, slot) = -tauyz;
          Fvisc(fpt, 4, 1, slot) = -(u * tauxy + v * tauyy + w * tauyz + (mu / input->prandtl) *
              input->gamma * de_dy);

          Fvisc(fpt, 0, 2, slot) = 0;
          Fvisc(fpt, 1, 2, slot) = -tauxz;
          Fvisc(fpt, 2, 2, slot) = -tauyz;
          Fvisc(fpt, 3, 2, slot) = -tauzz;
          Fvisc(fpt, 4, 2, slot) = -(u * tauxz + v * tauyz + w * tauzz + (mu / input->prandtl) *
              input->gamma * de_dz);
        }
      }
    }
#endif

#ifdef _GPU
    compute_Fvisc_fpts_EulerNS_wrapper(Fvisc_d, U_d, dU_d, nFpts, nDims, input->gamma, 
        input->prandtl, input->mu, input->c_sth, input->rt, input->fix_vis,
        startFpt, endFpt);
    check_error();

    //Fvisc = Fvisc_d;
#endif
  }
}

void Faces::compute_common_F(unsigned int startFpt, unsigned int endFpt)
{
  if (input->fconv_type == "Rusanov")
  {
#ifdef _CPU
    rusanov_flux(startFpt, endFpt);
#endif

#ifdef _GPU
    rusanov_flux_wrapper(U_d, Fconv_d, Fcomm_d, P_d, input->AdvDiff_A_d, norm_d, waveSp_d, LDG_bias_d, 
        input->gamma, input->rus_k, nFpts, nVars, nDims, input->equation, startFpt, endFpt);

    check_error();

    //Fcomm = Fcomm_d;
    //waveSp = waveSp_d;
#endif
  }
  else if (input->fconv_type == "Roe")
  {
#ifdef _CPU
    roe_flux(startFpt, endFpt);
#endif

#ifdef _GPU
    roe_flux_wrapper(U_d, Fconv_d, Fcomm_d, norm_d, waveSp_d, input->gamma, 
        nFpts, nVars, nDims, input->equation, startFpt, endFpt);

    check_error();
#endif
  }
  else
  {
    ThrowException("Numerical convective flux type not recognized!");
  }

  if (input->viscous)
  {
    if (input->fvisc_type == "LDG")
    {
#ifdef _CPU
      LDG_flux(startFpt, endFpt);
#endif

#ifdef _GPU
      LDG_flux_wrapper(U_d, Fvisc_d, Fcomm_d, Fcomm_temp_d, norm_d, LDG_bias_d, input->ldg_b,
          input->ldg_tau, nFpts, nVars, nDims, input->equation, startFpt, endFpt);

      check_error();
#endif
    }
    /*
    else if (input->fvisc_type == "Central")
      central_flux();
    */
    else
    {
      ThrowException("Numerical viscous flux type not recognized!");
    }
  }

  //transform_flux();
}

void Faces::compute_common_U(unsigned int startFpt, unsigned int endFpt)
{
  
  /* Compute common solution */
  if (input->fvisc_type == "LDG")
  {
#ifdef _CPU
#pragma omp parallel for 
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      double beta = input->ldg_b;

      /* Setting sign of beta (from HiFiLES) */
      if (nDims == 2)
      {
        if (norm(fpt, 0, 0) + norm(fpt, 1, 0) < 0.0)
          beta = -beta;
      }
      else if (nDims == 3)
      {
        if (norm(fpt, 0, 0) + norm(fpt, 1, 0) + sqrt(2.) * norm(fpt, 2, 0) < 0.0)
          beta = -beta;
      }

      /* Get left and right state variables */
      // TODO: Verify that this is the correct formula. Seem different than papers...
      /* If interior, allow use of beta factor */
      if (LDG_bias(fpt) == 0)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          double UL = U(fpt, n, 0); double UR = U(fpt, n, 1);

           Ucomm(fpt, n, 0) = 0.5*(UL + UR) - beta*(UL - UR);
           Ucomm(fpt, n, 1) = 0.5*(UL + UR) - beta*(UL - UR);
        }
      }
      /* If on (non-periodic) boundary, don't use beta (this is from HiFILES. Need to check) */
      /* If on (non-periodic) boundary, set right state as common (strong) */
      else
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          //double UL = U(fpt, n, 0); 
          double UR = U(fpt, n, 1);

          Ucomm(fpt, n, 0) = UR;
          Ucomm(fpt, n, 1) = UR;

           //Ucomm(fpt, n, 0) = 0.5*(UL + UR);
           //Ucomm(fpt, n, 1) = 0.5*(UL + UR);
        }
      }

    }
#endif

#ifdef _GPU
    compute_common_U_LDG_wrapper(U_d, Ucomm_d, norm_d, input->ldg_b, nFpts, nVars, nDims, LDG_bias_d, 
        startFpt, endFpt);

    check_error();

    //Ucomm = Ucomm_d;
#endif
  }

  // TODO: Can potentially remove central treatment since LDG recovers.
  /*
  else if (input->fvisc_type == "Central")
  {
#pragma omp parallel for collapse(2)
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UL = U(fpt, n, 0); double UR = U(fpt, n, 1);

        Ucomm(fpt, n, 0) = 0.5*(UL + UR);
        Ucomm(fpt, n, 1) = 0.5*(UL + UR);
      }

    }
  }
*/
  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }

}

void Faces::rusanov_flux(unsigned int startFpt, unsigned int endFpt)
{

  double k = input->rus_k;

  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);

#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fconv(fpt, n, dim, 0) * norm(fpt, dim, 0);
        FR[n] += Fconv(fpt, n, dim, 1) * norm(fpt, dim, 0);
      }
    }

    if (LDG_bias(fpt) != 0)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm(fpt, n, 0) = FR[n];
        Fcomm(fpt, n, 1) = FR[n];
      }
      continue;
    }

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
    }

    /* Get numerical wavespeed */
    if (input->equation == AdvDiff)
    {
      double An = 0.;

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        An += input->AdvDiff_A(dim) * norm(fpt, dim, 0);
      }

      waveSp(fpt) = std::abs(An);
    }
    else if (input->equation == EulerNS)
    {
      /* Compute speed of sound */
      double aL = std::sqrt(std::abs(input->gamma * P(fpt, 0) / WL[0]));
      double aR = std::sqrt(std::abs(input->gamma * P(fpt, 1) / WR[0]));

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += WL[dim+1]/WL[0] * norm(fpt, dim, 0);
        VnR += WR[dim+1]/WR[0] * norm(fpt, dim, 0);
      }

      waveSp(fpt) = std::max(std::abs(VnL) + aL, std::abs(VnR) + aR);
    }

    /* Compute common normal flux */
    for (unsigned int n = 0; n < nVars; n++)
    {
      double F = 0.5 * (FR[n]+FL[n]) - 0.5 * waveSp(fpt) * (1.0-k) * (WR[n]-WL[n]);

      /* Correct for positive parent space sign convention */
      Fcomm(fpt, n, 0) = F;
      Fcomm(fpt, n, 1) = F;
    }
  }
}

void Faces::roe_flux(unsigned int startFpt, unsigned int endFpt)
{
  if (nDims != 2)
  {
    ThrowException("Roe Flux only implemented for 2D!");
  }

  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> F(nVars);
  std::vector<double> dW(nVars);

#pragma omp parallel for firstprivate(FL, FR, F, dW)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        FL[n] += Fconv(fpt, n, dim, 0) * norm(fpt, dim, 0);
        FR[n] += Fconv(fpt, n, dim, 1) * norm(fpt, dim, 0);
      }
    }

    /* Get difference in state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      dW[n] = U(fpt, n, 1) - U(fpt, n, 0);
    }

    /* Get numerical wavespeed */
    if (input->equation == EulerNS)
    {
      /* Primitive Variables */
      double gam = input->gamma;
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
      double Vnm = um * norm(fpt, 0, 0) + vm * norm(fpt, 1, 0);

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
      double a6 = Vnm * dW[0] - norm(fpt, 0, 0) * dW[1] - norm(fpt, 1, 0) * dW[2];
      double aL1 = a1 * a5 - a3 * a6;
      double bL1 = a4 * a5 - a2 * a6;

      F[0] = 0.5 * (FR[0] + FL[0]) - (lambda0 * dW[0] + aL1);
      F[1] = 0.5 * (FR[1] + FL[1]) - (lambda0 * dW[1] + aL1 * um + bL1 * norm(fpt, 0, 0));
      F[2] = 0.5 * (FR[2] + FL[2]) - (lambda0 * dW[2] + aL1 * vm + bL1 * norm(fpt, 1, 0));
      F[3] = 0.5 * (FR[3] + FL[3]) - (lambda0 * dW[3] + aL1 * hm + bL1 * Vnm);

      waveSp(fpt) = std::max(std::max(lambda0, lambdaP), lambdaM);
    }
    else
    {
      ThrowException("Roe flux not implemented for this equation type!");
    }

    /* Correct for positive parent space sign convention */
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(fpt, n, 0) = F[n];
      Fcomm(fpt, n, 1) = F[n];
    }
  }
}

void Faces::transform_flux()
{
#ifdef _CPU
#pragma omp parallel for collapse(2)
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(fpt, n, 0) *= dA(fpt);
      Fcomm(fpt, n, 1) *= -dA(fpt); // Right state flux has opposite sign
    }
  }
#endif

#ifdef _GPU
  //Fcomm_d = Fcomm;

  transform_flux_faces_wrapper(Fcomm_d, dA_d, nFpts, nVars);

  check_error();

  //Fcomm = Fcomm_d;
#endif

}

void Faces::LDG_flux(unsigned int startFpt, unsigned int endFpt)
{
  std::vector<double> FL(nVars);
  std::vector<double> FR(nVars);
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);
   
  double tau = input->ldg_tau;

  Fcomm_temp.fill(0.0);

#pragma omp parallel for firstprivate(FL, FR, WL, WR)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    double beta = input->ldg_b;

    /* Setting sign of beta (from HiFiLES) */
    if (nDims == 2)
    {
      if (norm(fpt, 0, 0) + norm(fpt, 1, 0) < 0.0)
        beta = -beta;
    }
    else if (nDims == 3)
    {
      if (norm(fpt, 0, 0) + norm(fpt, 1, 0) + sqrt(2.) * norm(fpt, 2, 0) < 0.0)
        beta = -beta;
    }

    /* Initialize FL, FR */
    std::fill(FL.begin(), FL.end(), 0.0);
    std::fill(FR.begin(), FR.end(), 0.0);

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
    /* If interior, use central */
    if (LDG_bias(fpt) == 0)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          Fcomm_temp(fpt, n, dim) += 0.5*(Fvisc(fpt, n, dim, 0) + Fvisc(fpt, n, dim, 1)) + tau * norm(fpt, dim, 0)* (WL[n]
              - WR[n]) + beta * norm(fpt, dim, 0)* (FL[n] - FR[n]);
        }
      }
    }
    /* If Neumann boundary, use right state only */
    else
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          //Fcomm_temp(fpt, n, dim) += Fvisc(fpt, n, dim, 1);
          Fcomm_temp(fpt, n, dim) += Fvisc(fpt, n, dim, 1) + tau * norm(fpt, dim, 0)* (WL[n] - WR[n]);
        }
      }
    }

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Fcomm(fpt, n, 0) += (Fcomm_temp(fpt, n, dim) * norm(fpt, dim, 0));
        Fcomm(fpt, n, 1) += (Fcomm_temp(fpt, n, dim) * norm(fpt, dim, 0));
      }
    }

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
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(fpt, n, 0) += (0.5 * (FL[n]+FR[n]));
      Fcomm(fpt, n, 1) += (0.5 * (FL[n]+FR[n]));
    }
  }
}

#ifdef _MPI
void Faces::send_U_data()
{
  /* Stage all the non-blocking receives */
  unsigned int ridx = 0;

#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, MPI_COMM_WORLD, &rreqs[ridx]);
    ridx++;
  }
#endif
#ifdef _GPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, MPI_COMM_WORLD, &rreqs[ridx]);
    ridx++;
  }
#endif


  unsigned int sidx = 0;
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    const auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int i = 0; i < fpts.size(); i++)
      {
        U_sbuffs[sendRank](i, n) = U(fpts(i), n, 0);
      }

    }

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, MPI_COMM_WORLD, &sreqs[sidx]);
    sidx++;
  }
#endif
#ifdef _GPU
  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    pack_U_wrapper(U_sbuffs_d[sendRank], fpts, U_d, nVars);
  }

  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    U_sbuffs[pairedRank] = U_sbuffs_d[pairedRank];
  }

  for (auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, MPI_COMM_WORLD, &sreqs[sidx]);
    sidx++;
  }
#endif
}

void Faces::recv_U_data()
{
  /* Wait for comms to finish */
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

  /* Unpack buffer */
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int i = 0; i < fpts.size(); i++)
      {
        U(fpts(i), n, 1) = U_rbuffs[recvRank](i, n);
      }
    }
  }
#endif

#ifdef _GPU
  /* Copy buffer to device (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    U_rbuffs_d[pairedRank] = U_rbuffs[pairedRank];
  }

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;

    unpack_U_wrapper(U_rbuffs_d[recvRank], fpts, U_d, nVars);
  }
#endif

}

void Faces::send_dU_data()
{
  /* Stage all the non-blocking receives */
  unsigned int ridx = 0;
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, MPI_COMM_WORLD, &rreqs[ridx]);
    ridx++;
  }
#endif
#ifdef _GPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, MPI_COMM_WORLD, &rreqs[ridx]);
    ridx++;
  }
#endif

  unsigned int sidx = 0;
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    const auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int i = 0; i < fpts.size(); i++)
        {
          U_sbuffs[sendRank](i, n, dim) = dU(fpts(i), n, dim, 0);
        }
      }
    }

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, MPI_COMM_WORLD, &sreqs[sidx]);
    sidx++;
  }
#endif
#ifdef _GPU
  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    pack_dU_wrapper(U_sbuffs_d[sendRank], fpts, dU_d, nVars, nDims);
  }

  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    U_sbuffs[pairedRank] = U_sbuffs_d[pairedRank];
  }

  for (auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, MPI_COMM_WORLD, &sreqs[sidx]);
    sidx++;
  }

#endif
}

void Faces::recv_dU_data()
{
  /* Wait for comms to finish */
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

  /* Unpack buffer */
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int i = 0; i < fpts.size(); i++)
        {
          dU(fpts(i), n, dim, 1) = U_rbuffs[recvRank](i, n, dim);
        }
      }
    }
  }
#endif
#ifdef _GPU
  /* Copy buffer to device (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    U_rbuffs_d[pairedRank] = U_rbuffs[pairedRank];
  }

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;

    unpack_dU_wrapper(U_rbuffs_d[recvRank], fpts, dU_d, nVars, nDims);
  }
#endif
}
#endif



