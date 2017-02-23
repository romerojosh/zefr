/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "input.hpp"
#include "faces_kernels.h"
#include "flux.hpp"
#include "mdvector_gpu.h"

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void apply_bcs(mdview_gpu<double> U, mdview_gpu<double> U_ldg, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, double rho_fs, 
    const mdvector_gpu<double> V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, const mdvector_gpu<double> V_wall, mdvector_gpu<double> Vg,
    const mdvector_gpu<double> norm_fs, const mdvector_gpu<double> norm,
    const mdvector_gpu<char> gfpt2bnd, mdvector_gpu<char> rus_bias, mdvector_gpu<char> LDG_bias, bool motion = false)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  switch(bnd_id)
  {
    case SUP_IN: /* Farfield and Supersonic Inlet */
    {
      if (equation == AdvDiff)
      {
        /* Set boundaries to zero */
        U(1, 0, fpt) = 0;
        U_ldg(1, 0, fpt) = 0;
      }
      else
      {
        /* Set boundaries to freestream values */
        U(1, 0, fpt) = rho_fs;
        U_ldg(1, 0, fpt) = rho_fs;

        double Vsq = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(1, dim+1, fpt) = rho_fs * V_fs(dim);
          U_ldg(1, dim+1, fpt) = rho_fs * V_fs(dim);
          Vsq += V_fs(dim) * V_fs(dim);
        }

        U_ldg(1, nDims + 1, fpt) = P_fs/(gamma-1.0) + 0.5*rho_fs * Vsq; 
        U(1, nDims + 1, fpt) = P_fs/(gamma-1.0) + 0.5*rho_fs * Vsq; 
      }

      break;
    }

    case SUP_OUT: /* Supersonic Outlet */
    {
      /* Extrapolate boundary values from interior */
      for (unsigned int n = 0; n < nVars; n++)
      {
        U(1, n, fpt) = U(0, n, fpt);
        U_ldg(1, n, fpt) = U(0, n, fpt);
      }

      break;
    }

    case SUB_IN: /* Subsonic Inlet */
    {
      /* TODO: implement */
      break;
    }

    case SUB_OUT: /* Subsonic Outlet */
    { 
      /* Extrapolate Density */
      U(1, 0, fpt) = U(0, 0, fpt);
      U_ldg(1, 0, fpt) = U(0, 0, fpt);

      /* Extrapolate Momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(1, dim+1, fpt) =  U(0, dim+1, fpt);
        U_ldg(1, dim+1, fpt) =  U(0, dim+1, fpt);
      }

      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(0, dim + 1, fpt) * U(0, dim + 1, fpt);
      }

      momF /= U(0, 0, fpt);

      /* Fix pressure */
      U(1, nDims + 1, fpt) = P_fs/(gamma-1.0) + 0.5 * momF; 
      U_ldg(1, nDims + 1, fpt) = P_fs/(gamma-1.0) + 0.5 * momF; 

      break;
    }

    case CHAR: /* Characteristic (from PyFR) */
    case CHAR_P: /* Characteristic (prescribed) */
    {
      /* Compute wall normal velocities */
      double VnL = 0.0; double VnR = 0.0;

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += U(0, dim+1, fpt) / U(0, 0, fpt) * norm(dim, fpt);
        VnR += V_fs(dim) * norm(dim, fpt);
      }

      /* Compute pressure. TODO: Compute pressure once!*/
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        momF += U(0, dim + 1, fpt) * U(0, dim + 1, fpt);
      }

      momF /= U(0, 0, fpt);

      double PL = (gamma - 1.0) * (U(0, nDims + 1, fpt) - 0.5 * momF);
      double PR = P_fs;

      double cL = std::sqrt(gamma * PL / U(0, 0, fpt));
      double cR = std::sqrt(gamma * PR / rho_fs);

      /* Compute Riemann Invariants */
      double RL;
      if (std::abs(VnR) >= cR && VnL >= 0)
        RL = VnR + 2.0 / (gamma - 1) * cR;
      else
        RL = VnL + 2.0 / (gamma - 1) * cL;

      double RB;
      if (std::abs(VnR) >= cR && VnL < 0)
        RB = VnL - 2.0 / (gamma - 1) * cL;
      else
        RB = VnR - 2.0 / (gamma - 1) * cR;

      double cstar = 0.25 * (gamma - 1) * (RL - RB);
      double ustarn = 0.5 * (RL + RB);

      double rhoR = cstar * cstar / gamma;
      double VR[3] = {0, 0, 0};

      if (VnL < 0.0) /* Case 1: Inflow */
      {
        rhoR *= pow(rho_fs, gamma) / PR;

        for (unsigned int dim = 0; dim < nDims; dim++)
          VR[dim] = V_fs(dim) + (ustarn - VnR) * norm(dim, fpt);
      }
      else  /* Case 2: Outflow */
      {
        rhoR *= pow(U(0, 0, fpt), gamma) / PL;

        for (unsigned int dim = 0; dim < nDims; dim++)
          VR[dim] = U(0, dim+1, fpt) / U(0, 0, fpt) + (ustarn - VnL) * norm(dim, fpt);
      }

      rhoR = std::pow(rhoR, 1.0 / (gamma - 1));

      U(1, 0, fpt) = rhoR;
      U_ldg(1, 0, fpt) = rhoR;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(1, dim + 1, fpt) = rhoR * VR[dim];
        U_ldg(1, dim + 1, fpt) = rhoR * VR[dim];
      }

      PR = rhoR / gamma * cstar * cstar;
      U(1, nDims + 1, fpt) = PR / (gamma - 1);
      U_ldg(1, nDims + 1, fpt) = PR / (gamma - 1);
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        U(1, nDims+1, fpt) += 0.5 * rhoR * VR[dim] * VR[dim];
        U_ldg(1, nDims+1, fpt) += 0.5 * rhoR * VR[dim] * VR[dim];
      }

      /* Set Char (prescribed) */
      if (bnd_id == CHAR_P)
      {
        rus_bias(fpt) = 1;
      }

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case SYMMETRY_P: /* Symmetry (prescribed) */
    case SLIP_WALL_P: /* Slip Wall (prescribed) */
    {
      double momN = 0.0;

      /* Compute wall normal momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
        momN += U(0, dim+1, fpt) * norm(dim, fpt);

      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN -= U(0, 0, fpt) * Vg(fpt, dim) * norm(dim, fpt);
      }

      U(1, 0, fpt) = U(0, 0, fpt);

      /* Set boundary state with cancelled normal velocity */
      for (unsigned int dim = 0; dim < nDims; dim++)
        U(1, dim+1, fpt) = U(0, dim+1, fpt) - momN * norm(dim, fpt);

      /* Set energy */
      /* Get left-state pressure */
      double momFL = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momFL += U(0, dim + 1, fpt) * U(0, dim + 1, fpt);

      double PL = (gamma - 1.0) * (U(0, nDims + 1 , fpt) - 0.5 * momFL / U(0, 0, fpt));

      /* Get right-state momentum flux after velocity correction */
      double momFR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momFR += U(1, dim + 1, fpt) * U(1, dim + 1, fpt);

      /* Recompute energy with extrapolated pressure and new momentum */
      U(1, nDims + 1, fpt) = PL / (gamma - 1)  + 0.5 * momFR / U(1, 0, fpt);

      /* Set bias */
      rus_bias(fpt) = 1;

      break;
    }

    case SYMMETRY_G: /* Symmetry (ghost) */
    case SLIP_WALL_G: /* Slip Wall (ghost) */
    {
      double momN = 0.0;

      /* Compute wall normal momentum */
      for (unsigned int dim = 0; dim < nDims; dim++)
        momN += U(0, dim+1, fpt) * norm(dim, fpt);

      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN -= U(0, 0, fpt) * Vg(fpt, dim) * norm(dim, fpt);
      }

      U(1, 0, fpt) = U(0, 0, fpt);

      for (unsigned int dim = 0; dim < nDims; dim++)
        /* Set boundary state to reflect normal velocity */
        U(1, dim+1, fpt) = U(0, dim+1, fpt) - 2.0 * momN * norm(dim, fpt);

      U(1, nDims + 1, fpt) = U(0, nDims + 1, fpt);

      break;
    }


    case ISOTHERMAL_NOSLIP_P: /* Isothermal No-slip Wall (prescribed) */
    {
      double VG[nVars] = {0.0};

      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          VG[dim] = Vg(fpt, dim);
      }

      double rhoL = U(0, 0, fpt);

      U(1, 0, fpt) = rhoL;
      U_ldg(1, 0, fpt) = rhoL;

      /* Set velocity to zero (or grid wall velocity) */
      double Vsq = 0; double Vsq_grid = 0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        double VL = U(0, dim+1, fpt) / rhoL;
        double V = -VL + 2 * VG[dim];
        U(1, dim+1, fpt) = rhoL * V;
        Vsq += V * V;

        U_ldg(1, dim+1, fpt) =  VG[dim];
        Vsq_grid += VG[dim] * VG[dim];
      }
        
      double cp_over_gam =  R_ref / (gamma - 1);

      U(1, nDims + 1, fpt) = rhoL * (cp_over_gam * T_wall + 0.5 * Vsq);
      U_ldg(1, nDims + 1, fpt) = rhoL * cp_over_gam * T_wall;

      /* Set bias */
      LDG_bias(fpt) = 1;
      break;
    }

    case ISOTHERMAL_NOSLIP_G: /* Isothermal No-slip Wall (ghost) */
    {
      // NOT IMPLEMENTED
      break;
    }


    case ISOTHERMAL_NOSLIP_MOVING_P: /* Moving Isothermal No-slip Wall (prescribed) */
    {
      double rhoL = U(0, 0, fpt);

      U(1, 0, fpt) = rhoL;
      U_ldg(1, 0, fpt) = rhoL;

      /* Set velocity to zero (or wall velocity) */
      double Vsq = 0; double Vsq_wall = 0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        double VL = U(0, dim+1, fpt) / U(0, 0, fpt);
        double V = -VL + 2*(V_wall(dim));
        U(1, dim+1, fpt) = rhoL * V;
        Vsq += V * V;

        U_ldg(1, dim+1, fpt) = rhoL * V_wall(dim);
        Vsq_wall += V_wall(dim) * V_wall(dim);
      }
        
      double cp_over_gam = R_ref / (gamma - 1);

      U(1, nDims + 1, fpt) = rhoL * (cp_over_gam * T_wall + 0.5 * Vsq);
      U_ldg(1, nDims + 1, fpt) = rhoL * (cp_over_gam * T_wall + 0.5 * Vsq_wall);

      /* Set bias */
      LDG_bias(fpt) = 1;

      break;
    }
    
    case ISOTHERMAL_NOSLIP_MOVING_G: /* Moving Isothermal No-slip Wall (ghost) */
    {
      // NOT IMPLEMENTED

      break;
    }


    case ADIABATIC_NOSLIP_P: /* Adiabatic No-slip Wall (prescribed) */
    {
      double VG[nVars] = {0.0};
      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          VG[dim] = Vg(fpt, dim);
      }

      /* Extrapolate density */
      double rhoL = U(0, 0, fpt);
      U(1, 0, fpt) = rhoL;
      U_ldg(1, 0, fpt) = rhoL;

      /* Set right state (common) velocity to zero (or wall velocity) */
      double Vsq = 0.0; double VLsq = 0.0; double Vsq_grid = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        double VL = U(0, dim+1, fpt) / rhoL; 
        double V = -VL + 2 * VG[dim];
        U(1, dim+1, fpt) = rhoL * V;
        U_ldg(1, dim+1, fpt) = rhoL * VG[dim];

        Vsq += V * V;
        VLsq += VL * VL;
        Vsq_grid += VG[dim] * VG[dim];
      }

      double EL = U(0, nDims + 1, fpt);
      U(1, nDims + 1, fpt) = EL + 0.5 * rhoL * (Vsq - VLsq);
      U_ldg(1, nDims + 1, fpt) = EL + 0.5 * rhoL * (Vsq_grid - VLsq);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case ADIABATIC_NOSLIP_G: /* Adiabatic No-slip Wall (ghost) */
    {

      // NOT IMPLEMENTED
      break;
    }


    case ADIABATIC_NOSLIP_MOVING_P: /* Moving Adiabatic No-slip Wall (prescribed) */
    {
      /* Extrapolate density */
      double rhoL = U(0, 0, fpt);
      U(1, 0, fpt) = rhoL;
      U_ldg(1, 0, fpt) = rhoL;

      /* Set right state (common) velocity to zero (or wall velocity) */
      double Vsq = 0.0; double VLsq = 0.0; double Vsq_wall = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        double VL = U(0, dim+1, fpt) / rhoL; 
        double V = -VL+ 2 * V_wall(dim);
        U(1, dim+1, fpt) = rhoL * V;
        U_ldg(1, dim+1, fpt) = rhoL * V_wall(dim);

        Vsq += V * V;
        VLsq += VL * VL;
        Vsq_wall += V_wall(dim) * V_wall(dim);
      }

      double EL = U(0, nDims + 1, fpt);
      U(1, nDims + 1, fpt) = EL + 0.5 * rhoL * (Vsq - VLsq);
      U_ldg(1, nDims + 1, fpt) = EL - 0.5 * rhoL * (VLsq + Vsq_wall);

      /* Set LDG bias */
      LDG_bias(fpt) = 1;

      break;
    }

    case ADIABATIC_NOSLIP_MOVING_G: /* Moving Adiabatic No-slip Wall (ghost) */
    {
      // NOT IMPLEMENTED

      break;
    }

    case OVERSET:
    {
      // Do nothing
      break;
    }
  }

}

void apply_bcs_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, double rho_fs, 
    mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, mdvector_gpu<double> &V_wall, mdvector_gpu<double> &Vg,
    mdvector_gpu<double> &norm_fs,  mdvector_gpu<double> &norm, mdvector_gpu<char> &gfpt2bnd,
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias, unsigned int equation,
    bool motion)
{
  if (nGfpts_bnd == 0) return;

  unsigned int threads = 128;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      apply_bcs<1, 2, AdvDiff><<<blocks, threads>>>(U, U_ldg, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
          gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, Vg, norm_fs, norm, gfpt2bnd, rus_bias, LDG_bias, motion);
    else
      apply_bcs<1, 3, AdvDiff><<<blocks, threads>>>(U, U_ldg, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
          gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, Vg, norm_fs, norm, gfpt2bnd, rus_bias, LDG_bias, motion);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      apply_bcs<4, 2, EulerNS><<<blocks, threads>>>(U, U_ldg, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
          gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, Vg, norm_fs, norm, gfpt2bnd, rus_bias, LDG_bias, motion);
    else
      apply_bcs<5, 3, EulerNS><<<blocks, threads>>>(U, U_ldg, nFpts, nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, 
          gamma, R_ref,T_tot_fs, P_tot_fs, T_wall, V_wall, Vg, norm_fs, norm, gfpt2bnd, rus_bias, LDG_bias, motion);
  }
}

template<unsigned int nVars, unsigned int nDims>
__global__
void apply_bcs_dU(mdview_gpu<double> dU, mdview_gpu<double> U, mdvector_gpu<double> norm_gfpt,
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    mdvector_gpu<char> gfpt2bnd)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  if(bnd_id == ADIABATIC_NOSLIP_P || bnd_id == ADIABATIC_NOSLIP_G ||
          bnd_id == ADIABATIC_NOSLIP_MOVING_P || bnd_id == ADIABATIC_NOSLIP_MOVING_G) /* Adibatic Wall */
  {
    double norm[nDims];

    for (unsigned int dim = 0; dim < nDims; dim++)
      norm[dim] = norm_gfpt(fpt, dim);

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
  else if (bnd_id == OVERSET)
  {
    // Do nothing (handled similarly to MPI)
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


void apply_bcs_dU_wrapper(mdview_gpu<double> &dU, mdview_gpu<double> &U, mdvector_gpu<double> &norm, 
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, 
    unsigned int nDims, mdvector_gpu<char> &gfpt2bnd, unsigned int equation)
{
  if (nGfpts_bnd == 0) return;

  unsigned int threads = 128;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      apply_bcs_dU<1, 2><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd);
    else
      apply_bcs_dU<1, 3><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      apply_bcs_dU<4, 2><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd);
    else
      apply_bcs_dU<5, 3><<<blocks, threads>>>(dU, U, norm, nFpts, nGfpts_int, nGfpts_bnd,
          gfpt2bnd);
  }
}

template<unsigned int nVars, unsigned int nDims>
__global__
void apply_bcs_dFdU(mdview_gpu<double> U, mdvector_gpu<double> dFdUconv, mdvector_gpu<double> dFdUvisc,
    mdvector_gpu<double> dUcdU, mdvector_gpu<double> dFddUvisc, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, double rho_fs, mdvector_gpu<double> V_fs, double P_fs, double gamma,
    mdvector_gpu<double> norm, mdvector_gpu<char> gfpt2bnd, bool viscous)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  double dURdUL[nVars][nVars];
  double dFdURconv[nVars][nVars][nDims];

  double dUcdUR[nVars][nVars];
  double dFdURvisc[nVars][nVars][nDims];

  double ddURddUL[nVars][nVars][nDims][nDims];
  double dFddURvisc[nVars][nVars][nDims][nDims];

  /* Copy right state values */
  if (bnd_id != PERIODIC && bnd_id != SUP_IN)
  {
    /* Copy right state dFdUconv */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFdURconv[ni][nj][dim] = dFdUconv(fpt, ni, nj, dim, 1);
        }
      }
    }

    if (viscous)
    {
      /* Copy right state dUcdU */
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dUcdUR[ni][nj] = dUcdU(fpt, ni, nj, 1);
        }
      }

      /* Copy right state dFdUvisc */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dFdURvisc[ni][nj][dim] = dFdUvisc(fpt, ni, nj, dim, 1);
          }
        }
      }

      /* Copy right state dFddUvisc */
      if (bnd_id == ADIABATIC_NOSLIP_P) /* Adiabatic Wall */
      {
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
        {
          for (unsigned int dimi = 0; dimi < nDims; dimi++)
          {
            for (unsigned int nj = 0; nj < nVars; nj++)
            {
              for (unsigned int ni = 0; ni < nVars; ni++)
              {
                dFddURvisc[ni][nj][dimi][dimj] = dFddUvisc(fpt, ni, nj, dimi, dimj, 1);
              }
            }
          }
        }
      }
    }
  }

  /* Apply specified boundary condition */
  switch(bnd_id)
  {
    case SUB_OUT: /* Subsonic Outlet */
    {
      /* Primitive Variables */
      double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
      double vL = U(fpt, 2, 0) / U(fpt, 0, 0);

      /* Compute dURdUL */
      dURdUL[0][0] = 1;
      dURdUL[1][0] = 0;
      dURdUL[2][0] = 0;
      dURdUL[3][0] = -0.5 * (uL*uL + vL*vL);

      dURdUL[0][1] = 0;
      dURdUL[1][1] = 1;
      dURdUL[2][1] = 0;
      dURdUL[3][1] = uL;

      dURdUL[0][2] = 0;
      dURdUL[1][2] = 0;
      dURdUL[2][2] = 1;
      dURdUL[3][2] = vL;

      dURdUL[0][3] = 0;
      dURdUL[1][3] = 0;
      dURdUL[2][3] = 0;
      dURdUL[3][3] = 0;

      break;
    }

    case CHAR: /* Characteristic (from PyFR) */
    case CHAR_P: /* Characteristic (prescribed) */
    {

      /* Compute wall normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += U(fpt, dim+1, 0) / U(fpt, 0, 0) * norm(fpt, dim);
        VnR += V_fs(dim) * norm(fpt, dim);
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

      double cL = std::sqrt(gamma * PL / U(fpt, 0, 0));
      double cR = std::sqrt(gamma * PR / rho_fs);

      /* Compute Riemann Invariants */
      // Note: Implicit Char BC not implemented for supersonic flow!
      double RL = VnL + 2.0 / (gamma - 1) * cL;
      double RB = VnR - 2.0 / (gamma - 1) * cR;

      double cstar = 0.25 * (gamma - 1) * (RL - RB);
      double ustarn = 0.5 * (RL + RB);

      if (nDims == 2)
      {
        double nx = norm(fpt, 0);
        double ny = norm(fpt, 1);
        double gam = gamma;

        /* Primitive Variables */
        double rhoL = U(fpt, 0, 0);
        double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
        double vL = U(fpt, 2, 0) / U(fpt, 0, 0);

        double rhoR = U(fpt, 0, 1);
        double uR = U(fpt, 1, 1) / U(fpt, 0, 1);
        double vR = U(fpt, 2, 1) / U(fpt, 0, 1);

        if (VnL < 0.0) /* Case 1: Inflow */
        {
          /* Matrix Parameters */
          double a1 = 0.5 * rhoR / cstar;
          double a2 = gam / (rhoL * cL);
          
          double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
          double b2 = nx / rhoL - a2 * uL;
          double b3 = ny / rhoL - a2 * vL;
          double b4 = a2 / cstar;

          double c1 = cstar * cstar / ((gam-1.0) * gam) + 0.5 * (uR*uR + vR*vR);
          double c2 = uR * nx + vR * ny + cstar / gam;

          /* Compute dURdUL */
          dURdUL[0][0] = a1 * b1;
          dURdUL[1][0] = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
          dURdUL[2][0] = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
          dURdUL[3][0] = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

          dURdUL[0][1] = a1 * b2;
          dURdUL[1][1] = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
          dURdUL[2][1] = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
          dURdUL[3][1] = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

          dURdUL[0][2] = a1 * b3;
          dURdUL[1][2] = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
          dURdUL[2][2] = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
          dURdUL[3][2] = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

          dURdUL[0][3] = 0.5 * rhoR * b4;
          dURdUL[1][3] = 0.5 * rhoR * (b4 * uR + a2 * nx);
          dURdUL[2][3] = 0.5 * rhoR * (b4 * vR + a2 * ny);
          dURdUL[3][3] = 0.5 * rhoR * (b4 * c1 + a2 * c2);
        }

        else  /* Case 2: Outflow */
        {
          /* Matrix Parameters */
          double a1 = gam * rhoR / (gam-1.0);
          double a2 = gam / (rhoL * cL);
          double a3 = (gam-1.0) / (gam * PL);
          double a4 = (gam-1.0) / (2.0 * gam * cstar);
          double a5 = rhoR * cstar * cstar / (gam-1.0) / (gam-1.0);
          double a6 = rhoR * cstar / (2.0 * gam);

          double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
          double b2 = nx / rhoL - a2 * uL;
          double b3 = ny / rhoL - a2 * vL;

          double c1 = 0.5 * b1 * nx - (VnL * nx + uL) / rhoL;
          double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
          double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
          double c4 = ustarn * nx + uL - VnL * nx;

          double d1 = 0.5 * b1 * ny - (VnL * ny + vL) / rhoL;
          double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
          double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
          double d4 = ustarn * ny + vL - VnL * ny;

          double e1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
          double e2 = a3 * uL + a4 * b2;
          double e3 = a3 * vL + a4 * b3;
          double e4 = a3 + a2 * a4;

          double f1 = 0.5 * a1 * (c4*c4 + d4*d4) + a5;

          /* Compute dURdUL */
          dURdUL[0][0] = a1 * e1;
          dURdUL[1][0] = a1 * e1 * c4 + rhoR * c1;
          dURdUL[2][0] = a1 * e1 * d4 + rhoR * d1;
          dURdUL[3][0] = rhoR * (c1*c4 + d1*d4) + e1 * f1 + a6 * b1;

          dURdUL[0][1] = a1 * e2;
          dURdUL[1][1] = a1 * e2 * c4 + rhoR * c2;
          dURdUL[2][1] = a1 * e2 * d4 + rhoR * d2;
          dURdUL[3][1] = rhoR * (c2*c4 + d2*d4) + e2 * f1 + a6 * b2;

          dURdUL[0][2] = a1 * e3;
          dURdUL[1][2] = a1 * e3 * c4 + rhoR * c3;
          dURdUL[2][2] = a1 * e3 * d4 + rhoR * d3;
          dURdUL[3][2] = rhoR * (c3*c4 + d3*d4) + e3 * f1 + a6 * b3;

          dURdUL[0][3] = a1 * e4;
          dURdUL[1][3] = a1 * e4 * c4 + 0.5 * rhoR * a2 * nx;
          dURdUL[2][3] = a1 * e4 * d4 + 0.5 * rhoR * a2 * ny;
          dURdUL[3][3] = 0.5 * rhoR * a2 * (c4*nx + d4*ny) + e4 * f1 + a2 * a6;
        }
      }

      else if (nDims == 3)
      {
        double nx = norm(fpt, 0);
        double ny = norm(fpt, 1);
        double nz = norm(fpt, 2);
        double gam = gamma;

        /* Primitive Variables */
        double rhoL = U(fpt, 0, 0);
        double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
        double vL = U(fpt, 2, 0) / U(fpt, 0, 0);
        double wL = U(fpt, 3, 0) / U(fpt, 0, 0);

        double rhoR = U(fpt, 0, 1);
        double uR = U(fpt, 1, 1) / U(fpt, 0, 1);
        double vR = U(fpt, 2, 1) / U(fpt, 0, 1);
        double wR = U(fpt, 3, 1) / U(fpt, 0, 1);

        if (VnL < 0.0) /* Case 1: Inflow */
        {
          /* Matrix Parameters */
          double a1 = 0.5 * rhoR / cstar;
          double a2 = gam / (rhoL * cL);
          
          double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
          double b2 = nx / rhoL - a2 * uL;
          double b3 = ny / rhoL - a2 * vL;
          double b4 = nz / rhoL - a2 * wL;
          double b5 = a2 / cstar;

          double c1 = cstar * cstar / ((gam-1.0) * gam) + 0.5 * (uR*uR + vR*vR + wR*wR);
          double c2 = uR * nx + vR * ny + wR * nz + cstar / gam;

          /* Compute dURdUL */
          dURdUL[0][0] = a1 * b1;
          dURdUL[1][0] = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
          dURdUL[2][0] = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
          dURdUL[3][0] = a1 * b1 * wR + 0.5 * rhoR * b1 * nz;
          dURdUL[4][0] = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

          dURdUL[0][1] = a1 * b2;
          dURdUL[1][1] = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
          dURdUL[2][1] = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
          dURdUL[3][1] = a1 * b2 * wR + 0.5 * rhoR * b2 * nz;
          dURdUL[4][1] = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

          dURdUL[0][2] = a1 * b3;
          dURdUL[1][2] = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
          dURdUL[2][2] = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
          dURdUL[3][2] = a1 * b3 * wR + 0.5 * rhoR * b3 * nz;
          dURdUL[4][2] = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

          dURdUL[0][3] = a1 * b4;
          dURdUL[1][3] = a1 * b4 * uR + 0.5 * rhoR * b4 * nx;
          dURdUL[2][3] = a1 * b4 * vR + 0.5 * rhoR * b4 * ny;
          dURdUL[3][3] = a1 * b4 * wR + 0.5 * rhoR * b4 * nz;
          dURdUL[4][3] = a1 * b4 * c1 + 0.5 * rhoR * b4 * c2;

          dURdUL[0][4] = 0.5 * rhoR * b5;
          dURdUL[1][4] = 0.5 * rhoR * (b5 * uR + a2 * nx);
          dURdUL[2][4] = 0.5 * rhoR * (b5 * vR + a2 * ny);
          dURdUL[3][4] = 0.5 * rhoR * (b5 * wR + a2 * nz);
          dURdUL[4][4] = 0.5 * rhoR * (b5 * c1 + a2 * c2);
        }

        else  /* Case 2: Outflow */
        {
          /* Matrix Parameters */
          double a1 = gam * rhoR / (gam-1.0);
          double a2 = gam / (rhoL * cL);
          double a3 = (gam-1.0) / (gam * PL);
          double a4 = (gam-1.0) / (2.0 * gam * cstar);
          double a5 = rhoR * cstar * cstar / (gam-1.0) / (gam-1.0);
          double a6 = rhoR * cstar / (2.0 * gam);

          double b1 = -VnL / rhoL - a2 / rhoL * (PL / (gam-1.0) - 0.5 * momF);
          double b2 = nx / rhoL - a2 * uL;
          double b3 = ny / rhoL - a2 * vL;
          double b4 = nz / rhoL - a2 * wL;

          double c1 = 0.5 * b1 * nx - (VnL * nx + uL) / rhoL;
          double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
          double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
          double c4 = 0.5 * b4 * nx - nx * nz / rhoL;
          double c5 = ustarn * nx + uL - VnL * nx;

          double d1 = 0.5 * b1 * ny - (VnL * ny + vL) / rhoL;
          double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
          double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
          double d4 = 0.5 * b4 * ny - ny * nz / rhoL;
          double d5 = ustarn * ny + vL - VnL * ny;

          double e1 = 0.5 * b1 * nz - (VnL * nz + wL) / rhoL;
          double e2 = 0.5 * b2 * nz - nx * nz / rhoL;
          double e3 = 0.5 * b3 * nz - ny * nz / rhoL;
          double e4 = 0.5 * b4 * nz + (1.0 - nz*nz) / rhoL;
          double e5 = ustarn * nz + wL - VnL * nz;

          double f1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
          double f2 = a3 * uL + a4 * b2;
          double f3 = a3 * vL + a4 * b3;
          double f4 = a3 * wL + a4 * b4;
          double f5 = a3 + a2 * a4;

          double g1 = 0.5 * a1 * (c5*c5 + d5*d5 + e5*e5) + a5;

          /* Compute dURdUL */
          dURdUL[0][0] = a1 * f1;
          dURdUL[1][0] = a1 * f1 * c5 + rhoR * c1;
          dURdUL[2][0] = a1 * f1 * d5 + rhoR * d1;
          dURdUL[3][0] = a1 * f1 * e5 + rhoR * e1;
          dURdUL[4][0] = rhoR * (c1*c5 + d1*d5 + e1*e5) + f1 * g1 + a6 * b1;

          dURdUL[0][1] = a1 * f2;
          dURdUL[1][1] = a1 * f2 * c5 + rhoR * c2;
          dURdUL[2][1] = a1 * f2 * d5 + rhoR * d2;
          dURdUL[3][1] = a1 * f2 * e5 + rhoR * e2;
          dURdUL[4][1] = rhoR * (c2*c5 + d2*d5 + e2*e5) + f2 * g1 + a6 * b2;

          dURdUL[0][2] = a1 * f3;
          dURdUL[1][2] = a1 * f3 * c5 + rhoR * c3;
          dURdUL[2][2] = a1 * f3 * d5 + rhoR * d3;
          dURdUL[3][2] = a1 * f3 * e5 + rhoR * e3;
          dURdUL[4][2] = rhoR * (c3*c5 + d3*d5 + e3*e5) + f3 * g1 + a6 * b3;

          dURdUL[0][3] = a1 * f4;
          dURdUL[1][3] = a1 * f4 * c5 + rhoR * c4;
          dURdUL[2][3] = a1 * f4 * d5 + rhoR * d4;
          dURdUL[3][3] = a1 * f4 * e5 + rhoR * e4;
          dURdUL[4][3] = rhoR * (c4*c5 + d4*d5 + e4*e5) + f4 * g1 + a6 * b4;

          dURdUL[0][4] = a1 * f5;
          dURdUL[1][4] = a1 * f5 * c5 + 0.5 * rhoR * a2 * nx;
          dURdUL[2][4] = a1 * f5 * d5 + 0.5 * rhoR * a2 * ny;
          dURdUL[3][4] = a1 * f5 * e5 + 0.5 * rhoR * a2 * nz;
          dURdUL[4][4] = 0.5 * rhoR * a2 * (c5*nx + d5*ny + e5*nz) + f5 * g1 + a2 * a6;
        }
      }

      break;
    }

    case SYMMETRY_P: /* Symmetry (prescribed) */
    case SLIP_WALL_P: /* Slip Wall (prescribed) */
    {
      if (nDims == 2)
      {
        double nx = norm(fpt, 0);
        double ny = norm(fpt, 1);

        /* Primitive Variables */
        double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
        double vL = U(fpt, 2, 0) / U(fpt, 0, 0);

        double uR = U(fpt, 1, 1) / U(fpt, 0, 1);
        double vR = U(fpt, 2, 1) / U(fpt, 0, 1);

        /* Compute dURdUL */
        dURdUL[0][0] = 1;
        dURdUL[1][0] = 0;
        dURdUL[2][0] = 0;
        dURdUL[3][0] = 0.5 * (uL*uL + vL*vL - uR*uR - vR*vR);

        dURdUL[0][1] = 0;
        dURdUL[1][1] = 1.0-nx*nx;
        dURdUL[2][1] = -nx*ny;
        dURdUL[3][1] = -uL + (1.0-nx*nx)*uR - nx*ny*vR;

        dURdUL[0][2] = 0;
        dURdUL[1][2] = -nx*ny;
        dURdUL[2][2] = 1.0-ny*ny;
        dURdUL[3][2] = -vL - nx*ny*uR + (1.0-ny*ny)*vR;

        dURdUL[0][3] = 0;
        dURdUL[1][3] = 0;
        dURdUL[2][3] = 0;
        dURdUL[3][3] = 1;
      }

      else if (nDims == 3)
      {
        double nx = norm(fpt, 0);
        double ny = norm(fpt, 1);
        double nz = norm(fpt, 2);

        /* Primitive Variables */
        double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
        double vL = U(fpt, 2, 0) / U(fpt, 0, 0);
        double wL = U(fpt, 3, 0) / U(fpt, 0, 0);

        double uR = U(fpt, 1, 1) / U(fpt, 0, 1);
        double vR = U(fpt, 2, 1) / U(fpt, 0, 1);
        double wR = U(fpt, 3, 1) / U(fpt, 0, 1);

        /* Compute dURdUL */
        dURdUL[0][0] = 1;
        dURdUL[1][0] = 0;
        dURdUL[2][0] = 0;
        dURdUL[3][0] = 0;
        dURdUL[4][0] = 0.5 * (uL*uL + vL*vL + wL*wL - uR*uR - vR*vR - wR*wR);

        dURdUL[0][1] = 0;
        dURdUL[1][1] = 1.0-nx*nx;
        dURdUL[2][1] = -nx*ny;
        dURdUL[3][1] = -nx*nz;
        dURdUL[4][1] = -uL + (1.0-nx*nx)*uR - nx*ny*vR - nx*nz*wR;

        dURdUL[0][2] = 0;
        dURdUL[1][2] = -nx*ny;
        dURdUL[2][2] = 1.0-ny*ny;
        dURdUL[3][2] = -ny*nz;
        dURdUL[4][2] = -vL - nx*ny*uR + (1.0-ny*ny)*vR - ny*nz*wR;

        dURdUL[0][3] = 0;
        dURdUL[1][3] = -nx*nz;
        dURdUL[2][3] = -ny*nz;
        dURdUL[3][3] = 1.0-nz*nz;
        dURdUL[4][3] = -wL - nx*nz*uR - ny*nz*vR + (1.0-nz*nz)*wR;

        dURdUL[0][4] = 0;
        dURdUL[1][4] = 0;
        dURdUL[2][4] = 0;
        dURdUL[3][4] = 0;
        dURdUL[4][4] = 1;
      }

      break;
    }

    case SYMMETRY_G: /* Symmetry (ghost) */
    case SLIP_WALL_G: /* Slip Wall (ghost) */
    {
      double nx = norm(fpt, 0);
      double ny = norm(fpt, 1);

      dURdUL[0][0] = 1;
      dURdUL[1][0] = 0;
      dURdUL[2][0] = 0;
      dURdUL[3][0] = 0;

      dURdUL[0][1] = 0;
      dURdUL[1][1] = 1.0 - 2.0 * nx * nx;
      dURdUL[2][1] = -2.0 * nx * ny;
      dURdUL[3][1] = 0;

      dURdUL[0][2] = 0;
      dURdUL[1][2] = -2.0 * nx * ny;
      dURdUL[2][2] = 1.0 - 2.0 * ny * ny;
      dURdUL[3][2] = 0;

      dURdUL[0][3] = 0;
      dURdUL[1][3] = 0;
      dURdUL[2][3] = 0;
      dURdUL[3][3] = 1;

      break;
    }

    case ADIABATIC_NOSLIP_P: /* No-slip Wall (adiabatic) */
    {
      double nx = norm(fpt, 0);
      double ny = norm(fpt, 1);

      /* Primitive Variables */
      double rhoL = U(fpt, 0, 0);
      double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
      double vL = U(fpt, 2, 0) / U(fpt, 0, 0);
      double eL = U(fpt, 3, 0);

      /* Compute dURdUL */
      dURdUL[0][0] = 1;
      dURdUL[1][0] = 0;
      dURdUL[2][0] = 0;
      dURdUL[3][0] = 0.5 * (uL*uL + vL*vL);

      dURdUL[0][1] = 0;
      dURdUL[1][1] = 0;
      dURdUL[2][1] = 0;
      dURdUL[3][1] = -uL;

      dURdUL[0][2] = 0;
      dURdUL[1][2] = 0;
      dURdUL[2][2] = 0;
      dURdUL[3][2] = -vL;

      dURdUL[0][3] = 0;
      dURdUL[1][3] = 0;
      dURdUL[2][3] = 0;
      dURdUL[3][3] = 1;

      if (viscous)
      {
        /* Compute dUxR/dUxL */
        ddURddUL[0][0][0][0] = 1;
        ddURddUL[1][0][0][0] = 0;
        ddURddUL[2][0][0][0] = 0;
        ddURddUL[3][0][0][0] = nx*nx * (eL / rhoL - (uL*uL + vL*vL));

        ddURddUL[0][1][0][0] = 0;
        ddURddUL[1][1][0][0] = 1;
        ddURddUL[2][1][0][0] = 0;
        ddURddUL[3][1][0][0] = nx*nx * uL;

        ddURddUL[0][2][0][0] = 0;
        ddURddUL[1][2][0][0] = 0;
        ddURddUL[2][2][0][0] = 1;
        ddURddUL[3][2][0][0] = nx*nx * vL;

        ddURddUL[0][3][0][0] = 0;
        ddURddUL[1][3][0][0] = 0;
        ddURddUL[2][3][0][0] = 0;
        ddURddUL[3][3][0][0] = 1.0 - nx*nx;

        /* Compute dUyR/dUxL */
        ddURddUL[0][0][1][0] = 0;
        ddURddUL[1][0][1][0] = 0;
        ddURddUL[2][0][1][0] = 0;
        ddURddUL[3][0][1][0] = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddURddUL[0][1][1][0] = 0;
        ddURddUL[1][1][1][0] = 0;
        ddURddUL[2][1][1][0] = 0;
        ddURddUL[3][1][1][0] = nx*ny * uL;

        ddURddUL[0][2][1][0] = 0;
        ddURddUL[1][2][1][0] = 0;
        ddURddUL[2][2][1][0] = 0;
        ddURddUL[3][2][1][0] = nx*ny * vL;

        ddURddUL[0][3][1][0] = 0;
        ddURddUL[1][3][1][0] = 0;
        ddURddUL[2][3][1][0] = 0;
        ddURddUL[3][3][1][0] = -nx * ny;

        /* Compute dUxR/dUyL */
        ddURddUL[0][0][0][1] = 0;
        ddURddUL[1][0][0][1] = 0;
        ddURddUL[2][0][0][1] = 0;
        ddURddUL[3][0][0][1] = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddURddUL[0][1][0][1] = 0;
        ddURddUL[1][1][0][1] = 0;
        ddURddUL[2][1][0][1] = 0;
        ddURddUL[3][1][0][1] = nx*ny * uL;

        ddURddUL[0][2][0][1] = 0;
        ddURddUL[1][2][0][1] = 0;
        ddURddUL[2][2][0][1] = 0;
        ddURddUL[3][2][0][1] = nx*ny * vL;

        ddURddUL[0][3][0][1] = 0;
        ddURddUL[1][3][0][1] = 0;
        ddURddUL[2][3][0][1] = 0;
        ddURddUL[3][3][0][1] = -nx * ny;

        /* Compute dUyR/dUyL */
        ddURddUL[0][0][1][1] = 1;
        ddURddUL[1][0][1][1] = 0;
        ddURddUL[2][0][1][1] = 0;
        ddURddUL[3][0][1][1] = ny*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddURddUL[0][1][1][1] = 0;
        ddURddUL[1][1][1][1] = 1;
        ddURddUL[2][1][1][1] = 0;
        ddURddUL[3][1][1][1] = ny*ny * uL;

        ddURddUL[0][2][1][1] = 0;
        ddURddUL[1][2][1][1] = 0;
        ddURddUL[2][2][1][1] = 1;
        ddURddUL[3][2][1][1] = ny*ny * vL;

        ddURddUL[0][3][1][1] = 0;
        ddURddUL[1][3][1][1] = 0;
        ddURddUL[2][3][1][1] = 0;
        ddURddUL[3][3][1][1] = 1.0 - ny*ny;
      }

      break;
    }
  }

  /* Compute new right state values */
  if (bnd_id != PERIODIC && bnd_id != SUP_IN)
  {
    /* Compute dFdULconv for right state */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int j = 0; j < nVars; j++)
      {
        for (unsigned int i = 0; i < nVars; i++)
        {
          double val = 0;
          for (unsigned int k = 0; k < nVars; k++)
          {
            val += dFdURconv[i][k][dim] * dURdUL[k][j];
          }
          dFdUconv(fpt, i, j, dim, 1) = val;
        }
      }
    }

    if (viscous)
    {
      /* Compute dUcdUL for right state */
      for (unsigned int j = 0; j < nVars; j++)
      {
        for (unsigned int i = 0; i < nVars; i++)
        {
          double val = 0;
          for (unsigned int k = 0; k < nVars; k++)
          {
            val += dUcdUR[i][k] * dURdUL[k][j];
          }
          dUcdU(fpt, i, j, 1) = val;
        }
      }

      /* Compute dFdULvisc for right state */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int j = 0; j < nVars; j++)
        {
          for (unsigned int i = 0; i < nVars; i++)
          {
            double val = 0;
            for (unsigned int k = 0; k < nVars; k++)
            {
              val += dFdURvisc[i][k][dim] * dURdUL[k][j];
            }
            dFdUvisc(fpt, i, j, dim, 1) = val;
          }
        }
      }

      /* Compute dFddULvisc for right state */
      if (bnd_id == ADIABATIC_NOSLIP_P) /* Adiabatic Wall */
      {
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
        {
          for (unsigned int dimi = 0; dimi < nDims; dimi++)
          {
            for (unsigned int j = 0; j < nVars; j++)
            {
              for (unsigned int i = 0; i < nVars; i++)
              {
                double val = 0;
                for (unsigned int dimk = 0; dimk < nDims; dimk++)
                {
                  for (unsigned int k = 0; k < nVars; k++)
                  {
                    val += dFddURvisc[i][k][dimi][dimk] * ddURddUL[k][j][dimk][dimj];
                  }
                }
                dFddUvisc(fpt, i, j, dimi, dimj, 1) = val;
              }
            }
          }
        }
      }
    }
  }
}

void apply_bcs_dFdU_wrapper(mdview_gpu<double> &U, mdvector_gpu<double> &dFdUconv, mdvector_gpu<double> &dFdUvisc,
    mdvector_gpu<double> &dUcdU, mdvector_gpu<double> &dFddUvisc, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nVars, unsigned int nDims, double rho_fs, mdvector_gpu<double> &V_fs, double P_fs, double gamma, 
    mdvector_gpu<double> &norm, mdvector_gpu<char> &gfpt2bnd, unsigned int equation, bool viscous)
{
  if (nGfpts_bnd == 0) return;

  unsigned int threads = 128;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (equation == EulerNS)
  {
    if (nDims == 2)
      apply_bcs_dFdU<4, 2><<<blocks, threads>>>(U, dFdUconv, dFdUvisc, dUcdU, dFddUvisc,
          nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, gamma, norm, gfpt2bnd, viscous);
    else
      apply_bcs_dFdU<5, 3><<<blocks, threads>>>(U, dFdUconv, dFdUvisc, dUcdU, dFddUvisc,
          nGfpts_int, nGfpts_bnd, rho_fs, V_fs, P_fs, gamma, norm, gfpt2bnd, viscous);
  }
}

template <unsigned int nDims, unsigned int nVars>
__global__
void compute_common_U_LDG(const mdview_gpu<double> U, mdview_gpu<double> Ucomm, 
    const mdvector_gpu<double> norm, double beta, unsigned int nFpts,
    const mdvector_gpu<char> LDG_bias, unsigned int startFpt, unsigned int endFpt,
    bool overset = false, const int* iblank = NULL)
{
    const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

    if (fpt >= endFpt)
      return;
    
    double UL[nVars];
    double UR[nVars];

    if (overset)
      if (iblank[fpt] == 0)
        return;

    /* Setting sign of beta (from HiFiLES) */
    if (nDims == 2)
    {
      if (norm(fpt, 0) + norm(fpt, 1) < 0.0)
        beta = -beta;
    }
    else if (nDims == 3)
    {
      if (norm(fpt, 0) + norm(fpt, 1) + sqrt(2.) * norm(fpt, 2) < 0.0)
        beta = -beta;
    }


    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UL[n] = U(fpt, n, 0); UR[n] = U(fpt, n, 1);
    }

    if (LDG_bias(fpt) == 0)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double UC = 0.5*(UL[n] + UR[n]) - beta*(UL[n] - UR[n]);
        Ucomm(fpt, n, 0) = UC;
        Ucomm(fpt, n, 1) = UC;
      }
    }
    /* If on boundary, don't use beta */
    else
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        Ucomm(fpt, n, 0) = UR[n];
        Ucomm(fpt, n, 1) = UR[n];
      }
    }
}

void compute_common_U_LDG_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, mdvector_gpu<char> &LDG_bias, unsigned int startFpt,
    unsigned int endFpt, bool overset, int* iblank) 
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_common_U_LDG<2, 1><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt);
    else
      compute_common_U_LDG<3, 1><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_U_LDG<2, 4><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt);
    else
      compute_common_U_LDG<3, 5><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, overset, iblank);
  }

}

template <unsigned int nDims, unsigned int nVars>
__global__
void common_U_to_F(mdview_gpu<double> Fcomm, mdview_gpu<double> Ucomm, mdvector_gpu<double> norm, 
    mdvector_gpu<double> dA, unsigned int nFpts, unsigned int startFpt, unsigned int endFpt, unsigned int dim)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  double norm_dim = norm(fpt, dim);
  double dAL = dA(fpt, 0);
  double dAR = dA(fpt, 1);

  for (unsigned int var = 0; var < nVars; var++)
  {
    double F = Ucomm(fpt, var, 0) * norm_dim;
    Fcomm(fpt, var, 0) = F * dAL;
    Fcomm(fpt, var, 1) = -F * dAR;
  }
    
}

void common_U_to_F_wrapper(mdview_gpu<double> &Fcomm, mdview_gpu<double> &Ucomm, mdvector_gpu<double> &norm, 
    mdvector_gpu<double> &dA, unsigned int nFpts, unsigned int nVars, unsigned int nDims, unsigned int equation,
    unsigned int startFpt, unsigned int endFpt, unsigned int dim)
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      common_U_to_F<2, 1><<<blocks, threads>>>(Fcomm, Ucomm, norm, dA, nFpts, startFpt, endFpt, dim);
    else
      common_U_to_F<3, 1><<<blocks, threads>>>(Fcomm, Ucomm, norm, dA, nFpts, startFpt, endFpt, dim);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      common_U_to_F<2, 4><<<blocks, threads>>>(Fcomm, Ucomm, norm, dA, nFpts, startFpt, endFpt, dim);
    else
      common_U_to_F<3, 5><<<blocks, threads>>>(Fcomm, Ucomm, norm, dA, nFpts, startFpt, endFpt, dim);
  }

}


template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__device__ __forceinline__
void rusanov_flux(double UL[nVars], double UR[nVars], double Fcomm[nVars], 
    double &PL, double &PR,  double norm[nDims], double &waveSp, double *AdvDiff_A, 
    double Vgn, double gamma, double rus_k, char rus_bias)
{

  double FL[nVars][nDims];
  double FR[nVars][nDims];
  double FnL[nVars] = {0.0};
  double FnR[nVars] = {0.0};

  double wS = 0., eig = 0.;
  if (equation == AdvDiff) 
  {
    double A[nDims];
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      A[dim] = *(AdvDiff_A + dim);
      wS += A[dim] * norm[dim];
    }

    waveSp = std::abs(wS - Vgn);

    eig = std::abs(waveSp);

    compute_Fconv_AdvDiff<nVars, nDims>(UL, FL, A);
    compute_Fconv_AdvDiff<nVars, nDims>(UR, FR, A);
  }
  else if (equation == EulerNS)
  {
    double P;
    compute_Fconv_EulerNS<nVars, nDims>(UL, FL, P, gamma);
    double aL = std::sqrt(gamma * P / UL[0]);
    PL = P;
    compute_Fconv_EulerNS<nVars, nDims>(UR, FR, P, gamma);
    double aR = std::sqrt(gamma * P / UR[0]);
    PR = P;

    /* Compute speed of sound */
    //double aL = std::sqrt(std::abs(gamma * P(fpt, 0) / UL[0]));
    //double aR = std::sqrt(std::abs(gamma * P(fpt, 1) / UR[0]));

    /* Compute normal velocities */
    double VnL = 0.0; double VnR = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      VnL += UL[dim+1]/UL[0] * norm[dim];
      VnR += UR[dim+1]/UR[0] * norm[dim];
    }

    //waveSp = max(std::abs(VnL) + aL, std::abs(VnR) + aR);
    wS = std::abs(VnL - Vgn) + aL;
    wS = max(wS, std::abs(VnR - Vgn) + aR);

    eig = std::abs(VnL) + aL;
    eig = max(eig, std::abs(VnR) + aR);

    waveSp = wS;
  }

  /* Get interface-normal flux components  (from L to R) */
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      FnL[n] += FL[n][dim] * norm[dim];
      FnR[n] += FR[n][dim] * norm[dim];
    }
  }

  /* Compute common normal flux */
  if (rus_bias == 0) /* Upwinded */
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm[n] = (0.5 * (FnR[n]+FnL[n]) - 0.5 * eig * (1.0 - rus_k) * (UR[n]-UL[n]));
    }
  }
  else if (rus_bias == 2) /* Centered */
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm[n] = 0.5 * (FnR[n] + FnL[n]);
    }
  }
  else /* Prescribed right-state */
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm[n] = FnR[n];
    }
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__device__ __forceinline__
void LDG_flux(double UL[nVars], double UR[nVars], double dUL[nVars][nDims], double dUR[nVars][nDims], double Fcomm[nVars],
    double norm[nDims], double AdvDiff_D, double &diffCo, double gamma, double prandtl, double mu, double rt, double c_sth, 
    bool fix_vis, int LDG_bias, double beta, double tau)
{
  double FL[nVars][nDims] = {{0.0}}; double FR[nVars][nDims] = {{0.0}};
  double FnL[nVars] = {0.0}; double FnR[nVars] = {0.0};

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

  /* Get numerical diffusion coefficient */
  if (equation == AdvDiff)
  {
    compute_Fvisc_AdvDiff_add<nVars, nDims>(dUL, FL, AdvDiff_D);
    compute_Fvisc_AdvDiff_add<nVars, nDims>(dUR, FR, AdvDiff_D);

    diffCo = AdvDiff_D;
  }
  else if (equation == EulerNS)
  {
    compute_Fvisc_EulerNS_add<nVars, nDims>(UL, dUL, FL, gamma, prandtl, mu, rt, c_sth, fix_vis);
    compute_Fvisc_EulerNS_add<nVars, nDims>(UR, dUR, FR, gamma, prandtl, mu, rt, c_sth, fix_vis);


    // TODO: Add or store mu from Sutherland's law
    double diffCoL = max(mu / UL[0], gamma * mu / (prandtl * UL[0]));
    double diffCoR = max(mu / UR[0], gamma * mu / (prandtl * UR[0]));
    diffCo = max(diffCoL, diffCoR);
  }

  /* Get interface-normal flux components  (from L to R)*/
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      FnL[n] += FL[n][dim] * norm[dim];
      FnR[n] += FR[n][dim] * norm[dim];
    }
  }


  /* Compute common normal viscous flux and accumulate */
  /* If interior, use central */
  if (LDG_bias == 0)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        FR[n][dim] = 0.5 * (FL[n][dim] + FR[n][dim]) + 
          tau * norm[dim] * (UL[n] - UR[n]) + beta * norm[dim] * (FnL[n] - FnR[n]);
      }
    }
  }
  /* If boundary, use right state only */
  else
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        FR[n][dim] += tau * norm[dim] * (UL[n] - UR[n]);
      }
    }
  }

  for (unsigned int n = 0; n < nVars; n++)
  {
    double F = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      F += FR[n][dim] * norm[dim];
    }

    Fcomm[n] += F;
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__ 
void compute_common_F(mdview_gpu<double> U, mdview_gpu<double> U_ldg, mdview_gpu<double> dU,
    mdview_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> AdvDiff_A, 
    mdvector_gpu<double> norm_gfpts, mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<double> diffCo,
    mdvector_gpu<char> rus_bias, mdvector_gpu<char> LDG_bias,  mdvector_gpu<double> dA_in, mdvector_gpu<double> Vg, double AdvDiff_D, double gamma, double rus_k, 
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, 
    unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, bool viscous, bool motion, bool overset, int* iblank)
{

  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  if (overset and iblank[fpt] == 0)
    return;

  double UL[nVars]; double UR[nVars];
  double Fc[nVars];
  double norm[nDims];
  double Vgn = 0.0;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(dim, fpt);
  }

  if (motion)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      Vgn += Vg(fpt, dim) * norm[dim];
    }
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    UL[n] = U(0, n, fpt); UR[n] = U(1, n, fpt);
  }

  /* Compute convective contribution to common flux */
  if (fconv_type == Rusanov)
  {
    rusanov_flux<nVars, nDims, equation>(UL, UR, Fc, P(0, fpt), P(1, fpt), norm, waveSp_gfpts(fpt),
        AdvDiff_A.data(), Vgn, gamma, rus_k, rus_bias(fpt));
  }

  if (viscous)
  {
    /* Compute viscous contribution to common flux */
    double dUL[nVars][nDims];
    double dUR[nVars][nDims];

    /* Get left and right gradients */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UR[n] = U_ldg(1, n, fpt); // Overwrite right state with "LDG" boundary values

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dUL[n][dim] = dU(0, dim, n, fpt); dUR[n][dim] = dU(1, dim, n, fpt);
      }
    }

    if (fvisc_type == LDG)
      LDG_flux<nVars, nDims, equation>(UL, UR, dUL, dUR, Fc, norm, AdvDiff_D, diffCo(fpt), gamma, 
          prandtl, mu, rt, c_sth, fix_vis, LDG_bias(fpt), beta, tau);
  }

  /* Write common flux to global memory */
  double dAL = dA_in(0, fpt);
  double dAR = dA_in(1, fpt);
  for (unsigned int n = 0; n < nVars; n++)
  {
    Fcomm(0, n, fpt) = Fc[n] * dAL;
    Fcomm(1, n, fpt) = -Fc[n] * dAR;
  }
}

void compute_common_F_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, mdview_gpu<double> &dU,
    mdview_gpu<double> &Fcomm, mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, 
    mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, mdvector_gpu<double> &diffCo,
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias,  mdvector_gpu<double> &dA, mdvector_gpu<double>& Vg, double AdvDiff_D, double gamma, double rus_k, 
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, 
    bool viscous, bool motion, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_common_F<1, 2, AdvDiff><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, fconv_type, fvisc_type, startFpt, endFpt, viscous, motion, overset, iblank);
    else
      compute_common_F<1, 3, AdvDiff><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, fconv_type, fvisc_type, startFpt, endFpt, viscous, motion, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_F<4, 2, EulerNS><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, fconv_type, fvisc_type, startFpt, endFpt, viscous, motion, overset, iblank);
    else
      compute_common_F<5, 3, EulerNS><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, fconv_type, fvisc_type, startFpt, endFpt, viscous, motion, overset, iblank);
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void rusanov_dFcdU(mdview_gpu<double> U, mdvector_gpu<double> dFdUconv, 
    mdvector_gpu<double> dFcdU, mdvector_gpu<double> P, mdvector_gpu<double> norm_gfpts, 
    mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<char> rus_bias,
    double gamma, double rus_k, unsigned int startFpt, unsigned int endFpt)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  /* Apply central flux at boundaries */
  double k = rus_k;

  double dFndUL[nVars][nVars]; double dFndUR[nVars][nVars];
  double WL[nVars]; double WR[nVars];
  double norm[nDims]; 

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(fpt, dim);
  }

  /* Initialize dFndUL, dFndUR */
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      dFndUL[ni][nj] = 0;
      dFndUR[ni][nj] = 0;
    }
  }

  /* Get interface-normal dFdU components  (from L to R)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int nj = 0; nj < nVars; nj++)
    {
      for (unsigned int ni = 0; ni < nVars; ni++)
      {
        dFndUL[ni][nj] += dFdUconv(fpt, ni, nj, dim, 0) * norm[dim];
        dFndUR[ni][nj] += dFdUconv(fpt, ni, nj, dim, 1) * norm[dim];
      }
    }
  }

  if (rus_bias(fpt) == 2) /* Central */
  {
    for (unsigned int nj = 0; nj < nVars; nj++)
    {
      for (unsigned int ni = 0; ni < nVars; ni++)
      {
        dFcdU(fpt, ni, nj, 0, 0) = 0.5 * dFndUL[ni][nj];
        dFcdU(fpt, ni, nj, 1, 0) = 0.5 * dFndUR[ni][nj];

        dFcdU(fpt, ni, nj, 0, 1) = 0.5 * dFndUL[ni][nj];
        dFcdU(fpt, ni, nj, 1, 1) = 0.5 * dFndUR[ni][nj];
      }
    }
    return;
  }
  else if (rus_bias(fpt) == 1) /* Set flux state */
  {
    for (unsigned int nj = 0; nj < nVars; nj++)
    {
      for (unsigned int ni = 0; ni < nVars; ni++)
      {
        dFcdU(fpt, ni, nj, 0, 0) = 0;
        dFcdU(fpt, ni, nj, 1, 0) = dFndUR[ni][nj];

        dFcdU(fpt, ni, nj, 0, 1) = 0;
        dFcdU(fpt, ni, nj, 1, 1) = dFndUR[ni][nj];
      }
    }
    return;
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
  }

  /* Get numerical wavespeed and derivative */
  double waveSp = 0;
  double dwSdUL[nVars];
  double dwSdUR[nVars];
  if (equation == AdvDiff)
  {
    waveSp = waveSp_gfpts(fpt);
    dwSdUL[0] = 0;
    dwSdUR[0] = 0;
  }
  else if (equation == EulerNS)
  {
    /* Compute speed of sound */
    double aL = std::sqrt(std::abs(gamma * P(fpt, 0) / WL[0]));
    double aR = std::sqrt(std::abs(gamma * P(fpt, 1) / WR[0]));
    double VnL = 0.0; double VnR = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      VnL += WL[dim+1]/WL[0] * norm[dim];
      VnR += WR[dim+1]/WR[0] * norm[dim];
    }

    /* Compute wavespeed and wavespeed derivative */
    double gam = gamma;
    double wSL = std::abs(VnL) + aL;
    double wSR = std::abs(VnR) + aR;
    if (wSL > wSR)
    {
      /* Determine direction */
      int sgn = (VnL > 0) - (VnL < 0);

      /* Primitive Variables */
      double rho = WL[0];
      double V[nDims];
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = WL[dim+1] / WL[0];
      }

      /* Wavespeed */
      waveSp = wSL;

      /* Compute wavespeed derivative */
      dwSdUL[0] = -sgn*VnL/rho - aL/(2.0*rho);
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dwSdUL[0] += gam * (gam-1.0) * V[dim]*V[dim] / (4.0*aL*rho);
        dwSdUL[dim+1] = sgn*norm[dim]/rho - gam * (gam-1.0) * V[dim] / (2.0*aL*rho);
      }
      dwSdUL[nDims+1] = gam * (gam-1.0) / (2.0*aL*rho);

      for (unsigned int n = 0; n < nVars; n++)
      {
        dwSdUR[n] = 0;
      }
    }

    else
    {
      /* Determine direction */
      int sgn = (VnR > 0) - (VnR < 0);

      /* Primitive Variables */
      double rho = WR[0];
      double V[nDims];
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = WR[dim+1] / WR[0];
      }

      /* Wavespeed */
      waveSp = wSR;

      /* Compute wavespeed derivative */
      dwSdUR[0] = -sgn*VnR/rho - aR/(2.0*rho);
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dwSdUR[0] += gam * (gam-1.0) * V[dim]*V[dim] / (4.0*aR*rho);
        dwSdUR[dim+1] = sgn*norm[dim]/rho - gam * (gam-1.0) * V[dim] / (2.0*aR*rho);
      }
      dwSdUR[nDims+1] = gam * (gam-1.0) / (2.0*aR*rho);

      for (unsigned int n = 0; n < nVars; n++)
      {
        dwSdUL[n] = 0;
      }
    }
  }

  /* Compute common dFdU */
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      if (ni == nj)
      {
        dFcdU(fpt, ni, nj, 0, 0) = 0.5 * (dFndUL[ni][nj] - ((WR[ni]-WL[ni]) * dwSdUL[nj] - waveSp) * (1.0-k));
        dFcdU(fpt, ni, nj, 1, 0) = 0.5 * (dFndUR[ni][nj] - ((WR[ni]-WL[ni]) * dwSdUR[nj] + waveSp) * (1.0-k));

        dFcdU(fpt, ni, nj, 0, 1) = 0.5 * (dFndUL[ni][nj] - ((WR[ni]-WL[ni]) * dwSdUL[nj] - waveSp) * (1.0-k));
        dFcdU(fpt, ni, nj, 1, 1) = 0.5 * (dFndUR[ni][nj] - ((WR[ni]-WL[ni]) * dwSdUR[nj] + waveSp) * (1.0-k));
      }
      else
      {
        dFcdU(fpt, ni, nj, 0, 0) = 0.5 * (dFndUL[ni][nj] - (WR[ni]-WL[ni]) * dwSdUL[nj] * (1.0-k));
        dFcdU(fpt, ni, nj, 1, 0) = 0.5 * (dFndUR[ni][nj] - (WR[ni]-WL[ni]) * dwSdUR[nj] * (1.0-k));

        dFcdU(fpt, ni, nj, 0, 1) = 0.5 * (dFndUL[ni][nj] - (WR[ni]-WL[ni]) * dwSdUL[nj] * (1.0-k));
        dFcdU(fpt, ni, nj, 1, 1) = 0.5 * (dFndUR[ni][nj] - (WR[ni]-WL[ni]) * dwSdUR[nj] * (1.0-k));
      }
    }
  }
}

void rusanov_dFcdU_wrapper(mdview_gpu<double> &U, mdvector_gpu<double> &dFdUconv, 
    mdvector_gpu<double> &dFcdU, mdvector_gpu<double> &P, mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<char> &rus_bias, double gamma, double rus_k, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt)
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      rusanov_dFcdU<1, 2, AdvDiff><<<blocks, threads>>>(U, dFdUconv, dFcdU, P, norm, 
          waveSp, rus_bias, gamma, rus_k, startFpt, endFpt);
    else
      rusanov_dFcdU<1, 3, AdvDiff><<<blocks, threads>>>(U, dFdUconv, dFcdU, P, norm, 
          waveSp, rus_bias, gamma, rus_k, startFpt, endFpt);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      rusanov_dFcdU<4, 2, EulerNS><<<blocks, threads>>>(U, dFdUconv, dFcdU, P, norm, 
          waveSp, rus_bias, gamma, rus_k, startFpt, endFpt);
    else
      rusanov_dFcdU<5, 3, EulerNS><<<blocks, threads>>>(U, dFdUconv, dFcdU, P, norm, 
          waveSp, rus_bias, gamma, rus_k, startFpt, endFpt);
  }
}

__global__
void unpack_fringe_u(mdvector_gpu<double> U_fringe,
    mdview_gpu<double> U, mdvector_gpu<unsigned int> fringe_fpts,
    mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe, unsigned int nFpts,
    unsigned int nVars)
{
  const unsigned int var  = blockIdx.y;
  const unsigned int fpt  = threadIdx.x;
  const unsigned int face = blockIdx.x;

  if (fpt >= nFpts || face >= nFringe || var > nVars)
    return;

  const unsigned int gfpt = fringe_fpts(fpt, face);
  const unsigned int side = fringe_side(fpt, face);
  U(gfpt, var, side) = U_fringe(fpt, face, var);
}

void unpack_fringe_u_wrapper(mdvector_gpu<double> &U_fringe,
    mdview_gpu<double> &U, mdvector_gpu<unsigned int> &fringe_fpts,
    mdvector_gpu<unsigned int> &fringe_side, unsigned int nFringe, unsigned int nFpts, unsigned int nVars)
{
  int threads = nFpts;
  dim3 blocks(nFringe, nVars);

  unpack_fringe_u<<<blocks, threads>>>(U_fringe, U, fringe_fpts, fringe_side,
      nFringe, nFpts, nVars);
}

__global__
void unpack_fringe_grad(mdvector_gpu<double> dU_fringe,
    mdview_gpu<double> dU, mdvector_gpu<unsigned int> fringe_fpts,
    mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars)
{
  const unsigned int var = blockIdx.y % nVars;
  const unsigned int dim = blockIdx.y / nVars;
  const unsigned int fpt  = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int face = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (fpt >= nFpts || face >= nFringe)
    return;

  const unsigned int gfpt = fringe_fpts(fpt, face);
  const unsigned int side = fringe_side(fpt, face);
  dU(gfpt, var, dim, side) = dU_fringe(fpt, face, var, dim);
}

void unpack_fringe_grad_wrapper(mdvector_gpu<double> &dU_fringe,
    mdview_gpu<double> &dU, mdvector_gpu<unsigned int> &fringe_fpts,
    mdvector_gpu<unsigned int> &fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims)
{
  int threads  = 192;
  int nblock_x = (nFringe * nFpts + threads - 1)/ threads;
  dim3 blocks( nblock_x, nVars*nDims);

  unpack_fringe_grad<<<blocks, threads>>>(dU_fringe, dU, fringe_fpts,
      fringe_side, nFringe, nFpts, nVars);
}
