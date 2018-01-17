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

    case CHAR: /* Characteristic (from PyFR) */
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

      break;
    }

    case SYMMETRY: /* Symmetry */
    case SLIP_WALL: /* Slip Wall */
    {
      /* Rusanov Prescribed */
      if (rus_bias(fpt) == 1)
      {
        double momN = 0.0;

        /* Compute wall normal momentum */
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN += U(0, dim+1, fpt) * norm(dim, fpt);

        if (motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN -= U(0, 0, fpt) * Vg(dim, fpt) * norm(dim, fpt);
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
      }

      /* Rusanov Ghost */
      else
      {
        double momN = 0.0;

        /* Compute wall normal momentum */
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN += U(0, dim+1, fpt) * norm(dim, fpt);

        if (motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN -= U(0, 0, fpt) * Vg(dim, fpt) * norm(dim, fpt);
        }

        U(1, 0, fpt) = U(0, 0, fpt);

        for (unsigned int dim = 0; dim < nDims; dim++)
          /* Set boundary state to reflect normal velocity */
          U(1, dim+1, fpt) = U(0, dim+1, fpt) - 2.0 * momN * norm(dim, fpt);

        U(1, nDims + 1, fpt) = U(0, nDims + 1, fpt);
      }

      break;
    }

    case ISOTHERMAL_NOSLIP: /* Isothermal No-slip Wall */
    {
      double VG[nVars] = {0.0};

      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          VG[dim] = Vg(dim, fpt);
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

      /* Rusanov Prescribed */
      if (rus_bias(fpt) == 1)
        for (unsigned int var = 0; var < nVars; var++)
          U(1, var, fpt) = U_ldg(1, var, fpt);

      break;
    }

    case ISOTHERMAL_NOSLIP_MOVING: /* Moving Isothermal No-slip Wall */
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

      /* Rusanov Prescribed */
      if (rus_bias(fpt) == 1)
        for (unsigned int var = 0; var < nVars; var++)
          U(1, var, fpt) = U_ldg(1, var, fpt);

      break;
    }
    
    case ADIABATIC_NOSLIP: /* Adiabatic No-slip Wall */
    {
      double VG[nVars] = {0.0};
      if (motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          VG[dim] = Vg(dim, fpt);
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

      /* Rusanov Prescribed */
      if (rus_bias(fpt) == 1)
        for (unsigned int var = 0; var < nVars; var++)
          U(1, var, fpt) = U_ldg(1, var, fpt);

      break;
    }

    case ADIABATIC_NOSLIP_MOVING: /* Moving Adiabatic No-slip Wall */
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

      /* Rusanov Prescribed */
      if (rus_bias(fpt) == 1)
        for (unsigned int var = 0; var < nVars; var++)
          U(1, var, fpt) = U_ldg(1, var, fpt);

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
  if(bnd_id == ADIABATIC_NOSLIP || bnd_id == ADIABATIC_NOSLIP_MOVING) /* Adibatic Wall */
  {
    double norm[nDims];

    for (unsigned int dim = 0; dim < nDims; dim++)
      norm[dim] = norm_gfpt(dim, fpt);

    /* Extrapolate density gradient */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      dU(1, dim, 0, fpt) = dU(0, dim, 0, fpt);
    }

    if (nDims == 2)
    {
      /* Compute energy gradient */
      /* Get right states and velocity gradients*/
      double rho = U(0, 0, fpt);
      double momx = U(0, 1, fpt);
      double momy = U(0, 2, fpt);
      double E = U(0, 3, fpt);

      double u = momx / rho;
      double v = momy / rho;
      //double e_int = e / rho - 0.5 * (u*u + v*v);

      double rho_dx = dU(0, 0, 0, fpt);
      double momx_dx = dU(0, 0, 1, fpt);
      double momy_dx = dU(0, 0, 2, fpt);
      double E_dx = dU(0, 0, 3, fpt);

      double rho_dy = dU(0, 1, 0, fpt);
      double momx_dy = dU(0, 1, 1, fpt);
      double momy_dy = dU(0, 1, 2, fpt);
      double E_dy = dU(0, 1, 3, fpt);

      double du_dx = (momx_dx - rho_dx * u) / rho;
      double du_dy = (momx_dy - rho_dy * u) / rho;

      double dv_dx = (momy_dx - rho_dx * v) / rho;
      double dv_dy = (momy_dy - rho_dy * v) / rho;

      /* Option 1: Extrapolate momentum gradients */
      dU(1, 0, 1, fpt) = dU(0, 0, 1, fpt);
      dU(1, 1, 1, fpt) = dU(0, 1, 1, fpt);
      dU(1, 0, 2, fpt) = dU(0, 0, 2, fpt);
      dU(1, 1, 2, fpt) = dU(0, 1, 2, fpt);

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
      dU(1, 0, 3, fpt) = E_dx - dT_dn * norm[0]; 
      dU(1, 1, 3, fpt) = E_dy - dT_dn * norm[1]; 

      /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
      //dU(fpt, 3, 0, 1) = (dT_dx - dT_dn * norm[0]) + rho_dx * U(fpt, 3, 1) / rho; 
      //dU(fpt, 3, 1, 1) = (dT_dy - dT_dn * norm[1]) + rho_dy * U(fpt, 3, 1) / rho; 
    }
    else
    {
      /* Compute energy gradient */
      /* Get right states and velocity gradients*/
      double rho = U(0, 0, fpt);
      double momx = U(0, 1, fpt);
      double momy = U(0, 2, fpt);
      double momz = U(0, 3, fpt);
      double E = U(0, 4, fpt);

      double u = momx / rho;
      double v = momy / rho;
      double w = momz / rho;

      /* Gradients */
      double rho_dx = dU(0, 0, 0, fpt);
      double momx_dx = dU(0, 0, 1, fpt);
      double momy_dx = dU(0, 0, 2, fpt);
      double momz_dx = dU(0, 0, 3, fpt);
      double E_dx = dU(0, 0, 4, fpt);

      double rho_dy = dU(0, 1, 0, fpt);
      double momx_dy = dU(0, 1, 1, fpt);
      double momy_dy = dU(0, 1, 2, fpt);
      double momz_dy = dU(0, 1, 3, fpt);
      double E_dy = dU(0, 1, 4, fpt);

      double rho_dz = dU(0, 2, 0, fpt);
      double momx_dz = dU(0, 2, 1, fpt);
      double momy_dz = dU(0, 2, 2, fpt);
      double momz_dz = dU(0, 2, 3, fpt);
      double E_dz = dU(0, 2, 4, fpt);

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
      dU(1, 0, 1, fpt) = dU(0, 0, 1, fpt);
      dU(1, 1, 1, fpt) = dU(0, 1, 1, fpt);
      dU(1, 2, 1, fpt) = dU(0, 2, 1, fpt);

      dU(1, 0, 2, fpt) = dU(0, 0, 2, fpt);
      dU(1, 1, 2, fpt) = dU(0, 1, 2, fpt);
      dU(1, 2, 2, fpt) = dU(0, 2, 2, fpt);

      dU(1, 0, 3, fpt) = dU(0, 0, 3, fpt);
      dU(1, 1, 3, fpt) = dU(0, 1, 3, fpt);
      dU(1, 2, 3, fpt) = dU(0, 2, 3, fpt);

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
      dU(1, 0, 4, fpt) = E_dx - dT_dn * norm[0]; 
      dU(1, 1, 4, fpt) = E_dy - dT_dn * norm[1]; 
      dU(1, 2, 4, fpt) = E_dz - dT_dn * norm[2]; 

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
        dU(1, dim, n, fpt) = dU(0, dim, n, fpt);
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


template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__
void apply_bcs_dFdU(const mdview_gpu<double> U, mdvector_gpu<double> dUbdU, mdvector_gpu<double> ddUbddU, 
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, bool viscous, double rho_fs, 
    const mdvector_gpu<double> V_fs, double P_fs, double gamma, double R_ref, double T_wall, 
    const mdvector_gpu<double> V_wall, const mdvector_gpu<double> norm, const mdvector_gpu<char> gfpt2bnd)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + nGfpts_int;

  if (fpt >= nGfpts_int + nGfpts_bnd)
    return;

  unsigned int bnd_id = gfpt2bnd(fpt - nGfpts_int);

  /* Apply specified boundary condition */
  switch(bnd_id)
  {
    case PERIODIC:/* Periodic */
    {
      break;
    }

    case SUP_IN: /* Farfield and Supersonic Inlet */
    {
      /* Set solution to freestream */
      /* Extrapolate gradients */
      if (viscous)
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int var = 0; var < nVars; var++)
            ddUbddU(dim, dim, var, var, fpt) = 1;

      break;
    }

    case SUP_OUT: /* Supersonic Outlet */
    {
      // ThrowException("Supersonic Outlet boundary condition not implemented for implicit method");
      break;
    }

    case CHAR: /* Characteristic (from PyFR) */
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
      // if (std::abs(VnR) >= cR && VnL >= 0)
      //   ThrowException("Implicit Char BC not implemented for supersonic flow!")
      // else
      RL = VnL + 2.0 / (gamma - 1) * cL;

      double RB;
      // if (std::abs(VnR) >= cR && VnL < 0)
      //   ThrowException("Implicit Char BC not implemented for supersonic flow!")
      // else
      RB = VnR - 2.0 / (gamma - 1) * cR;

      double cstar = 0.25 * (gamma - 1) * (RL - RB);
      double ustarn = 0.5 * (RL + RB);

      if (nDims == 2)
      {
        double nx = norm(0, fpt);
        double ny = norm(1, fpt);
        double gam = gamma;

        /* Primitive Variables */
        double rhoL = U(0, 0, fpt);
        double uL = U(0, 1, fpt) / U(0, 0, fpt);
        double vL = U(0, 2, fpt) / U(0, 0, fpt);

        double rhoR = U(1, 0, fpt);
        double uR = U(1, 1, fpt) / U(1, 0, fpt);
        double vR = U(1, 2, fpt) / U(1, 0, fpt);

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

          /* Compute dUbdU */
          dUbdU(0, 0, fpt) = a1 * b1;
          dUbdU(1, 0, fpt) = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
          dUbdU(2, 0, fpt) = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
          dUbdU(3, 0, fpt) = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

          dUbdU(0, 1, fpt) = a1 * b2;
          dUbdU(1, 1, fpt) = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
          dUbdU(2, 1, fpt) = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
          dUbdU(3, 1, fpt) = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

          dUbdU(0, 2, fpt) = a1 * b3;
          dUbdU(1, 2, fpt) = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
          dUbdU(2, 2, fpt) = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
          dUbdU(3, 2, fpt) = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

          dUbdU(0, 3, fpt) = 0.5 * rhoR * b4;
          dUbdU(1, 3, fpt) = 0.5 * rhoR * (b4 * uR + a2 * nx);
          dUbdU(2, 3, fpt) = 0.5 * rhoR * (b4 * vR + a2 * ny);
          dUbdU(3, 3, fpt) = 0.5 * rhoR * (b4 * c1 + a2 * c2);
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

          double c1 = 0.5 * b1 * nx - (-VnL * nx + uL) / rhoL;
          double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
          double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
          double c4 = ustarn * nx + uL - VnL * nx;

          double d1 = 0.5 * b1 * ny - (-VnL * ny + vL) / rhoL;
          double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
          double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
          double d4 = ustarn * ny + vL - VnL * ny;

          double e1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
          double e2 = a3 * uL + a4 * b2;
          double e3 = a3 * vL + a4 * b3;
          double e4 = -a3 + a2 * a4;

          double f1 = 0.5 * a1 * (c4*c4 + d4*d4) + a5;

          /* Compute dUbdU */
          dUbdU(0, 0, fpt) = a1 * e1;
          dUbdU(1, 0, fpt) = a1 * e1 * c4 + rhoR * c1;
          dUbdU(2, 0, fpt) = a1 * e1 * d4 + rhoR * d1;
          dUbdU(3, 0, fpt) = rhoR * (c1*c4 + d1*d4) + e1 * f1 + a6 * b1;

          dUbdU(0, 1, fpt) = a1 * e2;
          dUbdU(1, 1, fpt) = a1 * e2 * c4 + rhoR * c2;
          dUbdU(2, 1, fpt) = a1 * e2 * d4 + rhoR * d2;
          dUbdU(3, 1, fpt) = rhoR * (c2*c4 + d2*d4) + e2 * f1 + a6 * b2;

          dUbdU(0, 2, fpt) = a1 * e3;
          dUbdU(1, 2, fpt) = a1 * e3 * c4 + rhoR * c3;
          dUbdU(2, 2, fpt) = a1 * e3 * d4 + rhoR * d3;
          dUbdU(3, 2, fpt) = rhoR * (c3*c4 + d3*d4) + e3 * f1 + a6 * b3;

          dUbdU(0, 3, fpt) = a1 * e4;
          dUbdU(1, 3, fpt) = a1 * e4 * c4 + 0.5 * rhoR * a2 * nx;
          dUbdU(2, 3, fpt) = a1 * e4 * d4 + 0.5 * rhoR * a2 * ny;
          dUbdU(3, 3, fpt) = 0.5 * rhoR * a2 * (c4*nx + d4*ny) + e4 * f1 + a2 * a6;
        }
      }

      else if (nDims == 3)
      {
        double nx = norm(0, fpt);
        double ny = norm(1, fpt);
        double nz = norm(2, fpt);
        double gam = gamma;

        /* Primitive Variables */
        double rhoL = U(0, 0, fpt);
        double uL = U(0, 1, fpt) / U(0, 0, fpt);
        double vL = U(0, 2, fpt) / U(0, 0, fpt);
        double wL = U(0, 3, fpt) / U(0, 0, fpt);

        double rhoR = U(1, 0, fpt);
        double uR = U(1, 1, fpt) / U(1, 0, fpt);
        double vR = U(1, 2, fpt) / U(1, 0, fpt);
        double wR = U(1, 3, fpt) / U(1, 0, fpt);

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

          /* Compute dUbdU */
          dUbdU(0, 0, fpt) = a1 * b1;
          dUbdU(1, 0, fpt) = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
          dUbdU(2, 0, fpt) = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
          dUbdU(3, 0, fpt) = a1 * b1 * wR + 0.5 * rhoR * b1 * nz;
          dUbdU(4, 0, fpt) = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

          dUbdU(0, 1, fpt) = a1 * b2;
          dUbdU(1, 1, fpt) = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
          dUbdU(2, 1, fpt) = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
          dUbdU(3, 1, fpt) = a1 * b2 * wR + 0.5 * rhoR * b2 * nz;
          dUbdU(4, 1, fpt) = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

          dUbdU(0, 2, fpt) = a1 * b3;
          dUbdU(1, 2, fpt) = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
          dUbdU(2, 2, fpt) = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
          dUbdU(3, 2, fpt) = a1 * b3 * wR + 0.5 * rhoR * b3 * nz;
          dUbdU(4, 2, fpt) = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

          dUbdU(0, 3, fpt) = a1 * b4;
          dUbdU(1, 3, fpt) = a1 * b4 * uR + 0.5 * rhoR * b4 * nx;
          dUbdU(2, 3, fpt) = a1 * b4 * vR + 0.5 * rhoR * b4 * ny;
          dUbdU(3, 3, fpt) = a1 * b4 * wR + 0.5 * rhoR * b4 * nz;
          dUbdU(4, 3, fpt) = a1 * b4 * c1 + 0.5 * rhoR * b4 * c2;

          dUbdU(0, 4, fpt) = 0.5 * rhoR * b5;
          dUbdU(1, 4, fpt) = 0.5 * rhoR * (b5 * uR + a2 * nx);
          dUbdU(2, 4, fpt) = 0.5 * rhoR * (b5 * vR + a2 * ny);
          dUbdU(3, 4, fpt) = 0.5 * rhoR * (b5 * wR + a2 * nz);
          dUbdU(4, 4, fpt) = 0.5 * rhoR * (b5 * c1 + a2 * c2);
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

          double c1 = 0.5 * b1 * nx - (-VnL * nx + uL) / rhoL;
          double c2 = 0.5 * b2 * nx + (1.0 - nx*nx) / rhoL;
          double c3 = 0.5 * b3 * nx - nx * ny / rhoL;
          double c4 = 0.5 * b4 * nx - nx * nz / rhoL;
          double c5 = ustarn * nx + uL - VnL * nx;

          double d1 = 0.5 * b1 * ny - (-VnL * ny + vL) / rhoL;
          double d2 = 0.5 * b2 * ny - nx * ny / rhoL;
          double d3 = 0.5 * b3 * ny + (1.0 - ny*ny) / rhoL;
          double d4 = 0.5 * b4 * ny - ny * nz / rhoL;
          double d5 = ustarn * ny + vL - VnL * ny;

          double e1 = 0.5 * b1 * nz - (-VnL * nz + wL) / rhoL;
          double e2 = 0.5 * b2 * nz - nx * nz / rhoL;
          double e3 = 0.5 * b3 * nz - ny * nz / rhoL;
          double e4 = 0.5 * b4 * nz + (1.0 - nz*nz) / rhoL;
          double e5 = ustarn * nz + wL - VnL * nz;

          double f1 = 1.0 / rhoL - 0.5 * a3 * momF / rhoL + a4 * b1;
          double f2 = a3 * uL + a4 * b2;
          double f3 = a3 * vL + a4 * b3;
          double f4 = a3 * wL + a4 * b4;
          double f5 = -a3 + a2 * a4;

          double g1 = 0.5 * a1 * (c5*c5 + d5*d5 + e5*e5) + a5;

          /* Compute dUbdU */
          dUbdU(0, 0, fpt) = a1 * f1;
          dUbdU(1, 0, fpt) = a1 * f1 * c5 + rhoR * c1;
          dUbdU(2, 0, fpt) = a1 * f1 * d5 + rhoR * d1;
          dUbdU(3, 0, fpt) = a1 * f1 * e5 + rhoR * e1;
          dUbdU(4, 0, fpt) = rhoR * (c1*c5 + d1*d5 + e1*e5) + f1 * g1 + a6 * b1;

          dUbdU(0, 1, fpt) = a1 * f2;
          dUbdU(1, 1, fpt) = a1 * f2 * c5 + rhoR * c2;
          dUbdU(2, 1, fpt) = a1 * f2 * d5 + rhoR * d2;
          dUbdU(3, 1, fpt) = a1 * f2 * e5 + rhoR * e2;
          dUbdU(4, 1, fpt) = rhoR * (c2*c5 + d2*d5 + e2*e5) + f2 * g1 + a6 * b2;

          dUbdU(0, 2, fpt) = a1 * f3;
          dUbdU(1, 2, fpt) = a1 * f3 * c5 + rhoR * c3;
          dUbdU(2, 2, fpt) = a1 * f3 * d5 + rhoR * d3;
          dUbdU(3, 2, fpt) = a1 * f3 * e5 + rhoR * e3;
          dUbdU(4, 2, fpt) = rhoR * (c3*c5 + d3*d5 + e3*e5) + f3 * g1 + a6 * b3;

          dUbdU(0, 3, fpt) = a1 * f4;
          dUbdU(1, 3, fpt) = a1 * f4 * c5 + rhoR * c4;
          dUbdU(2, 3, fpt) = a1 * f4 * d5 + rhoR * d4;
          dUbdU(3, 3, fpt) = a1 * f4 * e5 + rhoR * e4;
          dUbdU(4, 3, fpt) = rhoR * (c4*c5 + d4*d5 + e4*e5) + f4 * g1 + a6 * b4;

          dUbdU(0, 4, fpt) = a1 * f5;
          dUbdU(1, 4, fpt) = a1 * f5 * c5 + 0.5 * rhoR * a2 * nx;
          dUbdU(2, 4, fpt) = a1 * f5 * d5 + 0.5 * rhoR * a2 * ny;
          dUbdU(3, 4, fpt) = a1 * f5 * e5 + 0.5 * rhoR * a2 * nz;
          dUbdU(4, 4, fpt) = 0.5 * rhoR * a2 * (c5*nx + d5*ny + e5*nz) + f5 * g1 + a2 * a6;
        }
      }

      /* Extrapolate gradients */
      if (viscous)
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int var = 0; var < nVars; var++)
            ddUbddU(dim, dim, var, var, fpt) = 1;

      break;
    }

    case SYMMETRY: /* Symmetry */
    case SLIP_WALL: /* Slip Wall */
    {
      if (nDims == 2)
      {
        double nx = norm(0, fpt);
        double ny = norm(1, fpt);

        /* Primitive Variables */
        double uL = U(0, 1, fpt) / U(0, 0, fpt);
        double vL = U(0, 2, fpt) / U(0, 0, fpt);

        double uR = U(1, 1, fpt) / U(1, 0, fpt);
        double vR = U(1, 2, fpt) / U(1, 0, fpt);

        /* Compute dUbdU */
        dUbdU(0, 0, fpt) = 1;
        dUbdU(3, 0, fpt) = 0.5 * (uL*uL + vL*vL - uR*uR - vR*vR);

        dUbdU(1, 1, fpt) = 1.0-nx*nx;
        dUbdU(2, 1, fpt) = -nx*ny;
        dUbdU(3, 1, fpt) = -uL + (1.0-nx*nx)*uR - nx*ny*vR;

        dUbdU(1, 2, fpt) = -nx*ny;
        dUbdU(2, 2, fpt) = 1.0-ny*ny;
        dUbdU(3, 2, fpt) = -vL - nx*ny*uR + (1.0-ny*ny)*vR;

        dUbdU(3, 3, fpt) = 1;
      }

      else if (nDims == 3)
      {
        double nx = norm(0, fpt);
        double ny = norm(1, fpt);
        double nz = norm(2, fpt);

        /* Primitive Variables */
        double uL = U(0, 1, fpt) / U(0, 0, fpt);
        double vL = U(0, 2, fpt) / U(0, 0, fpt);
        double wL = U(0, 3, fpt) / U(0, 0, fpt);

        double uR = U(1, 1, fpt) / U(1, 0, fpt);
        double vR = U(1, 2, fpt) / U(1, 0, fpt);
        double wR = U(1, 3, fpt) / U(1, 0, fpt);

        /* Compute dUbdU */
        dUbdU(0, 0, fpt) = 1;
        dUbdU(4, 0, fpt) = 0.5 * (uL*uL + vL*vL + wL*wL - uR*uR - vR*vR - wR*wR);

        dUbdU(1, 1, fpt) = 1.0-nx*nx;
        dUbdU(2, 1, fpt) = -nx*ny;
        dUbdU(3, 1, fpt) = -nx*nz;
        dUbdU(4, 1, fpt) = -uL + (1.0-nx*nx)*uR - nx*ny*vR - nx*nz*wR;

        dUbdU(1, 2, fpt) = -nx*ny;
        dUbdU(2, 2, fpt) = 1.0-ny*ny;
        dUbdU(3, 2, fpt) = -ny*nz;
        dUbdU(4, 2, fpt) = -vL - nx*ny*uR + (1.0-ny*ny)*vR - ny*nz*wR;

        dUbdU(1, 3, fpt) = -nx*nz;
        dUbdU(2, 3, fpt) = -ny*nz;
        dUbdU(3, 3, fpt) = 1.0-nz*nz;
        dUbdU(4, 3, fpt) = -wL - nx*nz*uR - ny*nz*vR + (1.0-nz*nz)*wR;

        dUbdU(4, 4, fpt) = 1;
      }

      break;
    }

    case ISOTHERMAL_NOSLIP: /* Isothermal No-slip Wall */
    {
      dUbdU(0, 0, fpt) = 1;
      dUbdU(nDims+1, 0, fpt) = (R_ref * T_wall) / (gamma-1.0);

      /* Extrapolate gradients */
      if (viscous)
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int var = 0; var < nVars; var++)
            ddUbddU(dim, dim, var, var, fpt) = 1;

      break;
    }

    case ISOTHERMAL_NOSLIP_MOVING: /* Isothermal No-slip Wall, moving */
    {
      double Vsq = 0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        Vsq += V_wall(dim) * V_wall(dim);

      dUbdU(0, 0, fpt) = 1;
      for (unsigned int dim = 0; dim < nDims; dim++)
        dUbdU(dim+1, 0, fpt) = V_wall(dim);
      dUbdU(nDims+1, 0, fpt) = (R_ref * T_wall) / (gamma-1.0) + 0.5 * Vsq;

      /* Extrapolate gradients */
      if (viscous)
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int var = 0; var < nVars; var++)
            ddUbddU(dim, dim, var, var, fpt) = 1;

      break;
    }

    case ADIABATIC_NOSLIP: /* Adiabatic No-slip Wall */
    {
      // if (nDims == 3)
      //   ThrowException("3D Adiabatic No-slip Wall (prescribed) boundary condition not implemented for implicit method");

      double nx = norm(0, fpt);
      double ny = norm(1, fpt);

      /* Primitive Variables */
      double rhoL = U(0, 0, fpt);
      double uL = U(0, 1, fpt) / U(0, 0, fpt);
      double vL = U(0, 2, fpt) / U(0, 0, fpt);
      double eL = U(0, 3, fpt);

      /* Compute dUbdU */
      dUbdU(0, 0, fpt) = 1;
      dUbdU(3, 0, fpt) = 0.5 * (uL*uL + vL*vL);

      dUbdU(3, 1, fpt) = -uL;

      dUbdU(3, 2, fpt) = -vL;

      dUbdU(3, 3, fpt) = 1;

      if (viscous)
      {
        /* Compute dUxR/dUxL */
        ddUbddU(0, 0, 0, 0, fpt) = 1;
        ddUbddU(0, 0, 3, 0, fpt) = nx*nx * (eL / rhoL - (uL*uL + vL*vL));

        ddUbddU(0, 0, 1, 1, fpt) = 1;
        ddUbddU(0, 0, 3, 1, fpt) = nx*nx * uL;

        ddUbddU(0, 0, 2, 2, fpt) = 1;
        ddUbddU(0, 0, 3, 2, fpt) = nx*nx * vL;

        ddUbddU(0, 0, 3, 3, fpt) = 1.0 - nx*nx;

        /* Compute dUyR/dUxL */
        ddUbddU(1, 0, 3, 0, fpt) = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddUbddU(1, 0, 3, 1, fpt) = nx*ny * uL;

        ddUbddU(1, 0, 3, 2, fpt) = nx*ny * vL;

        ddUbddU(1, 0, 3, 3, fpt) = -nx * ny;

        /* Compute dUxR/dUyL */
        ddUbddU(0, 1, 3, 0, fpt) = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddUbddU(0, 1, 3, 1, fpt) = nx*ny * uL;

        ddUbddU(0, 1, 3, 2, fpt) = nx*ny * vL;

        ddUbddU(0, 1, 3, 3, fpt) = -nx * ny;

        /* Compute dUyR/dUyL */
        ddUbddU(1, 1, 0, 0, fpt) = 1;
        ddUbddU(1, 1, 3, 0, fpt) = ny*ny * (eL / rhoL - (uL*uL + vL*vL));

        ddUbddU(1, 1, 1, 1, fpt) = 1;
        ddUbddU(1, 1, 3, 1, fpt) = ny*ny * uL;

        ddUbddU(1, 1, 2, 2, fpt) = 1;
        ddUbddU(1, 1, 3, 2, fpt) = ny*ny * vL;

        ddUbddU(1, 1, 3, 3, fpt) = 1.0 - ny*ny;
      }

      break;
    }

    case ADIABATIC_NOSLIP_MOVING: /* Adiabatic No-slip Wall, moving */
    {
      // ThrowException("Adiabatic No-slip Wall, moving boundary condition not implemented for implicit method");
      break;
    }

    default:
    {
      // ThrowException("Boundary condition not implemented for implicit method");
      break;
    }
  }
}

void apply_bcs_dFdU_wrapper(mdview_gpu<double> &U, mdvector_gpu<double> &dUbdU, mdvector_gpu<double> &ddUbddU,
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, 
    bool viscous, double rho_fs, mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_wall, 
    mdvector_gpu<double> &V_wall, mdvector_gpu<double> &norm, mdvector_gpu<char> &gfpt2bnd, unsigned int equation)
{
  if (nGfpts_bnd == 0) return;

  unsigned int threads = 128;
  unsigned int blocks = (nGfpts_bnd + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      apply_bcs_dFdU<1, 2, AdvDiff><<<blocks, threads>>>(U, dUbdU, ddUbddU, nFpts, nGfpts_int, nGfpts_bnd, viscous, 
          rho_fs, V_fs, P_fs, gamma, R_ref, T_wall, V_wall, norm, gfpt2bnd);
    else
      apply_bcs_dFdU<1, 3, AdvDiff><<<blocks, threads>>>(U, dUbdU, ddUbddU, nFpts, nGfpts_int, nGfpts_bnd, viscous, 
          rho_fs, V_fs, P_fs, gamma, R_ref, T_wall, V_wall, norm, gfpt2bnd);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      apply_bcs_dFdU<4, 2, EulerNS><<<blocks, threads>>>(U, dUbdU, ddUbddU, nFpts, nGfpts_int, nGfpts_bnd, viscous, 
          rho_fs, V_fs, P_fs, gamma, R_ref, T_wall, V_wall, norm, gfpt2bnd);
    else
      apply_bcs_dFdU<5, 3, EulerNS><<<blocks, threads>>>(U, dUbdU, ddUbddU, nFpts, nGfpts_int, nGfpts_bnd, viscous, 
          rho_fs, V_fs, P_fs, gamma, R_ref, T_wall, V_wall, norm, gfpt2bnd);
  }
}


template <unsigned int nDims, unsigned int nVars>
__global__
void compute_common_U_LDG(const mdview_gpu<double> U, mdview_gpu<double> Ucomm, 
    const mdvector_gpu<double> norm, double beta, unsigned int nFpts,
    const mdvector_gpu<char> LDG_bias, unsigned int startFpt, unsigned int endFpt,
    const mdvector_gpu<char> flip_beta, bool overset = false, const int* iblank = NULL)
{
    const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

    if (fpt >= endFpt)
      return;
    
    double UL[nVars] = {};
    double UR[nVars]= {};

    if (overset)
      if (iblank[fpt] == 0)
        return;

    beta *= flip_beta(fpt);

    if (LDG_bias(fpt) == 0)
    {
      /* Get left and/or right state variables */
      if (beta == 0.5)
      {
        for (unsigned int n = 0; n < nVars; n++)
          UR[n] = U(1, n, fpt);
      }
      else if (beta == -0.5)
      {
        for (unsigned int n = 0; n < nVars; n++)
          UL[n] = U(0, n, fpt);
      }
      else
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          UL[n] = U(0, n, fpt); UR[n] = U(1, n, fpt);
        }
      }

      for (unsigned int n = 0; n < nVars; n++)
      {
        double UC = 0.5*(UL[n] + UR[n]) - beta*(UL[n] - UR[n]);
        Ucomm(0, n, fpt) = UC;
        Ucomm(1, n, fpt) = UC;
      }
    }
    /* If on boundary, don't use beta */
    else
    {
      for (unsigned int n = 0; n < nVars; n++)
        UR[n] = U(1, n, fpt);

      for (unsigned int n = 0; n < nVars; n++)
      {
        Ucomm(0, n, fpt) = UR[n];
        Ucomm(1, n, fpt) = UR[n];
      }
    }
}

void compute_common_U_LDG_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, mdvector_gpu<char> &LDG_bias, unsigned int startFpt,
    unsigned int endFpt, mdvector_gpu<char> &flip_beta, bool overset, int* iblank) 
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_common_U_LDG<2, 1><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, flip_beta);
    else
      compute_common_U_LDG<3, 1><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, flip_beta, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_U_LDG<2, 4><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, flip_beta);
    else
      compute_common_U_LDG<3, 5><<<blocks, threads>>>(U, Ucomm, norm, beta, nFpts,
          LDG_bias, startFpt, endFpt, flip_beta, overset, iblank);
  }

}


template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__device__ __forceinline__
void rusanov_flux(double UL[nVars], double UR[nVars], double Fcomm[nVars], double Vg[nDims],
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

    compute_Fconv_AdvDiff<nVars, nDims>(UL, FL, A, Vg);
    compute_Fconv_AdvDiff<nVars, nDims>(UR, FR, A, Vg);
  }
  else if (equation == EulerNS)
  {
    double P;
    compute_Fconv_EulerNS<nVars, nDims>(UL, FL, Vg, P, gamma);
    double aL = std::sqrt(gamma * P / UL[0]);
    PL = P;
    compute_Fconv_EulerNS<nVars, nDims>(UR, FR, Vg, P, gamma);
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
      Fcomm[n] += 0.5 * (FnL[n] + FnR[n]) + tau * (UL[n] - UR[n]) + beta * (FnL[n] - FnR[n]);
  }
  /* If boundary, use right state only */
  else
  {
    for (unsigned int n = 0; n < nVars; n++)
      Fcomm[n] += FnR[n] + tau * (UL[n] - UR[n]);
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__ 
void compute_common_F(mdview_gpu<double> U, mdview_gpu<double> U_ldg, mdview_gpu<double> dU,
    mdview_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> AdvDiff_A, 
    mdvector_gpu<double> norm_gfpts, mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<double> diffCo,
    mdvector_gpu<char> rus_bias, mdvector_gpu<char> LDG_bias,  mdvector_gpu<double> dA_in, mdvector_gpu<double> Vg, double AdvDiff_D, double gamma, double rus_k, 
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, unsigned int nFpts_int,
    unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, bool viscous, mdvector_gpu<char> flip_beta, bool motion, bool overset, int* iblank)
{

  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  if (overset and iblank[fpt] == 0)
    return;

  double UL[nVars]; double UR[nVars];
  double Fc[nVars];
  double norm[nDims];
  double V[nDims] = {0.0};
  double Vgn = 0.0;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    norm[dim] = norm_gfpts(dim, fpt);
  }

  if (motion)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      V[dim] = Vg(dim, fpt);
      Vgn += V[dim] * norm[dim];
    }
  }

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    UL[n] = U(0, n, fpt); UR[n] = U(1, n, fpt);
  }

  /* Compute convective contribution to common flux */
  double PL, PR;
  rusanov_flux<nVars, nDims, equation>(UL, UR, Fc, V, PL, PR, norm, waveSp_gfpts(fpt),
      AdvDiff_A.data(), Vgn, gamma, rus_k, rus_bias(fpt));

  if (equation == EulerNS)
  {
    //if (fpt >= nFpts_int) P(0, fpt) = PL; // Write out pressure on boundary only
    P(0, fpt) = PL;
    P(1, fpt) = PR;
  }

  if (viscous)
  {
    /* Compute viscous contribution to common flux */
    double dUL[nVars][nDims] = {{}};
    double dUR[nVars][nDims] = {{}};
    char LDG_bias_ = LDG_bias(fpt);

    beta *= flip_beta(fpt);

    if (LDG_bias_ == 0)
    {
      /* Get left and/or right gradients */
      if (beta == 0.5)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int n = 0; n < nVars; n++)
            dUL[n][dim] = dU(0, dim, n, fpt);
      }
      else if (beta == -0.5)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int n = 0; n < nVars; n++)
            dUR[n][dim] = dU(1, dim, n, fpt);
      }
      else
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          for (unsigned int n = 0; n < nVars; n++)
          {
            dUL[n][dim] = dU(0, dim, n, fpt); dUR[n][dim] = dU(1, dim, n, fpt);
          }
        }
      }
    }
    else
    {
      /* Get right gradients */
      for (unsigned int dim = 0; dim < nDims; dim++)
        for (unsigned int n = 0; n < nVars; n++)
          dUR[n][dim] = dU(1, dim, n, fpt);
 
      /* Replace right state with LDG right state */
      for (unsigned int n = 0; n < nVars; n++)
        UR[n] = U_ldg(1, n, fpt); 
    }

    LDG_flux<nVars, nDims, equation>(UL, UR, dUL, dUR, Fc, norm, AdvDiff_D, diffCo(fpt), gamma,
        prandtl, mu, rt, c_sth, fix_vis, LDG_bias_, beta, tau);
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
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, unsigned int nFpts_int, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, 
    bool viscous, mdvector_gpu<char> &flip_beta, bool motion, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_common_F<1, 2, AdvDiff><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, nFpts_int, fconv_type, fvisc_type, startFpt, endFpt, viscous, flip_beta, motion, overset, iblank);
    else
      compute_common_F<1, 3, AdvDiff><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, nFpts_int, fconv_type, fvisc_type, startFpt, endFpt, viscous, flip_beta, motion, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_F<4, 2, EulerNS><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, nFpts_int, fconv_type, fvisc_type, startFpt, endFpt, viscous, flip_beta, motion, overset, iblank);
    else
      compute_common_F<5, 3, EulerNS><<<blocks, threads>>>(U, U_ldg, dU, Fcomm, P, AdvDiff_A, norm, waveSp, diffCo, rus_bias, LDG_bias, dA, Vg, AdvDiff_D, gamma, rus_k, 
        mu, prandtl, rt, c_sth, fix_vis, beta, tau, nFpts, nFpts_int, fconv_type, fvisc_type, startFpt, endFpt, viscous, flip_beta, motion, overset, iblank);
  }
}


template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__device__ __forceinline__
void rusanov_dFdU(double UL[nVars], double UR[nVars], double dURdUL[nVars][nVars], 
    double dFcdUL[nVars][nVars], double dFcdUR[nVars][nVars], double PL, double PR, 
    double norm[nDims], double waveSp, double *AdvDiff_A, double gamma, double rus_k, 
    char rus_bias)
{
  double dFdUL[nVars][nVars][nDims] = {0.0};
  double dFdUR[nVars][nVars][nDims] = {0.0};
  double dwSdUL[nVars] = {0.0};
  double dwSdUR[nVars] = {0.0};
  double dFndUL[nVars][nVars] = {0.0};
  double dFndUR[nVars][nVars] = {0.0};

  /* Compute flux Jacobians and numerical wavespeed derivative */
  if (equation == AdvDiff)
  {
    double A[nDims];
    for (unsigned int dim = 0; dim < nDims; dim++)
      A[dim] = *(AdvDiff_A + dim);

    compute_dFdUconv_AdvDiff<nVars, nDims>(dFdUL, A);
    compute_dFdUconv_AdvDiff<nVars, nDims>(dFdUR, A);
  }
  else if (equation == EulerNS)
  {
    compute_dFdUconv_EulerNS<nVars, nDims>(UL, dFdUL, gamma);
    compute_dFdUconv_EulerNS<nVars, nDims>(UR, dFdUR, gamma);

    /* Compute speed of sound */
    double aL = std::sqrt(gamma * PL / UL[0]);
    double aR = std::sqrt(gamma * PR / UR[0]);

    /* Compute normal velocities */
    double VnL = 0.0; double VnR = 0.0;
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      VnL += UL[dim+1]/UL[0] * norm[dim];
      VnR += UR[dim+1]/UR[0] * norm[dim];
    }

    /* Compute numerical wavespeed derivative */
    double wSL = std::abs(VnL) + aL;
    double wSR = std::abs(VnR) + aR;
    if (wSL > wSR)
    {
      /* Determine direction */
      int sgn = (VnL > 0) ? 1 : -1;

      /* Primitive Variables */
      double rho = UL[0];
      double V[nDims];
      for (unsigned int dim = 0; dim < nDims; dim++)
        V[dim] = UL[dim+1] / UL[0];

      /* Compute wavespeed derivative */
      dwSdUL[0] = -sgn*VnL/rho - aL/(2.0*rho);
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dwSdUL[0] += gamma * (gamma-1.0) * V[dim]*V[dim] / (4.0*aL*rho);
        dwSdUL[dim+1] = sgn*norm[dim]/rho - gamma * (gamma-1.0) * V[dim] / (2.0*aL*rho);
      }
      dwSdUL[nDims+1] = gamma * (gamma-1.0) / (2.0*aL*rho);
    }

    else
    {
      /* Determine direction */
      int sgn = (VnR > 0) ? 1 : -1;

      /* Primitive Variables */
      double rho = UR[0];
      double V[nDims];
      for (unsigned int dim = 0; dim < nDims; dim++)
        V[dim] = UR[dim+1] / UR[0];

      /* Compute wavespeed derivative */
      dwSdUR[0] = -sgn*VnR/rho - aR/(2.0*rho);
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dwSdUR[0] += gamma * (gamma-1.0) * V[dim]*V[dim] / (4.0*aR*rho);
        dwSdUR[dim+1] = sgn*norm[dim]/rho - gamma * (gamma-1.0) * V[dim] / (2.0*aR*rho);
      }
      dwSdUR[nDims+1] = gamma * (gamma-1.0) / (2.0*aR*rho);
    }
  }

  /* Get interface-normal dFdU components  (from L to R) */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dFndUL[vari][varj] += dFdUL[vari][varj][dim] * norm[dim];
        dFndUR[vari][varj] += dFdUR[vari][varj][dim] * norm[dim];
      }

  /* Compute common dFdU */
  if (rus_bias == 0) /* Upwinded */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        if (vari == varj)
        {
          dFcdUL[vari][varj] = 0.5 * dFndUL[vari][varj] - 0.5 * ((UR[vari]-UL[vari]) * dwSdUL[varj] - waveSp) * (1.0-rus_k);
          dFcdUR[vari][varj] = 0.5 * dFndUR[vari][varj] - 0.5 * ((UR[vari]-UL[vari]) * dwSdUR[varj] + waveSp) * (1.0-rus_k);
        }
        else
        {
          dFcdUL[vari][varj] = 0.5 * dFndUL[vari][varj] - 0.5 * (UR[vari]-UL[vari]) * dwSdUL[varj] * (1.0-rus_k);
          dFcdUR[vari][varj] = 0.5 * dFndUR[vari][varj] - 0.5 * (UR[vari]-UL[vari]) * dwSdUR[varj] * (1.0-rus_k);
        }
      }

  else if (rus_bias == 2) /* Central */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        dFcdUL[vari][varj] = 0.5 * dFndUL[vari][varj];
        dFcdUR[vari][varj] = 0.5 * dFndUR[vari][varj];
      }

  else if (rus_bias == 1) /* Set flux state */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        double dFcdU = 0;
        for (unsigned int vark = 0; vark < nVars; vark++)
          dFcdU += dFndUR[vari][vark] * dURdUL[vark][varj];

        dFcdUL[vari][varj] = dFcdU;
        dFcdUR[vari][varj] = dFcdU;
      }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__device__ __forceinline__
void LDG_dFdU(double UL[nVars], double UR[nVars], double dUL[nVars][nDims], double dUR[nVars][nDims], 
    double dURdUL[nVars][nVars], double ddURddUL[nDims][nDims][nVars][nVars], double dFcdUL[nVars][nVars], 
    double dFcdUR[nVars][nVars], double dFcddUL[nDims][nVars][nVars], double dFcddUR[nDims][nVars][nVars],
    double norm[nDims], double AdvDiff_D, double gamma, double prandtl, double mu, int LDG_bias, 
    double beta, double tau)
{
  double dFdUL[nVars][nVars][nDims] = {0};
  double dFdUR[nVars][nVars][nDims] = {0};
  double dFddUL[nVars][nVars][nDims][nDims] = {0};
  double dFddUR[nVars][nVars][nDims][nDims] = {0};
  double dFndUL[nVars][nVars] = {0};
  double dFndUR[nVars][nVars] = {0};
  double dFnddUL[nVars][nVars][nDims] = {0};
  double dFnddUR[nVars][nVars][nDims] = {0};

  /* Compute viscous flux Jacobians */
  if (equation == AdvDiff)
  {
    compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddUL, AdvDiff_D);
    compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddUR, AdvDiff_D);
  }
  else if (equation == EulerNS)
  {
    compute_dFdUvisc_EulerNS_add<nVars, nDims>(UL, dUL, dFdUL, gamma, prandtl, mu);
    compute_dFdUvisc_EulerNS_add<nVars, nDims>(UR, dUR, dFdUR, gamma, prandtl, mu);

    compute_dFddUvisc_EulerNS<nVars, nDims>(UL, dFddUL, gamma, prandtl, mu);
    compute_dFddUvisc_EulerNS<nVars, nDims>(UR, dFddUR, gamma, prandtl, mu);
  }

  /* Get interface-normal dFdU components (from L to R) */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dFndUL[vari][varj] += dFdUL[vari][varj][dim] * norm[dim];
        dFndUR[vari][varj] += dFdUR[vari][varj][dim] * norm[dim];
      }

  /* Get interface-normal dFddU components (from L to R) */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
      for (unsigned int dimi = 0; dimi < nDims; dimi++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
        {
          dFnddUL[vari][varj][dimj] += dFddUL[vari][varj][dimi][dimj] * norm[dimi];
          dFnddUR[vari][varj][dimj] += dFddUR[vari][varj][dimi][dimj] * norm[dimi];
        }

  /* Compute common normal viscous dFdU */
  /* If interior, use central */
  if (LDG_bias == 0)
  {
    /* Compute common viscous flux Jacobian (dFcdU) */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        dFcdUL[vari][varj] += (0.5 + beta) * dFndUL[vari][varj];
        dFcdUR[vari][varj] += (0.5 - beta) * dFndUR[vari][varj];

        if (vari == varj)
        {
          dFcdUL[vari][varj] += tau;
          dFcdUR[vari][varj] -= tau;
        }
      }

    /* Compute common viscous flux Jacobian (dFcddU) */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
        {
          dFcddUL[dimj][vari][varj] = (0.5 + beta) * dFnddUL[vari][varj][dimj];
          dFcddUR[dimj][vari][varj] = (0.5 - beta) * dFnddUR[vari][varj][dimj];
        }
  }

  /* If boundary, use right state only */
  else
  {
    /* Compute common viscous flux Jacobian (dFcdU) */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        /* Compute boundary dFdU */
        double dFcdU = 0;
        for (unsigned int vark = 0; vark < nVars; vark++)
          dFcdU += dFndUR[vari][vark] * dURdUL[vark][varj];

        if (vari == varj)
          dFcdU += tau;
        dFcdU -= tau * dURdUL[vari][varj];

        dFcdUL[vari][varj] += dFcdU;
      }

    /* Compute boundary dFddU */
    for (unsigned int dimj = 0; dimj < nDims; dimj++)
      for (unsigned int dimk = 0; dimk < nDims; dimk++)
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            for (unsigned int vark = 0; vark < nVars; vark++)
              dFcddUL[dimj][vari][varj] += dFnddUR[vari][vark][dimk] * ddURddUL[dimk][dimj][vark][varj];
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
__global__ 
void compute_common_dFdU(mdview_gpu<double> U, mdview_gpu<double> dU, mdview_gpu<double> dFcdU, 
    mdview_gpu<double> dUcdU, mdview_gpu<double> dFcddU, mdvector_gpu<double> dUbdU, mdvector_gpu<double> ddUbddU, 
    mdvector_gpu<double> P, mdvector_gpu<double> AdvDiff_A, mdvector_gpu<double> norm_gfpts, 
    mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<char> rus_bias, mdvector_gpu<char> LDG_bias, 
    mdvector_gpu<double> dA_in, double AdvDiff_D, double gamma, double rus_k, double mu, double prandtl, 
    double beta, double tau, unsigned int startFpt, unsigned int endFpt, bool viscous, mdvector_gpu<char> flip_beta)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x + startFpt;

  if (fpt >= endFpt)
    return;

  double UL[nVars]; double UR[nVars];
  double dURdUL[nVars][nVars];
  double dFcdUL[nVars][nVars];
  double dFcdUR[nVars][nVars];
  double norm[nDims];

  for (unsigned int dim = 0; dim < nDims; dim++)
    norm[dim] = norm_gfpts(dim, fpt);

  /* Get left and right state variables */
  for (unsigned int n = 0; n < nVars; n++)
  {
    UL[n] = U(0, n, fpt); UR[n] = U(1, n, fpt);
  }

  /* Get boundary Jacobian of solution */
  if (rus_bias(fpt) == 1) /* Set flux state */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        dURdUL[vari][varj] = dUbdU(vari, varj, fpt);

  /* Compute convective contribution to common dFdU */
  double PL, PR;
  if (equation == EulerNS)
  {
    PL = P(0, fpt);
    PR = P(1, fpt);
  }
  rusanov_dFdU<nVars, nDims, equation>(UL, UR, dURdUL, dFcdUL, dFcdUR, 
      PL, PR, norm, waveSp_gfpts(fpt), AdvDiff_A.data(), gamma, rus_k, 
      rus_bias(fpt));

  double dAL = dA_in(0, fpt);
  double dAR = dA_in(1, fpt);
  if (viscous)
  {
    double dUL[nVars][nDims];
    double dUR[nVars][nDims];
    double ddURddUL[nDims][nDims][nVars][nVars];
    double dFcddUL[nDims][nVars][nVars] = {0};
    double dFcddUR[nDims][nVars][nVars] = {0};
    char LDG_bias_ = LDG_bias(fpt);

    beta *= flip_beta(fpt);

    /* Get left and right state gradients */
    for (unsigned int dim = 0; dim < nDims; dim++)
      for (unsigned int n = 0; n < nVars; n++)
      {
        dUL[n][dim] = dU(0, dim, n, fpt); 
        dUR[n][dim] = dU(1, dim, n, fpt);
      }

    /* If interior, use central */
    if (LDG_bias_ == 0)
    {
      /* Compute common solution Jacobian (dUcdU) */
      for (unsigned int var = 0; var < nVars; var++)
      {
        dUcdU(0, var, var, fpt) = (0.5 - beta);
        dUcdU(1, var, var, fpt) = (0.5 + beta);
      }
    }

    /* If boundary, use right state only */
    else
    {
      /* Compute common solution Jacobian (dUcdU) */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          dUcdU(0, vari, varj, fpt) = dURdUL[vari][varj];

      /* Get boundary Jacobian of gradient */
      for (unsigned int dimi = 0; dimi < nDims; dimi++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int varj = 0; varj < nVars; varj++)
              ddURddUL[dimi][dimj][vari][varj] = ddUbddU(dimi, dimj, vari, varj, fpt);
    }

    /* Compute viscous contribution to common dFdU */
    LDG_dFdU<nVars, nDims, equation>(UL, UR, dUL, dUR, dURdUL, ddURddUL, dFcdUL, dFcdUR, 
        dFcddUL, dFcddUR, norm, AdvDiff_D, gamma, prandtl, mu, LDG_bias_, beta, tau);

    /* Write common dFcddU to global memory */
    for (unsigned int dimj = 0; dimj < nDims; dimj++)
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          dFcddU(0, 0, dimj, vari, varj, fpt) =  dFcddUL[dimj][vari][varj] * dAL;

          // HACK: Temporarily use dA(0, fpt) since dA(1, fpt) doesn't exist on mpi faces
          // Note: (May not work for triangles/tets) consider removing dA dependence on slots
          dFcddU(0, 1, dimj, vari, varj, fpt) =  dFcddUR[dimj][vari][varj] * dAL;
          //dFcddU(0, 1, dimj, vari, varj, fpt) =  dFcddUR[dimj][vari][varj] * dA(1, fpt);

          dFcddU(1, 0, dimj, vari, varj, fpt) = -dFcddUR[dimj][vari][varj] * dAR;
          dFcddU(1, 1, dimj, vari, varj, fpt) = -dFcddUL[dimj][vari][varj] * dAL;
        }
  }

  /* Write common dFcdU to global memory */
  for (unsigned int vari = 0; vari < nVars; vari++)
    for (unsigned int varj = 0; varj < nVars; varj++)
    {
      dFcdU(0, vari, varj, fpt) =  dFcdUL[vari][varj] * dAL;
      dFcdU(1, vari, varj, fpt) = -dFcdUR[vari][varj] * dAR;
    }
}

void compute_common_dFdU_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &dU, mdview_gpu<double> &dFcdU, 
    mdview_gpu<double> &dUcdU, mdview_gpu<double> &dFcddU, mdvector_gpu<double> &dUbdU, mdvector_gpu<double> &ddUbddU, 
    mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias, mdvector_gpu<double> &dA, double AdvDiff_D, double gamma, 
    double rus_k, double mu, double prandtl, double beta, double tau, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int startFpt, unsigned int endFpt, bool viscous, mdvector_gpu<char> &flip_beta)
{
  unsigned int threads = 128;
  unsigned int blocks = ((endFpt - startFpt + 1) + threads - 1)/threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_common_dFdU<1, 2, AdvDiff><<<blocks, threads>>>(U, dU, dFcdU, dUcdU, dFcddU, dUbdU, ddUbddU, P, AdvDiff_A, norm, 
          waveSp, rus_bias, LDG_bias, dA, AdvDiff_D, gamma, rus_k, mu, prandtl, beta, tau, startFpt, endFpt, viscous, flip_beta);
    else
      compute_common_dFdU<1, 3, AdvDiff><<<blocks, threads>>>(U, dU, dFcdU, dUcdU, dFcddU, dUbdU, ddUbddU, P, AdvDiff_A, norm, 
          waveSp, rus_bias, LDG_bias, dA, AdvDiff_D, gamma, rus_k, mu, prandtl, beta, tau, startFpt, endFpt, viscous, flip_beta);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_dFdU<4, 2, EulerNS><<<blocks, threads>>>(U, dU, dFcdU, dUcdU, dFcddU, dUbdU, ddUbddU, P, AdvDiff_A, norm, 
          waveSp, rus_bias, LDG_bias, dA, AdvDiff_D, gamma, rus_k, mu, prandtl, beta, tau, startFpt, endFpt, viscous, flip_beta);
    else
      compute_common_dFdU<5, 3, EulerNS><<<blocks, threads>>>(U, dU, dFcdU, dUcdU, dFcddU, dUbdU, ddUbddU, P, AdvDiff_A, norm, 
          waveSp, rus_bias, LDG_bias, dA, AdvDiff_D, gamma, rus_k, mu, prandtl, beta, tau, startFpt, endFpt, viscous, flip_beta);
  }
}
