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

#include <cmath>

#include "faces.hpp"
#include "flux.hpp"
#include "geometry.hpp"
#include "input.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "faces_kernels.h"
#include "solver_kernels.h"
#endif

Faces::Faces(GeoStruct *geo, InputStruct *input, _mpi_comm comm_in)
{
  this->input = input;
  this->geo = geo;
  nFpts = geo->nGfpts;
  myComm = comm_in;

}

void Faces::setup(unsigned int nDims, unsigned int nVars)
{
  this->nVars = nVars;
  this->nDims = nDims;

  /* Allocate memory for solution structures */
  U_bnd.assign({nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
  U_bnd_ldg.assign({nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
  Fcomm_bnd.assign({nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});

  /* If viscous, allocate arrays used for LDG flux */
  if(input->viscous)
  {
    dU_bnd.assign({nDims, nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
    Ucomm_bnd.assign({nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
  }

  /* Loop over boundary flux points and initialize Riemann/LDG bias variables */
  rus_bias.assign({nFpts}, 0);
  LDG_bias.assign({nFpts}, 0);
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    if (bnd_id != PERIODIC || bnd_id != OVERSET)
    {
      /* Default: Rusanov BC Ghost, LDG BC Prescribed */
      LDG_bias(fpt) = 1;

      /* Implicit Default: Rusanov BC Prescribed, LDG BC Prescribed */
      if (input->implicit_method)
      {
        rus_bias(fpt) = 1;
      }
    }
  }

  /* Allocate memory for implicit method data structures */
  if (input->implicit_method)
  {
    dFcdU_bnd.assign({nVars, nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
    dUbdU.assign({nVars, nVars, nFpts}, 0);

    /* If viscous, allocate arrays used for LDG flux */
    if(input->viscous)
    {
      dUcdU_bnd.assign({nVars, nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
      dFcddU_bnd.assign({2, nDims, nVars, nVars, geo->nGfpts_bnd + geo->nGfpts_mpi});
      ddUbddU.assign({nDims, nDims, nVars, nVars, nFpts}, 0);
    }
  }

  /* If running Euler/NS, allocate memory for pressure */
  if (input->equation == EulerNS)
    P.assign({2, nFpts});

  waveSp.assign({nFpts}, 0.0);
  if (input->viscous)
    diffCo.assign({nFpts}, 0.0);

  /* Allocate memory for geometry structures */
  coord.assign({nDims, nFpts});
  norm.assign({nDims, nFpts});
  dA.assign({2, nFpts},0.0);
  //jaco.assign({nFpts, nDims, nDims , 2}); // TODO - remove

  /* Moving-grid-related structures */
  if (input->motion)
  {
    Vg.assign({nDims, nFpts}, 0.0);
  }

#ifdef _MPI
  /* Allocate memory for send/receive buffers */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int pairedRank = entry.first;
    const auto &fpts = entry.second;

    if (input->viscous)
    {
      U_sbuffs[pairedRank].assign({nDims, nVars, (unsigned int) fpts.size()}, 0.0, true);
      U_rbuffs[pairedRank].assign({nDims, nVars, (unsigned int) fpts.size()}, 0.0, true);
    }
    else
    {
      U_sbuffs[pairedRank].assign({nVars, (unsigned int) fpts.size()}, 0.0, true);
      U_rbuffs[pairedRank].assign({nVars, (unsigned int) fpts.size()}, 0.0, true);
    }
  }

  sreqs.resize(geo->fpt_buffer_map.size());
  rreqs.resize(geo->fpt_buffer_map.size());
#endif

}

void Faces::apply_bcs()
{
#ifdef _CPU
  /* Create some useful variables outside loop */
  std::array<double, 3> VL, VR, VG;

  /* Loop over boundary flux points */
#pragma omp parallel for
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    switch(bnd_id)
    {
      case SUP_IN: /* Farfield and Supersonic Inlet */
      {
        if (input->equation == AdvDiff)
        {
          /* Set boundaries to zero */
          U(1, 0, fpt) = 0;
          U_ldg(1, 0, fpt) = 0;
        }
        else
        {
          /* Set boundaries to freestream values */
          U(1, 0, fpt) = input->rho_fs;
          U_ldg(1, 0, fpt) = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            U(1, dim+1, fpt) = input->rho_fs * input->V_fs(dim);
            U_ldg(1, dim+1, fpt) = U(1, dim+1, fpt);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          U(1, nDims + 1, fpt) = input->P_fs/(input->gamma-1.0) + 0.5*input->rho_fs * Vsq; 
          U_ldg(1, nDims + 1, fpt) = U(1, nDims + 1, fpt);
        }

        break;
      }

      case SUP_OUT: /* Supersonic Outlet */
      {
        /* Extrapolate boundary values from interior */
        for (unsigned int n = 0; n < nVars; n++)
        {
          U(1, n, fpt) = U(0, n, fpt);
          U_ldg(1, n, fpt) = U(1, n, fpt);
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
          VnR += input->V_fs(dim) * norm(dim, fpt);
        }

        /* Compute pressure. TODO: Compute pressure once!*/
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(0, dim + 1, fpt) * U(0, dim + 1, fpt);
        }

        momF /= U(0, 0, fpt);

        double PL = (input->gamma - 1.0) * (U(0, nDims + 1, fpt) - 0.5 * momF);
        double PR = input->P_fs;

        double cL = std::sqrt(input->gamma * PL / U(0, 0, fpt));
        double cR = std::sqrt(input->gamma * PR / input->rho_fs);

        /* Compute Riemann Invariants */
        double RL;
        if (std::abs(VnR) >= cR && VnL >= 0)
          RL = VnR + 2.0 / (input->gamma - 1) * cR;
        else
          RL = VnL + 2.0 / (input->gamma - 1) * cL;

        double RB;
        if (std::abs(VnR) >= cR && VnL < 0)
          RB = VnL - 2.0 / (input->gamma - 1) * cL;
        else
          RB = VnR - 2.0 / (input->gamma - 1) * cR;

        double cstar = 0.25 * (input->gamma - 1) * (RL - RB);
        double ustarn = 0.5 * (RL + RB);

        double rhoR = cstar * cstar / input->gamma;
        double VR[3] = {0, 0, 0};

        if (VnL < 0.0) /* Case 1: Inflow */
        {
          rhoR *= std::pow(input->rho_fs, input->gamma) / PR;

          for (unsigned int dim = 0; dim < nDims; dim++)
            VR[dim] = input->V_fs(dim) + (ustarn - VnR) * norm(dim, fpt);
        }
        else  /* Case 2: Outflow */
        {
          rhoR *= std::pow(U(0, 0, fpt), input->gamma) / PL;

          for (unsigned int dim = 0; dim < nDims; dim++)
            VR[dim] = U(0, dim+1, fpt) / U(0, 0, fpt) + (ustarn - VnL) * norm(dim, fpt);
        }

        rhoR = std::pow(rhoR, 1.0 / (input->gamma - 1));

        U(1, 0, fpt) = rhoR;
        U_ldg(1, 0, fpt) = rhoR;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(1, dim + 1, fpt) = rhoR * VR[dim];
          U_ldg(1, dim + 1, fpt) = rhoR * VR[dim];
        }

        PR = rhoR / input->gamma * cstar * cstar;
        U(1, nDims + 1, fpt) = PR / (input->gamma - 1);
        U_ldg(1, nDims + 1, fpt) = PR / (input->gamma - 1);
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
        if (input->viscous)
          ThrowException("SLIP_WALL_P not supported for viscous!");

        /* Rusanov Prescribed */
        if (rus_bias(fpt) == 1)
        {
          double momN = 0.0;

          /* Compute wall normal momentum */
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN += U(0, dim+1, fpt) * norm(dim, fpt);

          if (input->motion)
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

          double PL = (input->gamma - 1.0) * (U(0, nDims + 1 , fpt) - 0.5 * momFL / U(0, 0, fpt));

          /* Get right-state momentum flux after velocity correction */
          double momFR = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
            momFR += U(1, dim + 1, fpt) * U(1, dim + 1, fpt);

          /* Compute energy with extrapolated pressure and new momentum */
          U(1, nDims + 1, fpt) = PL / (input->gamma - 1)  + 0.5 * momFR / U(1, 0, fpt);
        }

        /* Rusanov Ghost */
        else
        {
          double momN = 0.0;

          /* Compute wall normal momentum */
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN += U(0, dim+1, fpt) * norm(dim, fpt);

          if (input->motion)
          {
            for (unsigned int dim = 0; dim < nDims; dim++)
              momN -= U(0, 0, fpt) * Vg(dim, fpt) * norm(dim, fpt);
          }

          U(1, 0, fpt) = U(0, 0, fpt);

          /* Set boundary state to reflect normal velocity */
          for (unsigned int dim = 0; dim < nDims; dim++)
            U(1, dim+1, fpt) = U(0, dim+1, fpt) - 2.0 * momN * norm(dim, fpt);

          /* Set energy */
          U(1, nDims + 1, fpt) = U(0, nDims + 1, fpt);
        }

        break;
      }

      case ISOTHERMAL_NOSLIP: /* Isothermal No-slip Wall */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        if (input->motion)
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
          
        double cp_over_gam =  input->R_ref / (input->gamma - 1);

        U(1, nDims + 1, fpt) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq) ;
        U_ldg(1, nDims + 1, fpt) = rhoL * cp_over_gam * input->T_wall;

        /* Rusanov Prescribed */
        if (rus_bias(fpt) == 1)
          for (unsigned int var = 0; var < nVars; var++)
            U(1, var, fpt) = U_ldg(1, var, fpt);

        break;
      }

      case ISOTHERMAL_NOSLIP_MOVING: /* Isothermal No-slip Wall, moving */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double rhoL = U(0, 0, fpt);

        U(1, 0, fpt) = rhoL;
        U_ldg(1, 0, fpt) = rhoL;

        /* Set velocity to zero (or wall velocity) */
        double Vsq = 0; double Vsq_wall = 0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(0, dim+1, fpt) / U(0, 0, fpt);
          double V = -VL + 2*(input->V_wall(dim));
          U(1, dim+1, fpt) = rhoL * V;
          Vsq += V * V;

          U_ldg(1, dim+1, fpt) = rhoL*input->V_wall(dim);
          Vsq_wall += input->V_wall(dim) * input->V_wall(dim);
        }
          
        double cp_over_gam =  input->R_ref / (input->gamma - 1);

        U(1, nDims + 1, fpt) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq);
        U_ldg(1, nDims + 1, fpt) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq_wall);

        /* Rusanov Prescribed */
        if (rus_bias(fpt) == 1)
          for (unsigned int var = 0; var < nVars; var++)
            U(1, var, fpt) = U_ldg(1, var, fpt);

        break;
      }

      case ADIABATIC_NOSLIP: /* Adiabatic No-slip Wall */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        if (input->motion)
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

      case ADIABATIC_NOSLIP_MOVING: /* Adiabatic No-slip Wall, moving */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        /* Extrapolate density */
        double rhoL = U(0, 0, fpt);
        U(1, 0, fpt) = rhoL;
        U_ldg(1, 0, fpt) = rhoL;

        /* Set right state (common) velocity to zero (or wall velocity) */
        double Vsq = 0.0; double VLsq = 0.0; double Vsq_wall = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(0, dim+1, fpt) / rhoL; 
          double V = -VL + 2 * input->V_wall(dim);
          U(1, dim+1, fpt) = rhoL * V;
          U_ldg(1, dim+1, fpt) = rhoL * input->V_wall(dim);

          Vsq += V * V;
          VLsq += VL * VL;
          Vsq_wall += input->V_wall(dim) * input->V_wall(dim);
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
        // Do nothing [works similarly to MPI]
      }
    } 
  }
#endif

#ifdef _GPU
  apply_bcs_wrapper(U_d, U_ldg_d, nFpts, geo->nGfpts_int, geo->nGfpts_bnd, nVars, nDims, input->rho_fs, input->V_fs_d, 
      input->P_fs, input->gamma, input->R, input->T_tot_fs, input->P_tot_fs, input->T_wall, input->V_wall_d,
      Vg_d, input->norm_fs_d, norm_d, geo->gfpt2bnd_d, rus_bias_d, LDG_bias_d, input->equation, input->motion);

  check_error();
#endif

}

void Faces::apply_bcs_dU()
{
#ifdef _CPU
  /* Apply boundaries to solution derivative */
#pragma omp parallel for
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    if(bnd_id == ADIABATIC_NOSLIP || bnd_id == ADIABATIC_NOSLIP_MOVING) /* Adibatic Wall */
    {
      /* Extrapolate density gradient */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dU(1, dim, 0, fpt) = dU(0, dim, 0, fpt);
      }

      if (nDims == 2)
      {
        /* Compute energy gradient */
        /* Get left states and velocity gradients*/
        double rho = U(0, 0, fpt);
        double momx = U(0, 1, fpt);
        double momy = U(0, 2, fpt);
        double E = U(0, 3, fpt);

        double u = momx / rho;
        double v = momy / rho;


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
        //double du_dn = du_dx * norm(fpt, 0) + du_dy * norm(fpt, 1);
        //double dv_dn = dv_dx * norm(fpt, 0) + dv_dy * norm(fpt, 1);

        //dU(fpt, 1, 0, 1) = rho * du_dn * norm(fpt, 0);
        //dU(fpt, 1, 1, 1) = rho * du_dn * norm(fpt, 1);
        //dU(fpt, 2, 0, 1) = rho * dv_dn * norm(fpt, 0);
        //dU(fpt, 2, 1, 1) =  rho * dv_dn * norm(fpt, 1);

        // double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
        // double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

        /* Compute temperature gradient (actually C_v * rho * dT) */
        double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx);
        double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy);

        /* Compute wall normal temperature gradient */
        double dT_dn = dT_dx * norm(0, fpt) + dT_dy * norm(1, fpt);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(1, 0, 3, fpt) = E_dx - dT_dn * norm(0, fpt);
        dU(1, 1, 3, fpt) = E_dy - dT_dn * norm(1, fpt);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 3, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0)) + rho_dx * U(fpt, 3, 1) / rho; 
        //dU(fpt, 3, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1)) + rho_dy * U(fpt, 3, 1) / rho; 
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
        //double du_dn = du_dx * norm(fpt, 0) + du_dy * norm(fpt, 1) + du_dz * norm(fpt, 2);
        //double dv_dn = dv_dx * norm(fpt, 0) + dv_dy * norm(fpt, 1) + dv_dz * norm(fpt, 2);
        //double dw_dn = dw_dx * norm(fpt, 0) + dw_dy * norm(fpt, 1) + dw_dz * norm(fpt, 2);

        //dU(fpt, 1, 0, 1) = rho * du_dn * norm(fpt, 0);
        //dU(fpt, 1, 1, 1) = rho * du_dn * norm(fpt, 1);
        //dU(fpt, 1, 2, 1) = rho * du_dn * norm(fpt, 2);
        //dU(fpt, 2, 0, 1) = rho * dv_dn * norm(fpt, 0);
        //dU(fpt, 2, 1, 1) = rho * dv_dn * norm(fpt, 1);
        //dU(fpt, 2, 2, 1) = rho * dv_dn * norm(fpt, 2);
        //dU(fpt, 3, 0, 1) = rho * dw_dn * norm(fpt, 0);
        //dU(fpt, 3, 1, 1) = rho * dw_dn * norm(fpt, 1);
        //dU(fpt, 3, 2, 1) = rho * dw_dn * norm(fpt, 2);

       // double dke_dx = 0.5 * (u*u + v*v + w*w) * rho_dx + rho * (u * du_dx + v * dv_dx + w * dw_dx);
       // double dke_dy = 0.5 * (u*u + v*v + w*w) * rho_dy + rho * (u * du_dy + v * dv_dy + w * dw_dy);
       // double dke_dz = 0.5 * (u*u + v*v + w*w) * rho_dz + rho * (u * du_dz + v * dv_dz + w * dw_dz);

        /* Compute temperature gradient (actually C_v * rho * dT) */
        double dT_dx = E_dx - rho_dx * E/rho - rho * (u * du_dx + v * dv_dx + w * dw_dx);
        double dT_dy = E_dy - rho_dy * E/rho - rho * (u * du_dy + v * dv_dy + w * dw_dy);
        double dT_dz = E_dz - rho_dz * E/rho - rho * (u * du_dz + v * dv_dz + w * dw_dz);

        /* Compute wall normal temperature gradient */
        double dT_dn = dT_dx * norm(0, fpt) + dT_dy * norm(1, fpt) + dT_dz * norm(2, fpt);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(1, 0, 4, fpt) = E_dx - dT_dn * norm(0, fpt);
        dU(1, 1, 4, fpt) = E_dy - dT_dn * norm(1, fpt);
        dU(1, 2, 4, fpt) = E_dz - dT_dn * norm(2, fpt);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 4, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0)) + rho_dx * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1)) + rho_dy * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 2, 1) = (dT_dz - dT_dn * norm(fpt, 2)) + rho_dz * U(fpt, 4, 1) / rho; 
      }

    }
    else if (bnd_id == OVERSET)
    {
      // Do nothing...? [need to treat same as internal]
    }
    else /* Otherwise, right state gradient equals left state gradient */
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
#endif

#ifdef _GPU
  apply_bcs_dU_wrapper(dU_d, U_d, norm_d, nFpts, geo->nGfpts_int, geo->nGfpts_bnd, nVars, 
      nDims, geo->gfpt2bnd_d, input->equation);

  check_error();
#endif
}

// TODO: Collapse 2D and 3D boundary condition cases
void Faces::apply_bcs_dFdU()
{
#ifdef _CPU
  /* Loop over boundary flux points */
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

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
        if (input->viscous)
          for (unsigned int dim = 0; dim < nDims; dim++)
            for (unsigned int var = 0; var < nVars; var++)
              ddUbddU(dim, dim, var, var, fpt) = 1;

        break;
      }

      case SUP_OUT: /* Supersonic Outlet */
      {
        ThrowException("Supersonic Outlet boundary condition not implemented for implicit method");
        break;
      }

      case CHAR: /* Characteristic (from PyFR) */
      {
        /* Compute wall normal velocities */
        double VnL = 0.0; double VnR = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += U(0, dim+1, fpt) / U(0, 0, fpt) * norm(dim, fpt);
          VnR += input->V_fs(dim) * norm(dim, fpt);
        }

        /* Compute pressure. TODO: Compute pressure once!*/
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          momF += U(0, dim + 1, fpt) * U(0, dim + 1, fpt);
        }

        momF /= U(0, 0, fpt);

        double PL = (input->gamma - 1.0) * (U(0, nDims + 1, fpt) - 0.5 * momF);
        double PR = input->P_fs;

        double cL = std::sqrt(input->gamma * PL / U(0, 0, fpt));
        double cR = std::sqrt(input->gamma * PR / input->rho_fs);

        /* Compute Riemann Invariants */
        double RL;
        if (std::abs(VnR) >= cR && VnL >= 0)
          ThrowException("Implicit Char BC not implemented for supersonic flow!")
        else
          RL = VnL + 2.0 / (input->gamma - 1) * cL;

        double RB;
        if (std::abs(VnR) >= cR && VnL < 0)
          ThrowException("Implicit Char BC not implemented for supersonic flow!")
        else
          RB = VnR - 2.0 / (input->gamma - 1) * cR;

        double cstar = 0.25 * (input->gamma - 1) * (RL - RB);
        double ustarn = 0.5 * (RL + RB);

        if (nDims == 2)
        {
          double nx = norm(0, fpt);
          double ny = norm(1, fpt);
          double gam = input->gamma;

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
          double gam = input->gamma;

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
        if (input->viscous)
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
        dUbdU(nDims+1, 0, fpt) = (input->R * input->T_wall) / (input->gamma-1.0);

        /* Extrapolate gradients */
        if (input->viscous)
          for (unsigned int dim = 0; dim < nDims; dim++)
            for (unsigned int var = 0; var < nVars; var++)
              ddUbddU(dim, dim, var, var, fpt) = 1;

        break;
      }

      case ISOTHERMAL_NOSLIP_MOVING: /* Isothermal No-slip Wall, moving */
      {
        double Vsq = 0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          Vsq += input->V_wall(dim) * input->V_wall(dim);

        dUbdU(0, 0, fpt) = 1;
        for (unsigned int dim = 0; dim < nDims; dim++)
          dUbdU(dim+1, 0, fpt) = input->V_wall(dim);
        dUbdU(nDims+1, 0, fpt) = (input->R * input->T_wall) / (input->gamma-1.0) + 0.5 * Vsq;

        /* Extrapolate gradients */
        if (input->viscous)
          for (unsigned int dim = 0; dim < nDims; dim++)
            for (unsigned int var = 0; var < nVars; var++)
              ddUbddU(dim, dim, var, var, fpt) = 1;

        break;
      }

      case ADIABATIC_NOSLIP: /* Adiabatic No-slip Wall */
      {
        if (nDims == 3)
          ThrowException("3D Adiabatic No-slip Wall (prescribed) boundary condition not implemented for implicit method");

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

        if (input->viscous)
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
        ThrowException("Adiabatic No-slip Wall, moving boundary condition not implemented for implicit method");
        break;
      }

      default:
      {
        ThrowException("Boundary condition not implemented for implicit method");
        break;
      }
    }
  }
#endif

#ifdef _GPU
  apply_bcs_dFdU_wrapper(U_d, dUbdU_d, ddUbddU_d, nFpts, geo->nGfpts_int, geo->nGfpts_bnd, nVars, nDims, input->viscous, 
      input->rho_fs, input->V_fs_d, input->P_fs, input->gamma, input->R, input->T_wall, input->V_wall_d, norm_d, 
      geo->gfpt2bnd_d, input->equation);

  check_error();
#endif
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Faces::rusanov_flux(unsigned int startFpt, unsigned int endFpt)
{
  double FL[nVars][nDims];
  double FR[nVars][nDims];
  double UL[nVars];
  double UR[nVars];
  double V[nDims] = {0.0}; // Grid velocity - only updated if moving grid

#pragma omp parallel for private(FL,FR,UL,UR) firstprivate(V)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UL[n] = U(0, n, fpt); UR[n] = U(1, n, fpt);
    }

    double eig = 0;
    double Vgn = 0;
    if (input->motion)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = Vg(dim, fpt);
        Vgn += V[dim] * norm(dim, fpt);
      }
    }

    /* Get numerical wavespeed */
    if (input->equation == AdvDiff)
    {
      double An = 0.;
      double A[nDims];

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        An += input->AdvDiff_A(dim) * norm(dim, fpt);
        A[dim] = input->AdvDiff_A(dim);
      }

      eig = std::abs(An);
      waveSp(fpt) = std::abs(An - Vgn);

      compute_Fconv_AdvDiff<nVars, nDims>(UL, FL, A, V);
      compute_Fconv_AdvDiff<nVars, nDims>(UR, FR, A, V);
    }
    else if (input->equation == EulerNS)
    {
      double PL, PR;

      compute_Fconv_EulerNS<nVars, nDims>(UL, FL, V, PL, input->gamma);
      compute_Fconv_EulerNS<nVars, nDims>(UR, FR, V, PR, input->gamma);

      /* Store pressures for force computation */
      P(0, fpt) = PL;
      P(1, fpt) = PR;

      /* Compute speed of sound */
      double aL = std::sqrt(input->gamma * PL / UL[0]);
      double aR = std::sqrt(input->gamma * PR / UR[0]);

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += UL[dim+1]/UL[0] * norm(dim, fpt);
        VnR += UR[dim+1]/UR[0] * norm(dim, fpt);
      }

      eig = std::max(std::abs(VnL) + aL, std::abs(VnR) + aR);
      waveSp(fpt) = std::max(std::abs(VnL-Vgn) + aL, std::abs(VnR-Vgn) + aR);
    }

    double FnL[nVars] = {0.0};
    double FnR[nVars] = {0.0};

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        FnL[n] += FL[n][dim] * norm(dim, fpt);
        FnR[n] += FR[n][dim] * norm(dim, fpt);
      }
    }

    /* Compute common normal flux */
    if (rus_bias(fpt) == 0) /* Upwinded */
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double F = (0.5 * (FnR[n]+FnL[n]) - 0.5 * eig * (1.0-input->rus_k) * (UR[n]-UL[n]));

        /* Correct for positive parent space sign convention */
        Fcomm(0, n, fpt) = F * dA(0, fpt);
        Fcomm(1, n, fpt) = -F * dA(1, fpt);
      }
    }
    else if (rus_bias(fpt) == 2) /* Central */
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double F = 0.5 * (FnL[n] + FnR[n]);
        Fcomm(0, n, fpt) = F * dA(0, fpt);
        Fcomm(1, n, fpt) = -F * dA(1, fpt);
      }
    }
    else if (rus_bias(fpt) == 1)
    {
      for (unsigned int n = 0; n < nVars; n++) /* Set flux state */
      {
        double F = FnR[n];
        Fcomm(0, n, fpt) = F * dA(0, fpt);
        Fcomm(1, n, fpt) = -F * dA(1, fpt);
      }
    }

  }
}

template<unsigned int nVars>
void Faces::compute_common_U(unsigned int startFpt, unsigned int endFpt)
{
  
  /* Compute common solution */
  if (input->fvisc_type == LDG)
  {
#ifdef _CPU
#pragma omp parallel for
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

      double beta = geo->flip_beta(fpt)*input->ldg_b;

      /* Get left and right state variables */
      /* If interior, allow use of beta factor */
      if (LDG_bias(fpt) == 0)
      {
        double UL[nVars] = {};
        double UR[nVars] = {};

        if (beta == 0.5)
        {
          for (unsigned int n = 0; n < nVars; n++)
            UR[n] = U_ldg(1, n, fpt);
        }
        else if (beta == -0.5)
        {
          for (unsigned int n = 0; n < nVars; n++)
            UL[n] = U_ldg(0, n, fpt);
        }
        else
        {
          for (unsigned int n = 0; n < nVars; n++)
          {
            UL[n] = U_ldg(0, n, fpt); UR[n] = U_ldg(1, n, fpt);
          }
        }

        for (unsigned int n = 0; n < nVars; n++)
        {
          double UC = 0.5*(UL[n] + UR[n]) - beta*(UL[n] - UR[n]);
          Ucomm(0, n, fpt) = UC;
          Ucomm(1, n, fpt) = UC;
        }
      }
      /* If on (non-periodic) boundary, set right state as common (strong) */
      else
      {
        double UR[nVars] = {};
        for (unsigned int n = 0; n < nVars; n++)
          UR[n] = U_ldg(1, n, fpt);

        for (unsigned int n = 0; n < nVars; n++)
        {
          Ucomm(0, n, fpt) = UR[n];
          Ucomm(1, n, fpt) = UR[n];
        }
      }
    }
#endif

#ifdef _GPU
    compute_common_U_LDG_wrapper(U_ldg_d, Ucomm_d, norm_d, input->ldg_b, nFpts, nVars, nDims, input->equation, 
        LDG_bias_d, startFpt, endFpt, geo->flip_beta_d, input->overset, geo->iblank_fpts_d.data());

    check_error();

#endif
  }

  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }
}

void Faces::compute_common_U(unsigned int startFpt, unsigned int endFpt)
{
  if (input->equation == AdvDiff)
    compute_common_U<1>(startFpt, endFpt);
  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
      compute_common_U<4>(startFpt, endFpt);
    else
      compute_common_U<5>(startFpt, endFpt);
  }

}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Faces::LDG_flux(unsigned int startFpt, unsigned int endFpt)
{
   
  double tau = input->ldg_tau;

  double UL[nVars];
  double UR[nVars];
  double Fc[nVars];

#pragma omp parallel for private(UL,UR,Fc) firstprivate(tau)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE)
        continue;

    bool iflag = false;
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == FRINGE)
      iflag = true;

    double beta = geo->flip_beta(fpt)*input->ldg_b;

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UL[n] = U_ldg(0, n, fpt); UR[n] = iflag ? U(1, n, fpt) : U_ldg(1, n, fpt);
    }

    double dUL[nVars][nDims] = {{}};
    double dUR[nVars][nDims] = {{}};

    if (LDG_bias(fpt) == 0)
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
    }

    double FL[nVars][nDims] = {{0.0}};
    double FR[nVars][nDims] = {{0.0}};

    /* Get numerical diffusion coefficient */
    if (input->equation == AdvDiff)
    {
      compute_Fvisc_AdvDiff_add<nVars, nDims>(dUL, FL, input->AdvDiff_D);
      compute_Fvisc_AdvDiff_add<nVars, nDims>(dUR, FR, input->AdvDiff_D);

      diffCo(fpt) = input->AdvDiff_D;
    }
    else if (input->equation == EulerNS)
    {
      compute_Fvisc_EulerNS_add<nVars, nDims>(UL, dUL, FL, input->gamma, input->prandtl, input->mu, 
          input->rt, input->c_sth, input->fix_vis);
      compute_Fvisc_EulerNS_add<nVars, nDims>(UR, dUR, FR, input->gamma, input->prandtl, input->mu, 
          input->rt, input->c_sth, input->fix_vis);

      // TODO: Add or store mu from Sutherland's law
      double diffCoL = std::max(input->mu / UL[0], input->gamma * input->mu / (input->prandtl * UL[0]));
      double diffCoR = std::max(input->mu / UR[0], input->gamma * input->mu / (input->prandtl * UR[0]));
      diffCo(fpt) = std::max(diffCoL, diffCoR);
    }

    double FnL[nVars] = {0.0};
    double FnR[nVars] = {0.0};

    /* Get interface-normal flux components  (from L to R)*/
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        FnL[n] += FL[n][dim] * norm(dim, fpt);
        FnR[n] += FR[n][dim] * norm(dim, fpt);
      }
    }


    /* Compute common normal viscous flux and accumulate */
    /* If interior, use central */
    if (LDG_bias(fpt) == 0)
    {
      for (unsigned int n = 0; n < nVars; n++)
        Fc[n] = 0.5 * (FnL[n] + FnR[n]) + tau * (UL[n] - UR[n]) + beta * (FnL[n] - FnR[n]);
    }
    /* If Neumann boundary, use right state only */
    else
    {
      for (unsigned int n = 0; n < nVars; n++)
        Fc[n] = FnR[n] + tau * (UL[n] - UR[n]);
    }

    for (unsigned int n = 0; n < nVars; n++)
    {
      Fcomm(0, n, fpt) += Fc[n] * dA(0, fpt);
      Fcomm(1, n, fpt) -= Fc[n] * dA(1, fpt);
    }
  }
}

void Faces::compute_common_F(unsigned int startFpt, unsigned int endFpt)
{
#ifdef _CPU
  if (input->fconv_type == Rusanov)
  {
    if (input->equation == AdvDiff)
    {
      if (nDims == 2)
        rusanov_flux<1, 2, AdvDiff>(startFpt, endFpt);
      else
        rusanov_flux<1, 3, AdvDiff>(startFpt, endFpt);
    }
    else if (input->equation == EulerNS)
    {
      if (nDims == 2)
        rusanov_flux<4, 2, EulerNS>(startFpt, endFpt);
      else
        rusanov_flux<5, 3, EulerNS>(startFpt, endFpt);
    }

  }
  else
  {
    ThrowException("Numerical convective flux type not recognized!");
  }

  if (input->viscous)
  {
    if (input->fvisc_type == LDG)
    {
      if (input->equation == AdvDiff)
      {
        if (nDims == 2)
          LDG_flux<1, 2, AdvDiff>(startFpt, endFpt);
        else
          LDG_flux<1, 3, AdvDiff>(startFpt, endFpt);
      }
      else if (input->equation == EulerNS)
      {
        if (nDims == 2)
          LDG_flux<4, 2, EulerNS>(startFpt, endFpt);
        else
          LDG_flux<5, 3, EulerNS>(startFpt, endFpt);
      }
    }
    else
    {
      ThrowException("Numerical viscous flux type not recognized!");
    }
  }
#endif

#ifdef _GPU
    compute_common_F_wrapper(U_d, U_ldg_d, dU_d, Fcomm_d, P_d, input->AdvDiff_A_d, norm_d, waveSp_d, diffCo_d,
    rus_bias_d, LDG_bias_d,  dA_d, Vg_d, input->AdvDiff_D, input->gamma, input->rus_k, input->mu, input->prandtl, 
    input->rt, input->c_sth, input->fix_vis, input->ldg_b, input->ldg_tau, nFpts, geo->nGfpts_int, nVars, nDims, input->equation, 
    input->fconv_type, input->fvisc_type, startFpt, endFpt, input->viscous, geo->flip_beta_d, input->motion, input->overset, geo->iblank_fpts_d.data());

    check_error();
#endif
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Faces::rusanov_dFdU(unsigned int startFpt, unsigned int endFpt)
{
  double dFdUL[nVars][nVars][nDims] = {0};
  double dFdUR[nVars][nVars][nDims] = {0};
  double UL[nVars];
  double UR[nVars];
  double dURdUL[nVars][nVars];

  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    /* Get left and right state variables */
    for (unsigned int var = 0; var < nVars; var++)
    {
      UL[var] = U(0, var, fpt); UR[var] = U(1, var, fpt);
    }

    /* Compute flux Jacobians and numerical wavespeed derivative */
    double dwSdUL[nVars] = {0.0};
    double dwSdUR[nVars] = {0.0};
    if (input->equation == AdvDiff)
    {
      double A[nDims];
      for (unsigned int dim = 0; dim < nDims; dim++)
        A[dim] = input->AdvDiff_A(dim);

      compute_dFdUconv_AdvDiff<nVars, nDims>(dFdUL, A);
      compute_dFdUconv_AdvDiff<nVars, nDims>(dFdUR, A);
    }
    else if (input->equation == EulerNS)
    {
      compute_dFdUconv_EulerNS<nVars, nDims>(UL, dFdUL, input->gamma);
      compute_dFdUconv_EulerNS<nVars, nDims>(UR, dFdUR, input->gamma);

      /* Get pressure */
      double PL = P(0, fpt);
      double PR = P(1, fpt);

      /* Compute speed of sound */
      double aL = std::sqrt(input->gamma * PL / UL[0]);
      double aR = std::sqrt(input->gamma * PR / UR[0]);

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += UL[dim+1]/UL[0] * norm(dim, fpt);
        VnR += UR[dim+1]/UR[0] * norm(dim, fpt);
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
          dwSdUL[0] += input->gamma * (input->gamma-1.0) * V[dim]*V[dim] / (4.0*aL*rho);
          dwSdUL[dim+1] = sgn*norm(dim, fpt)/rho - input->gamma * (input->gamma-1.0) * V[dim] / (2.0*aL*rho);
        }
        dwSdUL[nDims+1] = input->gamma * (input->gamma-1.0) / (2.0*aL*rho);
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
          dwSdUR[0] += input->gamma * (input->gamma-1.0) * V[dim]*V[dim] / (4.0*aR*rho);
          dwSdUR[dim+1] = sgn*norm(dim, fpt)/rho - input->gamma * (input->gamma-1.0) * V[dim] / (2.0*aR*rho);
        }
        dwSdUR[nDims+1] = input->gamma * (input->gamma-1.0) / (2.0*aR*rho);
      }
    }

    /* Get interface-normal dFdU components  (from L to R)*/
    double dFndUL[nVars][nVars] = {0.0};
    double dFndUR[nVars][nVars] = {0.0};
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          dFndUL[vari][varj] += dFdUL[vari][varj][dim] * norm(dim, fpt);
          dFndUR[vari][varj] += dFdUR[vari][varj][dim] * norm(dim, fpt);
        }

    /* Compute common dFdU */
    if (rus_bias(fpt) == 0) /* Upwinded */
    {
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          double dFcdUL, dFcdUR;
          if (vari == varj)
          {
            dFcdUL = 0.5 * dFndUL[vari][varj] - 0.5 * ((UR[vari]-UL[vari]) * dwSdUL[varj] - waveSp(fpt)) * (1.0-input->rus_k);
            dFcdUR = 0.5 * dFndUR[vari][varj] - 0.5 * ((UR[vari]-UL[vari]) * dwSdUR[varj] + waveSp(fpt)) * (1.0-input->rus_k);
          }
          else
          {
            dFcdUL = 0.5 * dFndUL[vari][varj] - 0.5 * (UR[vari]-UL[vari]) * dwSdUL[varj] * (1.0-input->rus_k);
            dFcdUR = 0.5 * dFndUR[vari][varj] - 0.5 * (UR[vari]-UL[vari]) * dwSdUR[varj] * (1.0-input->rus_k);
          }
          dFcdU(0, vari, varj, fpt) =  dFcdUL * dA(0, fpt);
          dFcdU(1, vari, varj, fpt) = -dFcdUR * dA(1, fpt);
        }
    }
    else if (rus_bias(fpt) == 2) /* Central */
    {
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          dFcdU(0, vari, varj, fpt) =  0.5 * dFndUL[vari][varj] * dA(0, fpt);
          dFcdU(1, vari, varj, fpt) = -0.5 * dFndUR[vari][varj] * dA(1, fpt);
        }
    }
    else if (rus_bias(fpt) == 1) /* Set flux state */
    {
      /* Get boundary Jacobian of solution */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          dURdUL[vari][varj] = dUbdU(vari, varj, fpt);

      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          double dFcdUR = 0;
          for (unsigned int vark = 0; vark < nVars; vark++)
            dFcdUR += dFndUR[vari][vark] * dURdUL[vark][varj];

          dFcdU(0, vari, varj, fpt) =  dFcdUR * dA(0, fpt);
          dFcdU(1, vari, varj, fpt) = -dFcdUR * dA(1, fpt);
        }
    }
  }
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Faces::LDG_dFdU(unsigned int startFpt, unsigned int endFpt)
{
  double tau = input->ldg_tau;

  double UL[nVars];
  double UR[nVars];
  double dUL[nVars][nDims];
  double dUR[nVars][nDims];
  double dURdUL[nVars][nVars];
  double ddURddUL[nDims][nDims][nVars][nVars];

  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face(geo->fpt2face[fpt]) == HOLE) continue;

    double beta = geo->flip_beta(fpt)*input->ldg_b;

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UL[n] = U_ldg(0, n, fpt); UR[n] = U_ldg(1, n, fpt);

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dUL[n][dim] = dU(0, dim, n, fpt);
        dUR[n][dim] = dU(1, dim, n, fpt);
      }
    }

    /* Compute viscous flux Jacobians */
    double dFdUL[nVars][nVars][nDims] = {0};
    double dFdUR[nVars][nVars][nDims] = {0};
    double dFddUL[nVars][nVars][nDims][nDims] = {0};
    double dFddUR[nVars][nVars][nDims][nDims] = {0};
    if (input->equation == AdvDiff)
    {
      compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddUL, input->AdvDiff_D);
      compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddUR, input->AdvDiff_D);
    }
    else if (input->equation == EulerNS)
    {
      compute_dFdUvisc_EulerNS_add<nVars, nDims>(UL, dUL, dFdUL, input->gamma, input->prandtl, input->mu);
      compute_dFdUvisc_EulerNS_add<nVars, nDims>(UR, dUR, dFdUR, input->gamma, input->prandtl, input->mu);

      compute_dFddUvisc_EulerNS<nVars, nDims>(UL, dFddUL, input->gamma, input->prandtl, input->mu);
      compute_dFddUvisc_EulerNS<nVars, nDims>(UR, dFddUR, input->gamma, input->prandtl, input->mu);
    }

    /* Get interface-normal dFdU components (from L to R) */
    double dFndUL[nVars][nVars] = {0};
    double dFndUR[nVars][nVars] = {0};
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          dFndUL[vari][varj] += dFdUL[vari][varj][dim] * norm(dim, fpt);
          dFndUR[vari][varj] += dFdUR[vari][varj][dim] * norm(dim, fpt);
        }

    /* Get interface-normal dFddU components (from L to R) */
    double dFnddUL[nVars][nVars][nDims] = {0};
    double dFnddUR[nVars][nVars][nDims] = {0};
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
        for (unsigned int dimi = 0; dimi < nDims; dimi++)
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
          {
            dFnddUL[vari][varj][dimj] += dFddUL[vari][varj][dimi][dimj] * norm(dimi, fpt);
            dFnddUR[vari][varj][dimj] += dFddUR[vari][varj][dimi][dimj] * norm(dimi, fpt);
          }

    /* Compute common normal viscous dFdU */
    /* If interior, use central */
    if (LDG_bias(fpt) == 0)
    {
      /* Compute common solution Jacobian (dUcdU) */
      for (unsigned int var = 0; var < nVars; var++)
      {
        dUcdU(0, var, var, fpt) = (0.5 - beta);
        dUcdU(1, var, var, fpt) = (0.5 + beta);
      }

      /* Compute common viscous flux Jacobian (dFcdU) */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          double dFcdUL = (0.5 + beta) * dFndUL[vari][varj];
          double dFcdUR = (0.5 - beta) * dFndUR[vari][varj];

          if (vari == varj)
          {
            dFcdUL += tau;
            dFcdUR -= tau;
          }

          dFcdU(0, vari, varj, fpt) += dFcdUL * dA(0, fpt);
          dFcdU(1, vari, varj, fpt) -= dFcdUR * dA(1, fpt);
        }

      /* Compute common viscous flux Jacobian (dFcddU) */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
          {
            double dFcddUL = (0.5 + beta) * dFnddUL[vari][varj][dimj];
            double dFcddUR = (0.5 - beta) * dFnddUR[vari][varj][dimj];

            dFcddU(0, 0, dimj, vari, varj, fpt) =  dFcddUL * dA(0, fpt);

            // HACK: Temporarily use dA(0, fpt) since dA(1, fpt) doesn't exist on mpi faces
            // Note: (May not work for triangles/tets) consider removing dA dependence on slots
            dFcddU(0, 1, dimj, vari, varj, fpt) =  dFcddUR * dA(0, fpt);
            //dFcddU(0, 1, dimj, vari, varj, fpt) =  dFcddUR * dA(1, fpt);

            dFcddU(1, 0, dimj, vari, varj, fpt) = -dFcddUR * dA(1, fpt);
            dFcddU(1, 1, dimj, vari, varj, fpt) = -dFcddUL * dA(0, fpt);
          }
    }

    /* If boundary, use right state only */
    else
    {
      /* Get boundary Jacobian of solution */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          dURdUL[vari][varj] = dUbdU(vari, varj, fpt);

      /* Compute common solution Jacobian (dUcdU) */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          dUcdU(0, vari, varj, fpt) = dURdUL[vari][varj];

      /* Compute common viscous flux Jacobian (dFcdU) */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          /* Compute boundary dFdU */
          double dFcdUR = 0;
          for (unsigned int vark = 0; vark < nVars; vark++)
            dFcdUR += dFndUR[vari][vark] * dURdUL[vark][varj];

          if (vari == varj)
            dFcdUR += tau;
          dFcdUR -= tau * dURdUL[vari][varj];

          dFcdU(0, vari, varj, fpt) += dFcdUR * dA(0, fpt);
        }

      /* Get boundary Jacobian of gradient */
      for (unsigned int dimi = 0; dimi < nDims; dimi++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int varj = 0; varj < nVars; varj++)
              ddURddUL[dimi][dimj][vari][varj] = ddUbddU(dimi, dimj, vari, varj, fpt);

      /* Compute boundary dFddU */
      double dFcddUR[nDims][nVars][nVars] = {0};
      for (unsigned int dimj = 0; dimj < nDims; dimj++)
        for (unsigned int dimk = 0; dimk < nDims; dimk++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int varj = 0; varj < nVars; varj++)
              for (unsigned int vark = 0; vark < nVars; vark++)
                dFcddUR[dimj][vari][varj] += dFnddUR[vari][vark][dimk] * ddURddUL[dimk][dimj][vark][varj];

      /* Compute common viscous flux Jacobian (dFcddU) */
      for (unsigned int dimj = 0; dimj < nDims; dimj++)
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            dFcddU(0, 0, dimj, vari, varj, fpt) =  dFcddUR[dimj][vari][varj] * dA(0, fpt);
    }
  }
}

void Faces::compute_common_dFdU(unsigned int startFpt, unsigned int endFpt)
{
#ifdef _CPU
  if (input->fconv_type == Rusanov)
  {
    if (input->equation == AdvDiff)
    {
      if (nDims == 2)
        rusanov_dFdU<1, 2, AdvDiff>(startFpt, endFpt);
      else
        rusanov_dFdU<1, 3, AdvDiff>(startFpt, endFpt);
    }
    else if (input->equation == EulerNS)
    {
      if (nDims == 2)
        rusanov_dFdU<4, 2, EulerNS>(startFpt, endFpt);
      else
        rusanov_dFdU<5, 3, EulerNS>(startFpt, endFpt);
    }
  }
  else
  {
    ThrowException("Numerical convective flux type not recognized!");
  }

  if (input->viscous)
  {
    if (input->fvisc_type == LDG)
    {
      if (input->equation == AdvDiff)
      {
        if (nDims == 2)
          LDG_dFdU<1, 2, AdvDiff>(startFpt, endFpt);
        else
          LDG_dFdU<1, 3, AdvDiff>(startFpt, endFpt);
      }
      else if (input->equation == EulerNS)
      {
        if (nDims == 2)
          LDG_dFdU<4, 2, EulerNS>(startFpt, endFpt);
        else
          LDG_dFdU<5, 3, EulerNS>(startFpt, endFpt);
      }
    }
    else
    {
      ThrowException("Numerical viscous flux type not recognized!");
    }
  }
#endif

#ifdef _GPU
    compute_common_dFdU_wrapper(U_d, dU_d, dFcdU_d, dUcdU_d, dFcddU_d, dUbdU_d, ddUbddU_d, P_d, input->AdvDiff_A_d, norm_d, 
        waveSp_d, rus_bias_d, LDG_bias_d, dA_d, input->AdvDiff_D, input->gamma, input->rus_k, input->mu, input->prandtl, 
        input->ldg_b, input->ldg_tau, nVars, nDims, input->equation, startFpt, endFpt, input->viscous, geo->flip_beta_d);

    check_error();
#endif
}

#ifdef _MPI
void Faces::send_U_data()
{
#ifdef _CPU
  /* Stage nonblocking receives */
  unsigned int ridx = 0;
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
    ridx++;
  }

  unsigned int sidx = 0;
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    const auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int i = 0; i < fpts.size(); i++)
      {
        U_sbuffs[sendRank](n, i) = U(0, n, fpts(i));
      }
    }

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }
#endif

#ifdef _GPU
  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    pack_U_wrapper(U_sbuffs_d[sendRank], fpts, U_d, nVars, 0);
  }

  sync_stream(0);

#ifndef _CUDA_AWARE
  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    copy_from_device(U_sbuffs[pairedRank].data(), U_sbuffs_d[pairedRank].data(), U_sbuffs[pairedRank].max_size(), 1);
  }
#endif

  check_error();

#endif
}

void Faces::recv_U_data()
{
  PUSH_NVTX_RANGE("MPI", 0);
#ifdef _GPU
  int ridx = 0;
  /* Stage non-blocking receives */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

#ifndef _CUDA_AWARE
    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
#else
    MPI_Irecv(U_rbuffs_d[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
#endif
    ridx++;
  }

  /* Wait for buffer to host copy to complete */
  sync_stream(1);

  int sidx = 0;
  for (auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;

    /* Send buffer to paired rank */
#ifndef _CUDA_AWARE
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
#else
    MPI_Isend(U_sbuffs_d[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
#endif
    sidx++;
  }
#endif

  /* Wait for comms to finish */
  input->waitTimer.startTimer();
  MPI_Pcontrol(1, "recv_U_data");
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Pcontrol(-1, "recv_U_data");
  POP_NVTX_RANGE;
  input->waitTimer.stopTimer();

#ifdef  _CPU
  /* Unpack buffer */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int i = 0; i < fpts.size(); i++)
      {
        if (input->overset && geo->iblank_face(geo->fpt2face[fpts(i)]) != NORMAL)
          continue;

        U(1, n, fpts(i)) = U_rbuffs[recvRank](n, i);
      }
    }
  }
#endif

#ifdef _GPU
  /* Copy buffer to device (TODO: Use cuda aware MPI for direct transfer) */
#ifndef _CUDA_AWARE
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    copy_to_device(U_rbuffs_d[pairedRank].data(), U_rbuffs[pairedRank].data(), U_rbuffs_d[pairedRank].max_size(), 1);
  }
#endif
  
  sync_stream(1);

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;
    unpack_U_wrapper(U_rbuffs_d[recvRank], fpts, U_d, nVars, 0, input->overset, geo->iblank_fpts_d.data());    
  }

  /* Halt main compute stream until U is unpacked */
  //event_record_wait_pair(1, 1, 0);

  check_error();
#endif

}

void Faces::send_dU_data()
{
  /* Stage all the non-blocking receives */
#ifdef _CPU
  unsigned int ridx = 0;
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    if (input->ldg_b == 0.5 and geo->flip_beta(fpts(0)) == 1) continue;
    else if (input->ldg_b == -0.5 and geo->flip_beta(fpts(0)) == -1) continue;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
    ridx++;
  }

  unsigned int sidx = 0;
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    const auto &fpts = entry.second;

    if (input->ldg_b == 0.5 and geo->flip_beta(fpts(0)) == -1) continue;
    else if (input->ldg_b == -0.5 and geo->flip_beta(fpts(0)) == 1) continue;
    
    /* Pack buffer of solution data at flux points in list */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int i = 0; i < fpts.size(); i++)
        {
          U_sbuffs[sendRank](dim, n, i) = dU(0, dim, n, fpts(i));
        }
      }
    }

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }
#endif

#ifdef _GPU
  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;

    auto flip = geo->flip_beta(geo->fpt_buffer_map[sendRank](0));
    if (input->ldg_b == 0.5 and flip == -1) continue;
    else if (input->ldg_b == -0.5 and flip == 1) continue;
    
    /* Pack buffer of solution data at flux points in list */
    pack_dU_wrapper(U_sbuffs_d[sendRank], fpts, dU_d, nVars, nDims, 0);
  }

  sync_stream(0);

#ifndef _CUDA_AWARE
  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    auto flip = geo->flip_beta(entry.second(0));
    if (input->ldg_b == 0.5 and flip == -1) continue;
    else if (input->ldg_b == -0.5 and flip == 1) continue;

    copy_from_device(U_sbuffs[pairedRank].data(), U_sbuffs_d[pairedRank].data(), U_sbuffs[pairedRank].max_size(), 1);
  }
#endif

  check_error();
#endif
}

void Faces::recv_dU_data()
{
#ifdef _GPU
  int ridx = 0;
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;
    if (input->ldg_b == 0.5 and geo->flip_beta(fpts(0)) == 1) continue;
    else if (input->ldg_b == -0.5 and geo->flip_beta(fpts(0)) == -1) continue;

#ifndef _CUDA_AWARE
    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
#else
    MPI_Irecv(U_rbuffs_d[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
#endif
    ridx++;
  }
  
  sync_stream(1);

  int sidx = 0;
  for (auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    if (input->ldg_b == 0.5 and geo->flip_beta(fpts(0)) == -1) continue;
    else if (input->ldg_b == -0.5 and geo->flip_beta(fpts(0)) == 1) continue;

    /* Send buffer to paired rank */
#ifndef _CUDA_AWARE
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
#else
    MPI_Isend(U_sbuffs_d[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
#endif
    sidx++;
  }

#endif

  PUSH_NVTX_RANGE("MPI", 0)
  /* Wait for comms to finish */
  MPI_Pcontrol(1, "recv_dU_data");
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Pcontrol(-1, "recv_dU_data");
  POP_NVTX_RANGE

  /* Unpack buffer */
#ifdef _CPU
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;
    if (input->ldg_b == 0.5 and geo->flip_beta(fpts(0)) == 1) continue;
    else if (input->ldg_b == -0.5 and geo->flip_beta(fpts(0)) == -1) continue;

    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int i = 0; i < fpts.size(); i++)
        {
          if (input->overset && geo->iblank_face(geo->fpt2face[fpts(i)]) != NORMAL)
            continue;
          dU(1, dim, n, fpts(i)) = U_rbuffs[recvRank](dim, n, i);
        }
      }
    }
  }
#endif

#ifdef _GPU
  /* Copy buffer to device (TODO: Use cuda aware MPI for direct transfer) */
#ifndef _CUDA_AWARE
  for (auto &entry : geo->fpt_buffer_map)
  {
    int pairedRank = entry.first;
    auto flip = geo->flip_beta(entry.second(0));
    if (input->ldg_b == 0.5 and flip == 1) continue;
    else if (input->ldg_b == -0.5 and flip == -1) continue;

    copy_to_device(U_rbuffs_d[pairedRank].data(), U_rbuffs[pairedRank].data(), U_rbuffs_d[pairedRank].max_size(), 1);
  }
#endif

  sync_stream(1);

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;
    auto flip = geo->flip_beta(geo->fpt_buffer_map[recvRank](0));
    if (input->ldg_b == 0.5 and flip == 1) continue;
    else if (input->ldg_b == -0.5 and flip == -1) continue;

    unpack_dU_wrapper(U_rbuffs_d[recvRank], fpts, dU_d, nVars, nDims, 0, input->overset, geo->iblank_fpts_d.data());
  }

  /* Halt main stream until dU is unpacked */
  //event_record_wait_pair(1, 1, 0);

  check_error();
#endif
}
#endif

void Faces::get_U_index(int faceID, int fpt, int& ind, int& stride)
{
  /* U : nFpts x nVars x 2 */
  auto ftype = geo->faceType(faceID);
  int i = geo->face2fpts[ftype](fpt, faceID);
  int ic1 = geo->face2eles(faceID,0);
  int ic2 = geo->face2eles(faceID,1);

  int side = -1;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;
  else if (ic2 >= 0 && geo->iblank_cell(ic2) == NORMAL)
    side = 0;
  else
  {
    printf("face %d: ibf %d | ic1,2: %d,%d, ibc1,2: %d,%d\n",faceID,geo->iblank_face(faceID),ic1,ic2,geo->iblank_cell(ic1),geo->iblank_cell(ic2));
    ThrowException("Face not blanked but both elements are!");
  }

  ind    = std::distance(&U(0,0,0), &U(side,i,i));
  stride = std::distance(&U(side,0,i), &U(side,1,i));
}

double& Faces::get_u_fpt(int faceID, int fpt, int var)
{
  /* U : nFpts x nVars x 2 */
  auto ftype = geo->faceType(faceID);
  int i = geo->face2fpts[ftype](fpt, faceID);
  int ic1 = geo->face2eles(faceID,0);
  int ic2 = geo->face2eles(faceID,1);

  unsigned int side = 0;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;
  else if (ic2 < 0)
    ThrowException("get_u_fpt: Invalid face/cell blanking - check your connectivity.");

  return U(side,var,i);
}

double& Faces::get_grad_fpt(int faceID, int fpt, int var, int dim)
{
  /* U : nFpts x nVars x 2 */
  auto ftype = geo->faceType(faceID);
  int i = geo->face2fpts[ftype](fpt, faceID);
  int ic1 = geo->face2eles(faceID,0);
  int ic2 = geo->face2eles(faceID,1);

  unsigned int side = 0;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;

  return dU(side,dim,var,i);
}

#if defined(_GPU) && defined(_BUILD_LIB)
void Faces::fringe_u_to_device(int* fringeIDs, int nFringe)
{
  if (nFringe == 0) return;

  for (auto ftype : geo->face_set)
    nfringe_type[ftype] = 0;

  for (int i = 0; i < nFringe; i++)
    nfringe_type[geo->faceType(fringeIDs[i])]++;

  for (auto ftype : geo->face_set)
  {
    U_fringe[ftype].resize({geo->nFptsPerFace[ftype], nfringe_type[ftype], nVars});
    fringe_fpts[ftype].resize({geo->nFptsPerFace[ftype], nfringe_type[ftype]});
    fringe_side[ftype].resize({geo->nFptsPerFace[ftype], nfringe_type[ftype]});
  }

  for (auto ftype : geo->face_set)
    nfringe_type[ftype] = 0;

  for (int face = 0; face < nFringe; face++)
  {
    int fid = fringeIDs[face];
    ELE_TYPE ftype = geo->faceType(fid);

    unsigned int side = 0;
    int ic1 = geo->face2eles(fid,0);
    if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
      side = 1;

    for (unsigned int fpt = 0; fpt < geo->nFptsPerFace[ftype]; fpt++)
    {
      fringe_fpts[ftype](fpt,nfringe_type[ftype]) = geo->face2fpts[ftype](fpt, fid);
      fringe_side[ftype](fpt,nfringe_type[ftype]) = side;
    }

    nfringe_type[ftype]++;
  }

  for (unsigned int var = 0; var < nVars; var++)
  {
    std::map<ELE_TYPE, unsigned int> loc_fid;
    for (auto ftype : geo->face_set)
      loc_fid[ftype] = 0;

    for (unsigned int face = 0; face < nFringe; face++)
    {
      ELE_TYPE ftype = geo->faceType(fringeIDs[face]);
      for (unsigned int fpt = 0; fpt < geo->nFptsPerFace[ftype]; fpt++)
      {
        unsigned int gfpt = fringe_fpts[ftype](fpt,face);
        unsigned int side = fringe_side[ftype](fpt,face);
        U_fringe[ftype](fpt,loc_fid[ftype],var) = U(side,var,gfpt);
      }

      loc_fid[ftype]++;
    }
  }

  for (auto ftype : geo->face_set)
  {
    U_fringe_d[ftype] = U_fringe[ftype];
    fringe_fpts_d[ftype] = fringe_fpts[ftype];
    fringe_side_d[ftype] = fringe_side[ftype];

    unpack_fringe_u_wrapper(U_fringe_d[ftype],U_d,U_ldg_d,fringe_fpts_d[ftype],
        fringe_side_d[ftype],nfringe_type[ftype],geo->nFptsPerFace[ftype],nVars,3);
  }

  check_error();
}

void Faces::fringe_grad_to_device(int* fringeIDs, int nFringe)
{
  /* NOTE: Expecting that fringe_u_to_device has already been called for the
   * same set of fringe faces */

  if (nFringe == 0) return;

  dU_fringe.resize({geo->nFptsPerFace, (uint)nFringe, nVars, nDims});

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      for (unsigned int face = 0; face < nFringe; face++)
      {
        for (unsigned int fpt = 0; fpt < geo->nFptsPerFace; fpt++)
        {
          unsigned int gfpt = fringe_fpts(fpt,face);
          unsigned int side = fringe_side(fpt,face);
          dU_fringe(fpt,face,var,dim) = dU(side,dim,var,gfpt);
        }
      }
    }
  }

  dU_fringe_d = dU_fringe;

  unpack_fringe_grad_wrapper(dU_fringe_d,dU_d,fringe_fpts_d,fringe_side_d,
      nFringe,geo->nFptsPerFace,nVars,nDims,3);

  check_error();
}


void Faces::fringe_u_to_device(int* fringeIDs, int nFringe, double* data)
{
  if (nFringe == 0) return;

  U_fringe_d.assign({(uint)nFringe, geo->nFptsPerFace, nVars}, data, 3);

  if (input->motion || input->iter <= input->initIter+1) /// TODO: double-check
  {
    fringe_fpts.resize({geo->nFptsPerFace, (uint)nFringe});
    fringe_side.resize({geo->nFptsPerFace, (uint)nFringe});
    for (int face = 0; face < nFringe; face++)
    {
      unsigned int side = 0;
      int ic1 = geo->face2eles(fringeIDs[face],0);
      if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
        side = 1;

      for (unsigned int fpt = 0; fpt < geo->nFptsPerFace; fpt++)
      {
        fringe_fpts(fpt,face) = geo->face2fpts(fpt, fringeIDs[face]);
        fringe_side(fpt,face) = side;
      }
    }

    fringe_fpts_d.set_size({geo->nFptsPerFace, (uint)nFringe});
    fringe_side_d.set_size({geo->nFptsPerFace, (uint)nFringe});

    fringe_fpts_d = fringe_fpts;
    fringe_side_d = fringe_side;

    sync_stream(0);
  }

  unpack_fringe_u_wrapper(U_fringe_d,U_d,U_ldg_d,fringe_fpts_d,fringe_side_d,nFringe,
      geo->nFptsPerFace,nVars,3);

  check_error();
}

void Faces::fringe_grad_to_device(int* fringeIDs, int nFringe, double *data)
{
  /* NOTE: Expecting that fringe_u_to_device has already been called for the
   * same set of fringe faces */

  if (nFringe == 0) return;

  dU_fringe_d.assign({(uint)nFringe, geo->nFptsPerFace, nDims, nVars}, data, 3);

  unpack_fringe_grad_wrapper(dU_fringe_d,dU_d,fringe_fpts_d,fringe_side_d,
      nFringe,geo->nFptsPerFace,nVars,nDims,3);

  check_error();
}

void Faces::get_face_coords(int* fringeIDs, int nFringe, int* nPtsFace, double* xyz)
{
  if (nFringe == 0) return;

  fringeGFpts.resize({(uint)nFringe, geo->nFptsPerFace});
  for (int face = 0; face < nFringe; face++)
    for (unsigned int fpt = 0; fpt < geo->nFptsPerFace; fpt++)
      fringeGFpts(face,fpt) = geo->face2fpts(fpt, fringeIDs[face]);

  fringeGFpts_d.set_size(fringeGFpts);
  fringeGFpts_d = fringeGFpts;

  //fringeIDs_d.assign({nFaces}, faceIDs);
  fringeCoords_d.set_size({(uint)nFringe,geo->nFptsPerFace,nDims});

  pack_fringe_coords_wrapper(fringeGFpts_d, fringeCoords_d, coord_d, nFringe, geo->nFptsPerFace, nDims);

  copy_from_device(xyz, fringeCoords_d.data(), fringeCoords_d.size());
}

#endif
