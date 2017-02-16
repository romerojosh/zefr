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
  U_bnd.assign({geo->nGfpts_bnd + geo->nGfpts_mpi, nVars});
  U_bnd_ldg.assign({geo->nGfpts_bnd + geo->nGfpts_mpi, nVars});
  Fcomm_bnd.assign({geo->nGfpts_bnd + geo->nGfpts_mpi, nVars});

  /* If viscous, allocate arrays used for LDG flux */
  if(input->viscous)
  {
    dU_bnd.assign({geo->nGfpts_bnd + geo->nGfpts_mpi, nVars, nDims});
    Ucomm_bnd.assign({geo->nGfpts_bnd + geo->nGfpts_mpi, nVars});
  }

  /* Initialize Riemann/LDG bias variables to zero */
  LDG_bias.assign({nFpts}, 0);
  rus_bias.assign({nFpts}, 0);

  /* Allocate memory for implicit method data structures */
  if (input->dt_scheme == "MCGS")
  {
    /* Index Map Notes:
     * Common Value Slots:  Sloti:  dFdUL, dFdUR  Slotj:  L, R
     * dFddUvisc:           nDimsi: Fx, Fy        nDimsj: dUdx, dUdy
     * ddURddUL:            nDimsi: dURdx, dURdy  nDimsj: dULdx, dULdy
     */
    dFdUconv.assign({nFpts, nVars, nVars, nDims, 2});

    dFcdU.assign({nFpts, nVars, nVars, 2, 2});
    dFndUL_temp.assign({nFpts, nVars, nVars});
    dFndUR_temp.assign({nFpts, nVars, nVars});

    /* Temp strutures for inviscid boundary conditions */
    dFdURconv.assign({nVars, nVars, nDims});
    dURdUL.assign({nVars, nVars});

    if(input->viscous)
    {
      dFdUvisc.assign({nFpts, nVars, nVars, nDims, 2});
      dFddUvisc.assign({nFpts, nVars, nVars, nDims, nDims, 2}); 
      dUcdU.assign({nFpts, nVars, nVars, 2});
      dFcddU.assign({nFpts, nVars, nVars, nDims, 2, 2});
      dFnddUL_temp.assign({nFpts, nVars, nVars, nDims});
      dFnddUR_temp.assign({nFpts, nVars, nVars, nDims});

      // TODO: May be able to remove these
      dFcdU_temp.assign({nFpts, nVars, nVars, 2});
      dFcddU_temp.assign({nFpts, nVars, nVars, nDims, 2});

      /* Temp strutures for viscous boundary conditions */
      dUcdUR.assign({nVars, nVars});
      dFdURvisc.assign({nVars, nVars, nDims});
      dFddURvisc.assign({nVars, nVars, nDims, nDims});
      ddURddUL.assign({nVars, nVars, nDims, nDims});
    }
  }

  /* If running Euler/NS, allocate memory for pressure */
  if (input->equation == EulerNS)
    P.assign({nFpts, 2});

  waveSp.assign({nFpts}, 0.0);
  if (input->viscous)
    diffCo.assign({nFpts}, 0.0);

  /* Allocate memory for geometry structures */
  coord.assign({nFpts, nDims});
  norm.assign({nFpts, nDims});
  dA.assign({nFpts, 2},0.0);
  jaco.assign({nFpts, nDims, nDims , 2}); // TODO - remove

  /* Moving-grid-related structures */
  if (input->motion)
  {
    Vg.assign({nFpts, nDims}, 0.0);
  }

#ifdef _MPI
  /* Allocate memory for send/receive buffers */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int pairedRank = entry.first;
    const auto &fpts = entry.second;

    if (input->viscous)
    {
      U_sbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars, nDims}, 0.0, true);
      U_rbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars, nDims}, 0.0, true);
    }
    else
    {
      U_sbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars}, 0.0, true);
      U_rbuffs[pairedRank].assign({(unsigned int) fpts.size(), nVars}, 0.0, true);
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
#pragma omp parallel for private(VL,VR)
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    switch(bnd_id)
    {
      case SUP_IN: /* Farfield and Supersonic Inlet */
      {
        if (input->equation == AdvDiff)
        {
          /* Set boundaries to zero */
          U(fpt, 0, 1) = 0;
          U_ldg(fpt, 0, 1) = 0;
        }
        else
        {
          /* Set boundaries to freestream values */
          U(fpt, 0, 1) = input->rho_fs;
          U_ldg(fpt, 0, 1) = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            U(fpt, dim+1, 1) = input->rho_fs * input->V_fs(dim);
            U_ldg(fpt, dim+1, 1) = input->rho_fs * input->V_fs(dim);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          U(fpt, nDims + 1, 1) = input->P_fs/(input->gamma-1.0) + 0.5*input->rho_fs * Vsq; 
          U_ldg(fpt, nDims + 1, 1) = input->P_fs/(input->gamma-1.0) + 0.5*input->rho_fs * Vsq; 
        }

        break;
      }

      case SUP_OUT: /* Supersonic Outlet */
      {
        if (input->viscous)
          ThrowException("SUP_OUT broken for viscous! Use characteristic or fix it!");

        /* Extrapolate boundary values from interior */
        for (unsigned int n = 0; n < nVars; n++)
          U(fpt, n, 1) = U(fpt, n, 0);

        break;
      }

      case SUB_IN: /* Subsonic Inlet */
      {
        if (input->viscous)
          ThrowException("SUB_IN broken for viscous! Use characteristic or fix it!");

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
          VnL += VL[dim] * norm(fpt, dim);
          alpha += input->norm_fs(dim) * norm(fpt, dim);
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

        break;
      }

      case SUB_OUT: /* Subsonic Outlet */
      {
        if (input->viscous)
          ThrowException("SUB_OUT broken for viscous! Use characteristic or fix it!");

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
        LDG_bias(fpt) = 1;

        break;
      }

      case CHAR: /* Characteristic (from PyFR) */
      {
        /* Compute wall normal velocities */
        double VnL = 0.0; double VnR = 0.0;

        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += U(fpt, dim+1, 0) / U(fpt, 0, 0) * norm(fpt, dim);
          VnR += input->V_fs(dim) * norm(fpt, dim);
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

        double cL = std::sqrt(input->gamma * PL / U(fpt, 0, 0));
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
            VR[dim] = input->V_fs(dim) + (ustarn - VnR) * norm(fpt, dim);
        }
        else  /* Case 2: Outflow */
        {
          rhoR *= std::pow(U(fpt, 0, 0), input->gamma) / PL;

          for (unsigned int dim = 0; dim < nDims; dim++)
            VR[dim] = U(fpt, dim+1, 0) / U(fpt, 0, 0) + (ustarn - VnL) * norm(fpt, dim);
        }

        rhoR = std::pow(rhoR, 1.0 / (input->gamma - 1));

        U(fpt, 0, 1) = rhoR;
        U_ldg(fpt, 0, 1) = rhoR;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(fpt, dim + 1, 1) = rhoR * VR[dim];
          U_ldg(fpt, dim + 1, 1) = rhoR * VR[dim];
        }

        PR = rhoR / input->gamma * cstar * cstar;
        U(fpt, nDims + 1, 1) = PR / (input->gamma - 1);
        U_ldg(fpt, nDims + 1, 1) = PR / (input->gamma - 1);
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          U(fpt, nDims+1, 1) += 0.5 * rhoR * VR[dim] * VR[dim];
          U_ldg(fpt, nDims+1, 1) += 0.5 * rhoR * VR[dim] * VR[dim];
        }

        /* Set bias */
        LDG_bias(fpt) = 1;

        break;
      }

      case SYMMETRY_P: /* Symmetry (prescribed) */
      case SLIP_WALL_P: /* Slip Wall (prescribed) */
      {
        if (input->viscous)
          ThrowException("SLIP_WALL_P not supported for viscous!");

        double momN = 0.0;

        /* Compute wall normal momentum */
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN += U(fpt, dim+1, 0) * norm(fpt, dim);

        if (input->motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN -= U(fpt, 0, 0) * Vg(fpt, dim) * norm(fpt, dim);
        }

        U(fpt, 0, 1) = U(fpt, 0, 0);

        /* Set boundary state with cancelled normal velocity */
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim+1, 1) = U(fpt, dim+1, 0) - momN * norm(fpt, dim);

        /* Set energy */
        /* Get left-state pressure */
        double momFL = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          momFL += U(fpt, dim + 1, 0) * U(fpt, dim + 1, 0);

        double PL = (input->gamma - 1.0) * (U(fpt, nDims + 1 , 0) - 0.5 * momFL / U(fpt, 0, 0));

        /* Get right-state momentum flux after velocity correction */
        double momFR = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
          momFR += U(fpt, dim + 1, 1) * U(fpt, dim + 1, 1);

        /* Compute energy with extrapolated pressure and new momentum */
        U(fpt, nDims + 1, 1) = PL / (input->gamma - 1)  + 0.5 * momFR / U(fpt, 0, 1);

        /* Set bias */
        rus_bias(fpt) = 1;

        break;
      }

      case SYMMETRY_G: /* Symmetry (ghost) */
      case SLIP_WALL_G: /* Slip Wall (ghost) */
      {
        if (input->viscous)
          ThrowException("SLIP_WALL_G not supported for viscous!");

        double momN = 0.0;

        /* Compute wall normal momentum */
        for (unsigned int dim = 0; dim < nDims; dim++)
          momN += U(fpt, dim+1, 0) * norm(fpt, dim);

        if (input->motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            momN -= U(fpt, 0, 0) * Vg(fpt, dim) * norm(fpt, dim);
        }

        U(fpt, 0, 1) = U(fpt, 0, 0);

        /* Set boundary state to reflect normal velocity */
        for (unsigned int dim = 0; dim < nDims; dim++)
          U(fpt, dim+1, 1) = U(fpt, dim+1, 0) - 2.0 * momN * norm(fpt, dim);

        /* Set energy */
        U(fpt, nDims + 1, 1) = U(fpt, nDims + 1, 0);

        break;
      }

      case ISOTHERMAL_NOSLIP_P: /* Isothermal No-slip Wall (prescribed) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        if (input->motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            VG[dim] = Vg(fpt, dim);
        }

        double rhoL = U(fpt, 0, 0);

        U(fpt, 0, 1) = rhoL;
        U_ldg(fpt, 0, 1) = rhoL;

        /* Set velocity to zero (or grid wall velocity) */
        double Vsq = 0; double Vsq_grid = 0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(fpt, dim+1, 0) / rhoL;
          double V = -VL + 2 * VG[dim];
          U(fpt, dim+1, 1) = rhoL * V;
          Vsq += V * V;

          U_ldg(fpt, dim+1, 1) =  VG[dim];
          Vsq_grid += VG[dim] * VG[dim];
        }
          
        double cp_over_gam =  input->R_ref / (input->gamma - 1);

        U(fpt, nDims + 1, 1) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq) ;
        U_ldg(fpt, nDims + 1, 1) = rhoL * cp_over_gam * input->T_wall;

        /* Set bias */
        LDG_bias(fpt) = 1;

        break;
      }

      case ISOTHERMAL_NOSLIP_G: /* Isothermal No-slip Wall (ghost) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("WALL_NS_ISO_G not implemented");

        break;
      }


      case ISOTHERMAL_NOSLIP_MOVING_P: /* No-slip Wall (isothermal and moving) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        double rhoL = U(fpt, 0, 0);

        U(fpt, 0, 1) = rhoL;
        U_ldg(fpt, 0, 1) = rhoL;

        /* Set velocity to zero (or wall velocity) */
        double Vsq = 0; double Vsq_wall = 0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(fpt, dim+1, 0) / U(fpt, 0, 0);
          double V = -VL + 2*(input->V_wall(dim));
          U(fpt, dim+1, 1) = rhoL * V;
          Vsq += V * V;

          U_ldg(fpt, dim+1, 1) = rhoL*input->V_wall(dim);
          Vsq_wall += input->V_wall(dim) * input->V_wall(dim);
        }
          
        double cp_over_gam =  input->R_ref / (input->gamma - 1);

        U(fpt, nDims + 1, 1) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq);
        U_ldg(fpt, nDims + 1, 1) = rhoL * (cp_over_gam * input->T_wall + 0.5 * Vsq_wall);

        /* Set bias */
        LDG_bias(fpt) = 1;
        
        break;
      }

      case ISOTHERMAL_NOSLIP_MOVING_G: /* Isothermal No-slip Wall, moving (ghost) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("WALL_NS_ISO_MOVE_P not implemented yet!");
        break;
      }


      case ADIABATIC_NOSLIP_P: /* Adiabatic No-slip Wall (prescribed) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        if (input->motion)
        {
          for (unsigned int dim = 0; dim < nDims; dim++)
            VG[dim] = Vg(fpt, dim);
        }

        /* Extrapolate density */
        double rhoL = U(fpt, 0, 0);
        U(fpt, 0, 1) = rhoL;
        U_ldg(fpt, 0, 1) = rhoL;

        /* Set right state (common) velocity to zero (or wall velocity) */
        double Vsq = 0.0; double VLsq = 0.0; double Vsq_grid = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(fpt, dim+1, 0) / rhoL; 
          double V = -VL + 2 * VG[dim];
          U(fpt, dim+1, 1) = rhoL * V;
          U_ldg(fpt, dim+1, 1) = rhoL * VG[dim];

          Vsq += V * V;
          VLsq += VL * VL;
          Vsq_grid += VG[dim] * VG[dim];
        }

        double EL = U(fpt, nDims + 1, 0);
        U(fpt, nDims + 1, 1) = EL + 0.5 * rhoL * (Vsq - VLsq);
        U_ldg(fpt, nDims + 1, 1) = EL + 0.5 * rhoL * (Vsq_grid - VLsq);

        /* Set LDG bias */
        LDG_bias(fpt) = 1;

        break;
      }

      case ADIABATIC_NOSLIP_G: /* Adiabatic No-slip Wall (ghost) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("WALL_NS_ADI_G not implemented");

        break;
      }

      case ADIABATIC_NOSLIP_MOVING_P: /* Adiabatic No-slip Wall, moving (prescribed) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        /* Extrapolate density */
        double rhoL = U(fpt, 0, 0);
        U(fpt, 0, 1) = rhoL;
        U_ldg(fpt, 0, 1) = rhoL;

        /* Set right state (common) velocity to zero (or wall velocity) */
        double Vsq = 0.0; double VLsq = 0.0; double Vsq_wall = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          double VL = U(fpt, dim+1, 0) / rhoL; 
          double V = -VL + 2 * input->V_wall(dim);
          U(fpt, dim+1, 1) = rhoL * V;
          U_ldg(fpt, dim+1, 1) = rhoL * input->V_wall(dim);

          Vsq += V * V;
          VLsq += VL * VL;
          Vsq_wall += input->V_wall(dim) * input->V_wall(dim);
        }

        double EL = U(fpt, nDims + 1, 0);
        U(fpt, nDims + 1, 1) = EL + 0.5 * rhoL * (Vsq - VLsq);
        U_ldg(fpt, nDims + 1, 1) = EL - 0.5 * rhoL * (VLsq + Vsq_wall);

        /* Set LDG bias */
        LDG_bias(fpt) = 1;

        break;
      }

      case ADIABATIC_NOSLIP_MOVING_G: /* Adiabatic No-slip Wall, moving (ghost) */
      {
        if (!input->viscous)
          ThrowException("No slip wall boundary only for viscous flows!");

        ThrowException("WALL_NS_ADI_MOVE_G not implemented yet!");

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
      input->P_fs, input->gamma, input->R_ref, input->T_tot_fs, input->P_tot_fs, input->T_wall, input->V_wall_d, 
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
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

    /* Apply specified boundary condition */
    if(bnd_id == ADIABATIC_NOSLIP_P || bnd_id == ADIABATIC_NOSLIP_G ||
            bnd_id == ADIABATIC_NOSLIP_MOVING_P || bnd_id == ADIABATIC_NOSLIP_MOVING_G) /* Adibatic Wall */
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
        double dT_dn = dT_dx * norm(fpt, 0) + dT_dy * norm(fpt, 1);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(fpt, 3, 0, 1) = E_dx - dT_dn * norm(fpt, 0);
        dU(fpt, 3, 1, 1) = E_dy - dT_dn * norm(fpt, 1);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 3, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0)) + rho_dx * U(fpt, 3, 1) / rho; 
        //dU(fpt, 3, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1)) + rho_dy * U(fpt, 3, 1) / rho; 
      }
      else if (bnd_id == OVERSET)
      {
        // Do nothing...? [need to treat same as internal]
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
        double dT_dn = dT_dx * norm(fpt, 0) + dT_dy * norm(fpt, 1) + dT_dz * norm(fpt, 2);

        /* Option 1: Simply remove contribution of dT from total energy gradient */
        dU(fpt, 4, 0, 1) = E_dx - dT_dn * norm(fpt, 0);
        dU(fpt, 4, 1, 1) = E_dy - dT_dn * norm(fpt, 1);
        dU(fpt, 4, 2, 1) = E_dz - dT_dn * norm(fpt, 2);

        /* Option 2: Reconstruct energy gradient using right states (E = E_r, u = 0, v = 0, rho = rho_r = rho_l) */
        //dU(fpt, 4, 0, 1) = (dT_dx - dT_dn * norm(fpt, 0)) + rho_dx * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 1, 1) = (dT_dy - dT_dn * norm(fpt, 1)) + rho_dy * U(fpt, 4, 1) / rho; 
        //dU(fpt, 4, 2, 1) = (dT_dz - dT_dn * norm(fpt, 2)) + rho_dz * U(fpt, 4, 1) / rho; 
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
      nDims, geo->gfpt2bnd_d, input->equation);

  check_error();
#endif
}

// TODO: Add a check to ensure proper boundary conditions are used
// TODO: Collapse 2D and 3D boundary condition cases
void Faces::apply_bcs_dFdU()
{
#ifdef _CPU
  /* Loop over boundary flux points */
#pragma omp parallel for
  for (unsigned int fpt = geo->nGfpts_int; fpt < geo->nGfpts_int + geo->nGfpts_bnd; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    unsigned int bnd_id = geo->gfpt2bnd(fpt - geo->nGfpts_int);

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
            dFdURconv(ni, nj, dim) = dFdUconv(fpt, ni, nj, dim, 1);
          }
        }
      }

      if (input->viscous)
      {
        /* Copy right state dUcdU */
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dUcdUR(ni, nj) = dUcdU(fpt, ni, nj, 1);
          }
        }

        /* Copy right state dFdUvisc */
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          for (unsigned int nj = 0; nj < nVars; nj++)
          {
            for (unsigned int ni = 0; ni < nVars; ni++)
            {
              dFdURvisc(ni, nj, dim) = dFdUvisc(fpt, ni, nj, dim, 1);
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
                  dFddURvisc(ni, nj, dimi, dimj) = dFddUvisc(fpt, ni, nj, dimi, dimj, 1);
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
      case PERIODIC:/* Periodic */
      {
        break;
      }

      case SUB_OUT: /* Subsonic Outlet */
      {
        /* Primitive Variables */
        double uL = U(fpt, 1, 0) / U(fpt, 0, 0);
        double vL = U(fpt, 2, 0) / U(fpt, 0, 0);

        /* Compute dURdUL */
        dURdUL(0, 0) = 1;
        dURdUL(1, 0) = 0;
        dURdUL(2, 0) = 0;
        dURdUL(3, 0) = -0.5 * (uL*uL + vL*vL);

        dURdUL(0, 1) = 0;
        dURdUL(1, 1) = 1;
        dURdUL(2, 1) = 0;
        dURdUL(3, 1) = uL;

        dURdUL(0, 2) = 0;
        dURdUL(1, 2) = 0;
        dURdUL(2, 2) = 1;
        dURdUL(3, 2) = vL;

        dURdUL(0, 3) = 0;
        dURdUL(1, 3) = 0;
        dURdUL(2, 3) = 0;
        dURdUL(3, 3) = 0;

        break;
      }

      case CHAR: /* Characteristic (from PyFR) */
      {
        /* Compute wall normal velocities */
        double VnL = 0.0; double VnR = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          VnL += U(fpt, dim+1, 0) / U(fpt, 0, 0) * norm(fpt, dim);
          VnR += input->V_fs(dim) * norm(fpt, dim);
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

        double cL = std::sqrt(input->gamma * PL / U(fpt, 0, 0));
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
          double nx = norm(fpt, 0);
          double ny = norm(fpt, 1);
          double gam = input->gamma;

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
            dURdUL(0, 0) = a1 * b1;
            dURdUL(1, 0) = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
            dURdUL(2, 0) = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
            dURdUL(3, 0) = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

            dURdUL(0, 1) = a1 * b2;
            dURdUL(1, 1) = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
            dURdUL(2, 1) = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
            dURdUL(3, 1) = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

            dURdUL(0, 2) = a1 * b3;
            dURdUL(1, 2) = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
            dURdUL(2, 2) = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
            dURdUL(3, 2) = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

            dURdUL(0, 3) = 0.5 * rhoR * b4;
            dURdUL(1, 3) = 0.5 * rhoR * (b4 * uR + a2 * nx);
            dURdUL(2, 3) = 0.5 * rhoR * (b4 * vR + a2 * ny);
            dURdUL(3, 3) = 0.5 * rhoR * (b4 * c1 + a2 * c2);
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
            dURdUL(0, 0) = a1 * e1;
            dURdUL(1, 0) = a1 * e1 * c4 + rhoR * c1;
            dURdUL(2, 0) = a1 * e1 * d4 + rhoR * d1;
            dURdUL(3, 0) = rhoR * (c1*c4 + d1*d4) + e1 * f1 + a6 * b1;

            dURdUL(0, 1) = a1 * e2;
            dURdUL(1, 1) = a1 * e2 * c4 + rhoR * c2;
            dURdUL(2, 1) = a1 * e2 * d4 + rhoR * d2;
            dURdUL(3, 1) = rhoR * (c2*c4 + d2*d4) + e2 * f1 + a6 * b2;

            dURdUL(0, 2) = a1 * e3;
            dURdUL(1, 2) = a1 * e3 * c4 + rhoR * c3;
            dURdUL(2, 2) = a1 * e3 * d4 + rhoR * d3;
            dURdUL(3, 2) = rhoR * (c3*c4 + d3*d4) + e3 * f1 + a6 * b3;

            dURdUL(0, 3) = a1 * e4;
            dURdUL(1, 3) = a1 * e4 * c4 + 0.5 * rhoR * a2 * nx;
            dURdUL(2, 3) = a1 * e4 * d4 + 0.5 * rhoR * a2 * ny;
            dURdUL(3, 3) = 0.5 * rhoR * a2 * (c4*nx + d4*ny) + e4 * f1 + a2 * a6;
          }
        }

        else if (nDims == 3)
        {
          double nx = norm(fpt, 0);
          double ny = norm(fpt, 1);
          double nz = norm(fpt, 2);
          double gam = input->gamma;

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
            dURdUL(0, 0) = a1 * b1;
            dURdUL(1, 0) = a1 * b1 * uR + 0.5 * rhoR * b1 * nx;
            dURdUL(2, 0) = a1 * b1 * vR + 0.5 * rhoR * b1 * ny;
            dURdUL(3, 0) = a1 * b1 * wR + 0.5 * rhoR * b1 * nz;
            dURdUL(4, 0) = a1 * b1 * c1 + 0.5 * rhoR * b1 * c2;

            dURdUL(0, 1) = a1 * b2;
            dURdUL(1, 1) = a1 * b2 * uR + 0.5 * rhoR * b2 * nx;
            dURdUL(2, 1) = a1 * b2 * vR + 0.5 * rhoR * b2 * ny;
            dURdUL(3, 1) = a1 * b2 * wR + 0.5 * rhoR * b2 * nz;
            dURdUL(4, 1) = a1 * b2 * c1 + 0.5 * rhoR * b2 * c2;

            dURdUL(0, 2) = a1 * b3;
            dURdUL(1, 2) = a1 * b3 * uR + 0.5 * rhoR * b3 * nx;
            dURdUL(2, 2) = a1 * b3 * vR + 0.5 * rhoR * b3 * ny;
            dURdUL(3, 2) = a1 * b3 * wR + 0.5 * rhoR * b3 * nz;
            dURdUL(4, 2) = a1 * b3 * c1 + 0.5 * rhoR * b3 * c2;

            dURdUL(0, 3) = a1 * b4;
            dURdUL(1, 3) = a1 * b4 * uR + 0.5 * rhoR * b4 * nx;
            dURdUL(2, 3) = a1 * b4 * vR + 0.5 * rhoR * b4 * ny;
            dURdUL(3, 3) = a1 * b4 * wR + 0.5 * rhoR * b4 * nz;
            dURdUL(4, 3) = a1 * b4 * c1 + 0.5 * rhoR * b4 * c2;

            dURdUL(0, 4) = 0.5 * rhoR * b5;
            dURdUL(1, 4) = 0.5 * rhoR * (b5 * uR + a2 * nx);
            dURdUL(2, 4) = 0.5 * rhoR * (b5 * vR + a2 * ny);
            dURdUL(3, 4) = 0.5 * rhoR * (b5 * wR + a2 * nz);
            dURdUL(4, 4) = 0.5 * rhoR * (b5 * c1 + a2 * c2);
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
            dURdUL(0, 0) = a1 * f1;
            dURdUL(1, 0) = a1 * f1 * c5 + rhoR * c1;
            dURdUL(2, 0) = a1 * f1 * d5 + rhoR * d1;
            dURdUL(3, 0) = a1 * f1 * e5 + rhoR * e1;
            dURdUL(4, 0) = rhoR * (c1*c5 + d1*d5 + e1*e5) + f1 * g1 + a6 * b1;

            dURdUL(0, 1) = a1 * f2;
            dURdUL(1, 1) = a1 * f2 * c5 + rhoR * c2;
            dURdUL(2, 1) = a1 * f2 * d5 + rhoR * d2;
            dURdUL(3, 1) = a1 * f2 * e5 + rhoR * e2;
            dURdUL(4, 1) = rhoR * (c2*c5 + d2*d5 + e2*e5) + f2 * g1 + a6 * b2;

            dURdUL(0, 2) = a1 * f3;
            dURdUL(1, 2) = a1 * f3 * c5 + rhoR * c3;
            dURdUL(2, 2) = a1 * f3 * d5 + rhoR * d3;
            dURdUL(3, 2) = a1 * f3 * e5 + rhoR * e3;
            dURdUL(4, 2) = rhoR * (c3*c5 + d3*d5 + e3*e5) + f3 * g1 + a6 * b3;

            dURdUL(0, 3) = a1 * f4;
            dURdUL(1, 3) = a1 * f4 * c5 + rhoR * c4;
            dURdUL(2, 3) = a1 * f4 * d5 + rhoR * d4;
            dURdUL(3, 3) = a1 * f4 * e5 + rhoR * e4;
            dURdUL(4, 3) = rhoR * (c4*c5 + d4*d5 + e4*e5) + f4 * g1 + a6 * b4;

            dURdUL(0, 4) = a1 * f5;
            dURdUL(1, 4) = a1 * f5 * c5 + 0.5 * rhoR * a2 * nx;
            dURdUL(2, 4) = a1 * f5 * d5 + 0.5 * rhoR * a2 * ny;
            dURdUL(3, 4) = a1 * f5 * e5 + 0.5 * rhoR * a2 * nz;
            dURdUL(4, 4) = 0.5 * rhoR * a2 * (c5*nx + d5*ny + e5*nz) + f5 * g1 + a2 * a6;
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
          dURdUL(0, 0) = 1;
          dURdUL(1, 0) = 0;
          dURdUL(2, 0) = 0;
          dURdUL(3, 0) = 0.5 * (uL*uL + vL*vL - uR*uR - vR*vR);

          dURdUL(0, 1) = 0;
          dURdUL(1, 1) = 1.0-nx*nx;
          dURdUL(2, 1) = -nx*ny;
          dURdUL(3, 1) = -uL + (1.0-nx*nx)*uR - nx*ny*vR;

          dURdUL(0, 2) = 0;
          dURdUL(1, 2) = -nx*ny;
          dURdUL(2, 2) = 1.0-ny*ny;
          dURdUL(3, 2) = -vL - nx*ny*uR + (1.0-ny*ny)*vR;

          dURdUL(0, 3) = 0;
          dURdUL(1, 3) = 0;
          dURdUL(2, 3) = 0;
          dURdUL(3, 3) = 1;
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
          dURdUL(0, 0) = 1;
          dURdUL(1, 0) = 0;
          dURdUL(2, 0) = 0;
          dURdUL(3, 0) = 0;
          dURdUL(4, 0) = 0.5 * (uL*uL + vL*vL + wL*wL - uR*uR - vR*vR - wR*wR);

          dURdUL(0, 1) = 0;
          dURdUL(1, 1) = 1.0-nx*nx;
          dURdUL(2, 1) = -nx*ny;
          dURdUL(3, 1) = -nx*nz;
          dURdUL(4, 1) = -uL + (1.0-nx*nx)*uR - nx*ny*vR - nx*nz*wR;

          dURdUL(0, 2) = 0;
          dURdUL(1, 2) = -nx*ny;
          dURdUL(2, 2) = 1.0-ny*ny;
          dURdUL(3, 2) = -ny*nz;
          dURdUL(4, 2) = -vL - nx*ny*uR + (1.0-ny*ny)*vR - ny*nz*wR;

          dURdUL(0, 3) = 0;
          dURdUL(1, 3) = -nx*nz;
          dURdUL(2, 3) = -ny*nz;
          dURdUL(3, 3) = 1.0-nz*nz;
          dURdUL(4, 3) = -wL - nx*nz*uR - ny*nz*vR + (1.0-nz*nz)*wR;

          dURdUL(0, 4) = 0;
          dURdUL(1, 4) = 0;
          dURdUL(2, 4) = 0;
          dURdUL(3, 4) = 0;
          dURdUL(4, 4) = 1;
        }

        break;
      }

      case SYMMETRY_G: /* Symmetry (ghost) */
      case SLIP_WALL_G: /* Slip Wall (ghost) */
      {
        double nx = norm(fpt, 0);
        double ny = norm(fpt, 1);

        dURdUL(0, 0) = 1;
        dURdUL(1, 0) = 0;
        dURdUL(2, 0) = 0;
        dURdUL(3, 0) = 0;

        dURdUL(0, 1) = 0;
        dURdUL(1, 1) = 1.0 - 2.0 * nx * nx;
        dURdUL(2, 1) = -2.0 * nx * ny;
        dURdUL(3, 1) = 0;

        dURdUL(0, 2) = 0;
        dURdUL(1, 2) = -2.0 * nx * ny;
        dURdUL(2, 2) = 1.0 - 2.0 * ny * ny;
        dURdUL(3, 2) = 0;

        dURdUL(0, 3) = 0;
        dURdUL(1, 3) = 0;
        dURdUL(2, 3) = 0;
        dURdUL(3, 3) = 1;

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
        dURdUL(0, 0) = 1;
        dURdUL(1, 0) = 0;
        dURdUL(2, 0) = 0;
        dURdUL(3, 0) = 0.5 * (uL*uL + vL*vL);

        dURdUL(0, 1) = 0;
        dURdUL(1, 1) = 0;
        dURdUL(2, 1) = 0;
        dURdUL(3, 1) = -uL;

        dURdUL(0, 2) = 0;
        dURdUL(1, 2) = 0;
        dURdUL(2, 2) = 0;
        dURdUL(3, 2) = -vL;

        dURdUL(0, 3) = 0;
        dURdUL(1, 3) = 0;
        dURdUL(2, 3) = 0;
        dURdUL(3, 3) = 1;

        if (input->viscous)
        {
          /* Compute dUxR/dUxL */
          ddURddUL(0, 0, 0, 0) = 1;
          ddURddUL(1, 0, 0, 0) = 0;
          ddURddUL(2, 0, 0, 0) = 0;
          ddURddUL(3, 0, 0, 0) = nx*nx * (eL / rhoL - (uL*uL + vL*vL));

          ddURddUL(0, 1, 0, 0) = 0;
          ddURddUL(1, 1, 0, 0) = 1;
          ddURddUL(2, 1, 0, 0) = 0;
          ddURddUL(3, 1, 0, 0) = nx*nx * uL;

          ddURddUL(0, 2, 0, 0) = 0;
          ddURddUL(1, 2, 0, 0) = 0;
          ddURddUL(2, 2, 0, 0) = 1;
          ddURddUL(3, 2, 0, 0) = nx*nx * vL;

          ddURddUL(0, 3, 0, 0) = 0;
          ddURddUL(1, 3, 0, 0) = 0;
          ddURddUL(2, 3, 0, 0) = 0;
          ddURddUL(3, 3, 0, 0) = 1.0 - nx*nx;

          /* Compute dUyR/dUxL */
          ddURddUL(0, 0, 1, 0) = 0;
          ddURddUL(1, 0, 1, 0) = 0;
          ddURddUL(2, 0, 1, 0) = 0;
          ddURddUL(3, 0, 1, 0) = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

          ddURddUL(0, 1, 1, 0) = 0;
          ddURddUL(1, 1, 1, 0) = 0;
          ddURddUL(2, 1, 1, 0) = 0;
          ddURddUL(3, 1, 1, 0) = nx*ny * uL;

          ddURddUL(0, 2, 1, 0) = 0;
          ddURddUL(1, 2, 1, 0) = 0;
          ddURddUL(2, 2, 1, 0) = 0;
          ddURddUL(3, 2, 1, 0) = nx*ny * vL;

          ddURddUL(0, 3, 1, 0) = 0;
          ddURddUL(1, 3, 1, 0) = 0;
          ddURddUL(2, 3, 1, 0) = 0;
          ddURddUL(3, 3, 1, 0) = -nx * ny;

          /* Compute dUxR/dUyL */
          ddURddUL(0, 0, 0, 1) = 0;
          ddURddUL(1, 0, 0, 1) = 0;
          ddURddUL(2, 0, 0, 1) = 0;
          ddURddUL(3, 0, 0, 1) = nx*ny * (eL / rhoL - (uL*uL + vL*vL));

          ddURddUL(0, 1, 0, 1) = 0;
          ddURddUL(1, 1, 0, 1) = 0;
          ddURddUL(2, 1, 0, 1) = 0;
          ddURddUL(3, 1, 0, 1) = nx*ny * uL;

          ddURddUL(0, 2, 0, 1) = 0;
          ddURddUL(1, 2, 0, 1) = 0;
          ddURddUL(2, 2, 0, 1) = 0;
          ddURddUL(3, 2, 0, 1) = nx*ny * vL;

          ddURddUL(0, 3, 0, 1) = 0;
          ddURddUL(1, 3, 0, 1) = 0;
          ddURddUL(2, 3, 0, 1) = 0;
          ddURddUL(3, 3, 0, 1) = -nx * ny;

          /* Compute dUyR/dUyL */
          ddURddUL(0, 0, 1, 1) = 1;
          ddURddUL(1, 0, 1, 1) = 0;
          ddURddUL(2, 0, 1, 1) = 0;
          ddURddUL(3, 0, 1, 1) = ny*ny * (eL / rhoL - (uL*uL + vL*vL));

          ddURddUL(0, 1, 1, 1) = 0;
          ddURddUL(1, 1, 1, 1) = 1;
          ddURddUL(2, 1, 1, 1) = 0;
          ddURddUL(3, 1, 1, 1) = ny*ny * uL;

          ddURddUL(0, 2, 1, 1) = 0;
          ddURddUL(1, 2, 1, 1) = 0;
          ddURddUL(2, 2, 1, 1) = 1;
          ddURddUL(3, 2, 1, 1) = ny*ny * vL;

          ddURddUL(0, 3, 1, 1) = 0;
          ddURddUL(1, 3, 1, 1) = 0;
          ddURddUL(2, 3, 1, 1) = 0;
          ddURddUL(3, 3, 1, 1) = 1.0 - ny*ny;
        }

        break;
      }

      default:
      {
        ThrowException("Boundary condition not implemented for implicit method");
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
              val += dFdURconv(i, k, dim) * dURdUL(k, j);
            }
            dFdUconv(fpt, i, j, dim, 1) = val;
          }
        }
      }

      if (input->viscous)
      {
        /* Compute dUcdUL for right state */
        for (unsigned int j = 0; j < nVars; j++)
        {
          for (unsigned int i = 0; i < nVars; i++)
          {
            double val = 0;
            for (unsigned int k = 0; k < nVars; k++)
            {
              val += dUcdUR(i, k) * dURdUL(k, j);
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
                val += dFdURvisc(i, k, dim) * dURdUL(k, j);
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
                      val += dFddURvisc(i, k, dimi, dimk) * ddURddUL(k, j, dimk, dimj);
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
#endif

#ifdef _GPU
  apply_bcs_dFdU_wrapper(U_d, dFdUconv_d, dFdUvisc_d, dUcdU_d, dFddUvisc_d,
      geo->nGfpts_int, geo->nGfpts_bnd, nVars, nDims, input->rho_fs, input->V_fs_d, 
      input->P_fs, input->gamma, norm_d, geo->gfpt2bnd_d, input->equation, input->viscous);
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


#pragma omp parallel for firstprivate(FL, FR, UL, UR)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      UL[n] = U(fpt, n, 0); UR[n] = U(fpt, n, 1);
    }


    double eig = 0;
    double Vgn = 0;
    if (input->motion)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        Vgn += Vg(fpt, dim) * norm(fpt, dim);
    }

    /* Get numerical wavespeed */
    if (input->equation == AdvDiff)
    {
      double An = 0.;
      double A[nDims];

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        An += input->AdvDiff_A(dim) * norm(fpt, dim);
        A[dim] = input->AdvDiff_A(dim);
      }

      eig = std::abs(An);
      waveSp(fpt) = std::abs(An - Vgn);

      compute_Fconv_AdvDiff<nVars, nDims>(UL, FL, A);
      compute_Fconv_AdvDiff<nVars, nDims>(UR, FR, A);
    }
    else if (input->equation == EulerNS)
    {
      double PL, PR;

      compute_Fconv_EulerNS<nVars, nDims>(UL, FL, PL, input->gamma);
      compute_Fconv_EulerNS<nVars, nDims>(UR, FR, PR, input->gamma);

      /* Store pressures for force computation */
      P(fpt, 0) = PL;
      P(fpt, 1) = PR;

      /* Compute speed of sound */
      double aL = std::sqrt(input->gamma * PL / UL[0]);
      double aR = std::sqrt(input->gamma * PR / UR[0]);

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += UL[dim+1]/UL[0] * norm(fpt, dim);
        VnR += UR[dim+1]/UR[0] * norm(fpt, dim);
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
        FnL[n] += FL[n][dim] * norm(fpt, dim);
        FnR[n] += FR[n][dim] * norm(fpt, dim);
      }
    }

    /* Compute common normal flux */
    if (rus_bias(fpt) == 0) /* Upwinded */
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double F = (0.5 * (FnR[n]+FnL[n]) - 0.5 * eig * (1.0-input->rus_k) * (UR[n]-UL[n]));

        /* Correct for positive parent space sign convention */
        Fcomm(fpt, n, 0) = F * dA(fpt, 0);
        Fcomm(fpt, n, 1) = -F * dA(fpt, 1);
      }
    }
    else if (rus_bias(fpt) == 2) /* Central */
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        double F = 0.5 * (FnL[n] + FnR[n]);
        Fcomm(fpt, n, 0) = F * dA(fpt, 0);
        Fcomm(fpt, n, 1) = -F * dA(fpt, 1);
      }
    }
    else if (rus_bias(fpt) == 1)
    {
      for (unsigned int n = 0; n < nVars; n++) /* Set flux state */
      {
        double F = FnR[n];
        Fcomm(fpt, n, 0) = F * dA(fpt, 0);
        Fcomm(fpt, n, 1) = -F * dA(fpt, 1);
      }
    }
  }
}

void Faces::compute_common_U(unsigned int startFpt, unsigned int endFpt)
{
  
  /* Compute common solution */
  if (input->fvisc_type == LDG)
  {
#ifdef _CPU
#pragma omp parallel for 
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

      double beta = input->ldg_b;


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
      /* If interior, allow use of beta factor */
      if (LDG_bias(fpt) == 0)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          double UL = U_ldg(fpt, n, 0); double UR = U_ldg(fpt, n, 1);

           Ucomm(fpt, n, 0) = 0.5*(UL + UR) - beta*(UL - UR);
           Ucomm(fpt, n, 1) = 0.5*(UL + UR) - beta*(UL - UR);
        }
      }
      /* If on (non-periodic) boundary, set right state as common (strong) */
      else
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          double UR = U_ldg(fpt, n, 1);

          Ucomm(fpt, n, 0) = UR;
          Ucomm(fpt, n, 1) = UR;
        }
      }

    }
#endif

#ifdef _GPU
    compute_common_U_LDG_wrapper(U_ldg_d, Ucomm_d, norm_d, input->ldg_b, nFpts, nVars, nDims, input->equation, 
        LDG_bias_d, startFpt, endFpt, input->overset, geo->iblank_fpts_d.data());

    check_error();

#endif
  }

  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }
}

void Faces::common_U_to_F(unsigned int startFpt, unsigned int endFpt, unsigned int dim)
{
#ifdef _CPU
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      double F = Ucomm(fpt, var, 0) * norm(fpt, dim);
      Fcomm(fpt, var, 0) = F * dA(fpt, 0);
      Fcomm(fpt, var, 1) = -F * dA(fpt, 1);
    }
  }
#endif

#ifdef _GPU
  common_U_to_F_wrapper(Fcomm_d, Ucomm_d, norm_d, dA_d, nFpts, nVars, nDims, input->equation, startFpt,
      endFpt, dim);
#endif
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Faces::LDG_flux(unsigned int startFpt, unsigned int endFpt)
{
   
  double tau = input->ldg_tau;

  double UL[nVars];
  double UR[nVars];
  double dUL[nVars][nDims];
  double dUR[nVars][nDims];

#pragma omp parallel for firstprivate(UL, UR, dUL, dUR)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    double beta = input->ldg_b;

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
      UL[n] = U_ldg(fpt, n, 0); UR[n] = U_ldg(fpt, n, 1);

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        dUL[n][dim] = dU(fpt, n, dim, 0);
        dUR[n][dim] = dU(fpt, n, dim, 1);
      }
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
        FnL[n] += FL[n][dim] * norm(fpt, dim);
        FnR[n] += FR[n][dim] * norm(fpt, dim);
      }
    }

    /* Compute common normal viscous flux and accumulate */
    /* If interior, use central */
    if (LDG_bias(fpt) == 0)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          FR[n][dim] = 0.5*(FL[n][dim] + FR[n][dim]) + tau * norm(fpt, dim)* (UL[n]
              - UR[n]) + beta * norm(fpt, dim)* (FnL[n] - FnR[n]);
        }
      }
    }
    /* If Neumann boundary, use right state only */
    else
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          FR[n][dim] += tau * norm(fpt, dim)* (UL[n] - UR[n]);
        }
      }
    }

    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        double F = FR[n][dim] * norm(fpt, dim);
        Fcomm(fpt, n, 0) += F * dA(fpt, 0);
        Fcomm(fpt, n, 1) -= F * dA(fpt, 1);
      }
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
    input->rt, input->c_sth, input->fix_vis, input->ldg_b, input->ldg_tau, nFpts, nVars, nDims, input->equation, 
    input->fconv_type, input->fvisc_type, startFpt, endFpt, input->viscous, input->motion, input->overset, geo->iblank_fpts_d.data());
#endif

}

void Faces::compute_dFcdU(unsigned int startFpt, unsigned int endFpt)
{
  if (input->fconv_type == Rusanov)
  {
#ifdef _CPU
    rusanov_dFcdU(startFpt, endFpt);
#endif

#ifdef _GPU
    rusanov_dFcdU_wrapper(U_d, dFdUconv_d, dFcdU_d, P_d, norm_d, waveSp_d, LDG_bias_d,
        input->gamma, input->rus_k, nFpts, nVars, nDims, input->equation, startFpt, endFpt);
    check_error();
#endif

  }
  else
  {
    ThrowException("Numerical convective flux type not recognized!");
  }

  if (input->viscous)
  {
    if (input->fvisc_type == LDG)
    {
#ifdef _CPU
      LDG_dFcdU(startFpt, endFpt);
#endif

#ifdef _GPU
      ThrowException("LDG flux for implicit method not implemented on GPU!");
#endif

    }
    else
    {
      ThrowException("Numerical viscous flux type not recognized!");
    }
  }
}

void Faces::compute_dUcdU(unsigned int startFpt, unsigned int endFpt)
{
  if (input->fvisc_type == LDG)
  {
    for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
    {
      if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

      double beta = input->ldg_b;

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

      /* If interior, allow use of beta factor */
      if (LDG_bias(fpt) == 0)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            if (ni == nj)
            {
              dUcdU(fpt, ni, nj, 0) = 0.5 - beta;
              dUcdU(fpt, ni, nj, 1) = 0.5 + beta;
            }
            else
            {
              dUcdU(fpt, ni, nj, 0) = 0;
              dUcdU(fpt, ni, nj, 1) = 0;
            }
          }
        }
      }
      /* If on (non-periodic) boundary, don't use beta (this is from HiFILES. Need to check) */
      /* If on (non-periodic) boundary, set right state as common (strong) */
      else
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            if (ni == nj)
            {
              dUcdU(fpt, ni, nj, 0) = 0;
              dUcdU(fpt, ni, nj, 1) = 1;
            }
            else
            {
              dUcdU(fpt, ni, nj, 0) = 0;
              dUcdU(fpt, ni, nj, 1) = 0;
            }
          }
        }
      }
    }
  }
  else
  {
    ThrowException("Numerical viscous flux type not recognized!");
  }
}

void Faces::rusanov_dFcdU(unsigned int startFpt, unsigned int endFpt)
{
  std::vector<double> WL(nVars);
  std::vector<double> WR(nVars);

  dFndUL_temp.fill(0);
  dFndUR_temp.fill(0);

#pragma omp parallel for firstprivate(WL, WR)
  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    /* Apply central flux at boundaries */
    double k = input->rus_k;

    /* Get interface-normal dFdU components  (from L to R)*/
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFndUL_temp(fpt, ni, nj) += dFdUconv(fpt, ni, nj, dim, 0) * norm(fpt, dim);
          dFndUR_temp(fpt, ni, nj) += dFdUconv(fpt, ni, nj, dim, 1) * norm(fpt, dim);
        }
      }
    }

    if (LDG_bias(fpt) == 2)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFcdU(fpt, ni, nj, 0, 0) = 0.5 * dFndUL_temp(fpt, ni, nj);
          dFcdU(fpt, ni, nj, 1, 0) = 0.5 * dFndUR_temp(fpt, ni, nj);

          dFcdU(fpt, ni, nj, 0, 1) = 0.5 * dFndUL_temp(fpt, ni, nj);
          dFcdU(fpt, ni, nj, 1, 1) = 0.5 * dFndUR_temp(fpt, ni, nj);
        }
      }
      continue;
    }
    else if (LDG_bias(fpt) != 0)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFcdU(fpt, ni, nj, 0, 0) = 0;
          dFcdU(fpt, ni, nj, 1, 0) = dFndUR_temp(fpt, ni, nj);

          dFcdU(fpt, ni, nj, 0, 1) = 0;
          dFcdU(fpt, ni, nj, 1, 1) = dFndUR_temp(fpt, ni, nj);
        }
      }
      continue;
    }
   
    /* Get left and right state variables */
    for (unsigned int n = 0; n < nVars; n++)
    {
      WL[n] = U(fpt, n, 0); WR[n] = U(fpt, n, 1);
    }

    /* Get numerical wavespeed */
    // TODO: This can be removed when NK implemented on CPU
    std::vector<double> dwSdUL(nVars, 0);
    std::vector<double> dwSdUR(nVars, 0);
    if (input->equation == AdvDiff)
    {
      double An = 0.;

      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        An += input->AdvDiff_A(dim) * norm(fpt, dim);
      }

      waveSp(fpt) = std::abs(An);
    }
    else if (input->equation == EulerNS)
    {
      // TODO: May be able to remove once pressure is stored
      for (unsigned int slot = 0; slot < 2; slot ++)
      {
        double momF = 0.0;
        for (unsigned int dim = 0; dim < nDims; dim ++)
        {
          momF += U(fpt, dim + 1, slot) * U(fpt, dim + 1, slot);
        }

        momF /= U(fpt, 0, slot);

        P(fpt, slot) = (input->gamma - 1.0) * (U(fpt, nDims + 1, slot) - 0.5 * momF);
      }

      /* Compute speed of sound */
      double aL = std::sqrt(std::abs(input->gamma * P(fpt, 0) / WL[0]));
      double aR = std::sqrt(std::abs(input->gamma * P(fpt, 1) / WR[0]));

      /* Compute normal velocities */
      double VnL = 0.0; double VnR = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        VnL += WL[dim+1]/WL[0] * norm(fpt, dim);
        VnR += WR[dim+1]/WR[0] * norm(fpt, dim);
      }

      /* Compute wavespeed and wavespeed derivative */
      double gam = input->gamma;
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
        waveSp(fpt) = wSL;

        /* Compute wavespeed derivative */
        dwSdUL[0] = -sgn*VnL/rho - aL/(2.0*rho);
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          dwSdUL[0] += gam * (gam-1.0) * V[dim]*V[dim] / (4.0*aL*rho);
          dwSdUL[dim+1] = sgn*norm(fpt, dim)/rho - gam * (gam-1.0) * V[dim] / (2.0*aL*rho);
        }
        dwSdUL[nDims+1] = gam * (gam-1.0) / (2.0*aL*rho);
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
        waveSp(fpt) = wSR;

        /* Compute wavespeed derivative */
        dwSdUR[0] = -sgn*VnR/rho - aR/(2.0*rho);
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          dwSdUR[0] += gam * (gam-1.0) * V[dim]*V[dim] / (4.0*aR*rho);
          dwSdUR[dim+1] = sgn*norm(fpt, dim)/rho - gam * (gam-1.0) * V[dim] / (2.0*aR*rho);
        }
        dwSdUR[nDims+1] = gam * (gam-1.0) / (2.0*aR*rho);
      }
    }

    /* Compute common dFdU */
    for (unsigned int nj = 0; nj < nVars; nj++)
    {
      for (unsigned int ni = 0; ni < nVars; ni++)
      {
        if (ni == nj)
        {
          dFcdU(fpt, ni, nj, 0, 0) = 0.5 * (dFndUL_temp(fpt, ni, nj) - ((WR[ni]-WL[ni]) * dwSdUL[nj] - waveSp(fpt)) * (1.0-k));
          dFcdU(fpt, ni, nj, 1, 0) = 0.5 * (dFndUR_temp(fpt, ni, nj) - ((WR[ni]-WL[ni]) * dwSdUR[nj] + waveSp(fpt)) * (1.0-k));

          dFcdU(fpt, ni, nj, 0, 1) = 0.5 * (dFndUL_temp(fpt, ni, nj) - ((WR[ni]-WL[ni]) * dwSdUL[nj] - waveSp(fpt)) * (1.0-k));
          dFcdU(fpt, ni, nj, 1, 1) = 0.5 * (dFndUR_temp(fpt, ni, nj) - ((WR[ni]-WL[ni]) * dwSdUR[nj] + waveSp(fpt)) * (1.0-k));
        }
        else
        {
          dFcdU(fpt, ni, nj, 0, 0) = 0.5 * (dFndUL_temp(fpt, ni, nj) - (WR[ni]-WL[ni]) * dwSdUL[nj] * (1.0-k));
          dFcdU(fpt, ni, nj, 1, 0) = 0.5 * (dFndUR_temp(fpt, ni, nj) - (WR[ni]-WL[ni]) * dwSdUR[nj] * (1.0-k));

          dFcdU(fpt, ni, nj, 0, 1) = 0.5 * (dFndUL_temp(fpt, ni, nj) - (WR[ni]-WL[ni]) * dwSdUL[nj] * (1.0-k));
          dFcdU(fpt, ni, nj, 1, 1) = 0.5 * (dFndUR_temp(fpt, ni, nj) - (WR[ni]-WL[ni]) * dwSdUR[nj] * (1.0-k));
        }
      }
    }
  }
}

void Faces::LDG_dFcdU(unsigned int startFpt, unsigned int endFpt)
{
  double tau = input->ldg_tau;

  dFndUL_temp.fill(0);
  dFndUR_temp.fill(0);
  dFnddUL_temp.fill(0);
  dFnddUR_temp.fill(0);

  dFcdU_temp.fill(0.0);
  dFcddU_temp.fill(0.0);

  for (unsigned int fpt = startFpt; fpt < endFpt; fpt++)
  {
    if (input->overset && geo->iblank_face[geo->fpt2face[fpt]] == HOLE) continue;

    /* Setting sign of beta (from HiFiLES) */
    double beta = input->ldg_b;
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

    /* Get numerical diffusion coefficient */
    // TODO: This can be removed when NK implemented on CPU
    if (input->equation == AdvDiff)
    {
      diffCo(fpt) = input->AdvDiff_D;
    }
    else if (input->equation == EulerNS)
    {
      // TODO: Add or store mu from Sutherland's law
      double diffCoL = std::max(input->mu / U(fpt, 0, 0), input->gamma * input->mu / (input->prandtl * U(fpt, 0, 0)));
      double diffCoR = std::max(input->mu / U(fpt, 0, 1), input->gamma * input->mu / (input->prandtl * U(fpt, 0, 1)));
      diffCo(fpt) = std::max(diffCoL, diffCoR);
    }

    /* Get interface-normal dFdU components (from L to R) */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFndUL_temp(fpt, ni, nj) += dFdUvisc(fpt, ni, nj, dim, 0) * norm(fpt, dim);
          dFndUR_temp(fpt, ni, nj) += dFdUvisc(fpt, ni, nj, dim, 1) * norm(fpt, dim);
        }
      }
    }

    /* Get interface-normal dFddU components (from L to R) */
    for (unsigned int dimj = 0; dimj < nDims; dimj++)
    {
      for (unsigned int dimi = 0; dimi < nDims; dimi++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dFnddUL_temp(fpt, ni, nj, dimj) += dFddUvisc(fpt, ni, nj, dimi, dimj, 0) * norm(fpt, dimi);
            dFnddUR_temp(fpt, ni, nj, dimj) += dFddUvisc(fpt, ni, nj, dimi, dimj, 1) * norm(fpt, dimi);
          }
        }
      }
    }

    /* Compute common interface values */
    /* If interior, use central */
    if (LDG_bias(fpt) == 0)
    {
      /* Compute common dFdU */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dFcdU_temp(fpt, ni, nj, 0) += (0.5 * dFdUvisc(fpt, ni, nj, dim, 0) + beta * norm(fpt, dim) * dFndUL_temp(fpt, ni, nj)) * norm(fpt, dim);
            dFcdU_temp(fpt, ni, nj, 1) += (0.5 * dFdUvisc(fpt, ni, nj, dim, 1) - beta * norm(fpt, dim) * dFndUR_temp(fpt, ni, nj)) * norm(fpt, dim);

            if (ni == nj)
            {
              dFcdU_temp(fpt, ni, nj, 0) += (tau * norm(fpt, dim)) * norm(fpt, dim);
              dFcdU_temp(fpt, ni, nj, 1) -= (tau * norm(fpt, dim)) * norm(fpt, dim);
            }
          }
        }
      }

      /* Compute common dFddU */
      for (unsigned int dimj = 0; dimj < nDims; dimj++)
      {
        for (unsigned int dimi = 0; dimi < nDims; dimi++)
        {
          for (unsigned int nj = 0; nj < nVars; nj++)
          {
            for (unsigned int ni = 0; ni < nVars; ni++)
            {
              dFcddU_temp(fpt, ni, nj, dimj, 0) += 
                (0.5 * dFddUvisc(fpt, ni, nj, dimi, dimj, 0) + beta * norm(fpt, dimi) * dFnddUL_temp(fpt, ni, nj, dimj)) * norm(fpt, dimi);
              dFcddU_temp(fpt, ni, nj, dimj, 1) += 
                (0.5 * dFddUvisc(fpt, ni, nj, dimi, dimj, 1) - beta * norm(fpt, dimi) * dFnddUR_temp(fpt, ni, nj, dimj)) * norm(fpt, dimi);
            }
          }
        }
      }
    }
    /* If Neumann boundary, use right state only */
    else
    {
      /* Compute common dFdU */
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dFcdU_temp(fpt, ni, nj, 1) += dFdUvisc(fpt, ni, nj, dim, 1) * norm(fpt, dim);

            if (ni == nj)
            {
              dFcdU_temp(fpt, ni, nj, 0) += (tau * norm(fpt, dim)) * norm(fpt, dim);
              dFcdU_temp(fpt, ni, nj, 1) -= (tau * norm(fpt, dim)) * norm(fpt, dim);
            }
          }
        }
      }

      /* Compute common dFddU */
      for (unsigned int dimj = 0; dimj < nDims; dimj++)
      {
        for (unsigned int dimi = 0; dimi < nDims; dimi++)
        {
          for (unsigned int nj = 0; nj < nVars; nj++)
          {
            for (unsigned int ni = 0; ni < nVars; ni++)
            {
              dFcddU_temp(fpt, ni, nj, dimj, 1) += dFddUvisc(fpt, ni, nj, dimi, dimj, 1) * norm(fpt, dimi);
            }
          }
        }
      }
    }

    /* Correct for positive parent space sign convention, dFcdU */
    for (unsigned int slot = 0; slot < 2; slot++)
    {
      for (unsigned int nj = 0; nj < nVars; nj++)
      {
        for (unsigned int ni = 0; ni < nVars; ni++)
        {
          dFcdU(fpt, ni, nj, slot, 0) += dFcdU_temp(fpt, ni, nj, slot);
          dFcdU(fpt, ni, nj, slot, 1) += dFcdU_temp(fpt, ni, nj, slot);
        }
      }
    }

    /* Correct for positive parent space sign convention, dFcddU */
    for (unsigned int slot = 0; slot < 2; slot++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            dFcddU(fpt, ni, nj, dim, slot, 0) = dFcddU_temp(fpt, ni, nj, dim, slot);
            dFcddU(fpt, ni, nj, dim, slot, 1) = dFcddU_temp(fpt, ni, nj, dim, slot);
          }
        }
      }
    }
  }
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
        U_sbuffs[sendRank](i, n) = U(fpts(i), n, 0);
      }
    }

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }
#endif

#ifdef _GPU
  /* Wait for extrapolate_U to complete in stream 0 */
  stream_wait_event(1, 0);
  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    pack_U_wrapper(U_sbuffs_d[sendRank], fpts, U_d, nVars, 1);
  }

  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    copy_from_device(U_sbuffs[pairedRank].data(), U_sbuffs_d[pairedRank].data(), U_sbuffs[pairedRank].max_size(), 1);
  }

  check_error();
#endif
}

void Faces::recv_U_data()
{
#ifdef _GPU
  int ridx = 0;
  /* Stage non-blocking receives */
  for (const auto &entry : geo->fpt_buffer_map)
  {
    int recvRank = entry.first;
    const auto &fpts = entry.second;

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
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
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }
#endif

  /* Wait for comms to finish */
  input->waitTimer.startTimer();
  PUSH_NVTX_RANGE("MPI", 0)
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);
  POP_NVTX_RANGE
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
        if (input->overset && geo->iblank_face[geo->fpt2face[fpts(i)]] != NORMAL)
          continue;

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
    copy_to_device(U_rbuffs_d[pairedRank].data(), U_rbuffs[pairedRank].data(), U_rbuffs_d[pairedRank].max_size(), 1);
  }

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;
    unpack_U_wrapper(U_rbuffs_d[recvRank], fpts, U_d, nVars, 1, input->overset, geo->iblank_fpts_d.data());
  }

  /* Halt main compute stream until U is unpacked */
  event_record(1, 1);
  stream_wait_event(0, 1);

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

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
    ridx++;
  }

  unsigned int sidx = 0;
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
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }
#endif

#ifdef _GPU
  /* Wait for extrapolate_dU to complete in stream 0 */
  stream_wait_event(1, 0);

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;
    
    /* Pack buffer of solution data at flux points in list */
    pack_dU_wrapper(U_sbuffs_d[sendRank], fpts, dU_d, nVars, nDims, 1);
  }

  /* Copy buffer to host (TODO: Use cuda aware MPI for direct transfer) */
  for (auto &entry : geo->fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    copy_from_device(U_sbuffs[pairedRank].data(), U_sbuffs_d[pairedRank].data(), U_sbuffs[pairedRank].max_size(), 1);
  }

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

    MPI_Irecv(U_rbuffs[recvRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, recvRank, 0, myComm, &rreqs[ridx]);
    ridx++;
  }
  
  sync_stream(1);

  int sidx = 0;
  for (auto &entry : geo->fpt_buffer_map)
  {
    int sendRank = entry.first;
    auto &fpts = entry.second;

    /* Send buffer to paired rank */
    MPI_Isend(U_sbuffs[sendRank].data(), (unsigned int) fpts.size() * nVars * nDims, MPI_DOUBLE, sendRank, 0, myComm, &sreqs[sidx]);
    sidx++;
  }

#endif

  PUSH_NVTX_RANGE("MPI", 0)
  /* Wait for comms to finish */
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);
  POP_NVTX_RANGE

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
          if (input->overset && geo->iblank_face[geo->fpt2face[fpts(i)]] != NORMAL)
            continue;
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
    copy_to_device(U_rbuffs_d[pairedRank].data(), U_rbuffs[pairedRank].data(), U_rbuffs_d[pairedRank].max_size(), 1);
  }

  for (auto &entry : geo->fpt_buffer_map_d)
  {
    int recvRank = entry.first;
    auto &fpts = entry.second;

    unpack_dU_wrapper(U_rbuffs_d[recvRank], fpts, dU_d, nVars, nDims, 1, input->overset, geo->iblank_fpts_d.data());
  }

  /* Halt main stream until dU is unpacked */
  event_record(1, 1);
  stream_wait_event(0, 1);

  check_error();
#endif
}
#endif

void Faces::get_U_index(int faceID, int fpt, int& ind, int& stride)
{
  /* U : nFpts x nVars x 2 */
  int i = geo->face2fpts(fpt, faceID);
  int ic1 = geo->face2eles(0,faceID);
  int ic2 = geo->face2eles(1,faceID);

  int side = -1;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;
  else if (ic2 >= 0 && geo->iblank_cell(ic2) == NORMAL)
    side = 0;
  else
  {
    printf("face %d: ibf %d | ic1,2: %d,%d, ibc1,2: %d,%d\n",faceID,geo->iblank_face[faceID],ic1,ic2,geo->iblank_cell(ic1),geo->iblank_cell(ic2));
    ThrowException("Face not blanked but both elements are!");
  }

  ind    = std::distance(&U(0,0,0), &U(i,0,side));
  stride = std::distance(&U(i,0,side), &U(i,1,side));
}

double& Faces::get_u_fpt(int faceID, int fpt, int var)
{
  /* U : nFpts x nVars x 2 */
  int i = geo->face2fpts(fpt, faceID);
  int ic1 = geo->face2eles(0,faceID);
  int ic2 = geo->face2eles(1,faceID);

  unsigned int side = 0;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;
  else if (ic2 < 0)
    ThrowException("get_u_fpt: Invalid face/cell blanking - check your connectivity.");

  return U(i,var,side);
}

double& Faces::get_grad_fpt(int faceID, int fpt, int var, int dim)
{
  /* U : nFpts x nVars x 2 */
  int i = geo->face2fpts(fpt, faceID);
  int ic1 = geo->face2eles(0,faceID);
  int ic2 = geo->face2eles(1,faceID);

  unsigned int side = 0;
  if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
    side = 1;

  return dU(i,var,dim,side);
}

#ifdef _GPU
void Faces::fringe_u_to_device(int* fringeIDs, int nFringe)
{
  if (nFringe == 0) return;

  U_fringe.assign({geo->nFptsPerFace, nFringe, nVars});
  fringe_fpts.assign({geo->nFptsPerFace, nFringe}, 0);
  fringe_side.assign({geo->nFptsPerFace, nFringe}, 0);
  for (int face = 0; face < nFringe; face++)
  {
    unsigned int side = 0;
    int ic1 = geo->face2eles(0,fringeIDs[face]);
    if (ic1 >= 0 && geo->iblank_cell(ic1) == NORMAL)
      side = 1;

    for (unsigned int fpt = 0; fpt < geo->nFptsPerFace; fpt++)
    {
      fringe_fpts(fpt,face) = geo->face2fpts(fpt, fringeIDs[face]);
      fringe_side(fpt,face) = side;
    }
  }

  for (unsigned int var = 0; var < nVars; var++)
  {
    for (unsigned int face = 0; face < nFringe; face++)
    {
      for (unsigned int fpt = 0; fpt < geo->nFptsPerFace; fpt++)
      {
        unsigned int gfpt = fringe_fpts(fpt,face);
        unsigned int side = fringe_side(fpt,face);
        U_fringe(fpt,face,var) = U(gfpt,var,side);
      }
    }
  }

  U_fringe_d = U_fringe;
  fringe_fpts_d = fringe_fpts;
  fringe_side_d = fringe_side;

  unpack_fringe_u_wrapper(U_fringe_d,U_d,fringe_fpts_d,fringe_side_d,nFringe,
      geo->nFptsPerFace,nVars);

  check_error();
}

void Faces::fringe_grad_to_device(int nFringe)
{
  /* NOTE: Expecting that fringe_u_to_device has already been called for the
   * same set of fringe faces */

  if (nFringe == 0) return;

  dU_fringe.assign({geo->nFptsPerFace, nFringe, nVars, nDims});

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
          dU_fringe(fpt,face,var,dim) = dU(gfpt,var,dim,side);
        }
      }
    }
  }

  dU_fringe_d = dU_fringe;

  unpack_fringe_grad_wrapper(dU_fringe_d,dU_d,fringe_fpts_d,fringe_side_d,
      nFringe,geo->nFptsPerFace,nVars,nDims);

  check_error();
}
#endif
