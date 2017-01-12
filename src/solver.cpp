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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <vector>

extern "C" {
#include "cblas.h"
}

#include "elements.hpp"
#include "faces.hpp"
#include "funcs.hpp"
#include "geometry.hpp"
#include "hexas.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _BUILD_LIB
//#include "tiogaInterface.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#endif

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
#endif

#ifndef _NO_HDF5
#include "H5Cpp.h"
#ifndef _H5_NO_NAMESPACE
using namespace H5;
#endif
#ifdef _MPI
  #ifdef H5_HAVE_PARALLEL
//    #define _USE_H5_PARALLEL
  #endif
#endif
#endif

FRSolver::FRSolver(InputStruct *input, int order)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;

}

void FRSolver::setup(_mpi_comm comm_in)
{
  myComm = comm_in;
#ifdef _MPI
  worldComm = MPI_COMM_WORLD;
#endif

  if (input->rank == 0) std::cout << "Reading mesh: " << input->meshfile << std::endl;
  geo = process_mesh(input, order, input->nDims, myComm);

  if (input->rank == 0) std::cout << "Setting up elements and faces..." << std::endl;

  if (input->nDims == 2)
    eles = std::make_shared<Quads>(&geo, input, order);
  else if (input->nDims == 3)
    eles = std::make_shared<Hexas>(&geo, input, order);

  faces = std::make_shared<Faces>(&geo, input, myComm);

  faces->setup(eles->nDims, eles->nVars);

  eles->setup(faces, myComm);

  if (input->rank == 0) std::cout << "Setting up timestepping..." << std::endl;
  setup_update();  

  if (input->rank == 0) std::cout << "Setting up output..." << std::endl;
  setup_output();

  if (input->rank == 0) std::cout << "Initializing solution..." << std::endl;
  initialize_U();


  if (input->restart)
  {
    if (input->restart_type == 0)
    {
      if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_file + " ..." << std::endl;

      // Backwards compatibility: full filename given
      if (input->restart_file.find(".vtu")  != std::string::npos or
          input->restart_file.find(".pvtu") != std::string::npos)
      {
        restart(input->restart_file);
      }
      else if (input->restart_file.find(".pyfr") != std::string::npos)
      {
        restart_pyfr(input->restart_file);
      }
      else
        ThrowException("Unknown file type for restart file.");
    }
    else
    {
      if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_case + "_" + std::to_string(input->restart_iter) + " ..." << std::endl;

      // New version: Use case name + iteration number to find file
      // [Overset compatible]
      if (input->restart_type == 1) // ParaView
        restart(input->restart_case, input->restart_iter);
      else if (input->restart_type == 2) // PyFR
        restart_pyfr(input->restart_case, input->restart_iter);
    }
  }

  if (input->filt_on)
  {
    if (input->rank == 0) std::cout << "Setting up filter..." << std::endl;
    filt.setup(input, *this);
  }

#ifdef _GPU
  if (input->rank == 0) std::cout << "Setting up data on GPU(s)..." << std::endl;
  solver_data_to_device();
#endif

  setup_views(); // Note: This function allocates addtional GPU memory for views

#ifdef _GPU
  report_gpu_mem_usage();
#endif

}

void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  if (input->dt_scheme == "Euler")
  {
    nStages = 1;
    rk_beta.assign({nStages}, 1.0);

  }
  else if (input->dt_scheme == "RK44")
  {
    nStages = 4;
    
    rk_alpha.assign({nStages-1});
    rk_alpha(0) = 0.5; rk_alpha(1) = 0.5; rk_alpha(2) = 1.0;

    rk_beta.assign({nStages});
    rk_beta(0) = 1./6.; rk_beta(1) = 1./3.; 
    rk_beta(2) = 1./3.; rk_beta(3) = 1./6.;
  }
  else if (input->dt_scheme == "RKj")
  {
    nStages = 4;
    rk_alpha.assign({nStages});
    /* Standard RK44 */
    //rk_alpha(0) = 1./4; rk_alpha(1) = 1./3.; 
    //rk_alpha(2) = 1./2.; rk_alpha(3) = 1.0;
    /* OptRK4 (r = 0.5) */
    rk_alpha(0) = 0.153; rk_alpha(1) = 0.442; 
    rk_alpha(2) = 0.930; rk_alpha(3) = 1.0;
  }
  else if (input->dt_scheme == "LSRK")
  {
    nStages = 5;
    rk_alpha.assign({nStages - 1});
    rk_alpha(0) =   970286171893. / 4311952581923.;
    rk_alpha(1) =  6584761158862. / 12103376702013.;
    rk_alpha(2) =  2251764453980. / 15575788980749.;
    rk_alpha(3) = 26877169314380. / 34165994151039.;

    rk_beta.assign({nStages});
    rk_beta(0) =  1153189308089. / 22510343858157.;
    rk_beta(1) =  1772645290293. / 4653164025191.;
    rk_beta(2) = -1672844663538. / 4480602732383.;
    rk_beta(3) =  2114624349019. / 3568978502595.;
    rk_beta(4) =  5198255086312. / 14908931495163.;

    rk_bhat.assign({nStages});
    rk_bhat(0) =  1016888040809. / 7410784769900.;
    rk_bhat(1) = 11231460423587. / 58533540763752.;
    rk_bhat(2) = -1563879915014. / 6823010717585.;
    rk_bhat(3) =   606302364029. / 971179775848.;
    rk_bhat(4) =  1097981568119. / 3980877426909.;

    rk_c.assign({nStages});
    for (int i = 1; i < nStages; i++)
    {
      rk_c(i) = rk_alpha(i-1);

      for (int j = 0; j < i-1; j++)
        rk_c(i) += rk_beta(j);
    }

    expa = input->pi_alpha / 4.;
    expb = input->pi_beta / 4.;
    prev_err = 1.;

    U_til.assign({eles->nSpts, eles->nEles, eles->nVars});
    rk_err.assign({eles->nSpts, eles->nEles, eles->nVars});
  }
  else if (input->dt_scheme == "MCGS")
  {
#ifdef _GPU
    if (input->viscous)
    {
      ThrowException("Viscous MCGS not implemented on GPU");
    }
#endif

    // HACK: (nStages = 1) doesn't work, fix later
    nStages = 2;
    rk_alpha.assign({nStages});
    rk_beta.assign({nStages});

    /* Forward or Forward/Backward sweep */
    nCounter = geo.nColors;
    if (input->backsweep)
    {
      nCounter *= 2;
    }
  }
  else
  {
    ThrowException("dt_scheme not recognized!");
  }

  U_ini.assign({eles->nSpts, eles->nEles, eles->nVars});
  dt.assign({eles->nEles},input->dt);
}

void FRSolver::setup_output()
{
  /* Create output directory to store data files */
  if (input->rank == 0)
  {
    std::string cmd = "mkdir -p " + input->output_prefix;
    system(cmd.c_str());
  }

#ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
       
  eles->setup_ppt_connectivity();
}

void FRSolver::restart(std::string restart_file, unsigned restart_iter)
{
  if (input->restart_type > 0) // append .pvtu / .vtu to case name
  {
    std::stringstream ss;

    ss << restart_file;

    if (input->overset)
    {
      ss << "_Grid" << input->gridID;
    }

    ss << "_" << std::setw(9) << std::setfill('0') << restart_iter;

#ifdef _MPI
    ss << ".pvtu";
#else
    ss << ".vtu";
#endif

    restart_file = ss.str();
  }

  size_t pos;
#ifdef _MPI
  /* From .pvtu, form partition specific filename */
  pos = restart_file.rfind(".pvtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .pvtu file for parallel restart!");
  }

  restart_file = restart_file.substr(0, pos);

  std::stringstream ss;
  ss << std::setw(3) << std::setfill('0') << input->rank;

  restart_file += "_" + ss.str() + ".vtu";
#endif

  /* Open .vtu file */
  std::ifstream f(restart_file);
  pos = restart_file.rfind(".vtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .vtu file for restart!");
  }

  if (!f.is_open())
  {
    ThrowException("Could not open restart file " + restart_file + "!");
  }

  std::string param, line;
  double val;
  unsigned int order_restart;
  mdvector<double> U_restart;

  unsigned int nEles = eles->nEles;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (int ele = 0; ele < eles->nEles; ele++)
      if (geo.iblank_cell(ele) != NORMAL) nEles--;
  }

  /* Load data from restart file */
  while (f >> param)
  {
    if (param == "TIME")
    {
      f >> flow_time;
    }
    if (param == "ITER")
    {
      f >> current_iter;
      restart_iter = current_iter;
    }
    if (param == "ORDER")
    {
      f >> order_restart;
    }

    if (param == "<AppendedData")
    {
      std::getline(f,line);
      f.ignore(1); 

      /* Setup extrapolation operator from equistant restart points */
      eles->set_oppRestart(order_restart, true);

      unsigned int nRpts = eles->oppRestart.get_dim(1);

      U_restart.assign({nRpts, eles->nEles, eles->nVars});

      unsigned int temp; 
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        binary_read(f, temp);

        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          /// TODO: make sure this is setup correctly first
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          for (unsigned int rpt = 0; rpt < nRpts; rpt++)
          {
            binary_read(f, U_restart(rpt, ele, n));
          }
        }
      }

      /* Extrapolate values from restart points to solution points */
      auto &A = eles->oppRestart(0, 0);
      auto &B = U_restart(0, 0, 0);
      auto &C = eles->U_spts(0, 0, 0);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nRpts, 1.0, &A, eles->oppRestart.ldim(), &B, 
          U_restart.ldim(), 0.0, &C, eles->U_spts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nRpts, 1.0, &A, eles->oppRestart.ldim(), &B, 
          U_restart.ldim(), 0.0, &C, eles->U_spts.ldim());
#endif

    }
  }

  f.close();
}

#ifdef _GPU
void FRSolver::solver_data_to_device()
{
  /* Initial copy of data to GPU. Assignment operator will allocate data on device when first
   * used. */

  /* FR operators */
  eles->oppE_d = eles->oppE;
  eles->oppD_d = eles->oppD;
  eles->oppD0_d = eles->oppD0;
  eles->oppD_fpts_d = eles->oppD_fpts;
  eles->oppDiv_fpts_d = eles->oppDiv_fpts;

  /* Solver data structures */
  U_ini_d = U_ini;
  rk_alpha_d = rk_alpha;
  rk_beta_d = rk_beta;
  dt_d = dt;

  if (input->dt_scheme == "LSRK")
  {
    U_til_d = U_til;
    rk_err_d = rk_err;
  }

  /* Implicit solver data structures */
  if (input->dt_scheme == "MCGS")
  {
    eles->deltaU_d = eles->deltaU;
    eles->RHS_d = eles->RHS;
    eles->LHS_d = eles->LHSs[0];

    if (input->inv_mode)
    {
      eles->LHSInv_d = eles->LHSInvs[0];
    }

    eles->LU_pivots_d = eles->LU_pivots;
    eles->LU_info_d = eles->LU_info;

    /* For cublas batched LU: Setup and transfer array of GPU pointers to 
     * LHS matrices and RHS vectors */
    unsigned int N = eles->nSpts * eles->nVars;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      eles->RHS_ptrs(ele) = eles->RHS_d.data() + ele * N;

      if (input->inv_mode)
      {
        eles->deltaU_ptrs(ele) = eles->deltaU_d.data() + ele * N;
      }
    }

    if (!input->stream_mode)
    {
      unsigned int nElesMax = ceil(geo.nEles / (double) input->n_LHS_blocks);
      for (unsigned int ele = 0; ele < nElesMax; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

        eles->LHS_ptrs(ele) = eles->LHS_d.data() + ele * (N * N);
      }
        
      if (input->inv_mode)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          eles->LHSInv_ptrs(ele) = eles->LHSInv_d.data() + ele * (N * N);
        }
      }
    }
    else
    {
      unsigned int nElesMax = *std::max_element(geo.ele_color_nEles.begin(), geo.ele_color_nEles.end());
      for (unsigned int ele = 0; ele < nElesMax; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

        eles->LHS_ptrs(ele) = eles->LHS_d.data() + ele * (N * N);
        if (input->inv_mode)
          eles->LHSInv_ptrs(ele) = eles->LHSInv_d.data() + ele * (N * N);
      }
    }

    eles->LHS_ptrs_d = eles->LHS_ptrs;
    eles->RHS_ptrs_d = eles->RHS_ptrs;

    if (input->inv_mode)
    {
      eles->LHSInv_ptrs_d = eles->LHSInv_ptrs;
      eles->deltaU_ptrs_d = eles->deltaU_ptrs;
    }

    /* Implicit flux derivative data structures (element local) */
    eles->dFdU_spts_d = eles->dFdU_spts;
    eles->dFcdU_fpts_d = eles->dFcdU_fpts;

    /* Implicit flux derivative data structures (faces) */
    faces->dFdUconv_d = faces->dFdUconv;
    faces->dFcdU_d = faces->dFcdU;
  }

  /* Solution data structures (element local) */
  eles->U_spts_d = eles->U_spts;
  eles->U_fpts_d = eles->U_fpts;
  eles->Uavg_d = eles->Uavg;
  eles->weights_spts_d = eles->weights_spts;
  eles->Fcomm_d = eles->Fcomm;
  eles->F_spts_d = eles->F_spts;
  eles->divF_spts_d = eles->divF_spts;
  eles->inv_jaco_spts_d = eles->inv_jaco_spts;
  eles->jaco_det_spts_d = eles->jaco_det_spts;
  eles->vol_d = eles->vol;

  if (input->CFL_type == 2)
    eles->h_ref_d = eles->h_ref;

  if (input->viscous)
  {
    eles->dU_spts_d = eles->dU_spts;
    eles->Ucomm_d = eles->Ucomm;
    eles->dU_fpts_d = eles->dU_fpts;
  }
  
  //TODO: Temporary fix. Need to remove usage of jaco_spts_d from all kernels.
  if (input->motion || input->dt_scheme == "MCGS")
  {
    eles->jaco_spts_d = eles->jaco_spts;
  }

  if (input->motion)
  {
    eles->nodes_d = eles->nodes;
    eles->grid_vel_nodes_d = eles->grid_vel_nodes;
    eles->grid_vel_spts_d = eles->grid_vel_spts;
    eles->grid_vel_fpts_d = eles->grid_vel_fpts;
    eles->shape_spts_d = eles->shape_spts;
    eles->shape_fpts_d = eles->shape_fpts;
    eles->dshape_spts_d = eles->dshape_spts;
    eles->dshape_fpts_d = eles->dshape_fpts;
    eles->jaco_fpts_d = eles->jaco_fpts;
    eles->inv_jaco_fpts_d = eles->inv_jaco_fpts;
    eles->tnorm_d = eles->tnorm;
    eles->dUr_spts_d = eles->dUr_spts;
    eles->dF_spts_d = eles->dF_spts;
    eles->dFn_fpts_d = eles->dFn_fpts;
    eles->tempF_fpts_d = eles->tempF_fpts;

    /* Moving-grid parameters for convenience / ease of future additions
     * (add to input.hpp, then also here) */
    motion_vars = new MotionVars[1];

    motion_vars->moveAx = input->moveAx;
    motion_vars->moveAy = input->moveAy;
    motion_vars->moveAz = input->moveAz;
    motion_vars->moveFx = input->moveFx;
    motion_vars->moveFy = input->moveFy;
    motion_vars->moveFz = input->moveFz;

    allocate_device_data(motion_vars_d, 1);
    copy_to_device(motion_vars_d, motion_vars, 1);
  }

  /* Solution data structures (faces) */
  faces->U_bnd_d = faces->U_bnd;
  faces->P_d = faces->P;
  faces->Ucomm_bnd_d = faces->Ucomm_bnd;
  faces->Fcomm_bnd_d = faces->Fcomm_bnd;
  faces->norm_d = faces->norm;
  faces->dA_d = faces->dA;
  faces->waveSp_d = faces->waveSp;
  faces->diffCo_d = faces->diffCo;
  faces->LDG_bias_d = faces->LDG_bias;

  if (input->viscous)
  {
    faces->dU_bnd_d = faces->dU_bnd;
  }

  if (input->motion)
  {
    faces->Vg_d = faces->Vg;
    faces->coord_d = faces->coord;
  }

  /* Additional data */
  /* Geometry */
  geo.fpt2gfpt_d = geo.fpt2gfpt;
  geo.fpt2gfpt_slot_d = geo.fpt2gfpt_slot;
  geo.gfpt2bnd_d = geo.gfpt2bnd;
  geo.per_fpt_list_d = geo.per_fpt_list;
  eles->coord_spts_d = eles->coord_spts;

  if (input->motion)
  {
    geo.ele2nodes_d = geo.ele2nodes;
    geo.coord_nodes_d = geo.coord_nodes;
    geo.coords_init_d = geo.coords_init;
    geo.grid_vel_nodes_d = geo.grid_vel_nodes;
    eles->coord_fpts_d = eles->coord_fpts;
  }

  /* Input parameters */
  input->V_fs_d = input->V_fs;
  input->V_wall_d = input->V_wall;
  input->norm_fs_d = input->norm_fs;
  input->AdvDiff_A_d = input->AdvDiff_A;

#ifdef _MPI
  /* MPI data */
  for (auto &entry : geo.fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    auto &fpts = entry.second;
    geo.fpt_buffer_map_d[pairedRank] = fpts;
    faces->U_sbuffs_d[pairedRank] = faces->U_sbuffs[pairedRank];
    faces->U_rbuffs_d[pairedRank] = faces->U_rbuffs[pairedRank];
  }

#endif
}
#endif

void FRSolver::compute_residual(unsigned int stage, unsigned int color)
{
  unsigned int startEle = 0; unsigned int endEle = eles->nEles;
  unsigned int startFpt = 0; unsigned int endFpt = geo.nGfpts;

#ifdef _MPI
  endFpt = geo.nGfpts_int + geo.nGfpts_bnd;
  unsigned int startFptMpi = endFpt;
#endif

  /* If using coloring, modify range to extrapolate data from previously updated colors */
  if (color && geo.nColors > 1)
  {
    startEle = geo.ele_color_range[prev_color - 1]; endEle = geo.ele_color_range[prev_color];
  }

#ifdef _BUILD_LIB
  if (input->overset)
  {
    ZEFR->overset_interp(faces->nVars, eles->U_spts.data(), faces->U.data(), 0);
  }
#endif

  /* Extrapolate solution to flux points */
  eles->extrapolate_U(startEle, endEle);

  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    eles->compute_Uavg();
    eles->poly_squeeze();
  }

  /* For coloring, modify range to sweep through current color */
  if (color && geo.nColors > 1)
  {
    startEle = geo.ele_color_range[color - 1]; endEle = geo.ele_color_range[color];
  }

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();


  /* If running inviscid, use this scheduling. */
  if(!input->viscous)
  {
    /* Compute flux at solution points */
    eles->compute_F(startEle, endEle);

    /* Transform solution point fluxes from physical to reference space */
    if (input->motion)
    {
      eles->compute_gradF_spts(startEle, endEle);
      eles->compute_dU0(startEle, endEle);
    }

    /* Compute parent space common flux at non-MPI flux points */
    faces->compute_common_F(startFpt, endFpt);

    /* Compute solution point contribution to divergence of flux */
    if (input->motion)
    {
      eles->transform_gradF_spts(stage, startEle, endEle);
    }
    else
      eles->compute_divF_spts(stage, startEle, endEle);

#ifdef _MPI
    /* Receive U data */
    faces->recv_U_data();

    /* Complete computation on remaning flux points. */
    faces->compute_common_F(startFptMpi, geo.nGfpts);
#endif
  }

  /* If running viscous, use this scheduling */
  else
  {
    /* Compute common interface solution and convective flux at non-MPI flux points */
    faces->compute_common_U(startFpt, endFpt);
    
    /* Compute solution point contribution to (corrected) gradient of state variables at solution points */
    eles->compute_dU_spts(startEle, endEle);

#ifdef _MPI
    /* Receieve U data */
    faces->recv_U_data();
    
    /* Complete computation on remaining flux points */
    faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

    /* Compute flux point contribution to (corrected) gradient of state variables at solution points */
    eles->compute_dU_fpts(startEle, endEle);

    /* Copy un-transformed dU to dUr for later use (L-M chain rule) */
    if (input->motion)
      eles->compute_dU0(startEle, endEle);

    /* Transform gradient of state variables to physical space from 
     * reference space */
    //eles->transform_dU(startEle, endEle);

    /* Compute flux at solution points */
    eles->compute_F(startEle, endEle);

    /* Extrapolate physical solution gradient (computed during compute_F) to flux points */
    eles->extrapolate_dU(startEle, endEle);

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();

    /* Interpolate gradient data to/from other grid(s) */
#ifdef _BUILD_LIB
    if (input->overset)
      ZEFR->overset_interp(faces->nVars, eles->dU_spts.data(), faces->dU.data(), 1);
#endif
#endif

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    
    if (input->motion)
    {
      /* Use Liang-Miyaji Chain-Rule form to compute divF */
      eles->compute_gradF_spts(startEle, endEle);

      eles->transform_gradF_spts(stage, startEle, endEle);
    }
    else
    {
      /* Compute solution point contribution to divergence of flux */
      eles->compute_divF_spts(stage, startEle, endEle);
    }

    /* Compute common interface flux at non-MPI flux points */
    faces->compute_common_F(startFpt, endFpt);

#ifdef _MPI
    /* Receive gradient data */
    faces->recv_dU_data();

    /* Complete computation of fluxes */
    faces->compute_common_F(startFptMpi, geo.nGfpts);
#endif
  }

  if (input->motion) // and input->gridID == 0)
  {
    /* Add standard FR correction to flux divergence (requires extrapolation) */
    eles->extrapolate_Fn(startEle, endEle, faces);
    eles->correct_divF_spts(stage, startEle, endEle);
  }
  else
  {
    /* Compute flux point contribution to divergence of flux */
    eles->compute_divF_fpts(stage, startEle, endEle);
  }

  /* Add source term (if required) */
  if (input->source)
    add_source(stage, startEle, endEle);

}

void FRSolver::compute_LHS()
{
  /* Compute derivative of convective flux with respect to state variables 
   * at solution and flux points */
  eles->compute_dFdUconv();
  faces->compute_dFdUconv(0, geo.nGfpts);

  if (input->viscous)
  {
    /* Compute derivative of common solution with respect to state variables
     * at flux points */
    faces->compute_dUcdU(0, geo.nGfpts);

    /* Compute derivative of viscous flux with respect to state variables 
     * at solution and flux points */
    eles->compute_dFdUvisc();
    faces->compute_dFdUvisc(0, geo.nGfpts);

    /* Compute derivative of viscous flux with respect to the gradient of 
     * state variables at solution and flux points */
    eles->compute_dFddUvisc();
    faces->compute_dFddUvisc(0, geo.nGfpts);
  }

  /* Apply boundary conditions for flux derivative data */
  faces->apply_bcs_dFdU();

  /* Compute normal flux derivative data at flux points */
  faces->compute_dFcdU(0, geo.nGfpts);

  /* Transform flux derivative data from physical to reference space */
  eles->transform_dFdU();
  faces->transform_dFcdU();

  /* Copy normal flux derivative data from face local storage to element local storage */
  dFcdU_from_faces();

  /* Compute LHS implicit Jacobian */
  if (!input->stream_mode)
  {
#ifdef _CPU
      eles->compute_localLHS(dt, 0, geo.nEles);
      compute_LHS_LU(0, geo.nEles);
#endif
#ifdef _GPU
    unsigned int blocksize = ceil(geo.nEles / (double) input->n_LHS_blocks);
    for (unsigned int startEle = 0; startEle <  geo.nEles; startEle += blocksize)
    {
      unsigned int endEle = std::min(startEle + blocksize, geo.nEles);

      eles->compute_localLHS(dt_d, startEle, endEle);
      compute_LHS_LU(startEle, endEle);
    }

    check_error();
#endif
  }
  else
  {
    for (unsigned int color = geo.nColors; color > 0; color--)
    {
      unsigned int startEle = geo.ele_color_range[color - 1];
      unsigned int endEle = geo.ele_color_range[color];

#ifdef _CPU
      eles->compute_localLHS(dt, startEle, endEle, color);
#endif
#ifdef _GPU
      eles->compute_localLHS(dt_d, startEle, endEle, color);
#endif
      compute_LHS_LU(startEle, endEle, color);

#ifdef _GPU
      copy_from_device(eles->LHSInvs[color - 1].data(), eles->LHSInv_d.data(), eles->LHSInv_d.max_size());
#endif
    }
  }
}

void FRSolver::compute_LHS_LU(unsigned int startEle, unsigned int endEle, unsigned int color)
{

#ifdef _GPU
  unsigned int N = eles->nSpts * eles->nVars;

  /* Perform batched LU using cuBLAS */
  if (input->LU_pivot)
  {
    cublasDgetrfBatched_wrapper(N, eles->LHS_ptrs_d.data(), N, eles->LU_pivots_d.data(), eles->LU_info_d.data(), 
        endEle - startEle);
  }
  else
  {
    cublasDgetrfBatched_wrapper(N, eles->LHS_ptrs_d.data(), N, nullptr, eles->LU_info_d.data(), 
        endEle - startEle);
  }

  if (input->inv_mode)
  {
    if (!input->stream_mode)
    {
      if (input->LU_pivot)
      {
        cublasDgetriBatched_wrapper(N, (const double**) eles->LHS_ptrs_d.data(), N, eles->LU_pivots_d.data(), 
            eles->LHSInv_ptrs_d.data() + startEle, N, eles->LU_info_d.data(), endEle - startEle);
      }
      else
      {
        cublasDgetriBatched_wrapper(N, (const double**) eles->LHS_ptrs_d.data(), N, nullptr, 
            eles->LHSInv_ptrs_d.data() + startEle, N, eles->LU_info_d.data(), endEle - startEle);
      }
    }
    else
    {
      cublasDgetriBatched_wrapper(N, (const double**) eles->LHS_ptrs_d.data(), N, eles->LU_pivots_d.data(), 
          eles->LHSInv_ptrs_d.data(), N, eles->LU_info_d.data(), endEle - startEle);
    }
  }
#endif

#ifdef _CPU
#ifndef _NO_TNT
  for (unsigned int ele = 0; ele < endEle - startEle; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

    /* Copy LHS into TNT object */
    // TODO: Copy can now be removed. Need to investigate column-major TNT array views.
    unsigned int N = eles->nSpts * eles->nVars;
    TNT::Array2D<double> A(N, N);
    for (unsigned int nj = 0; nj < eles->nVars; nj++)
    {
      for (unsigned int ni = 0; ni < eles->nVars; ni++)
      {
        for (unsigned int sj = 0; sj < eles->nSpts; sj++)
        {
          for (unsigned int si = 0; si < eles->nSpts; si++)
          {
            unsigned int i = ni * eles->nSpts + si;
            unsigned int j = nj * eles->nSpts + sj;
            A[i][j] = eles->LHSs[color - 1](si, ni, sj, nj, ele);
          }
        }
      }
    }

    /* Calculate and store LU object */
    LUptrs[color - 1][ele] = JAMA::LU<double>(A);
  }
#endif
#endif
}

void FRSolver::compute_RHS(unsigned int color)
{
  unsigned int startEle = geo.ele_color_range[color - 1];
  unsigned int endEle = geo.ele_color_range[color];
#ifdef _CPU
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        if (input->dt_type != 2)
        {
          eles->RHS(spt, n, ele) = -(dt(0) * eles->divF_spts(spt, ele, n, 0)) / eles->jaco_det_spts(spt, ele);
        }
        else
        {
          eles->RHS(spt, n, ele) = -(dt(ele) * eles->divF_spts(spt, ele, n, 0)) / eles->jaco_det_spts(spt, ele);
        }
      }
    }
  }
#endif

#ifdef _GPU
  compute_RHS_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, eles->RHS_d, input->dt_type, eles->nSpts, 
      eles->nEles, eles->nVars, startEle, endEle);
#endif
}

#ifdef _CPU
void FRSolver::compute_RHS_source(const mdvector<double> &source, unsigned int color)
{
  unsigned int startEle = geo.ele_color_range[color - 1];
  unsigned int endEle = geo.ele_color_range[color];
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        if (input->dt_type != 2)
        {
          eles->RHS(spt, n, ele) = -(dt(0) * (eles->divF_spts(spt, ele, n, 0) + source(spt, ele, n))) / eles->jaco_det_spts(spt, ele);
        }
        else
        {
          eles->RHS(spt, n, ele) = -(dt(ele) * (eles->divF_spts(spt, ele, n, 0) + source(spt, ele, n))) / eles->jaco_det_spts(spt, ele);
        }
      }
    }
  }
}
#endif

#ifdef _GPU
void FRSolver::compute_RHS_source(const mdvector_gpu<double> &source, unsigned int color)
{
  unsigned int startEle = geo.ele_color_range[color - 1];
  unsigned int endEle = geo.ele_color_range[color];

  compute_RHS_source_wrapper(eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, eles->RHS_d, input->dt_type, eles->nSpts, 
      eles->nEles, eles->nVars, startEle, endEle);
}
#endif

void FRSolver::compute_deltaU(unsigned int color)
{
  unsigned int startEle = geo.ele_color_range[color - 1];
  unsigned int endEle = geo.ele_color_range[color];

  if (!input->stream_mode)
    color = 1;

#ifdef _CPU
#ifndef _NO_TNT
  unsigned int idx = 0;

  if (!input->stream_mode)
    idx = startEle;

  for (unsigned int ele = startEle; ele < endEle; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

    /* Create Array1D view of RHS */
    unsigned int N = eles->nSpts * eles->nVars;
    TNT::Array1D<double> b(N, &eles->RHS(0, 0, ele));

    /* Solve for deltaU */
    TNT::Array1D<double> x(N, &eles->deltaU(0, 0, ele));
    x.inject(LUptrs[color - 1][idx].solve(b));

    if (x.dim() == 0)
    {
      ThrowException("LU solve failed!");
    }

    idx++;
  }

#endif
#endif

#ifdef _GPU
  if (!input->inv_mode)
  {
    /* Solve LU systems using batched cublas routine */
    unsigned int N = eles->nSpts * eles->nVars;
    int info;

    if (input->LU_pivot)
    {
      cublasDgetrsBatched_wrapper(N, 1, (const double**) (eles->LHS_ptrs_d.data() + startEle), N, eles->LU_pivots_d.data() + startEle * N, 
          eles->RHS_ptrs_d.data() + startEle, N, &info, endEle - startEle);
    }
    {
      cublasDgetrsBatched_wrapper(N, 1, (const double**) (eles->LHS_ptrs_d.data() + startEle), N, nullptr, 
          eles->RHS_ptrs_d.data() + startEle, N, &info, endEle - startEle);
    }

    if (info)
      ThrowException("cublasDgetrs failed. info = " + std::to_string(info));
  }
  else
  {
    unsigned int N = eles->nSpts * eles->nVars;

    if (!input->stream_mode)
      cublasDgemvBatched_wrapper(N, N, 1.0, (const double**) (eles->LHSInv_ptrs_d.data() + startEle), N, (const double**) eles->RHS_ptrs_d.data() + startEle, 
          1, 0.0, eles->deltaU_ptrs_d.data() + startEle, 1, endEle - startEle); 
    else
      cublasDgemvBatched_wrapper(N, N, 1.0, (const double**) (eles->LHSInv_ptrs_d.data()), N, (const double**) eles->RHS_ptrs_d.data() + startEle, 
          1, 0.0, eles->deltaU_ptrs_d.data() + startEle, 1, endEle - startEle); 

  }
#endif
}

void FRSolver::compute_U(unsigned int color)
{
  unsigned int startEle = geo.ele_color_range[color - 1];
  unsigned int endEle = geo.ele_color_range[color];

#ifdef _CPU
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        eles->U_spts(spt, ele, n) += eles->deltaU(spt, n, ele);
      }
    }
  }
#endif

#ifdef _GPU
  /* Add RHS (which contains deltaU) to U */
  if (input->inv_mode)
  {
    compute_U_wrapper(eles->U_spts_d, eles->deltaU_d, eles->nSpts, eles->nEles, eles->nVars, startEle, endEle);
  }
  else
  {
    compute_U_wrapper(eles->U_spts_d, eles->RHS_d, eles->nSpts, eles->nEles, eles->nVars, startEle, endEle);
  }
#endif
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  eles->U_fpts.assign({eles->nFpts, eles->nEles, eles->nVars});
  if (input->viscous)
    eles->Ucomm.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->U_ppts.assign({eles->nPpts, eles->nEles, eles->nVars});
  eles->U_qpts.assign({eles->nQpts, eles->nEles, eles->nVars});

  if (input->squeeze)
    eles->Uavg.assign({eles->nEles, eles->nVars});

  eles->F_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  //eles->F_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->Fcomm.assign({eles->nFpts, eles->nEles, eles->nVars});

  if (input->viscous)
  {
    eles->dU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
    eles->dU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
    eles->dU_qpts.assign({eles->nQpts, eles->nEles, eles->nVars, eles->nDims});
  }

  if (input->dt_scheme != "LSRK")
    eles->divF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, nStages});
  else
    eles->divF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, 1});

  if (input->motion)
  {
    eles->dUr_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
    eles->dF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims, eles->nDims});
    eles->dFn_fpts.assign({eles->nFpts, eles->nEles, eles->nVars});
    eles->tempF_fpts.assign({eles->nFpts, eles->nEles});
  }

  /* Allocate memory for implicit method data structures */
  if (input->dt_scheme == "MCGS")
  {
    if (!input->inv_mode and input->n_LHS_blocks != 1)
    {
      ThrowException("If inv_mode != 0, n_LHS_blocks must equal 1!");
    } 

    eles->dFdU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims});
    eles->dFcdU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, 2});

    if(input->viscous)
    {
      eles->dUcdU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, 2});

      /* Note: nDimsi: Fx, Fy // nDimsj: dUdx, dUdy */
      eles->dFddU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims, eles->nDims});
      eles->dFcddU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims, 2});
    }
      
    if (!input->stream_mode)
    {
      eles->LHSs.resize(1);
      eles->LHSInvs.resize(1);
      LUptrs.resize(1);

#ifdef _CPU
      unsigned int nElesMax = eles->nEles;
#endif
#ifdef _GPU
      unsigned int nElesMax = ceil(geo.nEles / (double) input->n_LHS_blocks);
#endif

      eles->LHSs[0].assign({eles->nSpts, eles->nVars, eles->nSpts, eles->nVars, nElesMax}, 0);
      LUptrs[0].resize(eles->nEles);
      
      if (input->inv_mode)
      {
        eles->LHSInvs[0].assign({eles->nSpts, eles->nVars, eles->nSpts, eles->nVars, eles->nEles}, 0);
        eles->LHSInv_ptrs.assign({eles->nEles});
        eles->deltaU_ptrs.assign({eles->nEles});
      }

      eles->LHS_ptrs.assign({nElesMax});
      eles->RHS_ptrs.assign({eles->nEles});
      eles->LU_pivots.assign({eles->nSpts * eles->nVars * nElesMax});
      eles->LU_info.assign({eles->nSpts * eles->nVars * nElesMax});

    }
    else
    {
      eles->LHSs.resize(geo.nColors);
      eles->LHSInvs.resize(geo.nColors);
      LUptrs.resize(geo.nColors);

      unsigned int nElesMax = *std::max_element(geo.ele_color_nEles.begin(), geo.ele_color_nEles.end());
      for (unsigned int color = 1; color <= geo.nColors; color++)
      {
        eles->LHSs[color - 1].assign({eles->nSpts, eles->nVars, eles->nSpts, eles->nVars, nElesMax}, 0);
        eles->LHSInvs[color - 1].assign({eles->nSpts, eles->nVars, eles->nSpts, eles->nVars, nElesMax}, 0, true);
        LUptrs[color - 1].resize(nElesMax);
      }
      
      if (input->inv_mode)
      {
        eles->LHSInv_ptrs.assign({nElesMax});
        eles->deltaU_ptrs.assign({eles->nEles});
      }

      eles->LHS_ptrs.assign({nElesMax});
      eles->RHS_ptrs.assign({eles->nEles});
      eles->LU_pivots.assign({eles->nSpts * eles->nVars * nElesMax});
      eles->LU_info.assign({eles->nSpts * eles->nVars * nElesMax});
    }

    eles->Cvisc0.assign({eles->nSpts, eles->nSpts, eles->nDims});
    eles->CviscN.assign({eles->nSpts, eles->nSpts, eles->nDims, eles->nFaces});
    eles->CdFddU0.assign({eles->nSpts, eles->nSpts, eles->nDims});
    eles->CtempSS.assign({eles->nSpts, eles->nSpts});
    eles->CtempFS.assign({eles->nFpts, eles->nSpts});
    eles->CtempFS2.assign({eles->nFpts, eles->nSpts});
    eles->CtempSF.assign({eles->nSpts, eles->nFpts});
    eles->CtempFSN.assign({eles->nSpts1D, eles->nSpts});
    eles->CtempFSN2.assign({eles->nSpts1D, eles->nSpts});

    eles->deltaU.assign({eles->nSpts, eles->nVars, eles->nEles});
    eles->RHS.assign({eles->nSpts, eles->nVars, eles->nEles});
  }

  /* Initialize solution */
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    if (input->ic_type == 0)
    {
      // Do nothing for now
    }
    else if (input->ic_type == 1)
    {
      if (input->nDims == 2)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = eles->coord_spts(spt, ele, 0);
            double y = eles->coord_spts(spt, ele, 1);

            eles->U_spts(spt, ele, 0) = compute_U_true(x, y, 0, 0, 0, input);
          }
        }
      }
      else if (input->nDims == 3)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = eles->coord_spts(spt, ele, 0);
            double y = eles->coord_spts(spt, ele, 1);
            double z = eles->coord_spts(spt, ele, 2);

            eles->U_spts(spt, ele, 0) = compute_U_true(x, y, z, 0, 0, input);

          }
        }
      }
    }
    else
    {
      ThrowException("ic_type not recognized!");
    }
  }
  else if (input->equation == EulerNS)
  {
    if (input->ic_type == 0)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          eles->U_spts(spt, ele, 0)  = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < eles->nDims; dim++)
          {
            eles->U_spts(spt, ele, dim+1)  = input->rho_fs * input->V_fs(dim);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          eles->U_spts(spt, ele, eles->nDims + 1)  = input->P_fs/(input->gamma-1.0) +
            0.5*input->rho_fs * Vsq;
        }
      }

    }
    else if (input->ic_type == 1)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = eles->coord_spts(spt, ele, 0);
            double y = eles->coord_spts(spt, ele, 1);

            eles->U_spts(spt, ele, n) = compute_U_true(x, y, 0, 0, n, input);
          }
        }
      }
    }
  }
  else
  {
    ThrowException("Solution initialization not recognized!");
  }
}

void FRSolver::setup_views()
{
  /* Setup face view of element solution data struture */
  // TODO: Might not want to allocate all these at once. Turn this into a function maybe?
  mdvector<double*> U_base_ptrs({2 * geo.nGfpts});
  mdvector<double*> Fcomm_base_ptrs({2 * geo.nGfpts});
  mdvector<unsigned int> U_strides({2 * geo.nGfpts});
  mdvector<double*> Ucomm_base_ptrs;
  mdvector<double*> dU_base_ptrs;
  mdvector<unsigned int> dU_strides;

  if (input->viscous)
  {
    Ucomm_base_ptrs.assign({2 * geo.nGfpts});
    dU_base_ptrs.assign({2 * geo.nGfpts});
    dU_strides.assign({2 * geo.nGfpts, 2});
  }
#ifdef _GPU
  mdvector<double*> U_base_ptrs_d({2 * geo.nGfpts});
  mdvector<double*> Fcomm_base_ptrs_d({2 * geo.nGfpts});
  mdvector<unsigned int> U_strides_d({2 * geo.nGfpts});
  mdvector<double*> Ucomm_base_ptrs_d;
  mdvector<double*> dU_base_ptrs_d;
  mdvector<unsigned int> dU_strides_d;

  if (input->viscous)
  {
    Ucomm_base_ptrs_d.assign({2 * geo.nGfpts});
    dU_base_ptrs_d.assign({2 * geo.nGfpts});
    dU_strides_d.assign({2 * geo.nGfpts, 2});
  }
#endif

  /* Set pointers for internal faces */
#pragma omp parallel for collapse(3)
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
    {
      int gfpt = geo.fpt2gfpt(fpt,ele);
      /* Check if flux point is on ghost edge */
      if (gfpt == -1)
      {
        continue;
      }

      int slot = geo.fpt2gfpt_slot(fpt,ele);

      U_base_ptrs(gfpt + slot * geo.nGfpts) = &eles->U_fpts(fpt, ele, 0);
      U_strides(gfpt + slot * geo.nGfpts) = eles->U_fpts.get_stride(1);

      Fcomm_base_ptrs(gfpt + slot * geo.nGfpts) = &eles->Fcomm(fpt, ele, 0);

      if (input->viscous) Ucomm_base_ptrs(gfpt + slot * geo.nGfpts) = &eles->Ucomm(fpt, ele, 0);
#ifdef _GPU
      U_base_ptrs_d(gfpt + slot * geo.nGfpts) = eles->U_fpts_d.get_ptr(fpt, ele, 0);
      U_strides_d(gfpt + slot * geo.nGfpts) = eles->U_fpts_d.get_stride(1);

      Fcomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = eles->Fcomm_d.get_ptr(fpt, ele, 0);

      if (input->viscous) Ucomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = eles->Ucomm_d.get_ptr(fpt, ele, 0);
#endif

      if (input->viscous)
      {
        dU_base_ptrs(gfpt + slot * geo.nGfpts) = &eles->dU_fpts(fpt, ele, 0, 0);
        dU_strides(gfpt + slot * geo.nGfpts, 0) = eles->dU_fpts.get_stride(1);
        dU_strides(gfpt + slot * geo.nGfpts, 1) = eles->dU_fpts.get_stride(2);

#ifdef _GPU
        dU_base_ptrs_d(gfpt + slot * geo.nGfpts) = eles->dU_fpts_d.get_ptr(fpt, ele, 0, 0);
        dU_strides_d(gfpt + slot * geo.nGfpts, 0) = eles->dU_fpts_d.get_stride(1);
        dU_strides_d(gfpt + slot * geo.nGfpts, 1) = eles->dU_fpts_d.get_stride(2);
#endif
       
      }
      
    }
  }

  /* Set pointers for remaining faces (includes boundary and MPI faces) */
  unsigned int i = 0;
  for (unsigned int gfpt = geo.nGfpts_int; gfpt < geo.nGfpts; gfpt++)
  {
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      U_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd(i, 0);
      U_strides(gfpt + 1 * geo.nGfpts) = faces->U_bnd.get_stride(0);

      Fcomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Fcomm_bnd(i, 0);

      if (input->viscous) Ucomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Ucomm_bnd(i, 0);

#ifdef _GPU
      U_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_ptr(i, 0);
      U_strides_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_stride(0);

      Fcomm_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->Fcomm_bnd_d.get_ptr(i, 0);

      if (input->viscous) Ucomm_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->Ucomm_bnd_d.get_ptr(i, 0);
#endif

      if (input->viscous)
      {
        dU_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->dU_bnd(i, 0, 0);
        dU_strides(gfpt + 1 * geo.nGfpts, 0) = faces->dU_bnd.get_stride(0);
        dU_strides(gfpt + 1 * geo.nGfpts, 1) = faces->dU_bnd.get_stride(1);
#ifdef _GPU
        dU_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->dU_bnd_d.get_ptr(i, 0, 0);
        dU_strides_d(gfpt + 1 * geo.nGfpts, 0) = faces->dU_bnd_d.get_stride(0);
        dU_strides_d(gfpt + 1 * geo.nGfpts, 1) = faces->dU_bnd_d.get_stride(1);
#endif
      }
    }

    i++;
  }

  /* Create views of element data for faces */
  faces->U.assign(U_base_ptrs, U_strides, geo.nGfpts);
  faces->Fcomm.assign(Fcomm_base_ptrs, U_strides, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm.assign(Ucomm_base_ptrs, U_strides, geo.nGfpts);
    faces->dU.assign(dU_base_ptrs, dU_strides, geo.nGfpts);
  }

#ifdef _GPU
  faces->U_d.assign(U_base_ptrs_d, U_strides_d, geo.nGfpts);
  faces->Fcomm_d.assign(Fcomm_base_ptrs_d, U_strides_d, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm_d.assign(Ucomm_base_ptrs_d, U_strides_d, geo.nGfpts);
    faces->dU_d.assign(dU_base_ptrs_d, dU_strides_d, geo.nGfpts);
  }
#endif

}

void FRSolver::dFcdU_from_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int nj = 0; nj < eles->nVars; nj++) 
  {
    for (unsigned int ni = 0; ni < eles->nVars; ni++) 
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfpt(fpt,ele);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          int slot = geo.fpt2gfpt_slot(fpt,ele);
          int notslot = 1;
          if (slot == 1)
          {
            notslot = 0;
          }

          /* Combine dFcdU on non-periodic boundaries */
          // TODO: might need to move this to faces
          if (gfpt >= (int)geo.nGfpts_int && gfpt < (int)(geo.nGfpts_int + geo.nGfpts_bnd))
          {
            unsigned int bnd_id = geo.gfpt2bnd(gfpt - geo.nGfpts_int);
            if (bnd_id != PERIODIC)
            {
              eles->dFcdU_fpts(fpt, ele, ni, nj, 0) = faces->dFcdU(gfpt, ni, nj, slot, slot) + 
                                                      faces->dFcdU(gfpt, ni, nj, notslot, slot);
              continue;
            }
          }
          eles->dFcdU_fpts(fpt, ele, ni, nj, 0) = faces->dFcdU(gfpt, ni, nj, slot, slot);
          eles->dFcdU_fpts(fpt, ele, ni, nj, 1) = faces->dFcdU(gfpt, ni, nj, notslot, slot);
        }
      }
    }
  }

  if(input->viscous)
  {
#pragma omp parallel for collapse(3)
    for (unsigned int nj = 0; nj < eles->nVars; nj++) 
    {
      for (unsigned int ni = 0; ni < eles->nVars; ni++) 
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
          {
            int gfpt = geo.fpt2gfpt(fpt,ele);
            /* Check if flux point is on ghost edge */
            if (gfpt == -1)
              continue;
            int slot = geo.fpt2gfpt_slot(fpt,ele);
            int notslot = 1;
            if (slot == 1)
            {
              notslot = 0;
            }

            eles->dUcdU_fpts(fpt, ele, ni, nj, 0) = faces->dUcdU(gfpt, ni, nj, slot);
            eles->dUcdU_fpts(fpt, ele, ni, nj, 1) = faces->dUcdU(gfpt, ni, nj, notslot);

            /* Combine dFcddU on non-periodic boundaries */
            // TODO: might need to move this to faces
            if (gfpt >= (int)geo.nGfpts_int && gfpt < (int)(geo.nGfpts_int + geo.nGfpts_bnd))
            {
              unsigned int bnd_id = geo.gfpt2bnd(gfpt - geo.nGfpts_int);
              if (bnd_id != PERIODIC)
              {
                for (unsigned int dim = 0; dim < eles->nDims; dim++)
                {
                  eles->dFcddU_fpts(fpt, ele, ni, nj, dim, 0) = faces->dFcddU(gfpt, ni, nj, dim, slot, slot) +
                                                                faces->dFcddU(gfpt, ni, nj, dim, notslot, slot);
                }
                continue;
              }
            }

            for (unsigned int dim = 0; dim < eles->nDims; dim++)
            {
              eles->dFcddU_fpts(fpt, ele, ni, nj, dim, 0) = faces->dFcddU(gfpt, ni, nj, dim, slot, slot);
              eles->dFcddU_fpts(fpt, ele, ni, nj, dim, 1) = faces->dFcddU(gfpt, ni, nj, dim, notslot, slot);
            }
          }
        }
      }
    }
  }
#endif

#ifdef _GPU
  dFcdU_from_faces_wrapper(faces->dFcdU_d, eles->dFcdU_fpts_d, geo.fpt2gfpt_d,
      geo.fpt2gfpt_slot_d, geo.gfpt2bnd_d, geo.nGfpts_int, geo.nGfpts_bnd, eles->nVars, 
      eles->nEles, eles->nFpts, eles->nDims, input->equation);
  check_error();
#endif
}

void FRSolver::add_source(unsigned int stage, unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
          double x = eles->coord_spts(spt, ele, 0);
          double y = eles->coord_spts(spt, ele, 1);
          double z = 0;
          if (eles->nDims == 3)
            z = eles->coord_spts(spt, ele, 2);

          eles->divF_spts(spt, ele, n, stage) += compute_source_term(x, y, z, flow_time, n, input) * 
            eles->jaco_det_spts(spt, ele);
      }
    }
  }

#endif

#ifdef _GPU
  add_source_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, eles->coord_spts_d, eles->nSpts, eles->nEles,
      eles->nVars, eles->nDims, input->equation, flow_time, stage, startEle, endEle);
  check_error();
#endif

}

/* Note: Source term in update() is used primarily for multigrid. To add a true source term, define
 * a source term in funcs.cpp and set source input flag to 1. */
#ifdef _CPU
void FRSolver::update(const mdvector<double> &source)
#endif 
#ifdef _GPU
void FRSolver::update(const mdvector_gpu<double> &source)
#endif
{
  prev_time = flow_time;

  if (input->dt_scheme == "LSRK")
  {
    step_adaptive_LSRK(source);
  }
  else
  {
    if (input->dt_scheme == "MCGS")
      step_MCGS(source);
    else
      step_RK(source);
  }

  flow_time = prev_time + dt(0);
  current_iter++;

  // Update grid to end of time step (if not already done so)
  if (input->dt_scheme != "MCGS" && (nStages == 1 || (nStages > 1 && rk_alpha(nStages-2) != 1)))
    move(flow_time);

#ifdef _BUILD_LIB
  // Update the overset connectivity to the new grid positions
  if (input->overset && input->motion)
  {
    ZEFR->tg_preprocess();
    ZEFR->tg_process_connectivity();
  }
#endif
}


#ifdef _CPU
void FRSolver::step_RK(const mdvector<double> &source)
#endif
#ifdef _GPU
void FRSolver::step_RK(const mdvector_gpu<double> &source)
#endif
{
#ifdef _CPU
  if (nStages > 1)
    U_ini = eles->U_spts;
#endif

#ifdef _GPU
  device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.max_size());
  check_error();
#endif

  unsigned int nSteps = (input->dt_scheme == "RKj") ? nStages : nStages - 1;

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < nSteps; stage++)
  {
    flow_time = prev_time + rk_alpha(stage) * dt(0);

    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#ifdef _CPU
    if (source.size() == 0)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int n = 0; n < eles->nVars; n++)
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(0) /
                  eles->jaco_det_spts(spt, ele) * eles->divF_spts(spt, ele, n, stage);
            }
            else
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(ele) /
                  eles->jaco_det_spts(spt, ele) * eles->divF_spts(spt, ele, n, stage);
            }
          }
        }
    }
    else
    {
#pragma omp parallel for collapse(2)
      for (unsigned int n = 0; n < eles->nVars; n++)
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            if (input->dt_type != 2)
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(0) /
                  eles->jaco_det_spts(spt,ele) * (eles->divF_spts(spt, ele, n, stage) + source(spt, ele, n));
            }
            else
            {
              eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha(stage) * dt(ele) /
                  eles->jaco_det_spts(spt,ele) * (eles->divF_spts(spt, ele, n, stage) + source(spt, ele, n));
            }
          }
        }
    }
#endif

#ifdef _GPU
    /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
    unsigned int last_stage = (input->dt_scheme == "RKj") ? nStages + 1 : nStages;

    if (source.size() == 0)
    {
      RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d,
                        rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
                        input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
    }
    else
    {
      RK_update_source_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d,
                               rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
                               input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
    }
    check_error();
#endif

    // Update grid to position of next time step
    move(flow_time);

#ifdef _BUILD_LIB
    // Update the overset connectivity to the new grid positions
    if (input->overset && input->motion)
    {
      ZEFR->tg_preprocess();
      ZEFR->tg_process_connectivity();
#ifdef _GPU
      ZEFR->update_iblank_gpu();
#endif
    }
#endif
  }

  /* Final stage combining residuals for full Butcher table style RK timestepping*/
  if (input->dt_scheme != "RKj")
  {
    if (nStages > 1)
      flow_time = prev_time + rk_alpha(nStages-2) * dt(0);

    compute_residual(nStages-1);
#ifdef _CPU
    if (nStages > 1)
      eles->U_spts = U_ini;
    else if (input->dt_type != 0)
      compute_element_dt();
#endif
#ifdef _GPU
    device_copy(eles->U_spts_d, U_ini_d, eles->U_spts_d.max_size());
#endif

#ifdef _CPU
    for (unsigned int stage = 0; stage < nStages; stage++)
    {
      if (source.size() == 0)
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < eles->nVars; n++)
          for (unsigned int ele = 0; ele < eles->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < eles->nSpts; spt++)
              if (input->dt_type != 2)
              {
                eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(0) / eles->jaco_det_spts(spt,ele) *
                    eles->divF_spts(spt, ele, n, stage);
              }
              else
              {
                eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(ele) / eles->jaco_det_spts(spt,ele) *
                    eles->divF_spts(spt, ele, n, stage);
              }
          }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < eles->nVars; n++)
          for (unsigned int ele = 0; ele < eles->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < eles->nSpts; spt++)
            {
              if (input->dt_type != 2)
              {
                eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(0) / eles->jaco_det_spts(spt,ele) *
                    (eles->divF_spts(spt, ele, n, stage) + source(spt, ele, n));
              }
              else
              {
                eles->U_spts(spt, ele, n) -= rk_beta(stage) * dt(ele) / eles->jaco_det_spts(spt,ele) *
                    (eles->divF_spts(spt, ele, n, stage) + source(spt, ele, n));
              }
            }
          }
      }
    }
#endif

#ifdef _GPU
    if (source.size() == 0)
    {
      RK_update_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d,
                        rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
                        input->equation, 0, nStages, true, input->overset, geo.iblank_cell_d.data());
    }
    else
    {
      RK_update_source_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d,
                               rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
                               input->equation, 0, nStages, true, input->overset, geo.iblank_cell_d.data());
    }

    check_error();
#endif
  }
}

#ifdef _CPU
void FRSolver::step_adaptive_LSRK(const mdvector<double> &source)
#endif
#ifdef _GPU
void FRSolver::step_adaptive_LSRK(const mdvector_gpu<double> &source)
#endif
{
  step_LSRK(source);

  // Calculate error (infinity norm of RK error) and scaling factor for dt
  double max_err = 0;
#ifdef _CPU
  for (uint n = 0; n < eles->nVars; n++)
  {
    for (uint ele = 0; ele < eles->nEles; ele++)
    {
      for (uint spt = 0; spt < eles->nSpts; spt++)
      {
        double err = std::abs(rk_err(spt,ele,n)) /
            (input->atol + input->rtol * std::max( std::abs(eles->U_spts(spt,ele,n)), std::abs(U_ini(spt,ele,n)) ));
        max_err = std::max(max_err, err);
      }
    }
  }

#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

  // Determine the time step scaling factor and the new time step
  double fac = pow(max_err, -expa) * pow(prev_err, expb);
  fac = std::min(input->maxfac, std::max(input->minfac, input->sfact*fac));

  dt(0) *= fac;
#endif

#ifdef _GPU
  max_err = set_adaptive_dt_wrapper(eles->U_spts_d, U_ini_d, rk_err_d, dt_d, dt(0),
      eles->nSpts, eles->nEles, eles->nVars, input->atol, input->rtol, expa, expb,
      input->minfac, input->maxfac, input->sfact, prev_err, worldComm, input->overset,
      geo.iblank_cell_d.data());
#endif

  if (dt(0) < 1e-14)
    ThrowException("dt approaching 0 - quitting simulation");

  if (max_err < 1.)
  {
    // Accept the time step and continue on
    prev_err = max_err;
  }
  else
  {
    // Reject step - reset solution back to beginning of time step
    flow_time = prev_time;
#ifdef _CPU
    eles->U_spts = U_ini;
#endif
#ifdef _GPU
    device_copy(eles->U_spts_d, U_ini_d, U_ini_d.max_size());
#endif
    // Try again with new dt
    step_adaptive_LSRK(source);
  }
}

#ifdef _CPU
void FRSolver::step_LSRK(const mdvector<double> &source)
#endif
#ifdef _GPU
void FRSolver::step_LSRK(const mdvector_gpu<double> &source)
#endif
{
  /* NOTE: this implementation is not the 'true' low-storage implementation
   * since we are using an additional array 'U_til' instead of swapping
   * pointers at each stage */

  // Copy current solution into "U_ini" ['rold' in PyFR]
#ifdef _CPU
  U_ini = eles->U_spts;
  U_til = eles->U_spts;
  rk_err.fill(0.0);
#endif

#ifdef _GPU
  device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.max_size());
  device_copy(U_til_d, eles->U_spts_d, eles->U_spts_d.max_size());
  device_fill(rk_err_d, rk_err_d.max_size());

  // Get current delta t [dt(0)] (updated on GPU)
  copy_from_device(dt.data(), dt_d.data(), 1);

  check_error();
#endif

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < nStages; stage++)
  {
    flow_time = prev_time + rk_c(stage) * dt(0);

    compute_residual(0);

    double ai = rk_alpha(stage);
    double bi = rk_beta(stage);
    double bhi = rk_bhat(stage);

#ifdef _CPU
    // Update Error
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          rk_err(spt,ele,n) -= (bi - bhi) * dt(0) /
              eles->jaco_det_spts(spt,ele) * eles->divF_spts(spt,ele,n,0);
        }
      }
    }

    // Update solution registers
    if (stage < nStages - 1)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            eles->U_spts(spt,ele,n) = U_til(spt,ele,n) - ai * dt(0) /
                eles->jaco_det_spts(spt,ele) * eles->divF_spts(spt,ele,n,0);

            U_til(spt,ele,n) = eles->U_spts(spt,ele,n) - (bi - ai) * dt(0) /
                eles->jaco_det_spts(spt,ele) * eles->divF_spts(spt,ele,n,0);
          }
        }
      }
    }
    else
    {
#pragma omp parallel for collapse(2)
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            eles->U_spts(spt,ele,n) = U_til(spt,ele,n) - bi * dt(0) /
                eles->jaco_det_spts(spt,ele) * eles->divF_spts(spt,ele,n,0);
          }
        }
      }
    }
#endif

#ifdef _GPU
    if (source.size() == 0)
    {
      LSRK_update_wrapper(eles->U_spts_d, U_til_d, rk_err_d, eles->divF_spts_d,
          eles->jaco_det_spts_d, dt(0), ai, bi, bhi, eles->nSpts, eles->nEles,
          eles->nVars, stage, nStages, input->overset, geo.iblank_cell_d.data());
    }
    else
    {
      LSRK_update_source_wrapper(eles->U_spts_d, U_til_d, rk_err_d,
          eles->divF_spts_d, source, eles->jaco_det_spts_d, dt(0), ai, bi, bhi,
          eles->nSpts, eles->nEles, eles->nVars, stage, nStages, input->overset,
          geo.iblank_cell_d.data());
    }
    check_error();
#endif

    // Update grid to position of next time step
    // move(flow_time);

#ifdef _BUILD_LIB
    // Update the overset connectivity to the new grid positions
    if (input->overset && input->motion)
    {
      ZEFR->tg_preprocess();
      ZEFR->tg_process_connectivity();
#ifdef _GPU
      ZEFR->update_iblank_gpu();
#endif
    }
#endif
  }

  flow_time = prev_time + dt(0);
}

#ifdef _CPU
void FRSolver::step_MCGS(const mdvector<double> &source)
#endif
#ifdef _GPU
void FRSolver::step_MCGS(const mdvector_gpu<double> &source)
#endif
{
  /* Sweep through colors */
  for (unsigned int counter = 1; counter <= nCounter; counter++)
  {
    /* Set color */
    unsigned int color = counter;
    if (color > geo.nColors)
      color = 2*geo.nColors+1 - counter;

    /* Compute residual and Jacobian on all elements */
    int iter = current_iter - restart_iter;
    if (counter == 1 && iter%input->Jfreeze_freq == 0)
    {
      compute_residual(0);

      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }

      /* Compute SER time step growth */
      if (input->SER)
      {
        compute_SER_dt();
#ifdef _GPU
        ThrowException("SER not available on GPU!");
#endif
      }

      /* Compute LHS implicit Jacobian */
      compute_LHS();
    }
    /* If running multigrid, assume solution has been updated externally. Compute res on
       * all elements to update face data */
    else if (input->p_multi and counter == 1)
    {
      compute_residual(0);
    }
    /* Compute residual on elements of this color only */
    else
    {
      compute_residual(0, color);
    }
    prev_color = color;

    /* Prepare RHS vector */
    if (source.size() == 0)
    {
      compute_RHS(color);
    }
    else
    {
      compute_RHS_source(source, color);
    }

#ifdef _GPU
    if (input->stream_mode)
      sync_stream(1);
#endif

    /* Solve system for deltaU */
    compute_deltaU(color);

#ifdef _GPU
    /* Begin transfer of LHSInv for next color */
    if (input->stream_mode)
    {
      cudaDeviceSynchronize();
      copy_to_device(eles->LHSInv_d.data(), eles->LHSInvs[(color) % geo.nColors].data(), eles->LHSInv_d.max_size(), 1);
    }
#endif

    /* Add deltaU to solution */
    compute_U(color);
  }
}

void FRSolver::compute_element_dt()
{
  /* Adapt CFL number */
  double CFL;
  if (input->adapt_CFL)
  {
    CFL_ratio *= input->CFL_ratio;
    CFL = input->CFL * CFL_ratio;
    if (CFL > input->CFL_max)
    {
      CFL = input->CFL_max;
    }
  }
  else
  {
    CFL = input->CFL;
  }

#ifdef _CPU
  /* CFL-estimate used by Liang, Lohner, and others. Factor of 2 to be 
   * consistent with 1D CFL estimates. */
  if (input->CFL_type == 1)
  {
#pragma omp parallel for
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    { 
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      double int_waveSp = 0.;  /* Edge/Face integrated wavespeed */

      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        /* Skip if on ghost edge. */
        int gfpt = geo.fpt2gfpt(fpt,ele);
        if (gfpt == -1)
          continue;

        if (eles->nDims == 2)
        {
          int_waveSp += eles->weights_spts(fpt % eles->nSpts1D) * faces->waveSp(gfpt) * faces->dA(gfpt);
        }
        else
        {
          int idx = fpt % (eles->nSpts1D * eles->nSpts1D);
          int i = idx % eles->nSpts1D;
          int j = idx / eles->nSpts1D;

          int_waveSp += eles->weights_spts(i) * eles->weights_spts(j) * faces->waveSp(gfpt) * faces->dA(gfpt);
        }
      }

      dt(ele) = 2.0 * CFL * get_cfl_limit_adv(order) * eles->vol(ele) / int_waveSp;
    }
  }

  /* CFL-estimate based on MacCormack for NS */
  else if (input->CFL_type == 2)
  {
    int nFptsFace = (eles->nDims == 2) ? eles->nSpts1D : eles->nSpts1D*eles->nSpts1D;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    { 
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      /* Compute inverse of timestep in each face */
      std::vector<double> dtinv(2*eles->nDims);
      for (unsigned int face = 0; face < 2*eles->nDims; face++)
      {
        for (unsigned int fpt = face * nFptsFace; fpt < (face+1) * nFptsFace; fpt++)
        {
          /* Skip if on ghost edge. */
          int gfpt = geo.fpt2gfpt(fpt,ele);
          if (gfpt == -1)
            continue;

          double dtinv_temp = faces->waveSp(gfpt) / (get_cfl_limit_adv(order) * eles->h_ref(fpt, ele));
          if (input->viscous)
            dtinv_temp += faces->diffCo(gfpt) / (get_cfl_limit_diff(order, input->ldg_b) * eles->h_ref(fpt, ele) * eles->h_ref(fpt, ele));
          dtinv[face] = std::max(dtinv[face], dtinv_temp);
        }
      }

      /* Find maximum in each dimension */
      if (eles->nDims == 2)
      {
        dtinv[0] = std::max(dtinv[0], dtinv[2]);
        dtinv[1] = std::max(dtinv[1], dtinv[3]);

        dt(ele) = CFL / (dtinv[0] + dtinv[1]);
      }
      else
      {
        dtinv[0] = std::max(dtinv[0],dtinv[1]);
        dtinv[1] = std::max(dtinv[2],dtinv[3]);
        dtinv[2] = std::max(dtinv[4],dtinv[5]);

        /// NOTE: this seems ultra-conservative.  Need additional scaling factor?
        dt(ele) = CFL / (dtinv[0] + dtinv[1] + dtinv[2]); // * 32; = empirically-found factor for sphere
      }
    }
  }

  if (input->dt_type == 1) /* Global minimum */
  {
    if (input->overset)
    {
      double minDT = INFINITY;
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        minDT = std::min(minDT, dt(ele));
      }
      dt(0) = minDT;
    }
    else
    {
      dt(0) = *std::min_element(dt.data(), dt.data()+eles->nEles);
    }

#ifdef _MPI
    /// TODO: If interfacing with other explicit solver, work together here
    MPI_Allreduce(MPI_IN_PLACE, &dt(0), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

  }
#endif

#ifdef _GPU
  compute_element_dt_wrapper(dt_d, faces->waveSp_d, faces->diffCo_d, faces->dA_d, geo.fpt2gfpt_d, 
      eles->weights_spts_d, eles->vol_d, eles->h_ref_d, eles->nSpts1D, CFL, input->ldg_b, order, 
      input->dt_type, input->CFL_type, eles->nFpts, eles->nEles, eles->nDims, myComm,
      input->overset, geo.iblank_cell_d.data());

  check_error();
#endif
}

void FRSolver::compute_SER_dt()
{
  /* Compute norm of residual */
  // TODO: Create norm function to eliminate repetition, add other norms
  SER_res[1] = SER_res[0];
  SER_res[0] = 0;
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele =0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        SER_res[0] += (eles->divF_spts(spt, ele, n, 0) / eles->jaco_det_spts(spt, ele)) *
                       (eles->divF_spts(spt, ele, n, 0) / eles->jaco_det_spts(spt, ele));
      }
    }
  }
  SER_res[0] = std::sqrt(SER_res[0]);

  /* Compute SER time step growth */
  double omg = SER_res[1] / SER_res[0];
  if (omg != 0)
  {
    /* Clipping */
    if (omg < 0.1)
      omg = 0.1;
    else if (omg > 2.0)
      omg = 2.0;

    /* Relax Growth */
    if (omg > 1.0)
      omg = std::sqrt(omg);

    /* Compute new time step */
    SER_omg *= omg;
    if (input->dt_type == 0)
    {
      dt(0) *= omg;
    }
    else if(input->dt_type == 1)
    {
      dt(0) *= SER_omg;
    }
    else if (input->dt_type == 2)
    {
#pragma omp parallel for
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        dt(ele) *= SER_omg;
      }
    }
  }
}

void FRSolver::write_solution_pyfr(const std::string &_prefix)
{
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
#endif

  std::string prefix = _prefix;

  if (input->overset) prefix += "-G" + std::to_string(input->gridID);

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  std::string filename = input->output_prefix + "/" + prefix + "-" + std::to_string(iter) + ".pyfrs";

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing data to file " << filename << std::endl;

  std::stringstream ss;

  // Create a dataspace and datatype for a std::string
  DataSpace dspace(H5S_SCALAR);
  hid_t string_type = H5Tcopy (H5T_C_S1);
  H5Tset_size (string_type, H5T_VARIABLE);

  // Setup config and stats strings

  /* --- Config String --- */
  ss.str(""); ss.clear();
  ss << "[constants]" << std::endl;
  ss << "gamma = 1.4" << std::endl;
  ss << std::endl;

  ss << "[solver]" << std::endl;
  ss << "system = ";
  if (input->equation == EulerNS) 
  {
    if (input->viscous)
      ss << "navier-stokes" << std::endl;
    else
      ss << "euler" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    if (input->viscous)
      ss << "advection-diffusion" << std::endl;
    else
      ss << "advection" << std::endl;
  }
  ss << "order = " << input->order << std::endl;
  ss << std::endl;

  if (geo.nDims == 2)
    ss << "[solver-elements-quad]" << std::endl;
  else
    ss << "[solver-elements-hex]" << std::endl;
  ss << "soln-pts = gauss-legendre" << std::endl;
  ss << std::endl;

  std::string config = ss.str();

  /* --- Stats String --- */
  ss.str(""); ss.clear();
  ss << "[data]" << std::endl;
  if (input->equation == EulerNS)
  {
    ss << "fields = rho,rhou,rhov,";
    if (geo.nDims == 3)
      ss << "rhoW,";
    ss << "E" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    ss << "fields = u" << std::endl;
  }
  ss << "prefix = soln" << std::endl;
  ss << std::endl;

  ss << "[solver-time-integrator]" << std::endl;
  ss << "tcurr = " << flow_time << std::endl;
  //ss << "wall-time = " << input->??"
  ss << std::endl;

  std::string stats = ss.str();

  int nEles = eles->nEles;
  int nVars = eles->nVars;
  int nSpts = eles->nSpts;

#ifdef _USE_H5_PARALLEL
  FileCreatPropList h5_mpi_plist = H5Pcreate(H5P_FILE_ACCESS);
  MPI_Info info;
  MPI_Info_create(&info);
  H5Pset_fapl_mpio(h5_mpi_plist, geo.myComm, info);
  MPI_Barrier(MPI_COMM_WORLD);
  H5File file(filename, H5F_ACC_TRUNC, h5_mpi_plist);

  std::vector<int> n_eles_p(input->nRanks);
  MPI_Allgather(&nEles, 1, MPI_INT, n_eles_p.data(), 1, MPI_INT, geo.myComm);

  /* --- Write Data to File --- */

  for (int p = 0; p < input->nRanks; p++)
  {
    nEles = n_eles_p[p];

    hsize_t dims[3] = {nSpts, nVars, nEles}; /// NOTE: We are col-major, HDF5 is row-major

    DataSpace dspaceU(3, dims);
    std::string solname = "soln_";
    solname += (geo.nDims == 2) ? "quad" : "hex";
    solname += "_p" + std::to_string(p);
    DataSet dsetU = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceU);

    if (p == input->rank)
    {
      mdvector<double> u_tmp({nEles, nVars, nSpts});
      for (int ele = 0; ele < nEles; ele++)
        for (int var = 0; var < nVars; var++)
          for (int spt = 0; spt < nSpts; spt++)
            u_tmp(ele,var,spt) = eles->U_spts(spt,ele,var);

      dsetU.write(u_tmp.data(), PredType::NATIVE_DOUBLE, dspaceU);
    }

    dsetU.close();
  }

  DataSet dset = file.createDataSet("config", string_type, dspace);
  if (input->rank == 0)
    dset.write(config, string_type, dspace);
  dset.close();

  dset = file.createDataSet("stats", string_type, dspace);
  if (input->rank == 0)
    dset.write(stats, string_type, dspace);
  dset.close();

  // Write mesh ID
  dset = file.createDataSet("mesh_uuid", string_type, dspace);
  if (input->rank == 0)
    dset.write(geo.mesh_uuid, string_type, dspace);
  dset.close();

  dspace.close();
#else

#ifdef _MPI
  // Need to traspose data to match PyFR layout - [spts,vars,eles] in row-major format
  std::vector<std::vector<double>> data_p(input->nRanks);
  std::vector<std::vector<int>> iblank_p(input->nRanks);
  if (input->rank == 0)
  {
    uint ind = 0;
    data_p[0].resize(nEles * nVars * nSpts);  // Create transposed copy
    for (int spt = 0; spt < nSpts; spt++)
    {
      for (int var = 0; var < nVars; var++)
      {
        for (int ele = 0; ele < nEles; ele++)
        {
          data_p[0][ind] = eles->U_spts(spt,ele,var);
          ind++;
        }
      }
    }

    if (input->overset)
    {
      iblank_p[0].resize(nEles);
      for (int ele = 0; ele < nEles; ele++)
        iblank_p[0][ele] = geo.iblank_cell(ele);
    }
  }

  /* --- Gather all the data onto Rank 0 for writing --- */

  std::vector<int> nEles_p(input->nRanks);
  MPI_Allgather(&nEles, 1, MPI_INT, nEles_p.data(), 1, MPI_INT, geo.myComm);

  for (int p = 1; p < input->nRanks; p++)
  {
    int nEles = nEles_p[p];
    int size = nEles * nSpts * nVars;

    if (input->rank == 0)
    {
      data_p[p].resize(size);
      MPI_Status status;
      MPI_Recv(data_p[p].data(), size, MPI_DOUBLE, p, 0, geo.myComm, &status);

      if (input->overset)
      {
        iblank_p[p].resize(nEles);
        MPI_Status status2;
        MPI_Recv(iblank_p[p].data(), nEles, MPI_INT, p, 0, geo.myComm, &status2);
      }
    }
    else
    {
      if (p == input->rank)
      {
        mdvector<double> u_tmp({nEles, nVars, nSpts});
        for (int ele = 0; ele < nEles; ele++)
          for (int var = 0; var < nVars; var++)
            for (int spt = 0; spt < nSpts; spt++)
              u_tmp(ele,var,spt) = eles->U_spts(spt,ele,var);

        MPI_Send(u_tmp.data(), size, MPI_DOUBLE, 0, 0, geo.myComm);

        if (input->overset)
        {
          MPI_Send(geo.iblank_cell.data(), nEles, MPI_INT, 0, 0, geo.myComm);
        }
      }
    }

    MPI_Barrier(geo.myComm);
  }

  /* --- Write Data to File (on Rank 0) --- */

  if (input->rank == 0)
  {
    H5File file(filename, H5F_ACC_TRUNC);

    // Write out all the data
    DataSet dset = file.createDataSet("config", string_type, dspace);
    dset.write(config, string_type, dspace);
    dset.close();

    dset = file.createDataSet("stats", string_type, dspace);
    dset.write(stats, string_type, dspace);
    dset.close();

    // Write mesh ID
    dset = file.createDataSet("mesh_uuid", string_type, dspace);
    dset.write(geo.mesh_uuid, string_type, dspace);
    dset.close();

    dspace.close();

    std::string sol_prefix = "soln_";
    sol_prefix += (geo.nDims == 2) ? "quad" : "hex";
    sol_prefix += "_p";
    for (int p = 0; p < input->nRanks; p++)
    {
      nEles = nEles_p[p];
      hsize_t dims[3] = {nSpts, nVars, nEles};
      DataSpace dspaceU(3, dims);

      std::string solname = sol_prefix + std::to_string(p);
      dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceU);
      dset.write(data_p[p].data(), PredType::NATIVE_DOUBLE, dspaceU);

      if (input->overset) // Write out iblank tags as DataSet attribute
      {
        hsize_t dims[1] = {nEles};
        DataSpace dspaceI(1, dims);
        Attribute att = dset.createAttribute("iblank", PredType::NATIVE_INT8, dspaceI);
        att.write(PredType::NATIVE_INT, iblank_p[p].data());
      }

      dspaceU.close();
      dset.close();
    }
  }
#else
  // Need to traspose data to match PyFR layout - [spts,vars,eles] in row-major format
  mdvector<double> u_tmp({nEles, nVars, nSpts});
  for (int ele = 0; ele < nEles; ele++)
    for (int var = 0; var < nVars; var++)
      for (int spt = 0; spt < nSpts; spt++)
        u_tmp(ele,var,spt) = eles->U_spts(spt,ele,var);

  hsize_t dims[3] = {nSpts, nVars, nEles}; /// NOTE: We are col-major, HDF5 is row-major

  /* --- Write Data to File --- */

  H5File file(filename, H5F_ACC_TRUNC);

  DataSet dset = file.createDataSet("config", string_type, dspace);
  dset.write(config, string_type, dspace);
  dset.close();

  dset = file.createDataSet("stats", string_type, dspace);
  dset.write(stats, string_type, dspace);
  dset.close();

  // Write mesh ID
  dset = file.createDataSet("mesh_uuid", string_type, dspace);
  dset.write(geo.mesh_uuid, string_type, dspace);
  dset.close();

  dspace.close();

  DataSpace dspaceU(3, dims);
  std::string solname = "soln_";
  solname += (geo.nDims == 2) ? "quad" : "hex";
  solname += "_p" + std::to_string(input->rank);
  dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceU);
  dset.write(u_tmp.data(), PredType::NATIVE_DOUBLE, dspaceU);
  dset.close();
#endif // no MPI

#endif // no USE_H5_PARALLEL
}

void FRSolver::restart_pyfr(std::string restart_file, unsigned restart_iter)
{
  std::string filename = restart_file;

  if (input->restart_type > 0) // append .pvtu / .vtu to case name
  {
    std::stringstream ss;

    ss << restart_file << "/" << restart_file;

    if (input->overset)
    {
      ss << "-G" << input->gridID;
    }

    ss << "-" << restart_iter;
    ss << ".pyfrs";

    filename = ss.str();
  }
  else
  {
    std::string str = filename;
    size_t ind = str.find("-");
    str.erase(str.begin(), str.begin()+ind+1);
    ind = str.find(".pyfrs");
    str.erase(str.begin()+ind,str.end());
    std::stringstream ss(str);
    ss >> restart_iter;
  }

  current_iter = restart_iter;
  input->iter = restart_iter;
  input->initIter = restart_iter;

  if (input->rank == 0)
    std::cout << "Reading data from file " << filename << std::endl;

  H5File file(filename, H5F_ACC_RDONLY);

  // Read the mesh ID string
  std::string mesh_uuid;

  DataSet dset = file.openDataSet("mesh_uuid");
  DataType dtype = dset.getDataType();
  DataSpace dspace(H5S_SCALAR);

  dset.read(mesh_uuid, dtype, dspace);
  dset.close();

  if (mesh_uuid != geo.mesh_uuid)
    ThrowException("Restart Error - Mesh and solution files do not mesh [mesh_uuid].");

  // Read the config string
  dset = file.openDataSet("config");
  dset.read(geo.config, dtype, dspace);
  dset.close();

  // Read the stats string
  dset = file.openDataSet("stats");
  dset.read(geo.stats, dtype, dspace);
  dset.close();

  // Read the solution data
  std::string solname = "soln_";
  solname += (geo.nDims == 2) ? "quad" : "hex";
  solname += "_p" + std::to_string(input->rank); /// TODO: write per rank in parallel...

  dset = file.openDataSet(solname);
  auto ds = dset.getSpace();

  hsize_t dims[3];
  int ds_rank = ds.getSimpleExtentDims(dims);

  if (ds_rank != 3)
    ThrowException("Improper DataSpace rank for solution data.");

  // Create a datatype for a std::string
  hid_t string_type = H5Tcopy (H5T_C_S1);
  H5Tset_size (string_type, H5T_VARIABLE);

  int nEles = eles->nEles;
  int nVars = eles->nVars;

  if (dims[2] != nEles || dims[1] != nVars)
    ThrowException("Size of solution data set does not match that from mesh.");

  int nSpts = dims[0];

  // Need to traspose data to match PyFR layout - [spts,vars,eles] in row-major format
  mdvector<double> u_tmp({nEles, nVars, nSpts});

  dset.read(u_tmp.data(), PredType::NATIVE_DOUBLE);

  if (input->overset)
  {
    Attribute att = dset.openAttribute("iblank");
    DataSpace dspaceI = att.getSpace();
    hsize_t dim[1];
    dspaceI.getSimpleExtentDims(dim);

    if (dim[0] != geo.nEles)
      ThrowException("Attribute error - expecting size of 'iblank' to be nEles");

    geo.iblank_cell.assign({geo.nEles});
    att.read(PredType::NATIVE_INT, geo.iblank_cell.data());
  }

  dset.close();

  if (nSpts == eles->nSpts)
  {
    eles->U_spts.assign({nSpts,nEles,nVars});
    for (int ele = 0; ele < nEles; ele++)
      for (int var = 0; var < nVars; var++)
        for (int spt = 0; spt < nSpts; spt++)
          eles->U_spts(spt,ele,var) = u_tmp(ele,var,spt);
  }
  else
  {
    int restartOrder = (input->nDims == 2)
                     ? (std::sqrt(nSpts)-1) : (std::cbrt(nSpts)-1);

    // Transpose solution data from PyFR to Zefr layout
    mdvector<double> U_restart({nSpts,nEles,nVars});
    for (int ele = 0; ele < nEles; ele++)
      for (int var = 0; var < nVars; var++)
        for (int spt = 0; spt < nSpts; spt++)
          U_restart(spt,ele,var) = u_tmp(ele,var,spt);

    // Setup extrapolation operator from restart points
    eles->set_oppRestart(restartOrder);


    // Extrapolate values from restart points to solution points
    auto &A = eles->oppRestart(0, 0);
    auto &B = U_restart(0, 0, 0);
    auto &C = eles->U_spts(0, 0, 0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts,
        eles->nEles * eles->nVars, nSpts, 1.0, &A, eles->oppRestart.ldim(), &B,
        U_restart.ldim(), 0.0, &C, eles->U_spts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts,
        eles->nEles * eles->nVars, nSpts, 1.0, &A, eles->oppRestart.ldim(), &B,
        U_restart.ldim(), 0.0, &C, eles->U_spts.ldim());
#endif
  }

  // Process the config / stats string
  std::string str, key, tmp;
  std::stringstream ss;
  std::istringstream stats(geo.stats);
  while (std::getline(stats, str))
  {
    ss.str(str);  ss >> key;
    if (key == "tcurr")
    {
      // "tcurr = ####"
      ss >> tmp >> flow_time;
      break;
    }

    ss.str(""); ss.clear();
  }
}

void FRSolver::write_solution(const std::string &_prefix)
{
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
#endif

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing data to file for case " << prefix << "..." << std::endl;

  if (input->overset) prefix += "_Grid" + std::to_string(input->gridID);

  std::stringstream ss;

#ifdef _MPI
  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_" << std::setw(9) << std::setfill('0');
    ss << iter << ".pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;

    std::vector<std::string> var;
    if (input->equation == AdvDiff || input->equation == Burgers)
    {
      var = {"u"};
    }
    else if (input->equation == EulerNS)
    {
      if (eles->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

    }

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      f << "<PDataArray type=\"Float64\" Name=\"" << var[n] << "\"/>" << std::endl;
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
      f << std::endl;
    }
    if (input->motion)
    {
      f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" format=\"ascii\"/>";
      f << std::endl;
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" />" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
      ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write partition solution to file in binary .vtu format */
  std::ofstream f(outputfile, std::ios::binary);

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" >" << std::endl;

  /* Write comments for solution order, iteration number and flowtime */
  f << "<!-- ORDER " << input->order << " -->" << std::endl;
  f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << iter << " -->" << std::endl;

  if (input->motion)
  {
    eles->update_plot_point_coords();
#ifdef _GPU
    eles->grid_vel_nodes = eles->grid_vel_nodes_d;
#endif
  }

  unsigned int nEles = eles->nEles;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (int ele = 0; ele < eles->nEles; ele++)
      if (geo.iblank_cell(ele) != NORMAL) nEles--;
  }

  unsigned int nCells = 0;
  unsigned int nPts = 0;
  nCells += eles->nSubelements * nEles;
  nPts += eles->nPpts * nEles;

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPts << "\" ";
  f << "NumberOfCells=\"" << nCells << "\">";
  f << std::endl;

  

  size_t b_offset = 0;
  /* Write solution information */
  f << "<PointData>" << std::endl;

  std::vector<std::string> var;
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    var = {"u"};
  }
  else if(input->equation == EulerNS)
  {
    if (eles->nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};
  }

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    f << "<DataArray type=\"Float64\" Name=\"" << var[n] << "\" ";
    f << "format=\"appended\" offset=\"" << b_offset << "\"/>"<< std::endl;
    b_offset += (nEles * eles->nPpts * sizeof(double) + sizeof(unsigned int));
  }

  if (input->filt_on && input->sen_write)
  {
#ifdef _GPU
    filt.sensor = filt.sensor_d;
#endif
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << filt.sensor(ele) << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  if (input->motion)
  {
    eles->get_grid_velocity_ppts();

    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        for (unsigned int dim = 0; dim < eles->nDims; dim++)
        {
          f << std::scientific << std::setprecision(16);
          f << eles->grid_vel_ppts(ppt, ele, dim);
          f  << " ";
        }
        if (eles->nDims == 2) f << 0.0 << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }
  f << "</PointData>" << std::endl;

  /* Write plot point information (single precision) */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  f << "</Points>" << std::endl;
  b_offset += (nEles * eles->nPpts * 3 * sizeof(float) + sizeof(unsigned int));

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += (nEles * eles->nSubelements * eles->nNodesPerSubelement * sizeof(unsigned int) + sizeof(unsigned int));



  f << "<DataArray type=\"UInt32\" Name=\"offsets\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += (nEles * eles->nSubelements * sizeof(unsigned int) + sizeof(unsigned int));

  f << "<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += (nEles * eles->nSubelements * sizeof(char) + sizeof(unsigned int));
  f << "</Cells>" << std::endl;

  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;

  /* Adding raw binary data as AppendedData*/
  f << "<AppendedData encoding=\"raw\">" << std::endl;
  f << "_"; // leading underscore

  /* Write solution data */
  /* Extrapolate solution to plot points */
  auto &A = eles->oppE_ppts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_ppts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, else->oppE_ppts.ldim(), &B, 
      eles->U_spts.ldim(), 0.0, &C, eles->U_ppts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->oppE_ppts.ldim(), &B, 
      eles->U_spts.ldim(), 0.0, &C, eles->U_ppts.ldim());
#endif

  /* Apply squeezing if needed */
  if (input->squeeze)
  {
    eles->compute_Uavg();

#ifdef _GPU
    eles->Uavg = eles->Uavg_d;
#endif

    eles->poly_squeeze_ppts();
  }

  unsigned int nBytes = nEles * eles->nPpts * sizeof(double);

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    binary_write(f, nBytes);
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        binary_write(f, eles->U_ppts(ppt, ele, n));
      }
    }
  }

  /* Write plot point coordinates */
  nBytes = nEles * eles->nPpts * 3 * sizeof(float);
  binary_write(f, nBytes);
  double dzero = 0.0;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
    {
      binary_write(f, (float) eles->coord_ppts(ppt, ele, 0));
      binary_write(f, (float) eles->coord_ppts(ppt, ele, 1));
      if (geo.nDims == 2)
        binary_write(f, 0.0f);
      else
        binary_write(f, (float) eles->coord_ppts(ppt, ele, 2));
    }
  }

  /* Write cell information */
  // Write connectivity
  nBytes = nEles * eles->nSubelements * eles->nNodesPerSubelement * sizeof(unsigned int);
  binary_write(f, nBytes);
  int shift = 0; // To account for blanked elements
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        binary_write(f, eles->ppt_connect(i, subele) + shift);
      }
    }
    shift += eles->nPpts;
  }

  // Offsets
  nBytes = nEles * eles->nSubelements * sizeof(unsigned int);
  binary_write(f, nBytes);
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      binary_write(f, offset);
      offset += eles->nNodesPerSubelement;
    }
  }

  // Types
  nBytes = nEles * eles->nSubelements * sizeof(char);
  binary_write(f, nBytes);
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      if (eles->etype == QUAD)
        binary_write(f, (char) 9);
      else if (eles->etype == TRI)
        binary_write(f, (char) 5);
      else if (eles->etype == HEX)
        binary_write(f, (char) 12);
    }
  }

  f << std::endl;
  f << "</AppendedData>" << std::endl;

  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::write_color()
{
  if (input->rank == 0) std::cout << "Writing colors to file..." << std::endl;

  std::stringstream ss;
#ifdef _MPI

  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << input->output_prefix << "_color.pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;
    f << "<PDataArray type=\"Int32\" Name=\"color\" format=\"ascii\"/>";
    f << std::endl;
    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;



    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << input->output_prefix << "_color_"; 
      ss << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_color_";
  ss << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_color";
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write partition color to file in .vtu format */
  std::ofstream f(outputfile);

  int nEles = eles->nEles;
  if (input->overset)
  {
    /* Remove blanked elements from total element count */
    for (int ele = 0; ele < eles->nEles; ele++)
      if (geo.iblank_cell(ele) != NORMAL) nEles--;
  }

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;
  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << eles->nPpts * nEles << "\" ";
  f << "NumberOfCells=\"" << eles->nSubelements * nEles << "\">";
  f << std::endl;
  
  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl; 

  if (eles->nDims == 2)
  {
    // TODO: Change order of ppt structures for better looping 
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << eles->coord_ppts(ppt, ele, 0) << " ";
        f << eles->coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << eles->coord_ppts(ppt, ele, 0) << " ";
        f << eles->coord_ppts(ppt, ele, 1) << " ";
        f << eles->coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }
  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  int count = 0;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << eles->ppt_connect(i, subele) + count*eles->nPpts << " ";
      }
      f << std::endl;
    }
    count++;
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      f << offset << " ";
      offset += eles->nNodesPerSubelement;
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int nCells = eles->nSubelements * nEles;
  if (eles->nDims == 2)
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 9 << " ";
  }
  else
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 12 << " ";
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write color information */
  f << "<PointData>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"color\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
    for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
    {
      f << std::scientific << std::setprecision(16) << geo.ele_color(ele);
      f  << " ";
    }
    f << std::endl;
  }
  f << "</DataArray>" << std::endl;
  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::write_overset_boundary(const std::string &_prefix)
{
  if (!input->overset) ThrowException("Overset surface export must have overset grid.");

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing overset boundary surface data to " << prefix << "..." << std::endl;

  prefix += "_Grid" + std::to_string(input->gridID);

  // Prep the index lists [to grab data from a face of an ele]
  int nDims = geo.nDims;
  int nPts1D = order+3;
  int nPtsFace = nPts1D;
  if (nDims==3) nPtsFace *= nPts1D;
  int nSubCells = nPts1D - 1;
  if (nDims==3) nSubCells *= nSubCells;
  int nFacesEle = geo.nFacesPerEle;

  mdvector<int> index_map({nFacesEle, nPtsFace});

  if (nDims == 2)
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                    // Bottom
      index_map(1,j) = nPts1D-1 + j*nPts1D;        // Right
      index_map(2,j) = nPts1D*nPts1D - 1 + j*(-1); // Top
      index_map(3,j) = 0 + j*nPts1D;               // Left
    }
  }
  else
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                      // Zmin / Bottom
      index_map(1,j) = nPts1D*nPtsFace - 1 + j*(-1); // Zmax / Top
      index_map(2,j) = 0 + j*nPts1D;                 // Xmin / Left
      index_map(3,j) = nPts1D - 1 + j*nPts1D;        // Xmax / Right
    }

    // Ymin / Front
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + j1*nPtsFace;
        index_map(4,J) = J2;
      }
    }

    // Ymax / Back
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + (j1+1)*nPtsFace - nPts1D;
        index_map(5,J) = J2;
      }
    }
  }

  // Write the ParaView file for each Gmsh boundary
  std::stringstream ss;

#ifdef _MPI
  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0');
    ss << iter << ".pvtu";

    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;
    if (input->equation == AdvDiff || input->equation == Burgers)
    {
      f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
      f << std::endl;
    }
    else if (input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (eles->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        f << "<PDataArray type=\"Float32\" Name=\"" << var[n];
        f << "\" format=\"ascii\"/>";
        f << std::endl;
      }
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
      f << std::endl;
    }

    if (input->motion)
    {
      f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" format=\"ascii\"/>";
      f << std::endl;
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    {
      ss.str("");
      ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
      ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write parition solution to file in .vtu format */
  std::ofstream f(outputfile);

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

  /* Write comments for solution order, iteration number and flowtime */
  f << "<!-- ORDER " << input->order << " -->" << std::endl;
  f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << iter << " -->" << std::endl;

  // Load list of eles (and sub-ele face indices) from which to load data
  std::vector<int> eleList, indList;
  unsigned int nFaces = 0;

  for (unsigned int ff = 0; ff < geo.nFaces; ff++)
  {
    if (geo.iblank_face[ff] == FRINGE)
    {
      int ic1 = geo.face2eles(0,ff);
      int ic2 = geo.face2eles(1,ff);
      if (geo.iblank_cell(ic1) == NORMAL)
      {
        eleList.push_back(ic1);
      }
      else if (ic2 > 0 && geo.iblank_cell(ic2) == NORMAL)
      {
        eleList.push_back(ic2);
      }

      for (int i = 0; i < nFacesEle; i++)
      {
        if (geo.ele2face(i,eleList.back()) == ff)
        {
          indList.push_back(i);
          break;
        }
      }

      nFaces++;
    }
  }

  // Write data to file

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPtsFace * nFaces << "\" ";
  f << "NumberOfCells=\"" << nSubCells * nFaces << "\">";
  f << std::endl;


  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl;

  if (eles->nDims == 2)
  {
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << eles->coord_ppts(ppt, ele, 0) << " ";
        f << eles->coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << eles->coord_ppts(ppt, ele, 0) << " ";
        f << eles->coord_ppts(ppt, ele, 1) << " ";
        f << eles->coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }

  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;

  if (nDims == 2)
  {
    int count = 0;
    for (int face = 0; face < nFaces; face++)
    {
      for (int j = 0; j < nPts1D-1; j++)
      {
        f << count + j << " ";
        f << count + j + 1 << " ";
        f << endl;
      }
      count += nPtsFace;
    }
  }
  else
  {
    int count = 0;
    for (int face = 0; face < nFaces; face++)
    {
      for (int j = 0; j < nPts1D-1; j++)
      {
        for (int i = 0; i < nPts1D-1; i++)
        {
          f << count + j*nPts1D     + i   << " ";
          f << count + j*nPts1D     + i+1 << " ";
          f << count + (j+1)*nPts1D + i+1 << " ";
          f << count + (j+1)*nPts1D + i   << " ";
          f << endl;
        }
      }
      count += nPtsFace;
    }
  }

  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  int nvPerFace = (nDims == 2) ? 2 : 4;
  int offset = nvPerFace;
  for (int face = 0; face < nFaces; face++)
  {
    for (int subele = 0; subele < nSubCells; subele++)
    {
      f << offset << " ";
      offset += nvPerFace;
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  int nCells = nSubCells * nFaces;
  if (nDims == 2)
  {
    for (int cell = 0; cell < nCells; cell++)
      f << 3 << " ";
  }
  else
  {
    for (int cell = 0; cell < nCells; cell++)
      f << 9 << " ";
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write solution information */
  f << "<PointData>" << std::endl;

  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    f << "<DataArray type=\"Float32\" Name=\"u\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << std::scientific << std::setprecision(16) << eles->U_ppts(ppt, ele, 0);
        f  << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }
  else if(input->equation == EulerNS)
  {
    std::vector<std::string> var;
    if (eles->nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};

    for (int n = 0; n < eles->nVars; n++)
    {
      f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
      f << "format=\"ascii\">"<< std::endl;

      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << std::scientific << std::setprecision(16);
          f << eles->U_ppts(ppt, ele, n);
          f << " ";
        }

        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
  }

  if (input->filt_on && input->sen_write)
  {
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        f << filt.sensor(ele) << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  if (input->motion)
  {
    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (unsigned int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (unsigned int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        for (unsigned int dim = 0; dim < eles->nDims; dim++)
        {
          f << std::scientific << std::setprecision(16);
          f << eles->grid_vel_ppts(ppt, ele, dim);
          f  << " ";
        }
        if (eles->nDims == 2) f << 0.0 << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}


void FRSolver::write_surfaces(const std::string &_prefix)
{
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
#endif

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing surface data to " << prefix << "..." << std::endl;

  if (input->overset) prefix += "_Grid" + std::to_string(input->gridID);

  // Prep the index lists [to grab data from a face of an ele]
  int nDims = geo.nDims;
  int nPts1D = order+1;
  int nPtsFace = nPts1D;
  if (nDims==3) nPtsFace *= nPts1D;
  int nSubCells = nPts1D - 1;
  if (nDims==3) nSubCells *= nSubCells;
  int nFacesEle = geo.nFacesPerEle;

  mdvector<int> index_map({nFacesEle, nPtsFace});

  if (nDims == 2)
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                    // Bottom
      index_map(1,j) = nPts1D-1 + j*nPts1D;        // Right
      index_map(2,j) = nPts1D*nPts1D - 1 + j*(-1); // Top
      index_map(3,j) = 0 + j*nPts1D;               // Left
    }
  }
  else
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                      // Zmin / Bottom
      index_map(1,j) = nPts1D*nPtsFace - 1 + j*(-1); // Zmax / Top
      index_map(2,j) = 0 + j*nPts1D;                 // Xmin / Left
      index_map(3,j) = nPts1D - 1 + j*nPts1D;        // Xmax / Right
    }

    // Ymin / Front
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + j1*nPtsFace;
        index_map(4,J) = J2;
      }
    }

    // Ymax / Back
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + (j1+1)*nPtsFace - nPts1D;
        index_map(5,J) = J2;
      }
    }
  }

  // General Solution Preprocessing Stuff

  if (input->motion)
  {
    eles->update_plot_point_coords();
#ifdef _GPU
    eles->grid_vel_nodes = eles->grid_vel_nodes_d;
#endif
    eles->get_grid_velocity_ppts();
  }

  /* Extrapolate solution to plot points */
  auto &A = eles->oppE_ppts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_ppts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts,
                    eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->oppE_ppts.ldim(), &B,
                    eles->U_spts.ldim(), 0.0, &C, eles->U_ppts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts,
              eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->oppE_ppts.ldim(), &B,
              eles->U_spts.ldim(), 0.0, &C, eles->U_ppts.ldim());
#endif

  /* Apply squeezing if needed */
  if (input->squeeze)
  {
    eles->compute_Uavg();

#ifdef _GPU
    eles->Uavg = eles->Uavg_d;
#endif

    eles->poly_squeeze_ppts();
  }

#ifdef _GPU
  if (input->filt_on && input->sen_write)
    filt.sensor = filt.sensor_d;
#endif

  // Write the ParaView file for each Gmsh boundary
  for (int bnd = 0; bnd < geo.nBounds; bnd++)
  {
    std::stringstream ss;

#ifdef _MPI
    /* Write .pvtu file on rank 0 if running in parallel */
    if (input->rank == 0)
    {
      ss << input->output_prefix << "/";
      ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0');
      ss << iter << ".pvtu";

      std::ofstream f(ss.str());
      f << "<?xml version=\"1.0\"?>" << std::endl;
      f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
      f << "byte_order=\"LittleEndian\" ";
      f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

      f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
      f << "<PPointData>" << std::endl;
      if (input->equation == AdvDiff || input->equation == Burgers)
      {
        f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
        f << std::endl;
      }
      else if (input->equation == EulerNS)
      {
        std::vector<std::string> var;
        if (eles->nDims == 2)
          var = {"rho", "xmom", "ymom", "energy"};
        else
          var = {"rho", "xmom", "ymom", "zmom", "energy"};

        for (unsigned int n = 0; n < eles->nVars; n++)
        {
          f << "<PDataArray type=\"Float32\" Name=\"" << var[n];
          f << "\" format=\"ascii\"/>";
          f << std::endl;
        }
      }

      if (input->filt_on && input->sen_write)
      {
        f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
        f << std::endl;
      }

      if (input->motion)
      {
        f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" format=\"ascii\"/>";
        f << std::endl;
      }

      f << "</PPointData>" << std::endl;
      f << "<PPoints>" << std::endl;
      f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
      f << "format=\"ascii\"/>" << std::endl;
      f << "</PPoints>" << std::endl;

      for (unsigned int n = 0; n < input->nRanks; n++)
      {
        ss.str("");
        ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
        ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
        f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
      }

      f << "</PUnstructuredGrid>" << std::endl;
      f << "</VTKFile>" << std::endl;

      f.close();
    }
#endif

    ss.str("");
#ifdef _MPI
    ss << input->output_prefix << "/";
    ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
    ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
    ss << input->output_prefix << "/";
    ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
    ss << ".vtu";
#endif

    auto outputfile = ss.str();

    /* Write parition solution to file in .vtu format */
    std::ofstream f(outputfile);

    /* Write header */
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    /* Write comments for solution order, iteration number and flowtime */
    f << "<!-- ORDER " << input->order << " -->" << std::endl;
    f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
    f << "<!-- ITER " << iter << " -->" << std::endl;

    // Load list of eles (and sub-ele face indices) from which to load data
    std::vector<int> eleList, indList;
    unsigned int nFaces = 0;

    for (auto &ff : geo.boundFaces[bnd])
    {
      // Load data from each face on boundary
      int ele = geo.face2eles(0,ff);

      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      int j = -1;
      for (int i = 0; i < nFacesEle; i++)
      {
        if (geo.ele2face(i, ele) == ff)
        {
          j = i;
          break;
        }
      }
      if (j < 0) ThrowException("write_surfaces: Error with ele/face connectivity!");

      eleList.push_back(ele);
      indList.push_back(j);
      nFaces++;
    }

    // Write data to file

    f << "<UnstructuredGrid>" << std::endl;
    f << "<Piece NumberOfPoints=\"" << nPtsFace * nFaces << "\" ";
    f << "NumberOfCells=\"" << nSubCells * nFaces << "\">";
    f << std::endl;


    /* Write plot point coordinates */
    f << "<Points>" << std::endl;
    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\">" << std::endl;

    if (eles->nDims == 2)
    {
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << eles->coord_ppts(ppt, ele, 0) << " ";
          f << eles->coord_ppts(ppt, ele, 1) << " ";
          f << 0.0 << std::endl;
        }
      }
    }
    else
    {
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << eles->coord_ppts(ppt, ele, 0) << " ";
          f << eles->coord_ppts(ppt, ele, 1) << " ";
          f << eles->coord_ppts(ppt, ele, 2) << std::endl;
        }
      }
    }

    f << "</DataArray>" << std::endl;
    f << "</Points>" << std::endl;

    /* Write cell information */
    f << "<Cells>" << std::endl;
    f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
    f << "format=\"ascii\">"<< std::endl;

    if (nDims == 2)
    {
      int count = 0;
      for (int face = 0; face < nFaces; face++)
      {
        for (int j = 0; j < nPts1D-1; j++)
        {
          f << count + j << " ";
          f << count + j + 1 << " ";
          f << endl;
        }
        count += nPtsFace;
      }
    }
    else
    {
      int count = 0;
      for (int face = 0; face < nFaces; face++)
      {
        for (int j = 0; j < nPts1D-1; j++)
        {
          for (int i = 0; i < nPts1D-1; i++)
          {
            f << count + j*nPts1D     + i   << " ";
            f << count + j*nPts1D     + i+1 << " ";
            f << count + (j+1)*nPts1D + i+1 << " ";
            f << count + (j+1)*nPts1D + i   << " ";
            f << endl;
          }
        }
        count += nPtsFace;
      }
    }

    f << "</DataArray>" << std::endl;

    f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
    f << "format=\"ascii\">"<< std::endl;
    int nvPerFace = (nDims == 2) ? 2 : 4;
    int offset = nvPerFace;
    for (int face = 0; face < nFaces; face++)
    {
      for (int subele = 0; subele < nSubCells; subele++)
      {
        f << offset << " ";
        offset += nvPerFace;
      }
    }
    f << std::endl;
    f << "</DataArray>" << std::endl;

    f << "<DataArray type=\"UInt8\" Name=\"types\" ";
    f << "format=\"ascii\">"<< std::endl;
    int nCells = nSubCells * nFaces;
    if (nDims == 2)
    {
      for (int cell = 0; cell < nCells; cell++)
        f << 3 << " ";
    }
    else
    {
      for (int cell = 0; cell < nCells; cell++)
        f << 9 << " ";
    }
    f << std::endl;
    f << "</DataArray>" << std::endl;
    f << "</Cells>" << std::endl;

    /* Write solution information */
    f << "<PointData>" << std::endl;

    if (input->equation == AdvDiff || input->equation == Burgers)
    {
      f << "<DataArray type=\"Float32\" Name=\"u\" ";
      f << "format=\"ascii\">"<< std::endl;
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << std::scientific << std::setprecision(16) << eles->U_ppts(ppt, ele, 0);
          f  << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
    else if(input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (eles->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (int n = 0; n < eles->nVars; n++)
      {
        f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
        f << "format=\"ascii\">"<< std::endl;

        for (int face = 0; face < nFaces; face++)
        {
          int ele = eleList[face];
          int ind = indList[face];
          for (int pt = 0; pt < nPtsFace; pt++)
          {
            int ppt = index_map(ind,pt);
            f << std::scientific << std::setprecision(16);
            f << eles->U_ppts(ppt, ele, n);
            f << " ";
          }

          f << std::endl;
        }
        f << "</DataArray>" << std::endl;
      }
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          f << filt.sensor(ele) << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }

    if (input->motion)
    {
      f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
      f << "format=\"ascii\">"<< std::endl;
      for (unsigned int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (unsigned int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          for (unsigned int dim = 0; dim < eles->nDims; dim++)
          {
            f << std::scientific << std::setprecision(16);
            f << eles->grid_vel_ppts(ppt, ele, dim);
            f  << " ";
          }
          if (eles->nDims == 2) f << 0.0 << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }

    f << "</PointData>" << std::endl;
    f << "</Piece>" << std::endl;
    f << "</UnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;
    f.close();
  }

  if (input->overset && input->plot_overset)
  {
    write_overset_boundary(_prefix);
  }
}

void FRSolver::write_LHS(const std::string &_prefix)
{
#if !defined (_MPI) && !defined (_GPU)
  if (input->dt_scheme == "MCGS" && !input->stream_mode)
  {
    if (input->rank == 0) std::cout << "Writing LHS to file..." << std::endl;

    unsigned int iter = current_iter;
    if (input->p_multi)
      iter = iter / input->mg_steps[0];

    std::string prefix = _prefix;
    std::stringstream ss;
    ss << input->output_prefix << "/";
    ss << prefix << "_LHS_" << std::setw(9) << std::setfill('0');
    ss << iter << ".dat";

    H5File file(ss.str(), H5F_ACC_TRUNC);
    unsigned int N = eles->nSpts * eles->nVars;
    hsize_t dims[3] = {geo.nEles, N, N};
    DataSpace dspaceU(3, dims);

    std::string name = "LHS_" + std::to_string(iter);
    DataSet dset = file.createDataSet(name, PredType::NATIVE_DOUBLE, dspaceU);
    dset.write(eles->LHSs[0].data(), PredType::NATIVE_DOUBLE, dspaceU);

    dspaceU.close();
    dset.close();
    file.close();
  }
#endif
}

void FRSolver::report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If running on GPU, copy out divergence */
#ifdef _GPU
  eles->divF_spts = eles->divF_spts_d;
  dt = dt_d;
#endif

  // HACK: Change nStages to compute the correct residual
  if (input->dt_scheme == "MCGS")
  {
    nStages = 1;
  }

  std::vector<double> res(eles->nVars,0.0);

  unsigned int nEles = 0;
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        if (input->res_type == 0)
          res[n] = std::max(res[n], std::abs(eles->divF_spts(spt,ele,n,0)
                                             / eles->jaco_det_spts(spt, ele)));

        else if (input->res_type == 1)
          res[n] += std::abs(eles->divF_spts(spt,ele,n,0)
                             / eles->jaco_det_spts(spt, ele));

        else if (input->res_type == 2)
          res[n] += eles->divF_spts(spt,ele,n,0) * eles->divF_spts(spt,ele,n,0)
                  / (eles->jaco_det_spts(spt, ele) * eles->jaco_det_spts(spt, ele));
      }
      nEles++;
    }
  }

  unsigned int nDoF =  (eles->nSpts * nEles);

  // HACK: Change nStages back
  if (input->dt_scheme == "MCGS")
  {
    nStages = 2;
  }

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), eles->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), eles->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
#endif

  /* Print residual to terminal (normalized by number of solution points) */
  if (input->rank == 0) 
  {
    if (input->res_type == 2)
    {
      for (auto &val : res)  
        val = std::sqrt(val);
    }

    if (input->overset)
      std::cout << "G" << std::setw(4) << std::left << input->gridID;

    std::cout << std::setw(6) << std::left << iter << " ";

    for (auto val : res)
      std::cout << std::scientific << std::setprecision(6) << std::setw(15) << std::left << val / nDoF << " ";

    if (input->dt_type == 2)
    {
      std::cout << "dt: " <<  *std::min_element(dt.data(), dt.data()+eles->nEles) << " (min) ";
      std::cout << *std::max_element(dt.data(), dt.data()+eles->nEles) << " (max)";
    }
    else
    {
      std::cout << "dt: " << dt(0);
    }

    std::cout << std::endl;
    
    /* Write to history file */
    auto t2 = std::chrono::high_resolution_clock::now();
    auto current_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);

    f << iter << " " << std::scientific << flow_time << " " << current_runtime.count() << " ";

    for (auto val : res)
      f << val / nDoF << " ";
    f << std::endl;

    /* Store maximum residual */
    res_max = res[0] / nDoF;
  }

#ifdef _MPI
  /* Broadcast maximum residual */
  MPI_Bcast(&res_max, 1, MPI_DOUBLE, 0, myComm);
#endif
}

void FRSolver::report_forces(std::ofstream &f)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If using GPU, copy out solution, gradient and pressure */
#ifdef _GPU
  eles->U_fpts = eles->U_fpts_d;
  eles->dU_fpts = eles->dU_fpts_d;
  faces->P = faces->P_d;
#endif

  std::stringstream ss;
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".cp";
#else
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".cp";
#endif

  auto cpfile = ss.str();
  std::ofstream g(cpfile);

  std::array<double, 3> force_conv = {0,0,0};
  std::array<double, 3> force_visc = {0,0,0};
  compute_forces(force_conv, force_visc, &g);

  /* Convert dimensional forces into non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  for (int i = 0; i < 3; i++)
  {
    force_conv[i] *= fac;
    force_visc[i] *= fac;
  }

  /* Compute lift and drag coefficients */
  double CL_conv, CD_conv, CL_visc, CD_visc;

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(force_conv.data(), force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(force_visc.data(), force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
#endif

  /* Get angle of attack (and sideslip) */
  double aoa = std::atan2(input->V_fs(1), input->V_fs(0));
  double aos = 0.0;
  if (eles->nDims == 3)
    aos = std::atan2(input->V_fs(2), input->V_fs(0));

  if (input->rank == 0)
  {
    if (eles->nDims == 2)
    {
      CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
      CD_conv = force_conv[0] * std::cos(aoa) + force_conv[1] * std::sin(aoa);
      CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
      CD_visc = force_visc[0] * std::cos(aoa) + force_visc[1] * std::sin(aoa);
    }
    else if (eles->nDims == 3)
    {
      CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
      CD_conv = force_conv[0] * std::cos(aoa) * std::cos(aos) + force_conv[1] * std::sin(aoa) + 
        force_conv[2] * std::sin(aoa) * std::cos(aos);
      CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
      CD_visc = force_visc[0] * std::cos(aoa) * std::cos(aos) + force_visc[1] * std::sin(aoa) + 
        force_visc[2] * std::sin(aoa) * cos(aos);
    }

    std::cout << "CL_conv = " << CL_conv << " CD_conv = " << CD_conv;

    f << iter << " " << std::scientific << std::setprecision(16) << flow_time << " ";

    f  << CL_conv << " " << CD_conv;

    if (input->viscous)
    {
      std::cout << " CL_visc = " << CL_visc << " CD_visc = " << CD_visc;
      f << " " << CL_visc << " " << CD_visc;
    }

    std::cout << std::endl;
    f << std::endl;
  }
}

void FRSolver::report_error(std::ofstream &f)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If using GPU, copy out solution */
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
  if (input->viscous)
  {
    eles->dU_spts = eles->dU_spts_d;
  }
#endif

  /* Extrapolate solution to quadrature points */
  auto &A = eles->oppE_qpts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_qpts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->oppE_qpts.ldim(), &B, 
      eles->U_spts.ldim(), 0.0, &C, eles->U_qpts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->U_qpts.ldim(), &B, 
      eles->U_spts.ldim(), 0.0, &C, eles->U_qpts.ldim());
#endif

  /* Extrapolate derivatives to quadrature points */
  if (input->viscous)
  {
    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      auto &A = eles->oppE_qpts(0, 0);
      auto &B = eles->dU_spts(0, 0, 0, dim);
      auto &C = eles->dU_qpts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts,
                        eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->U_qpts.ldim(), &B,
                        eles->U_spts.ldim(), 0.0, &C, eles->U_qpts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts,
                  eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->U_qpts.ldim(), &B,
                  eles->U_spts.ldim(), 0.0, &C, eles->U_qpts.ldim());
#endif

    }
  }

  std::vector<double> l2_error(2,0.0);
  double vol = 0;

  unsigned int n = input->err_field;
  std::vector<double> dU_true(2, 0.0), dU_error(2, 0.0);
#pragma omp for 
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int qpt = 0; qpt < eles->nQpts; qpt++)
      {
        double U_true = 0.0;
        double weight = 0.0;

        if (eles->nDims == 2)
        {
          /* Compute true solution and derivatives */
          if (input->test_case == 3) // Isentropic Bump
          {
            U_true = input->P_fs / std::pow(input->rho_fs, input->gamma);
          }
          else 
          {
            U_true = compute_U_true(eles->coord_qpts(qpt,ele,0), eles->coord_qpts(qpt,ele,1), 0, 
                flow_time, n, input);
          }

          if (input->viscous)
          {
            dU_true[0] = compute_dU_true(eles->coord_qpts(qpt,ele,0), eles->coord_qpts(qpt,ele,1), 0,
                                         flow_time, n, 0, input);
            dU_true[1] = compute_dU_true(eles->coord_qpts(qpt,ele,0), eles->coord_qpts(qpt,ele,1), 0,
                                         flow_time, n, 1, input);
          }

          /* Get quadrature point index and weight */
          unsigned int i = eles->idx_qpts(qpt,0);
          unsigned int j = eles->idx_qpts(qpt,1);
          weight = eles->weights_qpts[i] * eles->weights_qpts[j];
        }
        else if (eles->nDims == 3)
        {
          ThrowException("Under construction!");
        }

        /* Compute errors */
        double U_error;
        if (input->test_case == 2) // Couette flow case
        {
          if (!input->viscous) 
            ThrowException("Couette flow test case selected but viscosity disabled.");
          
          double rho = eles->U_qpts(qpt, ele, 0);
          double u =  eles->U_qpts(qpt, ele, 1) / rho;
          double rho_dx = eles->dU_qpts(qpt, ele, 0, 0);
          double rho_dy = eles->dU_qpts(qpt, ele, 0, 1);
          double momx_dx = eles->dU_qpts(qpt, ele, 1, 0);
          double momx_dy = eles->dU_qpts(qpt, ele, 1, 1);

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          U_error = U_true - u;
          dU_error[0] = dU_true[0] - du_dx;
          dU_error[1] = dU_true[1] - du_dy;
          vol = 1;
        }
        else if (input->test_case == 3) // Isentropic bump
        {
          double momF = 0.0;
          for (unsigned int dim = 0; dim < eles->nDims; dim ++)
          {
            momF += eles->U_qpts(qpt, ele, dim + 1) * eles->U_qpts(qpt, ele, dim + 1);
          }

          momF /= eles->U_qpts(qpt, ele, 0);

          double P = (input->gamma - 1.0) * (eles->U_qpts(qpt, ele, 3) - 0.5 * momF);

          U_error = (U_true - P/std::pow(eles->U_qpts(qpt, ele, 0), input->gamma)) / U_true;
          vol += weight * eles->jaco_det_qpts(qpt, ele); 
        }
        else
        {
          U_error = U_true - eles->U_qpts(qpt, ele, n);
          if (input->viscous)
          {
            dU_error[0] = dU_true[0] - eles->dU_qpts(qpt, ele, n, 0); 
            dU_error[1] = dU_true[1] - eles->dU_qpts(qpt, ele, n, 1);
          }
          vol = 1;
        }

        l2_error[0] += weight * eles->jaco_det_qpts(qpt, ele) * U_error * U_error; 
        l2_error[1] += weight * eles->jaco_det_qpts(qpt, ele) * (U_error * U_error +
            dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1]); 
      }
  }

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(l2_error.data(), l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }

#endif


  /* Print to terminal */
  if (input->rank == 0)
  {
    std::cout << "l2_error: ";
    for (auto &val : l2_error)
      std::cout << std::scientific << std::sqrt(val / vol) << " ";
    std::cout << std::endl;

    /* Write to file */
    f << iter << " " << std::scientific << std::setprecision(16) << flow_time << " ";

    for (auto &val : l2_error)
      f << std::sqrt(val / vol) << " ";
    f << std::endl;
  }

}

void FRSolver::compute_forces(std::array<double,3> &force_conv, std::array<double,3> &force_visc, std::ofstream *cp_file = NULL)
{
  double taun[3] = {0,0,0};

  /* Factor for forming non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  bool write_cp = (cp_file != NULL && cp_file->is_open());

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);
    unsigned int idx = count % geo.nFptsPerFace;

    if (bnd_id == SLIP_WALL_P || bnd_id == SLIP_WALL_G || bnd_id == ISOTHERMAL_NOSLIP_P  || bnd_id == ISOTHERMAL_NOSLIP_G
        || bnd_id == ISOTHERMAL_NOSLIP_MOVING_P || bnd_id == ISOTHERMAL_NOSLIP_MOVING_G || bnd_id == ADIABATIC_NOSLIP_P
        || bnd_id == ADIABATIC_NOSLIP_G || bnd_id == ADIABATIC_NOSLIP_MOVING_P || bnd_id == ADIABATIC_NOSLIP_MOVING_G) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(fpt, 0);

      if (write_cp)
      {
        /* Write CP distrubtion to file */
        double CP = (PL - input->P_fs) * fac;
        for(unsigned int dim = 0; dim < eles->nDims; dim++)
          *cp_file << std::scientific << faces->coord(fpt, dim) << " ";
        *cp_file << std::scientific << CP << std::endl;
      }

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < eles->nDims; dim++)
      {
        force_conv[dim] += eles->weights_fpts(idx) * PL *
          faces->norm(fpt, dim) * faces->dA(fpt);
      }

      if (input->viscous)
      {
        if (eles->nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double e = faces->U(fpt, 3, 0);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);

          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);

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
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0) + tauxy * faces->norm(fpt, 1);
          taun[1] = tauxy * faces->norm(fpt, 0) + tauyy * faces->norm(fpt, 1);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_fpts(idx) * taun[dim] *
              faces->dA(fpt);

        }
        else if (eles->nDims == 3)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double momz = faces->U(fpt, 3, 0);
          double e = faces->U(fpt, 4, 0);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

           /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);
          double momz_dx = faces->dU(fpt, 3, 0, 0);

          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);
          double momz_dy = faces->dU(fpt, 3, 1, 0);

          double rho_dz = faces->dU(fpt, 0, 2, 0);
          double momx_dz = faces->dU(fpt, 1, 2, 0);
          double momy_dz = faces->dU(fpt, 2, 2, 0);
          double momz_dz = faces->dU(fpt, 3, 2, 0);

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
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
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

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0) + tauxy * faces->norm(fpt, 1) + tauxz * faces->norm(fpt, 2);
          taun[1] = tauxy * faces->norm(fpt, 0) + tauyy * faces->norm(fpt, 1) + tauyz * faces->norm(fpt, 2);
          taun[3] = tauxz * faces->norm(fpt, 0) + tauyz * faces->norm(fpt, 1) + tauzz * faces->norm(fpt, 2);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_fpts(idx) * taun[dim] *
              faces->dA(fpt);
        }

      }
      count++;
    }
  }
}

void FRSolver::compute_moments(std::array<double,3> &tot_force, std::array<double,3> &tot_moment)
{
  /*! ---- TAKING ALL MOMENTS ABOUT THE ORIGIN ---- */
  tot_force.fill(0.0);
  tot_moment.fill(0.0);

  double taun[3];
  double force[3] = {0,0,0}; //! NOTE: need z-component initialized to '0' for 2D

  int c1[3] = {1,2,0}; // Cross-product index maps
  int c2[3] = {2,0,1};

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);
    unsigned int idx = count % geo.nFptsPerFace;

    if (bnd_id == SLIP_WALL_P || bnd_id == SLIP_WALL_G || bnd_id == ISOTHERMAL_NOSLIP_P  || bnd_id == ISOTHERMAL_NOSLIP_G
        || bnd_id == ISOTHERMAL_NOSLIP_MOVING_P || bnd_id == ISOTHERMAL_NOSLIP_MOVING_G || bnd_id == ADIABATIC_NOSLIP_P
        || bnd_id == ADIABATIC_NOSLIP_G || bnd_id == ADIABATIC_NOSLIP_MOVING_P || bnd_id == ADIABATIC_NOSLIP_MOVING_G) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(fpt, 0);

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < eles->nDims; dim++)
        force[dim] = eles->weights_fpts(idx) * PL *
          faces->norm(fpt, dim) * faces->dA(fpt);

      if (input->viscous)
      {
        if (eles->nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double e = faces->U(fpt, 3, 0);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);

          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);

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
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0) + tauxy * faces->norm(fpt, 1);
          taun[1] = tauxy * faces->norm(fpt, 0) + tauyy * faces->norm(fpt, 1);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force[dim] -= eles->weights_fpts(idx) * taun[dim] * faces->dA(fpt);
        }
        else if (eles->nDims == 3)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(fpt, 0, 0);
          double momx = faces->U(fpt, 1, 0);
          double momy = faces->U(fpt, 2, 0);
          double momz = faces->U(fpt, 3, 0);
          double e = faces->U(fpt, 4, 0);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

           /* Gradients */
          double rho_dx = faces->dU(fpt, 0, 0, 0);
          double momx_dx = faces->dU(fpt, 1, 0, 0);
          double momy_dx = faces->dU(fpt, 2, 0, 0);
          double momz_dx = faces->dU(fpt, 3, 0, 0);

          double rho_dy = faces->dU(fpt, 0, 1, 0);
          double momx_dy = faces->dU(fpt, 1, 1, 0);
          double momy_dy = faces->dU(fpt, 2, 1, 0);
          double momz_dy = faces->dU(fpt, 3, 1, 0);

          double rho_dz = faces->dU(fpt, 0, 2, 0);
          double momx_dz = faces->dU(fpt, 1, 2, 0);
          double momy_dz = faces->dU(fpt, 2, 2, 0);
          double momz_dz = faces->dU(fpt, 3, 2, 0);

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
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
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

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(fpt, 0) + tauxy * faces->norm(fpt, 1) + tauxz * faces->norm(fpt, 2);
          taun[1] = tauxy * faces->norm(fpt, 0) + tauyy * faces->norm(fpt, 1) + tauyz * faces->norm(fpt, 2);
          taun[3] = tauxz * faces->norm(fpt, 0) + tauyz * faces->norm(fpt, 1) + tauzz * faces->norm(fpt, 2);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force[dim] -= eles->weights_fpts(idx) * taun[dim] *
              faces->dA(fpt);
        }

      }

      // Add fpt's contribution to total force and moment
      for (unsigned int d = 0; d < eles->nDims; d++)
        tot_force[d] += force[d];

      if (eles->nDims == 3)
      {
        for (unsigned int d = 0; d < eles->nDims; d++)
          tot_moment[d] += faces->coord(fpt,c1[d]) * force[c2[d]]
              - faces->coord(fpt,c2[d]) * force[c1[d]];
      }
      else
      {
        // Only a 'z' component in 2D
        tot_moment[2] += faces->coord(fpt,0) * force[1]
            - faces->coord(fpt,1) * force[0];
      }

      count++;
    }
  }
}

//void FRSolver::filter_solution()
//{
//  if (!input->filt_on) return;
  
//  /* Sense discontinuities and filter solution */
//  unsigned int status = 1;
//  for (unsigned int level = 0; level < input->filt_maxLevels && status; level++)
//  {
//    filt.apply_sensor();
//    status = filt.apply_expfilter(level);
//  }
//}

void FRSolver::filter_solution()
{
  if (input->filt_on)
  {
    filt.apply_sensor();

    /* Method 1 */
    filt.apply_expfilter();
    if(input->filt2on)
      filt.apply_expfilter_type2();
  }
}

void FRSolver::move(double time)
{
  if (!input->motion) return;

#ifdef _CPU
  move_grid(input, geo, time);
#endif

#ifdef _GPU
  move_grid_wrapper(geo.coord_nodes_d, geo.coords_init_d, geo.grid_vel_nodes_d,
      motion_vars_d, geo.nNodes, geo.nDims, input->motion_type, time, geo.gridID);

  check_error();
#endif

  eles->move(faces);
}

#ifdef _GPU
void FRSolver::report_gpu_mem_usage()
{
  size_t free, total, used;
  cudaMemGetInfo(&free, &total);
  used = total - free;

#ifndef _MPI
  std::cout << "GPU Memory Usage: " << used/1e6 << " MB used of " << total/1e6 << " MB available" << std::endl;
#else
  size_t used_max, used_min;

  if (input->rank == 0)
  {
    MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);  
    MPI_Reduce(&used, &used_min, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);  

    std::cout << "GPU Memory Usage: " << (used_min/1e6) << " (min) - " << (used_max/1e6) << " (max) MB used of " << total/1e6;
    std::cout << " MB available per GPU" << std::endl;
  }
  else
  {
    MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);  
    MPI_Reduce(&used, &used_min, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, MPI_COMM_WORLD);  
  }
#endif
}
#endif
