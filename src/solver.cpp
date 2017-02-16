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
#include "tris.hpp"
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

  if (input->rank == 0) std::cout << "Setting up timestepping..." << std::endl;
  setup_update();  

  if (input->rank == 0) std::cout << "Setting up elements and faces..." << std::endl;

  for (auto etype : geo.ele_set)
  {
    if (etype == QUAD)
    {
      if(!geo.nElesBT.count(TRI)) //note: nElesBT used here since TRI can be removed from ele_set during MPI preprocessing.
        elesObjs.push_back(std::make_shared<Quads>(&geo, input, order));
      else
      {
        std::cout << "Increased order of quads!" << std::endl;
        elesObjs.push_back(std::make_shared<Quads>(&geo, input, order + 1));
      }
    }
    else if (etype == TRI)
    {
      if (input->viscous and !input->grad_via_div)
        ThrowException("Need to enable grad_via_div to use triangles for viscous problems!");

       elesObjs.push_back(std::make_shared<Tris>(&geo, input, order));
    }
    else if (etype == HEX)
       elesObjs.push_back(std::make_shared<Hexas>(&geo, input, order));
    
  }

  eles = elesObjs[0];

  faces = std::make_shared<Faces>(&geo, input, myComm);

  faces->setup(geo.nDims, elesObjs[0]->nVars);

  /* Partial element setup for flux point orientation */
  for (auto e : elesObjs)
  {
    e->set_locs();
    e->set_shape();
    e->set_coords(faces);
  }

  /* For 3D cases, need to orient face flux points */
  if (geo.nDims == 3)
    orient_fpts();

  /* Complete element setup */
  for (auto e : elesObjs)
  {
    e->setup(faces, myComm);
  }


  if (input->rank == 0) std::cout << "Setting up output..." << std::endl;
  setup_output();

  if (input->rank == 0) std::cout << "Initializing solution..." << std::endl;
  //initialize_U();
  for (auto e : elesObjs)
    e->initialize_U();


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

void FRSolver::orient_fpts()
{
  mdvector<double> fpt_coords_L({geo.nDims, geo.nGfpts}), fpt_coords_R({geo.nDims, geo.nGfpts});
  std::vector<unsigned int> idxL(geo.nGfpts), idxR(geo.nGfpts), idxsort(geo.nGfpts);
 
  /* Gather all flux point coordinates */
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfptBT[e->etype](fpt,ele);
        int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,ele);

        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          if (slot == 0)
            fpt_coords_L(dim, gfpt) = e->coord_fpts(fpt, ele, dim);
          else
            fpt_coords_R(dim, gfpt) = e->coord_fpts(fpt, ele, dim);
        }
      }
    }
  }

  for (unsigned int fpt = 0; fpt < geo.nGfpts; fpt++)
  {
    idxL[fpt] = fpt; idxR[fpt] = fpt; idxsort[fpt] = fpt;
  }

  /* Get consistent coupling via fuzzysort */
  for (unsigned int f = 0; f < geo.nGfpts_int/geo.nFptsPerFace; f++)
  {
    unsigned int shift = f * geo.nFptsPerFace;
    fuzzysort_ind(fpt_coords_L, idxL.data() + shift, geo.nFptsPerFace, geo.nDims);
    fuzzysort_ind(fpt_coords_R, idxR.data() + shift, geo.nFptsPerFace, geo.nDims);
  }

  /* Sort again to make left face indexing contiguous */
  std::sort(idxsort.begin(), idxsort.end(), [&](unsigned int a, unsigned int b) {return idxL[a] < idxL[b];});
  auto idxR_copy = idxR;
  for (unsigned int fpt = 0; fpt < geo.nGfpts_int; fpt++)
    idxR[fpt] = idxR_copy[idxsort[fpt]];


  /* Reindex right face flux points */
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
      {
        int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,ele);

        if (slot == 1)
        {
          int gfpt_old = geo.fpt2gfptBT[e->etype](fpt,ele);
          geo.fpt2gfptBT[e->etype](fpt, ele) = idxR[gfpt_old];
        }
      }
    }
  }

#ifdef _MPI
  /* For MPI, just use coupling from fuzzysort directly */
  for (auto &entry : geo.fpt_buffer_map)
  {
    auto &fpts = entry.second;
    for (unsigned int f = 0; f < fpts.size()/geo.nFptsPerFace; f++)
    {
      unsigned int shift = f * geo.nFptsPerFace;
      fuzzysort_ind(fpt_coords_L, fpts.data() + shift, geo.nFptsPerFace, geo.nDims);
    }
  }
#endif
}

void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  if (input->dt_scheme == "Euler")
  {
    input->nStages = 1;
    rk_beta.assign({input->nStages}, 1.0);

  }
  else if (input->dt_scheme == "RK44")
  {
    input->nStages = 4;
    
    rk_alpha.assign({input->nStages-1});
    rk_alpha(0) = 0.5; rk_alpha(1) = 0.5; rk_alpha(2) = 1.0;

    rk_beta.assign({input->nStages});
    rk_beta(0) = 1./6.; rk_beta(1) = 1./3.; 
    rk_beta(2) = 1./3.; rk_beta(3) = 1./6.;
  }
  else if (input->dt_scheme == "RKj")
  {
    input->nStages = 4;
    rk_alpha.assign({input->nStages});
    /* Standard RK44 */
    //rk_alpha(0) = 1./4; rk_alpha(1) = 1./3.; 
    //rk_alpha(2) = 1./2.; rk_alpha(3) = 1.0;
    /* OptRK4 (r = 0.5) */
    rk_alpha(0) = 0.153; rk_alpha(1) = 0.442; 
    rk_alpha(2) = 0.930; rk_alpha(3) = 1.0;
  }
  else if (input->dt_scheme == "LSRK")
  {
    input->nStages = 5;
    rk_alpha.assign({input->nStages - 1});
    rk_alpha(0) =   970286171893. / 4311952581923.;
    rk_alpha(1) =  6584761158862. / 12103376702013.;
    rk_alpha(2) =  2251764453980. / 15575788980749.;
    rk_alpha(3) = 26877169314380. / 34165994151039.;

    rk_beta.assign({input->nStages});
    rk_beta(0) =  1153189308089. / 22510343858157.;
    rk_beta(1) =  1772645290293. / 4653164025191.;
    rk_beta(2) = -1672844663538. / 4480602732383.;
    rk_beta(3) =  2114624349019. / 3568978502595.;
    rk_beta(4) =  5198255086312. / 14908931495163.;

    rk_bhat.assign({input->nStages});
    rk_bhat(0) =  1016888040809. / 7410784769900.;
    rk_bhat(1) = 11231460423587. / 58533540763752.;
    rk_bhat(2) = -1563879915014. / 6823010717585.;
    rk_bhat(3) =   606302364029. / 971179775848.;
    rk_bhat(4) =  1097981568119. / 3980877426909.;

    rk_c.assign({input->nStages});
    for (int i = 1; i < input->nStages; i++)
    {
      rk_c(i) = rk_alpha(i-1);

      for (int j = 0; j < i-1; j++)
        rk_c(i) += rk_beta(j);
    }

    expa = input->pi_alpha / 4.;
    expb = input->pi_beta / 4.;
    prev_err = 1.;

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
    input->nStages = 2;
    rk_alpha.assign({input->nStages});
    rk_beta.assign({input->nStages});

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
       
  for (auto e : elesObjs)
    e->setup_ppt_connectivity();
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
  std::map<ELE_TYPE, mdvector<double>> U_restart;

  std::map<ELE_TYPE, unsigned int> nElesBT = geo.nElesBT;

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

      unsigned int nRpts;
      /* Setup extrapolation operator from equistant restart points */
      for (auto e : elesObjs)
      {
        if (e->etype == QUAD and geo.nElesBT.count(TRI)) // to deal with increased quad order with mixed grids
        {
          e->set_oppRestart(order_restart + 1, true);
        }
        else
          e->set_oppRestart(order_restart, true);

        nRpts = e->oppRestart.get_dim(1);
        U_restart[e->etype].assign({nRpts, e->nEles, e->nVars});
      }

      unsigned int temp; 
      for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
      {
        binary_read(f, temp);
        for (auto e: elesObjs)
        {
          nRpts = e->oppRestart.get_dim(1);

          for (unsigned int ele = 0; ele < e->etype; ele++)
          {
            /// TODO: make sure this is setup correctly first
            if (input->overset && geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) continue;

            for (unsigned int rpt = 0; rpt < nRpts; rpt++)
            {
              binary_read(f, U_restart[e->etype](rpt, ele, n));
            }
          }
        }
      }

      /* Extrapolate values from restart points to solution points */
      for (auto e : elesObjs)
      {
        nRpts = e->oppRestart.get_dim(1);

        auto &A = e->oppRestart(0, 0);
        auto &B = U_restart[e->etype](0, 0, 0);
        auto &C = e->U_spts(0, 0, 0);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nSpts, 
            e->nEles * e->nVars, nRpts, 1.0, &A, e->oppRestart.ldim(), &B, 
            U_restart[e->etype].ldim(), 0.0, &C, e->U_spts.ldim());
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nSpts, 
            e->nEles * e->nVars, nRpts, 1.0, &A, e->oppRestart.ldim(), &B, 
            U_restart[e->etype].ldim(), 0.0, &C, e->U_spts.ldim());
#endif
      }
    }
  }

  f.close();
}

#ifdef _GPU
void FRSolver::solver_data_to_device()
{
  /* Initial copy of data to GPU. Assignment operator will allocate data on device when first
   * used. */

  /* -- Element Data -- */
  for (auto e : elesObjs)
  {
    e->oppE_d = e->oppE;
    e->oppD_d = e->oppD;
    e->oppD0_d = e->oppD0;
    e->oppD_fpts_d = e->oppD_fpts;
    e->oppDiv_fpts_d = e->oppDiv_fpts;

    e->U_ini_d = e->U_ini;
    e->dt_d = e->dt;
    e->U_spts_d = e->U_spts;
    e->U_fpts_d = e->U_fpts;
    e->Uavg_d = e->Uavg;
    e->weights_spts_d = e->weights_spts;
    e->weights_fpts_d = e->weights_fpts;
    e->Fcomm_d = e->Fcomm;
    e->F_spts_d = e->F_spts;
    e->divF_spts_d = e->divF_spts;
    e->coord_spts_d = e->coord_spts;
    e->inv_jaco_spts_d = e->inv_jaco_spts;
    e->jaco_det_spts_d = e->jaco_det_spts;
    e->vol_d = e->vol;

    geo.fpt2gfptBT_d[e->etype] = geo.fpt2gfptBT[e->etype];
    geo.fpt2gfpt_slotBT_d[e->etype] = geo.fpt2gfpt_slotBT[e->etype];

    if (input->CFL_type == 2)
      e->h_ref_d = e->h_ref;

    if (input->viscous)
    {
      e->dU_spts_d = e->dU_spts;
      e->Ucomm_d = e->Ucomm;
      e->dU_fpts_d = e->dU_fpts;
    }
    

    if (input->dt_scheme == "LSRK")
    {
      e->U_til_d = e->U_til;
      e->rk_err_d = e->rk_err;
    }

    if (input->dt_scheme == "MCGS")
    {
      e->deltaU_d = e->deltaU;
      e->RHS_d = e->RHS;
      e->LHS_d = e->LHSs[0];

      if (input->inv_mode)
      {
        e->LHSInv_d = e->LHSInvs[0];
      }

      e->LU_pivots_d = e->LU_pivots;
      e->LU_info_d = e->LU_info;

      /* For cublas batched LU: Setup and transfer array of GPU pointers to 
       * LHS matrices and RHS vectors */
      unsigned int N = e->nSpts * e->nVars;
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

        e->RHS_ptrs(ele) = e->RHS_d.data() + ele * N;

        if (input->inv_mode)
        {
          e->deltaU_ptrs(ele) = e->deltaU_d.data() + ele * N;
        }
      }

      if (!input->stream_mode)
      {
        unsigned int nElesMax = ceil(geo.nEles / (double) input->n_LHS_blocks);
        for (unsigned int ele = 0; ele < nElesMax; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          e->LHS_ptrs(ele) = e->LHS_d.data() + ele * (N * N);
        }
          
        if (input->inv_mode)
        {
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

            e->LHSInv_ptrs(ele) = e->LHSInv_d.data() + ele * (N * N);
          }
        }
      }
      else
      {
        unsigned int nElesMax = *std::max_element(geo.ele_color_nEles.begin(), geo.ele_color_nEles.end());
        for (unsigned int ele = 0; ele < nElesMax; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          e->LHS_ptrs(ele) = e->LHS_d.data() + ele * (N * N);
          if (input->inv_mode)
            e->LHSInv_ptrs(ele) = e->LHSInv_d.data() + ele * (N * N);
        }
      }

      e->LHS_ptrs_d = e->LHS_ptrs;
      e->RHS_ptrs_d = e->RHS_ptrs;

      if (input->inv_mode)
      {
        e->LHSInv_ptrs_d = e->LHSInv_ptrs;
        e->deltaU_ptrs_d = e->deltaU_ptrs;
      }

      /* Implicit flux derivative data structures (element local) */
      e->dFdU_spts_d = e->dFdU_spts;
      e->dFcdU_fpts_d = e->dFcdU_fpts;

    }

    //TODO: Temporary fix. Need to remove usage of jaco_spts_d from all kernels.
    if (input->motion || input->dt_scheme == "MCGS")
    {
      e->jaco_spts_d = e->jaco_spts;
    }

    if (input->motion)
    {
      e->coord_fpts_d = e->coord_fpts;
      e->nodes_d = e->nodes;
      e->grid_vel_nodes_d = e->grid_vel_nodes;
      e->grid_vel_spts_d = e->grid_vel_spts;
      e->grid_vel_fpts_d = e->grid_vel_fpts;
      e->shape_spts_d = e->shape_spts;
      e->shape_fpts_d = e->shape_fpts;
      e->dshape_spts_d = e->dshape_spts;
      e->dshape_fpts_d = e->dshape_fpts;
      e->jaco_fpts_d = e->jaco_fpts;
      e->inv_jaco_fpts_d = e->inv_jaco_fpts;
      e->tnorm_d = e->tnorm;
      e->dUr_spts_d = e->dUr_spts;
      e->dF_spts_d = e->dF_spts;
      e->dFn_fpts_d = e->dFn_fpts;
      e->tempF_fpts_d = e->tempF_fpts;
      
      geo.ele2nodesBT_d[e->etype] = geo.ele2nodesBT[e->etype];
    }
  }

  /* -- Face Data -- */
  faces->U_bnd_d = faces->U_bnd;
  faces->U_bnd_ldg_d = faces->U_bnd_ldg;
  faces->P_d = faces->P;
  faces->Ucomm_bnd_d = faces->Ucomm_bnd;
  faces->Fcomm_bnd_d = faces->Fcomm_bnd;
  faces->norm_d = faces->norm;
  faces->dA_d = faces->dA;
  faces->waveSp_d = faces->waveSp;
  faces->diffCo_d = faces->diffCo;
  faces->rus_bias_d = faces->rus_bias;
  faces->LDG_bias_d = faces->LDG_bias;

  if (input->viscous)
  {
    faces->dU_bnd_d = faces->dU_bnd;
  }

  if (input->dt_scheme == "MCGS")
  {
    /* Implicit flux derivative data structures (faces) */
    faces->dFdUconv_d = faces->dFdUconv;
    faces->dFcdU_d = faces->dFcdU;
  }

  if (input->motion)
  {
    faces->Vg_d = faces->Vg;
    //faces->coord_d = faces->coord;
  }

  /* -- Additional data -- */
  /* Geometry */
  geo.gfpt2bnd_d = geo.gfpt2bnd;

  /* Input parameters */
  input->V_fs_d = input->V_fs;
  input->V_wall_d = input->V_wall;
  input->norm_fs_d = input->norm_fs;
  input->AdvDiff_A_d = input->AdvDiff_A;

  rk_alpha_d = rk_alpha;
  rk_beta_d = rk_beta;

  if (input->motion)
  {
    geo.coord_nodes_d = geo.coord_nodes;
    geo.coords_init_d = geo.coords_init;
    geo.grid_vel_nodes_d = geo.grid_vel_nodes;

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
  unsigned int startFpt = 0; unsigned int endFpt = geo.nGfpts;
  unsigned int startEle = 0;

#ifdef _MPI
  endFpt = geo.nGfpts_int + geo.nGfpts_bnd;
  unsigned int startFptMpi = endFpt;
#endif

  /* If using coloring, modify range to extrapolate data from previously updated colors */
  //if (color && geo.nColors > 1)
  //{
  //  startEle = geo.ele_color_range[prev_color - 1]; endEle = geo.ele_color_range[prev_color];
  //}

#ifdef _BUILD_LIB
  if (input->overset)
  {
    for (auto e: elesObjs)
      ZEFR->overset_interp(faces->nVars, e->U_spts.data(), faces->U.data(), 0);
  }
#endif

  /* Extrapolate solution to flux points */
  for (auto e : elesObjs)
    e->extrapolate_U(startEle, e->nEles);


  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    for (auto e : elesObjs)
    {
      e->compute_Uavg();
      e->poly_squeeze();
    }
  }

  /* For coloring, modify range to sweep through current color */
  //if (color && geo.nColors > 1)
  //{
  //  startEle = geo.ele_color_range[color - 1]; endEle = geo.ele_color_range[color];
  //}

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  /* If running inviscid, use this scheduling. */
  if(!input->viscous)
  {
    for (auto e : elesObjs)
    {
      /* Compute flux at solution points */
      e->compute_F(startEle, e->nEles);

      /* Transform solution point fluxes from physical to reference space */
      if (input->motion)
      {
        e->compute_gradF_spts(startEle, e->nEles);
        e->compute_dU0(startEle, e->nEles);
      }
    }

    /* Compute parent space common flux at non-MPI flux points */
    faces->compute_common_F(startFpt, endFpt);

    /* Compute solution point contribution to divergence of flux */
    if (input->motion)
    {
      for (auto e: elesObjs)
        e->transform_gradF_spts(stage, startEle, e->nEles);
    }
    else
    {
      for (auto e: elesObjs)
        e->compute_divF_spts(stage, startEle, e->nEles);
    }

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
    // EXPERIMENTAL gradient computation from divergence
    if (input->grad_via_div)
    {
      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();
      
      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      for (auto e : elesObjs)
      {
        for (unsigned int dim = 0; dim < e->nDims; dim++)
        {
          // Compute unit advection flux at solution points with wavespeed along dim
          e->compute_unit_advF(startEle, e->nEles, dim); 

          // Convert common U to common normal advection flux
          faces->common_U_to_F(startFpt, geo.nGfpts, dim);

          // Compute physical gradient (times jacobian determinant) along via divergence of F
          e->compute_dU_spts_via_divF(startEle, e->nEles, dim);
          e->compute_dU_fpts_via_divF(startEle, e->nEles, dim);
        }
      }
    }
    else
    {
      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

      /* Compute solution point contribution to (corrected) gradient of state variables at solution points */
      for (auto e : elesObjs)
        e->compute_dU_spts(startEle, e->nEles);

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();
      
      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      /* Compute flux point contribution to (corrected) gradient of state variables at solution points */
      for (auto e : elesObjs)
        e->compute_dU_fpts(startEle, e->nEles);

      /* Copy un-transformed dU to dUr for later use (L-M chain rule) */
      if (input->motion)
      {
        for (auto e : elesObjs)
          e->compute_dU0(startEle, e->nEles);
      }
    }

    for (auto e : elesObjs)
    {
      /* Compute flux at solution points */
      e->compute_F(startEle, e->nEles);

      /* Extrapolate physical solution gradient (computed during compute_F) to flux points */
      e->extrapolate_dU(startEle, e->nEles);
    }

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();

    /* Interpolate gradient data to/from other grid(s) */
#ifdef _BUILD_LIB
    if (input->overset)
    {
      for (auto e : elesObjs)
        ZEFR->overset_interp(faces->nVars, e->dU_spts.data(), faces->dU.data(), 1);
    }
#endif
#endif

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    
    if (input->motion)
    {
      /* Use Liang-Miyaji Chain-Rule form to compute divF */
      for (auto e : elesObjs)
        e->compute_gradF_spts(startEle, e->nEles);

      for (auto e : elesObjs)
        e->transform_gradF_spts(stage, startEle, e->nEles);
    }
    else
    {
      /* Compute solution point contribution to divergence of flux */
      for (auto e : elesObjs)
        e->compute_divF_spts(stage, startEle, e->nEles);
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
    for (auto e : elesObjs)
    {
      e->extrapolate_Fn(startEle, e->nEles, faces);
      e->correct_divF_spts(stage, startEle, e->nEles);
    }
  }
  else
  {
    /* Compute flux point contribution to divergence of flux */
    for (auto e : elesObjs)
      e->compute_divF_fpts(stage, startEle, e->nEles);
  }

  /* Add source term (if required) */
  if (input->source)
  {
    for (auto e : elesObjs)
      e->add_source(stage, flow_time, startEle, e->nEles);
  }

}

void FRSolver::compute_LHS()
{
}

void FRSolver::compute_LHS_LU(unsigned int startEle, unsigned int endEle, unsigned int color)
{
}

void FRSolver::compute_RHS(unsigned int color)
{
}

#ifdef _CPU
void FRSolver::compute_RHS_source(const mdvector<double> &source, unsigned int color)
{
}
#endif

#ifdef _GPU
void FRSolver::compute_RHS_source(const mdvector_gpu<double> &source, unsigned int color)
{
}
#endif

void FRSolver::compute_deltaU(unsigned int color)
{
}

void FRSolver::compute_U(unsigned int color)
{
}

void FRSolver::initialize_U()
{
  // initialization moved into elements
}

void FRSolver::setup_views()
{
  /* Setup face view of element solution data struture */
  // TODO: Might not want to allocate all these at once. Turn this into a function maybe?
  mdvector<double*> U_base_ptrs({2 * geo.nGfpts});
  mdvector<double*> U_ldg_base_ptrs({2 * geo.nGfpts});
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
  mdvector<double*> U_ldg_base_ptrs_d({2 * geo.nGfpts});
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
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfptBT[e->etype](fpt,ele);
        int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,ele);

        U_base_ptrs(gfpt + slot * geo.nGfpts) = &e->U_fpts(fpt, ele, 0);
        U_ldg_base_ptrs(gfpt + slot * geo.nGfpts) = &e->U_fpts(fpt, ele, 0);
        U_strides(gfpt + slot * geo.nGfpts) = e->U_fpts.get_stride(1);

        Fcomm_base_ptrs(gfpt + slot * geo.nGfpts) = &e->Fcomm(fpt, ele, 0);

        if (input->viscous) Ucomm_base_ptrs(gfpt + slot * geo.nGfpts) = &e->Ucomm(fpt, ele, 0);
#ifdef _GPU
        U_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_ptr(fpt, ele, 0);
        U_ldg_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_ptr(fpt, ele, 0);
        U_strides_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_stride(1);

        Fcomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->Fcomm_d.get_ptr(fpt, ele, 0);

        if (input->viscous) Ucomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->Ucomm_d.get_ptr(fpt, ele, 0);
#endif

        if (input->viscous)
        {
          dU_base_ptrs(gfpt + slot * geo.nGfpts) = &e->dU_fpts(fpt, ele, 0, 0);
          dU_strides(gfpt + slot * geo.nGfpts, 0) = e->dU_fpts.get_stride(1);
          dU_strides(gfpt + slot * geo.nGfpts, 1) = e->dU_fpts.get_stride(2);

#ifdef _GPU
          dU_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->dU_fpts_d.get_ptr(fpt, ele, 0, 0);
          dU_strides_d(gfpt + slot * geo.nGfpts, 0) = e->dU_fpts_d.get_stride(1);
          dU_strides_d(gfpt + slot * geo.nGfpts, 1) = e->dU_fpts_d.get_stride(2);
#endif
         
        }
        
      }
    }
  }

  /* Set pointers for remaining faces (includes boundary and MPI faces) */
  unsigned int i = 0;
  for (unsigned int gfpt = geo.nGfpts_int; gfpt < geo.nGfpts; gfpt++)
  {
    for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
    {
      U_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd(i, 0);
      
      if (gfpt < geo.nGfpts_int + geo.nGfpts_bnd)
        U_ldg_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd_ldg(i, 0);
      else
        U_ldg_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd(i, 0); // point U_ldg to correct MPI data;

      U_strides(gfpt + 1 * geo.nGfpts) = faces->U_bnd.get_stride(0);

      Fcomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Fcomm_bnd(i, 0);

      if (input->viscous) Ucomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Ucomm_bnd(i, 0);

#ifdef _GPU
      U_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_ptr(i, 0);
      if (gfpt < geo.nGfpts_int + geo.nGfpts_bnd)
        U_ldg_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_ldg_d.get_ptr(i, 0);
      else
        U_ldg_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_ptr(i, 0); // point U_ldg to correct MPI data;

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
  faces->U_ldg.assign(U_ldg_base_ptrs, U_strides, geo.nGfpts);
  faces->Fcomm.assign(Fcomm_base_ptrs, U_strides, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm.assign(Ucomm_base_ptrs, U_strides, geo.nGfpts);
    faces->dU.assign(dU_base_ptrs, dU_strides, geo.nGfpts);
  }

#ifdef _GPU
  faces->U_d.assign(U_base_ptrs_d, U_strides_d, geo.nGfpts);
  faces->U_ldg_d.assign(U_ldg_base_ptrs_d, U_strides_d, geo.nGfpts);
  faces->Fcomm_d.assign(Fcomm_base_ptrs_d, U_strides_d, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm_d.assign(Ucomm_base_ptrs_d, U_strides_d, geo.nGfpts);
    faces->dU_d.assign(dU_base_ptrs_d, dU_strides_d, geo.nGfpts);
  }
#endif

}

/* Note: Source term in update() is used primarily for multigrid. To add a true source term, define
 * a source term in funcs.cpp and set source input flag to 1. */
#ifdef _CPU
void FRSolver::update(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif 
#ifdef _GPU
void FRSolver::update(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  prev_time = flow_time;

  if (input->dt_scheme == "LSRK")
  {
    step_adaptive_LSRK(sourceBT);
  }
  else
  {
    if (input->dt_scheme == "MCGS")
      step_MCGS(sourceBT);
    else
      step_RK(sourceBT);
  }

  flow_time = prev_time + elesObjs[0]->dt(0);
  current_iter++;

  // Update grid to end of time step (if not already done so)
  if (input->dt_scheme != "MCGS" && (input->nStages == 1 || (input->nStages > 1 && rk_alpha(input->nStages-2) != 1)))
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
void FRSolver::step_RK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_RK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
#ifdef _CPU
  if (input->nStages > 1)
  {
    for (auto e : elesObjs)
      e->U_ini = e->U_spts;
  }
#endif

#ifdef _GPU
  for (auto e : elesObjs)
    device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
  check_error();
#endif

  unsigned int nSteps = (input->dt_scheme == "RKj") ? input->nStages : input->nStages - 1;

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < nSteps; stage++)
  {
    flow_time = prev_time + rk_alpha(stage) * elesObjs[0]->dt(0);

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
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < e->nVars; n++)
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
            {
              if (input->dt_type != 2)
              {
                e->U_spts(spt, ele, n) = e->U_ini(spt, ele, n) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(spt, ele, n, stage);
              }
              else
              {
                e->U_spts(spt, ele, n) = e->U_ini(spt, ele, n) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(spt, ele, n, stage);
              }
            }
          }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < e->nVars; n++)
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
            {
              if (input->dt_type != 2)
              {
                e->U_spts(spt, ele, n) = e->U_ini(spt, ele, n) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt,ele) * (e->divF_spts(spt, ele, n, stage) + sourceBT.at(e->etype)(spt, ele, n));
              }
              else
              {
                e->U_spts(spt, ele, n) = e->U_ini(spt, ele, n) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt,ele) * (e->divF_spts(spt, ele, n, stage) + sourceBT.at(e->etype)(spt, ele, n));
              }
            }
          }
      }
    }
#endif

#ifdef _GPU
      /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
      unsigned int last_stage = (input->dt_scheme == "RKj") ? input->nStages + 1 : input->nStages;

      for (auto e : elesObjs)
      {
        if (!sourceBT.count(e->etype))
        {
          RK_update_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                            rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                            input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
        }
        else
        {
          RK_update_source_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                   rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                   input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
        }
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
    if (input->nStages > 1)
      flow_time = prev_time + rk_alpha(input->nStages-2) * elesObjs[0]->dt(0);

    compute_residual(input->nStages-1);
#ifdef _CPU
    if (input->nStages > 1)
    {
      for (auto e : elesObjs)
        e->U_spts = e->U_ini;
    }
    else if (input->dt_type != 0)
    {
      compute_element_dt();
    }
#endif
#ifdef _GPU
    for (auto e : elesObjs)
      device_copy(e->U_spts_d, e->U_ini_d, e->U_spts_d.max_size());
#endif

#ifdef _CPU
    for (auto e : elesObjs)
    {
      for (unsigned int stage = 0; stage < input->nStages; stage++)
      {
        if (!sourceBT.count(e->etype))
        {
#pragma omp parallel for collapse(2)
          for (unsigned int n = 0; n < e->nVars; n++)
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
              for (unsigned int spt = 0; spt < e->nSpts; spt++)
                if (input->dt_type != 2)
                {
                  e->U_spts(spt, ele, n) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt,ele) *
                      e->divF_spts(spt, ele, n, stage);
                }
                else
                {
                  e->U_spts(spt, ele, n) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt,ele) *
                      e->divF_spts(spt, ele, n, stage);
                }
            }
        }
        else
        {
#pragma omp parallel for collapse(2)
          for (unsigned int n = 0; n < e->nVars; n++)
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
              for (unsigned int spt = 0; spt < e->nSpts; spt++)
              {
                if (input->dt_type != 2)
                {
                  e->U_spts(spt, ele, n) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt,ele) *
                      (e->divF_spts(spt, ele, n, stage) + sourceBT.at(e->etype)(spt, ele, n));
                }
                else
                {
                  e->U_spts(spt, ele, n) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt,ele) *
                      (e->divF_spts(spt, ele, n, stage) + sourceBT.at(e->etype)(spt, ele, n));
                }
              }
            }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                          rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                          input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
    }

    check_error();
#endif
  }

}

#ifdef _CPU
void FRSolver::step_adaptive_LSRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_adaptive_LSRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  step_LSRK(sourceBT);

  // Calculate error (infinity norm of RK error) and scaling factor for dt
  double max_err = 0;
#ifdef _CPU
  for (auto e : elesObjs)
  {
    for (uint n = 0; n < e->nVars; n++)
    {
      for (uint ele = 0; ele < e->nEles; ele++)
      {
        for (uint spt = 0; spt < e->nSpts; spt++)
        {
          double err = std::abs(e->rk_err(spt,ele,n)) /
              (input->atol + input->rtol * std::max( std::abs(e->U_spts(spt,ele,n)), std::abs(e->U_ini(spt,ele,n)) ));
          max_err = std::max(max_err, err);
        }
      }
    }
  }


#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

  // Determine the time step scaling factor and the new time step
  double fac = pow(max_err, -expa) * pow(prev_err, expb);
  fac = std::min(input->maxfac, std::max(input->minfac, input->sfact*fac));

  for (auto e : elesObjs)
    e->dt(0) *= fac;
#endif

#ifdef _GPU
  max_err = 0.0;
  for (auto e : elesObjs)
  {

    double err = get_rk_error_wrapper(e->U_spts_d, e->U_ini_d, e->rk_err_d, e->nSpts, e->nEles,
        e->nVars, input->atol, input->rtol, worldComm, input->overset, geo.iblank_cell_d.data());

    err = std::isnan(err) ? INFINITY : err; // convert NaNs to "large" error

    max_err = std::max(max_err, err);

  }

  for (auto e : elesObjs)
  {
    set_adaptive_dt_wrapper(e->U_spts_d, e->U_ini_d, e->rk_err_d, e->dt_d, e->dt(0),
        e->nSpts, e->nEles, e->nVars, input->atol, input->rtol, expa, expb,
        input->minfac, input->maxfac, input->sfact, max_err, prev_err, worldComm, input->overset,
        geo.iblank_cell_d.data());
  }
#endif


  if (elesObjs[0]->dt(0) < 1e-14)
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
    for (auto e : elesObjs)
      e->U_spts = e->U_ini;
#endif
#ifdef _GPU
    for (auto e : elesObjs)
      device_copy(e->U_spts_d, e->U_ini_d, e->U_ini_d.max_size());
#endif
    // Try again with new dt
    step_adaptive_LSRK(sourceBT);
  }
}

#ifdef _CPU
void FRSolver::step_LSRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_LSRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  /* NOTE: this implementation is not the 'true' low-storage implementation
   * since we are using an additional array 'U_til' instead of swapping
   * pointers at each stage */

  // Copy current solution into "U_ini" ['rold' in PyFR]
#ifdef _CPU
  for (auto e : elesObjs)
  {
    e->U_ini = e->U_spts;
    e->U_til = e->U_spts;
    e->rk_err.fill(0.0);
  }
#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    device_copy(e->U_til_d, e->U_spts_d, e->U_spts_d.max_size());
    device_fill(e->rk_err_d, e->rk_err_d.max_size());
    
    // Get current delta t [dt(0)] (updated on GPU)
    copy_from_device(e->dt.data(), e->dt_d.data(), 1);
  }
  

  check_error();
#endif

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < input->nStages; stage++)
  {
    flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

    compute_residual(0);

    //double ai = (stage < input->nStages - 1) ? rk_alpha(stage) : 0.0;
    double ai = rk_alpha(stage);
    double bi = rk_beta(stage);
    double bhi = rk_bhat(stage);

#ifdef _CPU
    // Update Error
    for (auto e : elesObjs)
    {
      for (unsigned int n = 0; n < e->nVars; n++)
      {
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            e->rk_err(spt,ele,n) -= (bi - bhi) * e->dt(0) /
                e->jaco_det_spts(spt,ele) * e->divF_spts(spt,ele,n,0);
          }
        }
      }

      // Update solution registers
      if (stage < input->nStages - 1)
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < e->nVars; n++)
        {
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
            {
              e->U_spts(spt,ele,n) = e->U_til(spt,ele,n) - ai * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(spt,ele,n,0);

              e->U_til(spt,ele,n) = e->U_spts(spt,ele,n) - (bi - ai) * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(spt,ele,n,0);
            }
          }
        }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int n = 0; n < e->nVars; n++)
        {
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
            {
              e->U_spts(spt,ele,n) = e->U_til(spt,ele,n) - bi * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(spt,ele,n,0);
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        LSRK_update_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d, e->divF_spts_d,
            e->jaco_det_spts_d, e->dt(0), ai, bi, bhi, e->nSpts, e->nEles,
            e->nVars, stage, input->nStages, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        LSRK_update_source_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d,
            e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt(0), ai, bi, bhi,
            e->nSpts, e->nEles, e->nVars, stage, input->nStages, input->overset,
            geo.iblank_cell_d.data());
      }
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

  flow_time = prev_time + elesObjs[0]->dt(0);
}

#ifdef _CPU
void FRSolver::step_MCGS(const std::map<ELE_TYPE, mdvector<double>> &source)
#endif
#ifdef _GPU
void FRSolver::step_MCGS(const std::map<ELE_TYPE, mdvector_gpu<double>> &source)
#endif
{
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
    for (auto e : elesObjs)
    {
#pragma omp parallel for
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      { 
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        double int_waveSp = 0.;  /* Edge/Face integrated wavespeed */

        for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfptBT[e->etype](fpt,ele);
          int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,ele);

          int_waveSp += e->weights_fpts(fpt % e->nFptsPerFace) * faces->waveSp(gfpt) * faces->dA(gfpt, slot); 
        }

        e->dt(ele) = 2.0 * CFL * get_cfl_limit_adv(order) * e->vol(ele) / int_waveSp;
      }
    }
  }

  /* CFL-estimate based on MacCormack for NS */
  else if (input->CFL_type == 2)
  {
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      { 
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        /* Compute inverse of timestep in each face */
        std::vector<double> dtinv(2*e->nDims);
        for (unsigned int face = 0; face < 2*e->nDims; face++)
        {
          for (unsigned int fpt = face * e->nFptsPerFace; fpt < (face+1) * e->nFptsPerFace; fpt++)
          {
            int gfpt = geo.fpt2gfptBT[e->etype](fpt,ele);

            double dtinv_temp = faces->waveSp(gfpt) / (get_cfl_limit_adv(order) * e->h_ref(fpt, ele));
            if (input->viscous)
              dtinv_temp += faces->diffCo(gfpt) / (get_cfl_limit_diff(order, input->ldg_b) * e->h_ref(fpt, ele) * e->h_ref(fpt, ele));
            dtinv[face] = std::max(dtinv[face], dtinv_temp);
          }
        }

        /* Find maximum in each dimension */
        if (e->nDims == 2)
        {
          dtinv[0] = std::max(dtinv[0], dtinv[2]);
          dtinv[1] = std::max(dtinv[1], dtinv[3]);

          e->dt(ele) = CFL / (dtinv[0] + dtinv[1]);
        }
        else
        {
          dtinv[0] = std::max(dtinv[0],dtinv[1]);
          dtinv[1] = std::max(dtinv[2],dtinv[3]);
          dtinv[2] = std::max(dtinv[4],dtinv[5]);

          /// NOTE: this seems ultra-conservative.  Need additional scaling factor?
          e->dt(ele) = CFL / (dtinv[0] + dtinv[1] + dtinv[2]); // * 32; = empirically-found factor for sphere
        }
      }
    }
  }

  if (input->dt_type == 1) /* Global minimum */
  {
    double minDT = INFINITY;

    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        minDT = std::min(minDT, e->dt(ele));
      }
    }

#ifdef _MPI
    /// TODO: If interfacing with other explicit solver, work together here
    MPI_Allreduce(MPI_IN_PLACE, &minDT, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

    for (auto e : elesObjs)
      e->dt(0) = minDT;

  }
#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    compute_element_dt_wrapper(e->dt_d, faces->waveSp_d, faces->diffCo_d, faces->dA_d, geo.fpt2gfptBT_d[e->etype], 
        geo.fpt2gfpt_slotBT_d[e->etype], e->weights_fpts_d, e->vol_d, e->h_ref_d, e->nFptsPerFace, CFL, input->ldg_b, order, 
        input->dt_type, input->CFL_type, e->nFpts, e->nEles, e->nDims, myComm,
        input->overset, geo.iblank_cell_d.data());
  }

  check_error();
#endif
}

void FRSolver::compute_SER_dt()
{
  /* Compute norm of residual */
  // TODO: Create norm function to eliminate repetition, add other norms
  SER_res[1] = SER_res[0];
  SER_res[0] = 0;

  for (auto e : elesObjs)
  {
    for (unsigned int n = 0; n < e->nVars; n++)
    {
      for (unsigned int ele =0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          SER_res[0] += (e->divF_spts(spt, ele, n, 0) / e->jaco_det_spts(spt, ele)) *
                         (e->divF_spts(spt, ele, n, 0) / e->jaco_det_spts(spt, ele));
        }
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
    for (auto e : elesObjs)
    {
      if (input->dt_type == 0)
      {
        e->dt(0) *= omg;
      }
      else if(input->dt_type == 1)
      {
        e->dt(0) *= SER_omg;
      }
      else if (input->dt_type == 2)
      {
#pragma omp parallel for
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          e->dt(ele) *= SER_omg;
        }
      }
    }
  }
}

void FRSolver::write_solution_pyfr(const std::string &_prefix)
{
  if (elesObjs.size() > 1)
    ThrowException("PyFR write not supported for mixed element grids.");

  auto e = elesObjs[0];

#ifdef _GPU
    e->U_spts = e->U_spts_d;
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

  int nEles = e->nEles;
  int nVars = e->nVars;
  int nSpts = e->nSpts;

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
          data_p[0][ind] = e->U_spts(spt,ele,var);
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
              u_tmp(ele,var,spt) = e->U_spts(spt,ele,var);

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
        u_tmp(ele,var,spt) = e->U_spts(spt,ele,var);

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
#endif 

}

void FRSolver::restart_pyfr(std::string restart_file, unsigned restart_iter)
{
  if (elesObjs.size() > 1)
    ThrowException("PyFR restart not supported for mixed element grids.");

  auto e = elesObjs[0];

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

  int nEles = e->nEles;
  int nVars = e->nVars;

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

  if (nSpts == e->nSpts)
  {
    e->U_spts.assign({nSpts,nEles,nVars});
    for (int ele = 0; ele < nEles; ele++)
      for (int var = 0; var < nVars; var++)
        for (int spt = 0; spt < nSpts; spt++)
          e->U_spts(spt,ele,var) = u_tmp(ele,var,spt);
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
    e->set_oppRestart(restartOrder);


    // Extrapolate values from restart points to solution points
    auto &A = e->oppRestart(0, 0);
    auto &B = U_restart(0, 0, 0);
    auto &C = e->U_spts(0, 0, 0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nSpts,
        e->nEles * e->nVars, nSpts, 1.0, &A, e->oppRestart.ldim(), &B,
        U_restart.ldim(), 0.0, &C, e->U_spts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nSpts,
        e->nEles * e->nVars, nSpts, 1.0, &A, e->oppRestart.ldim(), &B,
        U_restart.ldim(), 0.0, &C, e->U_spts.ldim());
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
  for (auto e : elesObjs)
    e->U_spts = e->U_spts_d;
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
    if (input->equation == AdvDiff)
    {
      var = {"u"};
    }
    else if (input->equation == EulerNS)
    {
      if (geo.nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

    }

    for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
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
    for (auto e : elesObjs)
    {
      e->update_plot_point_coords();
#ifdef _GPU
      e->grid_vel_nodes = e->grid_vel_nodes_d;
#endif
    }
  }

  std::map<ELE_TYPE, unsigned int> nElesBT = geo.nElesBT;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (auto e : elesObjs)
    {
      for (int ele = 0; ele < e->nEles; ele++)
        if (geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) nElesBT[e->etype]--;
    }
  }

  unsigned int nCells = 0;
  unsigned int nPts = 0;

  for (auto e : elesObjs)
  {
    nCells += e->nSubelements * nElesBT[e->etype];
    nPts += e->nPpts * nElesBT[e->etype];
  }

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPts << "\" ";
  f << "NumberOfCells=\"" << nCells << "\">";
  f << std::endl;


  size_t b_offset = 0;
  /* Write solution information */
  f << "<PointData>" << std::endl;

  std::vector<std::string> var;
  if (input->equation == AdvDiff)
  {
    var = {"u"};
  }
  else if(input->equation == EulerNS)
  {
    if (geo.nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};
  }

  for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
  {
    f << "<DataArray type=\"Float64\" Name=\"" << var[n] << "\" ";
    f << "format=\"appended\" offset=\"" << b_offset << "\"/>"<< std::endl;
    b_offset += sizeof(unsigned int);
    for (auto e : elesObjs)
      b_offset += (e->nEles * e->nPpts * sizeof(double));
  }

  if (input->filt_on && input->sen_write)
  {
#ifdef _GPU
  for (auto e : elesObjs)
    filt.sensor[e->etype] = filt.sensor_d[e->etype];
#endif
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << filt.sensor[e->etype](ele) << " ";
        }
        f << std::endl;
      }
    }
    f << "</DataArray>" << std::endl;
  }

  if (input->motion)
  {
    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
    f << "format=\"ascii\">"<< std::endl;

    for (auto e : elesObjs)
    {
      e->get_grid_velocity_ppts();

      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          for (unsigned int dim = 0; dim < e->nDims; dim++)
          {
            f << std::scientific << std::setprecision(16);
            f << e->grid_vel_ppts(ppt, ele, dim);
            f  << " ";
          }
          if (e->nDims == 2) f << 0.0 << " ";
        }
        f << std::endl;
      }
    }
    f << "</DataArray>" << std::endl;
  }
  f << "</PointData>" << std::endl;

  /* Write plot point information (single precision) */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  f << "</Points>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (e->nEles * e->nPpts * 3 * sizeof(float));

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (e->nEles * e->nSubelements * e->nNodesPerSubelement * sizeof(unsigned int));



  f << "<DataArray type=\"UInt32\" Name=\"offsets\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (e->nEles * e->nSubelements * sizeof(unsigned int));

  f << "<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (e->nEles * e->nSubelements * sizeof(char));
  f << "</Cells>" << std::endl;

  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;

  /* Adding raw binary data as AppendedData*/
  f << "<AppendedData encoding=\"raw\">" << std::endl;
  f << "_"; // leading underscore

  /* Write solution data */
  /* Extrapolate solution to plot points */
  for (auto e : elesObjs)
  {
    auto &A = e->oppE_ppts(0, 0);
    auto &B = e->U_spts(0, 0, 0);
    auto &C = e->U_ppts(0, 0, 0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nPpts, 
        e->nEles * e->nVars, e->nSpts, 1.0, &A, e->oppE_ppts.ldim(), &B, 
        e->U_spts.ldim(), 0.0, &C, e->U_ppts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nPpts, 
          e->nEles * e->nVars, e->nSpts, 1.0, &A, e->oppE_ppts.ldim(), &B, 
          e->U_spts.ldim(), 0.0, &C, e->U_ppts.ldim());
#endif

    /* Apply squeezing if needed */
    if (input->squeeze)
    {
      e->compute_Uavg();

#ifdef _GPU
      e->Uavg = e->Uavg_d;
#endif

      e->poly_squeeze_ppts();
    }
  }

  unsigned int nBytes = 0;

  for (auto e : elesObjs)
    nBytes += e->nEles * e->nPpts * sizeof(double);

  for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
  {
    binary_write(f, nBytes);
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          binary_write(f, e->U_ppts(ppt, ele, n));
        }
      }
    }
  }

  /* Write plot point coordinates */
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += e->nEles * e->nPpts * 3 * sizeof(float);
  binary_write(f, nBytes);

  double dzero = 0.0;

  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
      {
        binary_write(f, (float) e->coord_ppts(ppt, ele, 0));
        binary_write(f, (float) e->coord_ppts(ppt, ele, 1));
        if (geo.nDims == 2)
          binary_write(f, 0.0f);
        else
          binary_write(f, (float) e->coord_ppts(ppt, ele, 2));
      }
    }
  }

  /* Write cell information */
  // Write connectivity
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += e->nEles * e->nSubelements * e->nNodesPerSubelement * sizeof(unsigned int);
  binary_write(f, nBytes);

  int shift = 0; // To account for blanked elements
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        for (unsigned int i = 0; i < e->nNodesPerSubelement; i++)
        {
          binary_write(f, e->ppt_connect(i, subele) + shift);
        }
      }
      shift += e->nPpts;
    }
  }

  // Offsets
  nBytes = 0;
  for (auto e: elesObjs)
    nBytes += e->nEles * e->nSubelements * sizeof(unsigned int);

  binary_write(f, nBytes);

  unsigned int offset = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        offset += e->nNodesPerSubelement;
        binary_write(f, offset);
      }
    }
  }

  // Types
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += e->nEles * e->nSubelements * sizeof(char);
  binary_write(f, nBytes);

  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        if (e->etype == QUAD)
          binary_write(f, (char) 9);
        else if (e->etype == TRI)
          binary_write(f, (char) 5);
        else if (e->etype == HEX)
          binary_write(f, (char) 12);
      }
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

  std::map<ELE_TYPE, unsigned int> nElesBT = geo.nElesBT;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (auto e : elesObjs)
    {
      for (int ele = 0; ele < e->nEles; ele++)
        if (geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) nElesBT[e->etype]--;
    }
  }

  unsigned int nCells = 0;
  unsigned int nPts = 0;

  for (auto e : elesObjs)
  {
    nCells += e->nSubelements * nElesBT[e->etype];
    nPts += e->nPpts * nElesBT[e->etype];
  }

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;
  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPts << "\" ";
  f << "NumberOfCells=\"" << nCells << "\">";
  f << std::endl;
  
  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl; 

  for (auto e : elesObjs)
  {
    if (e->nDims == 2)
    {
      // TODO: Change order of ppt structures for better looping 
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << e->coord_ppts(ppt, ele, 0) << " ";
          f << e->coord_ppts(ppt, ele, 1) << " ";
          f << 0.0 << std::endl;
        }
      }
    }
    else
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << e->coord_ppts(ppt, ele, 0) << " ";
          f << e->coord_ppts(ppt, ele, 1) << " ";
          f << e->coord_ppts(ppt, ele, 2) << std::endl;
        }
      }
    }
  }
  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  int shift = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        for (unsigned int i = 0; i < e->nNodesPerSubelement; i++)
        {
          f << e->ppt_connect(i, subele) + shift << " ";
        }
        f << std::endl;
      }

      shift += e->nPpts;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        offset += e->nNodesPerSubelement;
        f << offset << " ";
      }
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        if (e->etype == QUAD)
          f << 9 << " ";
        else if (e->etype == TRI)
          f << 5 << " ";
        else if (e->etype == HEX)
          f << 12 << " ";
      }
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write color information */
  f << "<PointData>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"color\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
      {
        f << std::scientific << std::setprecision(16) << geo.ele_colorBT[e->etype](ele);
        f  << " ";
      }
      f << std::endl;
    }
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

  auto e = elesObjs[0];

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
  int nFacesEle = geo.nFacesPerEleBT[e->etype];

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
    if (input->equation == AdvDiff)
    {
      f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
      f << std::endl;
    }
    else if (input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (e->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (unsigned int n = 0; n < e->nVars; n++)
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

  if (e->nDims == 2)
  {
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << e->coord_ppts(ppt, ele, 0) << " ";
        f << e->coord_ppts(ppt, ele, 1) << " ";
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
        f << e->coord_ppts(ppt, ele, 0) << " ";
        f << e->coord_ppts(ppt, ele, 1) << " ";
        f << e->coord_ppts(ppt, ele, 2) << std::endl;
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

  if (input->equation == AdvDiff)
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
        f << std::scientific << std::setprecision(16) << e->U_ppts(ppt, ele, 0);
        f  << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }
  else if(input->equation == EulerNS)
  {
    std::vector<std::string> var;
    if (e->nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};

    for (int n = 0; n < e->nVars; n++)
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
          f << e->U_ppts(ppt, ele, n);
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
        f << filt.sensor[e->etype](ele) << " ";
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
        for (unsigned int dim = 0; dim < e->nDims; dim++)
        {
          f << std::scientific << std::setprecision(16);
          f << e->grid_vel_ppts(ppt, ele, dim);
          f  << " ";
        }
        if (e->nDims == 2) f << 0.0 << " ";
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
  if (geo.ele_set.count(TET))
    ThrowException("Surface write not implemented for triangular faces.");

  auto e = elesObjs[0];

#ifdef _GPU
  e->U_spts = e->U_spts_d;
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
  int nFacesEle = geo.nFacesPerEleBT[e->etype];

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
    e->update_plot_point_coords();
#ifdef _GPU
    e->grid_vel_nodes = e->grid_vel_nodes_d;
#endif
    e->get_grid_velocity_ppts();
  }

  /* Extrapolate solution to plot points */
  auto &A = e->oppE_ppts(0, 0);
  auto &B = e->U_spts(0, 0, 0);
  auto &C = e->U_ppts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nPpts,
                    e->nEles * e->nVars, e->nSpts, 1.0, &A, e->oppE_ppts.ldim(), &B,
                    e->U_spts.ldim(), 0.0, &C, e->U_ppts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nPpts,
              e->nEles * e->nVars, e->nSpts, 1.0, &A, e->oppE_ppts.ldim(), &B,
              e->U_spts.ldim(), 0.0, &C, e->U_ppts.ldim());
#endif

  /* Apply squeezing if needed */
  if (input->squeeze)
  {
    e->compute_Uavg();

#ifdef _GPU
    e->Uavg = e->Uavg_d;
#endif

    e->poly_squeeze_ppts();
  }

#ifdef _GPU
  if (input->filt_on && input->sen_write)
    //filt.sensor = filt.sensor_d;
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
      if (input->equation == AdvDiff)
      {
        f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
        f << std::endl;
      }
      else if (input->equation == EulerNS)
      {
        std::vector<std::string> var;
        if (e->nDims == 2)
          var = {"rho", "xmom", "ymom", "energy"};
        else
          var = {"rho", "xmom", "ymom", "zmom", "energy"};

        for (unsigned int n = 0; n < e->nVars; n++)
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

    if (e->nDims == 2)
    {
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << e->coord_ppts(ppt, ele, 0) << " ";
          f << e->coord_ppts(ppt, ele, 1) << " ";
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
          f << e->coord_ppts(ppt, ele, 0) << " ";
          f << e->coord_ppts(ppt, ele, 1) << " ";
          f << e->coord_ppts(ppt, ele, 2) << std::endl;
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

    if (input->equation == AdvDiff)
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
          f << std::scientific << std::setprecision(16) << e->U_ppts(ppt, ele, 0);
          f  << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
    else if(input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (e->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (int n = 0; n < e->nVars; n++)
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
            f << e->U_ppts(ppt, ele, n);
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
          f << filt.sensor[e->etype](ele) << " ";
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
          for (unsigned int dim = 0; dim < e->nDims; dim++)
          {
            f << std::scientific << std::setprecision(16);
            f << e->grid_vel_ppts(ppt, ele, dim);
            f  << " ";
          }
          if (e->nDims == 2) f << 0.0 << " ";
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
  auto e = elesObjs[0];
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
    unsigned int N = e->nSpts * e->nVars;
    hsize_t dims[3] = {geo.nEles, N, N};
    DataSpace dspaceU(3, dims);

    std::string name = "LHS_" + std::to_string(iter);
    DataSet dset = file.createDataSet(name, PredType::NATIVE_DOUBLE, dspaceU);
    dset.write(e->LHSs[0].data(), PredType::NATIVE_DOUBLE, dspaceU);

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
  for (auto e : elesObjs)
  {
    e->divF_spts = e->divF_spts_d;
    e->dt = e->dt_d;
  }
#endif

  // HACK: Change nStages to compute the correct residual
  if (input->dt_scheme == "MCGS")
  {
    input->nStages = 1;
  }

  std::vector<double> res(elesObjs[0]->nVars,0.0);

  unsigned int nEles = 0;

  for (auto e : elesObjs)
  {
    for (unsigned int n = 0; n < e->nVars; n++)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          if (input->res_type == 0)
            res[n] = std::max(res[n], std::abs(e->divF_spts(spt,ele,n,0)
                                               / e->jaco_det_spts(spt, ele)));

          else if (input->res_type == 1)
            res[n] += std::abs(e->divF_spts(spt,ele,n,0)
                               / e->jaco_det_spts(spt, ele));

          else if (input->res_type == 2)
            res[n] += e->divF_spts(spt,ele,n,0) * e->divF_spts(spt,ele,n,0)
                    / (e->jaco_det_spts(spt, ele) * e->jaco_det_spts(spt, ele));
        }
        nEles++;
      }
    }
  }

  unsigned int nDoF = 0;
  for (auto e : elesObjs)
    nDoF += (e->nSpts * e->nEles);

  // HACK: Change nStages back
  if (input->dt_scheme == "MCGS")
  {
    input->nStages = 2;
  }

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
#endif

  double minDT = INFINITY; double maxDT = 0.0; 

  if (input->dt_type == 2)
  {
    for (auto e : elesObjs)
    {
      minDT = std::min(minDT, *std::min_element(e->dt.data(), e->dt.data() + e->nEles));
      maxDT = std::max(maxDT, *std::max_element(e->dt.data(), e->dt.data() + e->nEles));
    }

#ifdef _MPI
    if (input->rank == 0)
    {
      MPI_Reduce(MPI_IN_PLACE, &minDT, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(MPI_IN_PLACE, &maxDT, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
    else
    {
      MPI_Reduce(&minDT, &minDT, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(&maxDT, &maxDT, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
#endif
  }

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

      std::cout << "dt: " <<  minDT << " (min) ";
      std::cout << maxDT << " (max)";
    }
    else
    {
      std::cout << "dt: " << elesObjs[0]->dt(0);
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
  for (auto e : elesObjs)
  {
    e->U_fpts = e->U_fpts_d;
    if (input->viscous)
      e->dU_fpts = e->dU_fpts_d;
  }
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
  for (unsigned int dim = 0; dim < geo.nDims; dim++)
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
    MPI_Reduce(MPI_IN_PLACE, force_conv.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, force_visc.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(force_conv.data(), force_conv.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(force_visc.data(), force_visc.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
#endif

  /* Get angle of attack (and sideslip) */
  double aoa = std::atan2(input->V_fs(1), input->V_fs(0));
  double aos = 0.0;
  if (geo.nDims == 3)
    aos = std::atan2(input->V_fs(2), input->V_fs(0));

  if (input->rank == 0)
  {
    if (geo.nDims == 2)
    {
      CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
      CD_conv = force_conv[0] * std::cos(aoa) + force_conv[1] * std::sin(aoa);
      CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
      CD_visc = force_visc[0] * std::cos(aoa) + force_visc[1] * std::sin(aoa);
    }
    else if (geo.nDims == 3)
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
  for (auto e : elesObjs)
  {
    e->U_spts = e->U_spts_d;
    if (input->viscous)
      e->dU_spts = e->dU_spts_d;
  }
#endif

  std::vector<double> l2_error(2,0.0);
  double vol = 0;

  for (auto e : elesObjs)
  {
    /* Extrapolate solution to quadrature points */
    auto &A = e->oppE_qpts(0, 0);
    auto &B = e->U_spts(0, 0, 0);
    auto &C = e->U_qpts(0, 0, 0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nQpts, 
        e->nEles * e->nVars, e->nSpts, 1.0, &A, e->oppE_qpts.ldim(), &B, 
        e->U_spts.ldim(), 0.0, &C, e->U_qpts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nQpts, 
        e->nEles * e->nVars, e->nSpts, 1.0, &A, e->U_qpts.ldim(), &B, 
        e->U_spts.ldim(), 0.0, &C, e->U_qpts.ldim());
#endif

    /* Extrapolate derivatives to quadrature points */
    if (input->viscous)
    {
      for (unsigned int dim = 0; dim < e->nDims; dim++)
      {
        auto &A = e->oppE_qpts(0, 0);
        auto &B = e->dU_spts(0, 0, 0, dim);
        auto &C = e->dU_qpts(0, 0, 0, dim);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nQpts,
                          e->nEles * e->nVars, e->nSpts, 1.0, &A, e->U_qpts.ldim(), &B,
                          e->U_spts.ldim(), 0.0, &C, e->U_qpts.ldim());
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, e->nQpts,
                    e->nEles * e->nVars, e->nSpts, 1.0, &A, e->U_qpts.ldim(), &B,
                    e->U_spts.ldim(), 0.0, &C, e->U_qpts.ldim());
#endif

      }
    }

    unsigned int n = input->err_field;
    std::vector<double> dU_true(2, 0.0), dU_error(2, 0.0);
#pragma omp for 
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int qpt = 0; qpt < e->nQpts; qpt++)
      {
        double U_true = 0.0;

        if (e->nDims == 2)
        {
          /* Compute true solution and derivatives */
          if (input->test_case == 3) // Isentropic Bump
          {
            U_true = input->P_fs / std::pow(input->rho_fs, input->gamma);
          }
          else 
          {
            U_true = compute_U_true(e->coord_qpts(qpt,ele,0), e->coord_qpts(qpt,ele,1), 0, 
                flow_time, n, input);
          }

          if (input->viscous)
          {
            dU_true[0] = compute_dU_true(e->coord_qpts(qpt,ele,0), e->coord_qpts(qpt,ele,1), 0,
                                         flow_time, n, 0, input);
            dU_true[1] = compute_dU_true(e->coord_qpts(qpt,ele,0), e->coord_qpts(qpt,ele,1), 0,
                                         flow_time, n, 1, input);
          }

        }
        else if (e->nDims == 3)
        {
          ThrowException("Under construction!");
        }

        /* Compute errors */
        double U_error;
        if (input->test_case == 3) // Isentropic bump
        {
          double momF = 0.0;
          for (unsigned int dim = 0; dim < e->nDims; dim ++)
          {
            momF += e->U_qpts(qpt, ele, dim + 1) * e->U_qpts(qpt, ele, dim + 1);
          }

          momF /= e->U_qpts(qpt, ele, 0);

          double P = (input->gamma - 1.0) * (e->U_qpts(qpt, ele, 3) - 0.5 * momF);

          U_error = (U_true - P/std::pow(e->U_qpts(qpt, ele, 0), input->gamma)) / U_true;
          vol += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele); 
        }
        else
        {
          U_error = U_true - e->U_qpts(qpt, ele, n);
          if (input->viscous)
          {
            dU_error[0] = dU_true[0] - e->dU_qpts(qpt, ele, n, 0); 
            dU_error[1] = dU_true[1] - e->dU_qpts(qpt, ele, n, 1);
          }
          vol = 1;
        }

        l2_error[0] += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele) * U_error * U_error; 
        l2_error[1] += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele) * (U_error * U_error +
            dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1]); 
      }
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
  for (unsigned int dim = 0; dim < geo.nDims; dim++)
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
        for(unsigned int dim = 0; dim < geo.nDims; dim++)
          *cp_file << std::scientific << faces->coord(fpt, dim) << " ";
        *cp_file << std::scientific << CP << std::endl;
      }

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        //TODO: need to fix quadrature weights for mixed element cases!
        force_conv[dim] += elesObjs[0]->weights_fpts(idx) * PL *
          faces->norm(fpt, dim) * faces->dA(fpt);
      }

      if (input->viscous)
      {
        if (geo.nDims == 2)
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

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force_visc[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
              faces->dA(fpt);

        }
        else if (geo.nDims == 3)
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

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force_visc[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
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
      //TODO: need to fix quadrature weights for mixed element cases!
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
        force[dim] = elesObjs[0]->weights_fpts(idx) * PL *
          faces->norm(fpt, dim) * faces->dA(fpt);

      if (input->viscous)
      {
        if (geo.nDims == 2)
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

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] * faces->dA(fpt);
        }
        else if (geo.nDims == 3)
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

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
              faces->dA(fpt);
        }

      }

      // Add fpt's contribution to total force and moment
      for (unsigned int d = 0; d < geo.nDims; d++)
        tot_force[d] += force[d];

      if (geo.nDims == 3)
      {
        for (unsigned int d = 0; d < geo.nDims; d++)
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

void FRSolver::filter_solution()
{
  if (input->filt_on)
  {
    filt.apply_sensor();
    filt.apply_expfilter();
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

  for (auto e : elesObjs)
    e->move(faces);
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
