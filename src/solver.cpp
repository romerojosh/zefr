#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <vector>

#include "cblas.h"

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
#include "spmatrix.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "spmatrix_gpu.h"
#include "solver_kernels.h"
#include "cublas_v2.h"
#endif

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
#endif

//FRSolver::FRSolver(const InputStruct *input, int order)
FRSolver::FRSolver(InputStruct *input, int order)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;
}

void FRSolver::setup()
{
  if (input->rank == 0) std::cout << "Reading mesh: " << input->meshfile << std::endl;
  geo = process_mesh(input, order, input->nDims);

  if (input->rank == 0) std::cout << "Setting up elements and faces..." << std::endl;

  if (input->nDims == 2)
    eles = std::make_shared<Quads>(&geo, input, order);
  else if (input->nDims == 3)
    eles = std::make_shared<Hexas>(&geo, input, order);

  faces = std::make_shared<Faces>(&geo, input);

  faces->setup(eles->nDims, eles->nVars);
  eles->setup(faces);

  if (input->rank == 0) std::cout << "Setting up timestepping..." << std::endl;
  setup_update();

  if (input->rank == 0) std::cout << "Setting up output..." << std::endl;
  setup_output();

  if (input->rank == 0) std::cout << "Initializing solution..." << std::endl;
  initialize_U();

  if (input->restart)
  {
    if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_file +" ..." << std::endl;
    restart(input->restart_file);
  }

  if (input->filt_on)
  {
    if (input->rank == 0) std::cout << "Setting up filter..." << std::endl;
    filt.setup(input, *this);
  }
  
#ifdef _GPU
  solver_data_to_device();
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
    rk_alpha(0) = 1./4; rk_alpha(1) = 1./3.; 
    rk_alpha(2) = 1./2.; rk_alpha(3) = 1.0;
  }
  else if (input->dt_scheme == "BDF1" || input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
    // HACK: (nStages = 1) doesn't work, fix later
    nStages = 2;

    /* Setup element colors */
    ele_color.assign({eles->nEles});
    if (input->dt_scheme == "LUJac")
    {
      nColors = 1;
      ele_color.fill(1);
    }
    else if (input->dt_scheme == "LUSGS")
    {
      nColors = 2;
      ele_color(0) = 1;
      std::queue<unsigned int> Q;
      Q.push(0);
      while (!Q.empty())
      {
        unsigned int ele1 = Q.front();
        Q.pop();

        /* Determine opposite color */
        unsigned int color = ele_color(ele1);
        if (color == 1)
        {
          color = 2;
        }
        else
        {
          color = 1;
        }

        /* Color neighbors */
        for (unsigned int face = 0; face < eles->nFaces; face++)
        {
          int ele2 = geo.ele_adj(face, ele1);
          if (ele2 != -1 && ele_color(ele2) == 0)
          {
            ele_color(ele2) = color;
            Q.push(ele2);
          }
        }
      }
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
    std::string cmd = "mkdir " + input->output_prefix;
    system(cmd.c_str());
  }

  if (eles->nDims == 2)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 4;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({4, eles->nSubelements});

    /* Setup plot "subelement" connectivity */
    std::vector<unsigned int> nd(4,0);

    unsigned int ele = 0;
    nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;

    for (unsigned int i = 0; i < nSubelements1D; i++)
    {
      for (unsigned int j = 0; j < nSubelements1D; j++)
      {
        for (unsigned int node = 0; node < 4; node ++)
        {
          geo.ppt_connect(node, ele) = nd[node] + j;
        }

        ele++;
      }

      for (unsigned int node = 0; node < 4; node ++)
        nd[node] += nSubelements1D + 1;
    }
  }
  else if (eles->nDims == 3)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 8;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({8, eles->nSubelements});

    /* Setup plot "subelement" connectivity */
    std::vector<unsigned int> nd(8,0);

    unsigned int ele = 0;
    nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;
    nd[4] = (nSubelements1D + 1) * (nSubelements1D + 1); nd[5] = nd[4] + 1; 
    nd[6] = nd[4] + nSubelements1D + 2; nd[7] = nd[4] + nSubelements1D + 1;

    for (unsigned int i = 0; i < nSubelements1D; i++)
    {
      for (unsigned int j = 0; j < nSubelements1D; j++)
      {
        for (unsigned int k = 0; k < nSubelements1D; k++)
        {
          for (unsigned int node = 0; node < 8; node ++)
          {
            geo.ppt_connect(node, ele) = nd[node] + k;
          }

          ele++;
        }

        for (unsigned int node = 0; node < 8; node ++)
          nd[node] += (nSubelements1D + 1);

      }

      for (unsigned int node = 0; node < 8; node ++)
        nd[node] += (nSubelements1D + 1);
    }

  }

}

void FRSolver::restart(std::string restart_file)
{
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
  mdvector<double> U_restart, oppRestart;

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

    if (param == "<PointData>")
    {

      unsigned int nSpts1D_restart = order_restart + 1;
      unsigned int nSpts2D_restart = nSpts1D_restart * nSpts1D_restart;
      unsigned int nPpts1D = nSpts1D_restart + 2;
      unsigned int nPpts2D = nPpts1D * nPpts1D;

      unsigned int nSpts_restart = (unsigned int) std::pow(nSpts1D_restart, input->nDims);
      unsigned int nPpts = (unsigned int) std::pow(nPpts1D, input->nDims);

      U_restart.assign({nSpts_restart, eles->nEles, eles->nVars});

      /* Setup extrapolation operator from restart points */
      oppRestart.assign({eles->nSpts, nSpts_restart});
      auto loc_spts_restart_1D = Gauss_Legendre_pts(order_restart + 1); 

      std::vector<double> loc(input->nDims);
      for (unsigned int rpt = 0; rpt < nSpts_restart; rpt++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          for (unsigned int dim = 0; dim < input->nDims; dim++)
            loc[dim] = eles->loc_spts(spt , dim);

          if (input->nDims == 2)
          {
            int i = rpt % nSpts1D_restart;
            int j = rpt / nSpts1D_restart;
            oppRestart(spt,rpt) = Lagrange(loc_spts_restart_1D, i, loc[0]) * 
                                  Lagrange(loc_spts_restart_1D, j, loc[1]);
          }
          else
          {
            int i = rpt % nSpts1D_restart;
            int j = (rpt / nSpts1D_restart) % nSpts1D_restart;
            int k = rpt / nSpts2D_restart;
            oppRestart(spt,rpt) = Lagrange(loc_spts_restart_1D, i, loc[0]) * 
                                  Lagrange(loc_spts_restart_1D, j, loc[1]) *
                                  Lagrange(loc_spts_restart_1D, k, loc[2]);
          }
        }
      }

      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        std::getline(f,line);
        std::getline(f,line);

        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          unsigned int spt = 0;
          for (unsigned int ppt = 0; ppt < nPpts; ppt++)
          {
            f >> val;

            /* Logic to deal with extra plot point (corner nodes and flux points). */
            if (input->nDims == 2)
            {
              if (ppt < nPpts1D || ppt > nPpts1D * (nPpts1D-1) || ppt%nPpts1D == 0 || 
                  (ppt+1)%nPpts1D == 0)
                continue;
            }
            else
            {
              int shift = (ppt / nPpts2D) * nPpts2D;
              if (ppt < nPpts2D || ppt < nPpts1D + shift || ppt > nPpts1D * (nPpts1D-1) + shift || 
                  (ppt-shift) % nPpts1D == 0 || (ppt+1-shift)%nPpts1D == 0 || ppt > nPpts2D * (nPpts2D - 1))
                continue;
            }

            U_restart(spt, ele, n) = val;
            spt++;
          }
        }
        std::getline(f,line);
      }

      /* Extrapolate values from restart points to solution points */
      auto &A = oppRestart(0, 0);
      auto &B = U_restart(0, 0, 0);
      auto &C = eles->U_spts(0, 0, 0);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nSpts_restart, 1.0, &A, eles->nSpts, &B, 
          nSpts_restart, 0.0, &C, eles->nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, 
          eles->nEles * eles->nVars, nSpts_restart, 1.0, &A, eles->nSpts, &B, 
          nSpts_restart, 0.0, &C, eles->nSpts);
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
  eles->oppD_fpts_d = eles->oppD_fpts;
  eles->oppDiv_fpts_d = eles->oppDiv_fpts;

  /* If using multigrid, copy relevant operators */
  if (input->p_multi)
  {
    eles->oppPro_d = eles->oppPro;
    eles->oppRes_d = eles->oppRes;
  }

  /* Solver data structures */
  U_ini_d = U_ini;
  rk_alpha_d = rk_alpha;
  rk_beta_d = rk_beta;
  dt_d = dt;

  /* Implicit solver data structures */
  eles->deltaU_d = eles->deltaU;
  eles->RHS_d = eles->RHS;

  if (input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
    eles->LHS_d = eles->LHS;
    eles->LU_pivots_d = eles->LU_pivots;
    eles->LU_info_d = eles->LU_info;
    eles->LHS_tempSF.assign({eles->nSpts, eles->nVars, eles->nFpts, eles->nVars, eles->nEles});
    eles->LHS_tempSF_d = eles->LHS_tempSF;

    /* For cublas batched LU: Setup and transfer array of GPU pointers to 
     * LHS matrices and RHS vectors */
    unsigned int N = eles->nSpts * eles->nVars;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      eles->LHS_ptrs(ele) = eles->LHS_d.data() + ele * (N * N);
      eles->RHS_ptrs(ele) = eles->RHS_d.data() + ele * N;
    }

    /* Additional pointers for batched DGEMM */
    for (unsigned int i = 0; i < eles->nEles * eles->nVars; i++)
    {
      eles->LHS_subptrs(i) = eles->LHS_d.data() + i * N * eles->nSpts;
      eles->LHS_tempSF_subptrs(i) = eles->LHS_tempSF_d.data() + i * N * eles->nFpts;
      eles->oppE_ptrs(i) = eles->oppE_d.data();
    }

    eles->LHS_ptrs_d = eles->LHS_ptrs;
    eles->LHS_subptrs_d = eles->LHS_subptrs;
    eles->RHS_ptrs_d = eles->RHS_ptrs;
    eles->LHS_tempSF_subptrs_d = eles->LHS_tempSF_subptrs;
    eles->oppE_ptrs_d = eles->oppE_ptrs;

  }

  /* Solution data structures (element local) */
  eles->U_spts_d = eles->U_spts;
  eles->U_fpts_d = eles->U_fpts;
  eles->Ucomm_d = eles->Ucomm;
  eles->Uavg_d = eles->Uavg;
  eles->weights_spts_d = eles->weights_spts;
  eles->dU_spts_d = eles->dU_spts;
  eles->dU_fpts_d = eles->dU_fpts;
  eles->Fcomm_d = eles->Fcomm;
  eles->F_spts_d = eles->F_spts;
  eles->divF_spts_d = eles->divF_spts;
  eles->jaco_spts_d = eles->jaco_spts;
  eles->inv_jaco_spts_d = eles->inv_jaco_spts;
  eles->jaco_det_spts_d = eles->jaco_det_spts;
  eles->vol_d = eles->vol;

  /* Implicit flux derivative data structures (element local) */
  eles->dFdU_spts_d = eles->dFdU_spts;
  eles->dFcdU_fpts_d = eles->dFcdU_fpts;

  /* Solution data structures (faces) */
  faces->U_d = faces->U;
  faces->dU_d = faces->dU;
  faces->Fconv_d = faces->Fconv;
  faces->Fvisc_d = faces->Fvisc;
  faces->P_d = faces->P;
  faces->Ucomm_d = faces->Ucomm;
  faces->Fcomm_d = faces->Fcomm;
  faces->Fcomm_temp_d = faces->Fcomm_temp;
  faces->norm_d = faces->norm;
  faces->dA_d = faces->dA;
  faces->waveSp_d = faces->waveSp;
  faces->LDG_bias_d = faces->LDG_bias;
  faces->bc_bias_d = faces->bc_bias;

  /* Implicit flux derivative data structures (faces) */
  faces->dFdUconv_d = faces->dFdUconv;

  /* Additional data */
  /* Geometry */
  geo.fpt2gfpt_d = geo.fpt2gfpt;
  geo.fpt2gfpt_slot_d = geo.fpt2gfpt_slot;
  geo.gfpt2bnd_d = geo.gfpt2bnd;
  geo.per_fpt_list_d = geo.per_fpt_list;
  geo.coord_spts_d = geo.coord_spts;

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

void FRSolver::compute_residual(unsigned int stage)
{
  /* Extrapolate solution to flux points */
  eles->extrapolate_U();

  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    eles->compute_Uavg();
    eles->poly_squeeze();
  }

  /* Copy flux point data from element local to face local storage */
  U_to_faces();

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  /* Compute convective flux at solution points */
  eles->compute_Fconv();

  /* If running inviscid, use this scheduling. */
  if(!input->viscous)
  {

#ifdef _MPI
  /* Compute convective flux and common flux at non-MPI flux points */
  faces->compute_Fconv(0, geo.nGfpts_int + geo.nGfpts_bnd);
  faces->compute_common_F(0, geo.nGfpts_int + geo.nGfpts_bnd);
  
  /* Receive U data */
  faces->recv_U_data();

  /* Complete computation on remaning flux points. */
  faces->compute_Fconv(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
  faces->compute_common_F(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
  /* Compute convective and common fluxes at flux points */
  faces->compute_Fconv(0, geo.nGfpts);
  faces->compute_common_F(0, geo.nGfpts);
#endif
  }

  /* If running viscous, use this scheduling */
  else
  {
#ifdef _MPI
    /* Compute common interface solution at non-MPI flux points */
    faces->compute_common_U(0, geo.nGfpts_int + geo.nGfpts_bnd);

    /* Receieve U data */
    faces->recv_U_data();

    /* Finish computation of common interface solution */
    faces->compute_common_U(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
    /* Compute common interface solution at flux points */
    faces->compute_common_U(0, geo.nGfpts);
#endif

    /* Copy solution data at flux points from face local to element local
     * storage */
    U_from_faces();

    /* Compute gradient of state variables at solution points */
    eles->compute_dU();

    /* Transform gradient of state variables to physical space from 
     * reference space */
    eles->transform_dU();

    /* Extrapolate solution gradient to flux points */
    eles->extrapolate_dU();

    /* Copy gradient data from element local to face local storage */
    dU_to_faces();

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* Compute viscous flux at solution points */
    eles->compute_Fvisc();

    /* Compute viscous and convective flux and common interface flux 
     * at non-MPI flux points */
    faces->compute_Fvisc(0, geo.nGfpts_int + geo.nGfpts_bnd);
    faces->compute_Fconv(0, geo.nGfpts_int + geo.nGfpts_bnd);
    faces->compute_common_F(0, geo.nGfpts_int + geo.nGfpts_bnd);

    /* Receive gradient data */
    faces->recv_dU_data();

    /* Complete computation of fluxes */
    faces->compute_Fvisc(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
    faces->compute_Fconv(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);
    faces->compute_common_F(geo.nGfpts_int + geo.nGfpts_bnd, geo.nGfpts);

#else
    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* Compute viscous flux at solution points */
    eles->compute_Fvisc();

    /* Compute viscous and convective flux and common interface fluxes 
     * at flux points*/ 
    faces->compute_Fvisc(0, geo.nGfpts);
    faces->compute_Fconv(0, geo.nGfpts);
    faces->compute_common_F(0, geo.nGfpts);
#endif

  }

  /* Transform fluxes from physical to reference space */
  eles->transform_flux();
  faces->transform_flux();

  /* Copy flux data from face local storage to element local storage */
  F_from_faces();

  /* Compute divergence of flux */
  eles->compute_divF(stage);

  /* Add source term (if required) */
  if (input->source)
    add_source(stage);
}

void FRSolver::compute_LHS()
{
#ifdef _GPU
  /* Copy new solution from GPU */
  // TODO: Temporary until placed in GPU
  eles->U_spts = eles->U_spts_d;
  faces->U = faces->U_d;
#endif

  /* Compute derivative of convective flux with respect to state variables 
   * at solution and flux points */
  eles->compute_dFdUconv();
  faces->compute_dFdUconv(0, geo.nGfpts);

#ifdef _GPU
  /* Copy new dFdUconv from GPU */
  // TODO: Temporary until placed in GPU
  eles->dFdU_spts = eles->dFdU_spts_d;
  faces->dFdUconv = faces->dFdUconv_d;
#endif

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
  if (input->dt_scheme == "BDF1")
  {
    eles->compute_globalLHS(dt);

#ifdef _GPU
    /* Copy to GPU */
    eles->GLHS_d.free_data();
    eles->GLHS_d = eles->GLHS;
#endif

  }
  else if (input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
#ifdef _CPU
    eles->compute_localLHS(dt);
#endif
#ifdef _GPU
    eles->compute_localLHS(dt_d);
#endif
    compute_LHS_LU();
  }
}

void FRSolver::compute_LHS_LU()
{

#ifdef _GPU
  unsigned int N = eles->nSpts * eles->nVars;

  /* Perform batched LU using cuBLAS */
  cublasDgetrfBatched_wrapper(N, eles->LHS_ptrs_d.data(), N, eles->LU_pivots_d.data(), eles->LU_info_d.data(), eles->nEles);
#endif

#ifdef _CPU
#ifndef _NO_TNT
  LUptrs.clear();
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    /* Copy LHS into TNT object */
    // TODO: Copy can now be removed
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
            A[i][j] = eles->LHS(si, ni, sj, nj, ele);
          }
        }
      }
    }

    /* Calculate and store LU object */
    LUptrs.push_back(std::make_shared<JAMA::LU<double>>(A));
  }
#endif
#endif
}

void FRSolver::compute_RHS(unsigned int color)
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (ele_color(ele) == color)
      {
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
  }
#endif

#ifdef _GPU
  compute_RHS_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, eles->RHS_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars);
#endif
}

#ifdef _CPU
void FRSolver::compute_RHS_source(const mdvector<double> &source, unsigned int color)
{
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (ele_color(ele) == color)
      {
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
}
#endif

#ifdef _GPU
void FRSolver::compute_RHS_source(const mdvector_gpu<double> &source, unsigned int color)
{
  compute_RHS_source_wrapper(eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, eles->RHS_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars);
}
#endif

void FRSolver::compute_deltaU(unsigned int color)
{
  if (input->dt_scheme == "BDF1")
  {
#ifdef _CPU
    ThrowException("Global BDF1 not supported on CPU.");
#endif

#ifdef _GPU
    compute_deltaU_globalLHS_wrapper(eles->GLHS_d, eles->deltaU_d, eles->RHS_d, GMRES_conv);
#endif

  }
  else if (input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
#ifdef _CPU
#ifndef _NO_TNT
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (ele_color(ele) == color)
      {
        /* Copy RHS into TNT object */
        unsigned int N = eles->nSpts * eles->nVars;
        TNT::Array1D<double> b(N);
        for (unsigned int n = 0; n < eles->nVars; n++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            unsigned int i = n * eles->nSpts + spt;
            b[i] = eles->RHS(spt, n, ele);
          }
        }

        /* Solve for deltaU */
        TNT::Array1D<double> x = LUptrs[ele]->solve(b);
        if (x.dim() == 0)
        {
          ThrowException("LU solve failed!");
        }

        /* Copy TNT object into deltaU */
        for (unsigned int n = 0; n < eles->nVars; n++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            unsigned int i = n * eles->nSpts + spt;
            eles->deltaU(spt, n, ele) = x[i];
          }
        }
      }


    }
    //ThrowException("PAUSE");
#endif
#endif

#ifdef _GPU
    /* Solve LU systems using batched cublas routine */
    unsigned int N = eles->nSpts * eles->nVars;
    int info;
    cublasDgetrsBatched_wrapper(N, 1, (const double**) eles->LHS_ptrs_d.data(), N, eles->LU_pivots_d.data(), 
        eles->RHS_ptrs_d.data(), N, &info, eles->nEles);

    if (info)
      ThrowException("cublasDgetrs failed. info = " + std::to_string(info));
#endif
  }
}

void FRSolver::compute_U(unsigned int color)
{
#ifdef _CPU
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      if (ele_color(ele) == color)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          eles->U_spts(spt, ele, n) += eles->deltaU(spt, n, ele);
        }
      }
    }
  }
#endif

#ifdef _GPU
  if (input->dt_scheme == "BDF1")
  {
    compute_U_wrapper(eles->U_spts_d, eles->deltaU_d, eles->nSpts, eles->nEles, eles->nVars);
  }
  else if (input->dt_scheme == "LUJac" or input->dt_scheme == "LUSGS")
  {
    /* Add RHS (which contains deltaU) to U */
    compute_U_wrapper(eles->U_spts_d, eles->RHS_d, eles->nSpts, eles->nEles, eles->nVars);
  }
#endif
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  eles->U_fpts.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->Ucomm.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->U_ppts.assign({eles->nPpts, eles->nEles, eles->nVars});
  eles->U_qpts.assign({eles->nQpts, eles->nEles, eles->nVars});
  eles->Uavg.assign({eles->nEles, eles->nVars});

  eles->F_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->F_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->Fcomm.assign({eles->nFpts, eles->nEles, eles->nVars});

  eles->dU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dU_qpts.assign({eles->nQpts, eles->nEles, eles->nVars, eles->nDims});

  eles->divF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, nStages});

  /* Allocate memory for implicit method data structures */
  if (input->dt_scheme == "BDF1" || input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
    /* Maximum number of unique matrices possible per element */
    unsigned int nMat = eles->nFaces + 1;

    eles->dFdU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims});
    eles->dFcdU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, 2});

    if(input->viscous)
    {
      nMat += eles->nFaces * (eles->nFaces - 1);

      eles->dUcdU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, 2});

      /* Note: nDimsi: Fx, Fy // nDimsj: dUdx, dUdy */
      eles->dFddU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims, eles->nDims});
      eles->dFcddU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nVars, eles->nDims, 2});
    }

    if (input->dt_scheme == "BDF1")
    {
      eles->LHS.assign({eles->nSpts, eles->nSpts, nMat});
    }
    else if (input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
    {
      eles->LHS.assign({eles->nSpts, eles->nVars, eles->nSpts, eles->nVars, eles->nEles});
      eles->LHS_tempSF.assign({eles->nSpts, eles->nVars, eles->nFpts, eles->nVars, eles->nEles});
      eles->LHS_ptrs.assign({eles->nEles});
      eles->RHS_ptrs.assign({eles->nEles});
      eles->LU_pivots.assign({eles->nSpts * eles->nVars * eles->nEles});
      eles->LU_info.assign({eles->nSpts * eles->nVars * eles->nEles});
      eles->LHS_subptrs.assign({eles->nEles * eles->nVars});
      eles->LHS_tempSF_subptrs.assign({eles->nEles * eles->nVars});
      eles->oppE_ptrs.assign({eles->nEles * eles->nVars});
    }
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
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);

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
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);
            double z = geo.coord_spts(spt, ele, 2);

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
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);

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

void FRSolver::U_to_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
        {
          if (input->viscous) // if viscous, put extrapolated solution into Ucomm
            eles->Ucomm(fpt, ele, n) = eles->U_fpts(fpt, ele, n);
          continue;
        }
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        faces->U(gfpt, n, slot) = eles->U_fpts(fpt, ele, n);
      }
    }
  }
#endif

#ifdef _GPU
  U_to_faces_wrapper(eles->U_fpts_d, faces->U_d, eles->Ucomm_d, geo.fpt2gfpt_d,
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts, eles->nDims,
      input->equation, input->viscous);

  check_error();
#endif
}

void FRSolver::U_from_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Ucomm(fpt, ele, n) = faces->Ucomm(gfpt, n, slot);
      }
    }
  }
#endif

#ifdef _GPU
  U_from_faces_wrapper(faces->Ucomm_d, eles->Ucomm_d, geo.fpt2gfpt_d,
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts,
      eles->nDims, input->equation);

  check_error();
#endif

}

void FRSolver::dU_to_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(4)
  for (unsigned int dim = 0; dim < eles->nDims; dim++) 
  {
    for (unsigned int n = 0; n < eles->nVars; n++) 
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfpt(fpt,ele);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          int slot = geo.fpt2gfpt_slot(fpt,ele);

          faces->dU(gfpt, n, dim, slot) = eles->dU_fpts(fpt, ele, n, dim);
        }
      }
    }
  }
#endif

#ifdef _GPU
  dU_to_faces_wrapper(eles->dU_fpts_d, faces->dU_d, geo.fpt2gfpt_d, geo.fpt2gfpt_slot_d, 
      eles->nVars, eles->nEles, eles->nFpts, eles->nDims, input->equation);

  check_error();
#endif
}

void FRSolver::F_from_faces()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Fcomm(fpt, ele, n) = faces->Fcomm(gfpt, n, slot);

      }
    }
  }
#endif

#ifdef _GPU
  /* Can reuse kernel here */
  U_from_faces_wrapper(faces->Fcomm_d, eles->Fcomm_d, geo.fpt2gfpt_d, 
      geo.fpt2gfpt_slot_d, eles->nVars, eles->nEles, eles->nFpts, 
      eles->nDims, input->equation);

  check_error();
#endif
}

void FRSolver::dFcdU_from_faces()
{
#pragma omp parallel for collapse(4)
  for (unsigned int nj = 0; nj < eles->nVars; nj++) 
  {
    for (unsigned int ni = 0; ni < eles->nVars; ni++) 
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
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

          //TODO: if gfpt > nGfpts_int (and < nGfpts_int + nGfpts_bnd), it is a boundary. You can add boundary contributions here I think.

          eles->dFcdU_fpts(fpt, ele, ni, nj, 0) = faces->dFcdU(gfpt, ni, nj, slot, slot);
          eles->dFcdU_fpts(fpt, ele, ni, nj, 1) = faces->dFcdU(gfpt, ni, nj, notslot, slot);
        }
      }
    }
  }

  if(input->viscous)
  {
#pragma omp parallel for collapse(4)
    for (unsigned int nj = 0; nj < eles->nVars; nj++) 
    {
      for (unsigned int ni = 0; ni < eles->nVars; ni++) 
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
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
}

void FRSolver::add_source(unsigned int stage)
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele =0; ele < eles->nEles; ele++)
    {
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
          double x = geo.coord_spts(spt, ele, 0);
          double y = geo.coord_spts(spt, ele, 1);
          double z = 0;
          if (eles->nDims == 3)
            z = geo.coord_spts(spt, ele, 2);

          eles->divF_spts(spt, ele, n, stage) += compute_source_term(x, y, z, flow_time, n, input) * 
            eles->jaco_det_spts(spt, ele);
      }
    }
  }

#endif

#ifdef _GPU
  add_source_wrapper(eles->divF_spts_d, eles->jaco_det_spts_d, geo.coord_spts_d, eles->nSpts, eles->nEles,
      eles->nVars, eles->nDims, input->equation, flow_time, stage);
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
  if (input->dt_scheme != "BDF1" && input->dt_scheme != "LUJac" && input->dt_scheme != "LUSGS")
  {
#ifdef _CPU
    U_ini = eles->U_spts;
#endif

#ifdef _GPU
    device_copy(U_ini_d, eles->U_spts_d, eles->U_spts_d.get_nvals());
#endif

    unsigned int nSteps = (input->dt_scheme == "RKj") ? nStages : nStages - 1;

    /* Main stage loop. Complete for Jameson-style RK timestepping */
    for (unsigned int stage = 0; stage < nSteps; stage++)
    {
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
#pragma omp parallel for collapse(3)
        for (unsigned int n = 0; n < eles->nVars; n++)
          for (unsigned int ele = 0; ele < eles->nEles; ele++)
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
      else
      {
#pragma omp parallel for collapse(3)
        for (unsigned int n = 0; n < eles->nVars; n++)
          for (unsigned int ele = 0; ele < eles->nEles; ele++)
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
#endif

#ifdef _GPU
      /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
      unsigned int last_stage = (input->dt_scheme == "RKj") ? nStages + 1 : nStages;

      if (source.size() == 0)
      {
        RK_update_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
            rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
            input->equation, stage, last_stage, false);
      }
      else
      {
        RK_update_source_wrapper(eles->U_spts_d, U_ini_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, 
            rk_alpha_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims, 
            input->equation, stage, last_stage, false);
      }
      check_error();
#endif
    }

    /* Final stage combining residuals for full Butcher table style RK timestepping*/
    if (input->dt_scheme != "RKj")
    {
      compute_residual(nStages-1);
#ifdef _CPU
      eles->U_spts = U_ini;
#endif
#ifdef _GPU
      device_copy(eles->U_spts_d, U_ini_d, eles->U_spts_d.get_nvals());
#endif

#ifdef _CPU
      for (unsigned int stage = 0; stage < nStages; stage++)
      {
        if (source.size() == 0)
        {
#pragma omp parallel for collapse(3)
          for (unsigned int n = 0; n < eles->nVars; n++)
            for (unsigned int ele = 0; ele < eles->nEles; ele++)
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
        else
        {
#pragma omp parallel for collapse(3)
          for (unsigned int n = 0; n < eles->nVars; n++)
            for (unsigned int ele = 0; ele < eles->nEles; ele++)
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
#endif

#ifdef _GPU
      if (source.size() == 0)
      {
        RK_update_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, eles->jaco_det_spts_d, dt_d, 
            rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
            input->equation, 0, nStages, true);
      }
      else
      {
        RK_update_source_wrapper(eles->U_spts_d, eles->U_spts_d, eles->divF_spts_d, source, eles->jaco_det_spts_d, dt_d, 
            rk_beta_d, input->dt_type, eles->nSpts, eles->nEles, eles->nVars, eles->nDims,
            input->equation, 0, nStages, true);
      }

      check_error();
#endif
    }
  }

  else if (input->dt_scheme == "BDF1")
  {
#ifdef _CPU
    ThrowException("BDF1 not implemented on CPU yet.");
#endif

#ifdef _GPU
    compute_residual(0);

    /* Freeze Jacobian */
    int iter = current_iter - restart_iter;
    if (iter%input->Jfreeze_freq == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
      dt = dt_d;

      /* Compute SER time step growth */
      if (input->SER)
      {
        eles->divF_spts = eles->divF_spts_d;
        compute_SER_dt();
        dt_d = dt;
      }

      /* Compute LHS implicit Jacobian */
      compute_LHS();
    }

    /* Prepare RHS vector */
    if (source.size() == 0)
    {
      compute_RHS();
    }
    else
    {
      compute_RHS_source(source);
    }

    /* Solve system for deltaU */
    eles->deltaU.fill(0); //TODO: Whoops! We shouldn't be resetting the guess to zero every iteration!
    eles->deltaU_d = eles->deltaU;
    compute_deltaU_globalLHS_wrapper(eles->GLHS_d, eles->deltaU_d, eles->RHS_d, GMRES_conv);

    /* Add deltaU to solution */
    compute_U();

#endif
  }

  else if (input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {

#ifdef _GPU
    if (nColors > 1)
      ThrowException("Only block-jacobi supported on GPU currently!");
#endif

    for (unsigned int color = 1; color <= nColors; color++)
    {
      compute_residual(0);

      /* Freeze Jacobian */
      int iter = current_iter - restart_iter;
      if (color == 1 && iter%input->Jfreeze_freq == 0)
      {
        // TODO: Revisit this as it is kind of expensive.
        if (input->dt_type != 0)
        {
          compute_element_dt();
#ifdef _GPU
          dt = dt_d; // Copy timestep out to CPU for LHS computation
#endif
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

      /* Prepare RHS vector */
      if (source.size() == 0)
      {
        compute_RHS(color);
      }
      else
      {
        compute_RHS_source(source, color);
      }

      /* Solve system for deltaU */
      compute_deltaU(color);

      /* Add deltaU to solution */
      compute_U(color);
    }
  }

  flow_time += dt(0);
  current_iter++;
}

void FRSolver::compute_element_dt()
{
#ifdef _CPU
#pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  { 
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

    /* CFL-estimate used by Liang, Lohner, and others. Factor of 2 to be 
     * consistent with 1D CFL estimates. */
    dt(ele) = 2.0 * input->CFL * get_cfl_limit_adv(order) * eles->vol(ele) / int_waveSp;
  }

  if (input->dt_type == 1) /* Global minimum */
  {
    dt(0) = *std::min_element(dt.data(), dt.data()+eles->nEles);

#ifdef _MPI
    MPI_Allreduce(MPI_IN_PLACE, &dt(0), 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); 
#endif

  }
#endif

#ifdef _GPU
  compute_element_dt_wrapper(dt_d, faces->waveSp_d, faces->dA_d, geo.fpt2gfpt_d, 
      eles->weights_spts_d, eles->vol_d, eles->nSpts1D, input->CFL, order, 
      input->dt_type, eles->nFpts, eles->nEles, eles->nDims);
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

    /* Relax if GMRES did not converge */
    if (input->dt_scheme == "BDF1" && !GMRES_conv)
    {
      omg = 0.5;
    }

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

void FRSolver::write_solution(const std::string &prefix)
{
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
#endif

  if (input->rank == 0) std::cout << "Writing data to file..." << std::endl;

  std::stringstream ss;
#ifdef _MPI

  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_" << std::setw(9) << std::setfill('0');
    ss << current_iter << ".pvtu";
   
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

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
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
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
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
  f << "<!-- ITER " << current_iter << " -->" << std::endl;

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << eles->nPpts * eles->nEles << "\" ";
  f << "NumberOfCells=\"" << eles->nSubelements * eles->nEles << "\">";
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
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << geo.coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }

  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << geo.ppt_connect(i, subele) + ele*eles->nPpts << " ";
      }
      f << std::endl;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
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
  unsigned int nCells = eles->nSubelements * eles->nEles;
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

  /* Write solution information */
  f << "<PointData>" << std::endl;

  /* TEST: Write cell average solution */
  //eles->compute_Uavg();

  /* Extrapolate solution to plot points */
  auto &A = eles->oppE_ppts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_ppts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nPpts, &B, 
      eles->nSpts, 0.0, &C, eles->nPpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nPpts, &B, 
      eles->nSpts, 0.0, &C, eles->nPpts);
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

  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    f << "<DataArray type=\"Float32\" Name=\"u\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
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

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
      f << "format=\"ascii\">"<< std::endl;
      
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
        {
          f << std::scientific << std::setprecision(16);
          f << eles->U_ppts(ppt, ele, n) << " ";
        }

        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
  }
  if (input->filt_on && input->sen_write)
  {
#ifdef _GPU
    filt.sensor = filt.sensor_d;
#endif
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << filt.sensor(ele) << " ";
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

void FRSolver::write_color()
{
  std::cout << "Writing colors to file..." << std::endl;
  std::stringstream ss;
  ss.str("");
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_color";
  ss << ".vtu";

  auto outputfile = ss.str();

  /* Write parition solution to file in .vtu format */
  std::ofstream f(outputfile);

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;
  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << eles->nPpts * eles->nEles << "\" ";
  f << "NumberOfCells=\"" << eles->nSubelements * eles->nEles << "\">";
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
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << geo.coord_ppts(ppt, ele, 2) << std::endl;
      }
    }
  }
  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << geo.ppt_connect(i, subele) + ele*eles->nPpts << " ";
      }
      f << std::endl;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = eles->nNodesPerSubelement;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
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
  unsigned int nCells = eles->nSubelements * eles->nEles;
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
  f << "<DataArray type=\"Float32\" Name=\"color\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
    {
      f << std::scientific << std::setprecision(16) << ele_color(ele);
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

void FRSolver::report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1)
{

  /* If running on GPU, copy out divergence */
#ifdef _GPU
  eles->divF_spts = eles->divF_spts_d;
  dt = dt_d;
#endif

  // HACK: Change nStages to compute the correct residual
  if (input->dt_scheme == "BDF1" || input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
    nStages = 1;
  }

  std::vector<double> res(eles->nVars,0.0);

#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
    for (unsigned int ele =0; ele < eles->nEles; ele++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        eles->divF_spts(spt, ele, n, 0) /= eles->jaco_det_spts(spt, ele);

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    /* Infinity norm */
    if (input->res_type == 0)
      res[n] =*std::max_element(&eles->divF_spts(0, 0, n, 0), 
          &eles->divF_spts(0, 0, n+1, 1));

    /* L1 norm */
    else if (input->res_type == 1)
      res[n] = std::accumulate(&eles->divF_spts(0, 0, n, 0), 
          &eles->divF_spts(0, 0, n+1, 1), 0.0, abs_sum<double>());

    /* L2 norm */
    else if (input->res_type == 2)
      res[n] = std::accumulate(&eles->divF_spts(0, 0, n, 0), 
            &eles->divF_spts(0, 0, n+1, 1), 0.0, square<double>());
  }

  unsigned int nDoF =  (eles->nSpts * eles->nEles);

  // HACK: Change nStages back
  if (input->dt_scheme == "BDF1" || input->dt_scheme == "LUJac" || input->dt_scheme == "LUSGS")
  {
    nStages = 2;
  }

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), eles->nVars, MPI_DOUBLE, oper, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), eles->nVars, MPI_DOUBLE, oper, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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

    std::cout << current_iter << " ";
    for (auto val : res)
      std::cout << std::scientific << val / nDoF << " ";

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
    f << current_iter << " " << current_runtime.count() << " ";

    for (auto val : res)
      f << std::scientific << val / nDoF << " ";
    f << std::endl;
  }
}

void FRSolver::report_forces(std::ofstream &f)
{
  /* If using GPU, copy out solution, gradient and pressure */
#ifdef _GPU
  faces->U = faces->U_d;
  faces->dU = faces->dU_d;
  faces->P = faces->P_d;
#endif

  std::array<double, 3> force_conv, force_visc, taun;
  force_conv.fill(0.0); force_visc.fill(0.0); taun.fill(0.0);

  std::stringstream ss;
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".cp";
#else
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_" << std::setw(9) << std::setfill('0') << current_iter;
  ss << ".cp";
#endif

  auto cpfile = ss.str();
  std::ofstream g(cpfile);

  /* Get angle of attack (and sideslip) */
  double aoa = std::atan2(input->V_fs(1), input->V_fs(0)); 
  double aos = 0.0;
  if (eles->nDims == 3)
    aos = std::atan2(input->V_fs(2), input->V_fs(0));

  /* Compute factor for non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);

    if (bnd_id >= 8) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(fpt, 0);

      double CP = (PL - input->P_fs) * fac;

      /* Write CP distrubtion to file */
      for(unsigned int dim = 0; dim < eles->nDims; dim++)
        g << std::scientific << faces->coord(fpt, dim) << " ";
      g << std::scientific << CP << std::endl;

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < eles->nDims; dim++)
      {
        force_conv[dim] += eles->weights_spts(count%eles->nSpts1D) * CP * 
          faces->norm(fpt, dim, 0) * faces->dA(fpt);
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
          taun[0] = tauxx * faces->norm(fpt, 0, 0) + tauxy * faces->norm(fpt, 1, 0);
          taun[1] = tauxy * faces->norm(fpt, 0, 0) + tauyy * faces->norm(fpt, 1, 0);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_spts(count%eles->nSpts1D) * taun[dim] * 
              faces->dA(fpt) * fac;

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
          taun[0] = tauxx * faces->norm(fpt, 0, 0) + tauxy * faces->norm(fpt, 1, 0) + tauxz * faces->norm(fpt, 2, 0);
          taun[1] = tauxy * faces->norm(fpt, 0, 0) + tauyy * faces->norm(fpt, 1, 0) + tauyz * faces->norm(fpt, 2, 0);
          taun[3] = tauxz * faces->norm(fpt, 0, 0) + tauyz * faces->norm(fpt, 1, 0) + tauzz * faces->norm(fpt, 2, 0);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_spts(count%eles->nSpts1D) * taun[dim] * 
              faces->dA(fpt) * fac;

        }
        
      }
      count++;
    }
  }

  /* Compute lift and drag coefficients */
  double CL_conv, CD_conv, CL_visc, CD_visc;

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(force_conv.data(), force_conv.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(force_visc.data(), force_visc.data(), eles->nDims, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

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
    f << current_iter << " ";
    f << std::scientific << std::setprecision(16) << CL_conv << " " << CD_conv;

    if (input->viscous)
    {
      std::cout << " CL_visc = " << CL_visc << " CD_visc = " << CD_visc;
      f << std::scientific << std::setprecision(16) << " " << CL_visc << " " << CD_visc;
    }

    std::cout << std::endl;
    f << std::endl;
  }
}

void FRSolver::report_error(std::ofstream &f)
{
  /* If using GPU, copy out solution */
#ifdef _GPU
  eles->U_spts = eles->U_spts_d;
  eles->dU_spts = eles->dU_spts_d;
#endif

  /* Extrapolate solution to quadrature points */
  auto &A = eles->oppE_qpts(0, 0);
  auto &B = eles->U_spts(0, 0, 0);
  auto &C = eles->U_qpts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
      eles->nSpts, 0.0, &C, eles->nQpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
      eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
      eles->nSpts, 0.0, &C, eles->nQpts);
#endif

  /* Extrapolate derivatives to quadrature points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
      auto &A = eles->oppE_qpts(0, 0);
      auto &B = eles->dU_spts(0, 0, 0, dim);
      auto &C = eles->dU_qpts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
          eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
          eles->nSpts, 0.0, &C, eles->nQpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, 
          eles->nEles * eles->nVars, eles->nSpts, 1.0, &A, eles->nQpts, &B, 
          eles->nSpts, 0.0, &C, eles->nQpts);
#endif

  }

  std::vector<double> l2_error(2,0.0);

  unsigned int n = input->err_field;
  std::vector<double> dU_true(2, 0.0), dU_error(2, 0.0);
#pragma omp for collapse (2)
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
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
            U_true = compute_U_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
                flow_time, n, input);
          }

          dU_true[0] = compute_dU_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
              flow_time, n, 0, input);
          dU_true[1] = compute_dU_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 0, 
              flow_time, n, 1, input);
          

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

          U_error = U_true - P/std::pow(eles->U_qpts(qpt, ele, 0), input->gamma);
        }
        else
        {
          U_error = U_true - eles->U_qpts(qpt, ele, n);
          dU_error[0] = dU_true[0] - eles->dU_qpts(qpt, ele, n, 0); 
          dU_error[1] = dU_true[1] - eles->dU_qpts(qpt, ele, n, 1);
        }

        l2_error[0] += weight * eles->jaco_det_qpts(qpt, ele) * U_error * U_error; 
        l2_error[1] += weight * eles->jaco_det_qpts(qpt, ele) * (U_error * U_error +
            dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1]); 
      }
  }

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Reduce(l2_error.data(), l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

#endif


  /* Print to terminal */
  if (input->rank == 0)
  {
    std::cout << "l2_error: ";
    for (auto &val : l2_error)
      std::cout << std::scientific << std::sqrt(val) << " ";
    std::cout << std::endl;

    /* Write to file */
    f << current_iter << " ";
    for (auto &val : l2_error)
      f << std::scientific << std::setprecision(16) << std::sqrt(val) << " ";
    f << std::endl;
  }

}


void FRSolver::filter_solution()
{
  if (!input->filt_on) return; 
  
  /* Sense discontinuities and filter solution */
  unsigned int status = 1;
  for (unsigned int level = 0; level < input->filt_maxLevels && status; level++)
  {
    filt.apply_sensor();
    status = filt.apply_filter(level);
  }
}
