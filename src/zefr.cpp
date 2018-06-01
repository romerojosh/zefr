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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _MPI
#include "mpi.h"
#ifdef _GPU
#include "cuda_runtime.h"
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif
#endif

#include "input.hpp"
#include "macros.hpp"
#include "multigrid.hpp"
#include "solver.hpp"
#include "filter.hpp"

#include "funcs.hpp"
#include "mdvector.hpp"

#ifndef _BUILD_LIB
int main(int argc, char* argv[])
{
  int rank = 0; int nRanks = 1;
  _mpi_comm comm;
#ifdef _MPI
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

#ifdef _GPU
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (nDevices < nRanks)
  {
    //ThrowException("Not enough GPUs for this run. Allocate more!");
  }

  cudaSetDevice(rank%nDevices); /// TODO: use MPI_local_rank % nDevices
  //cudaSetDevice(rank%4); // Hardcoded for ICME K80 nodes for now.
#endif
#else
  comm = 0;
#endif /* _MPI */

  if (argc != 2)
  {
    if (rank == 0)
    {
      std::cout << "Usage:" << std::endl;
      std::cout << "\t" << argv[0] << " <input file>" << std::endl;
    }
    exit(1);
  }

  /* Print out cool ascii art header */
  if (rank == 0)
  {
    std::cout << std::endl;
    std::cout << R"(          ______     ______     ______   ______             )" << std::endl;
    std::cout << R"(         /\___  \   /\  ___\   /\  ___\ /\  == \            )" << std::endl;
    std::cout << R"(         \/_/  /__  \ \  __\   \ \  __\ \ \  __<            )" << std::endl;
    std::cout << R"(           /\_____\  \ \_____\  \ \_\    \ \_\ \_\          )" << std::endl;
    std::cout << R"(           \/_____/   \/_____/   \/_/     \/_/ /_/          )" << std::endl;
    std::cout << R"(____________________________________________________________)" << std::endl;
    std::cout << R"( "...bear in mind that princes govern all things --         )" << std::endl;
    std::cout << R"(                              save the wind." -Victor Hugo  )" << std::endl;
    std::cout << std::endl;
  }

  std::string inputfile = argv[1];

  if (rank == 0) std::cout << "Reading input file: " << inputfile <<  std::endl;
  auto input = read_input_file(inputfile);
  input.rank = rank;
  input.nRanks = nRanks;

#ifdef _GPU
  initialize_cuda();
#endif

  if (rank == 0) std::cout << "Setting up FRSolver..." << std::endl;
  FRSolver solver(&input);
  solver.setup(comm);

  if (input.restart)
    solver.restart_solution();
  
  PMGrid pmg;
  if (input.p_multi)
  {
    if (rank == 0) std::cout << "Setting up multigrid..." << std::endl;
    pmg.setup(&input, solver, comm);
  }

  /* Open file to write history output */
  std::ofstream hist_file;
  std::ofstream force_file;
  std::ofstream error_file;
  std::ofstream turb_stats_file;

  if (input.restart) /* If restarted, append to existing file */
  {
    hist_file.open(input.output_prefix + "/" + input.output_prefix + "_hist.dat", std::ios::app);
    force_file.open(input.output_prefix + "/" + input.output_prefix + "_forces.dat", std::ios::app);
    error_file.open(input.output_prefix + "/" + input.output_prefix + "_error.dat", std::ios::app);
    turb_stats_file.open(input.output_prefix + "/" + input.output_prefix + "_turb_stats.dat", std::ios::app);
  }
  else
  {
    hist_file.open(input.output_prefix + "/" + input.output_prefix + "_hist.dat");
    force_file.open(input.output_prefix + "/" + input.output_prefix + "_forces.dat");
    error_file.open(input.output_prefix + "/" + input.output_prefix + "_error.dat");
    turb_stats_file.open(input.output_prefix + "/" + input.output_prefix + "_turb_stats.dat");
  }

  /* Write initial solution */
  if (input.write_freq != 0 && !input.restart)
  {
    if (input.write_paraview)
      solver.write_solution(input.output_prefix);
    if (input.plot_surfaces)
      solver.write_surfaces(input.output_prefix);
    if (input.write_pyfr)
      solver.write_solution_pyfr(input.output_prefix);
  }

  if (input.iterative_method == MCGS)
    solver.write_color();

  /* Write initial error (if required) */
  if (input.error_freq != 0)
    solver.report_error(error_file);

  /* Set convergence file and timer parameters in solver */
  auto t1 = std::chrono::high_resolution_clock::now();
  if (input.implicit_method) solver.set_conv_file(t1);

  input.time = solver.get_current_time();
  /* Main iteration loop */
  for (unsigned int n = input.initIter+1; (n <= input.n_steps) && (input.time < input.tfinal) && (solver.res_max > input.res_tol); n++)
  {
    if (!input.p_multi)
    {
      solver.update();
      solver.filter_solution();
    }
    else
    {
      pmg.cycle(solver, hist_file, t1);
    }
    
    input.iter++;
    input.time = solver.get_current_time();

    if (input.tavg)
    {
      bool do_accum = (n%input.tavg_freq == 0);
      bool do_write = (n%input.write_tavg_freq == 0 || n == input.n_steps);

      if (do_accum || do_write)
        solver.accumulate_time_averages();

      if (do_write)
        solver.write_averages(input.output_prefix);
    }

    /* Write output if required */
    if (input.report_freq != 0 && (n%input.report_freq == 0 || n == input.n_steps || n == 1))
    {
      solver.report_residuals(hist_file, t1);
    }

    if (input.write_freq != 0 && (n%input.write_freq == 0 || n == input.n_steps || input.time >= input.tfinal || solver.res_max <= input.res_tol))
    {
      if (input.write_paraview)
        solver.write_solution(input.output_prefix);
      if (input.plot_surfaces)
        solver.write_surfaces(input.output_prefix);
      if (input.write_pyfr)
        solver.write_solution_pyfr(input.output_prefix);
    }

    if (input.force_freq != 0 && (n%input.force_freq == 0 || n == input.n_steps || solver.res_max <= input.res_tol))
    {
      solver.report_forces(force_file);
    }

    if (input.error_freq != 0 && (n%input.error_freq == 0 || n == input.n_steps || solver.res_max <= input.res_tol))
    {
      solver.report_error(error_file);
    }

    if (input.turb_stat_freq != 0 && (n%input.turb_stat_freq == 0 || n == input.n_steps || solver.res_max <= input.res_tol))
    {
      solver.report_turbulent_stats(turb_stats_file);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
 
  auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  if (rank == 0) std::cout << "Elapsed time: " << elapsed_time.count() << " s" << std::endl;

#ifdef _MPI
  MPI_Finalize();
#endif

  return 0;
}
#endif

/* ==== Add in interface functions for use from external code ==== */
//#define _BUILD_LIB  // for QT editing
#ifdef _BUILD_LIB
#include "zefr.hpp"

#ifdef _MPI
Zefr::Zefr(MPI_Comm comm_in, int n_grids, int grid_id, MPI_Comm comm_world)
#else
Zefr::Zefr(void)
#endif
{
  // Basic constructor
#ifndef _MPI
  myComm = 0;
  myGrid = 0;
#else
  mpi_init(comm_in, comm_world, n_grids, grid_id);
#endif

  /* Print out cool ascii art header */
  if (myGrid == 0 && rank == 0)
  {
    std::cout << std::endl;
    std::cout << R"(          ______     ______     ______   ______             )" << std::endl;
    std::cout << R"(         /\___  \   /\  ___\   /\  ___\ /\  == \            )" << std::endl;
    std::cout << R"(         \/_/  /__  \ \  __\   \ \  __\ \ \  __<            )" << std::endl;
    std::cout << R"(           /\_____\  \ \_____\  \ \_\    \ \_\ \_\          )" << std::endl;
    std::cout << R"(           \/_____/   \/_____/   \/_/     \/_/ /_/          )" << std::endl;
    std::cout << R"(____________________________________________________________)" << std::endl;
    std::cout << R"( "...bear in mind that princes govern all things --         )" << std::endl;
    std::cout << R"(                              save the wind." -Victor Hugo  )" << std::endl;
    std::cout << std::endl;
  }

#ifdef _GPU
  initialize_cuda();
#endif
}

#ifdef _MPI
void Zefr::mpi_init(MPI_Comm comm_in, MPI_Comm comm_world, int n_grids, int grid_id)
{
  myComm = comm_in;
  worldComm = comm_world;
  nGrids = n_grids;
  myGrid = grid_id;

  MPI_Comm_rank(myComm, &rank);
  MPI_Comm_size(myComm, &nRanks);

  grank = rank;
  MPI_Comm_rank(worldComm,&grank);

#ifdef _GPU
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  /// TODO
  if (nDevices < nRanks)
  {
    //ThrowException("Not enough GPUs for this run. Allocate more!");
  } 

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(hostname, &len);

  int cid;
  //cid = grank%16; // For XStream nodes
  //cid = grank%4; // For ICME K80 nodes
  cid = grank % nDevices; /// TODO: use MPI_local_rank % nDevices
  //printf("rank %d on grid %d, global rank %d --> CUDA device %d\n",rank,myGrid,grank,cid);
  cudaSetDevice(cid); 
  check_error();

//  cudaDeviceProp prop;
//  cudaGetDeviceProperties(&prop, cid);
//  printf("%d: Device name: %s\n",grank, prop.name);
#endif
}
#endif

void Zefr::read_input(const char *inputfile)
{
  if (rank == 0) std::cout << "Reading input file: " << inputfile <<  std::endl;
  input = read_input_file(inputfile);

  input.rank = rank;
  input.nRanks = nRanks;
  input.grank = grank;

  if (nGrids > 1)
    input.overset = 1;
  //else
  //  input.overset = 0;

  if (input.overset)
  {
    if (input.oversetGrids.size() != nGrids)
      ThrowException("Number of overset grids in input file does not match nGrids");

    input.meshfile = input.oversetGrids[myGrid];
    input.gridID = myGrid;
    input.gridType = input.gridTypes[myGrid];
    gridType = input.gridType;
  }
}

void Zefr::init_inputs(void)
{
  initialize_inputs(input);
}

void Zefr::setup_solver(void)
{
  if (rank == 0) std::cout << "Setting up FRSolver..." << std::endl;
  solver = std::make_shared<FRSolver>(&input);
  solver->ZEFR = this;
  solver->setup(myComm, worldComm);

  if (input.p_multi)
  {
    pmg = std::make_shared<PMGrid>();

    if (rank == 0) std::cout << "Setting up multigrid..." << std::endl;
    pmg->setup(&input, *solver, myComm);
  }

  /* Open files to write residual / force / error history output */
  if (input.restart) /* If restarted, append to existing file */
  {
    hist_file.open(input.output_prefix + "/" + input.output_prefix + "_hist.dat", std::ios::app);
    force_file.open(input.output_prefix + "/" + input.output_prefix + "_forces.dat", std::ios::app);
    error_file.open(input.output_prefix + "/" + input.output_prefix + "_error.dat", std::ios::app);
  }
  else
  {
    hist_file.open(input.output_prefix + "/" + input.output_prefix + "_hist.dat");
    force_file.open(input.output_prefix + "/" + input.output_prefix + "_forces.dat");
    error_file.open(input.output_prefix + "/" + input.output_prefix + "_error.dat");
  }

  geo = &solver->geo;

  t_start = std::chrono::high_resolution_clock::now();

  solver->grid_time = -1;
  solver->init_grid_motion(solver->flow_time);
  
  // For easy access in Python
  simData.nfields = solver->eles->nVars;
  for (int i = 0; i < geo->ele_types.size(); i++)
  {
    simData.nspts[i] = solver->elesObjs[i]->nSpts;
    simData.u_spts[i] = solver->elesObjs[i]->U_spts.data();
    if (input.viscous)
      simData.du_spts[i] = solver->elesObjs[i]->dU_spts.data();
#ifdef _GPU
    simData.u_spts_d[i] = solver->elesObjs[i]->U_spts_d.data();
    if (input.viscous)
      simData.du_spts_d[i] = solver->elesObjs[i]->dU_spts_d.data();
#endif
  }
}

void Zefr::restart_solution(void)
{
  if (input.restart)
    solver->restart_solution();
}

void Zefr::do_step(void)
{
  if (!input.p_multi)
  {
    solver->update();
    solver->filter_solution();
  }
  else
  {
    pmg->cycle(*solver, hist_file, t_start);
  }

  input.iter++;
}

void Zefr::do_rk_stage(int iter, int stage)
{
  input.iter = iter;

  solver->step_RK_stage(stage);

  if (stage == input.nStages-1)
  {
    solver->filter_solution();
    input.iter++;
  }
}

void Zefr::do_rk_stage_start(int iter, int stage)
{
  if (input.dt_scheme == "RK54")
    solver->step_LSRK_stage_start(stage);
  else
    solver->step_RK_stage_start(stage);
}

void Zefr::do_rk_stage_mid(int iter, int stage)
{
  if (input.dt_scheme == "RK54")
    stage = 0; // Low-storage RK method has just 1 register for residual

  solver->step_RK_stage_mid(stage);
}

void Zefr::do_rk_stage_finish(int iter, int stage)
{
  if (input.dt_scheme == "RK54")
    solver->step_LSRK_stage_finish(stage);
  else
    solver->step_RK_stage_finish(stage);

  if (stage == input.nStages-1)
  {
    solver->filter_solution();
  }

  input.iter = iter;
  solver->current_iter = iter;
}

void Zefr::do_n_steps(int n)
{
  for (int i = 0; i < n; i++)
    do_step();
}

void Zefr::write_residual(void)
{
  solver->report_residuals(hist_file, t_start);
}

void Zefr::write_solution(void)
{
  if (input.write_paraview)
    solver->write_solution(input.output_prefix);
  if (input.plot_surfaces)
    solver->write_surfaces(input.output_prefix);
  if (input.write_pyfr)
    solver->write_solution_pyfr(input.output_prefix);
}

void Zefr::update_averages(void)
{
  if (input.tavg)
    solver->accumulate_time_averages();
}

void Zefr::write_averages(void)
{
  if (input.tavg)
    solver->write_averages(input.output_prefix);
}

void Zefr::write_forces(void)
{
  solver->report_forces(force_file);
}

void Zefr::get_forces(void)
{
  std::array<double, 3> force = {0,0,0};
  std::array<double, 3> moment = {0,0,0};
  solver->compute_moments(force, moment);

  for (int i = 0; i < geo->nDims; i++)
  {
    simData.forces[i] = force[i];
    simData.forces[i+3] = moment[i];
  }
}

void Zefr::write_error(void)
{
  solver->report_error(error_file);
}

void Zefr::write_wall_time(void)
{
  auto t2 = std::chrono::high_resolution_clock::now();

  auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t_start);
  if (input.gridID == 0 && input.rank == 0)
    std::cout << "Elapsed time: " << elapsed_time.count() << " s" << std::endl;
}

void Zefr::get_basic_geo_data(int& btag, int& gType, int& nnodes, double*& xyz,
    int*& iblank, int& nwall, int& nover, int*& wallNodes, int*& overNodes,
    int& nCellTypes, int*& nvert_cell, int*& nface_cell, int*& nCells_type, int**& c2v)
{
  btag = myGrid;
  gType = gridType;
  nnodes = geo->nNodes;
  xyz = geo->coord_nodes.data();
  iblank = geo->iblank_node.data();
  nwall = geo->nWall;
  nover = geo->nOver;
  wallNodes = geo->wallNodes.data();
  overNodes = geo->overNodes.data();
  nCellTypes = geo->ele_types.size();

  geo->nv_ptr.resize(nCellTypes);
  geo->nc_ptr.resize(nCellTypes);
  geo->ncf_ptr.resize(nCellTypes);
  geo->c2v_ptr.resize(nCellTypes);

  for (int i = 0; i < nCellTypes; i++)
  {
    ELE_TYPE etype = (ELE_TYPE)geo->ele_types(i);
    geo->nv_ptr[i] = geo->nNodesPerEleBT[etype];
    geo->nc_ptr[i] = geo->nElesBT[etype];
    geo->ncf_ptr[i] = geo->nFacesPerEleBT[etype];
    geo->c2v_ptr[i] = geo->ele2nodesBT[etype].data();
  }

  nvert_cell = geo->nv_ptr.data();
  nface_cell = geo->ncf_ptr.data();
  nCells_type = geo->nc_ptr.data();
  c2v = geo->c2v_ptr.data();
}

void Zefr::get_extra_geo_data(int& nFaceTypes, int*& faceTypes, int*& cellTypes,
                              int*& nvert_face, int*& nFaces_type,
                              int**& f2v, int*& f2c, int**& c2f, int*& iblank_face,
                              int*& iblank_cell, int &nOver, int*& overFaces,
                              int &nWall, int*& wallFaces, int &nMpiFaces,
                              int*& mpiFaces, int*& procR, int*& faceIdR,
                              double*& grid_vel, double*& offset, double*& Rmat)
{
  nFaceTypes = geo->face_types.size();
  faceTypes = geo->face_types.data();
  cellTypes = geo->ele_types.data();

  nvert_face = geo->nNode_face.data();

  geo->nf_ptr.resize(nFaceTypes);
  geo->f2v_ptr.resize(nFaceTypes);
  geo->c2f_ptr.resize(geo->ele_types.size());

  for (int i = 0; i < nFaceTypes; i++)
  {
    ELE_TYPE ftype = (ELE_TYPE)geo->face_types(i);
    geo->nf_ptr[i] = geo->nFacesBT[ftype];
    geo->f2v_ptr[i] = geo->face2nodes[(ELE_TYPE)ftype].data();
  }

  for (int i = 0; i < geo->ele_types.size(); i++)
  {
    ELE_TYPE etype = (ELE_TYPE)geo->ele_types(i);
    geo->c2f_ptr[i] = geo->ele2face[etype].data();
  }

  nFaces_type = geo->nf_ptr.data();
  f2v = geo->f2v_ptr.data();
  c2f = geo->c2f_ptr.data();

  f2c = geo->face2eles.data();
  iblank_face = geo->iblank_face.data();
  iblank_cell = geo->iblank_cell.data();
  nOver = geo->overFaceList.size();
  overFaces = geo->overFaceList.data();
  nWall = geo->wallFaceList.size();
  wallFaces = geo->wallFaceList.data();
  nMpiFaces = geo->nMpiFaces;
  mpiFaces = geo->mpiFaces.data();
  procR = geo->procR.data();
  faceIdR = geo->faceID_R.data();

  if (input.motion)
  {
    grid_vel = geo->grid_vel_nodes.data();
    offset = geo->x_cg.data();
    Rmat = geo->Rmat.data();
  }
  else
  {
    grid_vel = NULL;
    offset = NULL;
    Rmat = NULL;
  }
}

void Zefr::get_gpu_geo_data(double*& coord_nodes, double*& coord_eles,
    int*& iblank_cell, int*& iblank_face)
{
#ifdef _GPU
  auto etype = solver->elesObjs[0]->etype;

  coord_nodes = geo->coord_nodes_d.data();
  coord_eles = solver->elesObjs[0]->nodes_d.data();
  iblank_cell = geo->iblank_cell_d.data();
  iblank_face = geo->iblank_face_d.data();
#endif
}

double Zefr::get_u_spt(int ele, int spt, int var)
{
  int eleBT = geo->eleID_type(ele);
  return solver->elesObjs[solver->ele2elesObj(ele)]->U_spts(spt, var, eleBT);
}

double Zefr::get_grad_spt(int ele, int spt, int dim, int var)
{
  int eleBT = geo->eleID_type(ele);
  return solver->elesObjs[solver->ele2elesObj(ele)]->dU_spts(dim, spt, var, eleBT);
}

double *Zefr::get_u_spts(int &ele_stride, int &spt_stride, int &var_stride, int etype)
{
  auto e = solver->elesObjs[etype];

  ele_stride = 1;
  var_stride = e->nElesPad;
  spt_stride = e->nElesPad * e->nVars;

  return e->U_spts.data();
}

double *Zefr::get_du_spts(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype)
{
  auto e = solver->elesObjs[etype];

  ele_stride = 1;
  var_stride = e->nEles;
  spt_stride = e->nEles * e->nVars;
  dim_stride = e->nEles * e->nVars * e->nSpts;

  return e->dU_spts.data();
}


double *Zefr::get_u_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int etype)
{
#ifdef _GPU
  auto e = solver->elesObjs[etype];

  ele_stride = 1;
  var_stride = e->nElesPad;
  spt_stride = e->nElesPad * e->nVars;

  return e->U_spts_d.data();
#endif
#ifdef _CPU
  ThrowException("Should not be calling get_u_spts_d - ZEFR not compiled for GPUs!");
#endif
}

double *Zefr::get_du_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype)
{
#ifdef _GPU
  auto e = solver->elesObjs[etype];

  ele_stride = 1;
  var_stride = e->nEles;
  spt_stride = e->nEles * e->nVars;
  dim_stride = e->nEles * e->nVars * e->nSpts;

  return e->dU_spts_d.data();
#endif
#ifdef _CPU
  ThrowException("Should not be calling get_du_spts_d - ZEFR not compiled for GPUs!");
#endif
}

void Zefr::get_nodes_per_cell(int cellID, int &nNodes)
{
  nNodes = solver->elesObjs[solver->ele2elesObj(cellID)]->nSpts;
}

void Zefr::get_nodes_per_face(int faceID, int& nNodes)
{
  nNodes = (int)geo->nFptsPerFace[geo->faceType(faceID)];
}

void Zefr::get_receptor_nodes(int cellID, int& nNodes, double* xyz)
{
  auto e = solver->elesObjs[solver->ele2elesObj(cellID)];
  int ele = geo->eleID_type(cellID);

  nNodes = (int)e->nSpts;

  for (int spt = 0; spt < nNodes; spt++)
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*spt+dim] = e->coord_spts(spt, dim, ele);
}

void Zefr::get_face_nodes(int faceID, int &nNodes, double* xyz)
{
  auto ftype = geo->faceType(faceID);
  int fid = geo->faceID_type(faceID);

  nNodes = (int)geo->nFptsPerFace[ftype];

  for (int fpt = 0; fpt < nNodes; fpt++)
  {
    int gfpt = geo->face2fpts[ftype](fpt, fid);
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*fpt+dim] = solver->faces->coord(dim, gfpt);
  }
}

void Zefr::get_face_nodes_gpu(int* faceIDs, int nFaces, int* nPtsFace, double *xyz)
{
#ifdef _GPU
  solver->faces->get_face_coords(faceIDs,nFaces,nPtsFace,xyz);
#endif
}

void Zefr::get_cell_nodes_gpu(int* cellIDs, int nCells, int* nPtsCell, double *xyz)
{
#ifdef _GPU
  if (nCells == 0) return;

  std::map<ELE_TYPE, unsigned int> ncell_type;
  std::map<ELE_TYPE, std::vector<int>> cells_type;
  std::map<ELE_TYPE, std::vector<double>> coords_type;

  for (auto etype : geo->ele_set)
    ncell_type[etype] = 0;

  for (int i = 0; i < nCells; i++)
  {
    auto eles = solver->elesObjs[solver->ele2elesObj(cellIDs[i])];
    auto etype = eles->etype;
    nPtsCell[i] = eles->nSpts;
    ncell_type[etype]++;
  }

  for (int i = 0; i < geo->ele_types.size(); i++)
  {
    ELE_TYPE etype = (ELE_TYPE)geo->ele_types(i);

    cells_type[etype].resize(ncell_type[etype]);
    coords_type[etype].resize(ncell_type[etype]*solver->elesObjs[i]->nSpts*3);
    ncell_type[etype] = 0;
  }

  for (int i = 0; i < nCells; i++)
  {
    auto etype = geo->eleType(i);
    cells_type[etype][ncell_type[etype]++] = cellIDs[i];
  }

  for (auto eles : solver->elesObjs)
  {
    auto etype = eles->etype;

    eles->get_cell_coords(cells_type[etype].data(),ncell_type[etype],nPtsCell,coords_type[etype].data());
  }

  for (auto etype : geo->ele_set)
    ncell_type[etype] = 0;

  int ind = 0;
  for (int i = 0; i < nCells; i++)
  {
    auto etype = geo->eleType(cellIDs[i]);

    for (int k = 0; k < nPtsCell[i]; k++)
      for (int d = 0; d < 3; d++)
        xyz[ind + 3*k + d] = coords_type[etype][3*(nPtsCell[i]*ncell_type[etype] + k) + d];

    ncell_type[etype]++;
    ind += nPtsCell[i];
  }
#endif
}

void Zefr::donor_inclusion_test(int cellID, double* xyz, int& passFlag, double* rst)
{
  int ele = geo->eleID_type(cellID);
  passFlag = solver->elesObjs[solver->ele2elesObj(cellID)]->getRefLoc(ele,xyz,rst);
}

void Zefr::donor_frac(int cellID, int &nweights, int* inode, double* weights,
                      double* rst, int buffsize)
{
  /* NOTE: inode is not used */
  solver->elesObjs[solver->ele2elesObj(cellID)]->get_interp_weights(rst,weights,nweights,buffsize);
}

int Zefr::get_n_weights(int cellID)
{
  return (int)solver->elesObjs[solver->ele2elesObj(cellID)]->nSpts;
}

void Zefr::donor_frac_gpu(int* cellIDs, int nFringe, double* rst, double* weights)
{
#ifdef _GPU
  if (nFringe == 0) return;

  if (geo->ele_types.size() == 1)
  {
    PUSH_NVTX_RANGE("donorFracGPU-1",1);
    solver->elesObjs[0]->get_interp_weights_gpu(cellIDs,nFringe,rst,weights);
    POP_NVTX_RANGE;

    return;
  }

  PUSH_NVTX_RANGE("donorFracGPU-1",1);
  std::map<ELE_TYPE, unsigned int> ncell_type;
  std::map<ELE_TYPE, mdvector<int>> cells_type;
  std::map<ELE_TYPE, mdvector<double>> weights_type, rst_type;
  std::map<ELE_TYPE, mdvector_gpu<double>> weights_type_d, rst_type_d;

  for (auto etype : geo->ele_set)
    ncell_type[etype] = 0;

  for (int i = 0; i < nFringe; i++)
  {
    auto etype = geo->eleType(cellIDs[i]);
    ncell_type[etype]++;
  }

  for (auto eles : solver->elesObjs)
  {
    auto etype = eles->etype;
    cells_type[etype].assign({ncell_type[etype]});
    rst_type[etype].assign({ncell_type[etype],3});
    weights_type[etype].assign({ncell_type[etype],eles->nSpts});
    weights_type_d[etype].set_size(weights_type[etype]);
    ncell_type[etype] = 0;
  }

  for (int i = 0; i < nFringe; i++)
  {
    auto etype = geo->eleType(cellIDs[i]);

    int ic = ncell_type[etype];
    cells_type[etype](ic) = cellIDs[i];

    for (int d = 0; d < 3; d++)
      rst_type[etype](ic,d) = rst[3*i+d];

    ncell_type[etype]++;
  }

  for (auto eles : solver->elesObjs)
  {
    auto etype = eles->etype;

    rst_type_d[etype] = rst_type[etype];

    mdvector_gpu<int> donors_d;
    donors_d = cells_type[etype];

    eles->get_interp_weights_gpu(donors_d.data(),ncell_type[etype],
        rst_type_d[etype].data(),weights_type_d[etype].data());

    sync_stream(3);

    weights_type[etype] = weights_type_d[etype];
    donors_d.free_data();
    weights_type_d[etype].free_data();
    rst_type_d[etype].free_data();
  }
  POP_NVTX_RANGE;

  PUSH_NVTX_RANGE("donorFracGPU-2",2); /// DEBUGGING
  for (auto etype : geo->ele_set)
    ncell_type[etype] = 0;

  int ind = 0;
  for (int i = 0; i < nFringe; i++)
  {
    auto eles = solver->elesObjs[solver->ele2elesObj(cellIDs[i])];
    auto etype = eles->etype;
    int nSpts = eles->nSpts;

    for (int k = 0; k < nSpts; k++)
      weights[ind + k] = weights_type[etype](ncell_type[etype],k);

    ncell_type[etype]++;
    ind += nSpts;
  }
  POP_NVTX_RANGE;
#endif
}

double& Zefr::get_u_fpt(int faceID, int fpt, int var)
{
  return solver->faces->get_u_fpt(faceID,fpt,var);
}

double& Zefr::get_grad_fpt(int faceID, int fpt, int dim, int var)
{
  return solver->faces->get_grad_fpt(faceID,fpt,var,dim);
}

void Zefr::update_iblank_gpu(void)
{
#ifdef _GPU
  geo->iblank_fpts.assign({geo->nGfpts}, 1);
  for (unsigned int fpt = 0; fpt < geo->nGfpts; fpt++)
  {
    int face = geo->fpt2face[fpt];
    int icL = geo->face2eles(face,0);
    geo->iblank_fpts(fpt) = geo->iblank_face(face);

    // Set iblank value to tell us which side the hole cell is on
    if (geo->iblank_face(face) < 0 && geo->iblank_cell(icL) == HOLE)
    {
      geo->iblank_fpts(fpt) = -2;
      geo->iblank_face(face) = -2;
    }
  }

  geo->iblank_fpts_d = geo->iblank_fpts;
  geo->iblank_cell_d = geo->iblank_cell;
  geo->iblank_face_d = geo->iblank_face; /// TEMP / DEBUGGING - RETHINK LATER
#endif
}

void Zefr::donor_data_from_device(int *donorIDs, int nDonors, int gradFlag)
{
#ifdef _GPU
  /// TODO: eletype
  if (gradFlag == 0)
    solver->elesObjs[0]->donor_u_from_device(donorIDs, nDonors);
  else
    solver->elesObjs[0]->donor_grad_from_device(donorIDs, nDonors);

  check_error();
#endif
}

void Zefr::fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double *data)
{
#ifdef _GPU
  if (data)
  {
    // New version
    if (gradFlag == 0)
      solver->faces->fringe_u_to_device(fringeIDs, nFringe, data);
    else
      solver->faces->fringe_grad_to_device(fringeIDs, nFringe, data);
  }
  else
  {
    ThrowException("Don't use this version!!");
    // Original version
    if (gradFlag == 0)
      solver->faces->fringe_u_to_device(fringeIDs, nFringe);
    else
      solver->faces->fringe_grad_to_device(fringeIDs, nFringe);
  }

  check_error();
#endif
}

void Zefr::unblank_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double *data)
{
#ifdef _GPU
  /// TODO: eletype
  if (gradFlag == 0)
    solver->elesObjs[0]->unblank_u_to_device(fringeIDs, nFringe, data);
//  else
//    solver->elesObjs[0]->unblank_grad_to_device(nFringe, data);

  check_error();
#endif
}

void Zefr::set_tioga_callbacks(void (*point_connect)(void), void (*unblank_part_1)(void),
                               void (*unblank_part_2)(int), void (*dataUpdate_recv)(int, int),
                               void (*dataUpdate_send)(int, int))
{
  /*! NOTE: All of these callbacks are not required [in fact, discouraged]
   *  when using the new HELIOS-compatible Python layer */

  tg_point_connectivity = point_connect;
  unblank_1 = unblank_part_1;
  unblank_2 = unblank_part_2;
  overset_interp_send = dataUpdate_send;
  overset_interp_recv = dataUpdate_recv;
}

void Zefr::set_rigid_body_callbacks(void (*setTransform)(double* mat, double* off, int nDims))
{
  tg_update_transform = setTransform;
}

void Zefr::move_grid_next(double time)
{
  solver->move_grid_next(time);
}

void Zefr::move_grid(double time)
{
  solver->move_grid_now(time);
}

void Zefr::move_grid(int iter, int stage)
{
  double time = solver->prev_time + solver->rk_c(stage) * solver->eles->dt(0);
  solver->move_grid_now(time);
}

void* Zefr::get_tg_stream_handle(void)
{
#ifdef _GPU
  return (void*)get_stream_handle(3);
#else
  return NULL;
#endif
}

void* Zefr::get_tg_event_handle(void)
{
#ifdef _GPU
  return (void*)get_event_handle(0);
#else
  return NULL;
#endif
}

#endif /* _BUILD_LIB */
