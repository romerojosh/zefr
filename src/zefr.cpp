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

  cudaSetDevice(rank%nDevices); // Hardcoded for ICME nodes for now.
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

  if (rank == 0) std::cout << "Setting up FRSolver..." << std::endl;
  FRSolver solver(&input);
  solver.setup(comm);
  
  PMGrid pmg;
  if (input.p_multi)
  {
    if (rank == 0) std::cout << "Setting up multigrid..." << std::endl;
    pmg.setup(&input, solver, comm);
  }

#ifdef _GPU
  start_cublas();
#endif

  /* Open file to write history output */
  std::ofstream hist_file;
  std::ofstream force_file;
  std::ofstream error_file;

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

  /* Write initial solution */
  if (input.write_paraview)
    solver.write_solution(input.output_prefix);
  if (input.plot_surfaces)
    solver.write_surfaces(input.output_prefix);
  if (input.write_pyfr)
    solver.write_solution_pyfr(input.output_prefix);

  if (input.dt_scheme == "MCGS")
  {
    solver.write_color();
  }

  /* Write initial error (if required) */
  if (input.error_freq != 0)
    solver.report_error(error_file);

  auto t1 = std::chrono::high_resolution_clock::now();
  /* Main iteration loop */
  for (unsigned int n = 1; (n <= input.n_steps) && (solver.res_max > input.res_tol); n++)
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

    /* Write output if required */
    if (input.report_freq != 0 && (n%input.report_freq == 0 || n == input.n_steps || n == 1))
    {
      solver.report_residuals(hist_file, t1);
    }

    if (input.write_freq != 0 && (n%input.write_freq == 0 || n == input.n_steps || solver.res_max <= input.res_tol))
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
Zefr::Zefr(MPI_Comm comm_in, int n_grids, int grid_id)
#else
Zefr::Zefr(void)
#endif
{
  // Basic constructor
#ifndef _MPI
  myComm = 0;
  myGrid = 0;
#else
  mpi_init(comm_in, n_grids, grid_id);
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

  solver = NULL;
}

#ifdef _MPI
void Zefr::mpi_init(MPI_Comm comm_in, int n_grids, int grid_id)
{
  myComm = comm_in;
  nGrids = n_grids;
  myGrid = grid_id;

  MPI_Comm_rank(myComm, &rank);
  MPI_Comm_size(myComm, &nRanks);

#ifdef _GPU
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  /// TODO
  if (nDevices < nRanks)
  {
    //ThrowException("Not enough GPUs for this run. Allocate more!");
  }

  cudaSetDevice(0);
  //cudaSetDevice(rank%6); // Hardcoded for ICME nodes for now.
#endif
}
#endif

void Zefr::read_input(char *inputfile)
{
  if (rank == 0) std::cout << "Reading input file: " << inputfile <<  std::endl;
  input = read_input_file(inputfile);

  input.rank = rank;
  input.nRanks = nRanks;
  if (nGrids > 1) input.overset = true;

  if (input.overset)
  {
    input.meshfile = input.oversetGrids[myGrid];
    input.gridID = myGrid;
  }
}

void Zefr::setup_solver(void)
{
  if (rank == 0) std::cout << "Setting up FRSolver..." << std::endl;
  solver = std::make_shared<FRSolver>(&input);
  solver->setup(myComm);

  if (input.p_multi)
  {
    pmg = std::make_shared<PMGrid>();

    if (rank == 0) std::cout << "Setting up multigrid..." << std::endl;
    pmg->setup(&input, *solver, myComm);
  }

#ifdef _GPU
  start_cublas();
#endif

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

void Zefr::do_n_steps(int n)
{
  for (int i = 0; i < n; i++)
    do_step();
}

void Zefr::extrapolate_u(void)
{
  solver->eles->extrapolate_U(0, solver->eles->nEles);
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
    solver->write_surfaces(input.output_prefix);
}

void Zefr::write_forces(void)
{
  solver->report_forces(force_file);
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

void Zefr::get_basic_geo_data(int& btag, int& nnodes, double*& xyz, int*& iblank,
                              int& nwall, int& nover, int*& wallNodes,
                              int*& overNodes, int& nCellTypes, int &nvert_cell,
                              int &nCells_type, int*& c2v)
{
  btag = myGrid;
  nnodes = geo->nNodes;
  xyz = geo->coord_nodes.data();
  iblank = geo->iblank_node.data();
  nwall = geo->nWall;
  nover = geo->nOver;
  wallNodes = geo->wallNodes.data();
  overNodes = geo->overNodes.data();
  nCellTypes = 1;
  nvert_cell = geo->nNodesPerEle;
  nCells_type = geo->nEles;
  c2v = (int *)&geo->ele2nodes(0,0);
}

void Zefr::get_extra_geo_data(int& nFaceTypes, int& nvert_face,
                              int& nFaces_type, int*& f2v, int*& f2c, int*& c2f,
                              int*& iblank_face, int*& iblank_cell,
                              int &nOver, int*& overFaces, int &nMpiFaces, int*& mpiFaces, int*& procR,
                              int*& faceIdR)
{
  nFaceTypes = 1;
  nvert_face = geo->nNodesPerFace;
  nFaces_type = geo->nFaces;
  f2v = (int *)geo->face2nodes.data();
  f2c = geo->face2eles.data();
  c2f = geo->ele2face.data();
  iblank_face = geo->iblank_face.data();
  iblank_cell = geo->iblank_cell.data();
  nOver = geo->overFaceList.size();
  overFaces = geo->overFaceList.data();
  nMpiFaces = geo->nMpiFaces;
  mpiFaces = geo->mpiFaces.data();
  procR = geo->procR.data();
  faceIdR = geo->faceID_R.data();
}

double Zefr::get_u_spt(int ele, int spt, int var)
{
  return solver->eles->U_spts(spt, ele, var);
}

double Zefr::get_grad_spt(int ele, int spt, int dim, int var)
{
  return solver->eles->dU_spts(spt, ele, var, dim);
}

double *Zefr::get_u_spts(void)
{
  return solver->eles->U_spts.data();
}

double *Zefr::get_u_fpts(void)
{
  return solver->faces->U.data();
}

void Zefr::get_nodes_per_cell(int &nNodes)
{
  nNodes = (int)solver->eles->nSpts;
}

void Zefr::get_nodes_per_face(int& nNodes)
{
  nNodes = (int)geo->nFptsPerFace;
}

void Zefr::get_receptor_nodes(int cellID, int& nNodes, double* xyz)
{
  nNodes = (int)solver->eles->nSpts;

  for (int spt = 0; spt < nNodes; spt++)
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*spt+dim] = geo->coord_spts(spt, cellID, dim);
}

void Zefr::get_face_nodes(int faceID, int &nNodes, double* xyz)
{
  nNodes = (int)geo->nFptsPerFace;

  int start_fpt = geo->face2fpts(0, faceID);
  for (int fpt = 0; fpt < nNodes; fpt++)
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*fpt+dim] = solver->faces->coord(start_fpt + fpt, dim);
}

void Zefr::donor_inclusion_test(int cellID, double* xyz, int& passFlag, double* rst)
{
  passFlag = solver->eles->getRefLoc(cellID,xyz,rst);
}

void Zefr::donor_frac(int cellID, int &nweights, int* inode, double* weights,
                      double* rst, int buffsize)
{
  /* NOTE: inode is not used, and cellID is irrelevant when all cells are
   * identical (tensor-product, one polynomial order) */
  solver->eles->get_interp_weights(rst,weights,nweights,buffsize);
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
    geo->iblank_fpts(fpt) = geo->iblank_face[face];
  }
  geo->iblank_fpts_d = geo->iblank_fpts;
  geo->iblank_cell_d = geo->iblank_cell;
#endif
}

void Zefr::donor_data_from_device(int *donorIDs, int nDonors, int gradFlag)
{
#ifdef _GPU
  if (gradFlag == 0)
    solver->eles->donor_u_from_device(donorIDs, nDonors);
  else
    solver->eles->donor_grad_from_device(donorIDs, nDonors);

  check_error();
#endif
}

void Zefr::fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag)
{
#ifdef _GPU
  if (gradFlag == 0)
    solver->faces->fringe_u_to_device(fringeIDs, nFringe);
  else
    solver->faces->fringe_grad_to_device(nFringe);

  check_error();
#endif
}

void Zefr::set_dataUpdate_callback(void (*dataUpdate)(int nvar, double *q_spts, double *q_fpts, int gradFlag))
{
  solver->overset_interp = dataUpdate;
}

#endif /* _BUILD_LIB */
