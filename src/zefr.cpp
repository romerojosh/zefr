#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _MPI
#include "mpi.h"
#ifdef _GPU
#include "cuda_runtime.h"
#endif
#endif

#include "input.hpp"
#include "macros.hpp"
#include "multigrid.hpp"
#include "solver.hpp"
#include "solver_kernels.h"
#include "filter.hpp"

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

  cudaSetDevice(rank%6); // Hardcoded for ICME nodes for now.
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
  solver.write_solution(input.output_prefix);
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
      solver.write_solution(input.output_prefix);
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
//#define _MPI        // for QT editing
#ifdef _BUILD_LIB
#include "zefr.hpp"

#ifdef _MPI
zefr::zefr(MPI_Comm comm_in, int n_grids, int grid_id)
#else
zefr::zefr(void)
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
void zefr::mpi_init(MPI_Comm comm_in, int n_grids, int grid_id)
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

  cudaSetDevice(rank%6); // Hardcoded for ICME nodes for now.
#endif
}
#endif

void zefr::read_input(char *inputfile)
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

void zefr::setup_solver(void)
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

void zefr::do_step(void)
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

void zefr::do_n_steps(int n)
{
  for (int i = 0; i < n; i++)
    do_step();
}

void zefr::extrapolate_u(void)
{
  solver->eles->extrapolate_U(0, solver->eles->nEles);
}

void zefr::write_residual(void)
{
  solver->report_residuals(hist_file, t_start);
}

void zefr::write_solution(void)
{
  solver->write_solution(input.output_prefix);
}

void zefr::write_forces(void)
{
  solver->report_forces(force_file);
}

void zefr::write_error(void)
{
  solver->report_error(error_file);
}

void zefr::get_basic_geo_data(int& btag, int& nnodes, double*& xyz, int*& iblank,
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

void zefr::get_extra_geo_data(int& nFaceTypes, int& nvert_face,
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

double zefr::get_u_spt(int ele, int spt, int var)
{
  return solver->eles->U_spts(spt, ele, var);
}

double *zefr::get_u_spts(void)
{
  return solver->eles->U_spts.data();
}

double *zefr::get_u_fpts(void)
{
  return solver->faces->U.data();
}

void zefr::get_nodes_per_cell(int &nNodes)
{
  nNodes = (int)solver->eles->nSpts;
}

void zefr::get_nodes_per_face(int& nNodes)
{
  nNodes = (int)geo->nFptsPerFace;
}

void zefr::get_receptor_nodes(int cellID, int& nNodes, double* xyz)
{
  nNodes = (int)solver->eles->nSpts;

  for (int spt = 0; spt < nNodes; spt++)
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*spt+dim] = geo->coord_spts(spt, cellID, dim);
}

void zefr::get_face_nodes(int faceID, int &nNodes, double* xyz)
{
  nNodes = (int)geo->nFptsPerFace;

  int start_fpt = geo->face2fpts(0, faceID);
  for (int fpt = 0; fpt < nNodes; fpt++)
    for (int dim = 0; dim < geo->nDims; dim++)
      xyz[3*fpt+dim] = solver->faces->coord(start_fpt + fpt, dim);
}

void zefr::get_q_index_face(int faceID, int fpt, int& ind, int& stride)
{
  solver->faces->get_U_index(faceID,fpt,ind,stride);
}

void zefr::donor_inclusion_test(int cellID, double* xyz, int& passFlag, double* rst)
{
  passFlag = solver->eles->getRefLoc(cellID,xyz,rst);
}

void zefr::donor_frac(int cellID, int &nweights, int* inode, double* weights,
                      double* rst, int buffsize)
{
  solver->eles->get_interp_weights(cellID,rst,inode,weights,nweights,buffsize);
}

#endif /* _BUILD_LIB */
