#ifndef _zefr_hpp
#define _zefr_hpp

#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _MPI
#include "mpi.h"
#endif

class Elements;
class Faces;
class FRSolver;
class PMGrid;

#include "input.hpp"
#include "macros.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

class zefr
{
public:

  /*! Assign MPI communicator and overset-grid ID
   *
   * For MPI cases, Python will call MPI_Init, so we should NOT do that here
   * Instead, Python will give us our own communicator for internal use
   * (1 communicator per grid)
   */
#ifdef _MPI
  zefr(MPI_Comm comm_in = MPI_COMM_WORLD, int n_grids = 1, int grid_id = 0);
#else
  zefr(void);
#endif

  //! Read input file and set basic run parameters
  void read_input(char *inputfile);

  //! Perform preprocessing and prepare to run case
  void setup_solver(void);

  //! Run one full time step, including any filtering or multigrid operations
  void do_step(void);

  //! Call "do_step()" n times
  void do_n_steps(int n);

  void extrapolate_u();

  // Functions to write data to file and/or terminal
  void write_residual(void);
  void write_solution(void);
  void write_forces(void);
  void write_error(void);

  // Other Misc. Functions
  InputStruct &get_input(void) { return input; }

  /* ==== Overset-Related Functions ==== */

  // Geometry Access Functions
  void get_basic_geo_data(int &btag, int &nnodes, double *&xyz, int *&iblank,
                          int &nwall, int &nover, int *&wallNodes, int *&overNodes,
                          int &nCellTypes, int &nvert_cell, int &nCells_type,
                          int *&c2v);

  void get_extra_geo_data(int &nFaceTypes, int& nvert_face, int& nFaces_type,
                          int*& f2v, int*& f2c, int*& c2f, int*& iblank_face,
                          int*& iblank_cell);

  // Solution-data access functions
  double *get_u_spts(void);
  double *get_u_fpts(void);

  // Callback Functions for TIOGA
  void get_nodes_per_cell(int& nNodes);
  void get_nodes_per_face(int& nNodes);
  void get_receptor_nodes(int cellID, int& nNodes, double* xyz);
  void get_face_nodes(int faceID, int& nNodes, double* xyz);
  void get_q_index_face(int faceID, int fpt, int& ind, int& stride);
  void donor_inclusion_test(int cellID, double* xyz, int& passFlag, double* rst);
  void donor_frac(int cellID, int& nweights, int* inode,
                  double* weights, double* rst, int buffsize);

private:
  // Generic data about the run
  int rank = 0, nRanks = 1;
  int myGrid = 0;  //! For overset: which grid this rank belongs to
  int nGrids = 1;  //! For overset: # of grids in entire system

  // Basic ZEFR Solver Objects
  std::shared_ptr<FRSolver> solver;
  InputStruct input;
  std::shared_ptr<PMGrid> pmg;
  GeoStruct *geo;

  // Files to write history output
  std::chrono::high_resolution_clock::time_point t_start;

  std::ofstream hist_file;
  std::ofstream force_file;
  std::ofstream error_file;

#ifdef _MPI
  void mpi_init(MPI_Comm comm_in, int n_grids = 1, int grid_id = 0);
#endif

  // Again, to simplify MPI vs. no-MPI compilation, this will be either an
  // MPI_Comm or an int
  _mpi_comm myComm;
};

#endif /* _zefr_hpp */
