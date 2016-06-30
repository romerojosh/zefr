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

#include "input.hpp"
#include "macros.hpp"
#include "multigrid.hpp"
#include "solver.hpp"
#include "solver_kernels.h"
#include "filter.hpp"

class zefr
{
public:

  zefr(void);

  /*! Assign MPI communicator and overset-grid ID
   *
   * For MPI cases, Python will call MPI_Init, so we should NOT do that here
   * Instead, Python will give us our own communicator for internal use
   * (1 communicator per grid)
   */
#ifdef _MPI
  void mpi_init(MPI_Comm comm_in, int grid_id=0);
#endif

  //! Read input file and set basic run parameters
  void initialize(char *inputfile);

  //! Perform preprocessing and prepare to run case
  void setup_solver(void);

  //! Run one full time step, including any filtering or multigrid operations
  void do_step(void);

  // Functions to write data to file and/or terminal
  void write_residual(void);
  void write_solution(void);
  void write_forces(void);
  void write_error(void);

  ~zefr(void) { delete solver; }

private:
  // Generic data about the run
  int rank = 0, nRanks = 1;
  int myGrid = 0;  //! For overset: which grid this rank belongs to

  // Basic ZEFR Solver Objects
  FRSolver *solver;
  InputStruct input;
  PMGrid pmg;

  // Files to write history output
  std::chrono::high_resolution_clock::time_point t_start;

  std::ofstream hist_file;
  std::ofstream force_file;
  std::ofstream error_file;

  // Again, to simplify MPI vs. no-MPI compilation, this will be either an
  // MPI_Comm or an int
  _mpi_comm myComm;
};

#endif /* _zefr_hpp */
