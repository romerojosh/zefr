#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

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

  //! Read input file and set basic run parameters
  void initialize(char *inputfile);

  //! Perform preprocessing and prepare to run case
  void setup_solver(void);

  //! Run one full time step, including any filtering or multigrid operations
  void do_step(void);

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
};
