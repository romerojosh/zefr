#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"
#include "solver_kernels.h"

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << " <input file>" << std::endl;
    exit(1);
  }

  /* Print out cool ascii art header */
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
                                                   

  std::string inputfile = argv[1];

  std::cout << "Reading input file: " << inputfile <<  std::endl;
  auto input = read_input_file(inputfile);

  std::cout << "Setting up FRSolver..." << std::endl;
  FRSolver solver(&input);
  solver.setup();

  PMGrid pmg;
  if (input.p_multi)
  {
    std::cout << "Setting up multigrid..." << std::endl;
    pmg.setup(input.order, &input, solver);
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
    hist_file.open(input.output_prefix + "_hist.dat", std::ios::app);
    force_file.open(input.output_prefix + "_forces.dat", std::ios::app);
    error_file.open(input.output_prefix + "_error.dat", std::ios::app);
  }
  else
  {
    hist_file.open(input.output_prefix + "_hist.dat");
    force_file.open(input.output_prefix + "_forces.dat");
    error_file.open(input.output_prefix + "_error.dat");
  }

  /* Write initial solution */
  solver.write_solution(input.output_prefix, 0);

  /* Write initial error (if required) */
  if (input.error_freq != 0)
    solver.report_error(error_file, 0);

  auto t1 = std::chrono::high_resolution_clock::now();
  /* Main iteration loop */
  for (unsigned int n = 1; n<=input.n_steps ; n++)
  {
    solver.update();

    /* If using multigrid, perform correction cycle */
    if (input.p_multi)
      pmg.cycle(solver);

    /* Write output if required */
    if (input.report_freq != 0 && (n%input.report_freq == 0 || n == input.n_steps || n == 1))
    {
      solver.report_residuals(hist_file , n, t1);
    }

    if (input.write_freq != 0 && (n%input.write_freq == 0 || n == input.n_steps))
    {
      solver.write_solution(input.output_prefix,n);
    }

    if (input.force_freq != 0 && (n%input.force_freq == 0 || n == input.n_steps))
    {
      solver.report_forces(input.output_prefix, force_file, n);
    }

    if (input.error_freq != 0 && (n%input.error_freq == 0 || n == input.n_steps))
    {
      solver.report_error(error_file, n);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
 
  auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "Elapsed time: " << elapsed_time.count() << " s" << std::endl;

  return 0;
}
