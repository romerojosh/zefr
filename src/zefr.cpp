#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage:" << std::endl;
    std::cout << "\t" << argv[0] << " <input file>" << std::endl;
    exit(1);
  }

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

  solver.write_solution(input.output_prefix,0);
  std::ofstream hist_file;
  hist_file.open(input.output_prefix + "_hist.dat");

  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int n = 1; n<=input.n_steps ; n++)
  {
    solver.update();

    if (input.p_multi)
      pmg.cycle(solver);

    if (n%input.report_freq == 0 || n == input.n_steps)
    {
      solver.report_max_residuals(hist_file , n, t1);
    }

    if (n%input.write_freq == 0 || n == input.n_steps)
    {
      solver.write_solution(input.output_prefix,n);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  solver.compute_l2_error();
 
  auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  std::cout << "Elapsed time: " << elapsed_time.count() << " s" << std::endl;

  return 0;
}
