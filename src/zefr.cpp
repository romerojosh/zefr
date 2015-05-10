#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "input.hpp"
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

  solver.write_solution(input.output_prefix,0);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (unsigned int n = 1; n<=input.n_steps ; n++)
  {
    solver.update();

    if (n%input.report_freq == 0 || n == input.n_steps)
    {
      std::cout << n << " ";
      solver.report_max_residuals();
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
