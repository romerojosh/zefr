#include <cblas.h>
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

  std::stringstream ss;
  ss << "data_" << std::setw(7) << std::setfill('0') << 0 << ".vtk";
  std::cout << "Writing " << ss.str() << std::endl;
  solver.write_solution(ss.str(),0);

  for (unsigned int n = 1; n<=input.n_steps ; n++)
  {
    solver.update();

    if (n%input.report_freq == 0 || n == input.n_steps)
      solver.report(n);

    if (n%input.write_freq == 0 || n == input.n_steps)
    {
      std::cout << n << std::endl;
      std::stringstream ss;
      ss << "data_" << std::setw(7) << std::setfill('0') << n << ".vtk";
      std::cout << "Writing " << ss.str() << std::endl;
      solver.write_solution(ss.str(),n);
    }
  }

  return 0;
}
