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
    if (n%input.write_freq == 0 || n == input.n_steps)
    {
      std::cout << n << std::endl;
      std::stringstream ss;
      ss << "data_" << std::setw(7) << std::setfill('0') << n << ".vtk";
      std::cout << "Writing " << ss.str() << std::endl;
      solver.write_solution(ss.str(),n);
    }
  }




  /*
  Quads eles(1,  2, &input);
  Faces faces(8, &input);

  eles.associate_faces(&faces);
  eles.setup();
  */

  /*
  mdvector<double> A({2,3,3},1.);
  mdvector<double> B({2,3,3});
  mdvector<double> C({3,3});

  B(0,0,0) = 1.;
  B(0,1,0) = 2.;
  B(0,2,0) = 3.;
  B(0,0,1) = 4.;
  B(0,1,1) = 5.;
  B(0,2,1) = 6.;
  B(0,0,2) = 7.;
  B(0,1,2) = 8.;
  B(0,2,2) = 9.;

  B(1,0,0) = 1.;
  B(1,1,0) = 4.;
  B(1,2,0) = 7.;
  B(1,0,1) = 2.;
  B(1,1,1) = 5.;
  B(1,2,1) = 8.;
  B(1,0,2) = 3.;
  B(1,1,2) = 6.;
  B(1,2,2) = 9.;

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0, &A(1,0,0),
    3, &B(1,0,0), 3, 1.0, &C(0,0), 3);

  std::cout << C << std::endl;

  */
}
