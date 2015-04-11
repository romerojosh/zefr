#include <cblas.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "input.hpp"
#include "solver.hpp"

int main()
{
  auto input = read_input_file("input.txt");

  FRSolver solver(&input);
  solver.setup();
  //solver.compute_residual();
  std::stringstream ss;
  ss << "data" << 0 << ".vtk";
  solver.write_solution(ss.str());
  for (unsigned int n = 1; n<=input.n_steps ; n++)
  {
    std::cout << n << std::endl;
    solver.update();
    if (n%10 == 0)
    {
      std::stringstream ss;
      ss << "data" << n << ".vtk";
      solver.write_solution(ss.str());
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
