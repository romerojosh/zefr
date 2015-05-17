#include <iostream>
#include <memory>
#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

void PMGrid::setup(int order, const InputStruct *input)
{
  
  this-> order = order;
  corrections.resize(order);



  /* Instantiate coarse grid solvers */
  for (int n = 0; n < order; n++)
  {
    std::cout << "P = " << n << std::endl;
    grids.push_back(std::make_shared<FRSolver>(input,n));
    grids[n]->setup();
  }


}

void PMGrid::cycle(FRSolver &solver)
{
  for (int n = 0; n < order; n++)
    grids[n]->update();
}
