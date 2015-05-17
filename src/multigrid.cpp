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

    /* Allocate memory for corrections (same size as divF) */
    corrections[n] = grids[n]->divF;
  }
}

void PMGrid::cycle(FRSolver &solver)
{
  for (int n = 0; n < order; n++)
    grids[n]->update();
}

void PMGrid::restrict_pmg(FRSolver &grid_fine, FRSolver &grid_coarse)
{
}
void PMGrid::prolong_pmg(FRSolver &grid_fine, FRSolver &grid_coarse)
{
}
