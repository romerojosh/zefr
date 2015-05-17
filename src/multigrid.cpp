#include <iostream>
#include <memory>

#include "cblas.h"

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
  restrict_pmg(solver, *grids[1]);
  restrict_pmg(*grids[1], *grids[0]);

  grids[0]->write_solution("res0",0);
  grids[1]->write_solution("res1",0);
}

void PMGrid::restrict_pmg(FRSolver &grid_f, FRSolver &grid_c)
{
  if (grid_f.order - grid_c.order > 1)
    ThrowException("Cannot restrict more than 1 order currently!");

  for (unsigned int n = 0; n < grid_f.eles->nVars; n++)
  {
    auto &A = grid_f.eles->oppRes(0,0);
    auto &B = grid_f.eles->U_spts(0,0,n);
    auto &C = grid_c.eles->U_spts(0,0,n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
        grid_c.eles->nEles, grid_f.eles->nSpts, 1.0, &A, grid_c.eles->nSpts, 
        &B, grid_f.eles->nSpts, 0.0, &C, grid_c.eles->nSpts);
  }
}
void PMGrid::prolong_pmg(FRSolver &grid_f, FRSolver &grid_c)
{
}
