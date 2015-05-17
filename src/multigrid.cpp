#include <iostream>
#include <memory>

#include "cblas.h"

#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

void PMGrid::setup(int order, const InputStruct *input)
{
  this-> order = order;
  corrections.resize(order + 1);
  sources.resize(order);

  /* Instantiate coarse grid solvers */
  for (int P = 0; P < order; P++)
  {
    std::cout << "P = " << P << std::endl;
    grids.push_back(std::make_shared<FRSolver>(input,P));
    grids[P]->setup();

    /* Allocate memory for corrections and source terms */
    corrections[P] = grids[P]->eles->U_spts;
    sources[P] = grids[P]->eles->U_spts;
  }
}

void PMGrid::cycle(FRSolver &solver)
{
  /*
  restrict_pmg(solver, *grids[1]);
  prolong_pmg(*grids[1], solver);

  grids[1]->write_solution("res",0);
  solver.write_solution("pro",0);
  */

  corrections[order] = solver.eles->U_spts;
  for (unsigned int n = 0; n < solver.eles->nVars; n++)
    for (unsigned int ele = 0; ele < solver.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < solver.eles->nSpts; spt++)
        corrections[order](spt, ele, n) = 0.0;

  /* Update residual on finest grid level and restrict */
  solver.compute_residual(0);
  restrict_pmg(solver, *grids[order-1]);

  for (unsigned int P = order-1; P > 0; P--)
  {
    /* Generate source term */
    compute_source_term(*grids[P], sources[P]);

    /* Copy initial solution to correction storage */
    std::copy(&grids[P]->eles->U_spts(0,0,0), 
        &grids[P]->eles->U_spts(grids[P]->eles->nSpts-1, grids[P]->eles->nEles-1, 
          grids[P]->eles->nVars-1), &corrections[P](0,0,0));

    /* Update solution on coarse level */
    grids[P]->update_with_source(sources[P]);

    /* Generate error */
    for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
      for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
          corrections[P](spt, ele, n) -= grids[P]->eles->U_spts(spt, ele, n);
  }

  for (int P = 1; P < order-1; P++)
  {
    prolong_err(*grids[P], corrections[P], *grids[P+1], corrections[P+1]);
  }

  /* Prolong correction to finest grid */
  prolong_err(*grids[order-1], corrections[order-1], solver, corrections[order]);

  /* Add correction to fine grid solution */
  for (unsigned int n = 0; n < solver.eles->nVars; n++)
    for (unsigned int ele = 0; ele < solver.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < solver.eles->nSpts; spt++)
        solver.eles->U_spts(spt, ele, n) += corrections[order](spt, ele, n);

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

    
    auto &B2 = grid_f.divF(0,0,n,0);
    auto &C2 = grid_c.divF(0,0,n,0);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
        grid_c.eles->nEles, grid_f.eles->nSpts, 1.0, &A, grid_c.eles->nSpts, 
        &B2, grid_f.eles->nSpts, 0.0, &C2, grid_c.eles->nSpts);
  }
}
void PMGrid::prolong_pmg(FRSolver &grid_c, FRSolver &grid_f)
{
  if (grid_f.order - grid_c.order > 1)
    ThrowException("Cannot prolong more than 1 order currently!");

  for (unsigned int n = 0; n < grid_c.eles->nVars; n++)
  {
    auto &A = grid_c.eles->oppPro(0,0);
    auto &B = grid_c.eles->U_spts(0,0,n);
    auto &C = grid_f.eles->U_spts(0,0,n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_f.eles->nSpts, 
        &B, grid_c.eles->nSpts, 0.0, &C, grid_f.eles->nSpts);
  }

}

void PMGrid::prolong_err(FRSolver &grid_c, mdvector<double> &correction_c,
    FRSolver &grid_f, mdvector<double> &correction_f)
{
  for (unsigned int n = 0; n < grid_c.eles->nVars; n++)
  {
    auto &A = grid_c.eles->oppPro(0,0);
    auto &B = correction_c(0,0,n);
    auto &C = correction_f(0,0,n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_f.eles->nSpts, 
        &B, grid_c.eles->nSpts, 1.0, &C, grid_f.eles->nSpts);
  }
}

void PMGrid::compute_source_term(FRSolver &grid, mdvector<double> &source)
{
  /* Copy restricted fine grid residual to source term */
  std::copy(&grid.divF(0,0,0,0), &grid.divF(grid.eles->nSpts-1, grid.eles->nEles-1, 
        grid.eles->nVars-1, 1), &source(0,0,0));

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
  for (unsigned int n = 0; n < grid.eles->nVars; n++)
    for (unsigned int ele = 0; ele < grid.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < grid.eles->nSpts; spt++)
        source(spt, ele, n) -= grid.divF(spt, ele, n, 0);

}
