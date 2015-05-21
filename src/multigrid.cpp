#include <iostream>
#include <memory>

#include "cblas.h"
#include "omp.h"

#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

void PMGrid::setup(int order, const InputStruct *input, FRSolver &solver)
{
  this-> order = order;
  this-> input = input;
  corrections.resize(order + 1);
  sources.resize(order);
  solutions.resize(order);

  /* Instantiate coarse grid solvers */
  for (int P = 0; P < order; P++)
  {
    std::cout << "P = " << P << std::endl;
    grids.push_back(std::make_shared<FRSolver>(input,P));
    grids[P]->setup();

    /* Allocate memory for corrections and source terms */
    corrections[P] = grids[P]->eles->U_spts;
    sources[P] = grids[P]->eles->U_spts;
    solutions[P] = grids[P]->eles->U_spts;
    corrections[P].fill(0.0);
    sources[P].fill(0.0);
    solutions[P].fill(0.0);
  }

  /* Allocate memory for fine grid correction and initialize to zero */
  corrections[order] = solver.eles->U_spts;
  corrections[order].fill(0.0);
}

void PMGrid::cycle(FRSolver &solver)
{
  /* Update residual on finest grid level and restrict */
  solver.compute_residual(0);
  restrict_pmg(solver, *grids[order-1]);

  for (int P = order-1; P >= (int)solver.geo.shape_order-1; P--)
  {
    /* Generate source term */
    compute_source_term(*grids[P], sources[P]);

    /* Copy initial solution to solution storage */
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
      for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
          solutions[P](spt, ele, n) = grids[P]->eles->U_spts(spt, ele, n);

    /* Update solution on coarse level */
    for (unsigned int step = 0; step < input->smooth_steps; step++)
    {
      grids[P]->update_with_source(sources[P]);
    }

    if (P-1 >= (int)solver.geo.shape_order-1)
    {
      /* Update residual and add source */
      grids[P]->compute_residual(0);
#pragma omp parallel for collapse(3)
      for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
        for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
          for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
            grids[P]->divF(spt, ele, n, 0) += sources[P](spt, ele, n);

      /* Restrict to next coarse grid */
      restrict_pmg(*grids[P], *grids[P-1]);
    }
  }

  for (int P = solver.geo.shape_order-1; P <= order-1; P++)
  {

    /* Advance again (v-cycle)*/
    if (P != (int)solver.geo.shape_order-1)
    {
      for (unsigned int step = 0; step < input->smooth_steps; step++)
      {
        grids[P]->update_with_source(sources[P]);
      }
    }
    /*
    else
    {
      for (unsigned int step = 0; step < 4; step++)
      {
        grids[P]->update_with_source(sources[P]);
      }

    }
    */

    /* Generate error */
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
      for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
          corrections[P](spt, ele, n) = grids[P]->eles->U_spts(spt, ele, n) - solutions[P](spt, ele, n);

    /* Prolong error and add to fine grid solution */
    if (P < order-1)
      prolong_err(*grids[P], corrections[P], *grids[P+1], corrections[P+1]);
  }

  /* Prolong correction and add to finest grid solution */
  prolong_err(*grids[order-1], corrections[order-1], solver, corrections[order]);

}

void PMGrid::restrict_pmg(FRSolver &grid_f, FRSolver &grid_c)
{
  if (grid_f.order - grid_c.order > 1)
    ThrowException("Cannot restrict more than 1 order currently!");

#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = grid_f.eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += grid_f.eles->nEles % (block_size);

    for (unsigned int n = 0; n < grid_f.eles->nVars; n++)
    {
      auto &A = grid_f.eles->oppRes(0,0);
      auto &B = grid_f.eles->U_spts(0,start_idx,n);
      auto &C = grid_c.eles->U_spts(0,start_idx,n);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
          block_size, grid_f.eles->nSpts, 1.0, &A, grid_c.eles->nSpts, 
          &B, grid_f.eles->nSpts, 0.0, &C, grid_c.eles->nSpts);

      
      auto &B2 = grid_f.divF(0,start_idx,n,0);
      auto &C2 = grid_c.divF(0,start_idx,n,0);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
          block_size, grid_f.eles->nSpts, 1.0, &A, grid_c.eles->nSpts, 
          &B2, grid_f.eles->nSpts, 0.0, &C2, grid_c.eles->nSpts);
    }
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

#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = grid_f.eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += grid_f.eles->nEles % (block_size);

    for (unsigned int n = 0; n < grid_c.eles->nVars; n++)
    {
      auto &A = grid_c.eles->oppPro(0,0);
      auto &B = correction_c(0,start_idx,n);
      //auto &C = correction_f(0,start_idx,n);
      auto &C = grid_f.eles->U_spts(0,start_idx,n);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
          block_size, grid_c.eles->nSpts, input->rel_fac, &A, grid_f.eles->nSpts, 
          &B, grid_c.eles->nSpts, 1.0, &C, grid_f.eles->nSpts);
    }
  }
}

void PMGrid::compute_source_term(FRSolver &grid, mdvector<double> &source)
{
  /* Copy restricted fine grid residual to source term */
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < grid.eles->nVars; n++)
    for (unsigned int ele = 0; ele < grid.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < grid.eles->nSpts; spt++)
        source(spt,ele,n) = grid.divF(spt,ele,n,0);

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < grid.eles->nVars; n++)
    for (unsigned int ele = 0; ele < grid.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < grid.eles->nSpts; spt++)
        source(spt, ele, n) -= grid.divF(spt, ele, n, 0);

}
