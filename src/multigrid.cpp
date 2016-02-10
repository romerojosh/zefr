#include <iostream>
#include <memory>

#include "cblas.h"

#include "funcs.hpp"
#include "input.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

#ifdef _GPU
#include "solver_kernels.h"
#endif

//void PMGrid::setup(int order, const InputStruct *input, FRSolver &solver)
void PMGrid::setup(InputStruct *input, FRSolver &solver)
{
  this-> input = input;
  this-> order = input->order;
  corrections.resize(order + 1);
  sources.resize(order);
  solutions.resize(order);

  //TODO: Add check for h_multigrid levels
  if (input->hmg_levels > 0)
  {
    hcorrections.resize(input->hmg_levels);
    hsources.resize(input->hmg_levels);
    hsolutions.resize(input->hmg_levels);
  }

#ifdef _GPU
  corrections_d.resize(order + 1);
  sources_d.resize(order);
  solutions_d.resize(order);
#endif

  /* Instantiate coarse grid solvers */
  for (int P = 0; P < order; P++)
  {
    if (input->rank == 0) std::cout << "P = " << P << std::endl;
    grids.push_back(std::make_shared<FRSolver>(input,P));
    grids[P]->setup();

    /* Allocate memory for corrections and source terms */
    corrections[P] = grids[P]->eles->U_spts;
    sources[P] = grids[P]->eles->U_spts;
    solutions[P] = grids[P]->eles->U_spts;
    corrections[P].fill(0.0);
    sources[P].fill(0.0);
    solutions[P].fill(0.0);

#ifdef _GPU
    /* If using GPU, allocate device memory */
    corrections_d[P] = corrections[P];
    sources_d[P] = sources[P];
    solutions_d[P] = solutions[P];
#endif
  }


  /* Allocate memory for fine grid correction and initialize to zero */
  corrections[order] = solver.eles->U_spts;
  corrections[order].fill(0.0);

#ifdef _GPU
  corrections_d[order] = corrections[order];
#endif

  /* Instantiate h multigrid levels (if required) */
  if (input->hmg_levels > 0)
  {
    for (unsigned int H = 0; H < input->hmg_levels; H++)
    {
      unsigned int nElesFV = grids[0]->eles->nEles / (unsigned int) std::pow(2, input->hmg_levels - H);

      if (input->rank == 0) std::cout << "h_level = " << nElesFV << std::endl;

      hgrids.push_back(std::make_shared<FRSolver>(input,0));
      hgrids[H]->setup();

      /* Allocate memory for corrections and source terms */
      hcorrections[H] = hgrids[H]->eles->U_spts;
      hsources[H] = hgrids[H]->eles->U_spts;
      hsolutions[H] = hgrids[H]->eles->U_spts;
      hcorrections[H].fill(0.0);
      hsources[H].fill(0.0);
      hsolutions[H].fill(0.0);

    }
  }

}

void PMGrid::cycle(FRSolver &solver)
{
  /* Update residual on finest grid level and restrict */
  solver.compute_residual(0);
  restrict_pmg(solver, *grids[order-1]);

  for (int P = order-1; P >= (int) input->low_order; P--)
  {
    /* Generate source term */
#ifdef _CPU
    compute_source_term(*grids[P], sources[P]);
#endif

#ifdef _GPU
    compute_source_term(*grids[P], sources_d[P]);
#endif

    /* Copy initial solution to solution storage */
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
      for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
          solutions[P](spt, ele, n) = grids[P]->eles->U_spts(spt, ele, n);
#endif

#ifdef _GPU
    device_copy(solutions_d[P], grids[P]->eles->U_spts_d, solutions_d[P].get_nvals());
#endif

    /* Update solution on coarse level */
    for (unsigned int step = 0; step < input->smooth_steps; step++)
    {
#ifdef _CPU
      grids[P]->update_with_source(sources[P]);
#endif

#ifdef _GPU
      grids[P]->update_with_source(sources_d[P]);
#endif
    }

    if (P-1 >= (int) input->low_order)
    {
      /* Update residual and add source */
      grids[P]->compute_residual(0);
#ifdef _CPU
#pragma omp parallel for collapse(3)
      for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
        for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
          for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
            grids[P]->eles->divF_spts(spt, ele, n, 0) += sources[P](spt, ele, n);
#endif

#ifdef _GPU
      device_add(grids[P]->eles->divF_spts_d, sources_d[P], sources_d[P].get_nvals());
#endif

      /* Restrict to next coarse grid */
      restrict_pmg(*grids[P], *grids[P-1]);
    }
  }

  //TODO: Advance down and up h-levels here.
  ///* Restrict P0 solution to first FV level via averaging */
  // 1. Update residual on P0 and add source
  // 2. RESTRICT: Compute partition average solution and residual value
  //              Set solution point values to associated partition values
  // 3. Compute source term (need to be careful with what I define as the residual)
  // 4. Update using new FV style update method (with source):
  //    - Run step as normal up until dF/divF computation
  //    - Set "divF" in each element as flux integral around faces
  //    - Sum "divF" over partitions to get total residual for each partition
  //    - Update solution using total residual (will update average over entire partition)
  //
  //
  //for(int H = (int) input->hmg_levels - 1; H >= 0; H--)
  //{
  //
  //} 


  for (int P = (int) input->low_order; P <= order-1; P++)
  {

    /* Advance again (v-cycle)*/
    if (P != (int) input->low_order)
    {
      for (unsigned int step = 0; step < input->smooth_steps; step++)
      {
#ifdef _CPU
        grids[P]->update_with_source(sources[P]);
#endif

#ifdef _GPU
        grids[P]->update_with_source(sources_d[P]);
#endif
      }
    }
    else
    {
      for (unsigned int step = 0; step < input->smooth_steps; step++)
      {
#ifdef _CPU
        grids[P]->update_with_source(sources[P]);
#endif

#ifdef _GPU
        grids[P]->update_with_source(sources_d[P]);
#endif
      }

    }

    /* Generate error */
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < grids[P]->eles->nVars; n++)
      for (unsigned int ele = 0; ele < grids[P]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[P]->eles->nSpts; spt++)
          corrections[P](spt, ele, n) = grids[P]->eles->U_spts(spt, ele, n) - 
            solutions[P](spt, ele, n);
#endif

#ifdef _GPU
    /* Note: Doing this with two separate kernels might be more expensive. Can write a
     * single kernel for this eventually */
    device_subtract(grids[P]->eles->U_spts_d, solutions_d[P], solutions_d[P].get_nvals());
    device_copy(corrections_d[P], grids[P]->eles->U_spts_d, corrections_d[P].get_nvals());
#endif

    /* Prolong error and add to fine grid solution */
    if (P < order-1)
    {
#ifdef _CPU
      prolong_err(*grids[P], corrections[P], *grids[P+1]);
#endif

#ifdef _GPU
      prolong_err(*grids[P], corrections_d[P], *grids[P+1]);
#endif
    }
  }

  /* Prolong correction and add to finest grid solution */
#ifdef _CPU
  prolong_err(*grids[order-1], corrections[order-1], solver);
#endif

#ifdef _GPU
  prolong_err(*grids[order-1], corrections_d[order-1], solver);
#endif

}

void PMGrid::restrict_pmg(FRSolver &grid_f, FRSolver &grid_c)
{
  if (grid_f.order - grid_c.order > 1)
    ThrowException("Cannot restrict more than 1 order currently!");

#ifdef _CPU
  /* Restrict solution */
  auto &A = grid_f.eles->oppRes(0, 0);
  auto &B = grid_f.eles->U_spts(0, 0, 0);
  auto &C = grid_c.eles->U_spts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_c.eles->nSpts, &B, grid_f.eles->nSpts, 0.0, &C, grid_c.eles->nSpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_c.eles->nSpts, &B, grid_f.eles->nSpts, 0.0, &C, grid_c.eles->nSpts);
#endif

  
  auto &B2 = grid_f.eles->divF_spts(0, 0, 0, 0);
  auto &C2 = grid_c.eles->divF_spts(0, 0, 0, 0);

  /* Restrict residual */
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_c.eles->nSpts, &B2, grid_f.eles->nSpts, 0.0, &C2, grid_c.eles->nSpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_c.eles->nSpts, &B2, grid_f.eles->nSpts, 0.0, &C2, grid_c.eles->nSpts);
#endif

#endif

#ifdef _GPU
  /* Restrict solution */
  cublasDGEMM_wrapper(grid_c.eles->nSpts, grid_f.eles->nEles * grid_f.eles->nVars, 
      grid_f.eles->nSpts, 1.0, grid_f.eles->oppRes_d.data(), grid_c.eles->nSpts, 
      grid_f.eles->U_spts_d.data(), grid_f.eles->nSpts, 0.0, grid_c.eles->U_spts_d.data(), 
      grid_c.eles->nSpts);

  /* Restrict residual */
  cublasDGEMM_wrapper(grid_c.eles->nSpts, grid_f.eles->nEles * grid_f.eles->nVars, 
      grid_f.eles->nSpts, 1.0, grid_f.eles->oppRes_d.data(), grid_c.eles->nSpts, 
      grid_f.eles->divF_spts_d.data(), grid_f.eles->nSpts, 0.0, 
      grid_c.eles->divF_spts_d.data(), grid_c.eles->nSpts);
#endif
}

void PMGrid::prolong_pmg(FRSolver &grid_c, FRSolver &grid_f)
{
  if (grid_f.order - grid_c.order > 1)
    ThrowException("Cannot prolong more than 1 order currently!");

  for (unsigned int n = 0; n < grid_c.eles->nVars; n++)
  {
    auto &A = grid_c.eles->oppPro(0, 0);
    auto &B = grid_c.eles->U_spts(0, 0, n);
    auto &C = grid_f.eles->U_spts(0, 0, n);

#ifdef _OMP    
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_f.eles->nSpts, 
        &B, grid_c.eles->nSpts, 0.0, &C, grid_f.eles->nSpts);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_f.eles->nSpts, 
        &B, grid_c.eles->nSpts, 0.0, &C, grid_f.eles->nSpts);
#endif
  }

}

void PMGrid::prolong_err(FRSolver &grid_c, mdvector<double> &correction_c, FRSolver &grid_f)
{
  auto &A = grid_c.eles->oppPro(0, 0);
  auto &B = correction_c(0, 0, 0);
  auto &C = grid_f.eles->U_spts(0, 0, 0);

  /* Prolong error */
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
      grid_c.eles->nEles * grid_c.eles->nVars, grid_c.eles->nSpts, input->rel_fac, 
      &A, grid_f.eles->nSpts, &B, grid_c.eles->nSpts, 1.0, &C, grid_f.eles->nSpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
      grid_c.eles->nEles * grid_c.eles->nVars, grid_c.eles->nSpts, input->rel_fac, 
      &A, grid_f.eles->nSpts, &B, grid_c.eles->nSpts, 1.0, &C, grid_f.eles->nSpts);
#endif
}

#ifdef _GPU
void PMGrid::prolong_err(FRSolver &grid_c, mdvector_gpu<double> &correction_c, FRSolver &grid_f)
{
  /* Prolong error */
  cublasDGEMM_wrapper(grid_f.eles->nSpts, grid_f.eles->nEles * grid_f.eles->nVars, 
      grid_c.eles->nSpts, input->rel_fac, grid_c.eles->oppPro_d.data(), grid_f.eles->nSpts, 
      correction_c.data(), grid_c.eles->nSpts, 1.0, grid_f.eles->U_spts_d.data(),
      grid_f.eles->nSpts);
}
#endif 

void PMGrid::compute_source_term(FRSolver &grid, mdvector<double> &source)
{
  /* Copy restricted fine grid residual to source term */
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < grid.eles->nVars; n++)
    for (unsigned int ele = 0; ele < grid.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < grid.eles->nSpts; spt++)
        source(spt,ele,n) = grid.eles->divF_spts(spt,ele,n,0);

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < grid.eles->nVars; n++)
    for (unsigned int ele = 0; ele < grid.eles->nEles; ele++)
      for (unsigned int spt = 0; spt < grid.eles->nSpts; spt++)
        source(spt, ele, n) -= grid.eles->divF_spts(spt, ele, n, 0);

}

#ifdef _GPU
void PMGrid::compute_source_term(FRSolver &grid, mdvector_gpu<double> &source)
{
  /* Copy restricted fine grid residual to source term */
  device_copy(source, grid.eles->divF_spts_d, source.get_nvals());

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
  device_subtract(source, grid.eles->divF_spts_d, source.get_nvals());

}
#endif
