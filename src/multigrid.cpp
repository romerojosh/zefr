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
void PMGrid::setup(InputStruct *input, FRSolver &solver, _mpi_comm comm_in)
{
  this-> order = input->order;
  this-> input = input;
  nLevels = input->mg_levels.size();

  if (input->mg_levels.size() == 0 or input->mg_steps.size() == 0)
    ThrowException("Need to provide mg_levels and/or mg_steps to run multigrid!");
  if (input->mg_levels[0] != order)
    ThrowException("Invalid mg_levels provided. First level must equal order!");
  if (input->mg_levels.size() != input->mg_steps.size())
    ThrowException("Inconsistent number of mg_levels and mg_steps provided!");

  corrections.resize(nLevels);
  sources.resize(nLevels);
  solutions.resize(nLevels);

#ifdef _GPU
  corrections_d.resize(nLevels);
  sources_d.resize(nLevels);
  solutions_d.resize(nLevels);
#endif

  /* Setup fine grid PMG operators */
  solver.eles->setup_PMG(order, input->mg_levels[1]);
  grids.push_back(nullptr); // Placeholder spot in grids array

  /* Instantiate coarse grid solvers */
  for (int n = 1; n < nLevels; n++)
  {
    int P = input->mg_levels[n];
    int P_pro = input->mg_levels[n - 1];
    int P_res = (n + 1 < nLevels) ? input->mg_levels[n + 1] : 0;

    if (input->rank == 0) std::cout << "P = " << P << std::endl;
    if (input->rank == 0) std::cout << "P_pro = " << P_pro << std::endl;
    if (input->rank == 0) std::cout << "P_res = " << P_res << std::endl;
    grids.push_back(std::make_shared<FRSolver>(input, P));
    grids[n]->setup(comm_in);
    grids[n]->eles->setup_PMG(P_pro, P_res);

    /* Allocate memory for corrections and source terms */
    corrections[n] = grids[n]->eles->U_spts;
    sources[n] = grids[n]->eles->U_spts;
    solutions[n] = grids[n]->eles->U_spts;
    corrections[n].fill(0.0);
    sources[n].fill(0.0);
    solutions[n].fill(0.0);

#ifdef _GPU
    /* If using GPU, allocate device memory */
    corrections_d[n] = corrections[n];
    sources_d[n] = sources[n];
    solutions_d[n] = solutions[n];
#endif
  }

  /* Allocate memory for fine grid correction and initialize to zero */
  corrections[0] = solver.eles->U_spts;
  corrections[0].fill(0.0);

#ifdef _GPU
  corrections_d[0] = corrections[0];
#endif
}

void PMGrid::v_cycle(FRSolver &solver, int level)
{
  /* ---Downward cycle--- */
  /* Update residual on finest grid level and restrict */
  if (level != nLevels - 1)
  {
    solver.compute_residual(0);
    restrict_pmg(solver, *grids[level + 1]);
  }

  for (int n = level + 1; n < nLevels; n++)
  {
    /* Generate source term */
#ifdef _CPU
    compute_source_term(*grids[n], sources[n]);
#endif

#ifdef _GPU
    compute_source_term(*grids[n], sources_d[n]);
#endif

    /* Copy initial solution to solution storage */
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int var = 0; var < grids[n]->eles->nVars; var++)
      for (unsigned int ele = 0; ele < grids[n]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[n]->eles->nSpts; spt++)
          solutions[n](spt, ele, var) = grids[n]->eles->U_spts(spt, ele, var);
#endif

#ifdef _GPU
    device_copy(solutions_d[n], grids[n]->eles->U_spts_d, solutions_d[n].max_size());
#endif

    /* Update solution on coarse level */
    for (unsigned int step = 0; step < input->mg_steps[n]; step++)
    {
#ifdef _CPU
      grids[n]->update(sources[n]);
      grids[n]->filter_solution();
#endif

#ifdef _GPU
      grids[n]->update(sources_d[n]);
      grids[n]->filter_solution();
#endif
    }
    
    /* If coarser level exits, restrict */
    if (n + 1 < nLevels)
    {
      /* Update residual and add source */
      grids[n]->compute_residual(0);
#ifdef _CPU
#pragma omp parallel for collapse(3)
      for (unsigned int var = 0; var < grids[n]->eles->nVars; var++)
        for (unsigned int ele = 0; ele < grids[n]->eles->nEles; ele++)
          for (unsigned int spt = 0; spt < grids[n]->eles->nSpts; spt++)
            grids[n]->eles->divF_spts(spt, ele, var, 0) += sources[n](spt, ele, var);
#endif

#ifdef _GPU
      device_add(grids[n]->eles->divF_spts_d, sources_d[n], sources_d[n].max_size());
#endif

      /* Restrict to next coarse level */
      restrict_pmg(*grids[n], *grids[n + 1]);
    }
  }

  /* ---Upward cycle--- */
  for (int n = nLevels - 1; n > level; n--)
  {

    /* Advance again (v-cycle)*/
    if (n != nLevels - 1)
    {
      for (unsigned int step = 0; step < input->mg_steps[n]; step++)
      {
#ifdef _CPU
        grids[n]->update(sources[n]);
        grids[n]->filter_solution();
#endif

#ifdef _GPU
        grids[n]->update(sources_d[n]);
        grids[n]->filter_solution();
#endif
      }
    }

    /* Generate error */
#ifdef _CPU
#pragma omp parallel for collapse(3)
    for (unsigned int var = 0; var < grids[n]->eles->nVars; var++)
      for (unsigned int ele = 0; ele < grids[n]->eles->nEles; ele++)
        for (unsigned int spt = 0; spt < grids[n]->eles->nSpts; spt++)
          corrections[n](spt, ele, var) = grids[n]->eles->U_spts(spt, ele, var) - 
            solutions[n](spt, ele, var);
#endif

#ifdef _GPU
    /* Note: Doing this with two separate kernels might be more expensive. Can write a
     * single kernel for this eventually */
    device_subtract(grids[n]->eles->U_spts_d, solutions_d[n], solutions_d[n].max_size());
    device_copy(corrections_d[n], grids[n]->eles->U_spts_d, corrections_d[n].max_size());
#endif

    /* Prolong error and add to fine grid solution */
    if (n > level + 1)
    {
#ifdef _CPU
      prolong_err(*grids[n], corrections[n], *grids[n - 1]);
#endif

#ifdef _GPU
      prolong_err(*grids[n], corrections_d[n], *grids[n - 1]);
#endif
    }
  }

  /* Prolong correction and add to finest grid solution */
  if (level != nLevels - 1)
  {
#ifdef _CPU
    prolong_err(*grids[level + 1], corrections[level + 1], solver);
#endif

#ifdef _GPU
    prolong_err(*grids[level + 1], corrections_d[level + 1], solver);
#endif
  }

}

void PMGrid::cycle(FRSolver &solver, std::ofstream& histfile, std::chrono::high_resolution_clock::time_point t1)
{
  if (input->mg_cycle == "V")
  {
    for (int step = 0; step < input->mg_steps[0]; step++)
    {
      solver.update();
      solver.filter_solution();
    }

    v_cycle(solver, 0);
  }
  else if (input->mg_cycle == "FMG")
  {
    /* Perform FMG cycle */
    for (int n = nLevels - 1; n > 0; n--)
    {
      std::cout << "FMG P: " << input->mg_levels[n] << std::endl;
      for (unsigned int cycle = 0; cycle < input->FMG_vcycles; cycle++)
      {
        /* Update current level */
        grids[n]->update();

        /* Do a v-cycle on current level */
        v_cycle(*grids[n], n); //Loop this?

        if (cycle == 0 or cycle % input->report_freq == 0)
          grids[n]->report_residuals(histfile, t1);
      }

      /* Update before prolongation */
      grids[n]->update();


      /* Prolong solution to next level */
      if (n > 1)
      {
#ifdef _CPU
        prolong_U(*grids[n], *grids[n - 1]);
#endif

#ifdef _GPU
        prolong_U(*grids[n], *grids[n - 1]);
#endif
        grids[n - 1]->write_solution("FMG_prolong_"+std::to_string(input->mg_levels[n - 1]));
      }
    }

    /* Prolong solution to finest level */
#ifdef _CPU
    prolong_U(*grids[1], solver);
#endif

#ifdef _GPU
    prolong_U(*grids[1], solver);
#endif

    /* Set cycle to use V-cycle */
    input->mg_cycle = std::string("V");
    solver.write_solution("FMG_prolong_"+std::to_string(order));

    /* Perform first V-cycle */
    cycle(solver, histfile, t1);

  }
  else
  {
    ThrowException("Multigrid cycle type unknown!");
  }
}

void PMGrid::restrict_pmg(FRSolver &grid_f, FRSolver &grid_c)
{
#ifdef _CPU
  /* Restrict solution */
  auto &A = grid_f.eles->oppRes(0, 0);
  auto &B = grid_f.eles->U_spts(0, 0, 0);
  auto &C = grid_c.eles->U_spts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_f.eles->oppRes.ldim(), &B, grid_f.eles->U_spts.ldim(), 0.0, &C, grid_c.eles->U_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_f.eles->oppRes.ldim(), &B, grid_f.eles->U_spts.ldim(), 0.0, &C, grid_c.eles->U_spts.ldim());
#endif

  
  auto &B2 = grid_f.eles->divF_spts(0, 0, 0, 0);
  auto &C2 = grid_c.eles->divF_spts(0, 0, 0, 0);

  /* Restrict residual */
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_f.eles->oppRes.ldim(), &B2, grid_f.eles->divF_spts.ldim(), 0.0, &C2, grid_c.eles->divF_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_c.eles->nSpts, 
      grid_f.eles->nEles * grid_f.eles->nVars, grid_f.eles->nSpts, 1.0, &A, 
      grid_f.eles->oppRes.ldim(), &B2, grid_f.eles->divF_spts.ldim(), 0.0, &C2, grid_c.eles->divF_spts.ldim());
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
  for (unsigned int n = 0; n < grid_c.eles->nVars; n++)
  {
    auto &A = grid_c.eles->oppPro(0, 0);
    auto &B = grid_c.eles->U_spts(0, 0, n);
    auto &C = grid_f.eles->U_spts(0, 0, n);

#ifdef _OMP    
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_c.eles->oppPro.ldim(), 
        &B, grid_c.eles->U_spts.ldim(), 0.0, &C, grid_f.eles->U_spts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
        grid_f.eles->nEles, grid_c.eles->nSpts, 1.0, &A, grid_c.eles->oppPro.ldim(), 
        &B, grid_c.eles->U_spts.ldim(), 0.0, &C, grid_f.eles->U_spts.ldim());
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
      &A, grid_c.eles->oppPro.ldim(), &B, correction_c.ldim(), 1.0, &C, grid_f.eles->U_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
      grid_c.eles->nEles * grid_c.eles->nVars, grid_c.eles->nSpts, input->rel_fac, 
      &A, grid_c.eles->oppPro.ldim(), &B, correction_c.ldim(), 1.0, &C, grid_f.eles->U_spts.ldim());
#endif
}

void PMGrid::prolong_U(FRSolver &grid_c, FRSolver &grid_f)
{
#ifdef _CPU
  auto &A = grid_c.eles->oppPro(0, 0);
  auto &B = grid_c.eles->U_spts(0, 0, 0);
  auto &C = grid_f.eles->U_spts(0, 0, 0);

  /* Prolong error */
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
      grid_c.eles->nEles * grid_c.eles->nVars, grid_c.eles->nSpts, 1.0, 
      &A, grid_c.eles->oppPro.ldim(), &B, grid_c.eles->U_spts.ldim(), 0.0, &C, grid_f.eles->U_spts.ldim());
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, grid_f.eles->nSpts, 
      grid_c.eles->nEles * grid_c.eles->nVars, grid_c.eles->nSpts, 1.0, 
      &A, grid_c.eles->oppPro.ldim(), &B, grid_c.eles->U_spts.ldim(), 0.0, &C, grid_f.eles->U_spts.ldim());
#endif
#endif

#ifdef _GPU
  cublasDGEMM_wrapper(grid_f.eles->nSpts, grid_c.eles->nEles * grid_c.eles->nVars, 
      grid_c.eles->nSpts, 1.0, grid_c.eles->oppPro_d.data(), grid_f.eles->nSpts, 
      grid_c.eles->U_spts_d.data(), grid_c.eles->nSpts, 0.0, grid_f.eles->U_spts_d.data(), 
      grid_f.eles->nSpts);
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
  device_copy(source, grid.eles->divF_spts_d, source.max_size());

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
  device_subtract(source, grid.eles->divF_spts_d, source.max_size());

}
#endif
