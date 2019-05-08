#include <iostream>
#include <memory>

#ifdef _MKL_BLAS
#include "mkl_cblas.h"
#elif _ESSL_BLAS
#include "essl.h"
#else
#include "cblas.h"
#endif

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

  correctionsBT.resize(nLevels);
  sourcesBT.resize(nLevels);
  solutionsBT.resize(nLevels);

#ifdef _GPU
  correctionsBT_d.resize(nLevels);
  sourcesBT_d.resize(nLevels);
  solutionsBT_d.resize(nLevels);
#endif


  bool tri_flag = solver.geo.nElesBT.count(TRI); // check if triangles are in grid

  /* Setup fine grid PMG operators */
  for (auto e : solver.elesObjs)
  {
    if (e->etype == QUAD and tri_flag) // need to adjust for increased quad order in this case
    {
      e->setup_PMG(order + 1, input->mg_levels[1] + 1);
    }
    else
      e->setup_PMG(order, input->mg_levels[1]);
  }
  grids.emplace_back(); // Placeholder spot in grids array

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

    for (auto e : grids[n]->elesObjs)
    {
      if (e->etype == QUAD and tri_flag)
        e->setup_PMG(P_pro + 1, P_res + 1);
      else
        e->setup_PMG(P_pro, P_res);

      /* Allocate memory for corrections and source terms */
      correctionsBT[n][e->etype] = e->U_spts;
      sourcesBT[n][e->etype] = e->U_spts;
      solutionsBT[n][e->etype] = e->U_spts;
      correctionsBT[n][e->etype].fill(0.0);
      sourcesBT[n][e->etype].fill(0.0);
      solutionsBT[n][e->etype].fill(0.0);

#ifdef _GPU
      /* If using GPU, allocate device memory */
      correctionsBT_d[n][e->etype] = correctionsBT[n][e->etype];
      sourcesBT_d[n][e->etype] = sourcesBT[n][e->etype];
      solutionsBT_d[n][e->etype] = solutionsBT[n][e->etype];
#endif
    }
  }

  /* Allocate memory for fine grid correction and initialize to zero */
  for (auto e : solver.elesObjs)
  {
    correctionsBT[0][e->etype] = e->U_spts;
    correctionsBT[0][e->etype].fill(0.0);

#ifdef _GPU
    correctionsBT_d[0][e->etype] = correctionsBT[0][e->etype];
#endif
  }
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
    compute_source_term(*grids[n], sourcesBT[n]);
#endif

#ifdef _GPU
    compute_source_term(*grids[n], sourcesBT_d[n]);
#endif

    /* Copy initial solution to solution storage */
#ifdef _CPU
    for (auto e : grids[n]->elesObjs)
    {
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && grids[n]->geo.iblank_cell(ele) != NORMAL) continue;

          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            solutionsBT[n][e->etype](spt, var, ele) = e->U_spts(spt, var, ele);
        }
    }
#endif

#ifdef _GPU
    for (auto e : grids[n]->elesObjs)
      device_copy(solutionsBT_d[n][e->etype], e->U_spts_d, solutionsBT_d[n][e->etype].max_size());
#endif

    /* Update solution on coarse level */
    for (unsigned int step = 0; step < input->mg_steps[n]; step++)
    {
#ifdef _CPU
      grids[n]->update(sourcesBT[n]);
      grids[n]->filter_solution();
#endif

#ifdef _GPU
      grids[n]->update(sourcesBT_d[n]);
      grids[n]->filter_solution();
#endif
    }
    
    /* If coarser level exits, restrict */
    if (n + 1 < nLevels)
    {
      /* Update residual and add source */
      grids[n]->compute_residual(0);
#ifdef _CPU
      for (auto e : grids[n]->elesObjs)
      {
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && grids[n]->geo.iblank_cell(ele) != NORMAL) continue;

            for (unsigned int spt = 0; spt < e->nSpts; spt++)
              e->divF_spts(0, spt, var, ele) += sourcesBT[n][e->etype](spt, var, ele);
          }
      }
#endif

#ifdef _GPU
      for (auto e : grids[n]->elesObjs)
        device_add(e->divF_spts_d, sourcesBT_d[n][e->etype], sourcesBT_d[n][e->etype].max_size());
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
        grids[n]->update(sourcesBT[n]);
        grids[n]->filter_solution();
#endif

#ifdef _GPU
        grids[n]->update(sourcesBT_d[n]);
        grids[n]->filter_solution();
#endif
      }
    }

    /* Generate error */
#ifdef _CPU
    for (auto e : grids[n]->elesObjs)
    {
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && grids[n]->geo.iblank_cell(ele) != NORMAL) continue;

          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            correctionsBT[n][e->etype](spt, var, ele) = e->U_spts(spt, var, ele) - 
              solutionsBT[n][e->etype](spt, var, ele);
        }
    }
#endif

#ifdef _GPU
    /* Note: Doing this with two separate kernels might be more expensive. Can write a
     * single kernel for this eventually */
    for (auto e : grids[n]->elesObjs)
    {
      device_subtract(e->U_spts_d, solutionsBT_d[n][e->etype], solutionsBT_d[n][e->etype].max_size());
      device_copy(correctionsBT_d[n][e->etype], e->U_spts_d, correctionsBT_d[n][e->etype].max_size());
    }
#endif

    /* Prolong error and add to fine grid solution */
    if (n > level + 1)
    {
#ifdef _CPU
      prolong_err(*grids[n], correctionsBT[n], *grids[n - 1]);
#endif

#ifdef _GPU
      prolong_err(*grids[n], correctionsBT_d[n], *grids[n - 1]);
#endif
    }
  }

  /* Prolong correction and add to finest grid solution */
  if (level != nLevels - 1)
  {
#ifdef _CPU
    prolong_err(*grids[level + 1], correctionsBT[level + 1], solver);
#endif

#ifdef _GPU
    prolong_err(*grids[level + 1], correctionsBT_d[level + 1], solver);
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
      if (input->rank == 0)
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
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;

      /* Restrict solution */
      auto &A = ef->oppRes(0, 0);
      auto &B = ef->U_spts(0, 0, 0);
      auto &C = ec->U_spts(0, 0, 0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ec->nSpts, 
          ef->nEles * ef->nVars, ef->nSpts, 1.0, &A, 
          ef->nSpts, &B, ef->nEles * ef->nVars, 0.0, &C, ef->nEles * ef->nVars);

      
      auto &B2 = ef->divF_spts(0, 0, 0, 0);
      auto &C2 = ec->divF_spts(0, 0, 0, 0);

      /* Restrict residual */
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ec->nSpts, 
          ef->nEles * ef->nVars, ef->nSpts, 1.0, &A, 
          ef->nSpts, &B2, ef->nEles * ef->nVars, 0.0, &C2, ef->nEles * ef->nVars);
    }
  }

#endif

#ifdef _GPU
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;

      auto *A = ef->oppRes_d.data();
      auto *B = ef->U_spts_d.data();
      auto *C = ec->U_spts_d.data();

      /* Restrict solution */
      cublasDGEMM_wrapper(ef->nElesPad * ef->nVars, ec->nSpts,
          ef->nSpts, 1.0, B, ef->nElesPad * ef->nVars, A, ef->nSpts, 0.0, C, 
          ef->nElesPad * ef->nVars);

      auto *B2 = ef->divF_spts_d.data();
      auto *C2 = ec->divF_spts_d.data();

      /* Restrict residual */
      cublasDGEMM_wrapper(ef->nElesPad * ef->nVars, ec->nSpts,
          ef->nSpts, 1.0, B2, ef->nElesPad * ef->nVars,  
          A, ef->nSpts, 0.0, C2, ef->nElesPad * ef->nVars);
    }
  }
#endif
}

void PMGrid::prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector<double>> &correctionBT, FRSolver &grid_f)
{
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;

      auto &A = ec->oppPro(0, 0);
      auto &B = correctionBT[ec->etype](0, 0, 0);
      auto &C = ef->U_spts(0, 0, 0);

      /* Prolong error */
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ef->nSpts, 
          ec->nEles * ec->nVars, ec->nSpts, input->rel_fac, 
          &A, ec->nSpts, &B, ec->nEles * ec->nVars, 1.0, &C, ec->nEles * ec->nVars);
    }
  }
}

#ifdef _GPU
void PMGrid::prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector_gpu<double>> &correctionBT_d, FRSolver &grid_f)
{
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;

      auto *A = ec->oppPro_d.data();
      auto *B = correctionBT_d[ec->etype].data();
      auto *C = ef->U_spts_d.data();

      /* Prolong error */
      cublasDGEMM_wrapper(ec->nElesPad * ec->nVars, ef->nSpts, ec->nSpts, input->rel_fac, 
          B, ec->nElesPad * ec->nVars, A, ec->nSpts, 1.0, C, ec->nElesPad * ec->nVars);
    }
  }
}
#endif 

void PMGrid::prolong_U(FRSolver &grid_c, FRSolver &grid_f)
{
#ifdef _CPU
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;
      auto &A = ec->oppPro(0, 0);
      auto &B = ec->U_spts(0, 0, 0);
      auto &C = ef->U_spts(0, 0, 0);

      /* Prolong error */
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ef->nSpts, 
          ec->nEles * ec->nVars, ec->nSpts, 1.0, 
          &A, ec->nSpts, &B, ec->nEles * ec->nVars, 0.0, &C, ec->nEles * ec->nVars);
    }
  }
#endif

#ifdef _GPU
  for (auto ef : grid_f.elesObjs)
  {
    for (auto ec : grid_c.elesObjs)
    {
      if (ef->etype != ec->etype) continue;

      auto *A = ec->oppPro_d.data();
      auto *B = ec->U_spts_d.data();
      auto *C = ef->U_spts_d.data();

      cublasDGEMM_wrapper(ec->nElesPad * ec->nVars, ef->nSpts, ec->nSpts, 1.0, 
          B, ec->nElesPad * ec->nVars, A, ec->nSpts, 0.0, C, ec->nElesPad * ec->nVars);

    }
  }
#endif
}

void PMGrid::compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector<double>> &sourceBT)
{
  /* Copy restricted fine grid residual to source term */
  for (auto e : grid.elesObjs)
  {
    for (unsigned int n = 0; n < e->nVars; n++)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && grids[n]->geo.iblank_cell(ele) != NORMAL) continue;

        for (unsigned int spt = 0; spt < e->nSpts; spt++)
          sourceBT[e->etype](spt, n, ele) = e->divF_spts(0, spt, n, ele);
      }
  }

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
  for (auto e : grid.elesObjs)
  {
    for (unsigned int n = 0; n < e->nVars; n++)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
          sourceBT[e->etype](spt, n, ele) -= e->divF_spts(0, spt, n, ele);
  }

}

#ifdef _GPU
void PMGrid::compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT_d)
{
  /* Copy restricted fine grid residual to source term */
  for (auto e : grid.elesObjs)
    device_copy(sourceBT_d[e->etype], e->divF_spts_d, sourceBT_d[e->etype].max_size());

  /* Update residual on current coarse grid */
  grid.compute_residual(0);

  /* Subtract to generate source term */
  for (auto e : grid.elesObjs)
    device_subtract(sourceBT_d[e->etype], e->divF_spts_d, sourceBT_d[e->etype].max_size());

}
#endif
