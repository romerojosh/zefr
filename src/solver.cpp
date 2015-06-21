#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>

#include <cblas.h>
#include <omp.h>

#include "elements.hpp"
#include "faces.hpp"
#include "funcs.hpp"
#include "geometry.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

FRSolver::FRSolver(const InputStruct *input, int order)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;
}

void FRSolver::setup()
{
  std::cout << "Reading mesh: " << input->meshfile << std::endl;
  geo = process_mesh(input->meshfile, order, input->nDims);

  std::cout << "Setting up elements and faces..." << std::endl;
  eles = std::make_shared<Quads>(&geo, input, order);
  faces = std::make_shared<Faces>(&geo, input);
  eles->associate_faces(faces);
  eles->setup();

  std::cout << "Initializing solution..." << std::endl;
  initialize_U();

  if (input->restart)
  {
    std::cout << "Restarting solution from " + input->restart_file +" ..." << std::endl;
    restart(input->restart_file);
  }

  std::cout << "Setting up timestepping..." << std::endl;
  setup_update();

  std::cout << "Setting up output..." << std::endl;
  setup_output();

}

void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  if (input->dt_scheme == "Euler")
  {
    nStages = 1;
    rk_beta = {1.0};
  }
  else if (input->dt_scheme == "RK44")
  {
    nStages = 4;
    rk_alpha = {0.5, 0.5, 1.0};
    rk_beta = {1./6., 1./3., 1./3., 1./6.};
  }
  else if (input->dt_scheme == "RK54")
  {
    nStages = 5;
    rk_alpha = {-0.417890474499852, -1.192151694642677, -1.697784692471528, -1.514183444257156};
    rk_beta = {0.149659021999229, 0.379210312999627, 0.822955029386982, 0.699450455949122, 0.153057247968152};
  }
  else
  {
    ThrowException("dt_scheme not recognized!");
  }

  U_ini.assign({eles->nSpts, eles->nEles, eles->nVars});
  divF.assign({eles->nSpts, eles->nEles, eles->nVars, nStages});

  // TODO: Should I create this array for user-supplied to save branch?
  dt.assign(eles->nEles,input->dt);

}

void FRSolver::setup_output()
{
  if (eles->nDims == 2)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 4;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({4, eles->nSubelements});

    /* Setup plot "subelement" connectivity */
    std::vector<unsigned int> nd(4,0);

    unsigned int ele = 0;
    nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;

    for (unsigned int i = 0; i < nSubelements1D; i++)
    {
      for (unsigned int j = 0; j < nSubelements1D; j++)
    {
        for (unsigned int node = 0; node < 4; node ++)
        {
          geo.ppt_connect(node, ele) = nd[node] + j;
        }

        ele++;
      }

      for (unsigned int node = 0; node < 4; node ++)
        nd[node] += nSubelements1D + 1;
    }
  }
  else
  {
    ThrowException("3D not implemented yet!");
  }

}

void FRSolver::restart(std::string restart_file)
{
  std::ifstream f(restart_file);

  if (!f.is_open())
    ThrowException("Could not open specified restart file!");

  std::string param, line;
  double val;
  unsigned int n = 0;
  unsigned int nPpts1D = eles->nSpts1D + 2;

  while (f >> param)
  {
    if (param == "TIME")
    {
      std::getline(f,line);
      f >> flow_time;
    }
    if (param == "CYCLE")
    {
      std::getline(f,line);
      f >> restart_iter;
    }
    if (param == "SCALARS")
    {
      f >> param;
      std::cout << "Reading " << param  << std::endl;
      /* Skip lines */
      std::getline(f,line);
      std::getline(f,line);
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        unsigned int spt = 0;
        for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
        {
          f >> val;

          /* Logic to deal with extra plot point (corner nodes and flux points). */
          if (ppt < nPpts1D || ppt > nPpts1D * (nPpts1D-1) || ppt%nPpts1D == 0 || (ppt+1)%nPpts1D == 0)
            continue;

          eles->U_spts(spt, ele, n) = val;
          spt++;
        }
      }
      n++;
    }

  }

}

void FRSolver::compute_residual(unsigned int stage)
{
  extrapolate_U();

  U_to_faces();
  faces->apply_bcs();
  eles->compute_Fconv();
  faces->compute_Fconv();

  if (input->viscous)
  {
    faces->compute_common_U();
    U_from_faces(); 
    compute_dU();
    extrapolate_dU();
    dU_to_faces();
    faces->apply_bcs_dU();
    eles->compute_Fvisc();
    faces->compute_Fvisc();
  }

  faces->compute_common_F();
  eles->transform_flux();
  F_from_faces();


  compute_dF();
  compute_divF(stage);
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nSpts, eles->nEles, eles->nVars});
  eles->U_fpts.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->Ucomm.assign({eles->nFpts, eles->nEles, eles->nVars});
  eles->U_ppts.assign({eles->nPpts, eles->nEles, eles->nVars});
  eles->U_qpts.assign({eles->nQpts, eles->nEles, eles->nVars});

  eles->F_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->F_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->Fcomm.assign({eles->nFpts, eles->nEles, eles->nVars});

  eles->dU_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dU_fpts.assign({eles->nFpts, eles->nEles, eles->nVars, eles->nDims});
  eles->dF_spts.assign({eles->nSpts, eles->nEles, eles->nVars, eles->nDims});

  eles->divF_spts.assign({eles->nSpts, eles->nEles, eles->nVars});

  /* Initialize solution */
  // TODO: Fill in with actual logic. */
  if (input->equation == "AdvDiff")
  {
    if (input->ic_type == 0)
    {
      // Do nothing for now
    }
    else if (input->ic_type == 1)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          double x = geo.coord_spts(spt, ele, 0);
          double y = geo.coord_spts(spt, ele, 1);

          eles->U_spts(spt, ele, 0) = compute_U_true(x, y, 0, 0, input);
        }
      }
    }
    else
    {
      ThrowException("ic_type not recognized!");
    }
  }
  else if (input->equation == "EulerNS")
  {
    if (input->ic_type == 0)
    { 
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        {
          eles->U_spts(spt, ele, 0)  = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < eles->nDims; dim++)
          {
            eles->U_spts(spt, ele, dim+1)  = input->rho_fs * input->V_fs[dim];
            Vsq += input->V_fs[dim] * input->V_fs[dim];
          }

          eles->U_spts(spt, ele, 3)  = input->P_fs/(input->gamma-1.0) + 
            0.5*input->rho_fs * Vsq;
        }
      }

    }
    else if (input->ic_type == 1)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double x = geo.coord_spts(spt, ele, 0);
            double y = geo.coord_spts(spt, ele, 1);

            eles->U_spts(spt, ele, n) = compute_U_true(x, y, 0, n, input);
          }
        }
      }
    }
  }
  else
  {
    ThrowException("Solution initialization not recognized!");
  }
}

void FRSolver::extrapolate_U()
{
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppE(0,0);
      auto &B = eles->U_spts(0,start_idx,n);
      auto &C = eles->U_fpts(0,start_idx,n);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, block_size,
            eles->nSpts, 1.0, &A, eles->nFpts, &B, eles->nSpts, 0.0, &C, eles->nFpts);
    }
  }

}

void FRSolver::U_to_faces()
{
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
        {
          if (input->viscous) // if viscous, put extrapolated solution into Ucomm
            eles->Ucomm(fpt, ele, n) = eles->U_fpts(fpt, ele, n);
          continue;
        }
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        faces->U(slot, n, gfpt) = eles->U_fpts(fpt, ele, n);
      }
    }
  }
}

void FRSolver::U_from_faces()
{  
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Ucomm(fpt, ele, n) = faces->Ucomm(slot, n, gfpt);
      }
    }
  }

}

void FRSolver::compute_dU()
{
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);

    /* Compute contribution to derivative from solution at solution points */
    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        auto &A = eles->oppD(0,0,dim);
        auto &B = eles->U_spts(0,start_idx,n);
        auto &C = eles->dU_spts(0,start_idx,n,dim);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, block_size,
              eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->nSpts);
      }
    }

    /* Compute contribution to derivative from common solution at flux points */
    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        auto &A = eles->oppD_fpts(0,0,dim);
        auto &B = eles->Ucomm(0,start_idx,n);
        auto &C = eles->dU_spts(0,start_idx,n,dim);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, block_size,
              eles->nFpts, 1.0, &A, eles->nSpts, &B, eles->nFpts, 1.0, &C, eles->nSpts);
      }
    }
  }


    /* Transform dU back to physical space */
    if (eles->nDims == 2)
    {
#pragma omp parallel for collapse(3)
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          {
            double dUtemp = eles->dU_spts(spt, ele, n, 0);
            
            eles->dU_spts(spt, ele, n, 0) = eles->dU_spts(spt, ele, n, 0) * eles->jaco_spts(1, 1, spt, ele)-
                                      eles->dU_spts(spt, ele, n, 1) * eles->jaco_spts(1, 0, spt, ele); 
            eles->dU_spts(spt, ele, n, 1) = -dUtemp * eles->jaco_spts(0, 1, spt, ele) +
                                      eles->dU_spts(spt, ele, n, 1) * eles->jaco_spts(0, 0, spt, ele); 

            eles->dU_spts(spt, ele, n, 0) /= eles->jaco_det_spts(spt, ele);
            eles->dU_spts(spt, ele, n, 1) /= eles->jaco_det_spts(spt, ele);
          }
        }
      }
    }
}

void FRSolver::extrapolate_dU()
{
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);


    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        auto &A = eles->oppE(0,0);
        auto &B = eles->dU_spts(0,start_idx,n,dim);
        auto &C = eles->dU_fpts(0,start_idx,n,dim);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, block_size,
            eles->nSpts, 1.0, &A, eles->nFpts, &B, eles->nSpts, 0.0, &C, eles->nFpts);
      }
    }
  }

}

void FRSolver::dU_to_faces()
{
#pragma omp parallel for collapse(4)
  for (unsigned int dim = 0; dim < eles->nDims; dim++) 
  {
    for (unsigned int n = 0; n < eles->nVars; n++) 
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfpt(fpt,ele);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          int slot = geo.fpt2gfpt_slot(fpt,ele);

          faces->dU(slot, n, dim, gfpt) = eles->dU_fpts(fpt, ele, n, dim);
        }
      }
    }
  }
}

void FRSolver::F_from_faces()
{
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(fpt,ele);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(fpt,ele);

        eles->Fcomm(fpt, ele, n) = faces->Fcomm(slot, n, gfpt);
      }
    }
  }
}

void FRSolver::compute_dF()
{
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);


    /* Compute contribution to derivative from flux at solution points */
    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        auto &A = eles->oppD(0,0,dim);
        auto &B = eles->F_spts(0,start_idx,n,dim);
        auto &C = eles->dF_spts(0,start_idx,n,dim);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, block_size,
              eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->nSpts);
      }
    }

    /* Compute contribution to derivative from common flux at flux points */
    for (unsigned int dim = 0; dim < eles->nDims; dim++)
    {
      for (unsigned int n = 0; n < eles->nVars; n++)
      {
        auto &A = eles->oppD_fpts(0,0,dim);
        auto &B = eles->Fcomm(0,start_idx,n);
        auto &C = eles->dF_spts(0,start_idx,n,dim);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, block_size,
            eles->nFpts, 1.0, &A, eles->nSpts, &B, eles->nFpts, 1.0, &C, eles->nSpts);
      }
    }
  }

}

void FRSolver::compute_divF(unsigned int stage)
{
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
    for (unsigned int ele =0; ele < eles->nEles; ele++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        divF(spt, ele, n, stage) = eles->dF_spts(spt, ele, n, 0);

  for (unsigned int dim = 1; dim < eles->nDims; dim ++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele =0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          divF(spt, ele, n, stage) += eles->dF_spts(spt, ele, n, dim);

}

void FRSolver::update()
{
  
    U_ini = eles->U_spts;

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type == 1)
      {
        compute_element_dt();
        double min_dt = *std::min_element(dt.begin(), dt.end());
        std::fill(dt.begin(), dt.end(), min_dt);
      }
      else if (input->dt_type == 2)
      {
        compute_element_dt();
      }
    }

#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha[stage] * dt[ele] / eles->jaco_det_spts(spt,ele) * divF(spt, ele, n, stage);
  }

  /* Final stage */
  compute_residual(nStages-1);
  eles->U_spts = U_ini;

  for (unsigned int stage = 0; stage < nStages; stage++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(spt, ele, n) -=  rk_beta[stage] * dt[ele] / eles->jaco_det_spts(spt,ele) * divF(spt, ele, n, stage);

  flow_time += dt[0];
 
}

void FRSolver::update_with_source(mdvector<double> &source)
{
  
    U_ini = eles->U_spts;

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type == 1)
      {
        compute_element_dt();
        double min_dt = *std::min_element(dt.begin(), dt.end());
        std::fill(dt.begin(), dt.end(), min_dt);
      }
      else if (input->dt_type == 2)
      {
        compute_element_dt();
      }
    }

#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(spt, ele, n) = U_ini(spt, ele, n) - rk_alpha[stage] * dt[ele] / eles->jaco_det_spts(spt,ele)* 
            (divF(spt, ele, n, stage) + source(spt, ele, n));
  }

  /* Final stage */
  compute_residual(nStages-1);
  eles->U_spts = U_ini;

  for (unsigned int stage = 0; stage < nStages; stage++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(spt, ele, n) -=  rk_beta[stage] * dt[ele] / eles->jaco_det_spts(spt,ele) * (divF(spt, ele, n, stage) + source(spt, ele, n));

}

void FRSolver::compute_element_dt()
{
#pragma omp parallel for
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  { 
    double waveSp_max = 0.0;

    /* Compute maximum wavespeed */
    for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
    {
      int gfpt = geo.fpt2gfpt(fpt,ele);
      if (gfpt == -1)
        continue;

      double waveSp = faces->waveSp[gfpt] / faces->dA[gfpt];

      waveSp_max = std::max(waveSp, waveSp_max);
    }

    /* Experiment: Scale CFL with order */
    /* Note: CFL is applied to parent space element with width 2 */
    //dt[ele] = (input->CFL) / ((order+1)*(order+1)) * (2.0 / (waveSp_max+1.e-10));
    dt[ele] = (input->CFL) * get_cfl_limit(order) * (2.0 / (waveSp_max+1.e-10));
    //dt[ele] = (input->CFL) / ((order+1) * (order+1) + 0.4) * (2.0 / (waveSp_max+1.e-10));
  }
}

void FRSolver::write_solution(std::string prefix, unsigned int nIter)
{
  std::stringstream ss;
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << nIter + restart_iter << ".vtk";

  auto outputfile = ss.str();
  std::cout << "Writing " << outputfile << std::endl;

  /* Write solution to file in .vtk format */
  std::ofstream f(outputfile);

  /* Write header */
  f << "# vtk DataFile Version 3.0" << std::endl;
  f << "vtk output" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET UNSTRUCTURED_GRID" << std::endl;
  f << std::endl;

  /* Write field data */
  f << "FIELD FieldData 2" << std::endl;
  f << "TIME 1 1 double" << std::endl;
  f << nIter * input->dt << std::endl;
  f << "CYCLE 1 1 int" << std::endl;
  f << nIter + restart_iter << std::endl;
  f << std::endl;
  
  /* Write plot point coordinates */
  f << "POINTS " << eles->nPpts*eles->nEles << " double" << std::endl;
  if (eles->nDims == 2)
  {
    // TODO: Change order of ppt structures for better looping 
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << geo.coord_ppts(ppt, ele, 0) << " ";
        f << geo.coord_ppts(ppt, ele, 1) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    ThrowException("3D not implemented!");
  }
  f << std::endl;

  /* Write cell information */
  unsigned int nCells = eles->nSubelements * eles->nEles;
  f << "CELLS " << nCells << " " << (1+eles->nNodesPerSubelement)*nCells << std::endl;
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int subele = 0; subele < eles->nSubelements; subele++)
    {
      f << eles->nNodesPerSubelement << " "; 
      for (unsigned int i = 0; i < eles->nNodesPerSubelement; i++)
      {
        f << geo.ppt_connect(i, subele) + ele*eles->nPpts << " ";
      }
      f << std::endl;
    }
  }
  f << std::endl;

  f << "CELL_TYPES " << nCells << std::endl;
  if (eles->nDims == 2)
  {
    for (unsigned int cell = 0; cell < nCells; cell++)
      f << 9 << std::endl;
  }
  else
  {
    ThrowException("3D not implemented!");
  }
  f << std::endl;

  /* Write solution information */
 #pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);

   /* Extrapolate solution to plot points */
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppE_ppts(0,0);
      auto &B = eles->U_spts(0,start_idx,n);
      auto &C = eles->U_ppts(0,start_idx,n);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, block_size,
          eles->nSpts, 1.0, &A, eles->nPpts, &B, eles->nSpts, 0.0, &C, eles->nPpts);
    }
  }

  if (input->equation == "AdvDiff")
  {
    f << "POINT_DATA " << eles->nPpts*eles->nEles << std::endl;
    f << "SCALARS U double 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << std::scientific << std::setprecision(12) << eles->U_ppts(ppt, ele, 0) << " ";
      }
      f << std::endl;
    }
  }
  else if(input->equation == "EulerNS")
  {
    std::array<std::string,4> var = {"rho", "xmom", "ymom", "energy"};

    f << "POINT_DATA " << eles->nPpts*eles->nEles << std::endl;

    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      f << "SCALARS " << var[n] <<" double 1" << std::endl;
      f << "LOOKUP_TABLE default" << std::endl;
      
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
        {
          f << std::scientific << std::setprecision(12) << eles->U_ppts(ppt, ele, n) << " ";
        }

        f << std::endl;
      }

      f << std::endl;
    }
  }
}

void FRSolver::report_max_residuals(std::ofstream &f, unsigned int iter, 
    std::chrono::high_resolution_clock::time_point t1)
{
  std::vector<double> max_res(eles->nVars,0.0);

#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
    for (unsigned int ele =0; ele < eles->nEles; ele++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      divF(spt, ele, n, nStages-1) /= eles->jaco_det_spts(spt, ele);
  //for (unsigned int n = 0; n < eles->nVars; n++)
  //  max_res[n] = *std::max_element(&divF(0, 0, n, nStages-1), &divF(eles->nSpts-1, eles->nEles-1, n, nStages-1));
  for (unsigned int n = 0; n < eles->nVars; n++)
    max_res[n] = std::sqrt(std::accumulate(&divF(0, 0, n, nStages-1), &divF(eles->nSpts-1, eles->nEles-1, n, nStages-1), 0.0, square<double>()));

  std::cout << iter + restart_iter << " ";
  for (auto &val : max_res)
    std::cout << std::scientific << val << " ";

  std::cout << "dt: " << dt[0];
  std::cout << std::endl;
  
  /* Write to history file */
  auto t2 = std::chrono::high_resolution_clock::now();
  auto current_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
  f << iter + restart_iter << " " << current_runtime.count() << " ";

  for (auto &val : max_res)
    f << std::scientific << val << " ";
  f << std::endl;

}

void FRSolver::report_forces(std::string prefix, std::ofstream &f, unsigned int iter)
{
  std::array<double, 3> force_conv, force_visc, taun;
  force_conv.fill(0.0); force_visc.fill(0.0); taun.fill(0.0);

  std::stringstream ss;
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter + restart_iter << ".cp";
  auto cpfile = ss.str();
  std::ofstream g(cpfile);

  /* Get angle of attack */
  double aoa = std::atan2(input->V_fs[1], input->V_fs[0]); 

  /* Compute factor for non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
    Vsq += input->V_fs[dim] * input->V_fs[dim];

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd[fpt - geo.nGfpts_int];

    if (bnd_id >= 7) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(0,fpt);

      double CP = (PL - input->P_fs) * fac;

      /* Write CP distrubtion to file */
      g << std:: scientific << faces->coord(fpt, 0) << " " << faces-> coord(fpt, 1) << " " << CP << std::endl;

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < eles->nDims; dim++)
      {
        force_conv[dim] += eles->weights_spts[count%eles->nSpts1D] * CP * faces->norm(0, dim, fpt) * 
          faces->dA[fpt];
      }

      if (input->viscous)
      {
        if (eles->nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(0, 0, fpt);
          double momx = faces->U(0, 1, fpt);
          double momy = faces->U(0, 2, fpt);
          double e = faces->U(0, 3, fpt);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(0, 0, 0, fpt);
          double momx_dx = faces->dU(0, 1, 0, fpt);
          double momy_dx = faces->dU(0, 2, 0, fpt);
          
          double rho_dy = faces->dU(0, 0, 1, fpt);
          double momx_dy = faces->dU(0, 1, 1, fpt);
          double momy_dy = faces->dU(0, 2, 1, fpt);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(0, 0, fpt) + tauxy * faces->norm(0, 1, fpt);
          taun[1] = tauxy * faces->norm(0, 0, fpt) + tauyy * faces->norm(0, 1, fpt);

          for (unsigned int dim = 0; dim < eles->nDims; dim++)
            force_visc[dim] -= eles->weights_spts[count%eles->nSpts1D] * taun[dim] * faces->dA[fpt] * fac;

        }
        else
        {
          ThrowException("Under construction!");
        }
        
      }
      count++;
    }
  }

  /* Compute lift and drag coefficients */
  double CL_conv, CD_conv, CL_visc, CD_visc;
  if (eles->nDims == 2)
  {
    CL_conv = -force_conv[0] * std::sin(aoa) + force_conv[1] * std::cos(aoa);
    CD_conv = force_conv[0] * std::cos(aoa) + force_conv[1] * std::sin(aoa);
    CL_visc = -force_visc[0] * std::sin(aoa) + force_visc[1] * std::cos(aoa);
    CD_visc = force_visc[0] * std::cos(aoa) + force_visc[1] * std::sin(aoa);
  }
  else
  {
    ThrowException("Under construction!");
  }

  std::cout << "CL_conv = " << CL_conv << " CD_conv = " << CD_conv;
  f << iter + restart_iter << " ";
  f << std::scientific << CL_conv << " " << CD_conv;

  if (input->viscous)
  {
    std::cout << " CL_visc = " << CL_visc << " CD_visc = " << CD_visc;
    f << std::scientific << " " << CL_visc << " " << CD_visc;
  }

  std::cout << std::endl;
  f << std::endl;

}

void FRSolver::compute_l2_error()
{
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = eles->nEles / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += eles->nEles % (block_size);


    /* Extrapolate solution to quadrature points */
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppE_qpts(0,0);
      auto &B = eles->U_spts(0,start_idx,n);
      auto &C = eles->U_qpts(0,start_idx,n);

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, eles->nQpts, block_size,
          eles->nSpts, 1.0, &A, eles->nQpts, &B, eles->nSpts, 0.0, &C, eles->nQpts);
    }
  }


  std::vector<double> l2_error(eles->nVars,0.0);

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
#pragma omp for collapse (2)
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int qpt = 0; qpt < eles->nQpts; qpt++)
      {

        double U_true = 0.0;
        double weight = 0.0;

        if (eles->nDims == 2)
        {
          /* Compute true solution */
          U_true = compute_U_true(geo.coord_qpts(qpt,ele,0), geo.coord_qpts(qpt,ele,1), 
              flow_time, n, input);

          /* Get quadrature point index and weight */
          unsigned int i = eles->idx_qpts(qpt,0);
          unsigned int j = eles->idx_qpts(qpt,1);
          weight = eles->weights_qpts[i] * eles->weights_qpts[j];
        }
        else if (eles->nDims == 3)
        {
          ThrowException("Under construction!");
        }

        /* Compute error */
        double error = U_true - eles->U_qpts(qpt, ele, n);

        l2_error[n] += weight * eles->jaco_det_qpts(qpt, ele) * error * error; 
      }
    }
  }

  std::cout << "l2_error: ";
  for (auto &val : l2_error)
    std::cout << std::scientific << std::setprecision(12) << std::sqrt(val) << " ";
  std::cout << std::endl;
}
