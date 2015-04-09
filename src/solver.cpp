#include <cblas.h>
#include <iostream>
#include <memory>

#include "elements.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "solver.hpp"

FRSolver::FRSolver(const InputStruct *input, unsigned int order)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;
}

void FRSolver::setup()
{
  
  // TODO: Need to process geometry here 

  eles = std::make_shared<Quads>(1, 1, input, order);
  faces = std::make_shared<Faces>(8, input);

  eles->associate_faces(faces);
  eles->setup();

  initialize_U();

  extrapolate_U();

  U_to_faces();
  // TODO: Calls on faces to compute common flux
  F_from_faces();

  compute_dF();
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nVars, eles->nSpts, eles->nEles},1);
  eles->U_fpts.assign({eles->nVars, eles->nFpts, eles->nEles});

  eles->F_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});
  eles->F_fpts.assign({eles->nDims, eles->nVars, eles->nFpts, eles->nEles});

  eles->dU_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});
  eles->dF_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});

  eles->divF_spts.assign({eles->nVars, eles->nSpts, eles->nEles});

  /* Initialize solution */
  // TODO: Fill in with actual logic. */
}

void FRSolver::extrapolate_U()
{
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    auto &A = eles->oppE(0,0);
    auto &B = eles->U_spts(n,0,0);
    auto &C = eles->U_fpts(n,0,0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, eles->nEles,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 1.0, &C, eles->nEles);
  }

  /*
  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->U_fpts(0,i,j);
    }
    std::cout << std::endl;
  }
  */
}

void FRSolver::U_to_faces()
{
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        unsigned int gfpt = eles->fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        unsigned int slot = eles->fpt2gfpt_slot(ele,fpt);

        faces->U(n, gfpt, slot) = eles-> U_fpts(n, fpt, ele);
      }
    }
  }
}

void FRSolver::F_from_faces()
{
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    for (unsigned int n = 0; n < eles->nVars; n++) 
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
        {
          unsigned int gfpt = eles->fpt2gfpt(ele,fpt);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          unsigned int slot = eles->fpt2gfpt_slot(ele,fpt);

          eles->F_fpts(dim, n, fpt, ele) = faces->Fcomm(dim, n, gfpt, slot);
        }
      }
    }
  }

  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->F_fpts(0,0,i,j);
    }
    std::cout << std::endl;
  }
}

void FRSolver::compute_dF()
{
  /* Compute contribution to derivative from flux at solution points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppD(dim,0,0);
      auto &B = eles->F_spts(dim, n,0,0);
      auto &C = eles->dF_spts(dim,n,0,0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles,
          eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 1.0, &C, eles->nEles);
    }
  }

  /* Compute contribution to derivative from common flux at flux points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppD_fpts(dim,0,0);
      auto &B = eles->F_fpts(dim, n,0,0);
      auto &C = eles->dF_spts(dim,n,0,0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles,
          eles->nFpts, 1.0, &A, eles->nFpts, &B, eles->nEles, 1.0, &C, eles->nEles);
    }
  }

  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->dF_spts(0,0,i,j);
    }
    std::cout << std::endl;
  }
 
}
