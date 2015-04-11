#include <cblas.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "mdvector.hpp"
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
  geo = process_mesh(input->meshfile, order, input->nDims);

  eles = std::make_shared<Quads>(&geo, input, order);
  faces = std::make_shared<Faces>(&geo, input);

  eles->associate_faces(faces);
  eles->setup();

  initialize_U();
  setup_update();
  setup_output();
}

void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  U_ini.assign({eles->nVars, eles->nSpts, eles->nEles});
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
  else
  {
    ThrowException("dt_scheme not recognized!");
  }

  divF.assign({nStages, eles->nVars, eles->nSpts, eles->nEles});

}

void FRSolver::setup_output()
{
  if (eles->nDims == 2)
  {
    unsigned int nSubelements1D = eles->nSpts1D+1;
    eles->nSubelements = nSubelements1D * nSubelements1D;
    eles->nNodesPerSubelement = 4;

    /* Allocate memory for local plot point connectivity and solution at plot points */
    geo.ppt_connect.assign({eles->nSubelements, 4});
    eles->U_ppts.assign({eles->nVars, eles->nPpts, eles->nEles});

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
          geo.ppt_connect(ele,node) = nd[node] + j;
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

void FRSolver::compute_residual(unsigned int stage)
{
  extrapolate_U();

  U_to_faces();

  eles->compute_Fconv();
  faces->compute_Fconv();
  faces->compute_common_F();
  F_from_faces();

  eles->transform_flux();

  compute_dF();
  compute_divF(stage);
}

void FRSolver::initialize_U()
{
  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  eles->U_spts.assign({eles->nVars, eles->nSpts, eles->nEles});
  eles->U_fpts.assign({eles->nVars, eles->nFpts, eles->nEles});
  eles->U_ppts.assign({eles->nVars, eles->nPpts, eles->nEles});

  eles->F_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});
  eles->F_fpts.assign({eles->nDims, eles->nVars, eles->nFpts, eles->nEles});
  eles->commF.assign({eles->nVars, eles->nFpts, eles->nEles});

  eles->dU_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});
  eles->dF_spts.assign({eles->nDims, eles->nVars, eles->nSpts, eles->nEles});

  eles->divF_spts.assign({eles->nVars, eles->nSpts, eles->nEles});

  /* Initialize solution */
  // TODO: Fill in with actual logic. */
  for (unsigned int ele = 0; ele < eles->nEles; ele++)
  {
    for (unsigned int spt = 0; spt < eles->nSpts; spt++)
    {
      double x = geo.coord_spts(0,spt,ele);
      double y = geo.coord_spts(1,spt,ele);

      eles->U_spts(0,spt,ele) = std::exp(-20. * (x*x + y*y));

      /*
      if (ele == 4)
      {
        eles->U_spts(0,spt,ele) = 1.0;
      }
      */
     
    }
  }
}

void FRSolver::extrapolate_U()
{
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    auto &A = eles->oppE(0,0);
    auto &B = eles->U_spts(n,0,0);
    auto &C = eles->U_fpts(n,0,0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, eles->nEles,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
  }

 
  /*
  std::cout << "Uext" << std::endl;
  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->U_fpts(0,i,j) << " ";
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
        int gfpt = geo.fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(ele,fpt);

        faces->U(n, gfpt, slot) = eles->U_fpts(n, fpt, ele);
        //std::cout << gfpt << " " << slot << " " << faces->U(n,gfpt,slot) << std::endl;
      }
    }
  }
}

void FRSolver::F_from_faces()
{
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        unsigned int gfpt = geo.fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        unsigned int slot = geo.fpt2gfpt_slot(ele,fpt);

        eles->commF(n, fpt, ele) = faces->Fcomm(n, gfpt, slot);
      }
    }
  }

  /*
  std::cout << "commF" << std::endl;
  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->commF(0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */
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
          eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
    }
  }

  /* Compute contribution to derivative from common flux at flux points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    for (unsigned int n = 0; n < eles->nVars; n++)
    {
      auto &A = eles->oppD_fpts(dim,0,0);
      auto &B = eles->commF(n,0,0);
      auto &C = eles->dF_spts(dim,n,0,0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles,
          eles->nFpts, 1.0, &A, eles->nFpts, &B, eles->nEles, 1.0, &C, eles->nEles);
    }
  }

  /*
  std::cout << "dF" << std::endl;
  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->dF_spts(0,0,i,j) << " ";
    }
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->dF_spts(1,0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */
  
 
}

void FRSolver::compute_divF(unsigned int stage)
{
  for (unsigned int n = 0; n < eles->nVars; n++)
    for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      for (unsigned int ele =0; ele < eles->nEles; ele++)
        divF(stage, n, spt, ele) = eles->dF_spts(0, n, spt, ele);
        //eles->divF_spts(n,spt,ele) = eles->dF_spts(0,n,spt,ele);

  for (unsigned int dim = 1; dim < eles->nDims; dim ++)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        for (unsigned int ele =0; ele < eles->nEles; ele++)
          divF(stage, n, spt, ele) += eles->dF_spts(dim, n, spt, ele);
          //eles->divF_spts(n,spt,ele) += eles->dF_spts(dim,n,spt,ele);
          //
  for (unsigned int dim = 1; dim < eles->nDims; dim ++)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        for (unsigned int ele =0; ele < eles->nEles; ele++)
          divF(stage, n, spt, ele) /= eles->jaco_det_spts(ele, spt);
          //eles->divF_spts(n,spt,ele) += eles->dF_spts(dim,n,spt,ele);

  /*
  std::cout << "divF" << std::endl;
  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << divF(stage,0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */
 
}

void FRSolver::update()
{
  U_ini = eles->U_spts;

  /* Loop over stages to get intermediate residuals. (Inactive for Euler) */
  for (unsigned int stage = 0; stage < (nStages-1); stage++)
  {
    compute_residual(stage);
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
          eles->U_spts(n, spt, ele) = U_ini(n, spt, ele) - rk_alpha[stage] * input->dt * divF(stage, n, spt, ele);
  }

  /* Final stage */
  compute_residual(nStages-1);
  eles->U_spts = U_ini;

  for (unsigned int stage = 0; stage < nStages; stage++)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        for (unsigned int ele = 0; ele < eles->nEles; ele++)
          eles->U_spts(n, spt, ele) -=  rk_beta[stage] * input->dt * divF(stage, n, spt, ele);

  /*
  std::cout << "U" << std::endl;
  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->U_spts(0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */
 
}

void FRSolver::write_solution(std::string outputfile)
{
  /* Write solution to file in .vtk format */
  std::ofstream f(outputfile);

  /* Write header */
  f << "# vtk DataFile Version 3.0" << std::endl;
  f << "vtk output" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET UNSTRUCTURED_GRID" << std::endl;
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
        f << geo.coord_ppts(0, ppt, ele) << " ";
        f << geo.coord_ppts(1, ppt, ele) << " ";
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
        f << geo.ppt_connect(subele,i) + ele*eles->nPpts << " ";
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
  /* Extrapolate solution to plot points */
  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    auto &A = eles->oppE_plot(0,0);
    auto &B = eles->U_spts(n,0,0);
    auto &C = eles->U_ppts(n,0,0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nPpts, eles->nEles,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
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
        f << eles->U_ppts(0,ppt,ele) << " ";
      }
      f << std::endl;
    }

  }


}
