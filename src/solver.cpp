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

  std::cout << "Setting up timestepping..." << std::endl;
  setup_update();

  std::cout << "Setting up output..." << std::endl;
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

  divF.assign({nStages, eles->nVars, eles->nEles, eles->nSpts});

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
  eles->U_spts.assign({eles->nVars, eles->nEles, eles->nSpts});
  eles->U_fpts.assign({eles->nVars, eles->nEles, eles->nFpts});
  eles->Ucomm.assign({eles->nVars, eles->nEles, eles->nFpts});
  eles->U_ppts.assign({eles->nVars, eles->nEles, eles->nPpts});
  eles->U_qpts.assign({eles->nVars, eles->nEles, eles->nQpts});

  eles->F_spts.assign({eles->nDims, eles->nVars, eles->nEles, eles->nSpts});
  eles->F_fpts.assign({eles->nDims, eles->nVars, eles->nEles, eles->nFpts});
  eles->Fcomm.assign({eles->nVars, eles->nEles, eles->nFpts});

  eles->dU_spts.assign({eles->nDims, eles->nVars, eles->nEles, eles->nSpts});
  eles->dU_fpts.assign({eles->nDims, eles->nVars, eles->nEles, eles->nFpts});
  eles->dF_spts.assign({eles->nDims, eles->nVars, eles->nEles, eles->nSpts});

  eles->divF_spts.assign({eles->nVars, eles->nEles, eles->nSpts});

  /* Initialize solution */
  // TODO: Fill in with actual logic. */
  if (input->equation == "AdvDiff")
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      {
        double x = geo.coord_spts(0,ele,spt);
        double y = geo.coord_spts(1,ele,spt);

        /*
        if (input->ic_type == 0)
          eles->U_spts(0,spt,ele) = std::exp(-20. * (x*x + y*y));
        else if (input->ic_type == 1)
          eles->U_spts(0,spt,ele) = std::sin(M_PI*x)*sin(M_PI*y);
        else
          ThrowException("ic_type not recognized!");
        */

        eles->U_spts(0,ele,spt) = compute_U_true(x, y, 0, 0, input);
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
  //for (unsigned int n = 0; n < eles->nVars; n++)
  //{
    auto &A = eles->U_spts(0,0,0);
    auto &B = eles->oppE(0,0);
    auto &C = eles->U_fpts(0,0,0);

    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, eles->nEles,
    //    eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nFpts,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nFpts, 0.0, &C, eles->nFpts);
  //}

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
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++) 
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < eles->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
        {
          if (input->viscous) // if viscous, put extrapolated solution into Ucomm
            eles->Ucomm(n, ele, fpt) = eles->U_fpts(n, ele, fpt);
          continue;
        }
        int slot = geo.fpt2gfpt_slot(ele,fpt);

        faces->U(n, gfpt, slot) = eles->U_fpts(n, ele, fpt);
        //std::cout << gfpt << " " << slot << " " << faces->U(n,gfpt,slot) << std::endl;
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
        int gfpt = geo.fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(ele,fpt);

        eles->Ucomm(n, ele, fpt) = faces->Ucomm(n, gfpt, slot);
      }
    }
  }

  /*
  std::cout << "Ucomm" << std::endl;
  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->Ucomm(0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */

}

void FRSolver::compute_dU()
{
  /* Compute contribution to derivative from solution at solution points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    //for (unsigned int n = 0; n < eles->nVars; n++)
    //{
      auto &A = eles->U_spts(0,0,0);
      auto &B = eles->oppD(dim,0,0);
      auto &C = eles->dU_spts(dim,0,0,0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nSpts,
          eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->nSpts);
    //}
  }

  /* Compute contribution to derivative from common solution at flux points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    //for (unsigned int n = 0; n < eles->nVars; n++)
    //{
      auto &A = eles->Ucomm(0,0,0);
      auto &B = eles->oppD_fpts(dim,0,0);
      auto &C = eles->dU_spts(dim,0,0,0);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nSpts,
          eles->nFpts, 1.0, &A, eles->nFpts, &B, eles->nSpts, 1.0, &C, eles->nSpts);
    //}
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
          double dUtemp = eles->dU_spts(0, n, ele, spt);
          
          eles->dU_spts(0, n, ele, spt) = eles->dU_spts(0, n, ele, spt) * eles->jaco_spts(ele, spt, 1, 1)-
                                    eles->dU_spts(1, n, ele, spt) * eles->jaco_spts(ele, spt, 1, 0); 
          eles->dU_spts(1, n, ele, spt) = -dUtemp * eles->jaco_spts(ele, spt, 0, 1) +
                                    eles->dU_spts(1, n, ele, spt) * eles->jaco_spts(ele, spt, 0, 0); 

          eles->dU_spts(0, n, ele, spt) /= eles->jaco_det_spts(ele,spt);
          eles->dU_spts(1, n, ele, spt) /= eles->jaco_det_spts(ele,spt);
        }
      }
    }
  }

  /*
  std::cout << "dU" << std::endl;
  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->dU_spts(0,0,i,j) << " ";
    }
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < eles->nSpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->dU_spts(1,0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */

}

void FRSolver::extrapolate_dU()
{
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    //for (unsigned int n = 0; n < eles->nVars; n++)
    //{
      auto &A = eles->dU_spts(dim,0,0,0);
      auto &B = eles->oppE(0,0);
      auto &C = eles->dU_fpts(dim,0,0,0);

      //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nFpts, eles->nEles,
      //    eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nFpts,
          eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nFpts, 0.0, &C, eles->nFpts);
    //}
  }

 /* 
  std::cout << "dUext" << std::endl;

  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    for (unsigned int i = 0; i < eles->nFpts; i++)
    {
      for (unsigned int j = 0; j < eles->nEles; j++)
      {
        std::cout << eles->dU_fpts(dim, 0,i,j) << " ";
      }
      std::cout << std::endl;
    }
  }
*/ 
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
          int gfpt = geo.fpt2gfpt(ele,fpt);
          /* Check if flux point is on ghost edge */
          if (gfpt == -1)
            continue;
          int slot = geo.fpt2gfpt_slot(ele,fpt);

          faces->dU(dim, n, gfpt, slot) = eles->dU_fpts(dim, n, ele, fpt);
          //std::cout << gfpt << " " << slot << " " << faces->U(n,gfpt,slot) << std::endl;
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
        int gfpt = geo.fpt2gfpt(ele,fpt);
        /* Check if flux point is on ghost edge */
        if (gfpt == -1)
          continue;
        int slot = geo.fpt2gfpt_slot(ele,fpt);

        eles->Fcomm(n, ele, fpt) = faces->Fcomm(n, gfpt, slot);
      }
    }
  }

  /*
  std::cout << "Fcomm" << std::endl;
  for (unsigned int i = 0; i < eles->nFpts; i++)
  {
    for (unsigned int j = 0; j < eles->nEles; j++)
    {
      std::cout << eles->Fcomm(0,i,j) << " ";
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
    //for (unsigned int n = 0; n < eles->nVars; n++)
    //{
      auto &A = eles->F_spts(dim, 0,0,0);
      auto &B = eles->oppD(dim,0,0);
      auto &C = eles->dF_spts(dim,0,0,0);

      //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles,
      //    eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nEles, 0.0, &C, eles->nEles);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nSpts,
          eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nSpts, 0.0, &C, eles->nSpts);
    //}
  }

  /* Compute contribution to derivative from common flux at flux points */
  for (unsigned int dim = 0; dim < eles->nDims; dim++)
  {
    //for (unsigned int n = 0; n < eles->nVars; n++)
    //{
      auto &A = eles->Fcomm(0,0,0);
      auto &B = eles->oppD_fpts(dim,0,0);
      auto &C = eles->dF_spts(dim,0,0,0);

      //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nSpts, eles->nEles,
      //    eles->nFpts, 1.0, &A, eles->nFpts, &B, eles->nEles, 1.0, &C, eles->nEles);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nSpts,
          eles->nFpts, 1.0, &A, eles->nFpts, &B, eles->nSpts, 1.0, &C, eles->nSpts);
    //}
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
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < eles->nVars; n++)
    for (unsigned int ele =0; ele < eles->nEles; ele++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        divF(stage, n, ele, spt) = eles->dF_spts(0, n, ele, spt);
        //eles->divF_spts(n,spt,ele) = eles->dF_spts(0,n,spt,ele);

  for (unsigned int dim = 1; dim < eles->nDims; dim ++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele =0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          divF(stage, n, ele, spt) += eles->dF_spts(dim, n, ele, spt);
          //eles->divF_spts(n,spt,ele) += eles->dF_spts(dim,n,spt,ele);
          //
  for (unsigned int dim = 1; dim < eles->nDims; dim ++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele =0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          divF(stage, n, ele, spt) /= eles->jaco_det_spts(ele, spt);
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
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(n, ele, spt) = U_ini(n, ele, spt) - rk_alpha[stage] * input->dt * divF(stage, n, ele, spt);
  }

  /* Final stage */
  compute_residual(nStages-1);
  eles->U_spts = U_ini;

  for (unsigned int stage = 0; stage < nStages; stage++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < eles->nVars; n++)
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
        for (unsigned int spt = 0; spt < eles->nSpts; spt++)
          eles->U_spts(n, ele, spt) -=  rk_beta[stage] * input->dt * divF(stage, n, ele, spt);

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

  flow_time += input->dt;
 
}

void FRSolver::write_solution(std::string prefix, unsigned int nIter)
{
  std::stringstream ss;
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << nIter << ".vtk";

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
  f << nIter << std::endl;
  f << std::endl;
  
  /* Write plot point coordinates */
  f << "POINTS " << eles->nPpts*eles->nEles << " double" << std::endl;
  if (eles->nDims == 2)
  {
    // TODO: Change order of ppt structures for better looping 
    for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
    {
      for (unsigned int ele = 0; ele < eles->nEles; ele++)
      {
        f << geo.coord_ppts(0, ele, ppt) << " ";
        f << geo.coord_ppts(1, ele, ppt) << " ";
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
  //for (unsigned int n = 0; n < eles->nVars; n++)
  //{
    auto &A = eles->U_spts(0,0,0);
    auto &B = eles->oppE_ppts(0,0);
    auto &C = eles->U_ppts(0,0,0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nPpts,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nPpts, 0.0, &C, eles->nPpts);
  //}

  if (input->equation == "AdvDiff")
  {
    f << "POINT_DATA " << eles->nPpts*eles->nEles << std::endl;
    f << "SCALARS U double 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int ppt = 0; ppt < eles->nPpts; ppt++)
      {
        f << eles->U_ppts(0,ele,ppt) << " ";
      }
      f << std::endl;
    }

  }
}

void FRSolver::report_max_residuals()
{
  std::vector<double> max_res(eles->nVars,0.0);

  for (unsigned int n = 0; n < eles->nVars; n++)
    max_res[n] = *std::max_element(&divF(nStages-1, n, 0, 0), &divF(nStages-1, n, eles->nEles-1, eles->nSpts-1));

  for (auto &val : max_res)
    std::cout << std::scientific << val << " ";

  std::cout << std::endl;
}

void FRSolver::compute_l2_error()
{
  /* Extrapolate solution to quadrature points */
  //for (unsigned int n = 0; n < eles->nVars; n++)
  //{
    auto &A = eles->U_spts(0,0,0);
    auto &B = eles->oppE_qpts(0,0);
    auto &C = eles->U_qpts(0,0,0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, eles->nVars * eles->nEles, eles->nQpts,
        eles->nSpts, 1.0, &A, eles->nSpts, &B, eles->nQpts, 0.0, &C, eles->nQpts);
  //}


  std::vector<double> l2_error(eles->nVars,0.0);

  for (unsigned int n = 0; n < eles->nVars; n++)
  {
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      for (unsigned int qpt = 0; qpt < eles->nQpts; qpt++)
      {

        double U_true = 0.0;
        double weight = 0.0;

        if (eles->nDims == 2)
        {
          /* Compute true solution */
          U_true = compute_U_true(geo.coord_qpts(0, ele, qpt), geo.coord_qpts(1, ele, qpt) , 
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
        double error = U_true - eles->U_qpts(n,ele,qpt);

        l2_error[n] += weight * eles->jaco_det_qpts(ele, qpt) * error * error; 
      }
    }
  }

  std::cout << "l2_error: ";
  for (auto &val : l2_error)
    std::cout << std::scientific << std::setprecision(12) << std::sqrt(val) << " ";
  std::cout << std::endl;
}
