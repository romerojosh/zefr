#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <cblas.h>

#include "elements.hpp"
#include "faces.hpp"
#include "funcs.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

void Elements::setup(std::shared_ptr<Faces> faces)
{
  set_locs();
  set_shape();
  set_transforms(faces);
  set_normals(faces);
  setup_FR();
  setup_aux();
  set_coords(faces);

  /* If P-multigrid is enabled, create associated operators */
  if (input->p_multi)
    setup_PMG();
}

void Elements::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nNodes, nSpts},1);
  shape_fpts.assign({nNodes, nFpts},1);
  shape_ppts.assign({nNodes, nPpts},1);
  shape_qpts.assign({nNodes, nQpts},1);
  dshape_spts.assign({nNodes, nSpts, nDims},1);
  dshape_fpts.assign({nNodes, nFpts, nDims},1);
  dshape_ppts.assign({nNodes, nPpts, nDims},1);
  dshape_qpts.assign({nNodes, nQpts, nDims},1);


  std::vector<double> loc(nDims,0.0);

  /* Shape functions and derivatives at solution points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_spts(spt,dim);


    auto shape_val = calc_shape(shape_order, loc);
    auto dshape_val = calc_d_shape(shape_order, loc);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(node,spt) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_spts(node,spt,dim) = dshape_val(node, dim);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_fpts(fpt,dim);

    auto shape_val = calc_shape(shape_order, loc);
    auto dshape_val = calc_d_shape(shape_order, loc);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(node, fpt) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_fpts(node, fpt, dim) = dshape_val(node, dim);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_ppts(ppt,dim);

    auto shape_val = calc_shape(shape_order, loc);
    auto dshape_val = calc_d_shape(shape_order, loc);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(node, ppt) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_ppts(node, ppt, dim) = dshape_val(node, dim);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    auto shape_val = calc_shape(shape_order, loc);
    auto dshape_val = calc_d_shape(shape_order, loc);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(node, qpt) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_qpts(node, qpt, dim) = dshape_val(node, dim);
    }
  }
}

void Elements::set_coords(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for physical coordinates */
  geo->coord_spts.assign({nSpts, nEles, nDims});
  geo->coord_fpts.assign({nFpts, nEles, nDims});
  faces->coord.assign({geo->nGfpts, nDims});
  geo->coord_ppts.assign({nPpts, nEles, nDims});
  geo->coord_qpts.assign({nQpts, nEles, nDims});


  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {

      /* Setup physical coordinates at solution points */
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_spts(spt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_spts(node, spt);
        }
      }
  
      /* Setup physical coordinates at flux points */
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_fpts(fpt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_fpts(node, fpt);

          int gfpt = geo->fpt2gfpt(fpt,ele);

          /* Check if on ghost edge */
          if (gfpt != -1)
            faces->coord(gfpt, dim) += geo->coord_nodes(gnd,dim) * shape_fpts(node, fpt);

        }
      }

          /* Setup physical coordinates at plot points */
      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_ppts(ppt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_ppts(node, ppt);
        }
      }

      /* Setup physical coordinates at quadrature points */
      for (unsigned int qpt = 0; qpt < nQpts; qpt++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(node, ele);
          geo->coord_qpts(qpt, ele, dim) += geo->coord_nodes(gnd,dim) * shape_qpts(node, qpt);
        }
      }

    }
  }
}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppD.assign({nSpts, nSpts, nDims});
  oppD_fpts.assign({nSpts, nFpts, nDims});
  oppDiv_fpts.assign({nSpts, nFpts});

  std::vector<double> loc(nDims, 0.0);
  /* Setup spt to fpt extrapolation operator (oppE) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        loc[dim] = loc_fpts(fpt , dim);

      oppE(fpt,spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Setup differentiation operator (oppD) for solution points */
  /* Note: Can set up for standard FR eventually. Trying to keep things simple.. */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int jspt = 0; jspt < nSpts; jspt++)
    {
      for (unsigned int ispt = 0; ispt < nSpts; ispt++)
      {
        for (unsigned int d = 0; d < nDims; d++)
          loc[d] = loc_spts(ispt , d);

        oppD(ispt,jspt,dim) = calc_d_nodal_basis_spts(jspt, loc, dim);
      }
    }
  }

  /* Setup differentiation operator (oppD_fpts) for flux points (DFR Specific)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int d = 0; d < nDims; d++)
          loc[d] = loc_spts(spt , d);

        oppD_fpts(spt,fpt,dim) = calc_d_nodal_basis_fpts(fpt, loc, dim);
      }
    }
  }

  /* Setup divergence operator (oppDiv_fpts) for flux points by combining dimensions of oppD_fpts */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {

      /* Set positive parent sign convention into operator based on face */
      int fac = 1;
      if (nDims == 2) 
      {
        int face = fpt / nSpts1D;
        if (face == 0 or face == 3) // Bottom and Right face
          fac = -1;
      }
      else if (nDims == 3)
      {
        int face = fpt / (nSpts1D * nSpts1D);
        if (face % 2 == 0) // Bottom, Left, and Front face
          fac = -1;
      }

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        oppDiv_fpts(spt, fpt) += fac * oppD_fpts(spt, fpt, dim);
      }
    }
  }

}

void Elements::setup_aux()
{
  /* Allocate memory for plot point and quadrature point interpolation operator */
  oppE_ppts.assign({nPpts, nSpts});
  oppE_qpts.assign({nQpts, nSpts});

  std::vector<double> loc(nDims, 0.0);

  /* Setup spt to ppt extrapolation operator (oppE_ppts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        loc[dim] = loc_ppts(ppt , dim);

      oppE_ppts(ppt, spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Setup spt to qpt extrapolation operator (oppE_qpts) */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        loc[dim] = loc_qpts(qpt , dim);

      oppE_qpts(qpt,spt) = calc_nodal_basis(spt, loc);
    }
  }

}

void Elements::extrapolate_U(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto &A = oppE(0,0);
    auto &B = U_spts(0, startEle, var);
    auto &C = U_fpts(0, startEle, var);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
          nSpts, 1.0, &A, oppE.ldim(), &B, U_spts.ldim(), 0.0, &C, U_fpts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
          nSpts, 1.0, &A, oppE.ldim(), &B, U_spts.ldim(), 0.0, &C, U_fpts.ldim());
#endif
  }

#endif

#ifdef _GPU
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto *A = oppE_d.data();
    auto *B = U_spts_d.data() + startEle * U_spts_d.ldim() + var * (U_spts_d.ldim() * nEles);
    auto *C = U_fpts_d.data() + startEle * U_fpts_d.ldim() + var * (U_fpts_d.ldim() * nEles);
    cublasDGEMM_wrapper(nFpts, endEle - startEle, nSpts, 1.0,
        A, oppE_d.ldim(), B, U_spts_d.ldim(), 0.0, C, U_fpts_d.ldim());
  }

  check_error();
#endif

}

void Elements::extrapolate_dU(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      auto &A = oppE(0,0);
      auto &B = dU_spts(0, startEle, var, dim);
      auto &C = dU_fpts(0, startEle, var, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, 
          endEle - startEle, nSpts, 1.0, &A, oppE.ldim(), &B, dU_spts.ldim(), 0.0, &C, dU_fpts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
          nSpts, 1.0, &A, oppE.ldim(), &B, dU_spts.ldim(), 0.0, &C, dU_fpts.ldim());
#endif
    }
  }
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    cublasDGEMM_wrapper(nFpts, nEles * nVars, nSpts, 1.0, 
        oppE_d.data(), oppE_d.ldim(), dU_spts_d.data() + dim * (dU_spts_d.ldim() * 
        nVars * nEles), dU_spts_d.ldim(), 0.0, dU_fpts_d.data() + dim * 
        (dU_fpts_d.ldim() * nVars * nEles), dU_fpts_d.ldim());
  }
#endif
}

void Elements::compute_dU(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  /* Compute contribution to derivative from solution at solution points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto &A = oppD(0, 0, dim);
        auto &B = U_spts(0, startEle, var);
        auto &C = dU_spts(0, startEle, var, dim);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
            endEle - startEle, nSpts, 1.0, &A, oppD.ldim(), &B, U_spts.ldim(), 
            0.0, &C, dU_spts.ldim());
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
            endEle - startEle, nSpts, 1.0, &A, oppD.ldim(), &B, U_spts.ldim(), 
            0.0, &C, dU_spts.ldim());
#endif
      }
    }

    /* Compute contribution to derivative from common solution at flux points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto &A = oppD_fpts(0, 0, dim);
        auto &B = Ucomm(0, startEle, var);
        auto &C = dU_spts(0, startEle, var, dim);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
            endEle - startEle, nFpts, 1.0, &A, oppD_fpts.ldim(), &B, Ucomm.ldim(), 
            1.0, &C, dU_spts.ldim());
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
            endEle - startEle, nFpts, 1.0, &A, oppD_fpts.ldim(), &B, Ucomm.ldim(), 
            1.0, &C, dU_spts.ldim());
#endif
      }
    }

#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    /* Compute contribution to derivative from solution at solution points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nSpts, 1.0,
        oppD_d.data() + dim * (oppD_d.ldim() * nSpts), oppD_d.ldim(), 
        U_spts_d.data(), U_spts_d.ldim(), 0.0, dU_spts_d.data() + dim * 
        (dU_spts_d.ldim() * nVars * nEles), dU_spts_d.ldim());

    check_error();

    /* Compute contribution to derivative from common solution at flux points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nFpts, 1.0,
        oppD_fpts_d.data() + dim * (oppD_fpts_d.ldim() * nFpts), oppD_fpts_d.ldim(),
        Ucomm_d.data(), Ucomm_d.ldim(), 1.0, dU_spts_d.data() + dim * 
        (dU_spts_d.ldim() * nVars * nEles), dU_spts_d.ldim());

    check_error();
  }
#endif

}

void Elements::compute_divF(unsigned int stage, unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  /* Compute contribution to divergence from flux at solution points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    double fac = (dim == 0) ? 0.0 : 1.0;

    for (unsigned int var = 0; var < nVars; var++)
    {
      auto &A = oppD(0, 0, dim);
      auto &B = F_spts(0, startEle, var, dim);
      auto &C = divF_spts(0, startEle, var, stage);


#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          endEle - startEle, nSpts, 1.0, &A, oppD.ldim(), &B, F_spts.ldim(), fac, &C, divF_spts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, endEle - startEle,
            nSpts, 1.0, &A, oppD.ldim(), &B, F_spts.ldim(), fac, &C, divF_spts.ldim());
#endif
    }
  }

  /* Compute contribution to divergence from common flux at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto &A = oppDiv_fpts(0, 0);
    auto &B = Fcomm(0, startEle, var);
    auto &C = divF_spts(0, startEle, var, stage);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
        endEle - startEle, nFpts, 1.0, &A, oppDiv_fpts.ldim(), &B, Fcomm.ldim(), 1.0, &C, divF_spts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, endEle - startEle,
        nFpts, 1.0, &A, oppDiv_fpts.ldim(), &B, Fcomm.ldim(), 1.0, &C, divF_spts.ldim());
#endif
  }
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    double fac = (dim == 0) ? 0.0 : 1.0;

    for (unsigned int var = 0; var < nVars; var++)
    {
      auto *A = oppD_d.data() + dim * (oppD_d.ldim() * nSpts);
      auto *B = F_spts_d.data() + startEle * F_spts_d.ldim() + var * (F_spts_d.ldim() * nEles) + dim * (F_spts_d.ldim() * nEles * nVars);
      auto *C = divF_spts_d.data() + startEle * divF_spts_d.ldim() + var * (divF_spts_d.ldim() * nEles) + stage * (divF_spts_d.ldim() * nEles * nVars);

      /* Compute contribution to derivative from solution at solution points */
      cublasDGEMM_wrapper(nSpts, endEle - startEle, nSpts, 1.0,
          A, oppD_d.ldim(), B, F_spts_d.ldim(), fac, C, divF_spts_d.ldim());
    }
  }

  /* Compute contribution to derivative from common solution at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto *A = oppDiv_fpts_d.data();
    auto *B = Fcomm_d.data() + startEle * Fcomm_d.ldim() + var * (Fcomm_d.ldim() * nEles);
    auto *C = divF_spts_d.data() + startEle * divF_spts_d.ldim() + var * (divF_spts_d.ldim() * nEles) + stage * (divF_spts_d.ldim() * nEles * nVars);

    cublasDGEMM_wrapper(nSpts, endEle - startEle,  nFpts, 1.0,
        A, oppDiv_fpts_d.ldim(), B, Fcomm_d.ldim(), 1.0, C, divF_spts_d.ldim());
  }

  check_error();
#endif
}


void Elements::compute_Fconv(unsigned int startEle, unsigned int endEle)
{
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = startEle; ele < endEle; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            F_spts(spt, ele, n, dim) = input->AdvDiff_A(dim) * U_spts(spt, ele, n);
          }
        }
      }
    }
#endif

#ifdef _GPU
    compute_Fconv_spts_AdvDiff_wrapper(F_spts_d, U_spts_d, nSpts, nEles, nDims, input->AdvDiff_A_d, startEle, endEle);
    check_error();
#endif

  }

  else if (input->equation == Burgers)
  {
#ifdef _CPU
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = startEle; ele < endEle; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            F_spts(spt, ele, n, dim) = 0.5 * U_spts(spt, ele, n) * U_spts(spt, ele, n);
          }
        }
      }
    }
#endif

#ifdef _GPU
    compute_Fconv_spts_Burgers_wrapper(F_spts_d, U_spts_d, nSpts, nEles, nDims, startEle, endEle);
    check_error();
#endif

  }

  else if (input->equation == EulerNS)
  {
#ifdef _CPU
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = startEle; ele < endEle; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Compute some primitive variables */
          double momF = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim ++)
          {
            momF += U_spts(spt, ele, dim + 1) * U_spts(spt, ele, dim + 1);
          }

          momF /= U_spts(spt, ele, 0);

          double P = (input->gamma - 1.0) * (U_spts(spt, ele, 3) - 0.5 * momF);
          double H = (U_spts(spt, ele, 3) + P) / U_spts(spt, ele, 0);


          F_spts(spt, ele, 0, 0) = U_spts(spt, ele, 1);
          F_spts(spt, ele, 1, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 2, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 3, 0) = U_spts(spt, ele, 1) * H;

          F_spts(spt, ele, 0, 1) = U_spts(spt, ele, 2);
          F_spts(spt, ele, 1, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 2, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 3, 1) = U_spts(spt, ele, 2) * H;
        }
      }
    }
    else if (nDims == 3)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = startEle; ele < endEle; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Compute some primitive variables */
          double momF = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim ++)
          {
            momF += U_spts(spt, ele, dim + 1) * U_spts(spt, ele, dim + 1);
          }

          momF /= U_spts(spt, ele, 0);

          double P = (input->gamma - 1.0) * (U_spts(spt, ele, 4) - 0.5 * momF);
          double H = (U_spts(spt, ele, 4) + P) / U_spts(spt, ele, 0);


          F_spts(spt, ele, 0, 0) = U_spts(spt, ele, 1);
          F_spts(spt, ele, 1, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 2, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 3, 0) = U_spts(spt, ele, 1) * U_spts(spt, ele, 3) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 4, 0) = U_spts(spt, ele, 1) * H;

          F_spts(spt, ele, 0, 1) = U_spts(spt, ele, 2);
          F_spts(spt, ele, 1, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 2, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 3, 1) = U_spts(spt, ele, 2) * U_spts(spt, ele, 3) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 4, 1) = U_spts(spt, ele, 2) * H;

          F_spts(spt, ele, 0, 2) = U_spts(spt, ele, 3);
          F_spts(spt, ele, 1, 2) = U_spts(spt, ele, 3) * U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 2, 2) = U_spts(spt, ele, 3) * U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          F_spts(spt, ele, 3, 2) = U_spts(spt, ele, 3) * U_spts(spt, ele, 3) / U_spts(spt, ele, 0) + P;
          F_spts(spt, ele, 4, 2) = U_spts(spt, ele, 3) * H;
        }
      }
    }
#endif

#ifdef _GPU
    compute_Fconv_spts_EulerNS_wrapper(F_spts_d, U_spts_d, nSpts, nEles, nDims, input->gamma, startEle, endEle);
    check_error();
#endif

  }
}

void Elements::compute_Fvisc(unsigned int startEle, unsigned int endEle)
{
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
#ifdef _CPU
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = startEle; ele < endEle; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            /* Can just add viscous flux to existing convective flux */
            F_spts(spt, ele, n, dim) += -input->AdvDiff_D * dU_spts(spt, ele, n, dim);
          }
        }
      }
    }
#endif

#ifdef _GPU
    compute_Fvisc_spts_AdvDiff_wrapper(F_spts_d, dU_spts_d, nSpts, nEles, nDims, input->AdvDiff_D);
    check_error();
#endif

  }
  else if (input->equation == EulerNS)
  {
#ifdef _CPU
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = U_spts(spt, ele, 0);
          double momx = U_spts(spt, ele, 1);
          double momy = U_spts(spt, ele, 2);
          double e = U_spts(spt, ele, 3);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = dU_spts(spt, ele, 0, 0);
          double momx_dx = dU_spts(spt, ele, 1, 0);
          double momy_dx = dU_spts(spt, ele, 2, 0);
          double e_dx = dU_spts(spt, ele, 3, 0);
          
          double rho_dy = dU_spts(spt, ele, 0, 1);
          double momx_dy = dU_spts(spt, ele, 1, 1);
          double momy_dy = dU_spts(spt, ele, 2, 1);
          double e_dy = dU_spts(spt, ele, 3, 1);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double dke_dx = 0.5 * (u*u + v*v) * rho_dx + rho * (u * du_dx + v * dv_dx);
          double dke_dy = 0.5 * (u*u + v*v) * rho_dy + rho * (u * du_dy + v * dv_dy);

          double de_dx = (e_dx - dke_dx - rho_dx * e_int) / rho;
          double de_dy = (e_dy - dke_dy - rho_dy * e_int) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Set viscous flux values */
          F_spts(spt, ele, 1, 0) -= tauxx;
          F_spts(spt, ele, 2, 0) -= tauxy;
          F_spts(spt, ele, 3, 0) -= (u * tauxx + v * tauxy + (mu / input->prandtl) *
              input-> gamma * de_dx);

          F_spts(spt, ele, 1, 1) -= tauxy;
          F_spts(spt, ele, 2, 1) -= tauyy;
          F_spts(spt, ele, 3, 1) -= (u * tauxy + v * tauyy + (mu / input->prandtl) *
              input->gamma * de_dy);
        }
      }
    }
    else if (nDims == 3)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = U_spts(spt, ele, 0);
          double momx = U_spts(spt, ele, 1);
          double momy = U_spts(spt, ele, 2);
          double momz = U_spts(spt, ele, 3);
          double e = U_spts(spt, ele, 4);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

          /* Gradients */
          double rho_dx = dU_spts(spt, ele, 0, 0);
          double momx_dx = dU_spts(spt, ele, 1, 0);
          double momy_dx = dU_spts(spt, ele, 2, 0);
          double momz_dx = dU_spts(spt, ele, 3, 0);
          double e_dx = dU_spts(spt, ele, 4, 0);
          
          double rho_dy = dU_spts(spt, ele, 0, 1);
          double momx_dy = dU_spts(spt, ele, 1, 1);
          double momy_dy = dU_spts(spt, ele, 2, 1);
          double momz_dy = dU_spts(spt, ele, 3, 1);
          double e_dy = dU_spts(spt, ele, 4, 1);

          double rho_dz = dU_spts(spt, ele, 0, 2);
          double momx_dz = dU_spts(spt, ele, 1, 2);
          double momy_dz = dU_spts(spt, ele, 2, 2);
          double momz_dz = dU_spts(spt, ele, 3, 2);
          double e_dz = dU_spts(spt, ele, 4, 2);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio + input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;
          double du_dz = (momx_dz - rho_dz * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;
          double dv_dz = (momy_dz - rho_dz * v) / rho;

          double dw_dx = (momz_dx - rho_dx * w) / rho;
          double dw_dy = (momz_dy - rho_dy * w) / rho;
          double dw_dz = (momz_dz - rho_dz * w) / rho;

          double dke_dx = 0.5 * (u*u + v*v + w*w) * rho_dx + rho * (u * du_dx + v * dv_dx + w * dw_dx);
          double dke_dy = 0.5 * (u*u + v*v + w*w) * rho_dy + rho * (u * du_dy + v * dv_dy + w * dw_dy);
          double dke_dz = 0.5 * (u*u + v*v + w*w) * rho_dz + rho * (u * du_dz + v * dv_dz + w * dw_dz);

          double de_dx = (e_dx - dke_dx - rho_dx * e_int) / rho;
          double de_dy = (e_dy - dke_dy - rho_dy * e_int) / rho;
          double de_dz = (e_dz - dke_dz - rho_dz * e_int) / rho;

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Set viscous flux values */
          F_spts(spt, ele, 1, 0) -= tauxx;
          F_spts(spt, ele, 2, 0) -= tauxy;
          F_spts(spt, ele, 3, 0) -= tauxz;
          F_spts(spt, ele, 4, 0) -= (u * tauxx + v * tauxy + w * tauxz + (mu / input->prandtl) *
              input-> gamma * de_dx);

          F_spts(spt, ele, 1, 1) -= tauxy;
          F_spts(spt, ele, 2, 1) -= tauyy;
          F_spts(spt, ele, 3, 1) -= tauyz;
          F_spts(spt, ele, 4, 1) -= (u * tauxy + v * tauyy + w * tauyz + (mu / input->prandtl) *
              input->gamma * de_dy);

          F_spts(spt, ele, 1, 2) -= tauxz;
          F_spts(spt, ele, 2, 2) -= tauyz;
          F_spts(spt, ele, 3, 2) -= tauzz;
          F_spts(spt, ele, 4, 2) -= (u * tauxz + v * tauyz + w * tauzz + (mu / input->prandtl) *
              input->gamma * de_dz);
        }
      }

    }
#endif

#ifdef _GPU
      compute_Fvisc_spts_EulerNS_wrapper(F_spts_d, U_spts_d, dU_spts_d, nSpts, nEles, nDims, 
          input->gamma, input->prandtl, input->mu, input->c_sth, input->rt, input->fix_vis);
      check_error();
#endif

  }
}

void Elements::compute_dFdUconv()
{
  if (input->equation == AdvDiff)
  {
//#ifdef _CPU
    if (CPU_flag)
    {
#pragma omp parallel for collapse(5)
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            for (unsigned int ele = 0; ele < nEles; ele++)
            {
              for (unsigned int spt = 0; spt < nSpts; spt++)
              {
                dFdU_spts(spt, ele, ni, nj, dim) = input->AdvDiff_A(dim);
              }
            }
          }
        }
      }
    }
//#endif

#ifdef _GPU
    if (!CPU_flag)
    {
      compute_dFdUconv_spts_AdvDiff_wrapper(dFdU_spts_d, nSpts, nEles, nDims, input->AdvDiff_A_d);
      check_error();
    }
#endif

  }

  else if (input->equation == Burgers)
  {
//#ifdef _CPU
    if (CPU_flag)
    {
#pragma omp parallel for collapse(5)
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        for (unsigned int nj = 0; nj < nVars; nj++)
        {
          for (unsigned int ni = 0; ni < nVars; ni++)
          {
            for (unsigned int ele = 0; ele < nEles; ele++)
            {
              for (unsigned int spt = 0; spt < nSpts; spt++)
              {
                dFdU_spts(spt, ele, ni, nj, dim) = U_spts(spt, ele, 0);
              }
            }
          }
        }
      }
    }
//#endif

#ifdef _GPU
    if (!CPU_flag)
    {
      compute_dFdUconv_spts_Burgers_wrapper(dFdU_spts_d, U_spts_d, nSpts, nEles, nDims);
      check_error();
    }
#endif

  }

  else if (input->equation == EulerNS)
  {
//#ifdef _CPU
    if (CPU_flag)
    {
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Primitive Variables */
          double rho = U_spts(spt, ele, 0);
          double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          double e = U_spts(spt, ele, 3);
          double gam = input->gamma;

          /* Set convective dFdU values in the x-direction */
          dFdU_spts(spt, ele, 0, 0, 0) = 0;
          dFdU_spts(spt, ele, 1, 0, 0) = 0.5 * ((gam-3.0) * u*u + (gam-1.0) * v*v);
          dFdU_spts(spt, ele, 2, 0, 0) = -u * v;
          dFdU_spts(spt, ele, 3, 0, 0) = -gam * e * u / rho + (gam-1.0) * u * (u*u + v*v);

          dFdU_spts(spt, ele, 0, 1, 0) = 1;
          dFdU_spts(spt, ele, 1, 1, 0) = (3.0-gam) * u;
          dFdU_spts(spt, ele, 2, 1, 0) = v;
          dFdU_spts(spt, ele, 3, 1, 0) = gam * e / rho + 0.5 * (1.0-gam) * (3.0*u*u + v*v);

          dFdU_spts(spt, ele, 0, 2, 0) = 0;
          dFdU_spts(spt, ele, 1, 2, 0) = (1.0-gam) * v;
          dFdU_spts(spt, ele, 2, 2, 0) = u;
          dFdU_spts(spt, ele, 3, 2, 0) = (1.0-gam) * u * v;

          dFdU_spts(spt, ele, 0, 3, 0) = 0;
          dFdU_spts(spt, ele, 1, 3, 0) = (gam-1.0);
          dFdU_spts(spt, ele, 2, 3, 0) = 0;
          dFdU_spts(spt, ele, 3, 3, 0) = gam * u;

          /* Set convective dFdU values in the y-direction */
          dFdU_spts(spt, ele, 0, 0, 1) = 0;
          dFdU_spts(spt, ele, 1, 0, 1) = -u * v;
          dFdU_spts(spt, ele, 2, 0, 1) = 0.5 * ((gam-1.0) * u*u + (gam-3.0) * v*v);
          dFdU_spts(spt, ele, 3, 0, 1) = -gam * e * v / rho + (gam-1.0) * v * (u*u + v*v);

          dFdU_spts(spt, ele, 0, 1, 1) = 0;
          dFdU_spts(spt, ele, 1, 1, 1) = v;
          dFdU_spts(spt, ele, 2, 1, 1) = (1.0-gam) * u;
          dFdU_spts(spt, ele, 3, 1, 1) = (1.0-gam) * u * v;

          dFdU_spts(spt, ele, 0, 2, 1) = 1;
          dFdU_spts(spt, ele, 1, 2, 1) = u;
          dFdU_spts(spt, ele, 2, 2, 1) = (3.0-gam) * v;
          dFdU_spts(spt, ele, 3, 2, 1) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + 3.0*v*v);

          dFdU_spts(spt, ele, 0, 3, 1) = 0;
          dFdU_spts(spt, ele, 1, 3, 1) = 0;
          dFdU_spts(spt, ele, 2, 3, 1) = (gam-1.0);
          dFdU_spts(spt, ele, 3, 3, 1) = gam * v;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("compute_dFdUconv for 3D EulerNS not implemented yet!");
    }
    }
//#endif

#ifdef _GPU
    if (!CPU_flag)
    {
      compute_dFdUconv_spts_EulerNS_wrapper(dFdU_spts_d, U_spts_d, nSpts, nEles, nDims, input->gamma);
      check_error();
    }
#endif

  }
}

void Elements::compute_dFdUvisc()
{
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    /* Note: dFdUvisc = 0 for this case */
  }

  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = U_spts(spt, ele, 0);
          double momx = U_spts(spt, ele, 1);
          double momy = U_spts(spt, ele, 2);
          double e = U_spts(spt, ele, 3);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = dU_spts(spt, ele, 0, 0);
          double momx_dx = dU_spts(spt, ele, 1, 0);
          double momy_dx = dU_spts(spt, ele, 2, 0);
          
          double rho_dy = dU_spts(spt, ele, 0, 1);
          double momx_dy = dU_spts(spt, ele, 1, 1);
          double momy_dy = dU_spts(spt, ele, 2, 1);

          /* Set viscosity */
          // TODO: Store mu in array
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
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

          /* Set viscous dFdU values in the x-direction */
          dFdU_spts(spt, ele, 3, 0, 0) += -(u * tauxx + v * tauxy) / rho;
          dFdU_spts(spt, ele, 3, 1, 0) += tauxx / rho;
          dFdU_spts(spt, ele, 3, 2, 0) += tauxy / rho;
          dFdU_spts(spt, ele, 3, 3, 0) += 0;

          /* Set viscous dFdU values in the y-direction */
          dFdU_spts(spt, ele, 3, 0, 1) += -(u * tauxy + v * tauyy) / rho;
          dFdU_spts(spt, ele, 3, 1, 1) += tauxy / rho;
          dFdU_spts(spt, ele, 3, 2, 1) += tauyy / rho;
          dFdU_spts(spt, ele, 3, 3, 1) += 0;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("compute_dFdUvisc for 3D EulerNS not implemented yet!");
    }
  }
}

void Elements::compute_dFddUvisc()
{
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
#pragma omp parallel for collapse(4)
    for (unsigned int nj = 0; nj < nVars; nj++)
    {
      for (unsigned int ni = 0; ni < nVars; ni++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            dFddU_spts(spt, ele, ni, nj, 0, 0) = -input->AdvDiff_D;
            dFddU_spts(spt, ele, ni, nj, 1, 0) = 0;
            dFddU_spts(spt, ele, ni, nj, 0, 1) = 0;
            dFddU_spts(spt, ele, ni, nj, 1, 1) = -input->AdvDiff_D;
          }
        }
      }
    }
  }

  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Primitive Variables */
          double rho = U_spts(spt, ele, 0);
          double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          double e = U_spts(spt, ele, 3);

          // TODO: Add or store mu from Sutherland's law
          double diffCo1 = input->mu / rho;
          double diffCo2 = input->gamma * input->mu / (input->prandtl * rho);

          /* Set viscous dFxddUx values */
          dFddU_spts(spt, ele, 0, 0, 0, 0) = 0;
          dFddU_spts(spt, ele, 1, 0, 0, 0) = -4.0/3.0 * u * diffCo1;
          dFddU_spts(spt, ele, 2, 0, 0, 0) = -v * diffCo1;
          dFddU_spts(spt, ele, 3, 0, 0, 0) = -(4.0/3.0 * u*u + v*v) * diffCo1 + (u*u + v*v - e/rho) * diffCo2;

          dFddU_spts(spt, ele, 0, 1, 0, 0) = 0;
          dFddU_spts(spt, ele, 1, 1, 0, 0) = 4.0/3.0 * diffCo1;
          dFddU_spts(spt, ele, 2, 1, 0, 0) = 0;
          dFddU_spts(spt, ele, 3, 1, 0, 0) = u * (4.0/3.0 * diffCo1 - diffCo2);

          dFddU_spts(spt, ele, 0, 2, 0, 0) = 0;
          dFddU_spts(spt, ele, 1, 2, 0, 0) = 0;
          dFddU_spts(spt, ele, 2, 2, 0, 0) = diffCo1;
          dFddU_spts(spt, ele, 3, 2, 0, 0) = v * (diffCo1 - diffCo2);

          dFddU_spts(spt, ele, 0, 3, 0, 0) = 0;
          dFddU_spts(spt, ele, 1, 3, 0, 0) = 0;
          dFddU_spts(spt, ele, 2, 3, 0, 0) = 0;
          dFddU_spts(spt, ele, 3, 3, 0, 0) = diffCo2;

          /* Set viscous dFyddUx values */
          dFddU_spts(spt, ele, 0, 0, 1, 0) = 0;
          dFddU_spts(spt, ele, 1, 0, 1, 0) = -v * diffCo1;
          dFddU_spts(spt, ele, 2, 0, 1, 0) = 2.0/3.0 * u * diffCo1;
          dFddU_spts(spt, ele, 3, 0, 1, 0) = -1.0/3.0 * u * v * diffCo1;

          dFddU_spts(spt, ele, 0, 1, 1, 0) = 0;
          dFddU_spts(spt, ele, 1, 1, 1, 0) = 0;
          dFddU_spts(spt, ele, 2, 1, 1, 0) = -2.0/3.0 * diffCo1;
          dFddU_spts(spt, ele, 3, 1, 1, 0) = -2.0/3.0 * v * diffCo1;

          dFddU_spts(spt, ele, 0, 2, 1, 0) = 0;
          dFddU_spts(spt, ele, 1, 2, 1, 0) = diffCo1;
          dFddU_spts(spt, ele, 2, 2, 1, 0) = 0;
          dFddU_spts(spt, ele, 3, 2, 1, 0) = u * diffCo1;

          dFddU_spts(spt, ele, 0, 3, 1, 0) = 0;
          dFddU_spts(spt, ele, 1, 3, 1, 0) = 0;
          dFddU_spts(spt, ele, 2, 3, 1, 0) = 0;
          dFddU_spts(spt, ele, 3, 3, 1, 0) = 0;

          /* Set viscous dFxddUy values */
          dFddU_spts(spt, ele, 0, 0, 0, 1) = 0;
          dFddU_spts(spt, ele, 1, 0, 0, 1) = 2.0/3.0 * v * diffCo1;
          dFddU_spts(spt, ele, 2, 0, 0, 1) = -u * diffCo1;
          dFddU_spts(spt, ele, 3, 0, 0, 1) = -1.0/3.0 * u * v * diffCo1;

          dFddU_spts(spt, ele, 0, 1, 0, 1) = 0;
          dFddU_spts(spt, ele, 1, 1, 0, 1) = 0;
          dFddU_spts(spt, ele, 2, 1, 0, 1) = diffCo1;
          dFddU_spts(spt, ele, 3, 1, 0, 1) = v * diffCo1;

          dFddU_spts(spt, ele, 0, 2, 0, 1) = 0;
          dFddU_spts(spt, ele, 1, 2, 0, 1) = -2.0/3.0 * diffCo1;
          dFddU_spts(spt, ele, 2, 2, 0, 1) = 0;
          dFddU_spts(spt, ele, 3, 2, 0, 1) = -2.0/3.0 * u * diffCo1;

          dFddU_spts(spt, ele, 0, 3, 0, 1) = 0;
          dFddU_spts(spt, ele, 1, 3, 0, 1) = 0;
          dFddU_spts(spt, ele, 2, 3, 0, 1) = 0;
          dFddU_spts(spt, ele, 3, 3, 0, 1) = 0;

          /* Set viscous dFyddUy values */
          dFddU_spts(spt, ele, 0, 0, 1, 1) = 0;
          dFddU_spts(spt, ele, 1, 0, 1, 1) = -u * diffCo1;
          dFddU_spts(spt, ele, 2, 0, 1, 1) = -4.0/3.0 * v * diffCo1;
          dFddU_spts(spt, ele, 3, 0, 1, 1) = -(u*u + 4.0/3.0 * v*v) * diffCo1 + (u*u + v*v - e/rho) * diffCo2;

          dFddU_spts(spt, ele, 0, 1, 1, 1) = 0;
          dFddU_spts(spt, ele, 1, 1, 1, 1) = diffCo1;
          dFddU_spts(spt, ele, 2, 1, 1, 1) = 0;
          dFddU_spts(spt, ele, 3, 1, 1, 1) = u * (diffCo1 - diffCo2);

          dFddU_spts(spt, ele, 0, 2, 1, 1) = 0;
          dFddU_spts(spt, ele, 1, 2, 1, 1) = 0;
          dFddU_spts(spt, ele, 2, 2, 1, 1) = 4.0/3.0 * diffCo1;
          dFddU_spts(spt, ele, 3, 2, 1, 1) = v * (4.0/3.0 * diffCo1 - diffCo2);

          dFddU_spts(spt, ele, 0, 3, 1, 1) = 0;
          dFddU_spts(spt, ele, 1, 3, 1, 1) = 0;
          dFddU_spts(spt, ele, 2, 3, 1, 1) = 0;
          dFddU_spts(spt, ele, 3, 3, 1, 1) = diffCo2;
        }
      }
    }
    else if (nDims == 3)
    {
      ThrowException("compute_dFddUvisc for 3D EulerNS not implemented yet!");
    }
  }
}

void Elements::compute_globalLHS(mdvector<double> &dt)
{
  /* TODO: Move setup */
  std::vector<int> ele_list;

  mdvector<double> Cvisc0, CviscN, CdFddU0, CdFddUN;
  Cvisc0.assign({nSpts, nSpts, nDims});
  CviscN.assign({nSpts, nSpts, nDims, nFaces});
  CdFddU0.assign({nSpts, nSpts, nDims});
  CdFddUN.assign({nSpts, nSpts, nDims, nFaces});

  mdvector<double> CtempSS, CtempFS, CtempFS2;
  CtempSS.assign({nSpts, nSpts});
  CtempFS.assign({nFpts, nSpts});
  CtempFS2.assign({nFpts, nSpts});

  mdvector<double> CtempFSN, CtempFSN2;
  CtempFSN.assign({nSpts1D, nSpts});
  CtempFSN2.assign({nSpts1D, nSpts});

  /* Compute LHS */
  GLHS.clear();
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        /* Fill element list with center and neighbors */
        LHS.fill(0);
        ele_list.clear();
        ele_list.push_back((int)ele);
        for (unsigned int face = 0; face < nFaces; face++)
        {
          int eleN = geo->ele_adj(face, ele);
          ele_list.push_back(eleN);
        }

        /* Compute inviscid LHS implicit Jacobians */
        /* (Center) */
        CtempFS.fill(0);
        for (unsigned int j = 0; j < nSpts; j++)
        {
          for (unsigned int i = 0; i < nFpts; i++)
          {
            CtempFS(i, j) = dFcdU_fpts(i, ele, ni, nj, 0) * oppE(i, j);
          }
        }

        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              LHS(i, j, 0) += oppD(i, j, dim) * dFdU_spts(j, ele, ni, nj, dim);
              for (unsigned int k = 0; k < nFpts; k++)
              {
                LHS(i, j, 0) += oppD_fpts(i, k, dim) * CtempFS(k, j);
              }
            }
          }
        }

        /* (Neighbors) */
        for (unsigned int face = 0; face < nFaces; face++)
        {
          /* Neighbor element */
          int eleN = geo->ele_adj(face, ele);

          /* Neighbor contribution to inviscid LHS */
          if (eleN != -1)
          {
            /* Neighbor face */
            unsigned int faceN = 0;
            while (geo->ele_adj(faceN, eleN) != (int)ele)
              faceN++;

            CtempFSN.fill(0);
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts1D; i++)
              {
                unsigned int ind = face * nSpts1D + i;
                unsigned int indN = (faceN+1) * nSpts1D - (i+1);
                CtempFSN(i, j) = dFcdU_fpts(ind, ele, ni, nj, 1) * oppE(indN, j);
              }
            }

            for (unsigned int dim = 0; dim < nDims; dim++)
            {
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts; i++)
                {
                  for (unsigned int k = 0; k < nSpts1D; k++)
                  {
                    unsigned int ind = face * nSpts1D + k;
                    LHS(i, j, face+1) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                  }
                }
              }
            }
          }
        }

        /* Compute viscous LHS implicit Jacobians */
        if (input->viscous)
        {
          /* Add contribution from boundary conditions (dFcddU) */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            int eleN = geo->ele_adj(face, ele);
            if (eleN == -1)
            {
              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int i = 0; i < nSpts1D; i++)
                {
                  unsigned int ind = face * nSpts1D + i;
                  dFcddU_fpts(ind, ele, ni, nj, dim, 0) += dFcddU_fpts(ind, ele, ni, nj, dim, 1);
                }
              }
            }
          }

          /* (Center Contributions) */
          /* Compute viscous supplementary matrices */
          /* (Center) */
          Cvisc0.fill(0);
          CtempFS.fill(0);
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nFpts; i++)
            {
              CtempFS(i, j) = dUcdU_fpts(i, ele, ni, nj, 0) * oppE(i, j);
            }
          }

          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                Cvisc0(i, j, dim) += oppD(i, j, dim);
                for (unsigned int k = 0; k < nFpts; k++)
                {
                  Cvisc0(i, j, dim) += oppD_fpts(i, k, dim) * CtempFS(k, j);
                }
              }
            }
          }

          /* (Neighbors) */
          CviscN.fill(0);
          for (unsigned int face = 0; face < nFaces; face++)
          {
            /* Neighbor element */
            int eleN = geo->ele_adj(face, ele);

            /* Add contribution from solution boundary condition to Cvisc0 */
            if (eleN == -1)
            {
              CtempFSN.fill(0);
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts1D; i++)
                {
                  unsigned int ind = face * nSpts1D + i;
                  CtempFSN(i, j) = dUcdU_fpts(ind, ele, ni, nj, 1) * oppE(ind, j);
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts1D; k++)
                    {
                      unsigned int ind = face * nSpts1D + k;
                      Cvisc0(i, j, dim) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                    }
                  }
                }
              }
            }

            /* Compute CviscN from Neighbor element */
            else
            {
              /* Neighbor face */
              unsigned int faceN = 0;
              while (geo->ele_adj(faceN, eleN) != (int)ele)
                faceN++;

              CtempFSN.fill(0);
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts1D; i++)
                {
                  unsigned int ind = face * nSpts1D + i;
                  unsigned int indN = (faceN+1) * nSpts1D - (i+1);
                  CtempFSN(i, j) = dUcdU_fpts(ind, ele, ni, nj, 1) * oppE(indN, j);
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts1D; k++)
                    {
                      unsigned int ind = face * nSpts1D + k;
                      CviscN(i, j, dim, face) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                    }
                  }
                }
              }
            }
          }

          /* Transform viscous supplementary matrices (2D) */
          /* (Center) */
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              double Cvisc0temp = Cvisc0(i, j, 0);

              Cvisc0(i, j, 0) = Cvisc0(i, j, 0) * jaco_spts(1, 1, i, ele) -
                                Cvisc0(i, j, 1) * jaco_spts(1, 0, i, ele);

              Cvisc0(i, j, 1) = Cvisc0(i, j, 1) * jaco_spts(0, 0, i, ele) -
                                Cvisc0temp * jaco_spts(0, 1, i, ele);

              Cvisc0(i, j, 0) /= jaco_det_spts(i, ele);
              Cvisc0(i, j, 1) /= jaco_det_spts(i, ele);
            }
          }

          /* (Neighbors) */
          // Note: CviscN is zero if on boundary face
          for (unsigned int face = 0; face < nFaces; face++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                double CviscNtemp = CviscN(i, j, 0, face);

                CviscN(i, j, 0, face) = CviscN(i, j, 0, face) * jaco_spts(1, 1, i, ele) -
                                        CviscN(i, j, 1, face) * jaco_spts(1, 0, i, ele);

                CviscN(i, j, 1, face) = CviscN(i, j, 1, face) * jaco_spts(0, 0, i, ele) -
                                        CviscNtemp * jaco_spts(0, 1, i, ele);

                CviscN(i, j, 0, face) /= jaco_det_spts(i, ele);
                CviscN(i, j, 1, face) /= jaco_det_spts(i, ele);
              }
            }
          }

          /* Compute dFddU */
          /* (Center) */
          CdFddU0.fill(0);
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
          {
            for (unsigned int dimi = 0; dimi < nDims; dimi++)
            {
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts; i++)
                {
                  CdFddU0(i, j, dimi) += dFddU_spts(i, ele, ni, nj, dimi, dimj) * Cvisc0(i, j, dimj);
                }
              }
            }
          }

          /* (Neighbors) */
          // Note: CdFddUN is zero if on boundary face
          CdFddUN.fill(0);
          for (unsigned int face = 0; face < nFaces; face++)
          {
            for (unsigned int dimj = 0; dimj < nDims; dimj++)
            {
              for (unsigned int dimi = 0; dimi < nDims; dimi++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    CdFddUN(i, j, dimi, face) += dFddU_spts(i, ele, ni, nj, dimi, dimj) * CviscN(i, j, dimj, face);
                  }
                }
              }
            }
          }

          /* Transform dFddU (2D) */
          /* (Center) */
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              double CdFddU0temp = CdFddU0(i, j, 0);

              CdFddU0(i, j, 0) = CdFddU0(i, j, 0) * jaco_spts(1, 1, i, ele) -
                                 CdFddU0(i, j, 1) * jaco_spts(0, 1, i, ele);

              CdFddU0(i, j, 1) = CdFddU0(i, j, 1) * jaco_spts(0, 0, i, ele) -
                                 CdFddU0temp * jaco_spts(1, 0, i, ele);
            }
          }

          /* (Neighbors) */
          // Note: CdFddUN is zero if on boundary face
          for (unsigned int face = 0; face < nFaces; face++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                double CdFddUNtemp = CdFddUN(i, j, 0, face);

                CdFddUN(i, j, 0, face) = CdFddUN(i, j, 0, face) * jaco_spts(1, 1, i, ele) -
                                         CdFddUN(i, j, 1, face) * jaco_spts(0, 1, i, ele);

                CdFddUN(i, j, 1, face) = CdFddUN(i, j, 1, face) * jaco_spts(0, 0, i, ele) -
                                         CdFddUNtemp * jaco_spts(1, 0, i, ele);
              }
            }
          }

          /* Center contribution to viscous LHS */
          /* (Center) Term 1 */
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            CtempSS.fill(0);
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                CtempSS(i, j) += CdFddU0(i, j, dim);
              }
            }

            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                for (unsigned int k = 0; k < nSpts; k++)
                {
                  LHS(i, j, 0) += oppD(i, k, dim) * CtempSS(k, j);
                }
              }
            }
          }

          /* (Center) Term 2 */
          CtempFS.fill(0);
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            CtempFS2.fill(0);
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nFpts; i++)
              {
                CtempFS2(i, j) += dFcddU_fpts(i, ele, ni, nj, dim, 0) * oppE(i, j);
              }
            }

            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nFpts; i++)
              {
                for (unsigned int k = 0; k < nSpts; k++)
                {
                  CtempFS(i, j) += CtempFS2(i, k) * Cvisc0(k, j, dim);
                }
              }
            }
          }

          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                for (unsigned int k = 0; k < nFpts; k++)
                {
                  LHS(i, j, 0) += oppD_fpts(i, k, dim) * CtempFS(k, j);
                }
              }
            }
          }

          /* (Neighbors) */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            /* Neighbor element */
            int eleN = geo->ele_adj(face, ele);

            /* Center contribution to Neighbor viscous LHS */
            if (eleN != -1)
            {
              /* (Neighbor) Term 1 */
              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts; k++)
                    {
                      LHS(i, j, face+1) += oppD(i, k, dim) * CdFddUN(k, j, dim, face);
                    }
                  }
                }
              }

              /* (Neighbor) Term 2 */
              CtempFS.fill(0);
              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                CtempFS2.fill(0);
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nFpts; i++)
                  {
                    CtempFS2(i, j) += dFcddU_fpts(i, ele, ni, nj, dim, 0) * oppE(i, j);
                  }
                }

                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nFpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts; k++)
                    {
                      CtempFS(i, j) += CtempFS2(i, k) * CviscN(k, j, dim, face);
                    }
                  }
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nFpts; k++)
                    {
                      LHS(i, j, face+1) += oppD_fpts(i, k, dim) * CtempFS(k, j);
                    }
                  }
                }
              }
            }
          }

          /* (Neighbor Contributions) */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            /* Neighbor element */
            int eleN = geo->ele_adj(face, ele);

            /* Neighbor contribution to viscous LHS */
            if (eleN != -1)
            {
              /* Neighbor face */
              unsigned int faceN = 0;
              while (geo->ele_adj(faceN, eleN) != (int)ele)
                faceN++;

              /* Compute viscous supplementary matrices */
              /* (Neighbor Center) */
              Cvisc0.fill(0);         
              CtempFS.fill(0);
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nFpts; i++)
                {
                  CtempFS(i, j) = dUcdU_fpts(i, eleN, ni, nj, 0) * oppE(i, j);
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    Cvisc0(i, j, dim) += oppD(i, j, dim);
                    for (unsigned int k = 0; k < nFpts; k++)
                    {
                      Cvisc0(i, j, dim) += oppD_fpts(i, k, dim) * CtempFS(k, j);
                    }
                  }
                }
              }

              /* (2nd Neighbors) */
              CviscN.fill(0);
              for (unsigned int face2 = 0; face2 < nFaces; face2++)
              {
                /* 2nd Neighbor element */
                int eleN2 = geo->ele_adj(face2, eleN);

                /* Add contribution from solution boundary condition to Cvisc0 */
                if (eleN2 == -1)
                {
                  CtempFSN.fill(0);
                  for (unsigned int j = 0; j < nSpts; j++)
                  {
                    for (unsigned int i = 0; i < nSpts1D; i++)
                    {
                      unsigned int ind = face2 * nSpts1D + i;
                      CtempFSN(i, j) = dUcdU_fpts(ind, eleN, ni, nj, 1) * oppE(ind, j);
                    }
                  }

                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts; i++)
                      {
                        for (unsigned int k = 0; k < nSpts1D; k++)
                        {
                          unsigned int ind = face2 * nSpts1D + k;
                          Cvisc0(i, j, dim) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                        }
                      }
                    }
                  }
                }

                /* Compute CviscN from 2nd Neighbor element */
                else
                {
                  /* 2nd Neighbor face */
                  unsigned int faceN2 = 0;
                  while (geo->ele_adj(faceN2, eleN2) != (int)eleN)
                    faceN2++;

                  CtempFSN.fill(0);
                  for (unsigned int j = 0; j < nSpts; j++)
                  {
                    for (unsigned int i = 0; i < nSpts1D; i++)
                    {
                      unsigned int ind = face2 * nSpts1D + i;
                      unsigned int indN = (faceN2+1) * nSpts1D - (i+1);
                      CtempFSN(i, j) = dUcdU_fpts(ind, eleN, ni, nj, 1) * oppE(indN, j);
                    }
                  }

                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts; i++)
                      {
                        for (unsigned int k = 0; k < nSpts1D; k++)
                        {
                          unsigned int ind = face2 * nSpts1D + k;
                          CviscN(i, j, dim, face2) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                        }
                      }
                    }
                  }
                }
              }

              /* Transform viscous supplementary matrices (2D) */
              /* (Neighbor Center) */
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts; i++)
                {
                  double Cvisc0temp = Cvisc0(i, j, 0);

                  Cvisc0(i, j, 0) = Cvisc0(i, j, 0) * jaco_spts(1, 1, i, eleN) -
                                    Cvisc0(i, j, 1) * jaco_spts(1, 0, i, eleN);

                  Cvisc0(i, j, 1) = Cvisc0(i, j, 1) * jaco_spts(0, 0, i, eleN) -
                                    Cvisc0temp * jaco_spts(0, 1, i, eleN);

                  Cvisc0(i, j, 0) /= jaco_det_spts(i, eleN);
                  Cvisc0(i, j, 1) /= jaco_det_spts(i, eleN);
                }
              }

              /* (2nd Neighbors) */
              // Note: CviscN is zero if on boundary face
              for (unsigned int face2 = 0; face2 < nFaces; face2++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    double CviscNtemp = CviscN(i, j, 0, face2);

                    CviscN(i, j, 0, face2) = CviscN(i, j, 0, face2) * jaco_spts(1, 1, i, eleN) -
                                             CviscN(i, j, 1, face2) * jaco_spts(1, 0, i, eleN);

                    CviscN(i, j, 1, face2) = CviscN(i, j, 1, face2) * jaco_spts(0, 0, i, eleN) -
                                             CviscNtemp * jaco_spts(0, 1, i, eleN);

                    CviscN(i, j, 0, face2) /= jaco_det_spts(i, eleN);
                    CviscN(i, j, 1, face2) /= jaco_det_spts(i, eleN);
                  }
                }
              }

              /* Neighbors contribution to viscous LHS */
              /* (Neighbor Center) Term 1 */
              CtempFSN.fill(0);
              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                CtempFSN2.fill(0);
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts1D; i++)
                  {
                    unsigned int ind = face * nSpts1D + i;
                    unsigned int indN = (faceN+1) * nSpts1D - (i+1);
                    CtempFSN2(i, j) += dFcddU_fpts(ind, ele, ni, nj, dim, 1) * oppE(indN, j);
                  }
                }

                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts1D; i++)
                  {
                    for (unsigned int k = 0; k < nSpts; k++)
                    {
                      CtempFSN(i, j) += CtempFSN2(i, k) * Cvisc0(k, j, dim);
                    }
                  }
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts1D; k++)
                    {
                      unsigned int ind = face * nSpts1D + k;
                      LHS(i, j, face+1) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                    }
                  }
                }
              }

              /* (2nd Neighbors) */
              for (unsigned int face2 = 0; face2 < nFaces; face2++)
              {
                /* 2nd Neighbor element */
                int eleN2 = geo->ele_adj(face2, eleN);
                if (eleN2 != -1)
                {
                  unsigned int mat = (unsigned int)ele_list.size();
                  for (unsigned int i = 0; i < ele_list.size(); i++)
                  {
                    if (ele_list[i] == (int)eleN2)
                    {
                      mat = i;
                      break;
                    }
                  }
                  if (mat == (unsigned int)ele_list.size())
                  {
                    ele_list.push_back(eleN2);
                  }

                  /* (2nd Neighbors) Term 1 */
                  CtempFSN.fill(0);
                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    CtempFSN2.fill(0);
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts1D; i++)
                      {
                        unsigned int ind = face * nSpts1D + i;
                        unsigned int indN = (faceN+1) * nSpts1D - (i+1);
                        CtempFSN2(i, j) += dFcddU_fpts(ind, ele, ni, nj, dim, 1) * oppE(indN, j);
                      }
                    }

                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts1D; i++)
                      {
                        for (unsigned int k = 0; k < nSpts; k++)
                        {
                          CtempFSN(i, j) += CtempFSN2(i, k) * CviscN(k, j, dim, face2);
                        }
                      }
                    }
                  }

                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts; i++)
                      {
                        for (unsigned int k = 0; k < nSpts1D; k++)
                        {
                          unsigned int ind = face * nSpts1D + k;
                          LHS(i, j, mat) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }

        /* Fill LHS implicit Jacobian */
        /* (Center) */
        for (unsigned int j = 0; j < nSpts; j++)
        {
          for (unsigned int i = 0; i < nSpts; i++)
          {
            /* Determine index */
            int Gi = ele*nVars*nSpts + ni*nSpts + i;
            int Gj = ele*nVars*nSpts + nj*nSpts + j;

            /* Compute val */
            double Gval;
            if (input->dt_type != 2)
            {
              Gval = dt(0) * LHS(i, j, 0) / jaco_det_spts(i, ele);
            }
            else
            {
              Gval = dt(ele) * LHS(i, j, 0) / jaco_det_spts(i, ele);
            }

            if (i == j && ni == nj)
            {
              Gval += 1;
            }

            /* Fill Jacobian */
            if (Gval != 0)
            {
              GLHS.addEntry(Gi, Gj, Gval);
            }
          }
        }

        /* (Neighbors) */
        for (unsigned int mat = 1; mat < ele_list.size(); mat++)
        {
          // TODO: Include boundary condition case
          int eleN = ele_list[mat];
          if (eleN != -1)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                /* Determine index */
                int Gi = ele*nVars*nSpts + ni*nSpts + i;
                int Gj = eleN*nVars*nSpts + nj*nSpts + j;

                /* Compute val and fill */
                double Gval;
                if (input->dt_type != 2)
                {
                  Gval = dt(0) * LHS(i, j, mat) / jaco_det_spts(i, ele);
                }
                else
                {
                  Gval = dt(ele) * LHS(i, j, mat) / jaco_det_spts(i, ele);
                }

                /* Fill Jacobian */
                if (Gval != 0)
                {
                  GLHS.addEntry(Gi, Gj, Gval);
                }
              }
            }
          }
        }
      }
    }
  }
  GLHS.toCSR();
}

#ifdef _CPU
void Elements::compute_localLHS(mdvector<double> &dt)
#endif
#ifdef _GPU
void Elements::compute_localLHS(mdvector_gpu<double> &dt_d)
#endif
{

#ifdef _CPU

  /* Compute LHS */
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        /* Compute center inviscid LHS implicit Jacobian */
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          auto *A = &oppD(0, 0, dim);
          auto *B = &dFdU_spts(0, ele, ni, nj, dim);
          auto *C = &LHS(0, ni, 0, nj, ele);

          double fac = (dim == 0) ? 0.0 : 1.0;

          dgmm(nSpts, nSpts, 1, A, oppD.ldim(), B, 0, fac, C, nSpts*nVars);
        }

        auto *A = &oppDiv_fpts(0, 0);
        auto *B = &dFcdU_fpts(0, ele, ni, nj, 0);
        auto *C = &CtempSF(0, 0);
        dgmm(nSpts, nFpts, 1, A, oppDiv_fpts.ldim(), B, 0, 0, C, nSpts);

        A = &CtempSF(0, 0);
        B = &oppE(0, 0);
        C = &LHS(0, ni, 0, nj, ele);
        gemm(nSpts, nSpts, nFpts, 1, A, nSpts, B, oppE.ldim(), 1, C, nSpts*nVars);

        /* Compute center viscous LHS implicit Jacobian */
        if (input->viscous)
        {
          /* Add contribution from boundary conditions (dFcddU) */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            int eleN = geo->ele_adj(face, ele);
            if (eleN == -1)
            {
              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int i = 0; i < nSpts1D; i++)
                {
                  unsigned int ind = face * nSpts1D + i;
                  dFcddU_fpts(ind, ele, ni, nj, dim, 0) += dFcddU_fpts(ind, ele, ni, nj, dim, 1);
                }
              }
            }
          }

          /* Compute center viscous supplementary matrix */
          Cvisc0.fill(0);       
          CtempFS.fill(0);
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nFpts; i++)
            {
              CtempFS(i, j) = dUcdU_fpts(i, ele, ni, nj, 0) * oppE(i, j);
            }
          }

          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                Cvisc0(i, j, dim) += oppD(i, j, dim);
                for (unsigned int k = 0; k < nFpts; k++)
                {
                  Cvisc0(i, j, dim) += oppD_fpts(i, k, dim) * CtempFS(k, j);
                }
              }
            }
          }

          /* Add contribution from solution boundary condition to Cvisc0 */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            /* Neighbor element */
            int eleN = geo->ele_adj(face, ele);
            if (eleN == -1)
            {
              CtempFSN.fill(0);
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts1D; i++)
                {
                  unsigned int ind = face * nSpts1D + i;
                  CtempFSN(i, j) = dUcdU_fpts(ind, ele, ni, nj, 1) * oppE(ind, j);
                }
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
              {
                for (unsigned int j = 0; j < nSpts; j++)
                {
                  for (unsigned int i = 0; i < nSpts; i++)
                  {
                    for (unsigned int k = 0; k < nSpts1D; k++)
                    {
                      unsigned int ind = face * nSpts1D + k;
                      Cvisc0(i, j, dim) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                    }
                  }
                }
              }
            }
          }

          /* Transform center viscous supplementary matrices (2D) */
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              double Cvisc0temp = Cvisc0(i, j, 0);

              Cvisc0(i, j, 0) = Cvisc0(i, j, 0) * jaco_spts(1, 1, i, ele) -
                                Cvisc0(i, j, 1) * jaco_spts(1, 0, i, ele);

              Cvisc0(i, j, 1) = Cvisc0(i, j, 1) * jaco_spts(0, 0, i, ele) -
                                Cvisc0temp * jaco_spts(0, 1, i, ele);

              Cvisc0(i, j, 0) /= jaco_det_spts(i, ele);
              Cvisc0(i, j, 1) /= jaco_det_spts(i, ele);
            }
          }

          /* Compute center dFddU */
          CdFddU0.fill(0);
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
          {
            for (unsigned int dimi = 0; dimi < nDims; dimi++)
            {
              for (unsigned int j = 0; j < nSpts; j++)
              {
                for (unsigned int i = 0; i < nSpts; i++)
                {
                  CdFddU0(i, j, dimi) += dFddU_spts(i, ele, ni, nj, dimi, dimj) * Cvisc0(i, j, dimj);
                }
              }
            }
          }

          /* Transform center dFddU (2D) */
          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              double CdFddU0temp = CdFddU0(i, j, 0);

              CdFddU0(i, j, 0) = CdFddU0(i, j, 0) * jaco_spts(1, 1, i, ele) -
                                 CdFddU0(i, j, 1) * jaco_spts(0, 1, i, ele);

              CdFddU0(i, j, 1) = CdFddU0(i, j, 1) * jaco_spts(0, 0, i, ele) -
                                 CdFddU0temp * jaco_spts(1, 0, i, ele);
            }
          }

          /* (Center) Term 1 */
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            CtempSS.fill(0);
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                CtempSS(i, j) += CdFddU0(i, j, dim);
              }
            }

            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                double val = 0;
                for (unsigned int k = 0; k < nSpts; k++)
                {
                  val += oppD(i, k, dim) * CtempSS(k, j);
                }
                LHS(i, ni, j, nj, ele) += val;
              }
            }
          }

          /* (Center) Term 2 */
          CtempFS.fill(0);
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            CtempFS2.fill(0);
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nFpts; i++)
              {
                CtempFS2(i, j) += dFcddU_fpts(i, ele, ni, nj, dim, 0) * oppE(i, j);
              }
            }

            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nFpts; i++)
              {
                for (unsigned int k = 0; k < nSpts; k++)
                {
                  CtempFS(i, j) += CtempFS2(i, k) * Cvisc0(k, j, dim);
                }
              }
            }
          }

          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            for (unsigned int j = 0; j < nSpts; j++)
            {
              for (unsigned int i = 0; i < nSpts; i++)
              {
                double val = 0;
                for (unsigned int k = 0; k < nFpts; k++)
                {
                  val += oppD_fpts(i, k, dim) * CtempFS(k, j);
                }
                LHS(i, ni, j, nj, ele) += val;
              }
            }
          }

          /* Add center contribution to Neighbor gradient */
          for (unsigned int face = 0; face < nFaces; face++)
          {
            /* Neighbor element */
            int eleN = geo->ele_adj(face, ele);
            if (eleN != -1)
            {
              /* Neighbor face */
              unsigned int faceN = 0;
              while (geo->ele_adj(faceN, eleN) != (int)ele)
                faceN++;

              /* (2nd Neighbors) */
              for (unsigned int face2 = 0; face2 < nFaces; face2++)
              {
                /* 2nd Neighbor element */
                int eleN2 = geo->ele_adj(face2, eleN);
                if (eleN2 == (int)ele)
                {
                  /* 2nd Neighbor face */
                  unsigned int faceN2 = face;

                  /* Compute 2nd Neighbor viscous supplementary matrix */
                  // Note: Only need to zero out face2
                  CviscN.fill(0);
                  CtempFSN.fill(0);
                  for (unsigned int j = 0; j < nSpts; j++)
                  {
                    for (unsigned int i = 0; i < nSpts1D; i++)
                    {
                      unsigned int ind = face2 * nSpts1D + i;
                      unsigned int indN = (faceN2+1) * nSpts1D - (i+1);
                      CtempFSN(i, j) = dUcdU_fpts(ind, eleN, ni, nj, 1) * oppE(indN, j);
                    }
                  }

                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts; i++)
                      {
                        for (unsigned int k = 0; k < nSpts1D; k++)
                        {
                          unsigned int ind = face2 * nSpts1D + k;
                          CviscN(i, j, dim, face2) += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                        }
                      }
                    }
                  }

                  /* Transform 2nd Neighbor viscous supplementary matrices (2D) */
                  for (unsigned int j = 0; j < nSpts; j++)
                  {
                    for (unsigned int i = 0; i < nSpts; i++)
                    {
                      double CviscNtemp = CviscN(i, j, 0, face2);

                      CviscN(i, j, 0, face2) = CviscN(i, j, 0, face2) * jaco_spts(1, 1, i, eleN) -
                                               CviscN(i, j, 1, face2) * jaco_spts(1, 0, i, eleN);

                      CviscN(i, j, 1, face2) = CviscN(i, j, 1, face2) * jaco_spts(0, 0, i, eleN) -
                                               CviscNtemp * jaco_spts(0, 1, i, eleN);

                      CviscN(i, j, 0, face2) /= jaco_det_spts(i, eleN);
                      CviscN(i, j, 1, face2) /= jaco_det_spts(i, eleN);
                    }
                  }

                  /* (2nd Neighbors) Term 1 */
                  CtempFSN.fill(0);
                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    CtempFSN2.fill(0);
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts1D; i++)
                      {
                        unsigned int ind = face * nSpts1D + i;
                        unsigned int indN = (faceN+1) * nSpts1D - (i+1);
                        CtempFSN2(i, j) += dFcddU_fpts(ind, ele, ni, nj, dim, 1) * oppE(indN, j);
                      }
                    }

                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts1D; i++)
                      {
                        for (unsigned int k = 0; k < nSpts; k++)
                        {
                          CtempFSN(i, j) += CtempFSN2(i, k) * CviscN(k, j, dim, face2);
                        }
                      }
                    }
                  }

                  for (unsigned int dim = 0; dim < nDims; dim++)
                  {
                    for (unsigned int j = 0; j < nSpts; j++)
                    {
                      for (unsigned int i = 0; i < nSpts; i++)
                      {
                        double val = 0;
                        for (unsigned int k = 0; k < nSpts1D; k++)
                        {
                          unsigned int ind = face * nSpts1D + k;
                          val += oppD_fpts(i, ind, dim) * CtempFSN(k, j);
                        }
                        LHS(i, ni, j, nj, ele) += val;
                      }
                    }
                  }
                }
              }
            }
          }
        }

        /* Compute center LHS implicit Jacobian */
        // TODO: Create a new function for this operation
        for (unsigned int j = 0; j < nSpts; j++)
        {
          for (unsigned int i = 0; i < nSpts; i++)
          {
            if (input->dt_type != 2)
            {
              LHS(i, ni, j, nj, ele) = dt(0) * LHS(i, ni, j, nj, ele) / jaco_det_spts(i, ele);
            }

            else
            {
              LHS(i, ni, j, nj, ele) = dt(ele) * LHS(i, ni, j, nj, ele) / jaco_det_spts(i, ele);
            }

            if (i == j && ni == nj)
            {
              LHS(i, ni, j, nj, ele) += 1;
            }
          }
        }
      }
    }
  }
#endif

#ifdef _GPU
  /* Compute center inviscid LHS implicit Jacobian */

  /* Fill temporary matrix with oppDiv_fpts scaled by dFcdU_fpts */
  add_scaled_oppDiv_wrapper(LHS_tempSF_d, oppDiv_fpts_d, dFcdU_fpts_d, nSpts, nFpts, nVars, nEles);

  /* Multiply blocks by oppE, put result in LHS */
  cublasDgemmBatched_wrapper(nSpts * nVars, nSpts, nFpts, 1.0, (const double**)LHS_tempSF_subptrs_d.data(), nSpts * nVars, 
      (const double**) oppE_ptrs_d.data(), oppE_d.ldim(), 0.0, LHS_subptrs_d.data(), nSpts * nVars, nEles * nVars);

  /* Add oppD scaled by dFdU_spts to LHS */
  add_scaled_oppD_wrapper(LHS_d, oppD_d, dFdU_spts_d, nSpts, nVars, nEles, nDims);

  /* Finalize LHS (scale by jacobian, dt, and add identity) */
  finalize_LHS_wrapper(LHS_d, dt_d, jaco_det_spts_d, nSpts, nVars, nEles, input->dt_type);

  check_error();

#endif
}

void Elements::compute_Uavg()
{
#ifdef _CPU
#pragma omp parallel for collapse(2)
  /* Compute average solution using quadrature */
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      double sum = 0.0;
      double vol = 0.0;

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        /* Get quadrature weight */
        unsigned int i = idx_spts(spt,0);
        unsigned int j = idx_spts(spt,1);
        double weight = weights_spts(i) * weights_spts(j);

        sum += weight * jaco_det_spts(spt, ele) * U_spts(spt, ele, n);
        vol += weight * jaco_det_spts(spt, ele);
      }

      Uavg(ele, n) = sum / vol; 

    }
  }
#endif

#ifdef _GPU
  compute_Uavg_wrapper(U_spts_d, Uavg_d, jaco_det_spts_d, weights_spts_d, nSpts, nEles, nVars, order);
#endif
}

void Elements::poly_squeeze()
{
#ifdef _CPU
  double V[3]; 

  /* For each element, check for negative density at solution and flux points */
  double tol = 1e-10;
#pragma omp parallel for 
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    bool negRho = false;
    double minRho = U_spts(0, ele, 0);

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      if (U_spts(spt, ele, 0) < 0)
      {
        negRho = true;
        minRho = std::min(minRho, U_spts(spt, ele, 0));
      }
    }
    
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      if (U_fpts(fpt, ele, 0) < 0)
      {
        negRho = true;
        minRho = std::min(minRho, U_fpts(fpt, ele, 0));
      }
    }

    /* If negative density found, squeeze density */
    if (negRho)
    {
      double theta = (Uavg(ele, 0) - tol) / (Uavg(ele , 0) - minRho); 
      //double theta = 1.0;

      for (unsigned int spt = 0; spt < nSpts; spt++)
        U_spts(spt, ele, 0) = theta * U_spts(spt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);

      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        U_fpts(fpt, ele, 0) = theta * U_fpts(fpt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);
      
    }
  }


#pragma omp parallel for 
  /* For each element, check for entropy loss */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    double minTau = 1.0;

    /* Get minimum tau value */
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      double rho = U_spts(spt, ele, 0);
      double momF = (U_spts(spt, ele, 1) * U_spts(spt,ele,1) + U_spts(spt, ele, 2) * 
          U_spts(spt, ele,2)) / U_spts(spt, ele, 0);
      double P = (input->gamma - 1.0) * (U_spts(spt, ele, 3) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);

    }
    
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      double rho = U_fpts(fpt, ele, 0);
      double momF = (U_fpts(fpt, ele, 1) * U_fpts(fpt,ele,1) + U_spts(fpt, ele, 2) * 
          U_fpts(fpt, ele,2)) / U_fpts(fpt, ele, 0);
      double P = (input->gamma - 1.0) * (U_fpts(fpt, ele, 3) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);

    }

    /* If minTau is negative, squeeze solution */
    if (minTau < 0)
    {
      double rho = Uavg(ele, 0);
      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = Uavg(ele, dim+1) / rho;
        Vsq += V[dim] * V[dim];
      }

      double e = Uavg(ele, 3);
      double P = (input->gamma - 1.0) * (e - 0.5 * rho * Vsq);

      double eps = minTau / (minTau - P + input->exps0 * std::pow(rho, input->gamma));

//      if (P < input->exps0 * std::pow(rho, input->gamma))
//        std::cout << "Constraint violated. Lower CFL?" << std::endl;

      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          U_spts(spt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_spts(spt, ele, n);
        }
      }

      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        {
          U_fpts(fpt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_fpts(fpt, ele, n);
        }
      }

    }

  }
#endif

#ifdef _GPU
  poly_squeeze_wrapper(U_spts_d, U_fpts_d, Uavg_d, input->gamma, input->exps0, nSpts, nFpts,
      nEles, nVars, nDims);
#endif

}

void Elements::poly_squeeze_ppts()
{
  double V[3]; 

  /* For each element, check for negative density at plot points */
  double tol = 1e-10;
#pragma omp parallel for 
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    bool negRho = false;
    double minRho = U_ppts(0, ele, 0);

    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      if (U_ppts(ppt, ele, 0) < 0)
      {
        negRho = true;
        minRho = std::min(minRho, U_ppts(ppt, ele, 0));
      }
    }
    
    /* If negative density found, squeeze density */
    if (negRho)
    {
      double theta = std::abs(Uavg(ele, 0) - tol) / (Uavg(ele , 0) - minRho); 

      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
        U_ppts(ppt, ele, 0) = theta * U_ppts(ppt, ele, 0) + (1.0 - theta) * Uavg(ele, 0);
    }
  }

#pragma omp parallel for 
  /* For each element, check for entropy loss */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    double minTau = 1.0;

    /* Get minimum tau value */
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      double rho = U_ppts(ppt, ele, 0);
      double momF = (U_ppts(ppt, ele, 1) * U_ppts(ppt,ele,1) + U_ppts(ppt, ele, 2) * 
          U_ppts(ppt, ele,2)) / U_ppts(ppt, ele, 0);
      double P = (input->gamma - 1.0) * (U_ppts(ppt, ele, 3) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);
    }
    
    /* If minTau is negative, squeeze solution */
    if (minTau < 0)
    {
      double rho = Uavg(ele, 0);
      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = Uavg(ele, dim+1) / rho;
        Vsq += V[dim] * V[dim];
      }

      double e = Uavg(ele, 3);
      double P = (input->gamma - 1.0) * (e - 0.5 * rho * Vsq);

      double eps = minTau / (minTau - P + input->exps0 * std::pow(rho, input->gamma));

      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ppt = 0; ppt < nPpts; ppt++)
        {
          U_ppts(ppt, ele, n) = eps * Uavg(ele, n) + (1.0 - eps) * U_ppts(ppt, ele, n);
          //U_ppts(ppt, ele, n) = (1.0 - eps) * Uavg(ele, n) + eps * U_ppts(ppt, ele, n);
        }
      }

    }

  }

}

