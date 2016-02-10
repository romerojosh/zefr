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

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(node,spt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_spts(node,spt,dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_fpts(fpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(node, fpt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_fpts(node, fpt, dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_ppts(ppt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(node,ppt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_ppts(node,ppt,dim) = calc_d_shape(shape_order, node, loc, dim);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(node,qpt) = calc_shape(shape_order, node, loc);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_qpts(node,qpt,dim) = calc_d_shape(shape_order, node, loc, dim);
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
        for (unsigned int dim = 0; dim < nDims; dim++)
          loc[dim] = loc_spts(ispt , dim);

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
        for (unsigned int dim = 0; dim < nDims; dim++)
          loc[dim] = loc_spts(spt , dim);

        oppD_fpts(spt,fpt,dim) = calc_d_nodal_basis_fpts(fpt, loc, dim);
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

void Elements::extrapolate_U()
{
#ifdef _CPU
  auto &A = oppE(0,0);
  auto &B = U_spts(0, 0, 0);
  auto &C = U_fpts(0, 0, 0);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, nEles * nVars,
        nSpts, 1.0, &A, nFpts, &B, nSpts, 0.0, &C, nFpts);
#else
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, nEles * nVars,
        nSpts, 1.0, &A, nFpts, &B, nSpts, 0.0, &C, nFpts);
#endif

#endif

#ifdef _GPU
  cublasDGEMM_wrapper(nFpts, nEles * nVars, nSpts, 1.0,
      oppE_d.data(), nFpts, U_spts_d.data(), nSpts, 0.0,
      U_fpts_d.data(), nFpts);

  check_error();
#endif

}

void Elements::extrapolate_dU()
{
#ifdef _CPU
  for (unsigned int dim = 0; dim < nDims; dim++)
    {
        auto &A = oppE(0,0);
        auto &B = dU_spts(0, 0, 0, dim);
        auto &C = dU_fpts(0, 0, 0, dim);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, 
            nEles * nVars, nSpts, 1.0, &A, nFpts, &B, nSpts, 0.0, &C, nFpts);
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, nEles * nVars,
            nSpts, 1.0, &A, nFpts, &B, nSpts, 0.0, &C, nFpts);
#endif
    }
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    cublasDGEMM_wrapper(nFpts, nEles * nVars, nSpts, 1.0, 
        oppE_d.data(), nFpts, dU_spts_d.data() + dim * (nSpts * 
        nVars * nEles), nSpts, 0.0, dU_fpts_d.data() + dim * 
        (nFpts * nVars * nEles), nFpts);
  }
#endif
}

void Elements::compute_dU()
{
#ifdef _CPU
  /* Compute contribution to derivative from solution at solution points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      auto &A = oppD(0, 0, dim);
      auto &B = U_spts(0, 0, 0);
      auto &C = dU_spts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nSpts, 1.0, &A, nSpts, &B, nSpts, 
          0.0, &C, nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nSpts, 1.0, &A, nSpts, &B, nSpts, 
          0.0, &C, nSpts);
#endif
    }

    /* Compute contribution to derivative from common solution at flux points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      auto &A = oppD_fpts(0, 0, dim);
      auto &B = Ucomm(0, 0, 0);
      auto &C = dU_spts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nFpts, 1.0, &A, nSpts, &B, nFpts, 
          1.0, &C, nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nFpts, 1.0, &A, nSpts, &B, nFpts, 
          1.0, &C, nSpts);
#endif
    }

#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    /* Compute contribution to derivative from solution at solution points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nSpts, 1.0,
        oppD_d.data() + dim * (nSpts * nSpts), nSpts, 
        U_spts_d.data(), nSpts, 0.0, dU_spts_d.data() + dim * 
        (nSpts * nVars * nEles), nSpts);

    check_error();

    /* Compute contribution to derivative from common solution at flux points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nFpts, 1.0,
        oppD_fpts_d.data() + dim * (nSpts * nFpts), nSpts,
        Ucomm_d.data(), nFpts, 1.0, dU_spts_d.data() + dim * 
        (nSpts * nVars * nEles), nSpts);

    check_error();
  }
#endif

}

void Elements::compute_dF()
{
#ifdef _CPU
    /* Compute contribution to derivative from flux at solution points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      auto &A = oppD(0, 0,dim);
      auto &B = F_spts(0, 0, 0, dim);
      auto &C = dF_spts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nSpts, 1.0, &A, nSpts, &B, nSpts, 0.0, &C, nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, nEles * nVars,
            nSpts, 1.0, &A, nSpts, &B, nSpts, 0.0, &C, nSpts);
#endif

    }

    /* Compute contribution to derivative from common flux at flux points */
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      auto &A = oppD_fpts(0, 0, dim);
      auto &B = Fcomm(0, 0, 0);
      auto &C = dF_spts(0, 0, 0, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, 
          nEles * nVars, nFpts, 1.0, &A, nSpts, &B, nFpts, 1.0, &C, nSpts);
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, nEles * nVars,
          nFpts, 1.0, &A, nSpts, &B, nFpts, 1.0, &C, nSpts);
#endif
    }

#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    /* Compute contribution to derivative from flux at solution points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nSpts, 1.0,
        oppD_d.data() + dim * (nSpts * nSpts), nSpts, 
        F_spts_d.data() + dim * (nSpts * nVars * nEles), 
        nSpts, 0.0, dF_spts_d.data() + dim * (nSpts * nVars * 
        nEles), nSpts);

    check_error();

    /* Compute contribution to derivative from common flux at flux points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nFpts, 1.0,
        oppD_fpts_d.data() + dim * (nSpts * nFpts), nSpts, 
        Fcomm_d.data(), nFpts, 1.0, dF_spts_d.data() + dim * 
        (nSpts * nVars * nEles), nSpts);

    check_error();
  }
#endif

}

void Elements::compute_divF(unsigned int stage)
{
#ifdef _CPU

  /* Compute parent space divergence of flux */
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < nVars; n++)
    for (unsigned int ele =0; ele < nEles; ele++)
      for (unsigned int spt = 0; spt < nSpts; spt++)
        divF_spts(spt, ele, n, stage) = dF_spts(spt, ele, n, 0);

  for (unsigned int dim = 1; dim < nDims; dim ++)
#pragma omp parallel for collapse(3)
    for (unsigned int n = 0; n < nVars; n++)
      for (unsigned int ele =0; ele < nEles; ele++)
        for (unsigned int spt = 0; spt < nSpts; spt++)
          divF_spts(spt, ele, n, stage) += dF_spts(spt, ele, n, dim);

  /* Transform to physical space */
  for (unsigned int n = 0; n < nVars; n++)
#pragma omp parallel for collapse(2)
    for (unsigned int ele =0; ele < nEles; ele++)
      for (unsigned int spt = 0; spt < nSpts; spt++)
        divF_spts(spt, ele, n, stage) /= jaco_det_spts(spt, ele);

#endif

#ifdef _GPU
  compute_divF_wrapper(divF_spts_d, dF_spts_d, nSpts, nVars, nEles, 
      nDims, input->equation, stage);
  check_error();
#endif
}

void Elements::compute_intF(unsigned int stage)
{
#ifdef _CPU

  /* Compute integrated flux */
  for (unsigned int n = 0; n < nVars; n++)
    for (unsigned int ele = 0; ele < nEles; ele++)
      divF_spts(0, ele, n, stage) = -2*Fcomm(0, ele, n);

  for (unsigned int n = 0; n < nVars; n++)
    for (unsigned int ele = 0; ele < nEles; ele++)
      for (unsigned int fpt = 1; fpt < nFpts; fpt++)
      {
        if (fpt == 3)
          divF_spts(0, ele, n, stage) -= 2*Fcomm(fpt, ele, n);
        else
          divF_spts(0, ele, n, stage) += 2*Fcomm(fpt, ele, n);
      }

#endif

#ifdef _GPU
  compute_divF_wrapper(divF_spts_d, dF_spts_d, nSpts, nVars, nEles, 
      nDims, input->equation, stage);
  check_error();
#endif
}



void Elements::compute_Fconv()
{
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
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
    compute_Fconv_spts_AdvDiff_wrapper(F_spts_d, U_spts_d, nSpts, nEles, nDims, input->AdvDiff_A_d);
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
      for (unsigned int ele = 0; ele < nEles; ele++)
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
    compute_Fconv_spts_EulerNS_wrapper(F_spts_d, U_spts_d, nSpts, nEles, nDims, input->gamma);
    check_error();
#endif

  }

}

void Elements::compute_Fvisc()
{
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
#pragma omp parallel for collapse(4)
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
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

void Elements::compute_Uavg()
{
#ifdef _CPU
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
