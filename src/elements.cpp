/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

extern "C" {
#include "cblas.h"
}
#include "gimmik.h"

#include "elements.hpp"
#include "faces.hpp"
#include "filter.hpp"
#include "flux.hpp"
#include "funcs.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

#define _DEBUG
void Elements::setup(std::shared_ptr<Faces> faces, _mpi_comm comm_in)
{
  myComm = comm_in;

#ifdef _GPU
  nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#else
  nElesPad = nEles; //TODO: Padding for CPU
#endif

  set_locs();
  set_shape();
  set_coords(faces);
  set_normals(faces);
  set_vandermonde_mats();
  calc_transforms(faces);
  setup_FR();
  setup_aux();  

  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  U_spts.assign({nSpts, nVars, nElesPad});
  U_fpts.assign({nFpts, nVars, nElesPad});
  if (input->viscous)
    Ucomm.assign({nFpts, nVars, nElesPad});
  U_ppts.assign({nPpts, nVars, nElesPad});
  U_qpts.assign({nQpts, nVars, nElesPad});

  if (input->squeeze)
    Uavg.assign({nVars, nElesPad});

  F_spts.assign({nDims, nSpts, nVars, nElesPad});
  Fcomm.assign({nFpts, nVars, nElesPad});

  if (input->viscous)
  {
    dU_spts.assign({nDims, nSpts, nVars, nElesPad});
    dU_fpts.assign({nDims, nFpts, nVars, nElesPad});
    dU_qpts.assign({nDims, nQpts, nVars, nElesPad});
  }

  if (input->dt_scheme != "LSRK" && input->dt_scheme != "RK54")
  {
    divF_spts.assign({input->nStages, nSpts, nVars, nElesPad});
  }
  else
  {
    divF_spts.assign({1, nSpts, nVars, nElesPad});
    U_til.assign({nSpts, nVars, nElesPad});
  }

  // TODO: Use adapt_dt
  if (input->dt_scheme == "LSRK" || input->dt_scheme == "RK54" ||
      input->dt_scheme == "ESDIRK3" || input->dt_scheme == "ESDIRK4")
    rk_err.assign({nSpts, nVars, nElesPad});

  U_ini.assign({nSpts, nVars, nElesPad});
  dt.assign({nEles}, input->dt);

  /* Allocate memory for calculating time-avereaged statistics */
  if (input->tavg)
  {
    // Arrays for accumulating time averages using trapezoidal integration
    tavg_acc.assign({nSpts, nVars+nDims+1, nElesPad});
    tavg_prev.assign({nSpts, nVars+nDims+1, nElesPad});
    tavg_curr.assign({nSpts, nVars+nDims+1, nElesPad});
  }

  /* Allocate memory for implicit method data structures */
  if (input->implicit_method)
  {
    /* Data structures for constructing the implicit Jacobian */
    dFdU_spts.assign({nDims, nSpts, nVars, nVars, nEles});
    dFcdU.assign({nFpts, nVars, nVars, nEles});

    if (input->viscous)
    {
      dUcdU.assign({nFpts, nVars, nVars, nEles}, 0);

      /* Note: nDimsi: Fx, Fy // nDimsj: dUdx, dUdy */
      dFddU_spts.assign({nDims, nDims, nSpts, nVars, nVars, nEles});
      dFcddU.assign({2, nDims, nFpts, nVars, nVars, nEles});

      if (input->KPF_Jacobian)
      {
        ddUdUc.assign({nFpts, nDims, nEles}, 0);
      }
    }

    /* Temporary data structures for Jacobian on CPU */
#ifdef _CPU
    CtempSF.assign({nSpts, nFpts});
    if (input->viscous)
    {
      Cvisc0.assign({nDims, nVars, nVars, nSpts, nSpts});
      CviscN.assign({nDims, nVars, nSpts, nSpts});
      CdFddU0.assign({nDims, nVars, nVars, nSpts, nSpts});
      CdFcddU0.assign({nVars, nVars, nFpts, nSpts});

      CtempD.assign({nDims});
      CtempFS.assign({nFpts, nSpts});
      CtempFSN.assign({nFptsPerFace, nSpts});
    }
#endif

    /* Solver data structures */
    if (input->pseudo_time)
    {
      dtau.assign({nEles}, input->dtau);
      if (!input->remove_deltaU)
        U_iniNM.assign({nSpts, nVars, nElesPad});
    }
    LHS.assign({nEles, nVars, nSpts, nVars, nSpts});
    RHS.assign({nEles, nVars, nSpts});
    deltaU.assign({nEles, nVars, nSpts});
#ifdef _CPU
    if (input->linear_solver == LU)
      LU_ptrs.resize(nEles);
    else if (input->linear_solver == INV)
      LHSinv.assign({nEles, nVars, nSpts, nVars, nSpts});
    else if (input->linear_solver == SVD)
    {
      SVD_ptrs.resize(nEles);

      /* Low Rank Approximation */
      unsigned int N = nVars * nSpts;
      svd_rank = (unsigned int) (input->svd_cutoff * N);
      LHSinvD.assign({nEles, N});
      LHSinvS.assign({nEles, svd_rank});
      LHSU.assign({nEles, N, svd_rank});
      LHSV.assign({nEles, N, svd_rank});
    }
    else
      ThrowException("Linear solver not recognized!");
#endif
#ifdef _GPU
    LHS_ptrs.assign({nEles});
    RHS_ptrs.assign({nEles});
    deltaU_ptrs.assign({nEles});
    LHS_info.assign({nEles});

    if (input->linear_solver == INV)
    {
      LHSinv.assign({nEles, nVars, nSpts, nVars, nSpts});
      LHSinv_ptrs.assign({nEles});
    }
#endif
  }
}

void Elements::set_shape()
{
  // Pad set again here due to set_shape being called before setup.
#ifdef _GPU
  nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#else
  nElesPad = nEles; //TODO: Padding for CPU
#endif
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nSpts, nNodes},1);
  shape_fpts.assign({nFpts, nNodes},1);
  shape_ppts.assign({nPpts, nNodes},1);
  shape_qpts.assign({nQpts, nNodes},1);
  dshape_spts.assign({nDims, nSpts, nNodes},1);
  dshape_fpts.assign({nDims, nFpts, nNodes},1);
  dshape_qpts.assign({nDims, nQpts, nNodes},1);

  if (input->motion)
  {
    grid_vel_nodes.assign({nNodes, nDims, nEles}, 0.);
    grid_vel_spts.assign({nSpts, nDims, nEles}, 0.);
    grid_vel_fpts.assign({nFpts, nDims, nEles}, 0.);
    grid_vel_ppts.assign({nPpts, nDims, nEles}, 0.);
  }

  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nDims, nSpts, nDims, nEles});
  jaco_fpts.assign({nDims, nFpts, nDims, nEles});
  jaco_qpts.assign({nDims, nQpts, nDims, nEles});
  inv_jaco_spts.assign({nDims, nSpts, nDims, nElesPad});
  inv_jaco_fpts.assign({nDims, nFpts, nDims, nEles});
  jaco_det_spts.assign({nSpts, nElesPad});
  jaco_det_fpts.assign({nFpts, nEles});
  jaco_det_qpts.assign({nQpts, nEles});
  vol.assign({nEles});

  double loc[3] = {0., 0., 0.};
  mdvector<double> shape_val({nNodes});
  mdvector<double> dshape_val({nNodes,nDims});

  /* Shape functions and derivatives at solution points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_spts(spt,dim);


    calc_shape(shape_val, &loc[0]);
    calc_d_shape(dshape_val, &loc[0]);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(spt, node) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_spts(dim, spt, node) = dshape_val(node, dim);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_fpts(fpt,dim);

    calc_shape(shape_val, &loc[0]);
    calc_d_shape(dshape_val, &loc[0]);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(fpt, node) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_fpts(dim, fpt, node) = dshape_val(node, dim);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_ppts(ppt,dim);

    calc_shape(shape_val, &loc[0]);
    calc_d_shape(dshape_val, &loc[0]);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(ppt, node) = shape_val(node);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    calc_shape(shape_val, &loc[0]);
    calc_d_shape(dshape_val, &loc[0]);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(qpt, node) = shape_val(node);

      for (unsigned int dim = 0; dim < nDims; dim++)
        dshape_qpts(dim, qpt, node) = dshape_val(node, dim);
    }
  }
}

void Elements::set_coords(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for physical coordinates */
  coord_spts.assign({nSpts, nDims, nEles});
  coord_fpts.assign({nFpts, nDims, nEles});
  coord_ppts.assign({nPpts, nDims, nEles});
  coord_qpts.assign({nQpts, nDims, nEles});
  nodes.assign({nNodes, nDims, nEles});

  /* Setup positions of all element's shape nodes in one array */
  if (input->meshfile.find(".pyfr") != std::string::npos)
  {
    nodes = geo->ele_nodes; /// TODO: setup for Gmsh grids as well
  }
  else
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        unsigned int eleBT = ele + startEle;
        for (unsigned int node = 0; node < nNodes; node++)
        {
          nodes(node, dim, ele) = geo->coord_nodes(geo->ele2nodesBT[etype](eleBT,node), dim);
        }
      }
  }

  int ms = nSpts;
  int mf = nFpts;
  int mp = nPpts;
  int mq = nQpts;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = nodes(0,0,0);

  /* Setup physical coordinates at solution points */
  auto &As = shape_spts(0,0);
  auto &Cs = coord_spts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, n, 0.0, &Cs, n);

  /* Setup physical coordinates at flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = coord_fpts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, n, 0.0, &Cf, n);

  /* Setup physical coordinates at plot points */
  auto &Ap = shape_ppts(0,0);
  auto &Cp = coord_ppts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, n, 0.0, &Cp, n);

  /* Setup physical coordinates at quadrature points */
  auto &Aq = shape_qpts(0,0);
  auto &Cq = coord_qpts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, n, 0.0, &Cq, n);

  /* Setup physical coordinates at flux points [in faces class] */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      unsigned int eleBT = ele + startEle;
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        int gfpt = geo->fpt2gfptBT[etype](fpt,eleBT);
        int slot = geo->fpt2gfpt_slotBT[etype](fpt,eleBT);

        if (slot == 0)
        {
          faces->coord(dim, gfpt) = coord_fpts(fpt,dim,ele);
        }
      }
    }
  }

  if (input->CFL_type == 2 || input->CFL_tau_type == 2)
  {
    /* Allocate memory for tensor-line reference lengths */
    h_ref.assign({nFpts, nEles});

    /* Compute tensor-line lengths */
    if (nDims == 2)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int fpt = 0; fpt < nFpts/2; fpt++)
        {
          /* Some indexing to pair up flux points in 2D (on Quad) */
          unsigned int idx = fpt % nSpts1D;
          unsigned int fpt1 = fpt;
          unsigned int fpt2 =  (fpt / nSpts1D + 3) * nSpts1D - idx - 1;


          double dx = coord_fpts(fpt1, 0, ele) - coord_fpts(fpt2, 0, ele);
          double dy = coord_fpts(fpt1, 1, ele) - coord_fpts(fpt2, 1, ele);
          double dist = std::sqrt(dx*dx + dy*dy);

          h_ref(fpt1, ele) = dist;
          h_ref(fpt2, ele) = dist;
        }
      }
    }
    else /* nDims == 3 */
    {
      unsigned int nFptsPerFace = nSpts1D * nSpts1D;
      mdvector<double> fpts1({3, nFptsPerFace});
      mdvector<double> fpts2({3, nFptsPerFace});
      std::vector<std::vector<uint>> sortind1(3);
      std::vector<std::vector<uint>> sortind2(3);
      for (int f = 0; f < nFaces/2; f++) // btm/top, left/right, front/back
      {
        int ind1 = 2*f*nFptsPerFace;
        int ind2 = 2*f*nFptsPerFace + nFptsPerFace;
        for (int i = 0; i < nFptsPerFace; i++)
        {
          for (int d = 0; d < 3; d++)
          {
            fpts1(d,i) = loc_fpts(ind1+i,d);
            fpts2(d,i) = loc_fpts(ind2+i,d);
          }
        }

        sortind1[f] = fuzzysort(fpts1); // leave me alone, i'm lazy
        sortind2[f] = fuzzysort(fpts2); // don't feel like figuring out the map
      }

      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (int f = 0; f < nFaces/2; f++)
        {
          int ind1 = 2*f*nFptsPerFace;
          int ind2 = 2*f*nFptsPerFace + nFptsPerFace;
          for (int i = 0; i < nFptsPerFace; i++)
          {
            int fpt1 = ind1 + sortind1[f][i];
            int fpt2 = ind2 + sortind2[f][i];

            double dx[3] = {0,0,0};
            for (int d = 0; d < 3; d++)
              dx[d] = coord_fpts(fpt1,d,ele) - coord_fpts(fpt2,d,ele);

            double dist = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] * dx[2]*dx[2]);

            h_ref(fpt1, ele) = dist;
            h_ref(fpt2, ele) = dist;
          }
        }
      }
    }
  }

#ifdef _BUILD_LIB_blah
 if (input->overset)
 {
   if (etype != HEX) ThrowException("Don't use overset with non-hex grids!");

   /* For curved grids, Figure out which grid elements can be treated as linear */
   int nCorners = geo->nCornerNodesBT[etype];

   if (nCorners == 8) return; // No need to do anything

   mdvector<double> shape_ppts_lin({nCorners, nPpts});
   mdvector<double> coord_ppts_lin({nPpts, nDim, nEles});
   mdvector<double> corner_nodes({nCorners, nDims, nEles});

   /// HACK - since calc_shape() tied to element-class vars
   int tmp_nNodes = nNodes;
   int tmp_nNdSide = nNdSide;
   nNodes = nCorners;
   nNdSide = 2;

   double loc[3];
   mdvector<double> shape_val({nNodes});
   for (unsigned int ppt = 0; ppt < nPpts; ppt++)
   {
     for (unsigned int dim = 0; dim < nDims; dim++)
       loc[dim] = loc_ppts(ppt,dim);

     calc_shape(shape_val, &loc[0]);

     for (unsigned int node = 0; node < nNodes; node++)
     {
       shape_ppts_lin(node, ppt) = shape_val(node);
     }
   }

   /* Setup physical coordinates at plot points using only corner nodes*/
   auto &A = shape_ppts_lin(0,0);
   auto &B = corner_nodes(0,0,0);
   auto &C = coord_ppts_lin(0,0,0);
   int m = nPpts;
   int k = nNodes;
   int n = nEles*nDims;
   cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k,
               1.0, &A, k, &B, k, 0.0, &C, m);

   /* Tag all elements whose linearly-extrapolated plot points don't match their
    * regular plot points */
   geo->linear_tag.assign({nEles}, 1);
   for (int d = 0; d < nDims; d++)
   {
     for (int ele = 0; ele < nEles; ele++)
     {
       if (!geo->linear_tag(ele)) continue;

       for (int ppt = 0; ppt < nPpts; ppt++)
       {
         if (std::abs(coord_ppts(ppt,ele,d)-coord_ppts_lin(ppt,ele,d)) > 1e-6)
         {
           geo->linear_tag(ele) = 0;
           break;
         }
       }
     }
   }

   nNodes = tmp_nNodes;
   nNdSide = tmp_nNdSide;
 }
#endif
}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppD.assign({nDims, nSpts, nSpts});
  oppD_fpts.assign({nDims, nSpts, nFpts});
  oppDiv.assign({nSpts, nDims, nSpts});
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
          loc[d] = loc_spts(ispt, d);

        oppD(dim, ispt, jspt) = calc_d_nodal_basis_spts(jspt, loc, dim);
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

        oppD_fpts(dim, spt, fpt) = calc_d_nodal_basis_fpts(fpt, loc, dim);
      }
    }
  }

  /* Setup divergence operator (oppDiv) for solution points */
  /* Note: This is essentially the same as oppD, but with dimensions oriented in a row */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int jspt = 0; jspt < nSpts; jspt++)
    {
      for (unsigned int ispt = 0; ispt < nSpts; ispt++)
      {
        for (unsigned int d = 0; d < nDims; d++)
          loc[d] = loc_spts(ispt, d);

        oppDiv(ispt, dim, jspt) = calc_d_nodal_basis_spts(jspt, loc, dim);
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
      if (etype == QUAD || etype == HEX)
      {
        if (nDims == 2) 
        {
          int face = fpt / nSpts1D;
          if (face == 0 or face == 3) // Bottom and Left face
            fac = -1;
        }
        else if (nDims == 3)
        {
          int face = fpt / (nSpts1D * nSpts1D);
          if (face % 2 == 0) // Bottom, Left, and Front face
            fac = -1;
        }
      }

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        oppDiv_fpts(spt, fpt) += fac * oppD_fpts(dim, spt, fpt);
      }
    }
  }

#ifndef _NO_GIMMIK
  /* Setup operator ids for GiMMiK (ids = 0 by default to disable gimmik) */
  std::stringstream ss;

  ss << "oppE_P" << order << "_" << nDims << "D_e" << etype;
  std::string str = ss.str(); ss.str(std::string());
  oppE_id = hash_str(str.data()); 
  write_opp(oppE, str, oppE_id, nFpts, nSpts);

  ss << "oppD_P" << order << "_" << nDims << "D_e" << etype;
  str = ss.str(); ss.str(std::string());
  oppD_id = hash_str(str.data()); 
  write_opp(oppD, str, oppD_id, nDims * nSpts, nSpts);

  ss << "oppD_fpts_P" << order << "_" << nDims << "D_e" << etype;
  str = ss.str(); ss.str(std::string());
  oppD_fpts_id = hash_str(str.data()); 
  write_opp(oppD_fpts, str, oppD_fpts_id, nDims * nSpts, nFpts);

  ss << "oppDiv_P" << order << "_" << nDims << "D_e" << etype;
  str = ss.str(); ss.str(std::string());
  oppDiv_id = hash_str(str.data()); 
  write_opp(oppDiv, str, oppDiv_id, nSpts, nSpts * nDims);

  ss << "oppDiv_fpts_P" << order << "_" << nDims << "D_e" << etype;
  str = ss.str(); ss.str(std::string());
  oppDiv_fpts_id = hash_str(str.data()); 
  write_opp(oppDiv_fpts, str, oppDiv_fpts_id, nSpts, nFpts);
#endif

  /* Setup combined differentiation/extrapolation Jacobian operators (DFR Specific) */
  if (input->implicit_method && input->KPF_Jacobian)
  {
    oppD_spts1D.assign({nSpts1D, nSpts1D});
    for (unsigned int spti = 0; spti < nSpts1D; spti++)
      for (unsigned int sptj = 0; sptj < nSpts1D; sptj++)
      {
        oppD_spts1D(spti, sptj) = Lagrange_d1(loc_DFR_1D, sptj+1, loc_spts_1D[spti]);
      }

    oppDE_spts1D.assign({2, nSpts1D, nSpts1D});
    for (unsigned int spti = 0; spti < nSpts1D; spti++)
      for (unsigned int sptj = 0; sptj < nSpts1D; sptj++)
      {
        oppDE_spts1D(0, spti, sptj) =
          Lagrange_d1(loc_DFR_1D, 0, loc_spts_1D[spti]) *
          Lagrange(loc_spts_1D, sptj, -1);
        oppDE_spts1D(1, spti, sptj) = 
          Lagrange_d1(loc_DFR_1D, nSpts1D+1, loc_spts_1D[spti]) *
          Lagrange(loc_spts_1D, sptj,  1);
      }

    oppDivE_spts1D.assign({2, nSpts1D, nSpts1D});
    for (unsigned int spti = 0; spti < nSpts1D; spti++)
      for (unsigned int sptj = 0; sptj < nSpts1D; sptj++)
      {
        oppDivE_spts1D(0, spti, sptj) = -
          Lagrange_d1(loc_DFR_1D, 0, loc_spts_1D[spti]) *
          Lagrange(loc_spts_1D, sptj, -1);
        oppDivE_spts1D(1, spti, sptj) = 
          Lagrange_d1(loc_DFR_1D, nSpts1D+1, loc_spts_1D[spti]) *
          Lagrange(loc_spts_1D, sptj,  1);
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

void Elements::calc_transforms(std::shared_ptr<Faces> faces)
{
  /* --- Calculate Transformation at Solution Points --- */

  int ms = nSpts * nDims;
  int mf = nFpts * nDims;
  int mq = nQpts * nDims;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = nodes(0,0,0);
  auto &As = dshape_spts(0, 0, 0);
  auto &Af = dshape_fpts(0, 0, 0);
  auto &Aq = dshape_qpts(0, 0, 0);
  auto &Cs = jaco_spts(0, 0, 0, 0);
  auto &Cf = jaco_fpts(0, 0, 0, 0);
  auto &Cq = jaco_qpts(0, 0, 0, 0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, n, 0.0, &Cs, n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, n, 0.0, &Cf, n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, n, 0.0, &Cq, n);

  set_inverse_transforms(jaco_spts,inv_jaco_spts,jaco_det_spts,nSpts,nDims);
  
  mdvector<double> nullvec;
  set_inverse_transforms(jaco_fpts,inv_jaco_fpts,nullvec,nFpts,nDims);

  /* --- Compute Element Volumes --- */
  for (unsigned int e = 0; e < nEles; e++)
  {
    vol(e) = 0.0;

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      vol(e) += weights_spts(spt) * jaco_det_spts(spt, e);
    }
  }

  /* --- Calculate Transformation at Flux Points --- */
  /* If using modified gradient algorithm, store normals and dA in element local structures */
  if (input->grad_via_div)
  {
    norm_fpts.assign({nDims, nFpts, nEles}, 0);
    dA_fpts.assign({nFpts, nEles}, 0);
  }

  for (unsigned int e = 0; e < nEles; e++)
  {
    unsigned int eBT = e + startEle;
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfptBT[etype](fpt,eBT);

      unsigned int slot = geo->fpt2gfpt_slotBT[etype](fpt,eBT);

      /* --- Calculate outward unit normal vector at flux point ("left" element only) --- */
      double norm[3] = {0.0};
      // Transform face normal from reference to physical space [JGinv .dot. tNorm]
      for (uint dim1 = 0; dim1 < nDims; dim1++)
      {
        for (uint dim2 = 0; dim2 < nDims; dim2++)
          norm[dim1] += inv_jaco_fpts(dim2, fpt, dim1, e) * tnorm(fpt,dim2);
      }

      if (slot == 0)
      {
        for (uint dim = 0; dim < nDims; dim++)
          faces->norm(dim, gfpt) = norm[dim]; 
      }

      // Store magnitude of face normal (equivalent to face area in finite-volume land)
      faces->dA(slot, gfpt) = 0;
      for (uint dim = 0; dim < nDims; dim++)
        faces->dA(slot, gfpt) += norm[dim]*norm[dim];

      faces->dA(slot, gfpt) = sqrt(faces->dA(slot, gfpt));

      // Normalize
      // If we have a collapsed edge, the dA will be 0, so just set the normal to 0
      // (A normal vector at a point doesn't make sense anyways)
      //if (std::fabs(faces->dA(gfpt, slot)) < 1e-10)
      //{
      //  faces->dA(gfpt, slot) = 0.;
      //  for (uint dim = 0; dim < nDims; dim++)
      //    faces->norm(gfpt,dim) = 0;
      //}
      //else
      //{
      //  for (uint dim = 0; dim < nDims; dim++)
      //    faces->norm(gfpt,dim) /= faces->dA(gfpt);
      //}
      
      if (slot == 0)
      {
        for (uint dim = 0; dim < nDims; dim++)
          faces->norm(dim, gfpt) /= faces->dA(0, gfpt);
      }

      if (input->grad_via_div)
      {
        for (uint dim = 0; dim < nDims; dim++)
          dA_fpts(fpt, e) += norm[dim] * norm[dim];

        dA_fpts(fpt, e) = sqrt(dA_fpts(fpt, e));

        for (uint dim = 0; dim < nDims; dim++)
        {
          norm_fpts(dim, fpt, e) = norm[dim] / dA_fpts(fpt, e);
        }
      }
    }
  }

  /* Set jacobian matrix and determinant at quadrature points */
  for (unsigned int e = 0; e < nEles; e++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      if (nDims == 2)
      {
        // Determinant of transformation matrix
        jaco_det_qpts(qpt, e) = jaco_qpts(0, qpt, 0, e)*jaco_qpts(1, qpt, 1, e)-jaco_qpts(0, qpt, 1, e)*jaco_qpts(1, qpt, 0, e);
      }
      else if (nDims == 3)
      {
        double xr = jaco_qpts(0, qpt, 0, e);   double xs = jaco_qpts(1, qpt, 0, e);   double xt = jaco_qpts(2, qpt, 0, e);
        double yr = jaco_qpts(0, qpt, 1, e);   double ys = jaco_qpts(1, qpt, 1, e);   double yt = jaco_qpts(2, qpt, 1, e);
        double zr = jaco_qpts(0, qpt, 2, e);   double zs = jaco_qpts(1, qpt, 2, e);   double zt = jaco_qpts(2, qpt, 2, e);
        jaco_det_qpts(qpt,e) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);
      }
      if (jaco_det_qpts(qpt, e) < 0) ThrowException("Negative Jacobian at quadrature point.");
    }
  }
}

void Elements::set_inverse_transforms(const mdvector<double> &jaco,
               mdvector<double> &inv_jaco, mdvector<double> &jaco_det,
               unsigned int nPts, unsigned int nDims)
{
  for (unsigned int e = 0; e < nEles; e++)
  {
    for (unsigned int pt = 0; pt < nPts; pt++)
    {
      if (nDims == 2)
      {
        // Determinant of transformation matrix
        if (jaco_det.size()) jaco_det(pt, e) = jaco(0,pt,0,e) * jaco(1,pt,1,e) - jaco(0,pt,1,e) * jaco(1,pt,0,e);

        // Inverse of transformation matrix (times its determinant)
        inv_jaco(0,pt,0,e) = jaco(1,pt,1,e);  inv_jaco(0,pt,1,e) =-jaco(1,pt,0,e);
        inv_jaco(1,pt,0,e) =-jaco(0,pt,1,e);  inv_jaco(1,pt,1,e) = jaco(0,pt,0,e);
      }
      else if (nDims == 3)
      {
        double xr = jaco(0,pt,0,e);   double xs = jaco(1,pt,0,e);   double xt = jaco(2,pt,0,e);
        double yr = jaco(0,pt,1,e);   double ys = jaco(1,pt,1,e);   double yt = jaco(2,pt,1,e);
        double zr = jaco(0,pt,2,e);   double zs = jaco(1,pt,2,e);   double zt = jaco(2,pt,2,e);
        if (jaco_det.size()) jaco_det(pt,e) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

        inv_jaco(0,pt,0,e) = ys*zt - yt*zs;  inv_jaco(0,pt,1,e) = xt*zs - xs*zt;  inv_jaco(0,pt,2,e) = xs*yt - xt*ys;
        inv_jaco(1,pt,0,e) = yt*zr - yr*zt;  inv_jaco(1,pt,1,e) = xr*zt - xt*zr;  inv_jaco(1,pt,2,e) = xt*yr - xr*yt;
        inv_jaco(2,pt,0,e) = yr*zs - ys*zr;  inv_jaco(2,pt,1,e) = xs*zr - xr*zs;  inv_jaco(2,pt,2,e) = xr*ys - xs*yr;
      }

      if (jaco_det.size() and jaco_det(pt,e) < 0) ThrowException("Negative Jacobian at solution points.");
    }
  }
}

void Elements::initialize_U()
{
  /* Initialize solution */
  if (input->equation == AdvDiff)
  {
    if (input->ic_type == 0)
    {
      // Set to zero
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          U_spts(spt, 0, ele) = 0.0;
        }
      }
    }
    else if (input->ic_type == 1)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          double x = coord_spts(spt, 0, ele);
          double y = coord_spts(spt, 1, ele);
          double z = (nDims == 2) ? 0.0 : coord_spts(spt, 2, ele);

          U_spts(spt, 0, ele) = compute_U_init(x, y, z, 0, input);
        }
      }
    }
    else
    {
      ThrowException("ic_type not recognized!");
    }
  }
  else if (input->equation == EulerNS)
  {
    if (input->ic_type == 0)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          U_spts(spt, 0, ele)  = input->rho_fs;

          double Vsq = 0.0;
          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            U_spts(spt, dim+1, ele)  = input->rho_fs * input->V_fs(dim);
            Vsq += input->V_fs(dim) * input->V_fs(dim);
          }

          U_spts(spt, nDims + 1, ele)  = input->P_fs/(input->gamma-1.0) +
            0.5*input->rho_fs * Vsq;
        }
      }

    }
    else if (input->ic_type == 1)
    {
      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          for (unsigned int spt = 0; spt < nSpts; spt++)
          {
            double x = coord_spts(spt, 0, ele);
            double y = coord_spts(spt, 1, ele);
            double z = (nDims == 2) ? 0.0 : coord_spts(spt, 2, ele);

            U_spts(spt, n, ele) = compute_U_init(x, y, z, n, input);
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

void Elements::setup_filter()
{
  /* Compute 1D vandermonde matrices and inverse */
  mdvector<double> vand1D({nSpts1D, order + 1});
  for (unsigned int j = 0; j <= order; j++)
  {
    double normC = std::sqrt(2.0 / (2.0 * j + 1.0));
    for (unsigned int spt = 0; spt < nSpts1D; spt++)
      vand1D(spt, j) = Legendre(j, loc_spts_1D[spt]) / normC;
  }
  
  mdvector<double> inv_vand1D({nSpts1D, order + 1});
  vand1D.inverse(inv_vand1D);
  
  /* Compute 1D vandermonde matrix for the derivative */
  mdvector<double> vand1D_d1({nSpts1D, order + 1});
  for (unsigned int j = 0; j <= order; j++)
  {
    double normC = std::sqrt(2.0 / (2.0 * j + 1.0));
    for (unsigned int spt = 0; spt < nSpts1D; spt++)
      vand1D_d1(spt, j) = Legendre_d1(j, loc_spts_1D[spt]) / normC;
  }

  /* Form concentration sensor matrix */
  // Form 1D operator for tensor product line
  oppS_1D.assign({nSpts1D, order + 1});
  mdvector<double> conc({nSpts1D, order + 1}, 0.0);
  
  if (order > 0) 
  {
    for (unsigned int j = 0; j <= order; j++)
    {
      for (unsigned int spt = 0; spt < nSpts1D; spt++)
      {
        double x = loc_spts_1D[spt];
        conc(spt, j) = (M_PI / order) * std::sqrt(1.0 - x*x) * vand1D_d1(spt, j);
      }
    }
  }

  auto &A = conc(0, 0);
  auto &B = inv_vand1D(0, 0);
  auto &C = oppS_1D(0, 0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
    nSpts1D, nSpts1D, nSpts1D, 1.0, &A, nSpts1D, &B, nSpts1D, 0.0, &C, nSpts1D);

  // Form multidimensional operator
  // Note: This operator is constructed for tensor product elements. Application to tris/tets requires
  // additional multiplication by interpolation operator to "collapsed" tensor product solution points, done later.
  if (nDims == 2) // Quads, Tris
  {
    unsigned int nSpts2D = nSpts1D * nSpts1D;
    oppS.assign({nDims * nSpts2D, nSpts2D});

    // xi lines
    for (unsigned int k = 0; k < nSpts1D; k++)
      for (unsigned int j = 0; j < nSpts1D; j++)
        for (unsigned int i = 0; i < nSpts1D; i++)
          oppS(i + k*nSpts1D, j + k*nSpts1D) = oppS_1D(i,j);
      
    // eta lines
    for (unsigned int k = 0; k < nSpts1D; k++)
      for (unsigned int j = 0; j < nSpts1D; j++)
        for (unsigned int i = 0; i < nSpts1D; i++)
          oppS(nSpts2D + i + k*nSpts1D, j*nSpts1D + k) = oppS_1D(i,j);   
  } 
  else // Hexes, Tets
  {
    unsigned int nSpts2D = nSpts1D * nSpts1D;
    unsigned int nSpts3D = nSpts1D * nSpts1D * nSpts1D;
    oppS.assign({nDims * nSpts3D, nSpts3D});

    // xi lines
    for (unsigned int k = 0; k < nSpts2D; k++)
      for (unsigned int j = 0; j < nSpts1D; j++)
        for (unsigned int i = 0; i < nSpts1D; i++)
          oppS(i + k*nSpts1D, j + k*nSpts1D) = oppS_1D(i,j);

    // eta lines
    for(unsigned int l = 0; l < nSpts1D; l++)
      for (unsigned int k = 0; k < nSpts1D; k++)
        for (unsigned int j = 0; j < nSpts1D; j++)
          for (unsigned int i = 0; i < nSpts1D; i++)
            oppS(nSpts3D + i + k*nSpts1D + l*nSpts2D, j*nSpts1D + k + l*nSpts2D) = oppS_1D(i,j);

    // zeta lines
    for(unsigned int l = 0; l < nSpts1D; l++)
      for (unsigned int k = 0; k < nSpts1D; k++)
        for (unsigned int j = 0; j < nSpts1D; j++)
          for (unsigned int i = 0; i < nSpts1D; i++)
            oppS(2*nSpts3D + i + k*nSpts1D + l*nSpts2D, j*nSpts2D + k + l*nSpts1D) = oppS_1D(i,j);
  }

  modify_sensor(); // multiply by additional matrix for tri/tet

  /* Form exponential filter matrix */
  oppF.assign({nSpts, nSpts});
  mdvector<double> temp({nSpts, nSpts});

  for (unsigned int i = 0; i < nSpts; i++)
  {
    double sigma = calc_expfilter_coeffs(order, nSpts, input->alpha, input->filtexp, nDims, i);

    for (unsigned int j = 0; j < nSpts; j++)
      temp(i,j) = sigma * inv_vand(i,j);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts, nSpts, nSpts, 1.0, 
      vand.data(), nSpts, temp.data(), nSpts, 0.0, oppF.data(), nSpts);

#ifdef _GPU
  /* Copy operators to GPU */
  oppS_d = oppS;
  oppF_d = oppF;
#endif

}

void Elements::setup_ddUdUc()
{
  /* Setup fpt2spts */
  mdvector<unsigned int> fpt2spts({nFpts, nSpts1D});
  for (unsigned int face = 0; face < nFaces; face++)
  {
    if (etype == QUAD)
    {
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
      {
        unsigned int fpt = face*nFptsPerFace + fi;
        for (unsigned int sk = 0; sk < nSpts1D; sk++)
        {
          switch(face)
          {
            case 0: /* Bottom edge */
              fpt2spts(fpt, sk) = sk*nSpts1D + fi; break;

            case 1: /* Right edge */
              fpt2spts(fpt, sk) = fi*nSpts1D + sk; break;

            case 2: /* Upper edge */
              fpt2spts(fpt, sk) = sk*nSpts1D + (nSpts1D-fi-1); break;

            case 3: /* Left edge */
              fpt2spts(fpt, sk) = (nSpts1D-fi-1)*nSpts1D + sk; break;
          }
        }
      }
    }
    else if (etype == HEX)
    {
      for (unsigned int fi = 0; fi < nSpts1D; fi++)
      {
        for (unsigned int fj = 0; fj < nSpts1D; fj++)
        {
          unsigned int fpt = face*nFptsPerFace + fi*nSpts1D + fj;
          for (unsigned int sk = 0; sk < nSpts1D; sk++)
          {
            switch(face)
            {
              case 0: /* Bottom face */
                fpt2spts(fpt, sk) = sk*nFptsPerFace + fi*nSpts1D + fj; break;

              case 1: /* Top face */
                fpt2spts(fpt, sk) = sk*nFptsPerFace + fi*nSpts1D + (nSpts1D-fj-1); break;

              case 2: /* Left face */
                fpt2spts(fpt, sk) = fi*nFptsPerFace + fj*nSpts1D + sk; break;

              case 3: /* Right face */
                fpt2spts(fpt, sk) = fi*nFptsPerFace + (nSpts1D-fj-1)*nSpts1D + sk; break;

              case 4: /* Front face */
                fpt2spts(fpt, sk) = fi*nFptsPerFace + sk*nSpts1D + (nSpts1D-fj-1); break;

              case 5: /* Back face */
                fpt2spts(fpt, sk) = fi*nFptsPerFace + sk*nSpts1D + fj; break;
            }
          }
        }
      }
    }
    else
      ThrowException("fpt2spts should only be setup for QUAD or HEX!")
  }

  /* Setup ddUdUc */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    unsigned int eleID = geo->eleID[etype](ele + startEle);
    for (unsigned int face = 0; face < nFaces; face++)
    {
      /* Note: No contribution if element neighbor is a boundary */
      int eleNID = geo->ele2eleN(face, eleID);
      if (eleNID == -1) continue;

      if (etype == QUAD)
      {
        /* Determine neighbor's LR state and outward dimension on each face */
        int faceN = geo->face2faceN(face, eleID);
        unsigned int LR = (faceN == 0 || faceN == 3) ? 0 : 1;
        unsigned int dimN = (faceN == 1 || faceN == 3) ? 0 : 1;

        /* Compute inner product on each fpt */
        for (unsigned int fi = 0; fi < nSpts1D; fi++)
        {
          unsigned int fpt = face * nFptsPerFace + fi;
          int fptN = geo->fpt2fptN(fpt, eleID);

          /* Compute inner product */
          for (unsigned int dimj = 0; dimj < nDims; dimj++)
          {
            double val = 0.0;
            for (unsigned int sk = 0; sk < nSpts1D; sk++)
            {
              unsigned int sptk = fpt2spts(fptN, sk);
              val += oppDE_spts1D(LR, sk, sk) * inv_jacoN_spts(face, dimN, sptk, dimj, ele) / jacoN_det_spts(face, sptk, ele);
            }
            ddUdUc(fpt, dimj, ele) = val;
          }
        }
      }
      else if (etype == HEX)
      {
        /* Determine neighbor's LR state and outward dimension on each face */
        int faceN = geo->face2faceN(face, eleID);
        unsigned int LR = (faceN == 0 || faceN == 2 || faceN == 4) ? 0 : 1;
        unsigned int dimN = (faceN == 2 || faceN == 3) ? 0 : ((faceN == 4 || faceN == 5) ? 1 : 2);

        /* Compute inner product on each fpt */
        for (unsigned int fi = 0; fi < nSpts1D; fi++)
        {
          for (unsigned int fj = 0; fj < nSpts1D; fj++)
          {
            unsigned int fpt = face*nFptsPerFace + fi*nSpts1D + fj;
            int fptN = geo->fpt2fptN(fpt, eleID);

            /* Compute inner product */
            for (unsigned int dimj = 0; dimj < nDims; dimj++)
            {
              double val = 0.0;
              for (unsigned int sk = 0; sk < nSpts1D; sk++)
              {
                unsigned int sptk = fpt2spts(fptN, sk);
                val += oppDE_spts1D(LR, sk, sk) * inv_jacoN_spts(face, dimN, sptk, dimj, ele) / jacoN_det_spts(face, sptk, ele);
              }
              ddUdUc(fpt, dimj, ele) = val;
            }
          }
        }
      }
      else
        ThrowException("ddUdUc should only be setup for QUAD or HEX!")
    }
  }
}


void Elements::extrapolate_U()
{
#ifdef _CPU
  auto &A = oppE(0,0);
  auto &B = U_spts(0, 0, 0);
  auto &C = U_fpts(0, 0, 0);

  gimmik_mm_cpu(nFpts, nElesPad * nVars, nSpts, 1.0, &A, nSpts, &B, nElesPad * nVars,
      0.0, &C, nElesPad * nVars, oppE_id);
#endif

#ifdef _GPU
  auto *A = oppE_d.data();
  auto *B = U_spts_d.data();
  auto *C = U_fpts_d.data();
  //cublasDGEMM_wrapper(nEles * nVars, nFpts, nSpts, 1.0,
  //    B, nEles * nVars, A, nSpts, 0.0, C, nEles * nVars);
  gimmik_mm_gpu(nFpts, nElesPad * nVars, nSpts, 1.0, A, nSpts, B, nElesPad * nVars, 
      0.0, C, nElesPad * nVars, oppE_id);

  check_error();
#endif

}

void Elements::extrapolate_dU()
{
#ifdef _CPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    auto &A = oppE(0,0);
    auto &B = dU_spts(dim, 0, 0, 0);
    auto &C = dU_fpts(dim, 0, 0, 0);

    gimmik_mm_cpu(nFpts, nElesPad * nVars, nSpts, 1.0, &A, nSpts, &B, nElesPad * nVars,
        0.0, &C, nElesPad * nVars, oppE_id);
  }
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    auto *A = oppE_d.get_ptr(0,0);
    auto *B = dU_spts_d.get_ptr(dim, 0, 0, 0);
    auto *C = dU_fpts_d.get_ptr(dim, 0, 0, 0);

    //cublasDGEMM_wrapper(nEles * nVars, nFpts, nSpts, 1.0, B, nEles * nVars,
    //    A, nSpts, 0.0, C, nEles * nVars);
    gimmik_mm_gpu(nFpts, nElesPad * nVars, nSpts, 1.0, A, nSpts, B, nElesPad * nVars, 
        0.0, C, nElesPad * nVars, oppE_id);
  }

  check_error();
#endif
}

void Elements::compute_dU_spts()
{
#ifdef _CPU
  /* Compute contribution to derivative from solution at solution points */
  auto &A = oppD(0, 0, 0);
  auto &B = U_spts(0, 0, 0);
  auto &C = dU_spts(0, 0, 0, 0);

  gimmik_mm_cpu(nSpts * nDims, nElesPad * nVars, nSpts, 1.0, &A, nSpts, &B, nElesPad * nVars,
      0.0, &C, nElesPad * nVars, oppD_id);
#endif

#ifdef _GPU
  auto *A = oppD_d.get_ptr(0, 0, 0);
  auto *B = U_spts_d.get_ptr(0, 0, 0);
  auto *C = dU_spts_d.get_ptr(0, 0, 0, 0);

  /* Compute contribution to derivative from solution at solution points */
  //cublasDGEMM_wrapper(nEles * nVars, nSpts * nDims, nSpts, 1.0, B, nEles * nVars, 
  //    A, nSpts, 0.0, C, nEles * nVars);
  gimmik_mm_gpu(nSpts * nDims, nElesPad * nVars, nSpts, 1.0, A, nSpts, B, nElesPad * nVars, 
      0.0, C, nElesPad * nVars, oppD_id);

  check_error();
#endif

}

void Elements::compute_dU_fpts()
{
#ifdef _CPU
  /* Compute contribution to derivative from common solution at flux points */
  auto &A = oppD_fpts(0, 0, 0);
  auto &B = Ucomm(0, 0, 0);
  auto &C = dU_spts(0, 0, 0, 0);

  gimmik_mm_cpu(nSpts * nDims, nElesPad * nVars, nFpts, 1.0, &A, nFpts, &B, nElesPad * nVars,
      1.0, &C, nElesPad * nVars, oppD_fpts_id);
#endif

#ifdef _GPU
  auto *A = oppD_fpts_d.get_ptr(0, 0, 0);
  auto *B = Ucomm_d.get_ptr(0, 0, 0);
  auto *C = dU_spts_d.get_ptr(0, 0, 0, 0);

  /* Compute contribution to derivative from common solution at flux points */
  //cublasDGEMM_wrapper(nEles * nVars, nSpts * nDims, nFpts, 1.0,
  //    B, nEles * nVars, A, nFpts, 1.0, C, nEles * nVars);
  gimmik_mm_gpu(nSpts * nDims, nElesPad * nVars, nFpts, 1.0, A, nFpts, B, nElesPad * nVars, 
      1.0, C, nElesPad * nVars, oppD_fpts_id);

  check_error();
#endif

}

void Elements::compute_divF_spts(unsigned int stage)
{
#ifdef _CPU
  /* Compute contribution to divergence from flux at solution points */
  auto &A = oppDiv(0, 0, 0);
  auto &B = F_spts(0, 0, 0, 0);
  auto &C = divF_spts(stage, 0, 0, 0);

  gimmik_mm_cpu(nSpts, nElesPad * nVars, nSpts * nDims, 1.0, &A, nSpts * nDims, &B, nElesPad * nVars,
      0.0, &C, nElesPad * nVars, oppDiv_id);
#endif

#ifdef _GPU

  auto *A = oppDiv_d.get_ptr(0, 0, 0);
  auto *B = F_spts_d.get_ptr(0, 0, 0, 0);
  auto *C = divF_spts_d.get_ptr(stage, 0, 0, 0);

  /* Compute contribution to derivative from solution at solution points */
  //cublasDGEMM_wrapper(nEles * nVars, nSpts, nSpts * nDims, 1.0,
  //    B, nEles * nVars, A, nSpts * nDims, 0.0, C, nEles * nVars, 0);
  gimmik_mm_gpu(nSpts, nElesPad * nVars, nSpts * nDims, 1.0, A, nSpts * nDims, B, nElesPad * nVars, 
      0.0, C, nElesPad * nVars, oppDiv_id);
  check_error();
#endif
}

void Elements::compute_divF_fpts(unsigned int stage)
{
#ifdef _CPU
  /* Compute contribution to divergence from common flux at flux points */
  auto &A = oppDiv_fpts(0, 0);
  auto &B = Fcomm(0, 0, 0);
  auto &C = divF_spts(stage, 0, 0, 0);

  gimmik_mm_cpu(nSpts, nElesPad * nVars, nFpts, 1.0, &A, nFpts, &B, nElesPad * nVars,
      1.0, &C, nElesPad * nVars, oppDiv_fpts_id);
#endif

#ifdef _GPU
  /* Compute contribution to derivative from common solution at flux points */
  auto *A = oppDiv_fpts_d.get_ptr(0, 0);
  auto *B = Fcomm_d.get_ptr(0, 0, 0);
  auto *C = divF_spts_d.get_ptr(stage, 0, 0, 0);

  //cublasDGEMM_wrapper(nEles * nVars, nSpts,  nFpts, 1.0,
  //    B, nEles * nVars, A, nFpts, 1.0, C, nEles * nVars, 0);
  gimmik_mm_gpu(nSpts, nElesPad * nVars, nFpts, 1.0, A, nFpts, B, nElesPad * nVars, 
      1.0, C, nElesPad * nVars, oppDiv_fpts_id);

  check_error();
#endif
}

void Elements::add_source(unsigned int stage, double flow_time)
{
#ifdef _CPU
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        if (input->overset && geo->iblank_cell(ele) != NORMAL) continue;
          double x = coord_spts(spt, 0, ele);
          double y = coord_spts(spt, 1, ele);
          double z = 0;
          if (nDims == 3)
            z = coord_spts(spt, 2, ele);

          divF_spts(stage, spt, n, ele) += compute_source_term(x, y, z, flow_time, n, input) * 
            jaco_det_spts(spt, ele);
      }
    }
  }

#endif

#ifdef _GPU
  add_source_wrapper(divF_spts_d, jaco_det_spts_d, coord_spts_d, nSpts, nEles,
      nVars, nDims, input->equation, flow_time, stage);
  check_error();
#endif
}

void Elements::compute_dU_spts_via_divF(unsigned int dim)
{
#ifdef _CPU
  /* Compute contribution to divergence from flux at solution points */
  auto &A = oppDiv(0, 0, 0);
  auto &B = F_spts(0, 0, 0, 0);
  auto &C = dU_spts(dim, 0, 0, 0);

  gimmik_mm_cpu(nSpts, nElesPad * nVars, nSpts * nDims, 1.0, &A, nSpts * nDims, &B, nElesPad * nVars,
      0.0, &C, nElesPad * nVars, oppDiv_id);
#endif

#ifdef _GPU
  auto *A = oppDiv_d.get_ptr(0, 0, 0);
  auto *B = F_spts_d.get_ptr(0, 0, 0, 0);
  auto *C = dU_spts_d.get_ptr(dim, 0, 0, 0);

  /* Compute contribution to derivative from solution at solution points */
  //cublasDGEMM_wrapper(nEles * nVars, nSpts, nSpts * nDims, 1.0,
  //    B, nEles * nVars, A, nSpts * nDims, 0.0, C, nEles * nVars);
  gimmik_mm_gpu(nSpts, nElesPad * nVars, nSpts * nDims, 1.0, A, nSpts * nDims, B, nElesPad * nVars, 
      0.0, C, nElesPad * nVars, oppDiv_id);
  check_error();
#endif
}

void Elements::compute_dU_fpts_via_divF(unsigned int dim)
{
#ifdef _CPU
  /* Compute contribution to divergence from common flux at flux points */
  auto &A = oppDiv_fpts(0, 0);
  auto &B = Fcomm(0, 0, 0);
  auto &C = dU_spts(dim, 0, 0, 0);

  gimmik_mm_cpu(nSpts, nElesPad * nVars, nFpts, 1.0, &A, nFpts, &B, nElesPad * nVars,
      1.0, &C, nElesPad * nVars, oppDiv_fpts_id);
#endif

#ifdef _GPU
  /* Compute contribution to derivative from common solution at flux points */
  auto *A = oppDiv_fpts_d.get_ptr(0, 0);
  auto *B = Fcomm_d.get_ptr(0, 0, 0);
  auto *C = dU_spts_d.get_ptr(dim, 0, 0, 0);

  //cublasDGEMM_wrapper(nEles * nVars, nSpts,  nFpts, 1.0,
  //    B, nEles * nVars, A, nFpts, 1.0, C, nEles * nVars);
  gimmik_mm_gpu(nSpts, nElesPad * nVars, nFpts, 1.0, A, nFpts, B, nElesPad * nVars, 
      1.0, C, nElesPad * nVars, oppDiv_fpts_id);

  check_error();
#endif

}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Elements::compute_F()
{
  double U[nVars];
  double F[nVars][nDims];
  double tdU[nVars][nDims];

  double Vg[nDims] = {0.0}; // Only updated on moving grids

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {

      /* Get state variables and reference space gradients */
      for (unsigned int var = 0; var < nVars; var++)
      {
        U[var] = U_spts(spt, var, ele);

        if(input->viscous) 
        {
          for(unsigned int dim = 0; dim < nDims; dim++)
          {
            tdU[var][dim] = dU_spts(dim, spt, var, ele);
          }
        }
      }

      /* Get grid velocity */
      if (input->motion)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          Vg[dim] = grid_vel_spts(spt, dim, ele);
      }

      /* Get metric terms */
      double inv_jaco[nDims][nDims];

      for (int dim1 = 0; dim1 < nDims; dim1++)
        for (int dim2 = 0; dim2 < nDims; dim2++)
          inv_jaco[dim1][dim2] = inv_jaco_spts(dim1, spt, dim2, ele);

      double dU[nVars][nDims] = {{0.0}};

      if (input->viscous)
      {
        double inv_jaco_det = 1.0 / jaco_det_spts(spt,ele);

        /* Transform gradient to physical space */
        for (unsigned int var = 0; var < nVars; var++)
        {
          for (int dim1 = 0; dim1 < nDims; dim1++)
          {
            if (!input->grad_via_div)
            {
              for (int dim2 = 0; dim2 < nDims; dim2++)
              {
                dU[var][dim1] += (tdU[var][dim2] * inv_jaco[dim2][dim1]);
              }

              dU[var][dim1] *= inv_jaco_det;

            }
            else
            {
              dU[var][dim1] = tdU[var][dim1] * inv_jaco_det;
            }

            /* Write physical gradient to global memory */
            dU_spts(dim1, spt, var, ele) = dU[var][dim1];

          }
        }
      }

      /* Compute fluxes */
      if (equation == AdvDiff)
      {
        double A[nDims];
        for(unsigned int dim = 0; dim < nDims; dim++)
          A[dim] = input->AdvDiff_A(dim);

        compute_Fconv_AdvDiff<nVars, nDims>(U, F, A, Vg);
        if(input->viscous) compute_Fvisc_AdvDiff_add<nVars, nDims>(dU, F, input->AdvDiff_D);

      }
      else if (equation == EulerNS)
      {
        double P;
        compute_Fconv_EulerNS<nVars, nDims>(U, F, Vg, P, input->gamma);
        if(input->viscous) compute_Fvisc_EulerNS_add<nVars, nDims>(U, dU, F, input->gamma, input->prandtl, input->mu,
            input->rt, input->c_sth, input->fix_vis);
      }

      /* Transform flux to reference space */
      double tF[nVars][nDims] = {{0.0}};;

      for (unsigned int var = 0; var < nVars; var++)
      {
        for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
        {
          for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
          {
            tF[var][dim1] += F[var][dim2] * inv_jaco[dim1][dim2];
          }
        }
      }

      /* Write out transformed fluxes */
      for (unsigned int var = 0; var < nVars; var++)
      {
        for(unsigned int dim = 0; dim < nDims; dim++)
        {
          F_spts(dim, spt, var, ele) = tF[var][dim];
        }
      }
    }
  }

}

void Elements::compute_F()
{
#ifdef _CPU
  if (input->equation == AdvDiff)
  {
    if (nDims == 2)
      compute_F<1, 2, AdvDiff>();
    else if (nDims == 3)
      compute_F<1, 3, AdvDiff>();
  }
  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
      compute_F<4, 2, EulerNS>();
    else if (nDims == 3)
      compute_F<5, 3, EulerNS>();
  }
#endif

#ifdef _GPU
  compute_F_wrapper(F_spts_d, U_spts_d, dU_spts_d, grid_vel_spts_d, inv_jaco_spts_d, jaco_det_spts_d, nSpts, nEles, nDims, input->equation, input->AdvDiff_A_d,
      input->AdvDiff_D, input->gamma, input->prandtl, input->mu, input->c_sth, input->rt, 
      input->fix_vis, input->viscous, input->grad_via_div, input->overset, geo->iblank_cell_d.data(), input->motion);

  check_error();
#endif
}

void Elements::common_U_to_F(unsigned int dim)
{
#ifdef _CPU
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        double n = norm_fpts(dim, fpt, ele);
        double dA = dA_fpts(fpt, ele); 
        double F = Ucomm(fpt, var, ele) * n;

        Fcomm(fpt, var, ele) = F * dA;
      }
    }
  }
#endif

#ifdef _GPU
  common_U_to_F_wrapper(Fcomm_d, Ucomm_d, norm_fpts_d, dA_fpts_d, nEles, nFpts, nVars, nDims, input->equation, dim);

  check_error();
#endif
}

template<unsigned int nVars, unsigned int nDims>
void Elements::compute_unit_advF(unsigned int dim)
{
  double U[nVars];
  double inv_jaco[nDims];

  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get state variables */
      for (unsigned int var = 0; var < nVars; var++)
      {
        U[var] = U_spts(spt, var, ele);
      }

      /* Get required metric terms */
      for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
      {
          inv_jaco[dim1] = inv_jaco_spts(dim1, spt, dim, ele);
      }

      /* Compute transformed unit advection flux along provided dimension */
      for (unsigned int var = 0; var < nVars; var++)
      {
        for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
        {
            F_spts(dim1, spt, var, ele) = U[var] * inv_jaco[dim1];
        }
      }
    }
  }

}

void Elements::compute_unit_advF(unsigned int dim)
{
#ifdef _CPU
  if (input->equation == AdvDiff)
  {
    if (nDims == 2)
      compute_unit_advF<1, 2>(dim);
    else if (nDims == 3)
      compute_unit_advF<1, 3>(dim);
  }
  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
      compute_unit_advF<4, 2>(dim);
    else if (nDims == 3)
      compute_unit_advF<5, 3>(dim);
  }
#endif

#ifdef _GPU
  compute_unit_advF_wrapper(F_spts_d, U_spts_d, inv_jaco_spts_d, nSpts, nEles, nDims, input->equation, dim);

  check_error();
#endif
}


void Elements::compute_local_dRdU()
{
#ifdef _CPU
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    /* Compute inviscid element local Jacobians */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int varj = 0; varj < nVars; varj++)
      {
        /* Compute Jacobian at flux points */
        for (unsigned int spti = 0; spti < nSpts; spti++)
          for (unsigned int fptj = 0; fptj < nFpts; fptj++)
            CtempSF(spti, fptj) = oppDiv_fpts(spti, fptj) * dFcdU(fptj, vari, varj, ele);

        for (unsigned int spti = 0; spti < nSpts; spti++)
          for (unsigned int sptj = 0; sptj < nSpts; sptj++)
          {
            double val = 0;
            for (unsigned int fptk = 0; fptk < nFpts; fptk++)
              val += CtempSF(spti, fptk) * oppE(fptk, sptj);
            LHS(ele, vari, spti, varj, sptj) = val;
          }

        /* Compute Jacobian at solution points */
        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int spti = 0; spti < nSpts; spti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
              LHS(ele, vari, spti, varj, sptj) += oppD(dim, spti, sptj) * dFdU_spts(dim, sptj, vari, varj, ele);
      }

    /* Compute viscous element local Jacobians */
    if (input->viscous)
    {
      /* Compute gradient Jacobians */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
        {
          /* Compute solution gradient Jacobians in reference space */
          for (unsigned int fpti = 0; fpti < nFpts; fpti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
              CtempFS(fpti, sptj) = dUcdU(fpti, vari, varj, ele) * oppE(fpti, sptj);

          for (unsigned int dim = 0; dim < nDims; dim++)
          {
            for (unsigned int spti = 0; spti < nSpts; spti++)
              for (unsigned int sptj = 0; sptj < nSpts; sptj++)
              {
                double val = 0;
                for (unsigned int fptk = 0; fptk < nFpts; fptk++)
                  val += oppD_fpts(dim, spti, fptk) * CtempFS(fptk, sptj);
                Cvisc0(dim, vari, varj, spti, sptj) = val;
              }

            if (vari == varj)
              for (unsigned int spti = 0; spti < nSpts; spti++)
                for (unsigned int sptj = 0; sptj < nSpts; sptj++)
                  Cvisc0(dim, vari, varj, spti, sptj) += oppD(dim, spti, sptj);
          }

          /* Transform gradient Jacobians to physical space */
          for (unsigned int spti = 0; spti < nSpts; spti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
            {
              CtempD.fill(0);
              for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
              {
                for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
                  CtempD(dim1) += Cvisc0(dim2, vari, varj, spti, sptj) * inv_jaco_spts(dim2, spti, dim1, ele);
                CtempD(dim1) /= jaco_det_spts(spti, ele);
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
                Cvisc0(dim, vari, varj, spti, sptj) = CtempD(dim);
            }
        }

      /* Compute viscous Jacobian at solution points (dFddU only) */
      CdFddU0.fill(0);
      for (unsigned int dimi = 0; dimi < nDims; dimi++)
        for (unsigned int dimj = 0; dimj < nDims; dimj++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int varj = 0; varj < nVars; varj++)
              for (unsigned int vark = 0; vark < nVars; vark++)
                for (unsigned int spti = 0; spti < nSpts; spti++)
                  for (unsigned int sptj = 0; sptj < nSpts; sptj++)
                    CdFddU0(dimi, vari, varj, spti, sptj) += dFddU_spts(dimi, dimj, spti, vari, vark, ele) * Cvisc0(dimj, vark, varj, spti, sptj);

      for (unsigned int dim = 0; dim < nDims; dim++)
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            for (unsigned int spti = 0; spti < nSpts; spti++)
              for (unsigned int sptj = 0; sptj < nSpts; sptj++)
              {
                double val = 0;
                for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                  val += oppD(dim, spti, sptk) * CdFddU0(dim, vari, varj, sptk, sptj);
                LHS(ele, vari, spti, varj, sptj) += val;
              }

      /* Compute viscous Jacobian at flux points (dFcddU only) */
      CdFcddU0.fill(0);
      for (unsigned int dim = 0; dim < nDims; dim++)
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            for (unsigned int vark = 0; vark < nVars; vark++)
              for (unsigned int fpti = 0; fpti < nFpts; fpti++)
                for (unsigned int sptj = 0; sptj < nSpts; sptj++)
                  for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                    CdFcddU0(vari, varj, fpti, sptj) += dFcddU(0, dim, fpti, vari, vark, ele) * oppE(fpti, sptk) * Cvisc0(dim, vark, varj, sptk, sptj);

      /* Add center contribution to Neighbor gradient */
      unsigned int eleID = geo->eleID[etype](ele + startEle);
      for (unsigned int face = 0; face < nFaces; face++)
      {
        /* Note: No contribution if element neighbor is a boundary */
        int eleNID = geo->ele2eleN(face, eleID);
        if (eleNID == -1) continue;

        unsigned int faceN = geo->face2faceN(face, eleID);

        /* Compute Neighbor gradient Jacobian (only center contribution) */
        for (unsigned int var = 0; var < nVars; var++)
        {
          /* Compute Neighbor gradient Jacobian in reference space */
          for (unsigned int fpti = 0; fpti < nFptsPerFace; fpti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
            {
              unsigned int fptN = faceN * nFptsPerFace + fpti;

              // HACK: fpt2fptN of eleNID doesn't exist for mpi faces so search for fpt manually
              // Note: Consider creating this connectivity during preprocessing
              unsigned int fpt = face * nFptsPerFace;
              while (geo->fpt2fptN(fpt, eleID) != (int)fptN) fpt++;
              //unsigned int fpt = geo->fpt2fptN(fptN, eleNID);

              CtempFSN(fpti, sptj) = dUcdU(fpt, var, var, ele) * oppE(fpt, sptj);
            }

          for (unsigned int dim = 0; dim < nDims; dim++)
            for (unsigned int spti = 0; spti < nSpts; spti++)
              for (unsigned int sptj = 0; sptj < nSpts; sptj++)
              {
                double val = 0;
                for (unsigned int fptk = 0; fptk < nFptsPerFace; fptk++)
                {
                  unsigned int fptN = faceN * nFptsPerFace + fptk;
                  val += oppD_fpts(dim, spti, fptN) * CtempFSN(fptk, sptj);
                }
                CviscN(dim, var, spti, sptj) = val;
              }

          /* Transform Neighbor gradient Jacobian to physical space */
          for (unsigned int spti = 0; spti < nSpts; spti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
            {
              CtempD.fill(0);
              for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
              {
                for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
                  CtempD(dim1) += CviscN(dim2, var, spti, sptj) * inv_jacoN_spts(face, dim2, spti, dim1, ele);
                CtempD(dim1) /= jacoN_det_spts(face, spti, ele);
              }

              for (unsigned int dim = 0; dim < nDims; dim++)
                CviscN(dim, var, spti, sptj) = CtempD(dim);
            }
        }

        for (unsigned int dim = 0; dim < nDims; dim++)
          for (unsigned int vari = 0; vari < nVars; vari++)
            for (unsigned int varj = 0; varj < nVars; varj++)
              for (unsigned int fpti = 0; fpti < nFptsPerFace; fpti++)
                for (unsigned int sptj = 0; sptj < nSpts; sptj++)
                  for (unsigned int sptk = 0; sptk < nSpts; sptk++)
                  {
                    unsigned int fpt = face * nFptsPerFace + fpti;
                    int fptN = geo->fpt2fptN(fpt, eleID);
                    CdFcddU0(vari, varj, fpt, sptj) += dFcddU(1, dim, fpt, vari, varj, ele) * oppE(fptN, sptk) * CviscN(dim, varj, sptk, sptj);
                  }
      }

      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          for (unsigned int spti = 0; spti < nSpts; spti++)
            for (unsigned int sptj = 0; sptj < nSpts; sptj++)
            {
              double val = 0;
              for (unsigned int fptk = 0; fptk < nFpts; fptk++)
                val += oppDiv_fpts(spti, fptk) * CdFcddU0(vari, varj, fptk, sptj);
              LHS(ele, vari, spti, varj, sptj) += val;
            }
    }

    /* Scale residual Jacobian */
    for (unsigned int vari = 0; vari < nVars; vari++)
      for (unsigned int spti = 0; spti < nSpts; spti++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          for (unsigned int sptj = 0; sptj < nSpts; sptj++)
            LHS(ele, vari, spti, varj, sptj) /= jaco_det_spts(spti, ele);
  }
#endif

#ifdef _GPU
  if (input->KPF_Jacobian)
  {
    /* Compute element local Jacobians */
    compute_KPF_Jac_wrapper(LHS_d, oppD_spts1D_d, oppDivE_spts1D_d, dFdU_spts_d, dFcdU_d, 
        nSpts1D, nVars, nEles, nDims);

    /* Compute Jacobian (local gradient contributions) */
    if (input->viscous)
    {
      compute_KPF_Jac_grad_wrapper(LHS_d, oppD_spts1D_d, oppDivE_spts1D_d, oppDE_spts1D_d, 
          dUcdU_d, dFddU_spts_d, dFcddU_d, inv_jaco_spts_d, jaco_det_spts_d, nSpts1D, nVars, 
          nEles, nDims);
    }
  }

  else
  {
    /* Compute element local Jacobians */
    /* Compute Jacobian at solution points */
    compute_Jac_spts_wrapper(LHS_d, oppD_d, dFdU_spts_d, nSpts, nVars, nEles, nDims);

    /* Compute Jacobian at flux points */
    compute_Jac_fpts_wrapper(LHS_d, oppDiv_fpts_d, oppE_d, dFcdU_d, nSpts, nFpts, nVars, nEles);

    /* Compute element local Jacobians (gradient contributions) */
    if (input->viscous)
    {
      /* Compute Jacobian (local gradient contributions) */
      compute_Jac_grad_wrapper(LHS_d, oppD_d, oppDiv_fpts_d, oppD_fpts_d, oppE_d, 
          dUcdU_d, dFddU_spts_d, dFcddU_d, inv_jaco_spts_d, jaco_det_spts_d, nVars, nEles, 
          nDims, order);

      /* Compute Jacobian (neighbor gradient contributions) */
      compute_Jac_gradN_wrapper(LHS_d, oppDiv_fpts_d, oppD_fpts_d, oppE_d, dUcdU_d, 
          dFcddU_d, inv_jacoN_spts_d, jacoN_det_spts_d, geo->eleID_d[etype], geo->ele2eleN_d, 
          geo->face2faceN_d, geo->fpt2fptN_d, startEle, nFptsPerFace, nFaces, nVars, nEles, 
          nDims, order);
    }
  }

  /* Scale residual Jacobian */
  scale_Jac_wrapper(LHS_d, jaco_det_spts_d, nSpts, nVars, nEles);
  check_error();

#endif
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Elements::compute_dFdU()
{
  double U[nVars];
  double dU[nVars][nDims];
  double dFdU[nVars][nVars][nDims] = {0};
  double dFddU[nVars][nVars][nDims][nDims] = {0};

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      /* Get state variables and physical space gradients */
      for (unsigned int var = 0; var < nVars; var++)
      {
        U[var] = U_spts(spt, var, ele);

        if(input->viscous) 
          for(unsigned int dim = 0; dim < nDims; dim++)
            dU[var][dim] = dU_spts(dim, spt, var, ele);
      }

      /* Compute flux derivatives */
      if (equation == AdvDiff)
      {
        double A[nDims];
        for(unsigned int dim = 0; dim < nDims; dim++)
          A[dim] = input->AdvDiff_A(dim);

        compute_dFdUconv_AdvDiff<nVars, nDims>(dFdU, A);
        if(input->viscous) compute_dFddUvisc_AdvDiff<nVars, nDims>(dFddU, input->AdvDiff_D);
      }
      else if (equation == EulerNS)
      {
        compute_dFdUconv_EulerNS<nVars, nDims>(U, dFdU, input->gamma);
        if(input->viscous)
        {
          compute_dFdUvisc_EulerNS_add<nVars, nDims>(U, dU, dFdU, input->gamma, input->prandtl, input->mu);
          compute_dFddUvisc_EulerNS<nVars, nDims>(U, dFddU, input->gamma, input->prandtl, input->mu);
        }
      }

      /* Get metric terms */
      double inv_jaco[nDims][nDims];
      for (int dim1 = 0; dim1 < nDims; dim1++)
        for (int dim2 = 0; dim2 < nDims; dim2++)
          inv_jaco[dim1][dim2] = inv_jaco_spts(dim1, spt, dim2, ele);

      /* Transform flux derivative to reference space */
      double tdFdU[nVars][nVars][nDims] = {{0.0}};
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
            for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
              tdFdU[vari][varj][dim1] += dFdU[vari][varj][dim2] * inv_jaco[dim1][dim2];

      /* Write out transformed flux derivatives */
      for (unsigned int vari = 0; vari < nVars; vari++)
        for (unsigned int varj = 0; varj < nVars; varj++)
          for (unsigned int dim = 0; dim < nDims; dim++)
            dFdU_spts(dim, spt, vari, varj, ele) = tdFdU[vari][varj][dim];

      if(input->viscous)
      {
        /* Transform flux derivative to reference space */
        double tdFddU[nVars][nVars][nDims][nDims] = {{0.0}};
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            for (unsigned int dimj = 0; dimj < nDims; dimj++)
              for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
                for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
                  tdFddU[vari][varj][dim1][dimj] += dFddU[vari][varj][dim2][dimj] * inv_jaco[dim1][dim2];

        /* Write out transformed flux derivatives */
        for (unsigned int vari = 0; vari < nVars; vari++)
          for (unsigned int varj = 0; varj < nVars; varj++)
            for (unsigned int dimi = 0; dimi < nDims; dimi++)
              for (unsigned int dimj = 0; dimj < nDims; dimj++)
                dFddU_spts(dimi, dimj, spt, vari, varj, ele) = tdFddU[vari][varj][dimi][dimj];
      }
    }
  }
}

void Elements::compute_dFdU()
{
#ifdef _CPU
  if (input->equation == AdvDiff)
  {
    if (nDims == 2)
      compute_dFdU<1, 2, AdvDiff>();
    else if (nDims == 3)
      compute_dFdU<1, 3, AdvDiff>();
  }
  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
      compute_dFdU<4, 2, EulerNS>();
    else if (nDims == 3)
      compute_dFdU<5, 3, EulerNS>();
  }
#endif

#ifdef _GPU
  compute_dFdU_wrapper(dFdU_spts_d, dFddU_spts_d, U_spts_d, dU_spts_d, inv_jaco_spts_d, nSpts, nEles, nDims, input->equation, 
      input->AdvDiff_A_d, input->AdvDiff_D, input->gamma, input->prandtl, input->mu, input->viscous);

  check_error();
#endif
}

void Elements::compute_KPF_dFcdU_gradN()
{
#ifdef _CPU
  ThrowException("compute_KPF_dFcdU_gradN() not implemented on CPU!");
#endif

#ifdef _GPU
  compute_KPF_dFcdU_gradN_wrapper(dFcdU_d, dFcddU_d, ddUdUc_d, dUcdU_d, nFpts, nVars, nEles, nDims);

  check_error();
#endif
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

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        sum += weights_spts(spt) * jaco_det_spts(spt, ele) * U_spts(spt, n, ele);
      }

      Uavg(n, ele) = sum / vol(ele); 

    }
  }
#endif

#ifdef _GPU
  compute_Uavg_wrapper(U_spts_d, Uavg_d, jaco_det_spts_d, weights_spts_d, vol_d, nSpts, nEles, nVars, nDims, order);
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
        minRho = std::min(minRho, U_spts(spt, 0, ele));
      }
    }
    
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      if (U_fpts(fpt, 0, ele) < 0)
      {
        negRho = true;
        minRho = std::min(minRho, U_fpts(fpt, 0, ele));
      }
    }

    /* If negative density found, squeeze density */
    if (negRho)
    {
      double theta = (Uavg(0, ele) - tol) / (Uavg(0 , ele) - minRho); 
      //double theta = 1.0;

      for (unsigned int spt = 0; spt < nSpts; spt++)
        U_spts(spt, 0, ele) = theta * U_spts(spt, 0, ele) + (1.0 - theta) * Uavg(0, ele);

      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        U_fpts(fpt, 0, ele) = theta * U_fpts(fpt, 0, ele) + (1.0 - theta) * Uavg(0, ele);
      
    }
  }


  /* For each element, check for entropy loss */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    double minTau = 1.0;

    /* Get minimum tau value */
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      double rho = U_spts(spt, 0, ele);
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momF += U_spts(spt, dim + 1, ele) * U_spts(spt, dim + 1, ele);

      momF /= U_spts(spt, 0, ele);

      double P = (input->gamma - 1.0) * (U_spts(spt, nDims + 1, ele) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);

    }
    
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      double rho = U_fpts(fpt, 0, ele);
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momF += U_fpts(fpt, dim + 1, ele) * U_fpts(fpt, dim + 1, ele);

      momF /= U_fpts(fpt, 0, ele);
      double P = (input->gamma - 1.0) * (U_fpts(fpt, nDims + 1, ele) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);

    }

    /* If minTau is negative, squeeze solution */
    if (minTau < 0)
    {
      double rho = Uavg(0, ele);
      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = Uavg(dim+1, ele) / rho;
        Vsq += V[dim] * V[dim];
      }

      double e = Uavg(nDims + 1, ele);
      double P = (input->gamma - 1.0) * (e - 0.5 * rho * Vsq);

      double eps = minTau / (minTau - P + input->exps0 * std::pow(rho, input->gamma));

//      if (P < input->exps0 * std::pow(rho, input->gamma))
//        std::cout << "Constraint violated. Lower CFL?" << std::endl;

      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          U_spts(spt, n, ele) = eps * Uavg(n, ele) + (1.0 - eps) * U_spts(spt, n, ele);
        }
      }

      for (unsigned int n = 0; n < nVars; n++)
      {
        for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        {
          U_fpts(fpt, n, ele) = eps * Uavg(n, ele) + (1.0 - eps) * U_fpts(fpt, n, ele);
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
    double minRho = U_ppts(0, 0, ele);

    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      if (U_ppts(ppt, 0, ele) < 0)
      {
        negRho = true;
        minRho = std::min(minRho, U_ppts(ppt, 0, ele));
      }
    }
    
    /* If negative density found, squeeze density */
    if (negRho)
    {
      double theta = std::abs(Uavg(0, ele) - tol) / (Uavg(0, ele) - minRho); 

      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
        U_ppts(ppt, 0, ele) = theta * U_ppts(ppt, 0, ele) + (1.0 - theta) * Uavg(0, ele);
    }
  }

  /* For each element, check for entropy loss */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    double minTau = 1.0;

    /* Get minimum tau value */
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      double rho = U_ppts(ppt, 0, ele);
      double momF = 0.0;
      for (int dim = 0; dim < nDims; dim++)
        momF += U_ppts(ppt, dim+1, ele) * U_ppts(ppt, dim+1, ele);

      momF /= rho;
      double P = (input->gamma - 1.0) * (U_ppts(ppt, nDims+1, ele) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);
    }
    
    /* If minTau is negative, squeeze solution */
    if (minTau < 0)
    {
      double rho = Uavg(0, ele);
      double Vsq = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        V[dim] = Uavg(dim+1, ele) / rho;
        Vsq += V[dim] * V[dim];
      }

      double e = Uavg(nDims+1, ele);
      double P = (input->gamma - 1.0) * (e - 0.5 * rho * Vsq);

      double eps = minTau / (minTau - P + input->exps0 * std::pow(rho, input->gamma));

      for (unsigned int ppt = 0; ppt < nPpts; ppt++)
      {
        for (unsigned int n = 0; n < nVars; n++)
        {
          U_ppts(ppt, n, ele) = eps * Uavg(n, ele) + (1.0 - eps) * U_ppts(ppt, n, ele);
          //U_ppts(ppt, ele, n) = (1.0 - eps) * Uavg(ele, n) + eps * U_ppts(ppt, ele, n);
        }
      }

    }

  }

}

void Elements::move(std::shared_ptr<Faces> faces)
{
#ifdef _CPU
  if (input->motion_type == RIGID_BODY)
  {
    // Update grid position based on rigid-body motion: CG offset + rotation
    for (unsigned int i = 0; i < geo->nNodes; i++)
      for (unsigned int d = 0; d < nDims; d++)
        geo->coord_nodes(i,d) = geo->x_cg(d);

    auto &A = geo->coords_init;
    auto &B = geo->Rmat;  /// TODO: double-check orientation
    auto &C = geo->coord_nodes;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geo->nNodes, nDims, nDims,
        1.0, A.data(), A.ldim(), B.data(), B.ldim(), 1.0, C.data(), C.ldim());

    // Update grid velocity based on 'spin' matrix (omega cross r)
    for (unsigned int i = 0; i < geo->nNodes; i++)
      for (unsigned int d = 0; d < nDims; d++)
        geo->grid_vel_nodes(i,d) = geo->vel_cg(d);

    auto &Av = geo->coords_init;
    auto &Bv = geo->Wmat;  /// TODO: double-check orientation
    auto &Cv = geo->grid_vel_nodes;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geo->nNodes, nDims, nDims,
        1.0, Av.data(), A.ldim(), Bv.data(), B.ldim(), 1.0, Cv.data(), Cv.ldim());
  }

  update_point_coords(faces);
  update_grid_velocities(faces);
  if (input->motion_type != CIRCULAR_TRANS) // don't do for rigid translation
    calc_transforms(faces);  /// TODO: copy over GPU algo for rigid-body (to save cost)
#endif

#ifdef _GPU
  if (input->motion_type == RIGID_BODY)
  {
    // Positions
    update_nodes_rigid_wrapper(geo->coords_init_d, geo->coord_nodes_d, geo->Rmat_d,
        geo->x_cg_d, geo->nNodes, geo->nDims);

    update_nodes_rigid_wrapper(geo->coords_init_d, geo->grid_vel_nodes_d, geo->Wmat_d,
        geo->vel_cg_d, geo->nNodes, geo->nDims);
  }

  update_coords_wrapper(nodes_d, geo->coord_nodes_d, shape_spts_d,
      shape_fpts_d, coord_spts_d, coord_fpts_d, faces->coord_d,
      geo->ele2nodesBT_d[etype], geo->fpt2gfptBT_d[etype], nSpts, nFpts, nNodes, nEles, nDims);

  update_coords_wrapper(grid_vel_nodes_d, geo->grid_vel_nodes_d, shape_spts_d,
      shape_fpts_d, grid_vel_spts_d, grid_vel_fpts_d, faces->Vg_d,
      geo->ele2nodesBT_d[etype], geo->fpt2gfptBT_d[etype], nSpts, nFpts, nNodes, nEles, nDims);

  if (input->motion_type == RIGID_BODY)
  {
    /// TODO: kernels that take in 'body-coords' transforms and applies rotation matrix
    /* At times 1 and 2, jaco_1 = R_1 * jaco_0;  jaco_2 = R_2 * jaco_0
     * So to update from 1 to 2, jaco_2 = R_2 * R_1^inv * jaco_1
     * Where R is the matrix form of the body's roation quaternion */
    update_transforms_rigid_wrapper(jaco_spts_init_d, jaco_spts_d, inv_jaco_spts_d,
        faces->norm_init_d, faces->norm_d, geo->Rmat_d, nSpts, faces->nFpts, nEles, nDims, true);
  }
  else if (input->motion_type != CIRCULAR_TRANS)
  {
    calc_transforms_wrapper(nodes_d, jaco_spts_d, jaco_fpts_d, inv_jaco_spts_d,
                            inv_jaco_fpts_d, jaco_det_spts_d, dshape_spts_d, dshape_fpts_d, nSpts,
                            nFpts, nNodes, nEles, nDims);

    calc_normals_wrapper(faces->norm_d, faces->dA_d, inv_jaco_fpts_d, tnorm_d,
                         geo->fpt2gfptBT_d[etype], geo->fpt2gfpt_slotBT_d[etype], nFpts, nEles, nDims);

    if (input->CFL_type == 2 || input->CFL_tau_type == 2)
      update_h_ref_wrapper(h_ref_d, coord_fpts_d, nEles, nFpts, nSpts1D, nDims);
  }

  check_error();
#endif
}

void Elements::update_point_coords(std::shared_ptr<Faces> faces)
{
#ifdef _GPU
  // Copy back, since updated only on GPU
  geo->coord_nodes = geo->coord_nodes_d;
#endif
  for (uint node = 0; node < nNodes; node++)
    for (uint ele = 0; ele < nEles; ele++)
    {
      uint eleBT = ele + startEle;
      for (uint dim = 0; dim < nDims; dim++)
        nodes(node, dim, ele) = geo->coord_nodes(geo->ele2nodesBT[etype](eleBT,node),dim);
    }

  int ms = nSpts;
  int mf = nFpts;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = nodes(0,0,0);

  /* Setup physical coordinates at solution points */
  auto &As = shape_spts(0,0);
  auto &Cs = coord_spts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, n, 0.0, &Cs, n);

  /* Setup physical coordinates at flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = coord_fpts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, n, 0.0, &Cf, n);

  /* Setup physical coordinates at flux points [in faces class] */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      unsigned int eleBT = ele + startEle;
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        int gfpt = geo->fpt2gfptBT[etype](fpt,eleBT);

        faces->coord(dim,gfpt) = coord_fpts(fpt,dim,ele);
      }
    }
  }

  if (input->CFL_type == 2 || input->CFL_tau_type == 2)
  {
    /* Compute tensor-line lengths */
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < nFpts/2; fpt++)
      {
        if (nDims == 2)
        {
          /* Some indexing to pair up opposing flux points in 2D (on Quad) */
          unsigned int idx = fpt % nSpts1D;
          unsigned int fpt1 = fpt;
          unsigned int fpt2 =  (fpt / nSpts1D + 3) * nSpts1D - idx - 1;

          double dx = coord_fpts(fpt1, 0, ele) - coord_fpts(fpt2, 0, ele);
          double dy = coord_fpts(fpt1, 1, ele) - coord_fpts(fpt2, 1, ele);
          double dist = std::sqrt(dx*dx + dy*dy);

          h_ref(fpt1, ele) = dist;
          h_ref(fpt2, ele) = dist;
        }
        else
        {
          ThrowException("h_ref computation not setup in 3D yet!");
        }
      }
    }
  }
}

void Elements::update_plot_point_coords(void)
{
#ifdef _GPU
  // Copy back, since updated only on GPU
  nodes = nodes_d;
#endif

  int mp = nPpts;
  int mq = nQpts;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = nodes(0,0,0);

  /* Setup physical coordinates at plot points */
  auto &Ap = shape_ppts(0,0);
  auto &Cp = coord_ppts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, n, 0.0, &Cp, n);

  /* Setup physical coordinates at quadrature points */
  auto &Aq = shape_qpts(0,0);
  auto &Cq = coord_qpts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, n, 0.0, &Cq, n);
}

void Elements::update_grid_velocities(std::shared_ptr<Faces> faces)
{
  for (uint node = 0; node < nNodes; node++)
    for (uint ele = 0; ele < nEles; ele++)
    {
      uint eleBT = ele + startEle;
      for (uint dim = 0; dim < nDims; dim++)
        grid_vel_nodes(node, dim, ele) = geo->grid_vel_nodes(geo->ele2nodesBT[etype](eleBT,node), dim);
    }

  int ms = nSpts;
  int mf = nFpts;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = grid_vel_nodes(0,0,0);

  /* Interpolate grid velocities to solution points */
  auto &As = shape_spts(0,0);
  auto &Cs = grid_vel_spts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, n, 0.0, &Cs, n);

  /* Interpolate grid velocities to flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = grid_vel_fpts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, n, 0.0, &Cf, n);

  /* Store grid velocity in face class */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    unsigned int eleBT = ele + startEle;
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfptBT[etype](fpt,eleBT);
      unsigned int slot = geo->fpt2gfpt_slotBT[etype](fpt,eleBT);

      if (slot != 0)
        continue;

      for (uint dim = 0; dim < nDims; dim++)
      {
        faces->Vg(dim, gfpt) = grid_vel_fpts(fpt, dim, ele);
      }
    }
  }
}

void Elements::get_grid_velocity_ppts(void)
{
  int m = nPpts;
  int k = nNodes;
  int n = nEles * nDims;

  /* Interpolate grid velocities to plot points */
  auto &A = shape_ppts(0,0);
  auto &B = grid_vel_nodes(0,0,0);
  auto &C = grid_vel_ppts(0,0,0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              1.0, &A, k, &B, n, 0.0, &C, n);
}

std::vector<double> Elements::getBoundingBox(int ele)
{
  std::vector<double> bbox = { INFINITY, INFINITY, INFINITY,
                              -INFINITY,-INFINITY,-INFINITY};

  for (unsigned int node = 0; node < nNodes; node++)
  {
    unsigned int nd = geo->ele2nodesBT[etype](ele, node);
    for (int dim = 0; dim < nDims; dim++)
    {
      double pos = geo->coord_nodes(nd,dim);
      bbox[dim]   = std::min(bbox[dim],  pos);
      bbox[dim+3] = std::max(bbox[dim+3],pos);
    }
  }

  if (nDims == 2)
  {
    bbox[2] = 0;
    bbox[5] = 0;
  }

  return bbox;
}

void Elements::getBoundingBox(int ele, double bbox[6])
{
  for (unsigned int i = 0; i < 3; i++)
  {
    bbox[i]   =  INFINITY;
    bbox[i+3] = -INFINITY;
  }

  for (unsigned int node = 0; node < nNodes; node++)
  {
    unsigned int nd = geo->ele2nodesBT[etype](ele, node);
    for (int dim = 0; dim < nDims; dim++)
    {
      double pos = geo->coord_nodes(nd,dim);
      bbox[dim]   = std::min(bbox[dim],  pos);
      bbox[dim+3] = std::max(bbox[dim+3],pos);
    }
  }

  if (nDims == 2)
  {
    bbox[2] = 0;
    bbox[5] = 0;
  }
}

bool Elements::getRefLoc(int ele, double* xyz, double* rst)
{
  double xmin, ymin, zmin;
  double xmax, ymax, zmax;
  xmin = ymin = zmin =  1e15;
  xmax = ymax = zmax = -1e15;
  double eps = 1e-10;

  double box[6];
  getBoundingBox(ele,box);
  xmin = box[0];  ymin = box[1];  zmin = box[2];
  xmax = box[3];  ymax = box[4];  zmax = box[5];

  if (xyz[0] < xmin-eps || xyz[1] < ymin-eps || xyz[2] < zmin-eps ||
      xyz[0] > xmax+eps || xyz[1] > ymax+eps || xyz[2] > zmax+eps) {
    // Point does not lie within cell - return an obviously bad ref position
    rst[0] = 99.; rst[1] = 99.; rst[2] = 99.;
    return false;
  }

  // Use a relative tolerance to handle extreme grids
  double h = std::min(xmax-xmin,ymax-ymin);
  if (nDims==3) h = std::min(h,zmax-zmin);

  double tol = 1e-12*h;

  if (tmp_shape.size() != nNodes)
  {
    tmp_shape.resize({nNodes});
    tmp_dshape.resize({nNodes, nDims});
    tmp_coords.resize({nNodes, nDims});
  }

  int iter = 0;
  int iterMax = 20;
  double norm = 1;
  double norm_prev = 2;

  bool restart_rst = true;
  for (int i = 0; i < 3; i++)
  {
    if (std::abs(rst[i]) > 1.)
    {
      restart_rst = false;
      break;
    }
  }

  if (!restart_rst)
  {
    rst[0] = 0.; rst[1] = 0.; rst[2] = 0.;
  }

  for (int nd = 0; nd < nNodes; nd++)
    for (int d = 0; d < 3; d++)
      tmp_coords(nd,d) = geo->coord_nodes(geo->ele2nodesBT[etype](ele, nd),d);

  while (norm > tol && iter < iterMax)
  {
    calc_shape(tmp_shape, rst);
    calc_d_shape(tmp_dshape, rst);

    point dx(xyz[0],xyz[1],xyz[2]);

    double grad[3][3] = {{0.0}};
    double ginv[3][3] = {{0.0}};
    for (int nd = 0; nd < nNodes; nd++)
      for (int i = 0; i < nDims; i++)
        for (int j = 0; j < nDims; j++)
          grad[i][j] += tmp_coords(nd,i) * tmp_dshape(nd,j);

    for (int nd = 0; nd < nNodes; nd++)
      for (int i = 0; i < 3; i++)
        dx[i] -= tmp_shape(nd)*tmp_coords(nd,i);

    double detJ = det_3x3(&grad[0][0]);

    adjoint_3x3(&grad[0][0], &ginv[0][0]);

    double delta[3] = {0,0,0};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        delta[i] += ginv[i][j]*dx[j]/detJ;

    norm = dx.norm();
    for (int i = 0; i < 3; i++)
      rst[i] = std::max(std::min(rst[i]+delta[i],1.),-1.);

    if (iter > 1 && norm > .99*norm_prev) // If it's clear we're not converging
      break;

    norm_prev = norm;

    iter++;
  }

  if (norm <= tol)
    return true;
  else
    return false;
}

void Elements::get_interp_weights(double* rst, double* weights, int& nweights, int buffSize)
{
  assert(nSpts <= buffSize);

  nweights = nSpts;

  this->calc_nodal_basis(rst, weights);
}

#ifdef _GPU
void Elements::get_interp_weights_gpu(int* cellIDs, int nFringe, double* rst, double* weights)
{
  if (loc_spts_1D_d.size() != loc_spts_1D.size())
    loc_spts_1D_d.assign({loc_spts_1D.size()}, loc_spts_1D.data());

  get_nodal_basis_wrapper(cellIDs, rst, weights, loc_spts_1D_d.data(), nFringe, nSpts, nSpts1D, 3);
}

void Elements::donor_u_from_device(int* donorIDs_in, int nDonors_in)
{
  if (nDonors_in == 0) return;

  if (nDonors != nDonors_in)
  {
    if (nDonors > 0)
      free_device_data(donorIDs_d);

    nDonors = nDonors_in;

    U_donors.resize({nSpts,nDonors,nVars});
    U_donors_d.set_size(U_donors);

    if (input->viscous)
    {
      dU_donors.resize({nSpts,nDonors,nVars,nDims});
      dU_donors_d.set_size(dU_donors);
      dU_donors = dU_donors_d;
    }

    donorIDs.assign(donorIDs_in, donorIDs_in+nDonors);
    allocate_device_data(donorIDs_d, nDonors);
    copy_to_device(donorIDs_d, donorIDs_in, nDonors);
  }
  else
  {
    bool sameIDs = true;
    for (int i = 0; i < nDonors; i++)
    {
      if (donorIDs[i] != donorIDs_in[i])
      {
        sameIDs = false;
        break;
      }
    }

    if (!sameIDs)
    {
      donorIDs.assign(donorIDs_in, donorIDs_in+nDonors);
      copy_to_device(donorIDs_d, donorIDs_in, nDonors_in);
    }
  }

  pack_donor_u_wrapper(U_spts_d,U_donors_d,donorIDs_d,nDonors,nSpts,nVars);

  check_error();

  U_donors = U_donors_d;

  for (int var = 0; var < nVars; var++)
  {
    for (int donor = 0; donor < nDonors; donor++)
    {
      unsigned int ele = donorIDs[donor];
      for (int spt = 0; spt < nSpts; spt++)
      {
        U_spts(spt,ele,var) = U_donors(spt,donor,var);
      }
    }
  }
}

void Elements::donor_grad_from_device(int* donorIDs_in, int nDonors_in)
{
  if (nDonors_in == 0) return;

  if (nDonors_in != nDonors)
    ThrowException("Invalid nDonors/nDonors_in - should have been set up in donor_u_from_device!");

  pack_donor_grad_wrapper(dU_spts_d,dU_donors_d,donorIDs_d,nDonors_in,nSpts,nVars,nDims);

  check_error();

  dU_donors = dU_donors_d;

  for (int dim = 0; dim < nDims; dim++)
  {
    for (int var = 0; var < nVars; var++)
    {
      for (int donor = 0; donor < nDonors; donor++)
      {
        unsigned int ele = donorIDs[donor];
        for (int spt = 0; spt < nSpts; spt++)
        {
          dU_spts(spt,ele,var,dim) = dU_donors(spt,donor,var,dim);
        }
      }
    }
  }
}

void Elements::unblank_u_to_device(int *cellIDs, int nCells, double *data)
{
  if (nCells == 0) return;

  U_unblank_d.assign({nCells, nSpts, nVars}, data, 3);

  if (input->motion || input->iter <= input->initIter+1) /// TODO: double-check
    unblankIDs_d.assign({nCells}, cellIDs, 3);

  unpack_unblank_u_wrapper(U_unblank_d,U_spts_d,unblankIDs_d,nCells,nSpts,nVars,3);

  check_error();
}

void Elements::get_cell_coords(int* cellIDs, int nCells, int* nPtsCell, double* xyz)
{
  if (nCells == 0) return;

  unblankIDs_d.assign({nCells}, cellIDs); /// TODO: check this w/ unblanking...

  cellCoords_d.set_size({nCells,nSpts,nDims});

  pack_cell_coords_wrapper(unblankIDs_d, cellCoords_d, coord_spts_d, nCells, nSpts, nDims); /// TODO: async?

  copy_from_device(xyz, cellCoords_d.data(), cellCoords_d.size()); /// TODO: async?
}
#endif
