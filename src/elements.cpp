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

#include "elements.hpp"
#include "faces.hpp"
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

  set_locs();
  set_shape();
  set_coords(faces);
  set_normals(faces);
  calc_transforms(faces);
  setup_FR();
  setup_aux();  
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
  dshape_qpts.assign({nNodes, nQpts, nDims},1);

  if (input->motion)
  {
    grid_vel_nodes.assign({nNodes, nEles, nDims}, 0.);
    grid_vel_spts.assign({nSpts, nEles, nDims}, 0.);
    grid_vel_fpts.assign({nFpts, nEles, nDims}, 0.);
    grid_vel_ppts.assign({nPpts, nEles, nDims}, 0.);
  }

  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nSpts, nEles, nDims, nDims});
  jaco_fpts.assign({nFpts, nEles, nDims, nDims});
  jaco_qpts.assign({nQpts, nEles, nDims, nDims});
  inv_jaco_spts.assign({nSpts, nEles, nDims, nDims});
  inv_jaco_fpts.assign({nFpts, nEles, nDims, nDims});
  jaco_det_spts.assign({nSpts, nEles});
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


    calc_shape(shape_val, shape_order, loc);
    calc_d_shape(dshape_val, shape_order, loc);

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

    calc_shape(shape_val, shape_order, loc);
    calc_d_shape(dshape_val, shape_order, loc);

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

    calc_shape(shape_val, shape_order, loc);
    calc_d_shape(dshape_val, shape_order, loc);

    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(node, ppt) = shape_val(node);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      loc[dim] = loc_qpts(qpt,dim);

    calc_shape(shape_val, shape_order, loc);
    calc_d_shape(dshape_val, shape_order, loc);

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
  coord_spts.assign({nSpts, nEles, nDims});
  coord_fpts.assign({nFpts, nEles, nDims});
  faces->coord.assign({geo->nGfpts, nDims});
  coord_ppts.assign({nPpts, nEles, nDims});
  coord_qpts.assign({nQpts, nEles, nDims});
  nodes.assign({nNodes, nEles, nDims});

  /* Setup positions of all element's shape nodes in one array */
  if (input->meshfile.find(".pyfr") != std::string::npos)
  {
    nodes = geo->ele_nodes; /// TODO: setup for Gmsh grids as well
  }
  else
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
      for (unsigned int ele = 0; ele < nEles; ele++)
        for (unsigned int node = 0; node < nNodes; node++)
          nodes(node, ele, dim) = geo->coord_nodes(dim,geo->ele2nodes(node,ele));
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
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#endif

  /* Setup physical coordinates at flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = coord_fpts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
#endif

  /* Setup physical coordinates at plot points */
  auto &Ap = shape_ppts(0,0);
  auto &Cp = coord_ppts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, k, 0.0, &Cp, mp);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, k, 0.0, &Cp, mp);
#endif

  /* Setup physical coordinates at quadrature points */
  auto &Aq = shape_qpts(0,0);
  auto &Cq = coord_qpts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#endif

  /* Setup physical coordinates at flux points [in faces class] */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        int gfpt = geo->fpt2gfpt(fpt,ele);
        /* Check if on ghost edge */
        if (gfpt != -1)
        {
          faces->coord(gfpt, dim) = coord_fpts(fpt,ele,dim);
        }
      }
    }
  }

  if (input->CFL_type == 2)
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


          double dx = coord_fpts(fpt1, ele, 0) - coord_fpts(fpt2, ele, 0);
          double dy = coord_fpts(fpt1, ele, 1) - coord_fpts(fpt2, ele, 1);
          double dist = std::sqrt(dx*dx + dy*dy);

          h_ref(fpt1, ele) = dist;
          h_ref(fpt2, ele) = dist;
        }
      }
    }
    else /* nDims == 3 */
    {
      int nFptsPerFace = nSpts1D * nSpts1D;
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
              dx[d] = coord_fpts(fpt1,ele,d) - coord_fpts(fpt2,ele,d);

            double dist = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] * dx[2]*dx[2]);

            h_ref(fpt1, ele) = dist;
            h_ref(fpt2, ele) = dist;
          }
        }
      }
    }
  }
}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppE_Fn.assign({nFpts, nSpts, nDims});
  oppD.assign({nSpts, nSpts, nDims});
  oppD0.assign({nSpts, nSpts, nDims});
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

  /* Setup spt to fpt extrapolation operator for normal flux (oppE_Fn) */
  for (unsigned int dim = 0; dim < nDims; dim++)
    for (unsigned int spt = 0; spt < nSpts; spt++)
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
        oppE_Fn(fpt,spt,dim) = oppE(fpt,spt) * tnorm(fpt,dim);

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

  /* Setup differentiation operator (oppD) for solution points */
  /* Note: This one is the 'traditional' FR derivative [fpts not included] */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int jspt = 0; jspt < nSpts; jspt++)
    {
      for (unsigned int ispt = 0; ispt < nSpts; ispt++)
      {
        for (unsigned int d = 0; d < nDims; d++)
          loc[d] = loc_spts(ispt , d);

        oppD0(ispt,jspt,dim) = calc_d_nodal_basis_fr(jspt, loc, dim);
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
        if (face == 0 or face == 3) // Bottom and Left face
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

void Elements::calc_transforms(std::shared_ptr<Faces> faces)
{
  /* --- Calculate Transformation at Solution Points --- */

  int ms = nSpts;
  int mf = nFpts;
  int mq = nQpts;
  int k = nNodes;
  int n = nEles * nDims;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    auto &B = nodes(0,0,0);
    auto &As = dshape_spts(0, 0, dim);
    auto &Af = dshape_fpts(0, 0, dim);
    auto &Aq = dshape_qpts(0, 0, dim);
    auto &Cs = jaco_spts(0, 0, 0, dim);
    auto &Cf = jaco_fpts(0, 0, 0, dim);
    auto &Cq = jaco_qpts(0, 0, 0, dim);

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#endif
  }

  set_inverse_transforms(jaco_spts,inv_jaco_spts,jaco_det_spts,nSpts,nDims);

  mdvector<double> nullvec;
  set_inverse_transforms(jaco_fpts,inv_jaco_fpts,nullvec,nFpts,nDims);

  /* --- Compute Element Volumes --- */
#pragma omp parallel for collapse(2)
  for (unsigned int e = 0; e < nEles; e++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get quadrature weight */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);


      double weight = weights_spts(i) * weights_spts(j);

      if (nDims == 3)
        weight *= weights_spts(idx_spts(spt,2));

      vol(e) += weight * jaco_det_spts(spt, e);
    }
  }

  /* --- Calculate Transformation at Flux Points --- */
#pragma omp parallel for collapse(2)
  for (unsigned int e = 0; e < nEles; e++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfpt(fpt,e);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1)
        continue;

      unsigned int slot = geo->fpt2gfpt_slot(fpt,e);

      /* --- Calculate outward unit normal vector at flux point ("left" element only) --- */
      if (slot == 0)
      {
        // Transform face normal from reference to physical space [JGinv .dot. tNorm]
        for (uint dim1 = 0; dim1 < nDims; dim1++)
        {
          faces->norm(gfpt,dim1) = 0.;
          for (uint dim2 = 0; dim2 < nDims; dim2++)
            faces->norm(gfpt,dim1) += inv_jaco_fpts(fpt,e,dim2,dim1) * tnorm(fpt,dim2);
        }

        // Store magnitude of face normal (equivalent to face area in finite-volume land)
          faces->dA(gfpt) = 0;
          for (uint dim = 0; dim < nDims; dim++)
            faces->dA(gfpt) += faces->norm(gfpt,dim)*faces->norm(gfpt,dim);
          faces->dA(gfpt) = sqrt(faces->dA(gfpt));

        // Normalize
        // If we have a collapsed edge, the dA will be 0, so just set the normal to 0
        // (A normal vector at a point doesn't make sense anyways)
        if (std::fabs(faces->dA(gfpt)) < 1e-10)
        {
          faces->dA(gfpt) = 0.;
          for (uint dim = 0; dim < nDims; dim++)
            faces->norm(gfpt,dim) = 0;
        }
        else
        {
          for (uint dim = 0; dim < nDims; dim++)
            faces->norm(gfpt,dim) /= faces->dA(gfpt);
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
        jaco_det_qpts(qpt,e) = jaco_qpts(qpt,e,0,0)*jaco_qpts(qpt,e,1,1)-jaco_qpts(qpt,e,0,1)*jaco_qpts(qpt,e,1,0);
      }
      else if (nDims == 3)
      {
        double xr = jaco_qpts(qpt,e,0,0);   double xs = jaco_qpts(qpt,e,0,1);   double xt = jaco_qpts(qpt,e,0,2);
        double yr = jaco_qpts(qpt,e,1,0);   double ys = jaco_qpts(qpt,e,1,1);   double yt = jaco_qpts(qpt,e,1,2);
        double zr = jaco_qpts(qpt,e,2,0);   double zs = jaco_qpts(qpt,e,2,1);   double zt = jaco_qpts(qpt,e,2,2);
        jaco_det_qpts(qpt,e) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);
      }
      if (jaco_det_qpts(qpt,e)<0) ThrowException("Negative Jacobian at quadrature point.");
    }
  }
}

void Elements::set_inverse_transforms(const mdvector<double> &jaco,
               mdvector<double> &inv_jaco, mdvector<double> &jaco_det,
               unsigned int nPts, unsigned int nDims)
{
#pragma omp parallel for collapse(2)
  for (unsigned int e = 0; e < nEles; e++)
  {
    for (unsigned int pt = 0; pt < nPts; pt++)
    {
      if (nDims == 2)
      {
        // Determinant of transformation matrix
        if (jaco_det.size()) jaco_det(pt,e) = jaco(pt,e,0,0)*jaco(pt,e,1,1)-jaco(pt,e,0,1)*jaco(pt,e,1,0);

        // Inverse of transformation matrix (times its determinant)
        inv_jaco(pt,e,0,0) = jaco(pt,e,1,1);  inv_jaco(pt,e,0,1) =-jaco(pt,e,0,1);
        inv_jaco(pt,e,1,0) =-jaco(pt,e,1,0);  inv_jaco(pt,e,1,1) = jaco(pt,e,0,0);
      }
      else if (nDims == 3)
      {
        double xr = jaco(pt,e,0,0);   double xs = jaco(pt,e,0,1);   double xt = jaco(pt,e,0,2);
        double yr = jaco(pt,e,1,0);   double ys = jaco(pt,e,1,1);   double yt = jaco(pt,e,1,2);
        double zr = jaco(pt,e,2,0);   double zs = jaco(pt,e,2,1);   double zt = jaco(pt,e,2,2);
        if (jaco_det.size()) jaco_det(pt,e) = xr*(ys*zt - yt*zs) - xs*(yr*zt - yt*zr) + xt*(yr*zs - ys*zr);

        inv_jaco(pt,e,0,0) = ys*zt - yt*zs;  inv_jaco(pt,e,0,1) = xt*zs - xs*zt;  inv_jaco(pt,e,0,2) = xs*yt - xt*ys;
        inv_jaco(pt,e,1,0) = yt*zr - yr*zt;  inv_jaco(pt,e,1,1) = xr*zt - xt*zr;  inv_jaco(pt,e,1,2) = xt*yr - xr*yt;
        inv_jaco(pt,e,2,0) = yr*zs - ys*zr;  inv_jaco(pt,e,2,1) = xs*zr - xr*zs;  inv_jaco(pt,e,2,2) = xr*ys - xs*yr;
      }

      if (jaco_det.size() and jaco_det(pt,e) < 0) ThrowException("Negative Jacobian at solution points.");
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

  event_record(0, 0); // record event for MPI comms

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
    auto *A = oppE_d.get_ptr(0,0);
    auto *B = dU_spts_d.get_ptr(0, startEle, 0, dim);
    auto *C = dU_fpts_d.get_ptr(0, startEle, 0, dim);

    cublasDGEMM_wrapper(nFpts, nEles * nVars, nSpts, 1.0, A, oppE_d.ldim(), 
        B, dU_spts_d.ldim(), 0.0, C, dU_fpts_d.ldim());
  }

  event_record(0, 0); // record event for MPI comms
  check_error();
#endif
}

void Elements::extrapolate_Fn(unsigned int startEle, unsigned int endEle, std::shared_ptr<Faces> faces)
{
#ifdef _CPU
  dFn_fpts = Fcomm;

  if (input->motion)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto &A = oppE(0, 0);
        auto &B = F_spts(0, startEle, var, dim);
        auto &C = tempF_fpts(0, startEle);

  #ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
            nSpts, 1.0, &A, oppE.ldim(), &B, F_spts.ldim(), 0.0, &C, tempF_fpts.ldim());
  #else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
            nSpts, 1.0, &A, oppE.ldim(), &B, F_spts.ldim(), 0.0, &C, tempF_fpts.ldim());
  #endif
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          for (unsigned int fpt = 0; fpt < nFpts; fpt++)
          {
            int gfpt = geo->fpt2gfpt(fpt,ele);

            /* Check if flux point is on ghost edge */
            if (gfpt == -1)
              continue;

            unsigned int slot = geo->fpt2gfpt_slot(fpt,ele);
            double fac = (slot == 1) ? -1 : 1; // factor to negate normal if "right" element (slot = 1)
            dFn_fpts(fpt,ele,var) -= tempF_fpts(fpt,ele) * fac * faces->norm(gfpt,dim) * faces->dA(gfpt);
          }
        }
      }
    }
  }
  else
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto &A = oppE_Fn(0, 0, dim);
        auto &B = F_spts(0, startEle, var, dim);
        auto &C = dFn_fpts(0, startEle, var);

  #ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
            nSpts, 1.0, &A, oppE.ldim(), &B, F_spts.ldim(), -1.0, &C, dFn_fpts.ldim());
  #else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nFpts, endEle - startEle,
            nSpts, 1.0, &A, oppE.ldim(), &B, F_spts.ldim(), -1.0, &C, dFn_fpts.ldim());
  #endif
      }
    }
  }
#endif

#ifdef _GPU
  cudaDeviceSynchronize();
  check_error();

  device_copy(dFn_fpts_d, Fcomm_d, Fcomm_d.size());
cudaDeviceSynchronize();
  check_error();

  extrapolate_Fn_wrapper(oppE_d, F_spts_d, tempF_fpts_d, dFn_fpts_d,
      faces->norm_d, faces->dA_d, geo->fpt2gfpt_d, geo->fpt2gfpt_slot_d, nSpts,
      nFpts, nEles, nDims, nVars, input->motion);

  check_error();
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
    auto *A = oppD_d.get_ptr(0, 0, dim);
    auto *B = U_spts_d.get_ptr(0, startEle, 0);
    auto *C = dU_spts_d.get_ptr(0, startEle, 0, dim);

    /* Compute contribution to derivative from solution at solution points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nSpts, 1.0, A, oppD_d.ldim(), 
        B, U_spts_d.ldim(), 0.0, C, dU_spts_d.ldim());

    check_error();

    A = oppD_fpts_d.get_ptr(0, 0, dim);
    B = Ucomm_d.get_ptr(0, startEle, 0);
    C = dU_spts_d.get_ptr(0, startEle, 0, dim);

    /* Compute contribution to derivative from common solution at flux points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nFpts, 1.0, A, oppD_fpts_d.ldim(),
        B, Ucomm_d.ldim(), 1.0, C , dU_spts_d.ldim());

    check_error();
  }
#endif

}

void Elements::compute_dU_spts(unsigned int startEle, unsigned int endEle)
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

#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    auto *A = oppD_d.get_ptr(0, 0, dim);
    auto *B = U_spts_d.get_ptr(0, startEle, 0);
    auto *C = dU_spts_d.get_ptr(0, startEle, 0, dim);

    /* Compute contribution to derivative from solution at solution points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nSpts, 1.0, A, oppD_d.ldim(), 
        B, U_spts_d.ldim(), 0.0, C, dU_spts_d.ldim());

    check_error();
  }
#endif

}

void Elements::compute_dU_fpts(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
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
    auto *A = oppD_fpts_d.get_ptr(0, 0, dim);
    auto *B = Ucomm_d.get_ptr(0, startEle, 0);
    auto *C = dU_spts_d.get_ptr(0, startEle, 0, dim);

    /* Compute contribution to derivative from common solution at flux points */
    cublasDGEMM_wrapper(nSpts, nEles * nVars, nFpts, 1.0,
        A, oppD_fpts_d.ldim(), B, Ucomm_d.ldim(), 1.0, C, dU_spts_d.ldim());

    check_error();
  }
#endif

}

void Elements::compute_dU0(unsigned int startEle, unsigned int endEle)
{
  /* If running a viscous simulation, use dU that was already computed
   * NOTE: the point is to keep a copy of dU wrt reference domain coords */
  if (input->viscous)
  {
#ifdef _CPU
    std::copy(dU_spts.data(),dU_spts.data()+dU_spts.size(),dUr_spts.data());
#endif

#ifdef _GPU
    device_copy(dUr_spts_d, dU_spts_d, dU_spts_d.size());
    check_error();
#endif

    return;
  }

  /* Compute derivative of solution at solution points (Order-P FR basis) */
#ifdef _CPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      auto &A = oppD0(0, 0, dim);
      auto &B = U_spts(0, startEle, var);
      auto &C = dUr_spts(0, startEle, var, dim);

#ifdef _OMP
      omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts,
                        endEle - startEle, nSpts, 1.0, &A, oppD.ldim(), &B, U_spts.ldim(),
                        0.0, &C, dU_spts.ldim());
#else
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts,
                  endEle - startEle, nSpts, 1.0, &A, oppD0.ldim(), &B, U_spts.ldim(),
                  0.0, &C, dUr_spts.ldim());
#endif
    }
  }
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      auto *A = oppD0_d.get_ptr(0, 0, dim);
      auto *B = U_spts_d.get_ptr(0, startEle, var);
      auto *C = dUr_spts_d.get_ptr(0, startEle, var, dim);

      cublasDGEMM_wrapper(nSpts, endEle - startEle, nSpts, 1.0, A, oppD0_d.ldim(), B,
          U_spts_d.ldim(), 0.0, C, dUr_spts_d.ldim());
    }
  }
  check_error();
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
      auto *A = oppD_d.get_ptr(0, 0, dim);
      auto *B = F_spts_d.get_ptr(0, startEle, var, dim);
      auto *C = divF_spts_d.get_ptr(0, startEle, var, stage);

      /* Compute contribution to derivative from solution at solution points */
      cublasDGEMM_wrapper(nSpts, endEle - startEle, nSpts, 1.0,
          A, oppD_d.ldim(), B, F_spts_d.ldim(), fac, C, divF_spts_d.ldim());
    }
  }

  /* Compute contribution to derivative from common solution at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto *A = oppDiv_fpts_d.get_ptr(0, 0);
    auto *B = Fcomm_d.get_ptr(0, startEle, var);
    auto *C = divF_spts_d.get_ptr(0, startEle, var, stage);


    cublasDGEMM_wrapper(nSpts, endEle - startEle,  nFpts, 1.0,
        A, oppDiv_fpts_d.ldim(), B, Fcomm_d.ldim(), 1.0, C, divF_spts_d.ldim());
  }

  check_error();
#endif
}

void Elements::compute_gradF_spts(unsigned int startEle, unsigned int endEle)
{
  int m = nSpts;
  int n = endEle - startEle;
  int k = nSpts;

#ifdef _CPU
  /* Compute contribution to divergence from flux at solution points */
  for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
  {
    for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto &A = oppD0(0, 0, dim2);
        auto &B = F_spts(0, startEle, var, dim1);
        auto &C = dF_spts(0, startEle, var, dim1, dim2);

#ifdef _OMP
        omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                          1.0, &A, oppD.ldim(), &B, F_spts.ldim(), 0., &C, dF_spts.ldim());
#else
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                    1.0, &A, oppD0.ldim(), &B, F_spts.ldim(), 0., &C, dF_spts.ldim());
#endif
      }
    }
  }
#endif

#ifdef _GPU
  // nSpts, nEles, nVars, nDims, nDims
  for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
  {
    for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
    {
      for (unsigned int var = 0; var < nVars; var++)
      {
        auto *A = oppD0_d.get_ptr(0, 0, dim2);
        auto *B = F_spts_d.get_ptr(0, startEle, var, dim1);
        auto *C = dF_spts_d.get_ptr(0, startEle, var, dim1, dim2);

        /* Compute contribution to derivative from solution at solution points */
        cublasDGEMM_wrapper(m, n, k, 1.0, A, oppD0_d.ldim(), B, F_spts_d.ldim(),
            0., C, dF_spts_d.ldim());
      }
    }
  }
  check_error();
#endif
}

void Elements::transform_gradF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  if (nDims == 2)
  {
#pragma omp parallel for collapse(2)
    for (uint spt = 0; spt < nSpts; spt++)
    {
      for (uint e = startEle; e < endEle; e++)
      {
        double A = grid_vel_spts(spt,e,1)*jaco_spts(spt,e,0,1) - grid_vel_spts(spt,e,0)*jaco_spts(spt,e,1,1);
        double B = grid_vel_spts(spt,e,0)*jaco_spts(spt,e,1,0) - grid_vel_spts(spt,e,1)*jaco_spts(spt,e,0,0);
        for (uint k = 0; k < nVars; k++)
        {
          divF_spts(spt,e,k,stage) =  dF_spts(spt,e,k,0,0)*jaco_spts(spt,e,1,1) - dF_spts(spt,e,k,1,0)*jaco_spts(spt,e,0,1) + dUr_spts(spt,e,k,0)*A;
          divF_spts(spt,e,k,stage)+= -dF_spts(spt,e,k,0,1)*jaco_spts(spt,e,1,0) + dF_spts(spt,e,k,1,1)*jaco_spts(spt,e,0,0) + dUr_spts(spt,e,k,1)*B;
        }
      }
    }
  }
  else
  {
#pragma omp parallel for collapse(2)
    for (uint spt = 0; spt < nSpts; spt++)
    {
      for (uint e = startEle; e < endEle; e++)
      {
        for (uint k = 0; k < nVars; k++)
          divF_spts(spt,e,k,stage) = 0;

        mdvector<double> Jacobian({nDims+1,nDims+1});
        Jacobian(nDims,nDims) = 1;
        for (uint i = 0; i < nDims; i++)
        {
          for (uint j = 0; j < nDims; j++)
            Jacobian(i,j) = jaco_spts(spt,e,i,j);
          Jacobian(i,nDims) = grid_vel_spts(spt,e,i);
        }

        if (tmp_S.size() != 16)
          tmp_S.resize({4,4});
        adjoint_4x4(Jacobian.data(), tmp_S.data());

        for (uint dim1 = 0; dim1 < nDims; dim1++)
          for (uint dim2 = 0; dim2 < nDims; dim2++)
            for (uint k = 0; k < nVars; k++)
              divF_spts(spt,e,k,stage) += dF_spts(spt,e,k,dim1,dim2)*tmp_S(dim2,dim1);

        for (uint dim = 0; dim < nDims; dim++)
          for (uint k = 0; k < nVars; k++)
            divF_spts(spt,e,k,stage) += dUr_spts(spt,e,k,dim)*tmp_S(dim,nDims);
      }
    }
  }
#endif

#ifdef _GPU
  if (nDims == 2)
  {
    transform_gradF_quad_wrapper(divF_spts_d, dF_spts_d, jaco_spts_d, grid_vel_spts_d,
        dUr_spts_d, nSpts, nEles, stage, input->equation, input->overset,
        geo->iblank_cell_d.data());
  }
  else
  {
    transform_gradF_hexa_wrapper(divF_spts_d, dF_spts_d, jaco_spts_d, grid_vel_spts_d,
        dUr_spts_d, nSpts, nEles, stage, input->equation, input->overset,
        geo->iblank_cell_d.data());
  }
  check_error();
#endif
}

void Elements::compute_divF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle)
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
#endif

#ifdef _GPU
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    double fac = (dim == 0) ? 0.0 : 1.0;

    for (unsigned int var = 0; var < nVars; var++)
    {
      auto *A = oppD_d.get_ptr(0, 0, dim);
      auto *B = F_spts_d.get_ptr(0, startEle, var, dim);
      auto *C = divF_spts_d.get_ptr(0, startEle, var, stage);

      /* Compute contribution to derivative from solution at solution points */
      cublasDGEMM_wrapper(nSpts, endEle - startEle, nSpts, 1.0,
          A, oppD_d.ldim(), B, F_spts_d.ldim(), fac, C, divF_spts_d.ldim(), 0);
    }
  }
  check_error();
#endif
}

void Elements::compute_divF_fpts(unsigned int stage, unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
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
  /* Compute contribution to derivative from common solution at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto *A = oppDiv_fpts_d.get_ptr(0, 0);
    auto *B = Fcomm_d.get_ptr(0, startEle, var);
    auto *C = divF_spts_d.get_ptr(0, startEle, var, stage);

    cublasDGEMM_wrapper(nSpts, endEle - startEle,  nFpts, 1.0,
        A, oppDiv_fpts_d.ldim(), B, Fcomm_d.ldim(), 1.0, C, divF_spts_d.ldim(), 0);
  }

  check_error();
#endif

}

void Elements::correct_divF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  /* Compute contribution to divergence from common flux at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto &A = oppDiv_fpts(0, 0); // Identical to FR-DG correction operator
    auto &B = dFn_fpts(0, startEle, var);
    auto &C = divF_spts(0, startEle, var, stage);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts,
        endEle - startEle, nFpts, 1.0, &A, oppDiv_fpts.ldim(), &B, dFn_fpts.ldim(), 1.0, &C, divF_spts.ldim());
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts, endEle - startEle,
        nFpts, 1.0, &A, oppDiv_fpts.ldim(), &B, dFn_fpts.ldim(), 1.0, &C, divF_spts.ldim());
#endif
  }
#endif

#ifdef _GPU
  /* Compute contribution to derivative from common solution at flux points */
  for (unsigned int var = 0; var < nVars; var++)
  {
    auto *A = oppDiv_fpts_d.get_ptr(0, 0); // Identical to FR-DG correction operator
    auto *B = dFn_fpts_d.get_ptr(0, startEle, var);
    auto *C = divF_spts_d.get_ptr(0, startEle, var, stage);

    cublasDGEMM_wrapper(nSpts, endEle - startEle,  nFpts, 1.0,
        A, oppDiv_fpts_d.ldim(), B, dFn_fpts_d.ldim(), 1.0, C, divF_spts_d.ldim(), 0);
  }

  check_error();
#endif
}

template<unsigned int nVars, unsigned int nDims, unsigned int equation>
void Elements::compute_F(unsigned int startEle, unsigned int endEle)
{
  double U[nVars];
  double F[nVars][nDims];
  double tdU[nVars][nDims];
  double inv_jaco[nDims][nDims];

  for (unsigned int ele = startEle; ele < endEle; ele++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {

      /* Get state variables and reference space gradients */
      for (unsigned int var = 0; var < nVars; var++)
      {
        U[var] = U_spts(spt, ele, var);

        if(input->viscous) 
        {
          for(unsigned int dim = 0; dim < nDims; dim++)
          {
            tdU[var][dim] = dU_spts(spt, ele, var, dim);
          }
        }
      }

      /* Get metric terms */
      double inv_jaco[nDims][nDims];

      for (int dim1 = 0; dim1 < nDims; dim1++)
        for (int dim2 = 0; dim2 < nDims; dim2++)
          inv_jaco[dim1][dim2] = inv_jaco_spts(spt, ele, dim1, dim2);

      double dU[nVars][nDims] = {{0.0}};
      if (input->viscous)
      {
        double inv_jaco_det = 1.0 / jaco_det_spts(spt,ele);

        /* Transform gradient to physical space */
        for (unsigned int var = 0; var < nVars; var++)
        {
          for (int dim1 = 0; dim1 < nDims; dim1++)
          {
            for (int dim2 = 0; dim2 < nDims; dim2++)
            {
              dU[var][dim1] += (tdU[var][dim2] * inv_jaco[dim2][dim1]);
            }

            dU[var][dim1] *= inv_jaco_det;

            /* Write physical gradient to global memory */
            dU_spts(spt, ele, var, dim1) = dU[var][dim1];

          }
        }
      }

      /* Compute fluxes */
      if (equation == AdvDiff)
      {
        double A[nDims];
        for(unsigned int dim = 0; dim < nDims; dim++)
          A[dim] = input->AdvDiff_A(dim);

        compute_Fconv_AdvDiff<nVars, nDims>(U, F, A);
        if(input->viscous) compute_Fvisc_AdvDiff_add<nVars, nDims>(dU, F, input->AdvDiff_D);

      }
      else if (equation == Burgers)
      {
        compute_Fconv_Burgers<nVars, nDims>(U, F);
        if(input->viscous) compute_Fvisc_AdvDiff_add<nVars, nDims>(dU, F, input->AdvDiff_D);
      }
      else if (equation == EulerNS)
      {
        double P;
        compute_Fconv_EulerNS<nVars, nDims>(U, F, P, input->gamma);
        if(input->viscous) compute_Fvisc_EulerNS_add<nVars, nDims>(U, dU, F, input->gamma, input->prandtl, input->mu,
            input->rt, input->c_sth, input->fix_vis);
      }

      if (!input->motion)
      {
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
            F_spts(spt, ele, var, dim) = tF[var][dim];
          }
        }
      }
      else
      {
        /* Write out physical fluxes */
        for (unsigned int var = 0; var < nVars; var++)
        {
          for(unsigned int dim = 0; dim < nDims; dim++)
          {
            F_spts(spt, ele, var, dim) = F[var][dim];
          }
        }
      }
    }
  }

}

void Elements::compute_F(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
  if (input->equation == AdvDiff)
  {
    if (nDims == 2)
      compute_F<1, 2, AdvDiff>(startEle, endEle);
    else if (nDims == 3)
      compute_F<1, 3, AdvDiff>(startEle, endEle);
  }
  else if (input->equation == Burgers)
  {
    if (nDims == 2)
      compute_F<1, 2, Burgers>(startEle, endEle);
    else if (nDims == 3)
      compute_F<1, 3, Burgers>(startEle, endEle);
  }
  else if (input->equation == EulerNS)
  {
    if (nDims == 2)
      compute_F<4, 2, EulerNS>(startEle, endEle);
    else if (nDims == 3)
      compute_F<5, 3, EulerNS>(startEle, endEle);
  }
#endif

#ifdef _GPU
  compute_F_wrapper(F_spts_d, U_spts_d, dU_spts_d, inv_jaco_spts_d, jaco_det_spts_d, nSpts, nEles, nDims, input->equation, input->AdvDiff_A_d, 
      input->AdvDiff_D, input->gamma, input->prandtl, input->mu, input->c_sth, input->rt, 
      input->fix_vis, input->viscous, startEle, endEle, input->overset, geo->iblank_cell_d.data(), input->motion);

  check_error();
#endif
}


void Elements::compute_dFdUconv()
{
  if (input->equation == AdvDiff)
  {
#ifdef _CPU
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
#endif

#ifdef _GPU
    compute_dFdUconv_spts_AdvDiff_wrapper(dFdU_spts_d, nSpts, nEles, nDims, input->AdvDiff_A_d);
    check_error();
#endif

  }

  else if (input->equation == Burgers)
  {
#ifdef _CPU
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
#endif

#ifdef _GPU
    compute_dFdUconv_spts_Burgers_wrapper(dFdU_spts_d, U_spts_d, nSpts, nEles, nDims);
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
#pragma omp parallel for collapse(2)
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          /* Primitive Variables */
          double rho = U_spts(spt, ele, 0);
          double u = U_spts(spt, ele, 1) / U_spts(spt, ele, 0);
          double v = U_spts(spt, ele, 2) / U_spts(spt, ele, 0);
          double w = U_spts(spt, ele, 3) / U_spts(spt, ele, 0);
          double e = U_spts(spt, ele, 4);
          double gam = input->gamma;

          /* Set convective dFdU values in the x-direction */
          dFdU_spts(spt, ele, 0, 0, 0) = 0;
          dFdU_spts(spt, ele, 1, 0, 0) = 0.5 * ((gam-3.0) * u*u + (gam-1.0) * (v*v + w*w));
          dFdU_spts(spt, ele, 2, 0, 0) = -u * v;
          dFdU_spts(spt, ele, 3, 0, 0) = -u * w;
          dFdU_spts(spt, ele, 4, 0, 0) = -gam * e * u / rho + (gam-1.0) * u * (u*u + v*v + w*w);

          dFdU_spts(spt, ele, 0, 1, 0) = 1;
          dFdU_spts(spt, ele, 1, 1, 0) = (3.0-gam) * u;
          dFdU_spts(spt, ele, 2, 1, 0) = v;
          dFdU_spts(spt, ele, 3, 1, 0) = w;
          dFdU_spts(spt, ele, 4, 1, 0) = gam * e / rho + 0.5 * (1.0-gam) * (3.0*u*u + v*v + w*w);

          dFdU_spts(spt, ele, 0, 2, 0) = 0;
          dFdU_spts(spt, ele, 1, 2, 0) = (1.0-gam) * v;
          dFdU_spts(spt, ele, 2, 2, 0) = u;
          dFdU_spts(spt, ele, 3, 2, 0) = 0;
          dFdU_spts(spt, ele, 4, 2, 0) = (1.0-gam) * u * v;

          dFdU_spts(spt, ele, 0, 3, 0) = 0;
          dFdU_spts(spt, ele, 1, 3, 0) = (1.0-gam) * w;
          dFdU_spts(spt, ele, 2, 3, 0) = 0;
          dFdU_spts(spt, ele, 3, 3, 0) = u;
          dFdU_spts(spt, ele, 4, 3, 0) = (1.0-gam) * u * w;

          dFdU_spts(spt, ele, 0, 4, 0) = 0;
          dFdU_spts(spt, ele, 1, 4, 0) = (gam-1.0);
          dFdU_spts(spt, ele, 2, 4, 0) = 0;
          dFdU_spts(spt, ele, 3, 4, 0) = 0;
          dFdU_spts(spt, ele, 4, 4, 0) = gam * u;

          /* Set convective dFdU values in the y-direction */
          dFdU_spts(spt, ele, 0, 0, 1) = 0;
          dFdU_spts(spt, ele, 1, 0, 1) = -u * v;
          dFdU_spts(spt, ele, 2, 0, 1) = 0.5 * ((gam-1.0) * (u*u + w*w) + (gam-3.0) * v*v);
          dFdU_spts(spt, ele, 3, 0, 1) = -v * w;
          dFdU_spts(spt, ele, 4, 0, 1) = -gam * e * v / rho + (gam-1.0) * v * (u*u + v*v + w*w);

          dFdU_spts(spt, ele, 0, 1, 1) = 0;
          dFdU_spts(spt, ele, 1, 1, 1) = v;
          dFdU_spts(spt, ele, 2, 1, 1) = (1.0-gam) * u;
          dFdU_spts(spt, ele, 3, 1, 1) = 0;
          dFdU_spts(spt, ele, 4, 1, 1) = (1.0-gam) * u * v;

          dFdU_spts(spt, ele, 0, 2, 1) = 1;
          dFdU_spts(spt, ele, 1, 2, 1) = u;
          dFdU_spts(spt, ele, 2, 2, 1) = (3.0-gam) * v;
          dFdU_spts(spt, ele, 3, 2, 1) = w;
          dFdU_spts(spt, ele, 4, 2, 1) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + 3.0*v*v + w*w);

          dFdU_spts(spt, ele, 0, 3, 1) = 0;
          dFdU_spts(spt, ele, 1, 3, 1) = 0;
          dFdU_spts(spt, ele, 2, 3, 1) = (1.0-gam) * w;
          dFdU_spts(spt, ele, 3, 3, 1) = v;
          dFdU_spts(spt, ele, 4, 3, 1) = (1.0-gam) * v * w;

          dFdU_spts(spt, ele, 0, 4, 1) = 0;
          dFdU_spts(spt, ele, 1, 4, 1) = 0;
          dFdU_spts(spt, ele, 2, 4, 1) = (gam-1.0);
          dFdU_spts(spt, ele, 3, 4, 1) = 0;
          dFdU_spts(spt, ele, 4, 4, 1) = gam * v;

          /* Set convective dFdU values in the z-direction */
          dFdU_spts(spt, ele, 0, 0, 2) = 0;
          dFdU_spts(spt, ele, 1, 0, 2) = -u * w;
          dFdU_spts(spt, ele, 2, 0, 2) = -v * w;
          dFdU_spts(spt, ele, 3, 0, 2) = 0.5 * ((gam-1.0) * (u*u + v*v) + (gam-3.0) * w*w);
          dFdU_spts(spt, ele, 4, 0, 2) = -gam * e * w / rho + (gam-1.0) * w * (u*u + v*v + w*w);

          dFdU_spts(spt, ele, 0, 1, 2) = 0;
          dFdU_spts(spt, ele, 1, 1, 2) = w;
          dFdU_spts(spt, ele, 2, 1, 2) = 0;
          dFdU_spts(spt, ele, 3, 1, 2) = (1.0-gam) * u;
          dFdU_spts(spt, ele, 4, 1, 2) = (1.0-gam) * u * w;

          dFdU_spts(spt, ele, 0, 2, 2) = 0;
          dFdU_spts(spt, ele, 1, 2, 2) = 0;
          dFdU_spts(spt, ele, 2, 2, 2) = w;
          dFdU_spts(spt, ele, 3, 2, 2) = (1.0-gam) * v;
          dFdU_spts(spt, ele, 4, 2, 2) = (1.0-gam) * v * w;

          dFdU_spts(spt, ele, 0, 3, 2) = 1;
          dFdU_spts(spt, ele, 1, 3, 2) = u;
          dFdU_spts(spt, ele, 2, 3, 2) = v;
          dFdU_spts(spt, ele, 3, 3, 2) = (3.0-gam) * w;
          dFdU_spts(spt, ele, 4, 3, 2) = gam * e / rho + 0.5 * (1.0-gam) * (u*u + v*v + 3.0*w*w);

          dFdU_spts(spt, ele, 0, 4, 2) = 0;
          dFdU_spts(spt, ele, 1, 4, 2) = 0;
          dFdU_spts(spt, ele, 2, 4, 2) = 0;
          dFdU_spts(spt, ele, 3, 4, 2) = (gam-1.0);
          dFdU_spts(spt, ele, 4, 4, 2) = gam * w;
        }
      }
    }
#endif

#ifdef _GPU
    compute_dFdUconv_spts_EulerNS_wrapper(dFdU_spts_d, U_spts_d, nSpts, nEles, nDims, input->gamma);
    check_error();
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

#ifdef _CPU
void Elements::compute_localLHS(mdvector<double> &dt, unsigned int startEle, unsigned int endEle, unsigned int color)
#endif
#ifdef _GPU
void Elements::compute_localLHS(mdvector_gpu<double> &dt_d, unsigned int startEle, unsigned int endEle, unsigned int color)
#endif
{

#ifdef _CPU
  /* Compute LHS */
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      int idx = 0; /* Index for color local LHS */
      for (unsigned int ele = startEle; ele < endEle; ele++)
      {
        /* Compute center inviscid LHS implicit Jacobian */
        for (unsigned int dim = 0; dim < nDims; dim++)
        {
          auto *A = &oppD(0, 0, dim);
          auto *B = &dFdU_spts(0, ele, ni, nj, dim);
          auto *C = &LHSs[color - 1](0, ni, 0, nj, idx);

          double fac = (dim == 0) ? 0.0 : 1.0;

          dgmm(nSpts, nSpts, 1, A, oppD.ldim(), B, 0, fac, C, nSpts*nVars);
        }

        auto *A = &oppDiv_fpts(0, 0);
        auto *B = &dFcdU_fpts(0, ele, ni, nj, 0);
        auto *C = &CtempSF(0, 0);
        dgmm(nSpts, nFpts, 1, A, oppDiv_fpts.ldim(), B, 0, 0, C, nSpts);

        A = &CtempSF(0, 0);
        B = &oppE(0, 0);
        C = &LHSs[color - 1](0, ni, 0, nj, idx);
        gemm(nSpts, nSpts, nFpts, 1, A, nSpts, B, oppE.ldim(), 1, C, nSpts*nVars);

        /* Compute center viscous LHS implicit Jacobian */
        if (input->viscous)
        {
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

              Cvisc0(i, j, 0) = Cvisc0(i, j, 0) * jaco_spts(i, ele, 1, 1) -
                                Cvisc0(i, j, 1) * jaco_spts(i, ele, 1, 0);

              Cvisc0(i, j, 1) = Cvisc0(i, j, 1) * jaco_spts(i, ele, 0, 0) -
                                Cvisc0temp * jaco_spts(i, ele, 0, 1);

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

              CdFddU0(i, j, 0) = CdFddU0(i, j, 0) * jaco_spts(i, ele, 1, 1) -
                                 CdFddU0(i, j, 1) * jaco_spts(i, ele, 0, 1);

              CdFddU0(i, j, 1) = CdFddU0(i, j, 1) * jaco_spts(i, ele, 0, 0) -
                                 CdFddU0temp * jaco_spts(i, ele, 1, 0);
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
                LHSs[color - 1](i, ni, j, nj, idx) += val;
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

          for (unsigned int j = 0; j < nSpts; j++)
          {
            for (unsigned int i = 0; i < nSpts; i++)
            {
              double val = 0;
              for (unsigned int k = 0; k < nFpts; k++)
              {
                val += oppDiv_fpts(i, k) * CtempFS(k, j);
              }
              LHSs[color - 1](i, ni, j, nj, idx) += val;
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

                      CviscN(i, j, 0, face2) = CviscN(i, j, 0, face2) * jaco_spts(i, eleN, 1, 1) -
                                               CviscN(i, j, 1, face2) * jaco_spts(i, eleN, 1, 0);

                      CviscN(i, j, 1, face2) = CviscN(i, j, 1, face2) * jaco_spts(i, eleN, 0, 0) -
                                               CviscNtemp * jaco_spts(i, eleN, 0, 1);

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

                  for (unsigned int j = 0; j < nSpts; j++)
                  {
                    for (unsigned int i = 0; i < nSpts; i++)
                    {
                      double val = 0;
                      for (unsigned int k = 0; k < nSpts1D; k++)
                      {
                        unsigned int ind = face * nSpts1D + k;
                        val += oppDiv_fpts(i, ind) * CtempFSN(k, j);
                      }
                      LHSs[color - 1](i, ni, j, nj, idx) += val;
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
              LHSs[color - 1](i, ni, j, nj, idx) = dt(0) * LHSs[color - 1](i, ni, j, nj, idx) / jaco_det_spts(i, ele);
            }

            else
            {
              LHSs[color - 1](i, ni, j, nj, idx) = dt(ele) * LHSs[color - 1](i, ni, j, nj, idx) / jaco_det_spts(i, ele);
            }

            if (i == j && ni == nj)
            {
              LHSs[color - 1](i, ni, j, nj, idx) += 1;
            }
          }
        }
        idx++;
      }
    }
  }
#endif

#ifdef _GPU
  /* Compute center inviscid LHS implicit Jacobian */

  /* Add oppDiv_fpts scaled by dFcdU_fpts, multiplied by oppE, to LHS */
  add_scaled_oppDiv_times_oppE_wrapper(LHS_d, oppDiv_fpts_d, oppE_d, dFcdU_fpts_d, nSpts, nFpts, nVars, nEles, startEle, endEle);

  /* Add oppD scaled by dFdU_spts to LHS */
  add_scaled_oppD_wrapper(LHS_d, oppD_d, dFdU_spts_d, nSpts, nVars, nEles, nDims, startEle, endEle);

  /* Finalize LHS (scale by jacobian, dt, and add identity) */
  finalize_LHS_wrapper(LHS_d, dt_d, jaco_det_spts_d, nSpts, nVars, nEles, input->dt_type, startEle, endEle);

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

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        /* Get quadrature weight */
        unsigned int i = idx_spts(spt,0);
        unsigned int j = idx_spts(spt,1);
        double weight = weights_spts(i) * weights_spts(j);

        if (nDims == 3)
          weight *= weights_spts(idx_spts(spt,2));

        sum += weight * jaco_det_spts(spt, ele) * U_spts(spt, ele, n);
      }

      Uavg(ele, n) = sum / vol(ele); 

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
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momF += U_spts(spt, ele, dim + 1) * U_spts(spt, ele, dim + 1);

      momF /= U_spts(spt, ele, 0);

      double P = (input->gamma - 1.0) * (U_spts(spt, ele, nDims + 1) - 0.5 * momF);

      double tau = P - input->exps0 * std::pow(rho, input->gamma);
      minTau = std::min(minTau, tau);

    }
    
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      double rho = U_fpts(fpt, ele, 0);
      double momF = 0.0;
      for (unsigned int dim = 0; dim < nDims; dim++)
        momF += U_fpts(fpt, ele, dim + 1) * U_fpts(fpt, ele, dim + 1);

      momF /= U_fpts(fpt, ele, 0);
      double P = (input->gamma - 1.0) * (U_fpts(fpt, ele, nDims + 1) - 0.5 * momF);

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

      double e = Uavg(ele, nDims + 1);
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

void Elements::move(std::shared_ptr<Faces> faces)
{
#ifdef _CPU
  if (input->motion_type == RIGID_BODY)
  {
    // Update grid position based on rigid-body motion: CG offset + rotation
    for (unsigned int i = 0; i < geo->nNodes; i++)
      for (unsigned int d = 0; d < nDims; d++)
        geo->coord_nodes(d,i) = geo->x_cg(d);

    auto &A = geo->Rmat(0,0);
    auto &B = geo->coords_init(0,0);
    auto &C = geo->coord_nodes(0,0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nDims, geo->nNodes, nDims,
                      1.0, &A, nDims, &B, nDims, 1.0, &C, nDims);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nDims, geo->nNodes, nDims,
                1.0, &A, nDims, &B, nDims, 1.0, &C, nDims);
#endif

    // Update grid velocity based on 'spin' matrix (omega cross r)
    for (unsigned int i = 0; i < geo->nNodes; i++)
      for (unsigned int d = 0; d < nDims; d++)
        geo->grid_vel_nodes(d,i) = geo->vel_cg(d);

    auto &Av = geo->Wmat(0,0);
    auto &Bv = geo->coords_init(0,0);
    auto &Cv = geo->grid_vel_nodes(0,0);

#ifdef _OMP
    omp_blocked_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nDims, geo->nNodes, nDims,
                      1.0, &Av, nDims, &Bv, nDims, 1.0, &Cv, nDims);
#else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nDims, geo->nNodes, nDims,
                1.0, &Av, nDims, &Bv, nDims, 1.0, &Cv, nDims);
#endif
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

    // Velocities -- DOUBLE CHECK THIS IS CORRECT!
    update_nodes_rigid_wrapper(geo->coords_init_d, geo->grid_vel_nodes_d, geo->Wmat_d,
        geo->vel_cg_d, geo->nNodes, geo->nDims);
  }

  update_coords_wrapper(nodes_d, geo->coord_nodes_d, shape_spts_d,
      shape_fpts_d, coord_spts_d, coord_fpts_d, faces->coord_d,
      geo->ele2nodes_d, geo->fpt2gfpt_d, nSpts, nFpts, nNodes, nEles, nDims);

  update_coords_wrapper(grid_vel_nodes_d, geo->grid_vel_nodes_d, shape_spts_d,
      shape_fpts_d, grid_vel_spts_d, grid_vel_fpts_d, faces->Vg_d,
      geo->ele2nodes_d, geo->fpt2gfpt_d, nSpts, nFpts, nNodes, nEles, nDims);

  if (input->motion_type == RIGID_BODY)
  {
    /// TODO: kernels that take in 'body-coords' transforms and applies rotation matrix
    /* At times 1 and 2, jaco_1 = R_1 * jaco_0;  jaco_2 = R_2 * jaco_0
     * So to update from 1 to 2, jaco_2 = R_2 * R_1^inv * jaco_1
     * Where R is the matrix form of the body's roation quaternion */
    update_transforms_rigid_wrapper(jaco_spts_init_d, inv_jaco_spts_init_d, jaco_spts_d, inv_jaco_spts_d,
        faces->norm_init_d, faces->norm_d, geo->Rmat_d, nSpts, faces->nFpts, nEles, nDims, input->viscous);
  }
  else
  {
    calc_transforms_wrapper(nodes_d, jaco_spts_d, jaco_fpts_d, inv_jaco_spts_d,
                            inv_jaco_fpts_d, jaco_det_spts_d, dshape_spts_d, dshape_fpts_d, nSpts,
                            nFpts, nNodes, nEles, nDims);

    calc_normals_wrapper(faces->norm_d, faces->dA_d, inv_jaco_fpts_d, tnorm_d,
                         geo->fpt2gfpt_d, geo->fpt2gfpt_slot_d, nFpts, nEles, nDims);

    if (input->CFL_type == 2)
      update_h_ref_wrapper(h_ref_d, coord_fpts_d, nEles, nFpts, nSpts1D, nDims);
  }
#endif
}

void Elements::update_point_coords(std::shared_ptr<Faces> faces)
{
#ifdef _GPU
  // Copy back, since updated only on GPU
  geo->coord_nodes = geo->coord_nodes_d;
#endif

#pragma omp parallel for collapse(3)
  for (uint node = 0; node < nNodes; node++)
    for (uint ele = 0; ele < nEles; ele++)
      for (uint dim = 0; dim < nDims; dim++)
        nodes(node, ele, dim) = geo->coord_nodes(dim,geo->ele2nodes(node,ele));

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
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#endif

  /* Setup physical coordinates at flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = coord_fpts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasTrans, mf, n, k,
              1.0, &Af, k, &B, n, 0.0, &Cf, mf);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
#endif

  /* Setup physical coordinates at flux points [in faces class] */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        int gfpt = geo->fpt2gfpt(fpt,ele);
        /* Check if on ghost edge */
        if (gfpt != -1)
          faces->coord(gfpt, dim) = coord_fpts(fpt,ele,dim);
      }
    }
  }

  if (input->CFL_type == 2)
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

          double dx = coord_fpts(fpt1, ele, 0) - coord_fpts(fpt2, ele, 0);
          double dy = coord_fpts(fpt1, ele, 1) - coord_fpts(fpt2, ele, 1);
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
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, n, 0.0, &Cp, mp);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mp, n, k,
              1.0, &Ap, k, &B, k, 0.0, &Cp, mp);
#endif

  /* Setup physical coordinates at quadrature points */
  auto &Aq = shape_qpts(0,0);
  auto &Cq = coord_qpts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mq, n, k,
              1.0, &Aq, k, &B, k, 0.0, &Cq, mq);
#endif
}

void Elements::update_grid_velocities(std::shared_ptr<Faces> faces)
{
#pragma omp parallel for collapse(3)
  for (uint node = 0; node < nNodes; node++)
    for (uint ele = 0; ele < nEles; ele++)
      for (uint dim = 0; dim < nDims; dim++)
        grid_vel_nodes(node, ele, dim) = geo->grid_vel_nodes(dim, geo->ele2nodes(node,ele));

  int ms = nSpts;
  int mf = nFpts;
  int k = nNodes;
  int n = nEles * nDims;

  auto &B = grid_vel_nodes(0,0,0);

  /* Interpolate grid velocities to solution points */
  auto &As = shape_spts(0,0);
  auto &Cs = grid_vel_spts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, ms, n, k,
              1.0, &As, k, &B, k, 0.0, &Cs, ms);
#endif

  /* Interpolate grid velocities to flux points */
  auto &Af = shape_fpts(0,0);
  auto &Cf = grid_vel_fpts(0,0,0);
#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, mf, n, k,
              1.0, &Af, k, &B, k, 0.0, &Cf, mf);
#endif

  /* Store grid velocity in face class */
#pragma omp parallel for collapse(2)
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfpt(fpt,ele);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1)
        continue;

      unsigned int slot = geo->fpt2gfpt_slot(fpt,ele);

      if (slot != 0)
        continue;

      for (uint dim = 0; dim < nDims; dim++)
      {
        faces->Vg(gfpt, dim) = grid_vel_fpts(fpt, ele, dim);
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

#ifdef _OMP
  omp_blocked_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k,
              1.0, &A, k, &B, k, 0.0, &C, m);
#else
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, k,
              1.0, &A, k, &B, k, 0.0, &C, m);
#endif
}

std::vector<double> Elements::getBoundingBox(int ele)
{
  std::vector<double> bbox = { INFINITY, INFINITY, INFINITY,
                              -INFINITY,-INFINITY,-INFINITY};

  for (unsigned int node = 0; node < nNodes; node++)
  {
    unsigned int nd = geo->ele2nodes(node, ele);
    for (int dim = 0; dim < nDims; dim++)
    {
      double pos = geo->coord_nodes(dim,nd);
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
    unsigned int nd = geo->ele2nodes(node, ele);
    for (int dim = 0; dim < nDims; dim++)
    {
      double pos = geo->coord_nodes(dim,nd);
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
    tmp_grad.resize({nDims, nDims});
    tmp_ginv.resize({nDims, nDims});
    tmp_coords.resize({nDims, nNodes});
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
      tmp_coords(d,nd) = geo->coord_nodes(d, geo->ele2nodes(nd, ele));

  while (norm > tol && iter < iterMax)
  {
    calc_shape(tmp_shape, shape_order, rst);
    calc_d_shape(tmp_dshape, shape_order, rst);

    point dx(xyz[0],xyz[1],xyz[2]);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nDims, nDims, nNodes,
        1.0, tmp_coords.data(), nDims, tmp_dshape.data(), nNodes, 0.0, tmp_grad.data(), nDims);

    for (int node = 0; node < nNodes; node++)
      for (int i = 0; i < 3; i++)
        dx[i] -= tmp_shape(node)*tmp_coords(i,node);

    double detJ = det_3x3(tmp_grad.data());

    adjoint_3x3(tmp_grad.data(),tmp_ginv.data());

    double delta[3] = {0,0,0};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        delta[i] += tmp_ginv(i,j)*dx[j]/detJ;

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
  for (unsigned int spt = 0; spt < nSpts; spt++)
    weights[spt] = this->calc_nodal_basis(spt, rst);
}

#ifdef _GPU
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

  U_unblank_d.assign({nVars, nSpts, nCells}, data, 3);

  if (input->motion || input->iter <= input->initIter+1) /// TODO: double-check
    unblankIDs_d.assign({nCells}, cellIDs, 3);

  unpack_unblank_u_wrapper(U_unblank_d,U_spts_d,unblankIDs_d,geo->iblank_cell_d,
      nCells,nSpts,nVars,3);

  check_error();
}

#endif
