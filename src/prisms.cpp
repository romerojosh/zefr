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

#include <cmath>
#include <iostream>
#include <string>

#include "faces.hpp"
#include "geometry.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "prisms.hpp"
#include "funcs.hpp"

extern "C" {
#include "cblas.h"
}

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

Prisms::Prisms(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order)
{
  etype = PRI;
  this->geo = geo;
  this->input = input;  
  this->elesObjID = elesObjID;
  this->startEle = startEle;
  this->endEle = endEle;
  this->nEles = endEle - startEle;
  this->nQpts = 84; /// TODO // Note: Fixing quadrature points to Shunn-Hamm 84 point rule

  if (input->error_freq == 0) this->nQpts = 0; // disable allocation if not needed

  /* Generic tetrahedral geometry */
  nDims = 3;
  nFaces = 5;
  nNodes = geo->nNodesPerEleBT[PRI];
  
  /* If order argument is not provided, use order in input file */
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;

  nSpts1D = this->order + 1;
  nFptsTri = (this->order + 1) * (this->order + 2) / 2;
  nSpts2D = nFptsTri;
  nSpts = nSpts1D * nSpts2D;
  nFptsQuad = nSpts1D * nSpts1D;
  nFptsPerFace = nFptsQuad;
  nFpts_face = {nFptsTri, nFptsTri, nFptsQuad, nFptsQuad, nFptsQuad};
  nFpts = 2 * nFptsTri + 3 * nFptsQuad;
  nPpts = nSpts;
  
  if (input->equation == AdvDiff)
  {
    nVars = 1;
  }
  else if (input->equation == EulerNS)
  {
    nVars = 5;
  }
  else
  {
    ThrowException("Equation not recognized: " + input->equation);
  }
  
}

void Prisms::set_locs()
{
  /* Allocate memory for point location structures */
  loc_qpts.assign({nQpts,nDims}); 

  /* Get positions of points in 1D and 2D */
  if (input->spt_type != "Legendre")
    ThrowException("spt_type not recognized: " + input->spt_type);

  loc_spts_1D = Gauss_Legendre_pts(order+1); // loc_spts_1D used when generating filter matrices only
  loc_fpts_2D = WS_Tri_pts(order);

  weights_fptsTri = WS_Tri_weights(order);

  /* Setup solution point locations and quadrature weights */
  weights_fptsQuad.assign({nFptsQuad});
  auto weights_1D = Gauss_Legendre_weights(order+1);

  for (int i = 0; i < nSpts1D; i++)
    for (int j = 0; j < nSpts1D; j++)
      weights_fptsQuad(i*nSpts1D+j) = weights_1D[i] * weights_1D[j];

  weights_fpts.assign({nFptsTri+nFptsQuad});

  for (unsigned int fpt = 0; fpt < nFptsTri; fpt++)
    weights_fpts(fpt) = weights_fptsTri(fpt);

  for (unsigned int fpt = 0; fpt < nFptsQuad; fpt++)
    weights_fpts(nFptsTri+fpt) = weights_fptsQuad(fpt);

  loc_spts.assign({nSpts, nDims});
  weights_spts.assign({nSpts});

  int spt = 0;
  for (int i = 0; i < nSpts1D; i++)
  {
    for (int j = 0; j < nSpts2D; j++)
    {
      loc_spts(spt, 0) = loc_fpts_2D(j, 0);
      loc_spts(spt, 1) = loc_fpts_2D(j, 1);
      loc_spts(spt, 2) = loc_spts_1D[i];

      weights_spts(spt) = weights_fptsTri(j) * weights_1D[i];

      spt++;
    }
  }

  /* Setup flux point locations */
  loc_fpts.assign({nFpts,nDims});
  unsigned int fpt = 0;
  for (unsigned int i = 0; i < 2; i++)
  {
    for (unsigned int j = 0; j < nFptsTri; j++)
    {
      switch(i)
      {
        case 0: /* Bottom */
          loc_fpts(fpt,0) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,1) = loc_fpts_2D(j, 1);
          loc_fpts(fpt,2) = -1.0;
          break;

        case 1: /* Top */
          loc_fpts(fpt,0) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,1) = loc_fpts_2D(j, 1);
          loc_fpts(fpt,2) = 1.0;
          break;
      }

      fpt++;
    }
  }
  
  for (unsigned int i = 2; i < 5; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      for (unsigned int k = 0; k < nSpts1D; k++)
      {
        switch(i)
        {
          case 2: /* Front */
            loc_fpts(fpt,0) = loc_spts_1D[k];
            loc_fpts(fpt,1) = -1.0;
            loc_fpts(fpt,2) = loc_spts_1D[j];
            break;

          case 3: /* Left */
            loc_fpts(fpt,0) = -1.0;
            loc_fpts(fpt,1) = loc_spts_1D[k];
            loc_fpts(fpt,2) = loc_spts_1D[j];
            break;

          case 4: /* Angled Face */
            loc_fpts(fpt,0) = -loc_spts_1D[k];
            loc_fpts(fpt,1) = loc_spts_1D[k];
            loc_fpts(fpt,2) = loc_spts_1D[j];
            break;
        }

        fpt++;
      }
    }
  }

  /* Setup plot point locations */
  auto loc_ppts_1D = Shape_pts(order); unsigned int nPpts1D = loc_ppts_1D.size();
  loc_ppts.assign({nPpts,nDims});

  unsigned int ppt = 0;
  for (unsigned int k = 0; k < nPpts1D; k++)
  {
    for (unsigned int i = 0; i < nPpts1D; i++)
    {
      for (unsigned int j = 0; j < nPpts1D - i; j++)
      {
        loc_ppts(ppt,0) = loc_ppts_1D[j];
        loc_ppts(ppt,1) = loc_ppts_1D[i];
        loc_ppts(ppt,2) = loc_ppts_1D[k];
        ppt++;
      }
    }
  }

  /* Setup gauss quadrature point locations and weights */ /// TODO
  //loc_qpts = WS_Tet_pts(6);
  //weights_qpts = WS_Tet_weights(6);
}

void Prisms::set_normals(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for normals */
  tnorm.assign({nFpts,nDims});
  tdA.assign({nFpts});

  /* Setup parent-space (transformed) normals at flux points */
  unsigned int fpt = 0;
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < nFptsTri; j++)
    {
      switch (i)
      {
        case 0: /* Bottom */
          tnorm(fpt,0) = 0.0;
          tnorm(fpt,1) = 0.0;
          tnorm(fpt,2) = -1.0;
          tdA(fpt) = 1.0;
          break;

        case 1: /* Top */
          tnorm(fpt,0) = 0.0;
          tnorm(fpt,1) = 0.0;
          tnorm(fpt,2) = 1.0;
          tdA(fpt) = 1.0;
          break;
      }

      fpt++;
    }
  }

  for (int i = 2; i < 5; i++)
  {
    for (int j = 0; j < nFptsQuad; j++)
    {
      switch (i)
      {
        case 2: /* Front */
          tnorm(fpt,0) = 0.0;
          tnorm(fpt,1) = -1.0;
          tnorm(fpt,2) = 0.0;
          tdA(fpt) = 1.0;
          break;

        case 3: /* Left */
          tnorm(fpt,0) = -1.0;
          tnorm(fpt,1) = 0.0;
          tnorm(fpt,2) = 0.0;
          tdA(fpt) = 1.0;
          break;

        case 4: /* Angled Face */
          tnorm(fpt,0) = 1.0/std::sqrt(2.0);
          tnorm(fpt,1) = 1.0/std::sqrt(2.0);
          tnorm(fpt,2) = 0.0;
          tdA(fpt) = std::sqrt(2);
          break;
      }

      fpt++;
    }
  }
}

void Prisms::set_vandermonde_mats()
{
  /* Set vandermonde for orthonormal Dubiner basis */
  vand.assign({nSpts, nSpts});

  for (int i = 0; i < nSpts; i++)
  {
    for (int j = 0; j < nSpts; j++)
    {
      int m1 = j / nSpts2D;
      int m2 = j % nSpts2D;
      vand(i,j) = Dubiner2D(order, loc_spts(i,0), loc_spts(i,1), m2) * Legendre(m1, loc_spts(i,2)) * std::sqrt(m1+.5);
    }
  }

  inv_vand.assign({nSpts, nSpts}); 
  vand.inverse(inv_vand);

  vandTri.assign({nFptsTri, nFptsTri});

  for (unsigned int i = 0; i < nFptsTri; i++)
    for (unsigned int j = 0; j < nFptsTri; j++)
    {
      vandTri(i,j) = Dubiner2D(order, loc_fpts_2D(i, 0), loc_fpts_2D(i, 1), j);
    }

  inv_vandTri.assign({nFptsTri, nFptsTri});
  vandTri.inverse(inv_vandTri);
}

void Prisms::set_oppRestart(unsigned int order_restart, bool use_shape)
{
  /* Setup restart point locations */
  /// TODO
}

double Prisms::calc_nodal_basis(unsigned int spt, const std::vector<double> &loc)
{
  double val = 0.0;

  for (unsigned int i = 0; i < nSpts; i++)
  {
    unsigned int m1 = i / nSpts2D;
    unsigned int m2 = i % nSpts2D;
    val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
  }

  return val;
}

double Prisms::calc_nodal_basis(unsigned int spt, double *loc)
{
  double val = 0.0;

  for (unsigned int i = 0; i < nSpts; i++)
  {
    unsigned int m1 = i / nSpts2D;
    unsigned int m2 = i % nSpts2D;
    val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
  }

  return val;
}

void Prisms::calc_nodal_basis(double *loc, double* basis)
{
  // store values locally to avoid re-computing
  if (lag_i.size() < nSpts)
    lag_i.resize(nSpts);

  for (unsigned int i = 0; i < nSpts; i++)
  {
    unsigned int m1 = i / nSpts2D;
    unsigned int m2 = i % nSpts2D;
    lag_i[i] = Dubiner2D(order, loc[0], loc[1], m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
  }

  cblas_dgemv(CblasRowMajor,CblasTrans,nSpts,nSpts,1.0,inv_vand.data(),nSpts,
              lag_i.data(),1,0.0,basis,1);
}

double Prisms::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;

  for (unsigned int i = 0; i < nSpts; i++)
  {
    unsigned int m1 = i / nSpts2D;
    unsigned int m2 = i % nSpts2D;
    if (dim < 2)
    {
      val += inv_vand(i, spt) * dDubiner2D(order, loc[0], loc[1], dim, m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
    }
    else
    {
      val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], m2) * Legendre_d1(m1, loc[2]) * std::sqrt(m1+.5);
    }
  }

  return val;
}

double Prisms::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;

  for (unsigned int i = 0; i < nSpts; i++)
  {
    unsigned int m1 = i / nSpts2D;
    unsigned int m2 = i % nSpts2D;
    if (dim < 2)
    {
      val += inv_vand(i, spt) * dDubiner2D(order, loc[0], loc[1], dim, m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
    }
    else
    {
      val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], m2) * Legendre_d1(m1, loc[2]) * std::sqrt(m1+.5);
    }
  }

  return val;
}

double Prisms::calc_d_nodal_basis_fpts(unsigned int fpt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;
  return val;
}

mdvector<double> Prisms::get_face_nodes(unsigned int face, unsigned int P)
{
  if (face < 2)
  {
    return WS_Tri_pts(P);
  }
  else
  {
    mdvector<double> loc_pts({(P+1)*(P+1),nDims});

    auto loc_pts_1D = Gauss_Legendre_pts(P+1);

    unsigned int pt = 0;
    for (unsigned int i = 0; i < P+1; i++)
    {
      for (unsigned int j = 0; j < P+1; j++)
      {
        loc_pts(pt,0) = loc_pts_1D[j];
        loc_pts(pt,1) = loc_pts_1D[i];
        pt++;
      }
    }

    return loc_pts;
  }
}

mdvector<double> Prisms::get_face_weights(unsigned int face, unsigned int P)
{
  if (face < 2)
  {
    return WS_Tri_weights(P);
  }
  else
  {
    mdvector<double> wts_pts({(P+1)*(P+1)});

    auto weights_spts_1D = Gauss_Legendre_weights(P+1);

    unsigned int pt = 0;
    for (unsigned int i = 0; i < P+1; i++)
    {
      for (unsigned int j = 0; j < P+1; j++)
      {
        wts_pts(pt) = weights_spts_1D[i] * weights_spts_1D[j];
        pt++;
      }
    }

    return wts_pts;
  }
}

void Prisms::project_face_point(int face, const double* loc, double* ploc)
{
  switch(face)
  {
    case 0: /* Bottom */
      ploc[0] = loc[0];
      ploc[1] = loc[1];
      ploc[2] = -1.0;
      break;

    case 1: /* Top */
      ploc[0] = loc[0];
      ploc[1] = loc[1];
      ploc[2] = 1.0;
      break;

    case 2: /* Front */
      ploc[0] = loc[0];
      ploc[1] = -1.0;
      ploc[2] = loc[1];
      break;

    case 3: /* Left */
      ploc[0] = -1.0;
      ploc[1] = loc[0];
      ploc[2] = loc[1];
      break;

    case 4: /* Angled Face */
      ploc[0] = -loc[0];
      ploc[1] = loc[0];
      ploc[2] = loc[1];
      break;
  }
}

double Prisms::calc_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc)
{
  if (face < 2)
  {
    double val = 0.0;

    for (unsigned int i = 0; i < nFptsTri; i++)
    {
      val += inv_vandTri(i, pt) * Dubiner2D(order, loc[0], loc[1], i);
    }

    return val;
  }
  else
  {
    int i = pt % nSpts1D;
    int j = pt / nSpts1D;

    return Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);
  }
}

double Prisms::calc_orthonormal_basis(unsigned int mode, const double *loc)
{
  int m2 = mode % nSpts2D;
  int m1 = mode / nSpts2D;
  return Dubiner2D(order, loc[0], loc[1], m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
}

void Prisms::setup_PMG(int pro_order, int res_order)
{

}

void Prisms::setup_ppt_connectivity()
{
  unsigned int nPts1D = order + 1;
  nSubelements = (nPts1D - 1) * (nPts1D - 1) * (nPts1D - 1);
  nNodesPerSubelement = 6;

  /* Allocate memory for local plot point connectivity and solution at plot points */
  ppt_connect.assign({nNodesPerSubelement, nSubelements});

  /* Setup plot "subelement" connectivity [Adapted from HiFiLES] */
  std::vector<unsigned int> nd(nNodesPerSubelement,0);

  int stride = (nPts1D) * (nPts1D+1) / 2;

  unsigned int ele = 0;

  nd[3] = stride; nd[4] = nPts1D + stride;


  for (int k = 0; k < nPts1D-1; k++)
  {
    nd[0] = k*stride;
    nd[1] = nd[0] + nPts1D;

    for (unsigned int i = nPts1D-1; i >= 1; i--)
    {
      for (unsigned int j = 0; j < 2*i - 1; j++)
      {
        ppt_connect(2, ele) = nd[0];
        ppt_connect(1, ele) = nd[1];
        ppt_connect(0, ele) = nd[0]+1;

        ppt_connect(5, ele) = nd[0]+stride;
        ppt_connect(4, ele) = nd[1]+stride;
        ppt_connect(3, ele) = nd[0]+1+stride;

        ++nd[0];

        std::swap(nd[0], nd[1]);
        std::swap(nd[3], nd[4]);

        ele++;
      }
      std::swap(nd[0], nd[1]);
      std::swap(nd[3], nd[4]);
      ++nd[0]; ++nd[1];
      ++nd[3]; ++nd[4];
    }
  }
}

void Prisms::calc_shape(mdvector<double> &shape_val, const double* loc)
{
  int shape_order;

  if (ijk2gmsh.size() != nNodes)
    ijk2gmsh = structured_to_gmsh_pri(nNodes);

  switch (nNodes)
  {
    case 6:
      shape_order = 1; break;

    case 18:
      shape_order = 2; break;

    case 40:
      shape_order = 3; break;
  }

  auto loc_pts_1D = Shape_pts(shape_order);
  int nPts1D = loc_pts_1D.size();
  int nPts2D = nNodes / nPts1D;

  mdvector<double> loc_pts({nNodes,nDims});

  int pt = 0;
  for (int k = 0; k < nPts1D; k++)
  {
    for (int i = 0; i < nPts1D; i++)
    {
      for (int j = 0; j < nPts1D - i; j++)
      {
        loc_pts(pt, 0) = loc_pts_1D[j];
        loc_pts(pt, 1) = loc_pts_1D[i];
        loc_pts(pt, 2) = loc_pts_1D[k];
        pt++;
      }
    }
  }

  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
  {
    for (unsigned int j = 0; j < nNodes; j++)
    {
      int m1 = j / nPts2D;
      int m2 = j % nPts2D;
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), m2) * Legendre(m1, loc_pts(i, 2)) * std::sqrt(m1+.5);
    }
  }

  mdvector<double> inv_vand_s({nNodes, nNodes});
  vand_s.inverse(inv_vand_s);

  /* Compute shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    double val = 0.0;
    for (unsigned int i = 0; i < nNodes; i++)
    {
      int m1 = i / nPts2D;
      int m2 = i % nPts2D;
      val += inv_vand_s(i, nd) * Dubiner2D(shape_order, loc[0], loc[1], m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
    }

    shape_val(ijk2gmsh[nd]) = val;
  }
}

void Prisms::calc_d_shape(mdvector<double> &dshape_val, const double* loc)
{
  int shape_order;

  if (ijk2gmsh.size() != nNodes)
    ijk2gmsh = structured_to_gmsh_pri(nNodes);

  switch (nNodes)
  {
    case 6:
      shape_order = 1; break;

    case 18:
      shape_order = 2; break;

    case 40:
      shape_order = 3; break;
  }

  auto loc_pts_1D = Shape_pts(shape_order);
  int nPts1D = loc_pts_1D.size();
  int nPts2D = nNodes / nPts1D;

  mdvector<double> loc_pts({nNodes,nDims});

  int pt = 0;
  for (int k = 0; k < nPts1D; k++)
  {
    for (int i = 0; i < nPts1D; i++)
    {
      for (int j = 0; j < nPts1D - i; j++)
      {
        loc_pts(pt, 0) = loc_pts_1D[j];
        loc_pts(pt, 1) = loc_pts_1D[i];
        loc_pts(pt, 2) = loc_pts_1D[k];
        pt++;
      }
    }
  }

  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
  {
    for (unsigned int j = 0; j < nNodes; j++)
    {
      int m1 = j / nPts2D;
      int m2 = j % nPts2D;
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), m2) * Legendre(m1, loc_pts(i, 2)) * std::sqrt(m1+.5);
    }
  }

  mdvector<double> inv_vand_s({nNodes, nNodes});
  vand_s.inverse(inv_vand_s);

  /* Compute shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    // xi/eta derivatives [in triangular planes]
    for (unsigned int dim = 0; dim < 2; dim++)
    {
      double val = 0.0;
      for (unsigned int i = 0; i < nNodes; i++)
      {
        int m1 = i / nPts2D;
        int m2 = i % nPts2D;
        val += inv_vand_s(i, nd) * dDubiner2D(shape_order, loc[0], loc[1], dim, m2) * Legendre(m1, loc[2]) * std::sqrt(m1+.5);
      }

      dshape_val(ijk2gmsh[nd], dim) = val;
    }

    // zeta derivative
    double val = 0.0;
    for (unsigned int i = 0; i < nNodes; i++)
    {
      int m1 = i / nPts2D;
      int m2 = i % nPts2D;
      val += inv_vand_s(i, nd) * Dubiner2D(shape_order, loc[0], loc[1], m2) * Legendre_d1(m1, loc[2]) * std::sqrt(m1+.5);
    }

    dshape_val(ijk2gmsh[nd], 2) = val;
  }
}

void Prisms::modify_sensor()
{
  /* Obtain locations of "collapsed" hex solution points */
  unsigned int nNodesHex = 8;
  mdvector<double> nodes({nDims, nNodesHex}); 
  nodes(0, 0) = -1.0; nodes(1, 0) = -1.0; nodes(2, 0) = -1.0;
  nodes(0, 1) =  1.0; nodes(1, 1) = -1.0; nodes(2, 1) = -1.0;
  nodes(0, 2) = -1.0; nodes(1, 2) =  1.0; nodes(2, 2) = -1.0;
  nodes(0, 3) = -1.0; nodes(1, 3) =  1.0; nodes(2, 3) = -1.0;
  nodes(0, 4) = -1.0; nodes(1, 4) = -1.0; nodes(2, 4) =  1.0;
  nodes(0, 5) =  1.0; nodes(1, 5) = -1.0; nodes(2, 5) =  1.0;
  nodes(0, 6) = -1.0; nodes(1, 6) =  1.0; nodes(2, 6) =  1.0;
  nodes(0, 7) = -1.0; nodes(1, 7) =  1.0; nodes(2, 7) =  1.0;

  unsigned int nSpts3D = nSpts1D * nSpts1D * nSpts1D;
  mdvector<double> loc_spts_hex({nSpts3D, nDims}, 0);

  for (unsigned int spt = 0; spt < nSpts3D; spt++)
  {
    for (unsigned int nd = 0; nd < nNodesHex; nd++)
    {
      int i = nd % 2; int j = (nd / 2) % 2; int k = nd / 4;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        loc_spts_hex(spt, dim) += nodes(dim, nd) * Lagrange({-1, 1}, loc_spts_1D[spt % nSpts1D], i) * 
                                                   Lagrange({-1, 1}, loc_spts_1D[(spt / nSpts1D) % nSpts1D], j) *
                                                   Lagrange({-1, 1}, loc_spts_1D[spt / (nSpts1D *nSpts1D)], k);
      }
    }
  }

  /* Setup spt to collapsed spt extrapolation operator (oppEc) */
  std::vector<double> loc(nDims, 0.0);
  mdvector<double> oppEc({nSpts3D, nSpts});
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int spt_q = 0; spt_q < nSpts3D; spt_q++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        loc[dim] = loc_spts_hex(spt_q , dim);

      oppEc(spt_q, spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Multiply oppS by oppEc to get modified operator */
  auto temp = oppS;
  oppS.assign({nSpts3D * nDims, nSpts});

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts3D * nDims, nSpts, nSpts3D,
      1.0, temp.data(), nSpts3D, oppEc.data(), nSpts, 0.0, oppS.data(), nSpts);

} 

double Prisms::rst_max_lim(int dim, double* rst)
{
  switch (dim)
  {
    case 0:
      return std::min(rst[0], 1.0);
    case 1:
      return std::min(rst[1], -rst[0]);
    case 2:
      return std::min(rst[2], 1.0);
  }
}

double Prisms::rst_min_lim(int dim, double* rst)
{
  return std::max(rst[dim], -1.0);
}
