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
#include "quads.hpp"
#include "funcs.hpp"

extern "C" {
#include "cblas.h"
}

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

//Quads::Quads(GeoStruct *geo, const InputStruct *input, int order)
Quads::Quads(GeoStruct *geo, InputStruct *input, int order)
{
  etype = QUAD;
  this->geo = geo;
  this->input = input;  
  this->shape_order = geo->shape_orderBT[QUAD];  
  this->nEles = geo->nElesBT[QUAD];  
  this->nQpts = input->nQpts1D * input->nQpts1D;

  /* Generic quadrilateral geometry */
  nDims = 2;
  nFaces = 4;
  if (!input->serendipity)
    nNodes = (shape_order+1)*(shape_order+1); // Lagrange Elements
  else
    nNodes = 4*(shape_order); // Serendipity Elements
  
  //geo->nFacesPerEle = 4;
  //geo->nNodesPerFace = 2;
  //geo->nCornerNodes = 4;

  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order + 1) * (input->order + 1);
    nSpts1D = input->order + 1;
    this->order = input->order;
  }
  else
  {
    nSpts = (order + 1) * (order + 1);
    nSpts1D = order + 1;
    this->order = order;
  }

  nFptsPerFace = nSpts1D;
  nFpts = nSpts1D * nFaces;
  nPpts = nSpts;
  
  if (input->equation == AdvDiff || input->equation == Burgers)
  {
    nVars = 1;
  }
  else if (input->equation == EulerNS)
  {
    nVars = 4;
  }
  else
  {
    ThrowException("Equation not recognized: " + input->equation);
  }
  
}

void Quads::set_locs()
{
  /* Allocate memory for point location structures */
  loc_spts.assign({nSpts,nDims}); idx_spts.assign({nSpts,nDims});
  loc_fpts.assign({nFpts,nDims}); idx_fpts.assign({nFpts,nDims});
  loc_ppts.assign({nPpts,nDims}); idx_ppts.assign({nPpts,nDims});
  loc_qpts.assign({nQpts,nDims}); idx_qpts.assign({nQpts,nDims});

  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
   loc_spts_1D = Gauss_Legendre_pts(order+1); 
  else if (input->spt_type == "DFRsp")
    loc_spts_1D = DFRsp_pts(order+1, 0.339842589774454);
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  // NOTE: Currently assuming solution point locations always at Legendre.
  // Will need extrapolation operation in 1D otherwise
  auto weights_spts_1D = Gauss_Legendre_weights(nSpts1D); 
  weights_fpts.assign({nSpts1D});
  for (unsigned int fpt = 0; fpt < nSpts1D; fpt++)
    weights_fpts(fpt) = weights_spts_1D[fpt];


  loc_DFR_1D = loc_spts_1D;
  loc_DFR_1D.insert(loc_DFR_1D.begin(), -1.0);
  loc_DFR_1D.insert(loc_DFR_1D.end(), 1.0);

  /* Setup solution point locations and quadrature weights */
  weights_spts.assign({nSpts});
  unsigned int spt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      loc_spts(spt,0) = loc_spts_1D[j];
      loc_spts(spt,1) = loc_spts_1D[i];
      idx_spts(spt,0) = j;
      idx_spts(spt,1) = i;
      weights_spts(spt) = weights_spts_1D[i] * weights_spts_1D[j];
      spt++;
    }
  }

  /* Setup flux point locations */
  unsigned int fpt = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      switch(i)
      {
        case 0: /* Bottom edge */
          loc_fpts(fpt,0) = loc_spts_1D[j];
          loc_fpts(fpt,1) = -1.0; 
          idx_fpts(fpt,0) = j;
          idx_fpts(fpt,1) = -1; break;

        case 1: /* Right edge */
          loc_fpts(fpt,0) = 1.0; 
          loc_fpts(fpt,1) = loc_spts_1D[j];
          idx_fpts(fpt,0) = nSpts1D; 
          idx_fpts(fpt,1) = j; break;

        case 2: /* Upper edge */
          loc_fpts(fpt,0) = loc_spts_1D[nSpts1D-j-1];
          loc_fpts(fpt,1) = 1.0;
          idx_fpts(fpt,0) = nSpts1D-j-1;
          idx_fpts(fpt,1) = nSpts1D; break;

        case 3: /* Left edge */
          loc_fpts(fpt,0) = -1.0; 
          loc_fpts(fpt,1) = loc_spts_1D[nSpts1D-j-1];
          idx_fpts(fpt,0) = -1; 
          idx_fpts(fpt,1) = nSpts1D-j-1; break;
      }

      fpt++;
    }
  }
  
  /* Setup plot point locations (equispaced) */
  auto loc_ppts_1D = Shape_pts(order);

  unsigned int ppt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      loc_ppts(ppt,0) = loc_ppts_1D[j];
      loc_ppts(ppt,1) = loc_ppts_1D[i];
      idx_ppts(ppt,0) = j;
      idx_ppts(ppt,1) = i;
      ppt++;
    }
  }

  /* Setup gauss quadrature point locations and weights */
  loc_qpts_1D = Gauss_Legendre_pts(input->nQpts1D); 
  auto weights_qpts_1D = Gauss_Legendre_weights(input->nQpts1D);
  weights_qpts.assign({input->nQpts1D * input->nQpts1D});

  /* Setup quadrature point locations */
  unsigned int qpt = 0;
  for (unsigned int i = 0; i < input->nQpts1D; i++)
  {
    for (unsigned int j = 0; j < input->nQpts1D; j++)
    {
      loc_qpts(qpt,0) = loc_qpts_1D[j];
      loc_qpts(qpt,1) = loc_qpts_1D[i];
      idx_qpts(qpt,0) = j;
      idx_qpts(qpt,1) = i;
      weights_qpts(qpt) =  weights_qpts_1D[i] * weights_qpts_1D[j];
      qpt++;
    }
  }

}

void Quads::set_normals(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for normals */
  tnorm.assign({nFpts,nDims});

  /* Setup parent-space (transformed) normals at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    switch(fpt/nSpts1D)
    {
      case 0: /* Bottom edge */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = -1.0; break;

      case 1: /* Right edge */
        tnorm(fpt,0) = 1.0;
        tnorm(fpt,1) = 0.0; break;

      case 2: /* Top edge */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = 1.0; break;

      case 3: /* Left edge */
        tnorm(fpt,0) = -1.0;
        tnorm(fpt,1) = 0.0; break;
    }

  }
}

void Quads::set_oppRestart(unsigned int order_restart, bool use_shape)
{
  unsigned int nRpts1D = (order_restart + 1);
  unsigned int nRpts = nRpts1D * nRpts1D;

  /* Setup extrapolation operator from restart points */
  oppRestart.assign({nSpts, nRpts});

  std::vector<double> loc_rpts_1D;
  if (!use_shape)
    loc_rpts_1D = Gauss_Legendre_pts(order_restart + 1); 
  else
    loc_rpts_1D = Shape_pts(order_restart); 

  std::vector<double> loc(input->nDims);
  for (unsigned int rpt = 0; rpt < nRpts; rpt++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int dim = 0; dim < input->nDims; dim++)
        loc[dim] = loc_spts(spt , dim);

      int i = rpt % nRpts1D;
      int j = rpt / nRpts1D;
      oppRestart(spt,rpt) = Lagrange(loc_rpts_1D, i, loc[0]) * 
                            Lagrange(loc_rpts_1D, j, loc[1]);
    }
  }
}

double Quads::calc_nodal_basis(unsigned int spt, const std::vector<double> &loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);

  return val;
}

double Quads::calc_nodal_basis(unsigned int spt, double *loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);

  return val;
}

double Quads::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
   * boundary points for DFR) */
  unsigned int i = idx_spts(spt,0) + 1;
  unsigned int j = idx_spts(spt,1) + 1;

  double val = 0.0;

  if (dim == 0)
  {
      val = Lagrange_d1(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]);
  }
  else
  {
      val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange_d1(loc_DFR_1D, j, loc[1]);
  }

  return val;

}

double Quads::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);

  double val = 0.0;

  if (dim == 0)
  {
      val = Lagrange_d1(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);
  }
  else
  {
      val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange_d1(loc_spts_1D, j, loc[1]);
  }

  return val;

}

double Quads::calc_d_nodal_basis_fpts(unsigned int fpt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
   * boundary points for DFR) */
  unsigned int i = idx_fpts(fpt,0) + 1;
  unsigned int j = idx_fpts(fpt,1) + 1;

  double val = 0.0;

  if (dim == 0)
  {
      val = Lagrange_d1(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]);
  }
  else
  {
      val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange_d1(loc_DFR_1D, j, loc[1]);
  }

  return val;

}

void Quads::setup_PMG(int pro_order, int res_order)
{
  unsigned int nSpts_pro_1D = pro_order + 1;
  unsigned int nSpts_res_1D = res_order + 1;
  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D;
  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D;

  std::vector<double> loc(nDims, 0.0);

  if (order != pro_order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    std::vector<mdvector<double>> opps(pro_order + 1);

    /* Form operator by sequential multiplication of single order prolongation operators */
    for (int P = pro_order; P > order; P--)
    {
      opps[P].assign({(P+1) * (P+1), P * P}, 0);

      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
      auto loc_spts_P2_1D = Gauss_Legendre_pts(P); 

      for (unsigned int spt = 0; spt < P * P; spt++)
      {
        int i = spt % P;
        int j = spt / P;
        for (unsigned int pspt = 0; pspt < (P+1) * (P+1); pspt++)
        {
          loc[0] = loc_spts_P1_1D[pspt % (P+1)];
          loc[1] = loc_spts_P1_1D[pspt / (P+1)];

          opps[P](pspt, spt) = Lagrange(loc_spts_P2_1D, i, loc[0]) * 
                               Lagrange(loc_spts_P2_1D, j, loc[1]);
        }
      }
    }

    for (int P = pro_order; P > order + 1; P--)
    {
      mdvector<double> opp({nSpts_pro, (P-1) * (P-1)});

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, (P-1) * (P-1),
          P * P, 1.0, opps[P].data(), opps[P].ldim(), opps[P-1].data(), opps[P-1].ldim(), 0.0, opp.data(), opp.ldim());

      opps[P-1] = opp;
    }

    oppPro = opps[order + 1];
  }

  if (order != 0)
  {
    /* Setup restriction operator */
    oppRes.assign({nSpts_res, nSpts});

    std::vector<mdvector<double>> opps(order + 1);

    /* Form operator by sequential multiplication of single order restriction operators */
    for (int P = res_order; P < order; P++)
    {
      opps[P].assign({(P+1) * (P+1), (P+2) * (P+2)}, 0);

      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
      auto loc_spts_P2_1D = Gauss_Legendre_pts(P+2); 

      for (unsigned int spt = 0; spt < (P+2) * (P+2); spt++)
      {
        int i = spt % (P+2);
        int j = spt / (P+2);

        for (unsigned int rspt = 0; rspt < (P+1) * (P+1); rspt++)
        {
          loc[0] = loc_spts_P1_1D[rspt % (P+1)];
          loc[1] = loc_spts_P1_1D[rspt / (P+1)];

          opps[P](rspt, spt) = Lagrange(loc_spts_P2_1D, i, loc[0]) * 
                               Lagrange(loc_spts_P2_1D, j, loc[1]);
        }
      }
    }

    for (int P = res_order; P < order - 1; P++)
    {
      mdvector<double> opp({nSpts_res, (P+3) * (P+3)});

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_res, (P+3) * (P+3),
          (P+2) * (P+2), 1.0, opps[P].data(), opps[P].ldim(), opps[P+1].data(), opps[P+1].ldim(), 0.0, opp.data(), opp.ldim());

      opps[P+1] = opp;
    }

    oppRes = opps[order - 1];

  }

#ifdef _GPU
  /* Copy PMG operators to device */
  oppPro_d = oppPro;
  oppRes_d = oppRes;
#endif
}

void Quads::setup_ppt_connectivity()
{
  unsigned int nSubelements1D = nSpts1D - 1;
  nSubelements = nSubelements1D * nSubelements1D;
  nNodesPerSubelement = 4;

  /* Allocate memory for local plot point connectivity and solution at plot points */
  ppt_connect.assign({4, nSubelements});

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
        ppt_connect(node, ele) = nd[node] + j;
      }

      ele++;
    }

    for (unsigned int node = 0; node < 4; node ++)
      nd[node] += nSubelements1D + 1;
  }
}

void Quads::transform_dFdU()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        if (input->overset && geo->iblank_cell(ele) != NORMAL) continue;

        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          double dFdUtemp = dFdU_spts(spt, ele, ni, nj, 0);

          dFdU_spts(spt, ele, ni, nj, 0) = dFdU_spts(spt, ele, ni, nj, 0) * jaco_spts(spt, ele, 1, 1) -
                                           dFdU_spts(spt, ele, ni, nj, 1) * jaco_spts(spt, ele, 0, 1);
          dFdU_spts(spt, ele, ni, nj, 1) = dFdU_spts(spt, ele, ni, nj, 1) * jaco_spts(spt, ele, 0, 0) -
                                           dFdUtemp * jaco_spts(spt, ele, 1, 0);
        }
      }
    }
  }
#endif

#ifdef _GPU
  transform_dFdU_quad_wrapper(dFdU_spts_d, jaco_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);
  check_error();
#endif
}

mdvector<double> Quads::calc_shape(unsigned int shape_order, 
                                   const std::vector<double> &loc)
{
  mdvector<double> shape_val({nNodes}, 0.0);
  double xi = loc[0]; 
  double eta = loc[1];

  /* 8-node Serendipity Element */
  if (shape_order == 2 and input->serendipity)
  {
    shape_val(0) = -0.25*(1.-xi)*(1.-eta)*(1.+eta+xi);
    shape_val(1) = -0.25*(1.+xi)*(1.-eta)*(1.+eta-xi);
    shape_val(2) = -0.25*(1.+xi)*(1.+eta)*(1.-eta-xi);
    shape_val(3) = -0.25*(1.-xi)*(1.+eta)*(1.-eta+xi);
    shape_val(4) = 0.5*(1.-xi)*(1.+xi)*(1.-eta);
    shape_val(5) = 0.5*(1.+xi)*(1.+eta)*(1.-eta);
    shape_val(6) = 0.5*(1.-xi)*(1.+xi)*(1.+eta);
    shape_val(7) = 0.5*(1.-xi)*(1.+eta)*(1.-eta);
  }

  /* Lagrange Elements */
  else
  {
    int nSide = sqrt(nNodes);

    if (nSide*nSide != nNodes)
    {
      std::cout << "nNodes = " << nNodes << std::endl;
      ThrowException("For Lagrange quad of order N, must have (N+1)^2 shape points.");
    }

    std::vector<double> xlist(nSide);
    double dxi = 2./(nSide-1);

    for (int i=0; i<nSide; i++)
      xlist[i] = -1. + i*dxi;

    auto ijk2gmsh = structured_to_gmsh_quad(nNodes);

    int pt = 0;
    for (int j = 0; j < nSide; j++)
    {
      for (int i = 0; i < nSide; i++)
      {
        shape_val(ijk2gmsh[pt]) = Lagrange(xlist, xi, i) * Lagrange(xlist, eta, j);
        pt++;
      }
    }
  }

  return shape_val;
}

mdvector<double> Quads::calc_d_shape(unsigned int shape_order,
                                     const std::vector<double> &loc)
{
  mdvector<double> dshape_val({nNodes, nDims}, 0.0);
  double xi = loc[0];
  double eta = loc[1];

  /* 8-node Serendipity Element */
  if (shape_order == 2 and input->serendipity)
  {
      dshape_val(0, 0) = -0.25*(-1.+eta)*(2.*xi+eta);
      dshape_val(1, 0) = 0.25*(-1.+eta)*(eta - 2.*xi);
      dshape_val(2, 0) = 0.25*(1.+eta)*(2.*xi+eta);
      dshape_val(3, 0) = -0.25*(1.+eta)*(eta-2.*xi);
      dshape_val(4, 0) = xi*(-1.+eta);
      dshape_val(5, 0) = -0.5*(1+eta)*(-1.+eta);
      dshape_val(6, 0) = -xi*(1.+eta);
      dshape_val(7, 0) = 0.5*(1+eta)*(-1.+eta);

      dshape_val(0, 1) = -0.25*(-1.+xi)*(2.*eta+xi);
      dshape_val(1, 1) = 0.25*(1.+xi)*(2.*eta - xi);
      dshape_val(2, 1) = 0.25*(1.+xi)*(2.*eta+xi);
      dshape_val(3, 1) = -0.25*(-1.+xi)*(2.*eta-xi);
      dshape_val(4, 1) = 0.5*(1.+xi)*(-1.+xi);
      dshape_val(5, 1) = -eta*(1.+xi);
      dshape_val(6, 1) = -0.5*(1.+xi)*(-1.+xi);
      dshape_val(7, 1) = eta*(-1.+xi);
  }

  /* Lagrange Elements */
  else
  {
    int nSide = sqrt(nNodes);

    if (nSide*nSide != nNodes)
    {
      std::cout << "nNodes = " << nNodes << std::endl;
      ThrowException("For Lagrange quad of order N, must have (N+1)^2 shape points.");
    }

    std::vector<double> xlist(nSide);
    double dxi = 2./(nSide-1);

    for (int i=0; i<nSide; i++)
      xlist[i] = -1. + i*dxi;

    auto ijk2gmsh = structured_to_gmsh_quad(nNodes);

    int pt = 0;
    for (int j = 0; j < nSide; j++)
    {
      for (int i = 0; i < nSide; i++)
      {
        dshape_val(ijk2gmsh[pt], 0) = dLagrange(xlist, xi, i) *  Lagrange(xlist, eta, j);
        dshape_val(ijk2gmsh[pt], 1) =  Lagrange(xlist, xi, i) * dLagrange(xlist, eta, j);
        pt++;
      }
    }
  }

  return dshape_val;
}
