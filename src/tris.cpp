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
#include "tris.hpp"
#include "funcs.hpp"

extern "C" {
#include "cblas.h"
}

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

Tris::Tris(GeoStruct *geo, InputStruct *input, int order)
{
  etype = TRI;
  this->geo = geo;
  this->input = input;  
  this->shape_order = geo->shape_orderBT[TRI];  
  this->nEles = geo->nElesBT[TRI];  
  this->nQpts = 45; // Note: Fixing quadrature points to Williams-Shunn 45 point rule

  /* Generic triangular geometry */
  nDims = 2;
  nFaces = 3;
  nNodes = (shape_order+1) * (shape_order+2) / 2;
  
  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order + 1) * (input->order + 2) / 2;
    nSpts1D = input->order + 2;
    this->order = input->order;
  }
  else
  {
    nSpts = (order + 1) * (order + 2) / 2;
    nSpts1D = order + 2;
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

void Tris::set_locs()
{
  /* Allocate memory for point location structures */
  loc_qpts.assign({nQpts,nDims}); 

  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
   loc_spts_1D = Gauss_Legendre_pts(order+2); 
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  // NOTE: Currently assuming flux point locations always at Legendre.
  // Will need extrapolation operation in 1D otherwise
  auto weights_fpts_1D = Gauss_Legendre_weights(nFptsPerFace); 
  weights_fpts.assign({nFptsPerFace});
  for (unsigned int fpt = 0; fpt < nFptsPerFace; fpt++)
    weights_fpts(fpt) = weights_fpts_1D[fpt];


  /* Setup solution point locations and quadrature weights */
  //loc_spts = RW_Tri_pts(order);
  loc_spts = WS_Tri_pts(order);
  weights_spts = WS_Tri_weights(order); //TODO: weights at new points

  /* Setup flux point locations */
  loc_fpts.assign({nFpts,nDims});
  unsigned int fpt = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      switch(i)
      {
        case 0: /* Bottom edge */
          loc_fpts(fpt,0) = loc_spts_1D[j];
          loc_fpts(fpt,1) = -1.0; break;

        case 1: /* Hypotenuse */
          loc_fpts(fpt,0) = loc_spts_1D[nSpts1D-j-1]; 
          loc_fpts(fpt,1) = loc_spts_1D[j]; break;

        case 2: /* Left edge */
          loc_fpts(fpt,0) = -1.0;
          loc_fpts(fpt,1) = loc_spts_1D[nSpts1D-j-1]; break;
      }

      fpt++;
    }
  }
  
  /* Setup plot point locations */
  auto loc_ppts_1D = Shape_pts(order); unsigned int nPpts1D = loc_ppts_1D.size();
  loc_ppts.assign({nPpts,nDims});

  unsigned int ppt = 0;
  for (unsigned int i = 0; i < nPpts1D; i++)
  {
    for (unsigned int j = 0; j < nPpts1D; j++)
    {
      if (j <= nPpts1D - i - 1)
      {
        loc_ppts(ppt,0) = loc_ppts_1D[j];
        loc_ppts(ppt,1) = loc_ppts_1D[i];
        ppt++;
      }
    }
  }

  /* Setup gauss quadrature point locations and weights (fixed to 45 point WS rule) */
  loc_qpts = WS_Tri_pts(8);
  weights_qpts = WS_Tri_weights(8);

}

void Tris::set_normals(std::shared_ptr<Faces> faces)
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

      case 1: /* Hypotenuse */
        tnorm(fpt,0) = std::sqrt(2)/2;
        tnorm(fpt,1) = std::sqrt(2)/2; break;

      case 2: /* Left edge */
        tnorm(fpt,0) = -1.0;
        tnorm(fpt,1) = 0.0; break;

    }
  }
}

void Tris::set_vandermonde_mats()
{
  /* Set vandermonde for orthonormal Dubiner basis */
  vand.assign({nSpts, nSpts});

  for (unsigned int i = 0; i < nSpts; i++)
    for (unsigned int j = 0; j < nSpts; j++)
    {
      vand(i,j) = Dubiner2D(order, loc_spts(i, 0), loc_spts(i, 1), j); 
    }

  vand.calc_LU();

  inv_vand.assign({nSpts, nSpts}); 
  
  mdvector<double> eye({nSpts, nSpts}, 0); 
  for (unsigned int i = 0; i < nSpts; i++)
    eye(i,i) = 1.0;

  vand.solve(inv_vand, eye);


  /* Set vandermonde for Raviart-Thomas monomial basis over combined solution and flux point set*/
  vandRT.assign({2*nSpts+nFpts, 2*nSpts+nFpts}, 0.0);

  for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
  {
    for (unsigned int j = 0; j < 2*nSpts + nFpts; j++)
    {
      double tnormj[2];
      double loc[2];
      if (j < 2*nSpts)
      {
        //tnormj[0] = j % 2; tnormj[1] = (j+1) % 2; // alternates between +xi, and +eta directions
        tnormj[0] = (j < nSpts) ? 1 : 0; 
        tnormj[1] = (j < nSpts) ? 0 : 1; 
        loc[0] = loc_spts(j%nSpts, 0); loc[1] = loc_spts(j%nSpts, 1);
      }
      else
      {
        tnormj[0] = tnorm(j - 2*nSpts, 0); tnormj[1] = tnorm(j - 2*nSpts, 1);
        loc[0] = loc_fpts(j - 2*nSpts, 0); loc[1] = loc_fpts(j - 2*nSpts, 1);
      }

      vandRT(i,j) =  RTMonomial2D(order+1, loc[0], loc[1], 0, i) * tnormj[0];
      vandRT(i,j) += RTMonomial2D(order+1, loc[0], loc[1], 1, i) * tnormj[1];
    }
  }

  vandRT.calc_LU();

  inv_vandRT.assign({2*nSpts + nFpts, 2*nSpts * nFpts}); 
  
  mdvector<double>eye2({2*nSpts + nFpts, 2*nSpts + nFpts}, 0); 
  for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
    eye2(i,i) = 1.0;

  vandRT.solve(inv_vandRT, eye2);

}

void Tris::set_oppRestart(unsigned int order_restart, bool use_shape)
{
  /* Setup restart point locations */
  auto loc_rpts_1D = Shape_pts(order_restart); unsigned int nRpts1D = loc_rpts_1D.size();
  unsigned int nRpts = (order_restart + 1) * (order_restart + 2) / 2;

  mdvector<double> loc_rpts({nRpts,nDims});
  unsigned int rpt = 0;
  for (unsigned int i = 0; i < nRpts1D; i++)
  {
    for (unsigned int j = 0; j < nRpts1D; j++)
    {
      if (j <= nRpts1D - i - 1)
      {
        loc_rpts(rpt,0) = loc_rpts_1D[j];
        loc_rpts(rpt,1) = loc_rpts_1D[i];
        rpt++;
      }
    }
  }

  /* Setup extrapolation operator from restart points */
  oppRestart.assign({nSpts, nRpts});

  /* Set vandermonde and inverse for orthonormal Dubiner basis at restart points */
  mdvector<double> vand_r({nRpts, nRpts});

  for (unsigned int i = 0; i < nRpts; i++)
    for (unsigned int j = 0; j < nRpts; j++)
      vand_r(i,j) = Dubiner2D(order_restart, loc_rpts(i, 0), loc_rpts(i, 1), j); 

  vand_r.calc_LU();

  mdvector<double> inv_vand_r({nRpts, nRpts});
  
  mdvector<double> eye({nRpts, nRpts}, 0); 
  for (unsigned int i = 0; i < nRpts; i++)
    eye(i,i) = 1.0;

  vand_r.solve(inv_vand_r, eye);

  /* Compute Lagrange restart basis */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int rpt = 0; rpt < nRpts; rpt++)
    {
      double val = 0.0;
      for (unsigned int i = 0; i < nRpts; i++)
      {
        val += inv_vand_r(i, rpt) * Dubiner2D(order_restart, loc_spts(spt, 0), loc_spts(spt, 1), i);
      }

      oppRestart(spt, rpt) = val;
    }
  }

}

double Tris::calc_nodal_basis(unsigned int spt, const std::vector<double> &loc)
{

  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], i);
  }

  return val;
}

double Tris::calc_nodal_basis(unsigned int spt, double *loc)
{
  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * Dubiner2D(order, loc[0], loc[1], i);
  }

  return val;
}

double Tris::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{

  double val = 0.0;
  int mode;

  if (dim == 0)
  {
    mode = spt;
  }
  else
  {
    mode = spt + nSpts;
  }

  for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
  {
    val += inv_vandRT(mode, i) * divRTMonomial2D(order + 1, loc[0], loc[1], i);
  }

  return val;

}

double Tris::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{

  double val = 0.0;

  if (dim == 0)
  {
      val = 0.0;
  }
  else
  {
      val = 0.0;
  }

  return val;

}

double Tris::calc_d_nodal_basis_fpts(unsigned int fpt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;
  int mode;

  if (dim == 0)
  {
    mode = fpt + 2*nSpts;

    for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
    {
      val += inv_vandRT(mode, i) * divRTMonomial2D(order + 1, loc[0], loc[1], i);
    }
  }
  else
  {
    val = 0.0;
  }

  return val;

}

void Tris::setup_PMG(int pro_order, int res_order)
{
  unsigned int nSpts_pro = (pro_order + 1) * (pro_order + 2) / 2;
  unsigned int nSpts_res = (res_order + 1) * (res_order + 2) / 2;

  if (order != pro_order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    /* Set vandermonde matrix for pro_order solution points*/
    auto loc_spts_pro = WS_Tri_pts(pro_order);
    mdvector<double> vand_pro({nSpts_pro, nSpts_pro});

    for (unsigned int i = 0; i < nSpts_pro; i++)
      for (unsigned int j = 0; j < nSpts_pro; j++)
        vand_pro(i,j) = Dubiner2D(pro_order, loc_spts_pro(i, 0), loc_spts_pro(i, 1), j); 

    
    /* Set prolongation identity operator (injects modes) */
    mdvector<double> eye({nSpts_pro, nSpts}, 0); 
    for (unsigned int i = 0; i < nSpts; i++)
      eye(i,i) = 1.0;

    /* Form operator by triple matrix product (u_pro = vand_pro * I * inv_vand * u) */
    mdvector<double> temp({nSpts_pro, nSpts}, 0.0);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, nSpts, nSpts,
        1.0, eye.data(), eye.ldim(), inv_vand.data(), inv_vand.ldim(), 0.0, temp.data(), temp.ldim());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, nSpts, nSpts_pro,
        1.0, vand_pro.data(), vand_pro.ldim(), temp.data(), temp.ldim(), 0.0, oppPro.data(), oppPro.ldim());

  }

  if (order != 0)
  {
    /* Setup restriction operator */
    oppRes.assign({nSpts_res, nSpts});

    /* Set vandermonde matrix for res_order solution points*/
    auto  loc_spts_res = WS_Tri_pts(res_order);
    mdvector<double> vand_res({nSpts_res, nSpts_res});

    for (unsigned int i = 0; i < nSpts_res; i++)
      for (unsigned int j = 0; j < nSpts_res; j++)
        vand_res(i,j) = Dubiner2D(res_order, loc_spts_res(i, 0), loc_spts_res(i, 1), j); 

    
    /* Set restriction identity operator (truncates modes) */
    mdvector<double> eye({nSpts_res, nSpts}, 0); 
    for (unsigned int i = 0; i < nSpts_res; i++)
      eye(i,i) = 1.0;

    /* Form operator by triple matrix product (u_res = vand_res * I * inv_vand * u) */
    mdvector<double> temp({nSpts_res, nSpts}, 0.0);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_res, nSpts, nSpts,
        1.0, eye.data(), eye.ldim(), inv_vand.data(), inv_vand.ldim(), 0.0, temp.data(), temp.ldim());
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_res, nSpts, nSpts_res,
        1.0, vand_res.data(), vand_res.ldim(), temp.data(), temp.ldim(), 0.0, oppRes.data(), oppRes.ldim());

  }

#ifdef _GPU
  /* Copy PMG operators to device */
  oppPro_d = oppPro;
  oppRes_d = oppRes;
#endif
}

void Tris::setup_ppt_connectivity()
{
  unsigned int nPts1D = order + 1;
  nSubelements = (nPts1D - 1) * (nPts1D - 1);
  nNodesPerSubelement = 3;

  /* Allocate memory for local plot point connectivity and solution at plot points */
  ppt_connect.assign({3, nSubelements});

  /* Setup plot "subelement" connectivity */
  std::vector<unsigned int> nd(3,0);

  unsigned int ele = 0;
  nd[0] = 0; nd[1] = nPts1D; 

  for (unsigned int i = nPts1D-1; i >= 1; i--)
  {
    for (unsigned int j = 0; j < 2*i - 1; j++)
    {
      ppt_connect(2, ele) = nd[0];
      ppt_connect(1, ele) = nd[1];
      ppt_connect(0, ele) = ++nd[0];

      unsigned int tmp = nd[0];
      nd[0] = nd[1];
      nd[1] = tmp;

      ele++;
    }
    unsigned int tmp = nd[0];
    nd[0] = nd[1];
    nd[1] = tmp;
    ++nd[0];
    ++nd[1];
  }
}

void Tris::transform_dFdU()
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

mdvector<double> Tris::calc_shape(unsigned int shape_order, 
                                   const std::vector<double> &loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2};
  gmsh_nodes[2] =  {0, 3, 1, 5, 4, 2};

  if (shape_order > 2)
    ThrowException("Triangle with supplied shape_order unsupported!");

  mdvector<double> shape_val({nNodes}, 0.0);
  double xi = loc[0]; 
  double eta = loc[1];

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D; j++)
    {
      if (j <= nPts1D - i - 1)
      {
        loc_pts(pt,0) = loc_pts_1D[j];
        loc_pts(pt,1) = loc_pts_1D[i];
        pt++;
      }
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  vand_s.calc_LU();

  mdvector<double> inv_vand_s({nNodes, nNodes});
  
  mdvector<double> eye({nNodes, nNodes}, 0); 
  for (unsigned int i = 0; i < nNodes; i++)
    eye(i,i) = 1.0;

  vand_s.solve(inv_vand_s, eye);

  /* Compute Lagrange shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    double val = 0.0;
    for (unsigned int i = 0; i < nNodes; i++)
    {
      val += inv_vand_s(i, nd) * Dubiner2D(shape_order, loc[0], loc[1], i);
    }

    shape_val(gmsh_nodes[shape_order][nd]) = val;
  }

  return shape_val;

}

mdvector<double> Tris::calc_d_shape(unsigned int shape_order,
                                     const std::vector<double> &loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2};
  gmsh_nodes[2] =  {0, 3, 1, 5, 4, 2};

  if (shape_order > 2)
    ThrowException("Only linear triangles supported right now!");

  mdvector<double> dshape_val({nNodes, 2}, 0.0);
  double xi = loc[0];
  double eta = loc[1];

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D; j++)
    {
      if (j <= nPts1D - i - 1)
      {
        loc_pts(pt,0) = loc_pts_1D[j];
        loc_pts(pt,1) = loc_pts_1D[i];
        pt++;
      }
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  vand_s.calc_LU();

  mdvector<double> inv_vand_s({nNodes, nNodes});
  
  mdvector<double> eye({nNodes, nNodes}, 0); 
  for (unsigned int i = 0; i < nNodes; i++)
    eye(i,i) = 1.0;

  vand_s.solve(inv_vand_s, eye);

  /* Compute Lagrange shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      double val = 0.0;
      for (unsigned int i = 0; i < nNodes; i++)
      {
        val += inv_vand_s(i, nd) * dDubiner2D(shape_order, loc[0], loc[1], dim, i);
      }

      dshape_val(gmsh_nodes[shape_order][nd], dim) = val;
    }
  }

  return dshape_val;
}
