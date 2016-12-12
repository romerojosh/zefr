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

//Quads::Quads(GeoStruct *geo, const InputStruct *input, int order)
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
  
  //geo->nFacesPerEle = 4;
  //geo->nNodesPerFace = 2;
  //geo->nCornerNodes = 4;

  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order + 1) * (input->order + 2) / 2;
    //nSpts1D = input->order + 1;
    nSpts1D = input->order + 2;
    this->order = input->order;
  }
  else
  {
    nSpts = (order + 1) * (order + 2) / 2;
    //nSpts1D = order + 1;
    nSpts1D = order + 2;
    this->order = order;
  }

  nFptsPerFace = nSpts1D;
  nFpts = nSpts1D * nFaces;
  //nPpts = (nSpts1D + 2) * (nSpts1D + 2);
  //nPpts = (3 + 1) * (3 + 2) / 2;
  
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
  //loc_spts.assign({nSpts,nDims}); 
  loc_qpts.assign({nQpts,nDims}); 

  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
   loc_spts_1D = Gauss_Legendre_pts(order+2); 
   //loc_spts_1D = Gauss_Legendre_pts(order+1); 
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  // NOTE: Currently assuming flux point locations always at Legendre.
  // Will need extrapolation operation in 1D otherwise
  auto weights_fpts_1D = Gauss_Legendre_weights(nFptsPerFace); 
  weights_fpts.assign({nFptsPerFace});
  for (unsigned int fpt = 0; fpt < nFptsPerFace; fpt++)
    weights_fpts(fpt) = weights_fpts_1D[fpt];


  /* Setup solution point locations and quadrature weights */
  loc_spts = WS_Tri_pts(order);
  weights_spts = WS_Tri_weights(order);

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
  auto loc_ppts_1D = Shape_pts(3); unsigned int nPpts1D = loc_ppts_1D.size();
  nPpts = (nPpts1D) * (nPpts1D + 1) / 2;
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

  /* Setup gauss quadrature point locations and weights */
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

  /* Can set vandermonde matrices now */
  set_vandermonde_mats();
}

void Tris::set_vandermonde_mats()
{
  /* Set vandermonde for orthonormal Dubiner basis */
  vandDB.assign({nSpts, nSpts});

  for (unsigned int i = 0; i < nSpts; i++)
    for (unsigned int j = 0; j < nSpts; j++)
    {
      vandDB(i,j) = Dubiner2D(order, loc_spts(i, 0), loc_spts(i, 1), j); 
    }

  //for (unsigned int i = 0; i < nSpts; i++)
  //{
  //  for (unsigned int j = 0; j < nSpts; j++)
  //  {
  //    std::cout << vandDB(i,j) << " ";
  //  }
  //  std::cout << std::endl;
  //}

  vandDB.calc_LU();

  inv_vandDB.assign({nSpts, nSpts}); 
  
  mdvector<double> eye({nSpts, nSpts}, 0); 
  for (unsigned int i = 0; i < nSpts; i++)
    eye(i,i) = 1.0;

  vandDB.solve(inv_vandDB, eye);

  //for (unsigned int i = 0; i < nSpts; i++)
  //{
  //  for (unsigned int j = 0; j < nSpts; j++)
  //  {
  //    std::cout << inv_vandDB(i,j) << " ";
  //  }
  //  std::cout << std::endl;
  //}

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

  //for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
  //{
  //  for (unsigned int j = 0; j < 2*nSpts + nFpts; j++)
  //  {
  //    std::cout << vandRT(i,j) << " ";
  //  }
  //  std::cout << std::endl;
  //}

  vandRT.calc_LU();

  inv_vandRT.assign({2*nSpts + nFpts, 2*nSpts * nFpts}); 
  
  //eye.assign({2*nSpts + nFpts, 2*nSpts + nFpts}, 0); 
  mdvector<double>eye2({2*nSpts + nFpts, 2*nSpts + nFpts}, 0); 
  for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
    eye2(i,i) = 1.0;

  vandRT.solve(inv_vandRT, eye2);

  //for (unsigned int i = 0; i < 2*nSpts + nFpts; i++)
  //{
  //  for (unsigned int j = 0; j < 2*nSpts + nFpts; j++)
  //  {
  //    std::cout << inv_vandRT(i,j) << " ";
  //  }
  //  std::cout << std::endl;
  //}

  
}

void Tris::set_oppRestart(unsigned int order_restart, bool use_shape)
{
}

double Tris::calc_nodal_basis(unsigned int spt, const std::vector<double> &loc)
{

  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vandDB(i, spt) * Dubiner2D(order, loc[0], loc[1], i);
  }

  return val;
}

double Tris::calc_nodal_basis(unsigned int spt, double *loc)
{
  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vandDB(i, spt) * Dubiner2D(order, loc[0], loc[1], i);
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
//  unsigned int nSpts_pro_1D = pro_order + 1;
//  unsigned int nSpts_res_1D = res_order + 1;
//  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D;
//  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D;
//
//  std::vector<double> loc(nDims, 0.0);
//
//  if (order != input->order)
//  {
//    /* Setup prolongation operator */
//    oppPro.assign({nSpts_pro, nSpts});
//
//    std::vector<mdvector<double>> opps(pro_order + 1);
//
//    /* Form operator by sequential multiplication of single order prolongation operators */
//    for (int P = pro_order; P > order; P--)
//    {
//      opps[P].assign({(P+1) * (P+1), P * P}, 0);
//
//      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
//      auto loc_spts_P2_1D = Gauss_Legendre_pts(P); 
//
//      for (unsigned int spt = 0; spt < P * P; spt++)
//      {
//        int i = spt % P;
//        int j = spt / P;
//        for (unsigned int pspt = 0; pspt < (P+1) * (P+1); pspt++)
//        {
//          loc[0] = loc_spts_P1_1D[pspt % (P+1)];
//          loc[1] = loc_spts_P1_1D[pspt / (P+1)];
//
//          opps[P](pspt, spt) = Lagrange(loc_spts_P2_1D, i, loc[0]) * 
//                               Lagrange(loc_spts_P2_1D, j, loc[1]);
//        }
//      }
//    }
//
//    for (int P = pro_order; P > order + 1; P--)
//    {
//      mdvector<double> opp({nSpts_pro, (P-1) * (P-1)});
//
//      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, (P-1) * (P-1),
//          P * P, 1.0, opps[P].data(), opps[P].ldim(), opps[P-1].data(), opps[P-1].ldim(), 0.0, opp.data(), opp.ldim());
//
//      opps[P-1] = opp;
//    }
//
//    oppPro = opps[order + 1];
//
//  }
//
//  if (order != 0)
//  {
//    /* Setup restriction operator */
//    oppRes.assign({nSpts_res, nSpts});
//
//    std::vector<mdvector<double>> opps(order + 1);
//
//    /* Form operator by sequential multiplication of single order restriction operators */
//    for (int P = res_order; P < order; P++)
//    {
//      opps[P].assign({(P+1) * (P+1), (P+2) * (P+2)}, 0);
//
//      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
//      auto loc_spts_P2_1D = Gauss_Legendre_pts(P+2); 
//
//      for (unsigned int spt = 0; spt < (P+2) * (P+2); spt++)
//      {
//        int i = spt % (P+2);
//        int j = spt / (P+2);
//
//        for (unsigned int rspt = 0; rspt < (P+1) * (P+1); rspt++)
//        {
//          loc[0] = loc_spts_P1_1D[rspt % (P+1)];
//          loc[1] = loc_spts_P1_1D[rspt / (P+1)];
//
//          opps[P](rspt, spt) = Lagrange(loc_spts_P2_1D, i, loc[0]) * 
//                               Lagrange(loc_spts_P2_1D, j, loc[1]);
//        }
//      }
//    }
//
//    for (int P = res_order; P < order - 1; P++)
//    {
//      mdvector<double> opp({nSpts_res, (P+3) * (P+3)});
//
//      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_res, (P+3) * (P+3),
//          (P+2) * (P+2), 1.0, opps[P].data(), opps[P].ldim(), opps[P+1].data(), opps[P+1].ldim(), 0.0, opp.data(), opp.ldim());
//
//      opps[P+1] = opp;
//    }
//
//    oppRes = opps[order - 1];
//
//  }
//
//#ifdef _GPU
//  /* Copy PMG operators to device */
//  oppPro_d = oppPro;
//  oppRes_d = oppRes;
//#endif
}

void Tris::setup_ppt_connectivity()
{
  unsigned int nPts1D = 4;
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
  mdvector<double> vand({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  vand.calc_LU();

  mdvector<double> inv_vand({nNodes, nNodes});
  
  mdvector<double> eye({nNodes, nNodes}, 0); 
  for (unsigned int i = 0; i < nNodes; i++)
    eye(i,i) = 1.0;

  vand.solve(inv_vand, eye);

  /* Compute Lagrange shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    double val = 0.0;
    for (unsigned int i = 0; i < nNodes; i++)
    {
      val += inv_vand(i, nd) * Dubiner2D(shape_order, loc[0], loc[1], i);
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
  mdvector<double> vand({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  vand.calc_LU();

  mdvector<double> inv_vand({nNodes, nNodes});
  
  mdvector<double> eye({nNodes, nNodes}, 0); 
  for (unsigned int i = 0; i < nNodes; i++)
    eye(i,i) = 1.0;

  vand.solve(inv_vand, eye);

  /* Compute Lagrange shape basis */
  for (unsigned int nd = 0; nd < nNodes; nd++)
  {
    for (unsigned int dim = 0; dim < nDims; dim++)
    {
      double val = 0.0;
      for (unsigned int i = 0; i < nNodes; i++)
      {
        val += inv_vand(i, nd) * dDubiner2D(shape_order, loc[0], loc[1], dim, i);
      }

      dshape_val(gmsh_nodes[shape_order][nd], dim) = val;
    }
  }

  return dshape_val;
}
