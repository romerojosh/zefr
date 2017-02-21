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
#include "tets.hpp"
#include "funcs.hpp"

extern "C" {
#include "cblas.h"
}

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

Tets::Tets(GeoStruct *geo, InputStruct *input, int order)
{
  etype = TET;
  this->geo = geo;
  this->input = input;  
  this->nEles = geo->nElesBT[TET];  
  this->nQpts = 84; // Note: Fixing quadrature points to Shunn-Hamm 84 point rule

  /* Generic tetrahedral geometry */
  nDims = 3;
  nFaces = 4;
  nNodes = geo->nNodesPerEleBT[TET];
  
  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order + 1) * (input->order + 2) * (input->order + 3) / 6;
    nFptsPerFace = (input->order + 2) * (input->order + 3) / 2;
    nSpts1D = input->order + 1; //nSpts1D only used for filter matrix setup
    this->order = input->order;
  }
  else
  {
    nSpts = (order + 1) * (order + 2) * (order + 3) / 6;
    nFptsPerFace = (order + 2) * (order + 3) / 2;
    nSpts1D = order + 1; 
    this->order = order;
  }

  nFpts = nFptsPerFace * nFaces;
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

void Tets::set_locs()
{
  /* Allocate memory for point location structures */
  loc_qpts.assign({nQpts,nDims}); 

  mdvector<double> loc_fpts_2D; 
  /* Get positions of points in 1D and 2D */
  if (input->spt_type == "Legendre")
  {
   loc_spts_1D = Gauss_Legendre_pts(order + 1); // loc_spts_1D used when generating filter matrices only 
   loc_fpts_2D = WS_Tri_pts(order + 1);
  }
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  auto weights_fpts_2D = WS_Tri_weights(order + 1);
  weights_fpts.assign({nFptsPerFace});
  for (unsigned int fpt = 0; fpt < nFptsPerFace; fpt++)
    weights_fpts(fpt) = weights_fpts_2D(fpt);


  /* Setup solution point locations and quadrature weights */
  loc_spts = WS_Tet_pts(order);
  weights_spts = WS_Tet_weights(order); 

  /* Setup flux point locations */
  loc_fpts.assign({nFpts,nDims});
  unsigned int fpt = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < nFptsPerFace; j++)
    {
      switch(i)
      {
        case 0: /* Rear (xi-eta plane) */
          loc_fpts(fpt,0) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,1) = loc_fpts_2D(j, 1);
          loc_fpts(fpt,2) = -1.0;
          break;

        case 1: /* Angled Face */
          loc_fpts(fpt,0) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,1) = loc_fpts_2D(j, 1);
          loc_fpts(fpt,2) = -(loc_fpts_2D(j,0) + loc_fpts_2D(j,1) + 1);
          break;

        case 2: /* Left (eta-zeta plane) */
          loc_fpts(fpt,0) = -1.0;
          loc_fpts(fpt,1) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,2) = loc_fpts_2D(j, 1);
          break;

        case 3: /* Bottom (xi-zeta plane) */
          loc_fpts(fpt,0) = loc_fpts_2D(j, 0);
          loc_fpts(fpt,1) = -1.0;
          loc_fpts(fpt,2) = loc_fpts_2D(j, 1);
          break;
      }

      fpt++;
    }
  }
  
  /* Setup plot point locations */
  auto loc_ppts_1D = Shape_pts(order); unsigned int nPpts1D = loc_ppts_1D.size();
  //auto loc_ppts_1D = Shape_pts(1); unsigned int nPpts1D = loc_ppts_1D.size();
  loc_ppts.assign({nPpts,nDims});

  unsigned int ppt = 0;
//  for (unsigned int i = 0; i < nPpts1D; i++)
//  {
//    for (unsigned int j = 0; j < nPpts1D - i; j++)
//    {
//      for (unsigned int k = 0; k < nPpts1D - i - j; k++)
//      {
//        loc_ppts(ppt,0) = loc_ppts_1D[k];
//        loc_ppts(ppt,1) = loc_ppts_1D[j];
//        loc_ppts(ppt,2) = loc_ppts_1D[i];
//        ppt++;
//      }
//    }
//  }

/* Modified from HiFiLES */
for(int k = 0; k < order+1; k++)
{
  for(int j = 0; j < order+1-k; j++)
  {
    for(int i = 0; i < order+1-k-j; i++)
    {
      ppt = (order+1)*(order+2)*(order+3)/6 - (order+1-k)*(order+2-k)*(order+3-k)/6 +
        j*(order+1-k)-(j-1)*j/2 + i;

      loc_ppts(ppt, 0) = -1.0+(2.0*i/(order));
      loc_ppts(ppt, 1) = -1.0+(2.0*j/(order));
      loc_ppts(ppt, 2) = -1.0+(2.0*k/(order));
    }
  }
}

  /* Setup gauss quadrature point locations and weights (fixed to 84 point Shunn-Hamm rule) */
  loc_qpts = WS_Tet_pts(6);
  weights_qpts = WS_Tet_weights(6);

}

void Tets::set_normals(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for normals */
  tnorm.assign({nFpts,nDims});

  /* Setup parent-space (transformed) normals at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    switch(fpt/nFptsPerFace)
    {
      case 0: /* Rear */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = 0.0;
        tnorm(fpt,2) = -1.0;
        break;

      case 1: /* Angled Face */
        tnorm(fpt,0) = 1.0/std::sqrt(3);
        tnorm(fpt,1) = 1.0/std::sqrt(3);
        tnorm(fpt,2) = 1.0/std::sqrt(3);
        break;

      case 2: /* Left */
        tnorm(fpt,0) = -1.0;
        tnorm(fpt,1) = 0.0; 
        tnorm(fpt,2) = 0.0; 
        break;

      case 3: /* Bottom */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = -1.0; 
        tnorm(fpt,2) = 0.0; 
        break;

    }
  }
}

void Tets::set_vandermonde_mats()
{
  /* Set vandermonde for orthonormal Dubiner basis */
  vand.assign({nSpts, nSpts});

  for (unsigned int i = 0; i < nSpts; i++)
    for (unsigned int j = 0; j < nSpts; j++)
    {
      vand(i,j) = Dubiner3D(order, loc_spts(i, 0), loc_spts(i, 1), loc_spts(i, 2), j); 
    }

  vand.calc_LU();

  inv_vand.assign({nSpts, nSpts}); 
  
  mdvector<double> eye({nSpts, nSpts}, 0); 
  for (unsigned int i = 0; i < nSpts; i++)
    eye(i,i) = 1.0;

  vand.solve(inv_vand, eye);


  /* Set vandermonde for Raviart-Thomas monomial basis over combined solution and flux point set*/
  vandRT.assign({3*nSpts+nFpts, 3*nSpts+nFpts}, 0.0);

  for (unsigned int i = 0; i < 3*nSpts + nFpts; i++)
  {
    for (unsigned int j = 0; j < 3*nSpts + nFpts; j++)
    {
      double tnormj[3];
      double loc[3];
      if (j < 3*nSpts)
      {
        if (j < nSpts)
        {
          tnormj[0] = 1; tnormj[1] = 0; tnormj[2] = 0;
        }
        else if (j < 2*nSpts)
        {
          tnormj[0] = 0; tnormj[1] = 1; tnormj[2] = 0;
        }
        else if (j < 3*nSpts)
        {
          tnormj[0] = 0; tnormj[1] = 0; tnormj[2] = 1;
        }

        loc[0] = loc_spts(j%nSpts, 0); 
        loc[1] = loc_spts(j%nSpts, 1);
        loc[2] = loc_spts(j%nSpts, 2);
      }
      else
      {
        tnormj[0] = tnorm(j - 3*nSpts, 0); 
        tnormj[1] = tnorm(j - 3*nSpts, 1);
        tnormj[2] = tnorm(j - 3*nSpts, 2);

        loc[0] = loc_fpts(j - 3*nSpts, 0); 
        loc[1] = loc_fpts(j - 3*nSpts, 1);
        loc[2] = loc_fpts(j - 3*nSpts, 2);
      }

      vandRT(i,j) =  RTMonomial3D(order+1, loc[0], loc[1], loc[2], 0, i) * tnormj[0];
      vandRT(i,j) += RTMonomial3D(order+1, loc[0], loc[1], loc[2], 1, i) * tnormj[1];
      vandRT(i,j) += RTMonomial3D(order+1, loc[0], loc[1], loc[2], 2, i) * tnormj[2];
    }
  }

  vandRT.calc_LU();

  inv_vandRT.assign({3*nSpts + nFpts, 3*nSpts * nFpts}); 
  
  mdvector<double>eye2({3*nSpts + nFpts, 3*nSpts + nFpts}, 0); 
  for (unsigned int i = 0; i < 3*nSpts + nFpts; i++)
    eye2(i,i) = 1.0;

  vandRT.solve(inv_vandRT, eye2);

}

void Tets::set_oppRestart(unsigned int order_restart, bool use_shape)
{
  /* Setup restart point locations */
  auto loc_rpts_1D = Shape_pts(order_restart); unsigned int nRpts1D = loc_rpts_1D.size();
  unsigned int nRpts = (order_restart + 1) * (order_restart + 2) * (order_restart + 3) / 6;

  mdvector<double> loc_rpts({nRpts,nDims});
  unsigned int rpt = 0;
  for (unsigned int i = 0; i < nRpts1D; i++)
  {
    for (unsigned int j = 0; j < nRpts1D - i; j++)
    {
      for (unsigned int k = 0; k < nRpts1D - i - j; k++)
      {
        loc_rpts(rpt,0) = loc_rpts_1D[k];
        loc_rpts(rpt,1) = loc_rpts_1D[j];
        loc_rpts(rpt,2) = loc_rpts_1D[i];
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
      vand_r(i,j) = Dubiner3D(order_restart, loc_rpts(i, 0), loc_rpts(i, 1), loc_rpts(i, 2), j); 

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
        val += inv_vand_r(i, rpt) * Dubiner3D(order_restart, loc_spts(spt, 0), loc_spts(spt, 1), loc_spts(spt, 2), i);
      }

      oppRestart(spt, rpt) = val;
    }
  }

}

double Tets::calc_nodal_basis(unsigned int spt, const std::vector<double> &loc)
{

  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * Dubiner3D(order, loc[0], loc[1], loc[2], i);
  }

  return val;
}

double Tets::calc_nodal_basis(unsigned int spt, double *loc)
{
  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * Dubiner3D(order, loc[0], loc[1], loc[2], i);
  }

  return val;
}

double Tets::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{

  double val = 0.0;
  int mode;

  mode = spt + dim * nSpts;

  for (unsigned int i = 0; i < 3*nSpts + nFpts; i++)
  {
    val += inv_vandRT(mode, i) * divRTMonomial3D(order + 1, loc[0], loc[1], loc[2], i);
  }

  return val;

}

double Tets::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{

  double val = 0.0;

  return val;

}

double Tets::calc_d_nodal_basis_fpts(unsigned int fpt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;
  int mode;

  if (dim == 0)
  {
    mode = fpt + 3*nSpts;

    for (unsigned int i = 0; i < 3*nSpts + nFpts; i++)
    {
      val += inv_vandRT(mode, i) * divRTMonomial3D(order + 1, loc[0], loc[1], loc[2], i);
    }
  }
  else
  {
    val = 0.0;
  }

  return val;

}

void Tets::setup_PMG(int pro_order, int res_order)
{
  unsigned int nSpts_pro = (pro_order + 1) * (pro_order + 2) * (pro_order + 3) / 6;
  unsigned int nSpts_res = (res_order + 1) * (res_order + 2) * (res_order + 3) / 6;

  if (order != pro_order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    /* Set vandermonde matrix for pro_order solution points*/
    auto loc_spts_pro = WS_Tet_pts(pro_order);
    mdvector<double> vand_pro({nSpts_pro, nSpts_pro});

    for (unsigned int i = 0; i < nSpts_pro; i++)
      for (unsigned int j = 0; j < nSpts_pro; j++)
        vand_pro(i,j) = Dubiner3D(pro_order, loc_spts_pro(i, 0), loc_spts_pro(i, 1), loc_spts_pro(i, 2), j); 

    
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
    auto  loc_spts_res = WS_Tet_pts(res_order);
    mdvector<double> vand_res({nSpts_res, nSpts_res});

    for (unsigned int i = 0; i < nSpts_res; i++)
      for (unsigned int j = 0; j < nSpts_res; j++)
        vand_res(i,j) = Dubiner3D(res_order, loc_spts_res(i, 0), loc_spts_res(i, 1), loc_spts_res(i, 2), j); 

    
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

/* Modified from HiFiLES */
void Tets::setup_ppt_connectivity()
{
  int P = (int) order;
  nSubelements = (P) * (P+1) * (P+2)/6 + 4 * (P-1) * (P) * (P+1)/6 +(P-2) * (P-1) * (P)/6;
  nNodesPerSubelement = 4;

  /* Allocate memory for local plot point connectivity and solution at plot points */
  ppt_connect.assign({4, nSubelements});

  std::vector<unsigned int> nd(6);
  int temp = (P+1) * (P+2) * (P+3)/6;

  int ele = 0;
  /*! First set */
  for(int k=0; k < P; k++)
  {
    for(int j=0; j < P-k; j++)
    {
      for(int i=0; i < P-k-j; i++)
      {

      nd[0] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + j * (P+1-k) - (j-1) * j/2 + i;
      nd[1] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + j * (P+1-k) - (j-1) * j/2 + i + 1;
      nd[2] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + (j+1) * (P+1-k) - (j) * (j+1)/2 + i;
      nd[3] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + j * (P+1-(k+1)) - (j-1) * j/2 + i;

      ppt_connect(0, ele) = nd[0];
      ppt_connect(1, ele) = nd[1];
      ppt_connect(2, ele) = nd[2];
      ppt_connect(3, ele) = nd[3];
      ele++;
      }
    }
  }

  /*! Second set */
  for(int k=0; k < P-1; k++)
  {
    for(int j=0; j < P-1-k; j++)
    {
      for(int i=0; i < P-1-k-j; i++)
      {
        nd[0] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + j * (P+1-k) - (j-1) * j/2 + i + 1;
        nd[1] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + (j+1) * (P+1-k) - (j) * (j+1)/2 + i + 1;
        nd[2] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + j * (P+1-(k+1)) - (j-1) * j/2 + i + 1;
        nd[3] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + (j+1) * (P+1-(k+1)) - (j) * (j+1)/2 + (i-1) + 1;
        nd[4] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + (j) * (P+1-(k+1)) - (j-1) * (j)/2 + (i-1) + 1;
        nd[5] = temp - (P+1-(k)) * (P+2-(k)) * (P+3-(k))/6 + (j+1) * (P+1-(k)) - (j) * (j+1)/2 + (i-1) + 1;


        ppt_connect(0, ele) = nd[0];
        ppt_connect(1, ele) = nd[2];
        ppt_connect(2, ele) = nd[1];
        ppt_connect(3, ele) = nd[4];
        ele++;

        ppt_connect(0, ele) = nd[2];
        ppt_connect(1, ele) = nd[3];
        ppt_connect(2, ele) = nd[1];
        ppt_connect(3, ele) = nd[4];
        ele++;

        ppt_connect(0, ele) = nd[5];
        ppt_connect(1, ele) = nd[1];
        ppt_connect(2, ele) = nd[3];
        ppt_connect(3, ele) = nd[4];
        ele++;

        ppt_connect(0, ele) = nd[0];
        ppt_connect(1, ele) = nd[4];
        ppt_connect(2, ele) = nd[1];
        ppt_connect(3, ele) = nd[5];
        ele++;
      }
    }
  }

  /*! Third set */
  for(int k=0; k < P-2; k++)
  {
    for(int j=0; j < P-2-k; j++)
    {
      for(int i=0; i < P-2-k-j; i++)
      {
        nd[0] = temp - (P+1-k) * (P+2-k) * (P+3-k)/6 + (j+1) * (P+1-k) - (j) * (j+1)/2 + i + 1;
        nd[1] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + (j) * (P+1-(k+1)) - (j-1) * (j)/2 + i + 1;
        nd[2] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + (j+1) * (P+1-(k+1)) - (j) * (j+1)/2 + i ;
        nd[3] = temp - (P+1-(k+1)) * (P+2-(k+1)) * (P+3-(k+1))/6 + (j+1) * (P+1-(k+1)) - (j) * (j+1)/2 + i + 1;

        ppt_connect(0, ele) = nd[0];
        ppt_connect(1, ele) = nd[1];
        ppt_connect(2, ele) = nd[2];
        ppt_connect(3, ele) = nd[3];
        ele++;
      }
    }
  }
}

mdvector<double> Tets::calc_shape(const std::vector<double> &loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2, 3};
  gmsh_nodes[2] =  {0, 4, 1, 6, 5, 2, 7, 9, 8, 3};

  unsigned int shape_order = tet_nodes_to_order(nNodes);

  if (shape_order > 2)
    ThrowException("Tetrahedra with supplied shape_order unsupported!");

  mdvector<double> shape_val({nNodes}, 0.0);

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D - i; j++)
    {
      for (unsigned int k = 0; k < nPts1D - i - j; k++)
      {
        loc_pts(pt,0) = loc_pts_1D[k];
        loc_pts(pt,1) = loc_pts_1D[j];
        loc_pts(pt,2) = loc_pts_1D[i];
        pt++;
      }
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner3D(shape_order, loc_pts(i, 0), loc_pts(i, 1), loc_pts(i, 2), j); 

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
      val += inv_vand_s(i, nd) * Dubiner3D(shape_order, loc[0], loc[1], loc[2], i);
    }

    shape_val(gmsh_nodes[shape_order][nd]) = val;
  }

  return shape_val;

}

mdvector<double> Tets::calc_d_shape(const std::vector<double> &loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2, 3};
  gmsh_nodes[2] =  {0, 4, 1, 6, 5, 2, 7, 9, 8, 3};

  unsigned int shape_order = tet_nodes_to_order(nNodes);

  if (shape_order > 2)
    ThrowException("Tetrahedra with supplied shape_order unsupported!");

  mdvector<double> dshape_val({nNodes, 3}, 0.0);

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D - i; j++)
    {
      for (unsigned int k = 0; k < nPts1D - i - j; k++)
      {
        loc_pts(pt,0) = loc_pts_1D[k];
        loc_pts(pt,1) = loc_pts_1D[j];
        loc_pts(pt,2) = loc_pts_1D[i];
        pt++;
      }
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner3D(shape_order, loc_pts(i, 0), loc_pts(i, 1), loc_pts(i, 2), j); 

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
        val += inv_vand_s(i, nd) * dDubiner3D(shape_order, loc[0], loc[1], loc[2], dim, i);
      }

      dshape_val(gmsh_nodes[shape_order][nd], dim) = val;
    }
  }

  return dshape_val;
}

void Tets::modify_sensor()
{
//  /* Obtain locations of "collapsed" quad solution points */
//  unsigned int nNodesQuad = 4;
//  mdvector<double> nodes({nDims, nNodesQuad}); 
//  nodes(0, 0) = -1.0; nodes(1, 0) = -1.0; 
//  nodes(0, 1) = 1.0; nodes(1, 1) = -1.0; 
//  nodes(0, 2) = -1.0; nodes(1, 2) = 1.0; 
//  nodes(0, 3) = -1.0; nodes(1, 3) = 1.0; 
//
//  int nSpts2D = nSpts1D * nSpts1D;
//  mdvector<double> loc_spts_quad({nSpts2D, nDims}, 0);
//
//  for (unsigned int spt = 0; spt < nSpts2D; spt++)
//  {
//    for (unsigned int nd = 0; nd < nNodesQuad; nd++)
//    {
//      int i = nd % 2; int j = nd / 2;
//      for (unsigned int dim = 0; dim < nDims; dim++)
//      {
//        loc_spts_quad(spt, dim) += nodes(dim, nd) * Lagrange({-1, 1}, loc_spts_1D[spt % nSpts1D], i) * 
//                                                    Lagrange({-1, 1}, loc_spts_1D[spt / nSpts1D], j);
//      }
//    }
//  }
//
//  /* Setup spt to collapsed spt extrapolation operator (oppEc) */
//  std::vector<double> loc(nDims, 0.0);
//  mdvector<double> oppEc({nSpts2D, nSpts});
//  for (unsigned int spt = 0; spt < nSpts; spt++)
//  {
//    for (unsigned int spt_q = 0; spt_q < nSpts2D; spt_q++)
//    {
//      for (unsigned int dim = 0; dim < nDims; dim++)
//        loc[dim] = loc_spts_quad(spt_q , dim);
//
//      oppEc(spt_q, spt) = calc_nodal_basis(spt, loc);
//    }
//  }
//
//  /* Multiply oppS by oppEc to get modified operator */
//  auto temp = oppS;
//  oppS.assign({nSpts2D * nDims, nSpts});
//
//  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts2D * nDims, nSpts, nSpts2D,
//      1.0, temp.data(), temp.ldim(), oppEc.data(), oppEc.ldim(), 0.0, oppS.data(), oppS.ldim());
//
} 
