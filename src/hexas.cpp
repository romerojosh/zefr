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
#include "hexas.hpp"
#include "funcs.hpp"

extern "C" {
#include "cblas.h"
}

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

Hexas::Hexas(GeoStruct *geo, InputStruct *input, int order)
{
  this->geo = geo;
  this->input = input;  
  this->shape_order = geo->shape_order;  
  this->nEles = geo->nEles;  
  this->nQpts = input->nQpts1D * input->nQpts1D * input->nQpts1D;

  /* Generic hexahedral geometry */
  nDims = 3;
  nFaces = 6;
  nNodes = geo->nNodesPerEle;
  
  geo->nFacesPerEle = 6;
  geo->nNodesPerFace = 4;
  geo->nCornerNodes = 8;

  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order+1) * (input->order+1) * (input->order+1);
    nSpts1D = input->order+1;
    this->order = input->order;
  }
  else
  {
    nSpts = (order+1) * (order+1) * (order+1);
    nSpts1D = order+1;
    this->order = order;
  }

  nFpts = (nSpts1D * nSpts1D) * nFaces;
  nPpts = (nSpts1D + 2) * (nSpts1D + 2) * (nSpts1D + 2);
  
  if (input->equation == AdvDiff || input->equation == Burgers)
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

void Hexas::set_locs()
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
  auto weights_spts_temp = Gauss_Legendre_weights(nSpts1D); 
  weights_spts.assign({nSpts1D});
  for (unsigned int spt = 0; spt < nSpts1D; spt++)
    weights_spts(spt) = weights_spts_temp[spt];

  // For integration of quantities over faces
  weights_fpts.assign({nSpts1D,nSpts1D});
  for (unsigned int fpt1 = 0; fpt1 < nSpts1D; fpt1++)
    for (unsigned int fpt2 = 0; fpt2 < nSpts1D; fpt2++)
      weights_fpts(fpt1,fpt2) = weights_spts(fpt1)*weights_spts(fpt2);

  loc_DFR_1D = loc_spts_1D;
  loc_DFR_1D.insert(loc_DFR_1D.begin(), -1.0);
  loc_DFR_1D.insert(loc_DFR_1D.end(), 1.0);

  /* Setup solution point locations */
  unsigned int spt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      for (unsigned int k = 0; k < nSpts1D; k++)
      {
        loc_spts(spt,0) = loc_spts_1D[k];
        loc_spts(spt,1) = loc_spts_1D[j];
        loc_spts(spt,2) = loc_spts_1D[i];
        idx_spts(spt,0) = k;
        idx_spts(spt,1) = j;
        idx_spts(spt,2) = i;
        spt++;
      }
    }
  }

  /* Setup flux point locations */
  /* Note: Flux points are ordered progressing from corner nearest
   * to starting face node (see geometry.cpp) and sweeping up to
   * opposite corner.
   * Some diagrams:
   * Bottom/Top Faces:
   * ^ y
   * |
   * 3 ----- 2
   * | 2   3 |
   * | 0   1 |
   * 0-------1 --> x
   *
   * Front/Back Faces:
   * ^ z
   * |
   * 4 ----- 5
   * | 2   3 |
   * | 0   1 |
   * 0-------1 --> x
   *
   * Left/Right Faces:
   *               ^ z
   *               |
   *       7 ----- 4
   *       | 2   3 |
   *       | 0   1 |
   * y <-- 3-------0 
   * */

  unsigned int fpt = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      for (unsigned int k = 0; k < nSpts1D; k++)
      {
        switch(i)
        {
          case 0: /* Bottom face */
            loc_fpts(fpt,0) = loc_spts_1D[k];
            loc_fpts(fpt,1) = loc_spts_1D[j]; 
            loc_fpts(fpt,2) = -1.0; 
            idx_fpts(fpt,0) = k;
            idx_fpts(fpt,1) = j;
            idx_fpts(fpt,2) = -1; break;

          case 1: /* Top face */
            loc_fpts(fpt,0) = loc_spts_1D[nSpts1D - k - 1];
            loc_fpts(fpt,1) = loc_spts_1D[j]; 
            loc_fpts(fpt,2) = 1.0; 
            idx_fpts(fpt,0) = nSpts1D - k - 1;
            idx_fpts(fpt,1) = j;
            idx_fpts(fpt,2) = nSpts1D; break;

          case 2: /* Left face */
            loc_fpts(fpt,0) = -1.0;
            loc_fpts(fpt,1) = loc_spts_1D[k];
            loc_fpts(fpt,2) = loc_spts_1D[j];
            idx_fpts(fpt,0) = -1;
            idx_fpts(fpt,1) = k;
            idx_fpts(fpt,2) = j; break;

          case 3: /* Right face */
            loc_fpts(fpt,0) = 1.0;
            loc_fpts(fpt,1) = loc_spts_1D[nSpts1D - k - 1];
            loc_fpts(fpt,2) = loc_spts_1D[j];
            idx_fpts(fpt,0) = nSpts1D;
            idx_fpts(fpt,1) = nSpts1D - k - 1;
            idx_fpts(fpt,2) = j; break;

          case 4: /* Front face */
            loc_fpts(fpt,0) = loc_spts_1D[nSpts1D - k - 1];
            loc_fpts(fpt,1) = -1.0;
            loc_fpts(fpt,2) = loc_spts_1D[j];
            idx_fpts(fpt,0) = nSpts1D - k - 1;
            idx_fpts(fpt,1) = -1;
            idx_fpts(fpt,2) = j; break;

          case 5: /* Back face */
            loc_fpts(fpt,0) = loc_spts_1D[k];
            loc_fpts(fpt,1) = 1.0;
            loc_fpts(fpt,2) = loc_spts_1D[j];
            idx_fpts(fpt,0) = k;
            idx_fpts(fpt,1) = nSpts1D;
            idx_fpts(fpt,2) = j; break;
        }
        fpt++;

      }
    }
  }
  
  /* Setup plot point locations */
  auto loc_ppts_1D = loc_spts_1D;
  loc_ppts_1D.insert(loc_ppts_1D.begin(), -1.0);
  loc_ppts_1D.insert(loc_ppts_1D.end(), 1.0);

  unsigned int ppt = 0;
  for (unsigned int i = 0; i < nSpts1D+2; i++)
  {
    for (unsigned int j = 0; j < nSpts1D+2; j++)
    {
      for (unsigned int k = 0; k < nSpts1D+2; k++)
      {
        loc_ppts(ppt,0) = loc_ppts_1D[k];
        loc_ppts(ppt,1) = loc_ppts_1D[j];
        loc_ppts(ppt,2) = loc_ppts_1D[i];
        idx_ppts(ppt,0) = k;
        idx_ppts(ppt,1) = j;
        idx_ppts(ppt,2) = i;
        ppt++;
      }
    }
  }

  /* Setup gauss quadrature point locations and weights */
  loc_qpts_1D = Gauss_Legendre_pts(input->nQpts1D); 
  weights_qpts = Gauss_Legendre_weights(input->nQpts1D);

  /* Setup quadrature point locations */
  unsigned int qpt = 0;
  for (unsigned int i = 0; i < input->nQpts1D; i++)
  {
    for (unsigned int j = 0; j < input->nQpts1D; j++)
    {
      for (unsigned int k = 0; k < input->nQpts1D; k++)
      {
        loc_qpts(qpt,0) = loc_qpts_1D[k];
        loc_qpts(qpt,1) = loc_qpts_1D[j];
        loc_qpts(qpt,2) = loc_qpts_1D[i];
        idx_qpts(qpt,0) = k;
        idx_qpts(qpt,1) = j;
        idx_qpts(qpt,2) = i;
        qpt++;
      }
    }
  }

}

void Hexas::set_normals(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for normals */
  tnorm.assign({nFpts,nDims});

  /* Setup parent-space (transformed) normals at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    switch(fpt/(nSpts1D * nSpts1D))
    {
      case 0: /* Bottom */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = 0.0; 
        tnorm(fpt,2) = -1.0; break;

      case 1: /* Top */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = 0.0; 
        tnorm(fpt,2) = 1.0; break;

      case 2: /* Left */
        tnorm(fpt,0) = -1.0;
        tnorm(fpt,1) = 0.0; 
        tnorm(fpt,2) = 0.0; break;

      case 3: /* Right */
        tnorm(fpt,0) = 1.0;
        tnorm(fpt,1) = 0.0; 
        tnorm(fpt,2) = 0.0; break;

      case 4: /* Front */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = -1.0; 
        tnorm(fpt,2) = 0.0; break;

      case 5: /* Back */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = 1.0; 
        tnorm(fpt,2) = 0.0; break;
    }

  }
}

double Hexas::calc_nodal_basis(unsigned int spt,
                               const std::vector<double> &loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);
  unsigned int k = idx_spts(spt,2);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]) * Lagrange(loc_spts_1D, k, loc[2]);

  return val;
}

double Hexas::calc_nodal_basis(unsigned int spt, double *loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);
  unsigned int k = idx_spts(spt,2);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]) * Lagrange(loc_spts_1D, k, loc[2]);

  return val;
}

double Hexas::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
   * boundary points for DFR) */
  unsigned int i = idx_spts(spt,0) + 1;
  unsigned int j = idx_spts(spt,1) + 1;
  unsigned int k = idx_spts(spt,2) + 1;

  double val = 0.0;

  if (dim == 0)
  {
    val = Lagrange_d1(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]) * Lagrange(loc_DFR_1D, k, loc[2]);
  }
  else if (dim == 1)
  {
    val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange_d1(loc_DFR_1D, j, loc[1]) * Lagrange(loc_DFR_1D, k, loc[2]);
  }
  else
  {
    val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]) * Lagrange_d1(loc_DFR_1D, k, loc[2]);
  }

  return val;

}

double Hexas::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);
  unsigned int k = idx_spts(spt,2);

  double val = 0.0;

  if (dim == 0)
  {
    val = Lagrange_d1(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]) * Lagrange(loc_spts_1D, k, loc[2]);
  }
  else if (dim == 1)
  {
    val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange_d1(loc_spts_1D, j, loc[1]) * Lagrange(loc_spts_1D, k, loc[2]);
  }
  else
  {
    val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]) * Lagrange_d1(loc_spts_1D, k, loc[2]);
  }

  return val;

}

double Hexas::calc_d_nodal_basis_fpts(unsigned int fpt,
              const std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
   * boundary points for DFR) */
  unsigned int i = idx_fpts(fpt,0) + 1;
  unsigned int j = idx_fpts(fpt,1) + 1;
  unsigned int k = idx_fpts(fpt,2) + 1;

  double val = 0.0;

  if (dim == 0)
  {
    val = Lagrange_d1(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]) * Lagrange(loc_DFR_1D, k, loc[2]);
  }
  else if (dim == 1)
  {
    val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange_d1(loc_DFR_1D, j, loc[1]) * Lagrange(loc_DFR_1D, k, loc[2]);
  }
  else
  {
    val = Lagrange(loc_DFR_1D, i, loc[0]) * Lagrange(loc_DFR_1D, j, loc[1]) * Lagrange_d1(loc_DFR_1D, k, loc[2]);
  }

  return val;

}

void Hexas::setup_PMG(int pro_order, int res_order)
{
  unsigned int nSpts_pro_1D = pro_order + 1;
  unsigned int nSpts_res_1D = res_order + 1;
  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D * nSpts_pro_1D;
  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D * nSpts_res_1D;

  std::vector<double> loc(nDims, 0.0);

  if (order != input->order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    std::vector<mdvector<double>> opps(pro_order + 1);

    /* Form operator by sequential multiplication of single order prolongation operators */
    for (int P = pro_order; P > order; P--)
    {
      opps[P].assign({(P+1) * (P+1) * (P+1), P * P * P}, 0);

      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
      auto loc_spts_P2_1D = Gauss_Legendre_pts(P); 

      for (unsigned int spt = 0; spt < P * P * P; spt++)
      {
        int i1 = spt % P;
        int j1 = (spt / P) % P;
        int k1 = spt / (P * P);

        for (unsigned int pspt = 0; pspt < (P+1) * (P+1) * (P+1); pspt++)
        {
          int i2 = pspt % (P+1);
          int j2 = (pspt / (P+1)) % (P+1);
          int k2 = pspt / ((P+1) * (P+1));
          loc[0] = loc_spts_P1_1D[i2];
          loc[1] = loc_spts_P1_1D[j2];
          loc[2] = loc_spts_P1_1D[k2];

          opps[P](pspt, spt) = Lagrange(loc_spts_P2_1D, i1, loc[0]) * 
                               Lagrange(loc_spts_P2_1D, j1, loc[1]) *
                               Lagrange(loc_spts_P2_1D, k1, loc[2]);
        }
      }
    }

    for (int P = pro_order; P > order + 1; P--)
    {
      mdvector<double> opp({nSpts_pro, (P-1) * (P-1) * (P-1)});

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, (P-1) * (P-1) * (P-1),
          P * P * P, 1.0, opps[P].data(), opps[P].ldim(), opps[P-1].data(), opps[P-1].ldim(), 0.0, opp.data(), opp.ldim());

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
      opps[P].assign({(P+1) * (P+1) * (P+1), (P+2) * (P+2) * (P+2)}, 0);

      auto loc_spts_P1_1D = Gauss_Legendre_pts(P+1); 
      auto loc_spts_P2_1D = Gauss_Legendre_pts(P+2); 

      for (unsigned int spt = 0; spt < (P+2) * (P+2) * (P+2); spt++)
      {
        int i1 = spt % (P+2);
        int j1 = (spt / (P+2)) % (P+2);
        int k1 = spt / ((P+2) * (P+2));

        for (unsigned int rspt = 0; rspt < (P+1) * (P+1) * (P+1); rspt++)
        {
          int i2 = rspt % (P+1);
          int j2 = (rspt / (P+1)) % (P+1);
          int k2 = rspt / ((P+1) * (P+1));

          loc[0] = loc_spts_P1_1D[i2];
          loc[1] = loc_spts_P1_1D[j2];
          loc[2] = loc_spts_P1_1D[k2];

          opps[P](rspt, spt) = Lagrange(loc_spts_P2_1D, i1, loc[0]) * 
                               Lagrange(loc_spts_P2_1D, j1, loc[1]) *
                               Lagrange(loc_spts_P2_1D, k1, loc[2]);
        }
      }
    }

    for (int P = res_order; P < order - 1; P++)
    {
      mdvector<double> opp({nSpts_res, (P+3) * (P+3) * (P+3)});

      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nSpts_res, (P+3) * (P+3) * (P+3),
          (P+2) * (P+2) * (P+2), 1.0, opps[P].data(), opps[P].ldim(), opps[P+1].data(), opps[P+1].ldim(), 0.0, opp.data(), opp.ldim());

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

void Hexas::transform_dU(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo->iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double dUtemp0 = dU_spts(spt, ele, n, 0);
        double dUtemp1 = dU_spts(spt, ele, n, 1);

        dU_spts(spt, ele, n, 0) = dU_spts(spt, ele, n, 0) * inv_jaco_spts(spt, ele, 0, 0) +
                                  dU_spts(spt, ele, n, 1) * inv_jaco_spts(spt, ele, 1, 0) +
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 2, 0);

        dU_spts(spt, ele, n, 1) = dUtemp0 * inv_jaco_spts(spt, ele, 0, 1) +
                                  dU_spts(spt, ele, n, 1) * inv_jaco_spts(spt, ele, 1, 1) +
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 2, 1);
                                  
        dU_spts(spt, ele, n, 2) = dUtemp0 * inv_jaco_spts(spt, ele, 0, 2) +
                                  dUtemp1 * inv_jaco_spts(spt, ele, 1, 2) +
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 2, 2);

        dU_spts(spt, ele, n, 0) /= jaco_det_spts(spt, ele);
        dU_spts(spt, ele, n, 1) /= jaco_det_spts(spt, ele);
        dU_spts(spt, ele, n, 2) /= jaco_det_spts(spt, ele);
      }
    }
  }
#endif

#ifdef _GPU
  transform_dU_hexa_wrapper(dU_spts_d, inv_jaco_spts_d, jaco_det_spts_d, nSpts, nEles, nVars,
      nDims, input->equation, input->overset, geo->iblank_cell_d.data());
  //dU_spts = dU_spts_d;
  check_error();
#endif

}

void Hexas::transform_flux(unsigned int startEle, unsigned int endEle)
{
#ifdef _CPU
#pragma omp parallel for collapse(2)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = startEle; ele < endEle; ele++)
    {
      if (input->overset && geo->iblank_cell(ele) != NORMAL) continue;

      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double Ftemp0 = F_spts(spt, ele, n, 0);
        double Ftemp1 = F_spts(spt, ele, n, 1);

        F_spts(spt, ele, n, 0) = F_spts(spt, ele, n, 0) * inv_jaco_spts(spt, ele, 0, 0) +
                                  F_spts(spt, ele, n, 1) * inv_jaco_spts(spt, ele, 0, 1) +
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 0, 2);

        F_spts(spt, ele, n, 1) = Ftemp0 * inv_jaco_spts(spt, ele, 1, 0) +
                                  F_spts(spt, ele, n, 1) * inv_jaco_spts(spt, ele, 1, 1) +
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 1, 2);
                                  
        F_spts(spt, ele, n, 2) = Ftemp0 * inv_jaco_spts(spt, ele, 2, 0) +
                                  Ftemp1 * inv_jaco_spts(spt, ele, 2, 1) +
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(spt, ele, 2, 2);

      }

    }
  }

#endif

#ifdef _GPU
  //F_spts_d = F_spts;
  transform_flux_hexa_wrapper(F_spts_d, inv_jaco_spts_d, nSpts, nEles, nVars,
      nDims, input->equation, input->overset, geo->iblank_cell_d.data());

  check_error();

  //F_spts = F_spts_d;
#endif
}

void Hexas::transform_dFdU()
{
#ifdef _CPU
#pragma omp parallel for collapse(4)
  for (unsigned int nj = 0; nj < nVars; nj++)
  {
    for (unsigned int ni = 0; ni < nVars; ni++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int spt = 0; spt < nSpts; spt++)
        {
          double dFdUtemp0 = dFdU_spts(spt, ele, ni, nj, 0);
          double dFdUtemp1 = dFdU_spts(spt, ele, ni, nj, 1);

          dFdU_spts(spt, ele, ni, nj, 0) = dFdU_spts(spt, ele, ni, nj, 0) * inv_jaco_spts(spt, ele, 0, 0) +
                                           dFdU_spts(spt, ele, ni, nj, 1) * inv_jaco_spts(spt, ele, 0, 1) +
                                           dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco_spts(spt, ele, 0, 2);

          dFdU_spts(spt, ele, ni, nj, 1) = dFdUtemp0 * inv_jaco_spts(spt, ele, 1, 0) +
                                           dFdU_spts(spt, ele, ni, nj, 1) * inv_jaco_spts(spt, ele, 1, 1) +
                                           dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco_spts(spt, ele, 1, 2);
                                    
          dFdU_spts(spt, ele, ni, nj, 2) = dFdUtemp0 * inv_jaco_spts(spt, ele, 2, 0) +
                                           dFdUtemp1 * inv_jaco_spts(spt, ele, 2, 1) +
                                           dFdU_spts(spt, ele, ni, nj, 2) * inv_jaco_spts(spt, ele, 2, 2);
        }
      }
    }
  }

#endif

#ifdef _GPU
  transform_dFdU_hexa_wrapper(dFdU_spts_d, inv_jaco_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);
  check_error();

#endif
}

mdvector<double> Hexas::calc_shape(unsigned int shape_order,
                                   const std::vector<double> &loc)
{
  mdvector<double> shape_val({nNodes}, 0.0);
  double xi = loc[0]; 
  double eta = loc[1];
  double mu = loc[2];

  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  /* 20-node Seredipity */
  if (nNodes == 20 || (shape_order == 2 and input->serendipity))
  {
    /* Corner Nodes */
    shape_val(0) = 0.125 * (1. - xi) * (1. - eta) * (1. - mu) * (-xi - eta - mu - 2.); 
    shape_val(1) = 0.125 * (1. + xi) * (1. - eta) * (1. - mu) * (xi - eta - mu - 2.);
    shape_val(2) = 0.125 * (1. + xi) * (1. + eta) * (1. - mu) * (xi + eta - mu - 2.);
    shape_val(3) = 0.125 * (1. - xi) * (1. + eta) * (1. - mu) * (-xi + eta - mu - 2.);
    shape_val(4) = 0.125 * (1. - xi) * (1. - eta) * (1. + mu) * (-xi - eta + mu - 2.);
    shape_val(5) = 0.125 * (1. + xi) * (1. - eta) * (1. + mu) * (xi - eta + mu - 2.);
    shape_val(6) = 0.125 * (1. + xi) * (1. + eta) * (1. + mu) * (xi + eta + mu - 2.);
    shape_val(7) = 0.125 * (1. - xi) * (1. + eta) * (1. + mu) * (-xi + eta + mu - 2.);

    /* Edge Nodes */
    shape_val(8) = 0.25 * (1. - xi*xi) * (1. - eta) * (1. - mu);
    shape_val(9) = 0.25 * (1. + xi) * (1. - eta*eta) * (1. - mu);
    shape_val(10) = 0.25 * (1. - xi*xi) * (1. + eta) * (1. - mu);
    shape_val(11) = 0.25 * (1. - xi) * (1. - eta*eta) * (1. - mu);
    shape_val(12) = 0.25 * (1. - xi) * (1. - eta) * (1. - mu*mu);
    shape_val(13) = 0.25 * (1. + xi) * (1. - eta) * (1. - mu*mu);
    shape_val(14) = 0.25 * (1. + xi) * (1. + eta) * (1. - mu*mu);
    shape_val(15) = 0.25 * (1. - xi) * (1. + eta) * (1. - mu*mu);
    shape_val(16) = 0.25 * (1. - xi*xi) * (1. - eta) * (1. + mu);
    shape_val(17) = 0.25 * (1. + xi) * (1. - eta*eta) * (1. + mu);
    shape_val(18) = 0.25 * (1. - xi*xi) * (1. + eta) * (1. + mu);
    shape_val(19) = 0.25 * (1. - xi) * (1. - eta*eta) * (1. + mu);

  }
  else
  {
    int nSide = cbrt(nNodes);

    if (nSide*nSide*nSide != nNodes)
    {
      std::cout << "nNodes = " << nNodes << std::endl;
      ThrowException("For Lagrange hex of order N, must have (N+1)^3 shape points.");
    }

    std::vector<double> xlist(nSide);
    double dxi = 2./(nSide-1);

    for (int i=0; i<nSide; i++)
      xlist[i] = -1. + i*dxi;

    auto ijk2gmsh = structured_to_gmsh_hex(nNodes);

    int pt = 0;
    for (int k = 0; k < nSide; k++)
    {
      for (int j = 0; j < nSide; j++)
      {
        for (int i = 0; i < nSide; i++)
        {
          shape_val(ijk2gmsh[pt]) = Lagrange(xlist, xi, i) * Lagrange(xlist, eta, j) * Lagrange(xlist, mu, k);
          pt++;
        }
      }
    }
  }

  return shape_val;
}

mdvector<double> Hexas::calc_d_shape(unsigned int shape_order,
                                     const std::vector<double> &loc)
{
  mdvector<double> dshape_val({nNodes, nDims}, 0);
  double xi = loc[0];
  double eta = loc[1];
  double mu = loc[2];

  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  if (shape_order == 2 && input->serendipity)
  {
    dshape_val(0, 0) = -0.125 * (1. - eta) * (1. - mu) * (-2.*xi - eta - mu - 1.); 
    dshape_val(1, 0) = 0.125 * (1. - eta) * (1. - mu) * (2.*xi - eta - mu - 1.); 
    dshape_val(2, 0) = 0.125 * (1. + eta) * (1. - mu) * (2.*xi + eta - mu - 1.); 
    dshape_val(3, 0) = -0.125 * (1. + eta) * (1. - mu) * (-2.*xi + eta - mu - 1.); 
    dshape_val(4, 0) = -0.125 * (1. - eta) * (1. + mu) * (-2.*xi - eta + mu - 1.); 
    dshape_val(5, 0) = 0.125 * (1. - eta) * (1. + mu) * (2.*xi - eta + mu - 1.); 
    dshape_val(6, 0) = 0.125 * (1. + eta) * (1. + mu) * (2.*xi + eta + mu - 1.); 
    dshape_val(7, 0) = -0.125 * (1. + eta) * (1. + mu) * (-2.*xi + eta + mu - 1.); 
    dshape_val(8, 0) = -0.5 * xi * (1. - eta) * (1. - mu); 
    dshape_val(9, 0) = 0.25 * (1. - eta*eta) * (1. - mu); 
    dshape_val(10, 0) = -0.5 * xi * (1. + eta) * (1. - mu); 
    dshape_val(11, 0) = -0.25 * (1. - eta*eta) * (1. - mu); 
    dshape_val(12, 0) = -0.25 * (1. - eta) * (1. - mu*mu); 
    dshape_val(13, 0) = 0.25 * (1. - eta) * (1. - mu*mu); 
    dshape_val(14, 0) = 0.25 * (1. + eta) * (1. - mu*mu); 
    dshape_val(15, 0) = -0.25 * (1. + eta) * (1. - mu*mu); 
    dshape_val(16, 0) = -0.5 * xi * (1. - eta) * (1. + mu); 
    dshape_val(17, 0) = 0.25 * (1. - eta*eta) * (1. + mu); 
    dshape_val(18, 0) = -0.5 * xi * (1. + eta) * (1. + mu); 
    dshape_val(19, 0) = -0.25 * (1. - eta*eta) * (1. + mu); 

    dshape_val(0, 1) = -0.125 * (1. - xi) * (1. - mu) * (-xi -2.*eta - mu - 1.); 
    dshape_val(1, 1) = -0.125 * (1. + xi) * (1. - mu) * (xi - 2.*eta - mu - 1.); 
    dshape_val(2, 1) = 0.125 * (1. + xi) * (1. - mu) * (xi + 2.*eta - mu - 1.); 
    dshape_val(3, 1) = 0.125 * (1. - xi) * (1. - mu) * (-xi + 2.*eta - mu - 1.); 
    dshape_val(4, 1) = -0.125 * (1. - xi) * (1. + mu) * (-xi - 2.*eta + mu - 1.); 
    dshape_val(5, 1) = -0.125 * (1. + xi) * (1. + mu) * (xi - 2.*eta + mu - 1.); 
    dshape_val(6, 1) = 0.125 * (1. + xi) * (1. + mu) * (xi + 2.*eta + mu - 1.); 
    dshape_val(7, 1) = 0.125 * (1. - xi) * (1. + mu) * (-xi + 2.*eta + mu - 1.); 
    dshape_val(8, 1) = -0.25 * (1. - xi*xi) * (1. - mu); 
    dshape_val(9, 1) = -0.5 * eta * (1. + xi) * (1. - mu); 
    dshape_val(10, 1) = 0.25 * (1. - xi*xi) * (1. - mu); 
    dshape_val(11, 1) = -0.5 * eta * (1. - xi) * (1. - mu); 
    dshape_val(12, 1) = -0.25 * (1. - xi) * (1. - mu*mu); 
    dshape_val(13, 1) = -0.25 * (1. + xi) * (1. - mu*mu); 
    dshape_val(14, 1) = 0.25 * (1. + xi) * (1. - mu*mu); 
    dshape_val(15, 1) = 0.25 * (1. - xi) * (1. - mu*mu); 
    dshape_val(16, 1) = -0.25 * (1. - xi*xi) * (1. + mu); 
    dshape_val(17, 1) = -0.5 * eta * (1. + xi) * (1. + mu); 
    dshape_val(18, 1) = 0.25 * (1. - xi*xi) * (1. + mu); 
    dshape_val(19, 1) = -0.5 * eta * (1. - xi) * (1. + mu); 

    dshape_val(0, 2) = -0.125 * (1. - xi) * (1. - eta) * (-xi - eta - 2.*mu - 1.); 
    dshape_val(1, 2) = -0.125 * (1. + xi) * (1. - eta) * (xi - eta - 2.*mu - 1.); 
    dshape_val(2, 2) = -0.125 * (1. + xi) * (1. + eta) * (xi + eta - 2.*mu - 1.); 
    dshape_val(3, 2) = -0.125 * (1. - xi) * (1. + eta) * (-xi + eta - 2.*mu - 1.); 
    dshape_val(4, 2) = 0.125 * (1. - xi) * (1. - eta) * (-xi - eta + 2.*mu - 1.); 
    dshape_val(5, 2) = 0.125 * (1. + xi) * (1. - eta) * (xi - eta + 2.*mu - 1.); 
    dshape_val(6, 2) = 0.125 * (1. + xi) * (1. + eta) * (xi + eta + 2.*mu - 1.); 
    dshape_val(7, 2) = 0.125 * (1. - xi) * (1. + eta) * (-xi + eta + 2.*mu - 1.); 
    dshape_val(8, 2) = -0.25 * (1. - xi*xi) * (1. - eta); 
    dshape_val(9, 2) = -0.25 * (1. + xi) * (1. - eta*eta); 
    dshape_val(10, 2) = -0.25 * (1. - xi*xi) * (1. + eta); 
    dshape_val(11, 2) = -0.25 * (1. - xi) * (1. - eta*eta); 
    dshape_val(12, 2) = -0.5 * mu * (1. - xi) * (1. - eta); 
    dshape_val(13, 2) = -0.5 * mu * (1. + xi) * (1. - eta); 
    dshape_val(14, 2) = -0.5 * mu * (1. + xi) * (1. + eta); 
    dshape_val(15, 2) = -0.5 * mu * (1. - xi) * (1. + eta); 
    dshape_val(16, 2) = 0.25 * (1. - xi*xi) * (1. - eta); 
    dshape_val(17, 2) = 0.25 * (1. + xi) * (1. - eta*eta); 
    dshape_val(18, 2) = 0.25 * (1. - xi*xi) * (1. + eta); 
    dshape_val(19, 2) = 0.25 * (1. - xi) * (1. - eta*eta); 

  }
  else
  {
    int nSide = cbrt(nNodes);

    if (nSide*nSide*nSide != nNodes)
      ThrowException("For Lagrange hex of order N, must have (N+1)^3 shape points.");

    std::vector<double> xlist(nSide);
    double dxi = 2./(nSide-1);

    for (int i=0; i<nSide; i++)
      xlist[i] = -1. + i*dxi;

    auto ijk2gmsh = structured_to_gmsh_hex(nNodes);

    int pt = 0;
    for (int k = 0; k < nSide; k++)
    {
      for (int j = 0; j < nSide; j++)
      {
        for (int i = 0; i < nSide; i++)
        {
          dshape_val(ijk2gmsh[pt],0) = dLagrange(xlist, xi, i) *  Lagrange(xlist, eta, j) *  Lagrange(xlist, mu, k);
          dshape_val(ijk2gmsh[pt],1) =  Lagrange(xlist, xi, i) * dLagrange(xlist, eta, j) *  Lagrange(xlist, mu, k);
          dshape_val(ijk2gmsh[pt],2) =  Lagrange(xlist, xi, i) *  Lagrange(xlist, eta, j) * dLagrange(xlist, mu, k);
          pt++;
        }
      }
    }
  }

  return dshape_val;
}
