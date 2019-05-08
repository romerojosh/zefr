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

#ifdef _MKL_BLAS
#include "mkl_cblas.h"
#elif _ESSL_BLAS
#include "essl.h"
#else
#include "cblas.h"
#endif

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

Hexas::Hexas(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order)
{
  etype = HEX;

  this->init(geo,input,elesObjID,startEle,endEle,order);

  this->nQpts = input->nQpts1D * input->nQpts1D * input->nQpts1D;

  /* Generic hexahedral geometry */
  nDims = 3;
  nFaces = 6;

  nNodes = geo->nNodesPerEleBT[HEX];

  nNdSide = cbrt(nNodes);

  if (nNdSide*nNdSide*nNdSide != nNodes)
    ThrowException("For Lagrange hex of order N, must have (N+1)^3 shape points.");

  nSpts = (this->order+1) * (this->order+1) * (this->order+1);
  nSpts1D = this->order+1;

  nFptsPerFace = nSpts1D * nSpts1D;
  nFpts_face = {nFptsPerFace, nFptsPerFace, nFptsPerFace, nFptsPerFace, nFptsPerFace, nFptsPerFace};
  nFpts = (nSpts1D * nSpts1D) * nFaces;
  nPpts = nSpts;
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
  auto weights_spts_1D = Gauss_Legendre_weights(nSpts1D); 

  // For integration of quantities over faces [NOTE: can be used as 1D OR 2D array]
  weights_fpts.assign({nSpts1D,nSpts1D});
  for (unsigned int fpt1 = 0; fpt1 < nSpts1D; fpt1++)
    for (unsigned int fpt2 = 0; fpt2 < nSpts1D; fpt2++)
      weights_fpts(fpt1,fpt2) = weights_spts_1D[fpt1] * weights_spts_1D[fpt2];

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
      for (unsigned int k = 0; k < nSpts1D; k++)
      {
        loc_spts(spt,0) = loc_spts_1D[k];
        loc_spts(spt,1) = loc_spts_1D[j];
        loc_spts(spt,2) = loc_spts_1D[i];
        idx_spts(spt,0) = k;
        idx_spts(spt,1) = j;
        idx_spts(spt,2) = i;
        weights_spts(spt) = weights_spts_1D[i] * weights_spts_1D[j] * weights_spts_1D[k];

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
  
  /* Setup plot point locations (equidistant) */
  auto loc_ppts_1D = Shape_pts(order);

  unsigned int ppt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      for (unsigned int k = 0; k < nSpts1D; k++)
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
  auto weights_qpts_1D = Gauss_Legendre_weights(input->nQpts1D);
  weights_qpts.assign({input->nQpts1D * input->nQpts1D * input->nQpts1D});

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
        weights_qpts(qpt) =  weights_qpts_1D[i] * weights_qpts_1D[j] * weights_qpts_1D[k];
        qpt++;
      }
    }
  }

}

void Hexas::set_normals(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for normals */
  tnorm.assign({nFpts,nDims});
  tdA.assign({nFpts}, 1.0);

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

void Hexas::set_vandermonde_mats()
{
  /* Set vandermonde and inverse for 3D Legendre basis */
  vand.assign({nSpts, nSpts});

  for (unsigned int i = 0; i < nSpts; i++)
  {
    for (unsigned int j = 0; j < nSpts; j++)
      vand(i,j) = Legendre3D(order, loc_spts(i, 0), loc_spts(i, 1), loc_spts(i, 2), j);
  }

  inv_vand.assign({nSpts, nSpts});
  vand.inverse(inv_vand);
}

void Hexas::set_oppRestart(unsigned int order_restart, bool use_shape)
{
  unsigned int nRpts1D = (order_restart + 1);
  unsigned int nRpts2D = nRpts1D * nRpts1D;
  unsigned int nRpts = nRpts1D * nRpts1D * nRpts1D;

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
      int j = (rpt / nRpts1D) % nRpts1D;
      int k = rpt / nRpts2D;
      oppRestart(spt,rpt) = Lagrange(loc_rpts_1D, i, loc[0]) * 
                            Lagrange(loc_rpts_1D, j, loc[1]) *
                            Lagrange(loc_rpts_1D, k, loc[2]);
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

void Hexas::calc_nodal_basis(double *loc, double* basis)
{
  if (lag_i.size() < nSpts1D || lag_j.size() < nSpts1D || lag_k.size() < nSpts1D)
  {
    lag_i.resize(nSpts1D);
    lag_j.resize(nSpts1D);
    lag_k.resize(nSpts1D);
  }

  for (int spt = 0; spt < nSpts1D; spt++)
  {
    lag_i[spt] = Lagrange(loc_spts_1D, spt, loc[0]);
    lag_j[spt] = Lagrange(loc_spts_1D, spt, loc[1]);
    lag_k[spt] = Lagrange(loc_spts_1D, spt, loc[2]);
  }

  for (int k = 0; k < nSpts1D; k++)
    for (int j = 0; j < nSpts1D; j++)
      for (int i = 0; i < nSpts1D; i++)
        basis[i+nSpts1D*(j+nSpts1D*k)] = lag_i[i] * lag_j[j] * lag_k[k];
}

double Hexas::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
#ifndef _FR_HEXAS

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

#else
  // Use FR hexahedrons
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

#endif

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

mdvector<double> Hexas::get_face_nodes(unsigned int face, unsigned int P)
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

mdvector<double> Hexas::get_face_weights(unsigned int face, unsigned int P)
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

void Hexas::project_face_point(int face, const double* loc, double* ploc)
{
  switch(face)
  {
    case 0: /* Bottom face */
      ploc[0] = loc[0];
      ploc[1] = loc[1];
      ploc[2] = -1.0;
      break;

    case 1: /* Top face */
      ploc[0] = -loc[0];
      ploc[1] = loc[1];
      ploc[2] = 1.0;
      break;

    case 2: /* Left face */
      ploc[0] = -1.0;
      ploc[1] = loc[0];
      ploc[2] = loc[1];
      break;

    case 3: /* Right face */
      ploc[0] = 1.0;
      ploc[1] = -loc[0];
      ploc[2] = loc[1];
      break;

    case 4: /* Front face */
      ploc[0] = -loc[0];
      ploc[1] = -1.0;
      ploc[2] = loc[1];
      break;

    case 5: /* Back face */
      ploc[0] = loc[0];
      ploc[1] = 1.0;
      ploc[2] = loc[1];
      break;
  }
}

double Hexas::calc_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc)
{
  int i = pt % nSpts1D;
  int j = pt / nSpts1D;

  return Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);
}

double Hexas::calc_orthonormal_basis(unsigned int mode, const double *loc)
{
  return Legendre3D(order, loc[0], loc[1], loc[2], mode);
}

void Hexas::setup_PMG(int pro_order, int res_order)
{
  unsigned int nSpts_pro_1D = pro_order + 1;
  unsigned int nSpts_res_1D = res_order + 1;
  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D * nSpts_pro_1D;
  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D * nSpts_res_1D;

  std::vector<double> loc(nDims, 0.0);

  if (order != pro_order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    std::vector<mdvector<double>> opps(pro_order + 1);

    /* Form operator by sequential multiplication of single order prolongation operators */
    for (unsigned int P = pro_order; P > order; P--)
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

    for (unsigned int P = pro_order; P > order + 1; P--)
    {
      mdvector<double> opp({nSpts_pro, (P-1) * (P-1) * (P-1)});

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, (P-1) * (P-1) * (P-1),
          P * P * P, 1.0, opps[P].data(), P * P * P, opps[P-1].data(), (P-1) * (P-1) * (P-1), 0.0, 
          opp.data(), (P-1) * (P-1) * (P-1));

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
    for (unsigned int P = res_order; P < order; P++)
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

    for (unsigned int P = res_order; P < order - 1; P++)
    {
      mdvector<double> opp({nSpts_res, (P+3) * (P+3) * (P+3)});

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_res, (P+3) * (P+3) * (P+3),
          (P+2) * (P+2) * (P+2), 1.0, opps[P].data(), (P+2) * (P+2) * (P+2), opps[P+1].data(), 
          (P+3) * (P+3) * (P+3), 0.0, opp.data(), (P+3) * (P+3) * (P+3));

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

void Hexas::setup_ppt_connectivity()
{
  unsigned int nSubelements1D = nSpts1D - 1;
  nSubelements = nSubelements1D * nSubelements1D * nSubelements1D;
  nNodesPerSubelement = 8;

  /* Allocate memory for local plot point connectivity and solution at plot points */
  ppt_connect.assign({8, nSubelements});

  /* Setup plot "subelement" connectivity */
  std::vector<unsigned int> nd(8,0);

  unsigned int ele = 0;
  nd[0] = 0; nd[1] = 1; nd[2] = nSubelements1D + 2; nd[3] = nSubelements1D + 1;
  nd[4] = (nSubelements1D + 1) * (nSubelements1D + 1); nd[5] = nd[4] + 1; 
  nd[6] = nd[4] + nSubelements1D + 2; nd[7] = nd[4] + nSubelements1D + 1;

  for (unsigned int i = 0; i < nSubelements1D; i++)
  {
    for (unsigned int j = 0; j < nSubelements1D; j++)
    {
      for (unsigned int k = 0; k < nSubelements1D; k++)
      {
        for (unsigned int node = 0; node < 8; node ++)
        {
          ppt_connect(node, ele) = nd[node] + k;
        }

        ele++;
      }

      for (unsigned int node = 0; node < 8; node ++)
        nd[node] += (nSubelements1D + 1);

    }

    for (unsigned int node = 0; node < 8; node ++)
      nd[node] += (nSubelements1D + 1);
  }
}

void Hexas::calc_shape(mdvector<double> &shape_val, const double* loc)
{
  double xi = loc[0];
  double eta = loc[1];
  double mu = loc[2];

  if (xlist.size() != nNdSide)
  {
    xlist.resize(nNdSide);
    double dxi = 2./(nNdSide-1);

    for (int i=0; i<nNdSide; i++)
      xlist[i] = -1. + i*dxi;
  }

  if (ijk2gmsh.size() != nNodes)
    ijk2gmsh = structured_to_gmsh_hex(nNodes);

  // Pre-compute Lagrange function values to avoid redundant calculations
  if (lag_i.size() < nNdSide || lag_j.size() < nNdSide || lag_k.size() < nNdSide)
  {
    lag_i.resize(nNdSide); lag_j.resize(nNdSide); lag_k.resize(nNdSide);
  }

  for (int i = 0; i < nNdSide; i++)
  {
    lag_i[i] = Lagrange(xlist,  xi, i);
    lag_j[i] = Lagrange(xlist, eta, i);
    lag_k[i] = Lagrange(xlist,  mu, i);
  }

  int pt = 0;
  for (int k = 0; k < nNdSide; k++)
    for (int j = 0; j < nNdSide; j++)
      for (int i = 0; i < nNdSide; i++)
      {
        shape_val(ijk2gmsh[pt]) = lag_i[i] * lag_j[j] * lag_k[k];
        pt++;
      }
}

void Hexas::calc_d_shape(mdvector<double> &dshape_val, const double* loc)
{
  double xi = loc[0];
  double eta = loc[1];
  double mu = loc[2];

  if (xlist.size() != nNdSide)
  {
    xlist.resize(nNdSide);
    double dxi = 2./(nNdSide-1);

    for (int i=0; i<nNdSide; i++)
      xlist[i] = -1. + i*dxi;
  }

  if (ijk2gmsh.size() != nNodes)
    ijk2gmsh = structured_to_gmsh_hex(nNodes);

  // Pre-compute Lagrange function values to save redundant calculations
  if (dlag_i.size() != nNdSide)
  {
    lag_i.resize(nNdSide);   lag_j.resize(nNdSide);  lag_k.resize(nNdSide);
    dlag_i.resize(nNdSide); dlag_j.resize(nNdSide); dlag_k.resize(nNdSide);
  }

  for (int i = 0; i < nNdSide; i++)
  {
    lag_i[i] = Lagrange(xlist,  xi, i);
    lag_j[i] = Lagrange(xlist, eta, i);
    lag_k[i] = Lagrange(xlist,  mu, i);
    dlag_i[i] = Lagrange_d1(xlist, i,  xi);
    dlag_j[i] = Lagrange_d1(xlist, i, eta);
    dlag_k[i] = Lagrange_d1(xlist, i,  mu);
  }

  int pt = 0;
  for (int k = 0; k < nNdSide; k++)
    for (int j = 0; j < nNdSide; j++)
      for (int i = 0; i < nNdSide; i++)
      {
        dshape_val(ijk2gmsh[pt],0) = dlag_i[i] *  lag_j[j] *  lag_k[k];
        dshape_val(ijk2gmsh[pt],1) =  lag_i[i] * dlag_j[j] *  lag_k[k];
        dshape_val(ijk2gmsh[pt],2) =  lag_i[i] *  lag_j[j] * dlag_k[k];
        pt++;
      }
}

void Hexas::modify_sensor(){ /* Do nothing */ }

double Hexas::rst_max_lim(int dim, double* rst)
{
  return std::min(rst[dim], 1.0);
}

double Hexas::rst_min_lim(int dim, double* rst)
{
  return std::max(rst[dim], -1.0);
}
