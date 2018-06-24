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

Tris::Tris(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order)
{
  etype = TRI;

  this->init(geo,input,elesObjID,startEle,endEle,order);

  if (input->error_freq > 0) this->nQpts = 45; // Note: Fixing quadrature points to Williams-Shunn 45 point rule

  /* Generic triangular geometry */
  nDims = 2;
  nFaces = 3;
  nNodes = geo->nNodesPerEleBT[TRI];

  nSpts = (this->order + 1) * (this->order + 2) / 2;
  nSpts1D = this->order + 1;

  nFptsPerFace = this->order + 1;

  nFpts_face = {nFptsPerFace, nFptsPerFace, nFptsPerFace};
  nFpts = nFptsPerFace * nFaces;
  nPpts = nSpts;
}

void Tris::set_locs()
{
  /* Allocate memory for point location structures */
  loc_qpts.assign({nQpts,nDims}); 

  std::vector<double> loc_fpts_1D; 
  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
  {
    loc_spts_1D = Gauss_Legendre_pts(order+1); // loc_spts_1D used when generating filter matrices only
    loc_fpts_1D = Gauss_Legendre_pts(order+1);
  }
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
    for (unsigned int j = 0; j < nFptsPerFace; j++)
    {
      switch(i)
      {
        case 0: /* Bottom edge */
          loc_fpts(fpt,0) = loc_fpts_1D[j];
          loc_fpts(fpt,1) = -1.0; break;

        case 1: /* Hypotenuse */
          loc_fpts(fpt,0) = loc_fpts_1D[nFptsPerFace-j-1]; 
          loc_fpts(fpt,1) = loc_fpts_1D[j]; break;

        case 2: /* Left edge */
          loc_fpts(fpt,0) = -1.0;
          loc_fpts(fpt,1) = loc_fpts_1D[nFptsPerFace-j-1]; break;
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
    for (unsigned int j = 0; j < nPpts1D - i; j++)
    {
      loc_ppts(ppt,0) = loc_ppts_1D[j];
      loc_ppts(ppt,1) = loc_ppts_1D[i];
      ppt++;
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
  tdA.assign({nFpts});

  /* Setup parent-space (transformed) normals at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    switch(fpt/nFptsPerFace)
    {
      case 0: /* Bottom edge */
        tnorm(fpt,0) = 0.0;
        tnorm(fpt,1) = -1.0;
        tdA(fpt) = 1.0;
        break;

      case 1: /* Hypotenuse */
        tnorm(fpt,0) = std::sqrt(2)/2;
        tnorm(fpt,1) = std::sqrt(2)/2;
        tdA(fpt) = std::sqrt(2);
        break;

      case 2: /* Left edge */
        tnorm(fpt,0) = -1.0;
        tnorm(fpt,1) = 0.0;
        tdA(fpt) = 1.0;
        break;

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

  inv_vand.assign({nSpts, nSpts}); 
  vand.inverse(inv_vand);

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

  inv_vandRT.assign({2*nSpts + nFpts, 2*nSpts * nFpts}); 
  vandRT.inverse(inv_vandRT);
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
    for (unsigned int j = 0; j < nRpts1D - i; j++)
    {
      loc_rpts(rpt,0) = loc_rpts_1D[j];
      loc_rpts(rpt,1) = loc_rpts_1D[i];
      rpt++;
    }
  }

  /* Setup extrapolation operator from restart points */
  oppRestart.assign({nSpts, nRpts});

  /* Set vandermonde and inverse for orthonormal Dubiner basis at restart points */
  mdvector<double> vand_r({nRpts, nRpts});

  for (unsigned int i = 0; i < nRpts; i++)
    for (unsigned int j = 0; j < nRpts; j++)
      vand_r(i,j) = Dubiner2D(order_restart, loc_rpts(i, 0), loc_rpts(i, 1), j); 

  mdvector<double> inv_vand_r({nRpts, nRpts});
  vand_r.inverse(inv_vand_r);

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

void Tris::calc_nodal_basis(double *loc, double* basis)
{
  // store values locally to avoid re-computing
  if (lag_i.size() < nSpts)
    lag_i.resize(nSpts);

  for (unsigned int i = 0; i < nSpts; i++)
    lag_i[i] = Dubiner2D(order, loc[0], loc[1], i);

  cblas_dgemv(CblasRowMajor,CblasNoTrans,nSpts,nSpts,1.0,inv_vand.data(),nSpts,
              lag_i.data(),1,0.0,basis,1);
}

double Tris::calc_d_nodal_basis_spts(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;

  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * dDubiner2D(order, loc[0], loc[1], dim, i);
  }

  return val;

}

double Tris::calc_d_nodal_basis_fr(unsigned int spt,
              const std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    val += inv_vand(i, spt) * dDubiner2D(order, loc[0], loc[1], dim, i);
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


mdvector<double> Tris::get_face_nodes(unsigned int face, unsigned int P)
{
  auto vpts = Gauss_Legendre_pts(P+1);  // Given polynomial order; need N

  mdvector<double> pts({P+1});

  for (int i = 0; i < P+1; i++)
    pts(i) = vpts[i];

  return pts;
}

mdvector<double> Tris::get_face_weights(unsigned int face, unsigned int P)
{
  auto vwts = Gauss_Legendre_weights(P+1);  // Given polynomial order; need N

  mdvector<double> wts({P+1});

  for (int i = 0; i < P+1; i++)
    wts(i) = vwts[i];

  return wts;
}

void Tris::project_face_point(int face, const double* loc, double* ploc)
{
  switch(face)
  {
    case 0: /* Bottom edge */
      ploc[0] = loc[0];
      ploc[1] = -1.0;
      break;

    case 1: /* Hypotenuse */
      ploc[0] = -loc[0];
      ploc[1] = loc[0];
      break;

    case 2: /* Left edge */
      ploc[0] = -1.0;
      ploc[1] = -loc[0];
      break;
  }
}

double Tris::calc_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc)
{
  return Lagrange(loc_spts_1D, loc[0], pt); /// CHECK
}

double Tris::calc_orthonormal_basis(unsigned int mode, const double *loc)
{
  return Dubiner2D(order, loc[0], loc[1], mode);
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

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, nSpts, nSpts,
        1.0, eye.data(), nSpts, inv_vand.data(), nSpts, 0.0, temp.data(), nSpts);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_pro, nSpts, nSpts_pro,
        1.0, vand_pro.data(), nSpts_pro, temp.data(), nSpts, 0.0, oppPro.data(), nSpts);

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

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_res, nSpts, nSpts,
        1.0, eye.data(), nSpts, inv_vand.data(), nSpts, 0.0, temp.data(), nSpts);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts_res, nSpts, nSpts_res,
        1.0, vand_res.data(), nSpts_res, temp.data(), nSpts, 0.0, oppRes.data(), nSpts);

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

void Tris::calc_shape(mdvector<double> &shape_val, const double* loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2};
  gmsh_nodes[2] =  {0, 3, 1, 5, 4, 2};

  unsigned int shape_order = tri_nodes_to_order(nNodes);

  if (shape_order > 2)
    ThrowException("Triangle with supplied shape_order unsupported!");

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D - i; j++)
    {
      loc_pts(pt,0) = loc_pts_1D[j];
      loc_pts(pt,1) = loc_pts_1D[i];
      pt++;
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  mdvector<double> inv_vand_s({nNodes, nNodes});
  vand_s.inverse(inv_vand_s);

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
}

void Tris::calc_d_shape(mdvector<double> &dshape_val, const double* loc)
{
  std::vector<std::vector<unsigned int>> gmsh_nodes(3);
  gmsh_nodes[1] =  {0, 1, 2};
  gmsh_nodes[2] =  {0, 3, 1, 5, 4, 2};

  unsigned int shape_order = tri_nodes_to_order(nNodes);

  if (shape_order > 2)
    ThrowException("Triangle with supplied shape_order unsupported!");

  double xi = loc[0];
  double eta = loc[1];

  /* Setup shape node locations */
  auto loc_pts_1D = Shape_pts(shape_order); unsigned int nPts1D = loc_pts_1D.size();
  mdvector<double> loc_pts({nNodes, nDims});

  unsigned int pt = 0;
  for (unsigned int i = 0; i < nPts1D; i++)
  {
    for (unsigned int j = 0; j < nPts1D - i; j++)
    {
      loc_pts(pt,0) = loc_pts_1D[j];
      loc_pts(pt,1) = loc_pts_1D[i];
      pt++;
    }
  }

  /* Set vandermonde and inverse for orthonormal Dubiner basis at shape points*/
  mdvector<double> vand_s({nNodes, nNodes});

  for (unsigned int i = 0; i < nNodes; i++)
    for (unsigned int j = 0; j < nNodes; j++)
      vand_s(i,j) = Dubiner2D(shape_order, loc_pts(i, 0), loc_pts(i, 1), j); 

  mdvector<double> inv_vand_s({nNodes, nNodes});
  vand_s.inverse(inv_vand_s);

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
}

void Tris::modify_sensor()
{
  /* Obtain locations of "collapsed" quad solution points */
  unsigned int nNodesQuad = 4;
  mdvector<double> nodes({nDims, nNodesQuad}); 
  nodes(0, 0) = -1.0; nodes(1, 0) = -1.0; 
  nodes(0, 1) = 1.0; nodes(1, 1) = -1.0; 
  nodes(0, 2) = -1.0; nodes(1, 2) = 1.0; 
  nodes(0, 3) = -1.0; nodes(1, 3) = 1.0; 

  unsigned int nSpts2D = nSpts1D * nSpts1D;
  mdvector<double> loc_spts_quad({nSpts2D, nDims}, 0);

  for (unsigned int spt = 0; spt < nSpts2D; spt++)
  {
    for (unsigned int nd = 0; nd < nNodesQuad; nd++)
    {
      int i = nd % 2; int j = nd / 2;
      for (unsigned int dim = 0; dim < nDims; dim++)
      {
        loc_spts_quad(spt, dim) += nodes(dim, nd) * Lagrange({-1, 1}, loc_spts_1D[spt % nSpts1D], i) * 
                                                    Lagrange({-1, 1}, loc_spts_1D[spt / nSpts1D], j);
      }
    }
  }

  /* Setup spt to collapsed spt extrapolation operator (oppEc) */
  std::vector<double> loc(nDims, 0.0);
  mdvector<double> oppEc({nSpts2D, nSpts});
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int spt_q = 0; spt_q < nSpts2D; spt_q++)
    {
      for (unsigned int dim = 0; dim < nDims; dim++)
        loc[dim] = loc_spts_quad(spt_q , dim);

      oppEc(spt_q, spt) = calc_nodal_basis(spt, loc);
    }
  }

  /* Multiply oppS by oppEc to get modified operator */
  auto temp = oppS;
  oppS.assign({nSpts2D * nDims, nSpts});

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nSpts2D * nDims, nSpts, nSpts2D,
      1.0, temp.data(), nSpts2D, oppEc.data(), nSpts, 0.0, oppS.data(), nSpts);

} 

double Tris::rst_max_lim(int dim, double* rst)
{
  switch (dim)
  {
    case 0:
      return std::min(rst[0], 1.0);
    case 1:
      return std::min(rst[1], -rst[0]);
  }
}

double Tris::rst_min_lim(int dim, double* rst)
{
  return std::max(rst[dim], -1.0);
}
