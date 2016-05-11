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

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

//Quads::Quads(GeoStruct *geo, const InputStruct *input, int order)
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
  //nNodes = (shape_order+1)*(shape_order+1); // Lagrange Elements
  if (shape_order == 1)
    nNodes = 8;
  else if (shape_order == 2)
    nNodes = 20;
  
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


void Hexas::set_transforms(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nDims, nDims, nSpts, nEles});
  inv_jaco_spts.assign({nDims, nDims, nSpts, nEles});
  jaco_ppts.assign({nDims, nDims, nPpts, nEles});
  jaco_qpts.assign({nDims, nDims, nQpts, nEles});
  jaco_det_spts.assign({nSpts, nEles});
  jaco_det_qpts.assign({nQpts, nEles});
  vol.assign({nEles});


  /* Set jacobian matrix and determinant at solution points */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
      {
        for (unsigned int dimX = 0; dimX < nDims; dimX++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(node,ele);
            jaco_spts(dimX, dimXi, spt, ele) += geo->coord_nodes(gnd,dimX) * dshape_spts(node, spt, dimXi); 
          }
        }
      }
      
      double xr = jaco_spts(0, 0, spt, ele); double xs = jaco_spts(0, 1, spt, ele); double xt = jaco_spts(0, 2, spt, ele);
      double yr = jaco_spts(1, 0, spt, ele); double ys = jaco_spts(1, 1, spt, ele); double yt = jaco_spts(1, 2, spt, ele);
      double zr = jaco_spts(2, 0, spt, ele); double zs = jaco_spts(2, 1, spt, ele); double zt = jaco_spts(2, 2, spt, ele);

      inv_jaco_spts(0, 0, spt, ele) = ys * zt - yt * zs; inv_jaco_spts(0, 1, spt, ele) = xt * zs - xs * zt; inv_jaco_spts(0, 2, spt, ele) = xs * yt - xt * ys;
      inv_jaco_spts(1, 0, spt, ele) = yt * zr - yr * zt; inv_jaco_spts(1, 1, spt, ele) = xr * zt - xt * zr; inv_jaco_spts(1, 2, spt, ele) = xt * yr - xr * yt;
      inv_jaco_spts(2, 0, spt, ele) = yr * zs - ys * zr; inv_jaco_spts(2, 1, spt, ele) = xs * zr - xr * zs; inv_jaco_spts(2, 2, spt, ele) = xr * ys - xs * yr;

      jaco_det_spts(spt,ele) = xr * (ys * zt - yt * zs) - xs * (yr * zt - yt * zr) + 
        xt * (yr * zs - ys * zr);

      if (jaco_det_spts(spt,ele) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " spt:" + std::to_string(spt));

    }

    /* Compute element volume */
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get quadrature weight */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);
      unsigned int k = idx_spts(spt,2);

      double weight = weights_spts(i) * weights_spts(j) * weights_spts(k);

      vol(ele) += weight * jaco_det_spts(spt, ele);
    }

  }

  /* Set jacobian matrix at face flux points (do not need the determinant) */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfpt(fpt,ele);
      for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
      {
        for (unsigned int dimX = 0; dimX < nDims; dimX++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(node,ele);

            /* Skip fpts on ghost edges */
            if (gfpt == -1)
              continue;

            unsigned int slot = geo->fpt2gfpt_slot(fpt,ele);

            faces->jaco(gfpt, dimX, dimXi, slot) += geo->coord_nodes(gnd,dimX) * dshape_fpts(node, fpt, dimXi);
          }
        }
      }
    }
  }


  /* Set jacobian matrix at plot points (do not need the determinant) */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
      {
        for (unsigned int dimX = 0; dimX < nDims; dimX++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(node,ele);
            jaco_ppts(dimX,dimXi,ppt,ele) += geo->coord_nodes(gnd,dimX) * dshape_ppts(node, ppt, dimXi); 
          }
        }
      }
    }
  }
  /* Set jacobian matrix and determinant at quadrature points */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
      {
        for (unsigned int dimX = 0; dimX < nDims; dimX++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(node, ele);
            jaco_qpts(dimX,dimXi,qpt,ele) += geo->coord_nodes(gnd,dimX) * dshape_qpts(node,qpt,dimXi); 
          }
        }
      }

      double xr = jaco_qpts(0, 0, qpt, ele); double xs = jaco_qpts(0, 1, qpt, ele); double xt = jaco_qpts(0, 2, qpt, ele);
      double yr = jaco_qpts(1, 0, qpt, ele); double ys = jaco_qpts(1, 1, qpt, ele); double yt = jaco_qpts(1, 2, qpt, ele);
      double zr = jaco_qpts(2, 0, qpt, ele); double zs = jaco_qpts(2, 1, qpt, ele); double zt = jaco_qpts(2, 2, qpt, ele);

      jaco_det_qpts(qpt,ele) = xr * (ys * zt - yt * zs) - xs * (yr * zt - yt * zr) + 
        xt * (yr * zs - ys * zr);

      if (jaco_det_qpts(qpt,ele) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " qpt:" + std::to_string(qpt));

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

  /* Use transform to obtain physical normals at face flux points */
  mdvector<double> inv_jaco({nDims, nDims});
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfpt(fpt,ele);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1) 
        continue;

      unsigned int slot = geo->fpt2gfpt_slot(fpt,ele);

      double xr = faces->jaco(gfpt, 0, 0, slot); double xs = faces->jaco(gfpt, 0, 1, slot); double xt = faces->jaco(gfpt, 0, 2, slot);
      double yr = faces->jaco(gfpt, 1, 0, slot); double ys = faces->jaco(gfpt, 1, 1, slot); double yt = faces->jaco(gfpt, 1, 2, slot);
      double zr = faces->jaco(gfpt, 2, 0, slot); double zs = faces->jaco(gfpt, 2, 1, slot); double zt = faces->jaco(gfpt, 2, 2, slot);

      inv_jaco(0,0) = ys * zt - yt * zs; inv_jaco(0,1) = xt * zs - xs * zt; inv_jaco(0,2) = xs * yt - xt * ys;
      inv_jaco(1,0) = yt * zr - yr * zt; inv_jaco(1,1) = xr * zt - xt * zr; inv_jaco(1,2) = xt * yr - xr * yt;
      inv_jaco(2,0) = yr * zs - ys * zr; inv_jaco(2,1) = xs * zr - xr * zs; inv_jaco(2,2) = xr * ys - xs * yr;
      
      for (unsigned int dim1 = 0; dim1 < nDims; dim1++)
      {
        for (unsigned int dim2 = 0; dim2 < nDims; dim2++)
        {
          faces->norm(gfpt, dim1, slot) += inv_jaco(dim2, dim1) * tnorm(fpt, dim2); 
        }
      }

      if (slot == 0)
      {
        for (unsigned int dim = 0; dim < nDims; dim++)
          faces->dA(gfpt) += faces->norm(gfpt, dim, slot) * faces->norm(gfpt, dim, slot);

        faces->dA(gfpt) = std::sqrt(faces->dA(gfpt));
      }
                        

      for (unsigned int dim = 0; dim < nDims; dim++)
        faces->norm(gfpt, dim, slot) /= faces->dA(gfpt);

    }
  }

}

double Hexas::calc_nodal_basis(unsigned int spt, std::vector<double> &loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);
  unsigned int k = idx_spts(spt,2);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]) * Lagrange(loc_spts_1D, k, loc[2]);

  return val;
}

double Hexas::calc_d_nodal_basis_spts(unsigned int spt, std::vector<double> &loc, unsigned int dim)
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

double Hexas::calc_d_nodal_basis_spts_FR(unsigned int spt, std::vector<double> &loc, unsigned int dim)
{
  /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
   * boundary points for DFR) */
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

double Hexas::calc_d_nodal_basis_fpts(unsigned int fpt, std::vector<double> &loc, unsigned int dim)
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

void Hexas::setup_PMG()
{
  unsigned int nSpts_pro_1D = order+2;
  unsigned int nSpts_res_1D = order;
  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D * nSpts_pro_1D;
  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D * nSpts_res_1D;

  std::vector<double> loc(nDims, 0.0);

  if (order != input->order)
  {
    oppPro.assign({nSpts_pro, nSpts});

    auto loc_spts_pro_1D = Gauss_Legendre_pts(order+2); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int pspt = 0; pspt < nSpts_pro; pspt++)
      {
        unsigned int i = pspt % nSpts_pro_1D;
        unsigned int j = (pspt / nSpts_pro_1D) % nSpts_pro_1D;
        unsigned int k = pspt / (nSpts_pro_1D * nSpts_pro_1D);
        loc[0] = loc_spts_pro_1D[i];
        loc[1] = loc_spts_pro_1D[j];
        loc[2] = loc_spts_pro_1D[k]; //TODO: Correct?

        oppPro(pspt, spt) = calc_nodal_basis(spt, loc);
      }
    }
  }

  if (order != 0)
  {
    oppRes.assign({nSpts_res, nSpts});

    auto loc_spts_res_1D = Gauss_Legendre_pts(order); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int rspt = 0; rspt < nSpts_res; rspt++)
      {
        unsigned int i = rspt % nSpts_res_1D;
        unsigned int j = (rspt / nSpts_res_1D) % nSpts_res_1D;
        unsigned int k = rspt / (nSpts_res_1D * nSpts_res_1D);
        loc[0] = loc_spts_res_1D[i];
        loc[1] = loc_spts_res_1D[j];
        loc[2] = loc_spts_res_1D[k]; //TODO: Correct?

        oppRes(rspt, spt) = calc_nodal_basis(spt, loc);
      }
    }
  }
}

void Hexas::transform_dU()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double dUtemp0 = dU_spts(spt, ele, n, 0);
        double dUtemp1 = dU_spts(spt, ele, n, 1);

        dU_spts(spt, ele, n, 0) = dU_spts(spt, ele, n, 0) * inv_jaco_spts(0, 0, spt, ele) + 
                                  dU_spts(spt, ele, n, 1) * inv_jaco_spts(1, 0, spt, ele) +  
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(2, 0, spt, ele);  

        dU_spts(spt, ele, n, 1) = dUtemp0 * inv_jaco_spts(0, 1, spt, ele) + 
                                  dU_spts(spt, ele, n, 1) * inv_jaco_spts(1, 1, spt, ele) +  
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(2, 1, spt, ele);  
                                  
        dU_spts(spt, ele, n, 2) = dUtemp0 * inv_jaco_spts(0, 2, spt, ele) + 
                                  dUtemp1 * inv_jaco_spts(1, 2, spt, ele) +  
                                  dU_spts(spt, ele, n, 2) * inv_jaco_spts(2, 2, spt, ele);  

        dU_spts(spt, ele, n, 0) /= jaco_det_spts(spt, ele);
        dU_spts(spt, ele, n, 1) /= jaco_det_spts(spt, ele);
        dU_spts(spt, ele, n, 2) /= jaco_det_spts(spt, ele);
      }
    }
  }
#endif

#ifdef _GPU
  transform_dU_hexa_wrapper(dU_spts_d, inv_jaco_spts_d, jaco_det_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);
  //dU_spts = dU_spts_d;
  check_error();
#endif

}

void Hexas::transform_flux()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double Ftemp0 = F_spts(spt, ele, n, 0);
        double Ftemp1 = F_spts(spt, ele, n, 1);

        F_spts(spt, ele, n, 0) = F_spts(spt, ele, n, 0) * inv_jaco_spts(0, 0, spt, ele) + 
                                  F_spts(spt, ele, n, 1) * inv_jaco_spts(0, 1, spt, ele) +  
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(0, 2, spt, ele); 

        F_spts(spt, ele, n, 1) = Ftemp0 * inv_jaco_spts(1, 0, spt, ele) + 
                                  F_spts(spt, ele, n, 1) * inv_jaco_spts(1, 1, spt, ele) +  
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(1, 2, spt, ele);  
                                  
        F_spts(spt, ele, n, 2) = Ftemp0 * inv_jaco_spts(2, 0, spt, ele) + 
                                  Ftemp1 * inv_jaco_spts(2, 1, spt, ele) +  
                                  F_spts(spt, ele, n, 2) * inv_jaco_spts(2, 2, spt, ele); 

      }

    }
  }

#endif

#ifdef _GPU
  //F_spts_d = F_spts;
  transform_flux_hexa_wrapper(F_spts_d, inv_jaco_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);

  check_error();

  //F_spts = F_spts_d;
#endif
}

mdvector<double> Hexas::calc_shape(unsigned int shape_order, 
                         std::vector<double> &loc)
{
  mdvector<double> shape_val({nNodes}, 0.0);
  double xi = loc[0]; 
  double eta = loc[1];
  double mu = loc[2];

  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  /* Trilinear hexahedral/8-node Serendipity */
  if (shape_order == 1)
  {
    shape_val(0) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    shape_val(1) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    shape_val(2) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    shape_val(3) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    shape_val(4) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    shape_val(5) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    shape_val(6) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);
    shape_val(7) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);
  }
  /* 20-node Seredipity */
  else if (shape_order == 2 and input->serendipity)
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
    ThrowException("Element shape type for hexas given not supported!");
  }

  return shape_val;
}

mdvector<double> Hexas::calc_d_shape(unsigned int shape_order,
                          std::vector<double> &loc)
{
  mdvector<double> dshape_val({nNodes, nDims}, 0);
  double xi = loc[0];
  double eta = loc[1];
  double mu = loc[2];

  unsigned int i = 0;
  unsigned int j = 0;
  unsigned int k = 0;

  /* Bilinear hexahedral/8-node Serendipity */
  if (shape_order == 1)
  {
    dshape_val(0, 0) = Lagrange_d1({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(1, 0) = Lagrange_d1({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(2, 0) = Lagrange_d1({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(3, 0) = Lagrange_d1({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(4, 0) = Lagrange_d1({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(5, 0) = Lagrange_d1({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(6, 0) = Lagrange_d1({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(7, 0) = Lagrange_d1({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);

    dshape_val(0, 1) = Lagrange({-1.,1.}, 0, xi) * Lagrange_d1({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(1, 1) = Lagrange({-1.,1.}, 1, xi) * Lagrange_d1({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(2, 1) = Lagrange({-1.,1.}, 1, xi) * Lagrange_d1({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(3, 1) = Lagrange({-1.,1.}, 0, xi) * Lagrange_d1({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 0, mu);
    dshape_val(4, 1) = Lagrange({-1.,1.}, 0, xi) * Lagrange_d1({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(5, 1) = Lagrange({-1.,1.}, 1, xi) * Lagrange_d1({-1.,1.}, 0, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(6, 1) = Lagrange({-1.,1.}, 1, xi) * Lagrange_d1({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);
    dshape_val(7, 1) = Lagrange({-1.,1.}, 0, xi) * Lagrange_d1({-1.,1.}, 1, eta) * Lagrange({-1.,1.}, 1, mu);

    dshape_val(0, 2) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange_d1({-1.,1.}, 0, mu);
    dshape_val(1, 2) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange_d1({-1.,1.}, 0, mu);
    dshape_val(2, 2) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange_d1({-1.,1.}, 0, mu);
    dshape_val(3, 2) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange_d1({-1.,1.}, 0, mu);
    dshape_val(4, 2) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange_d1({-1.,1.}, 1, mu);
    dshape_val(5, 2) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 0, eta) * Lagrange_d1({-1.,1.}, 1, mu);
    dshape_val(6, 2) = Lagrange({-1.,1.}, 1, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange_d1({-1.,1.}, 1, mu);
    dshape_val(7, 2) = Lagrange({-1.,1.}, 0, xi) * Lagrange({-1.,1.}, 1, eta) * Lagrange_d1({-1.,1.}, 1, mu);
  }
  /* 20-node Serendipity */
  else if (shape_order == 2)
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


  return dshape_val;

}
