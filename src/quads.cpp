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
#include "shape.hpp"

#ifdef _GPU
#include "elements_kernels.h"
#include "solver_kernels.h"
#endif

//Quads::Quads(GeoStruct *geo, const InputStruct *input, int order)
Quads::Quads(GeoStruct *geo, InputStruct *input, int order)
{
  this->geo = geo;
  this->input = input;  
  this->shape_order = geo->shape_order;  
  this->nEles = geo->nEles;  
  this->nQpts = input->nQpts1D * input->nQpts1D;

  /* Generic quadrilateral geometry */
  nDims = 2;
  nFaces = 4;
  //nNodes = (shape_order+1)*(shape_order+1); // Lagrange Elements
  nNodes = 4*(shape_order); // Serendipity Elements
  
  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order+1)*(input->order+1);
    nSpts1D = input->order+1;
    this->order = input->order;
  }
  else
  {
    nSpts = (order+1)*(order+1);
    nSpts1D = order+1;
    this->order = order;
  }

  nFpts = nSpts1D * nFaces;
  nPpts = (nSpts1D+2)*(nSpts1D+2);
  
  if (input->equation == AdvDiff)
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
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  // NOTE: Currently assuming solution point locations always at Legendre.
  // Will need extrapolation operation in 1D otherwise
  weights_spts = Gauss_Legendre_weights(nSpts1D); 

  loc_DFR_1D = loc_spts_1D;
  loc_DFR_1D.insert(loc_DFR_1D.begin(), -1.0);
  loc_DFR_1D.insert(loc_DFR_1D.end(), 1.0);

  /* Setup solution point locations */
  unsigned int spt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      loc_spts(spt,0) = loc_spts_1D[j];
      loc_spts(spt,1) = loc_spts_1D[i];
      idx_spts(spt,0) = j;
      idx_spts(spt,1) = i;
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
  
  /* Setup plot point locations */
  auto loc_ppts_1D = loc_spts_1D;
  loc_ppts_1D.insert(loc_ppts_1D.begin(), -1.0);
  loc_ppts_1D.insert(loc_ppts_1D.end(), 1.0);

  unsigned int ppt = 0;
  for (unsigned int i = 0; i < nSpts1D+2; i++)
  {
    for (unsigned int j = 0; j < nSpts1D+2; j++)
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
  weights_qpts = Gauss_Legendre_weights(input->nQpts1D);

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
      qpt++;
    }
  }

}


void Quads::set_transforms(std::shared_ptr<Faces> faces)
{
  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nDims, nDims, nSpts, nEles});
  jaco_ppts.assign({nDims, nDims, nPpts, nEles});
  jaco_qpts.assign({nDims, nDims, nQpts, nEles});
  jaco_det_spts.assign({nSpts, nEles});
  jaco_det_qpts.assign({nQpts, nEles});

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

      jaco_det_spts(spt,ele) = jaco_spts(0,0,spt,ele) * jaco_spts(1,1,spt,ele) -
                               jaco_spts(0,1,spt,ele) * jaco_spts(1,0,spt,ele); 


      if (jaco_det_spts(spt,ele) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " spt:" + std::to_string(spt));

    }
  }

  /* Set jacobian matrix at face flux points (do not need the determinant) */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
      {
        for (unsigned int dimX = 0; dimX < nDims; dimX++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(node,ele);
            int gfpt = geo->fpt2gfpt(fpt,ele);

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

  /* Set jacobian matrix and determinant at plot points (do not need the determinant) */
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

      jaco_det_qpts(qpt,ele) = jaco_qpts(0,0,qpt,ele) * jaco_qpts(1,1,qpt,ele) -
                               jaco_qpts(0,1,qpt,ele) * jaco_qpts(1,0,qpt,ele); 


      if (jaco_det_qpts(qpt,ele) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " qpt:" + std::to_string(qpt));

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

  /* Use transform to obtain physical normals at face flux points */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = geo->fpt2gfpt(fpt,ele);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1) 
        continue;

      unsigned int slot = geo->fpt2gfpt_slot(fpt,ele);

      faces->norm(gfpt, 0, slot) = faces->jaco(gfpt, 1, 1, slot) * tnorm(fpt, 0) - 
                                 faces->jaco(gfpt, 1, 0, slot) * tnorm(fpt, 1);
      faces->norm(gfpt, 1, slot) = -faces->jaco(gfpt, 0, 1, slot) * tnorm(fpt, 0) + 
                                 faces->jaco(gfpt, 0, 0, slot) * tnorm(fpt, 1);

      faces->dA(gfpt) = std::sqrt(faces->norm(gfpt, 0, slot)*faces->norm(gfpt, 0, slot) + 
                        faces->norm(gfpt, 1, slot)*faces->norm(gfpt, 1, slot));

      faces->norm(gfpt, 0, slot) /= faces->dA(gfpt);
      faces->norm(gfpt, 1, slot) /= faces->dA(gfpt);

      unsigned int face_idx = fpt/nSpts1D;

      if(face_idx == 0 || face_idx == 3)
        faces->outnorm(gfpt, slot) = -1; 
      else 
        faces->outnorm(gfpt, slot) = 1; 

    }
  }

}

double Quads::calc_nodal_basis(unsigned int spt, std::vector<double> &loc)
{
  /* Get indices for Lagrange polynomial evaluation */
  unsigned int i = idx_spts(spt,0);
  unsigned int j = idx_spts(spt,1);

  double val = Lagrange(loc_spts_1D, i, loc[0]) * Lagrange(loc_spts_1D, j, loc[1]);

  return val;
}

double Quads::calc_d_nodal_basis_spts(unsigned int spt, std::vector<double> &loc, unsigned int dim)
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

double Quads::calc_d_nodal_basis_fpts(unsigned int fpt, std::vector<double> &loc, unsigned int dim)
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

void Quads::setup_PMG()
{
  /* Allocate memory for operators */
  unsigned int nSpts_pro_1D = order+2;
  unsigned int nSpts_res_1D = order;
  unsigned int nSpts_pro = nSpts_pro_1D * nSpts_pro_1D;
  unsigned int nSpts_res = nSpts_res_1D * nSpts_res_1D;

  std::vector<double> loc(nDims, 0.0);

  if (order != input->order)
  {
    /* Setup prolongation operator */
    oppPro.assign({nSpts_pro, nSpts});

    auto loc_spts_pro_1D = Gauss_Legendre_pts(order+2); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int pspt = 0; pspt < nSpts_pro; pspt++)
      {
        loc[0] = loc_spts_pro_1D[pspt%nSpts_pro_1D];
        loc[1] = loc_spts_pro_1D[pspt/nSpts_pro_1D];

        oppPro(pspt, spt) = calc_nodal_basis(spt, loc);
      }
    }
  }

  if (order != 0)
  {
    /* Setup restriction operator */
    oppRes.assign({nSpts_res, nSpts});

    auto loc_spts_res_1D = Gauss_Legendre_pts(order); 

    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int rspt = 0; rspt < nSpts_res; rspt++)
      {
        loc[0] = loc_spts_res_1D[rspt%nSpts_res_1D];
        loc[1] = loc_spts_res_1D[rspt/nSpts_res_1D];

        oppRes(rspt, spt) = calc_nodal_basis(spt, loc);
      }
    }
  }
}

void Quads::transform_dU()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double dUtemp = dU_spts(spt, ele, n, 0);

        dU_spts(spt, ele, n, 0) = dU_spts(spt, ele, n, 0) * jaco_spts(1, 1, spt, ele) - 
                                  dU_spts(spt, ele, n, 1) * jaco_spts(1, 0, spt, ele); 

        dU_spts(spt, ele, n, 1) = dU_spts(spt, ele, n, 1) * jaco_spts(0, 0, spt, ele) -
                                  dUtemp * jaco_spts(0, 1, spt, ele);

        dU_spts(spt, ele, n, 0) /= jaco_det_spts(spt, ele);
        dU_spts(spt, ele, n, 1) /= jaco_det_spts(spt, ele);
      }
    }
  }
#endif

#ifdef _GPU
  transform_dU_quad_wrapper(dU_spts_d, jaco_spts_d, jaco_det_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);
  //dU_spts = dU_spts_d;
  check_error();
#endif

}

void Quads::transform_flux()
{
#ifdef _CPU
#pragma omp parallel for collapse(3)
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int ele = 0; ele < nEles; ele++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        double Ftemp = F_spts(spt, ele, n, 0);

        F_spts(spt, ele, n, 0) = F_spts(spt, ele, n, 0) * jaco_spts(1, 1, spt, ele) -
                                 F_spts(spt, ele, n, 1) * jaco_spts(0, 1, spt, ele);
        F_spts(spt, ele, n, 1) = F_spts(spt, ele, n, 1) * jaco_spts(0, 0, spt, ele) -
                                 Ftemp * jaco_spts(1, 0, spt, ele);
      }
    }
  }
#endif

#ifdef _GPU
  //F_spts_d = F_spts;
  transform_flux_quad_wrapper(F_spts_d, jaco_spts_d, nSpts, nEles, nVars,
      nDims, input->equation);

  check_error();

  //F_spts = F_spts_d;
#endif
}

double Quads::calc_shape(unsigned int shape_order, unsigned int idx, 
                         std::vector<double> &loc)
{
  double val = 0.0;
  double xi = loc[0]; 
  double eta = loc[1];

  /* Bilinear quadrilateral/4-node Serendipity */
  if (shape_order == 1)
  {
    unsigned int i = 0;
    unsigned int j = 0;

    switch(idx)
    {
      case 0:
        i = 0; j = 0; break;
      case 1:
        i = 1; j = 0; break;
      case 2:
        i = 1; j = 1; break;
      case 3:
        i = 0; j = 1; break;
    }

    val = Lagrange({-1.,1.}, i, xi) * Lagrange({-1.,1.}, j, eta);
  }

  /* 8-node Serendipity Element */
  if (shape_order == 2)
  {
    switch(idx)
    {
      case 0:
        val = -0.25*(1.-xi)*(1.-eta)*(1.+eta+xi); break;
      case 4:
        val = 0.5*(1.-xi)*(1.+xi)*(1.-eta); break;
      case 1:
        val = -0.25*(1.+xi)*(1.-eta)*(1.+eta-xi); break;
      case 5:
        val = 0.5*(1.+xi)*(1.+eta)*(1.-eta); break;
      case 2:
        val = -0.25*(1.+xi)*(1.+eta)*(1.-eta-xi); break;
      case 6:
        val = 0.5*(1.-xi)*(1.+xi)*(1.+eta); break;
      case 3:
        val = -0.25*(1.-xi)*(1.+eta)*(1.-eta+xi); break;
      case 7:
        val = 0.5*(1.-xi)*(1.+eta)*(1.-eta); break;
    }
  }

  return val;
}

double Quads::calc_d_shape(unsigned int shape_order, unsigned int idx,
                          std::vector<double> &loc, unsigned int dim)
{
  double val = 0.0;
  double xi = loc[0];
  double eta = loc[1];

  /* Bilinear quadrilateral/4-node Serendipity */
  if (shape_order == 1)
  {
    unsigned int i = 0;
    unsigned int j = 0;

    switch(idx)
    {
      case 0:
        i = 0; j = 0; break;
      case 1:
        i = 1; j = 0; break;
      case 2:
        i = 1; j = 1; break;
      case 3:
        i = 0; j = 1; break;
    }

    if (dim == 0)
      val = Lagrange_d1({-1,1}, i, xi) * Lagrange({-1,1}, j, eta);
    else
      val = Lagrange({-1,1}, i, xi) * Lagrange_d1({-1,1}, j, eta);
  }

  /* 8-node Serendipity Element */
  else if (shape_order == 2)
  {
    if (dim == 0)
    {
      switch(idx)
      {
        case 0:
          val = -0.25*(-1.+eta)*(2.*xi+eta); break;
        case 4:
          val = xi*(-1.+eta); break;
        case 1:
          val = 0.25*(-1.+eta)*(eta - 2.*xi); break;
        case 5:
          val = -0.5*(1+eta)*(-1.+eta); break;
        case 2:
          val = 0.25*(1.+eta)*(2.*xi+eta); break;
        case 6:
          val = -xi*(1.+eta); break;
        case 3:
          val = -0.25*(1.+eta)*(eta-2.*xi); break;
        case 7:
          val = 0.5*(1+eta)*(-1.+eta); break;
      }
    }

    else if (dim == 1)
    {
      switch(idx)
      {
        case 0:
          val = -0.25*(-1.+xi)*(2.*eta+xi); break;
        case 4:
          val = 0.5*(1.+xi)*(-1.+xi); break;
        case 1:
          val = 0.25*(1.+xi)*(2.*eta - xi); break;
        case 5:
          val = -eta*(1.+xi); break;
        case 2:
          val = 0.25*(1.+xi)*(2.*eta+xi); break;
        case 6:
          val = -0.5*(1.+xi)*(-1.+xi); break;
        case 3:
          val = -0.25*(-1.+xi)*(2.*eta-xi); break;
        case 7:
          val = eta*(-1.+xi); break;
      }
    }

  }

  return val;

}
  /*
  std::cout << "tflux" << std::endl;
  for (unsigned int i = 0; i < nSpts; i++)
  {
    for (unsigned int j = 0; j < nEles; j++)
    {
      std::cout << F_spts(0,0,i,j) << " ";
    }
    std::cout << std::endl;
  }
  */

