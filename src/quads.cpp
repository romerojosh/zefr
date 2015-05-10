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

Quads::Quads(GeoStruct *geo, const InputStruct *input, int order)
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
  
  if (input->equation == "AdvDiff")
  {
    nVars = 1;
  }
  else if (input->equation == "EulerNS")
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
  loc_fpts.assign({nFpts,nDims}); idx_fpts.assign({nFpts, nDims});
  loc_ppts.assign({nPpts,nDims}); idx_ppts.assign({nPpts,nDims});
  loc_qpts.assign({nQpts,nDims}); idx_qpts.assign({nQpts,nDims});

  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
     loc_spts_1D = Gauss_Legendre_pts(order+1); 
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

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

void Quads::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nSpts, nNodes},1);
  shape_fpts.assign({nFpts, nNodes},1);
  shape_ppts.assign({nPpts, nNodes},1);
  shape_qpts.assign({nQpts, nNodes},1);
  dshape_spts.assign({nDims, nSpts, nNodes},1);
  dshape_fpts.assign({nDims, nFpts, nNodes},1);
  dshape_ppts.assign({nDims, nPpts, nNodes},1);
  dshape_qpts.assign({nDims, nQpts, nNodes},1);

  /* Shape functions and derivatives at solution points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(spt,node) = calc_shape_quad(shape_order, node, 
        loc_spts(spt,0),loc_spts(spt,1));
      dshape_spts(0,spt,node) = calc_dshape_quad(shape_order, node, 
        loc_spts(spt,0), loc_spts(spt,1), 0);
      dshape_spts(1,spt,node) = calc_dshape_quad(shape_order, node, 
        loc_spts(spt,0), loc_spts(spt,1), 1);
    }
  }

  /* Shape functions and derivatives at flux points */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(fpt,node) = calc_shape_quad(shape_order, node, 
        loc_fpts(fpt,0),loc_fpts(fpt,1));
      dshape_fpts(0,fpt,node) = calc_dshape_quad(shape_order, node, 
        loc_fpts(fpt,0), loc_fpts(fpt,1), 0);
      dshape_fpts(1,fpt,node) = calc_dshape_quad(shape_order, node, 
        loc_fpts(fpt,0), loc_fpts(fpt,1), 1);
    }
  }

    /* Shape function and derivatives at plot points */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_ppts(ppt,node) = calc_shape_quad(shape_order, node, 
        loc_ppts(ppt,0),loc_ppts(ppt,1));
      dshape_ppts(0,ppt,node) = calc_dshape_quad(shape_order, node, 
        loc_ppts(ppt,0), loc_ppts(ppt,1), 0);
      dshape_ppts(1,ppt,node) = calc_dshape_quad(shape_order, node, 
        loc_ppts(ppt,0), loc_ppts(ppt,1), 1);
    }
  }
  
  /* Shape function and derivatives at quadrature points */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_qpts(qpt,node) = calc_shape_quad(shape_order, node, 
        loc_qpts(qpt,0),loc_qpts(qpt,1));
      dshape_qpts(0,qpt,node) = calc_dshape_quad(shape_order, node, 
        loc_qpts(qpt,0), loc_qpts(qpt,1), 0);
      dshape_qpts(1,qpt,node) = calc_dshape_quad(shape_order, node, 
        loc_qpts(qpt,0), loc_qpts(qpt,1), 1);
    }
  }
}

void Quads::set_transforms()
{
  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nEles, nSpts, nDims, nDims});
  jaco_ppts.assign({nEles, nPpts, nDims, nDims});
  jaco_qpts.assign({nEles, nQpts, nDims, nDims});
  jaco_det_spts.assign({nEles, nSpts});
  jaco_det_qpts.assign({nEles, nQpts});

  /* Set jacobian matrix and determinant at solution points */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int dimX = 0; dimX < nDims; dimX++)
      {
        for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(ele, node);
            jaco_spts(ele,spt,dimX,dimXi) += geo->coord_nodes(gnd,dimX) * dshape_spts(dimXi,spt,node); 
          }
        }
      }

      jaco_det_spts(ele,spt) = jaco_spts(ele,spt,0,0) * jaco_spts(ele,spt,1,1) -
                               jaco_spts(ele,spt,0,1) * jaco_spts(ele,spt,1,0); 


      if (jaco_det_spts(ele,spt) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " spt:" + std::to_string(spt));

    }
  }

  /* Set jacobian matrix at face flux points (do not need the determinant) */
  for (unsigned int ele = 0; ele < nEles; ele++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int dimX = 0; dimX < nDims; dimX++)
      {
        for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(ele, node);
            int gfpt = geo->fpt2gfpt(ele,fpt);

            /* Skip fpts on ghost edges */
            if (gfpt == -1)
              continue;

            unsigned int slot = geo->fpt2gfpt_slot(ele,fpt);

            faces->jaco(gfpt, dimX, dimXi, slot) += geo->coord_nodes(gnd,dimX) * dshape_fpts(dimXi, fpt, node);
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
      for (unsigned int dimX = 0; dimX < nDims; dimX++)
      {
        for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(ele, node);
            jaco_ppts(ele,ppt,dimX,dimXi) += geo->coord_nodes(gnd,dimX) * dshape_ppts(dimXi,ppt,node); 
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
      for (unsigned int dimX = 0; dimX < nDims; dimX++)
      {
        for (unsigned int dimXi = 0; dimXi < nDims; dimXi++)
        {
          for (unsigned int node = 0; node < nNodes; node++)
          {
            unsigned int gnd = geo->nd2gnd(ele, node);
            jaco_qpts(ele,qpt,dimX,dimXi) += geo->coord_nodes(gnd,dimX) * dshape_qpts(dimXi,qpt,node); 
          }
        }
      }

      jaco_det_qpts(ele,qpt) = jaco_qpts(ele,qpt,0,0) * jaco_qpts(ele,qpt,1,1) -
                               jaco_qpts(ele,qpt,0,1) * jaco_qpts(ele,qpt,1,0); 


      if (jaco_det_qpts(ele,qpt) < 0.)
        ThrowException("Nonpositive Jacobian detected: ele: " + std::to_string(ele) + " qpt:" + std::to_string(qpt));

    }
  }

}

void Quads::set_normals()
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
      int gfpt = geo->fpt2gfpt(ele,fpt);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1) 
        continue;

      unsigned int slot = geo->fpt2gfpt_slot(ele,fpt);

      faces->norm(0,gfpt,slot) = faces->jaco(gfpt,1,1,slot) * tnorm(fpt,0) - 
                                 faces->jaco(gfpt,1,0,slot) * tnorm(fpt,1);
      faces->norm(1,gfpt,slot) = -faces->jaco(gfpt,0,1,slot) * tnorm(fpt,0) + 
                                 faces->jaco(gfpt,0,0,slot) * tnorm(fpt,1);

      faces->dA[gfpt] = std::sqrt(faces->norm(0,gfpt,slot)*faces->norm(0,gfpt,slot) + 
                        faces->norm(1,gfpt,slot)*faces->norm(1,gfpt,slot));

      faces->norm(0,gfpt,slot) /= faces->dA[gfpt];
      faces->norm(1,gfpt,slot) /= faces->dA[gfpt];

      unsigned int face_idx = fpt/nSpts1D;

      if(face_idx == 0 || face_idx == 3)
        faces->outnorm(gfpt,slot) = -1; 
      else 
        faces->outnorm(gfpt,slot) = 1; 

    }
  }

}

void Quads::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppD.assign({nDims, nSpts, nSpts});
  oppD_fpts.assign({nDims, nSpts, nFpts});

  /* Setup spt to fpt extrapolation operator (oppE) */
  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get indices for Lagrange polynomial evaluation */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);

      oppE(fpt,spt) = Lagrange(loc_spts_1D, i, loc_fpts(fpt,0)) * 
                      Lagrange(loc_spts_1D, j, loc_fpts(fpt,1));
    }
  }

  /* Setup differentiation operator (oppD) for solution points */
  /* Note: Can set up for standard FR eventually. Trying to keep things simple.. */
  auto loc_DFR_1D = loc_spts_1D;
  loc_DFR_1D.insert(loc_DFR_1D.begin(), -1.0);
  loc_DFR_1D.insert(loc_DFR_1D.end(), 1.0);

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ispt = 0; ispt < nSpts; ispt++)
    {
      for (unsigned int jspt = 0; jspt < nSpts; jspt++)
      {
        /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
         * boundary points for DFR) */
        unsigned int i = idx_spts(jspt,0) + 1;
        unsigned int j = idx_spts(jspt,1) + 1;

        if (dim == 0)
        {
            oppD(dim,ispt,jspt) = Lagrange_d1(loc_DFR_1D, i, loc_spts(ispt,0)) *
                                  Lagrange(loc_DFR_1D, j, loc_spts(ispt,1));
        }
        else
        {
            oppD(dim,ispt,jspt) = Lagrange(loc_DFR_1D, i, loc_spts(ispt,0)) *
                                  Lagrange_d1(loc_DFR_1D, j, loc_spts(ispt,1));
        }
      }
    }
  }

  /* Setup differentiation operator (oppD_fpts) for flux points (DFR Specific)*/
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int fpt = 0; fpt < nFpts; fpt++)
      {
        /* Get indices for Lagrange polynomial evaluation (shifted due to inclusion of
         * boundary points for DFR) */
        unsigned int i = idx_fpts(fpt,0) + 1;
        unsigned int j = idx_fpts(fpt,1) + 1;

        if (dim == 0)
        {
            oppD_fpts(dim,spt,fpt) = Lagrange_d1(loc_DFR_1D, i, loc_spts(spt,0)) *
                              Lagrange(loc_DFR_1D, j, loc_spts(spt,1));
        }
        else
        {
            oppD_fpts(dim,spt,fpt) = Lagrange(loc_DFR_1D, i, loc_spts(spt,0)) *
                              Lagrange_d1(loc_DFR_1D, j, loc_spts(spt,1));
        }
      }
    }
  }

}

void Quads::setup_aux()
{
  /* Allocate memory for plot point and quadrature point interpolation operator */
  oppE_ppts.assign({nPpts, nSpts});
  oppE_qpts.assign({nQpts, nSpts});

  /* Setup spt to ppt extrapolation operator (oppE_ppts) */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get indices for Lagrange polynomial evaluation */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);

      oppE_ppts(ppt,spt) = Lagrange(loc_spts_1D, i, loc_ppts(ppt,0)) * 
                      Lagrange(loc_spts_1D, j, loc_ppts(ppt,1));
    }
  }

  /* Setup spt to qpt extrapolation operator (oppE_qpts) */
  for (unsigned int qpt = 0; qpt < nQpts; qpt++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get indices for Lagrange polynomial evaluation */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);

      oppE_qpts(qpt,spt) = Lagrange(loc_spts_1D, i, loc_qpts(qpt,0)) * 
                      Lagrange(loc_spts_1D, j, loc_qpts(qpt,1));
    }
  }


}

void Quads::set_coords()
{
  /* Allocate memory for physical coordinates */
  geo->coord_spts.assign({nDims, nSpts, nEles});
  geo->coord_fpts.assign({nDims, nFpts, nEles});
  geo->coord_ppts.assign({nDims, nPpts, nEles});
  geo->coord_qpts.assign({nDims, nQpts, nEles});

  /* Setup physical coordinates at solution points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_spts(dim, spt, ele) += geo->coord_nodes(gnd,dim) * shape_spts(spt, node);
        }
      }
    }
  }
  
  /* Setup physical coordinates at flux points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_fpts(dim, fpt, ele) += geo->coord_nodes(gnd,dim) * shape_fpts(fpt, node);
        }
      }
    }
  }

  /* Setup physical coordinates at plot points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int ppt = 0; ppt < nPpts; ppt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_ppts(dim, ppt, ele) += geo->coord_nodes(gnd,dim) * shape_ppts(ppt, node);
        }
      }
    }
  }

  /* Setup physical coordinates at quadrature points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int qpt = 0; qpt < nQpts; qpt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = geo->nd2gnd(ele, node);
          geo->coord_qpts(dim, qpt, ele) += geo->coord_nodes(gnd,dim) * shape_qpts(qpt, node);
        }
      }
    }
  }

}

void Quads::compute_Fconv()
{
  if (input->equation == "AdvDiff")
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          F_spts(0, n, spt, ele) = input->AdvDiff_Ax * U_spts(n, spt, ele);
          F_spts(1, n, spt, ele) = input->AdvDiff_Ay * U_spts(n, spt, ele);
        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("Euler flux not implemented yet!");
  }
}

void Quads::compute_Fvisc()
{
  if (input->equation == "AdvDiff")
  {
    for (unsigned int n = 0; n < nVars; n++)
    {
      for (unsigned int spt = 0; spt < nSpts; spt++)
      {
        for (unsigned int ele = 0; ele < nEles; ele++)
        {
          /* Can just add viscous flux to existing convective flux */
          F_spts(0, n, spt, ele) += -input->AdvDiff_D * dU_spts(0, n, spt, ele);
          F_spts(1, n, spt, ele) += -input->AdvDiff_D * dU_spts(1, n, spt, ele);
        }
      }
    }
  }
  else if (input->equation == "EulerNS")
  {
    ThrowException("NS flux not implemented yet!");
  }

}
void Quads::transform_flux()
{
  for (unsigned int n = 0; n < nVars; n++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        double Ftemp = F_spts(0, n, spt, ele);

        F_spts(0, n, spt, ele) = F_spts(0, n, spt, ele) * jaco_spts(ele, spt, 1, 1) -
                                 F_spts(1, n, spt, ele) * jaco_spts(ele, spt, 0, 1);
        F_spts(1, n, spt, ele) = F_spts(1, n, spt, ele) * jaco_spts(ele, spt, 0, 0) -
                                 Ftemp * jaco_spts(ele, spt, 1, 0);
      }
    }
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
}
