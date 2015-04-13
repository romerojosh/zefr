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

Quads::Quads(GeoStruct *geo, const InputStruct *input, unsigned int order)
{
  this->geo = geo;
  this->input = input;  
  this->shape_order = geo->shape_order;  
  this->nEles = geo->nEles;  

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

  /* For debugging, setup reference quad element */
  /*
  nd2gnd.assign({nEles, nNodes});
  fpt2gfpt.assign({nEles, nFpts});
  fpt2gfpt_slot.assign({nEles, nFpts},0);

  for (unsigned int ele = 0; ele < nEles; ele++)
    for(unsigned int fpt = 0; fpt < nFpts; fpt++)
      fpt2gfpt(ele,fpt) = fpt;

  coord_nodes.assign({nNodes, nDims});

 
  // Bilinear Quad
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 1; 
  nd2gnd(0,2) = 2; nd2gnd(0,3) = 3; 

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 1.0; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 1.0; coord_nodes(2,1) = 1.0;
  coord_nodes(3,0) = 0.0; coord_nodes(3,1) = 1.0;
  */

  /*
  // Bilinear Quad to Triangle
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 1; 
  nd2gnd(0,2) = 2; nd2gnd(0,3) = 2; 

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 1.0; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 0.0; coord_nodes(2,1) = 1.0;
  */
  
  /*
  // Biquadratic Quad
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 4; nd2gnd(0,2) = 1;
  nd2gnd(0,3) = 7; nd2gnd(0,4) = 8; nd2gnd(0,5) = 5;
  nd2gnd(0,6) = 3; nd2gnd(0,7) = 6; nd2gnd(0,8) = 2;

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 1.0; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 1.0; coord_nodes(2,1) = 1.0;
  coord_nodes(3,0) = 0.0; coord_nodes(3,1) = 1.0;
  coord_nodes(4,0) = 0.5; coord_nodes(4,1) = 0.0;
  coord_nodes(5,0) = 1.0; coord_nodes(5,1) = 0.5;
  coord_nodes(6,0) = 0.5; coord_nodes(6,1) = 1.0;
  coord_nodes(7,0) = 0.0; coord_nodes(7,1) = 0.5;
  coord_nodes(8,0) = 0.5; coord_nodes(8,1) = 0.5;
  */

  /*
  // Biquadratic Quad to Triangle
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 3; nd2gnd(0,2) = 1;
  nd2gnd(0,3) = 5; nd2gnd(0,4) = 4; nd2gnd(0,5) = 4;
  nd2gnd(0,6) = 2; nd2gnd(0,7) = 2; nd2gnd(0,8) = 2;

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 1.0; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 0.0; coord_nodes(2,1) = 1.0;
  coord_nodes(3,0) = 0.5; coord_nodes(3,1) = 0.0;
  coord_nodes(4,0) = 0.5; coord_nodes(4,1) = 0.5;
  coord_nodes(5,0) = 0.0; coord_nodes(5,1) = 0.5;
  */

  /*  
  //8-node Serendipity Quad
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 1; nd2gnd(0,2) = 2;
  nd2gnd(0,3) = 3; nd2gnd(0,4) = 4; nd2gnd(0,5) = 5;
  nd2gnd(0,6) = 6; nd2gnd(0,7) = 7; 

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 0.5; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 1.0; coord_nodes(2,1) = 0.0;
  coord_nodes(3,0) = 1.0; coord_nodes(3,1) = 0.5;
  coord_nodes(4,0) = 1.0; coord_nodes(4,1) = 1.0;
  coord_nodes(5,0) = 0.5; coord_nodes(5,1) = 1.0;
  coord_nodes(6,0) = 0.0; coord_nodes(6,1) = 1.0;
  coord_nodes(7,0) = 0.0; coord_nodes(7,1) = 0.5;
  */

  /*
  // 8-node Serendipity Quad to Triangle
  nd2gnd(0,0) = 0; nd2gnd(0,1) = 1; nd2gnd(0,2) = 2;
  nd2gnd(0,3) = 3; nd2gnd(0,4) = 4; nd2gnd(0,5) = 4;
  nd2gnd(0,6) = 4; nd2gnd(0,7) = 5; 

  coord_nodes(0,0) = 0.0; coord_nodes(0,1) = 0.0;
  coord_nodes(1,0) = 0.5; coord_nodes(1,1) = 0.0;
  coord_nodes(2,0) = 1.0; coord_nodes(2,1) = 0.0;
  coord_nodes(3,0) = 0.5; coord_nodes(3,1) = 0.5;
  coord_nodes(4,0) = 0.0; coord_nodes(4,1) = 1.0;
  coord_nodes(5,0) = 0.0; coord_nodes(5,1) = 0.5;
  */
  
}

void Quads::set_locs()
{
  /* Allocate memory for point location structures */
  loc_spts.assign({nSpts,nDims}); idx_spts.assign({nSpts,nDims});
  loc_fpts.assign({nFpts,nDims}); idx_fpts.assign({nFpts, nDims});
  loc_ppts.assign({nPpts,nDims}); idx_ppts.assign({nPpts,nDims});

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

}

void Quads::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nSpts, nNodes},1);
  shape_fpts.assign({nFpts, nNodes},1);
  shape_ppts.assign({nPpts, nNodes},1);
  dshape_spts.assign({nDims, nSpts, nNodes},1);
  dshape_fpts.assign({nDims, nFpts, nNodes},1);
  dshape_ppts.assign({nDims, nPpts, nNodes},1);

  /* Shape functions at solution and flux points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_spts(spt,node) = calc_shape_quad(shape_order, node, 
        loc_spts(spt,0),loc_spts(spt,1));
    }
  }

  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      shape_fpts(fpt,node) = calc_shape_quad(shape_order, node, 
        loc_fpts(fpt,0),loc_fpts(fpt,1));
    }
  }


  /* Shape function derivatives at solution and flux points */
  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      dshape_spts(0,spt,node) = calc_dshape_quad(shape_order, node, 
        loc_spts(spt,0), loc_spts(spt,1), 0);
      dshape_spts(1,spt,node) = calc_dshape_quad(shape_order, node, 
        loc_spts(spt,0), loc_spts(spt,1), 1);
    }
  }

  for (unsigned int fpt = 0; fpt < nFpts; fpt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
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
    }
  }

  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int node = 0; node < nNodes; node++)
    {
      dshape_ppts(0,ppt,node) = calc_dshape_quad(shape_order, node, 
        loc_ppts(ppt,0), loc_ppts(ppt,1), 0);
      dshape_ppts(1,ppt,node) = calc_dshape_quad(shape_order, node, 
        loc_ppts(ppt,0), loc_ppts(ppt,1), 1);
    }
  }
}

void Quads::set_transforms()
{
  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nEles, nSpts, nDims, nDims});
  jaco_ppts.assign({nEles, nPpts, nDims, nDims});
  jaco_det_spts.assign({nEles, nSpts});

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
            unsigned int gfpt = geo->fpt2gfpt(ele,fpt);

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
      unsigned int gfpt = geo->fpt2gfpt(ele,fpt);

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

      //std::cout << faces->norm(0,gfpt, slot) <<" " <<faces->norm(1,gfpt,slot)<<std::endl;

      unsigned int faceidx = fpt/nSpts1D;

      if(fpt/nSpts1D == 0 || fpt/nSpts1D == 3)
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

void Quads::setup_plot()
{
  /* Allocate memory for plot point interpolation operator */
  oppE_plot.assign({nPpts, nSpts});

  /* Setup spt to ppt extrapolation operator (oppE_plot) */
  for (unsigned int ppt = 0; ppt < nPpts; ppt++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      /* Get indices for Lagrange polynomial evaluation */
      unsigned int i = idx_spts(spt,0);
      unsigned int j = idx_spts(spt,1);

      oppE_plot(ppt,spt) = Lagrange(loc_spts_1D, i, loc_ppts(ppt,0)) * 
                      Lagrange(loc_spts_1D, j, loc_ppts(ppt,1));
    }
  }


}

void Quads::set_coords()
{
  /* Allocate memory for physical coordinates */
  geo->coord_spts.assign({nDims, nSpts, nEles});
  geo->coord_fpts.assign({nDims, nFpts, nEles});
  geo->coord_ppts.assign({nDims, nPpts, nEles});

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
          F_spts(0, n, spt, ele) += input->AdvDiff_D * dU_spts(0, n, spt, ele);
          F_spts(1, n, spt, ele) += input->AdvDiff_D * dU_spts(1, n, spt, ele);
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
