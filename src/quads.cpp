#include <cmath>
#include <iostream>
#include <string>

#include "faces.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "quads.hpp"
#include "shape.hpp"

Quads::Quads(unsigned int nEles, unsigned int shape_order,
                   const InputStruct *input, unsigned int order)
{
  this->nEles = nEles;  
  this->input = input;  
  this->shape_order = shape_order;  

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
}

void Quads::set_shape()
{
  /* Allocate memory for shape function and related derivatives */
  shape_spts.assign({nSpts, nNodes},1);
  shape_fpts.assign({nFpts, nNodes},1);
  dshape_spts.assign({nDims, nSpts, nNodes},1);
  dshape_fpts.assign({nDims, nFpts, nNodes},1);

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
}

void Quads::set_transforms()
{
  /* Allocate memory for jacobian matrices and determinant */
  jaco_spts.assign({nEles, nSpts, nDims, nDims});
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
            unsigned int gnd = nd2gnd(ele, node);
            jaco_spts(ele,spt,dimX,dimXi) += coord_nodes(gnd,dimX) * dshape_spts(dimXi,spt,node); 
          }
        }
      }

      jaco_det_spts(ele,spt) = jaco_spts(ele,spt,0,0) * jaco_spts(ele,spt,1,1) -
                               jaco_spts(ele,spt,0,1) * jaco_spts(ele,spt,1,0); 


      std::cout << jaco_det_spts(ele,spt) << std::endl;
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
            unsigned int gnd = nd2gnd(ele, node);
            unsigned int gfpt = fpt2gfpt(ele,fpt);
            unsigned int slot = fpt2gfpt_slot(ele,fpt);

            faces->jaco(gfpt, dimX, dimXi, slot) += coord_nodes(gnd,dimX) * dshape_fpts(dimXi, fpt, node);
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
      unsigned int gfpt = fpt2gfpt(ele,fpt);

      /* Check if flux point is on ghost edge */
      if (gfpt == -1) 
        continue;

      unsigned int slot = fpt2gfpt_slot(ele,fpt);

      faces->norm(0,gfpt,slot) = faces->jaco(gfpt,1,1,slot) * tnorm(fpt,0) - 
                                 faces->jaco(gfpt,1,0,slot) * tnorm(fpt,1);
      faces->norm(1,gfpt,slot) = -faces->jaco(gfpt,0,1,slot) * tnorm(fpt,0) + 
                                 faces->jaco(gfpt,0,0,slot) * tnorm(fpt,1);

      faces->dA[gfpt] = std::sqrt(faces->norm(0,gfpt,slot)*faces->norm(0,gfpt,slot) + 
                        faces->norm(1,gfpt,slot)*faces->norm(1,gfpt,slot));

      faces->norm(0,gfpt,slot) /= faces->dA[gfpt];
      faces->norm(1,gfpt,slot) /= faces->dA[gfpt];

      std::cout << gfpt << " " << faces->norm(0,gfpt,slot) << " " << faces->norm(1,gfpt,slot) << std::endl;

      switch(fpt/nSpts1D)
      {
        case (0,3):
          faces->outnorm(gfpt,slot) = -1;
        case (1,2):
          faces->outnorm(gfpt,slot) = 1;
      }
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

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      std::cout << oppD_fpts(0,spt,fpt) << " ";
    }
    std::cout << std::endl;
  }

}

void Quads::set_coords()
{
  /* Allocate memory for physical coordinates */
  coord_spts.assign({nDims, nSpts, nEles});
  coord_fpts.assign({nDims, nFpts, nEles});

  /* Setup physical coordinates at solution points */
  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int spt = 0; spt < nSpts; spt++)
    {
      for (unsigned int ele = 0; ele < nEles; ele++)
      {
        for (unsigned int node = 0; node < nNodes; node++)
        {
          unsigned int gnd = nd2gnd(ele, node);
          coord_spts(dim, spt, ele) += coord_nodes(gnd,dim) * shape_spts(spt, node);
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
          unsigned int gnd = nd2gnd(ele, node);
          coord_fpts(dim, fpt, ele) += coord_nodes(gnd,dim) * shape_fpts(fpt, node);
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
