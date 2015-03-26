#include <iostream>
#include <string>

#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "quads.hpp"

Quads::Quads(unsigned int nEles, unsigned int shape_order,
                   const InputStruct *input, unsigned int order)
{
  this->nEles = nEles;  
  this->ele_type = ele_type;  
  this->input = input;  
  this->shape_order = shape_order;  

  /* Generic quadrilateral geometry */
  nDims = 2;
  nFaces = 4;
  nNodes = 4 * shape_order;
  
  /* If order argument is not provided, use order in input file */
  if (order == -1)
  {
    nSpts = (input->order+1)*(input->order+1);
    nSpts1D = input->order+1;
    nFptsPerFace = input->order+1;
    this->order = input->order;
  }
  else
  {
    nSpts = (order+1)*(order+1);
    nSpts1D = order+1;
    nFptsPerFace = order+1;
    this->order = order;
  }

  nFpts = nFptsPerFace * nFaces;
  
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
  loc_spts.assign({nSpts,nDims});
  loc_fpts.assign({nFpts,nDims});
  loc_nodes.assign({nNodes,nDims});

  /* Get positions of points in 1D */
  if (input->spt_type == "Legendre")
     loc_spts_1D = Gauss_Legendre_pts(order+1); 
  else
    ThrowException("spt_type not recognized: " + input->spt_type);

  loc_nodes_1D = Shape_pts(shape_order); 

  /* Setup solution point locations */
  unsigned int spt = 0;
  for (unsigned int i = 0; i < nSpts1D; i++)
  {
    for (unsigned int j = 0; j < nSpts1D; j++)
    {
      loc_spts(spt,0) = loc_spts_1D[j];
      loc_spts(spt,1) = loc_spts_1D[i];
      spt++;
    }
  }

  /* Setup flux point locations */
  unsigned int fpt = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < nFptsPerFace; j++)
    {
      switch(i)
      {
        case 0: /* Bottom edge */
          loc_fpts(fpt,0) = loc_spts_1D[j];
          loc_fpts(fpt,1) = -1.0; break;

        case 1: /* Right edge */
          loc_fpts(fpt,0) = 1.0; 
          loc_fpts(fpt,1) = loc_spts_1D[j]; break;

        case 2: /* Upper edge */
          loc_fpts(fpt,0) = loc_spts_1D[nSpts1D-j-1];
          loc_fpts(fpt,1) = 1.0; break;

        case 3: /* Left edge */
          loc_fpts(fpt,0) = -1.0; 
          loc_fpts(fpt,1) = loc_spts_1D[nSpts1D-j-1]; break;
      }

      fpt++;
    }
  }

  /* Setup shape point (node) locations */
  unsigned int node = 0;
  for (unsigned int i = 0; i < nFaces; i++)
  {
    for (unsigned int j = 0; j < shape_order + 1; j++)
    {
      if (node >= nNodes) break;

      switch(i)
      {
        case 0: /* Bottom edge */
          loc_nodes(node,0) = loc_nodes_1D[j];
          loc_nodes(node,1) = -1.0; break;

        case 1: /* Right edge */
          loc_nodes(node,0) = 1.0; 
          loc_nodes(node,1) = loc_nodes_1D[j]; break;

        case 2: /* Upper edge */
          loc_nodes(node,0) = loc_nodes_1D[shape_order-j];
          loc_nodes(node,1) = 1.0; break;

        case 3: /* Left edge */
          loc_nodes(node,0) = -1.0; 
          loc_nodes(node,1) = loc_nodes_1D[shape_order-j]; break;
      }

      node++;
    }
    node --;
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
      /* Compute indices for Lagrange polynomial evaluation */
      unsigned int i = (spt + nSpts1D) % nSpts1D;
      unsigned int j = spt/(order+1);

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
        /* Compute indices for Lagrange polynomial evaluation (shifted due to inclusion of
         * boundary points for DFR) */
        unsigned int i = (jspt + nSpts1D) % nSpts1D + 1;
        unsigned int j = jspt/(order+1) + 1;

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
        unsigned int i, j;

        switch(fpt/nFptsPerFace)
        {
          case 0: /* Bottom edge*/
            i = fpt + 1;
            j = 0; break;

          case 1: /* Right edge*/
            i = nFptsPerFace + 1;
            j = (fpt + nSpts1D) % nSpts1D + 1; break;

          case 2: /* Top edge*/
            i = 3*nFptsPerFace - fpt; 
            j = nFptsPerFace + 1; break;

          case 3: /* Left edge*/
            i = 0;
            j = 4*nFptsPerFace - fpt; break; 
        }

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

