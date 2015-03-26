#include <iostream>
#include <string>

#include "elements.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

Elements::Elements(unsigned int nEles, std::string ele_type, unsigned int shape_order,
                   const InputStruct *input, unsigned int P)
{
  this->nEles = nEles;  
  this->ele_type = ele_type;  
  this->input = input;  
  this->shape_order = shape_order;  

  if (ele_type == "quad")
  {
    /* Generic quadrilateral geometry */
    nDims = 2;
    nFaces = 4;
    nNodes = 4 * shape_order;
    
    if (P == -1)
    {
      nSpts = (input->order+1)*(input->order+1);
      nSpts1D = input->order+1;
      nFptsPerFace = input->order+1;
      order = input->order;
    }
    else
    {
      nSpts = (P+1)*(P+1);
      nSpts1D = P+1;
      nFptsPerFace = P+1;
      order = P;
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

  else if (ele_type == "hexa")
  {
    /* Generic hexahedral geometry */
    nDims = 3;
    nFaces = 6;
    nNodes = 8 * shape_order;
    
    if (P == -1)
    {
      nSpts = (input->order+1)*(input->order+1)*(input->order+1);
      nSpts1D = input->order+1;
      nFptsPerFace = (input->order+1)*(input->order+1);
      order = input->order;
    }
    else
    {
      nSpts = (P+1)*(P+1)*(P+1);
      nSpts1D = (P+1)*(P+1);
      nFptsPerFace = (P+1)*(P+1);
      order = P;
    }

    nFpts = nFptsPerFace * nFaces;

    if (input->equation == "AdvDiff")
    {
      nVars = 1;
    }
    else if (input->equation == "EulerNS")
    {
      nVars = 5;
    }
    else
    {
      ThrowException("Equation not recognized: " + input->equation);
    }

  }
  else
  {
    ThrowException("Unsupported element type: " + ele_type);
  }

}

void Elements::setup()
{
  set_locs();
  //initialize_U();
}

void Elements::set_locs()
{

}
void Elements::initialize_U()
{
  /* Note: Need to take care about data continuity */

  /* Allocate memory for solution data structures */
  /* Solution and Flux Variables */
  U_spts.assign({nVars, nEles, nSpts});
  U_fpts.assign({nVars, nEles, nFpts});

  F_spts.assign({nDims, nVars, nEles, nSpts});
  F_fpts.assign({nDims, nVars, nEles, nFpts});

  dU_spts.assign({nDims, nVars, nEles, nSpts});
  dF_spts.assign({nDims, nVars, nEles, nSpts});

  divF_spts.assign({nVars, nEles, nSpts});

}

void Elements::setup_FR()
{
  /* Allocate memory for FR operators */
  oppE.assign({nFpts, nSpts});
  oppD.assign({nDims, nSpts, nSpts});
  oppD_fpts.assign({nDims, nSpts, nFpts});

}
