#include <iostream>
#include <string>

#include "hexas.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

Hexas::Hexas(unsigned int nEles, unsigned int shape_order,
                   const InputStruct *input, unsigned int order)
{
  this->nEles = nEles;  
  this->input = input;  
  this->shape_order = shape_order;  

  /* Generic hexahedral geometry */
  nDims = 3;
  nFaces = 6;
  nNodes = 8 * shape_order;
  
  if (order == -1)
  {
    nSpts = (input->order+1)*(input->order+1)*(input->order+1);
    nSpts1D = input->order+1;
    nFptsPerFace = (input->order+1)*(input->order+1);
    this->order = input->order;
  }
  else
  {
    nSpts = (order+1)*(order+1)*(order+1);
    nSpts1D = order+1;
    nFptsPerFace = (order+1)*(order+1);
    this->order = order;
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

void Hexas::set_locs()
{

}
