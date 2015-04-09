#include "faces.hpp"

Faces::Faces(unsigned int nFpts, const InputStruct *input)
{
  this->nFpts = nFpts;
  this->input = input;
}

void Faces::setup(unsigned int nDims, unsigned int nVars)
{
  /* Allocate memory for solution structures */
  U.assign({nVars, nFpts, 2});
  dU.assign({nDims, nVars, nFpts, 2});
  F.assign({nDims, nVars, nFpts, 2});
  Fcomm.assign({nDims, nVars, nFpts, 2});

  /* Allocate memory for geometry structures */
  norm.assign({nDims, nFpts, 2});
  outnorm.assign({nFpts,2});
  dA.assign(nFpts,0.0);
  jaco.assign({nFpts, nDims, nDims , 2});
}


