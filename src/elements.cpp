#include <iostream>
#include <string>

#include "elements.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

void Elements::setup()
{
  set_locs();
  set_shape();
  set_transforms();
  set_normals();
  setup_FR();

  initialize_U();
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

  /* Initialize solution */

}


