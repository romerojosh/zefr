#include <iostream>
#include <memory>
#include <string>

#include "elements.hpp"
#include "faces.hpp"
#include "mdvector.hpp"
#include "macros.hpp"
#include "points.hpp"
#include "polynomials.hpp"

void Elements::associate_faces(std::shared_ptr<Faces> faces)
{
  this->faces = faces;
  this->faces->setup(nDims, nVars);
}

void Elements::setup()
{
  set_locs();
  set_shape();
  set_transforms();
  set_normals();
  setup_FR();
}
