#ifndef quads_hpp
#define quads_hpp

#include <string>

#include "elements.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "solver.hpp"

//class FRSolver;
class Quads: public Elements 
{
  //friend class FRSolver;
  private:
    void set_locs();
    void set_shape();
    void set_transforms();
    void set_normals();
    void setup_FR();
    void setup_plot();
    void set_coords();

  public:
    Quads(GeoStruct *geo, const InputStruct *input, 
          unsigned int order = -1);
    void compute_Fconv();
    void transform_flux();

};

#endif /* quads_hpp */
