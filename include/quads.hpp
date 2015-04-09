#ifndef quads_hpp
#define quads_hpp

#include <string>

#include "elements.hpp"
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

  public:
    Quads(unsigned int nEles, unsigned int shape_order, const InputStruct *input, 
          unsigned int order = -1);

};

#endif /* quads_hpp */
