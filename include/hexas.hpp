#ifndef hexas_hpp
#define hexas_hpp

#include <string>

#include "input.hpp"
#include "mdvector.hpp"

class Hexas: public Elements 
{
  private:
    void set_locs();

  public:
    Hexas(unsigned int nEles, unsigned int shape_order, const InputStruct *input, 
          unsigned int order = -1);

};

#endif /* hexas_hpp */
