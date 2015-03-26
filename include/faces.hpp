#ifndef faces_hpp
#define faces_hpp

#include <string>

#include "mdvector.hpp"

class faces 
{
  private:
    const inputStruct &input;
    std::string face_type;
    unsigned int nFpts, nDims, nVars;
    mdvector U, dU, F, Fcomm;

    void apply_bcs();
    void compute_Fconv();
    void compute_Fvisc();
    void rusanov_flux();
    void rusanov_flux();
    void rusanov_flux();

  public:
    faces(unsigned int nFpts, std::string face_type, const inputStruct &input);
    void setup();
    void set_face_U(unsigned int val, unsigned int gfpt, unsigned int var, 
                    unsigned int slot);
    void set_face_dU(unsigned int val, unsigned int gfpt, unsigned int var, 
                    unsigned int slot);
    void compute_common_U();
    void compute_common_F();
    
}

#endif /* elements_hpp */
