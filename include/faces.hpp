#ifndef faces_hpp
#define faces_hpp

#include <string>
#include <vector>

#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"

//class Elements;
class FRSolver;
class Quads;
class Faces 
{
  friend class FRSolver;
  friend class Quads;

  private:
    const InputStruct *input = NULL;
    GeoStruct *geo = NULL;
    unsigned int nFpts, nDims, nVars;

    void apply_bcs();
    void apply_bcs_dU();
    //void compute_Fconv();
    //void compute_Fvisc();
    void rusanov_flux();
    void LDG_flux();
    void transform_flux();

  protected:
    mdvector<double> U, dU, Fconv, Fvisc, Fcomm, Ucomm;
    mdvector<double> norm, jaco;
    mdvector<int> outnorm;
    std::vector<double> dA;

  public:
    Faces(GeoStruct *geo, const InputStruct *input);
    void setup(unsigned int nDims, unsigned int nVars);
    void compute_common_U();
    void compute_common_F();
    void compute_Fconv();
    void compute_Fvisc();

    
};

#endif /* faces_hpp */
