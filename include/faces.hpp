#ifndef faces_hpp
#define faces_hpp

#include <string>
#include <vector>

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
    unsigned int nFpts, nDims, nVars;

    void apply_bcs();
    void compute_Fconv();
    void compute_Fvisc();
    void rusanov_flux();

  protected:
    mdvector<double> U, dU, F, Fcomm;
    mdvector<double> norm, jaco;
    mdvector<int> outnorm;
    std::vector<double> dA;

  public:
    Faces(unsigned int nFpts, const InputStruct *input);
    void setup(unsigned int nDims, unsigned int nVars);
    void compute_common_U();
    void compute_common_F();

    
};

#endif /* faces_hpp */
