#ifndef faces_hpp
#define faces_hpp

#include <string>
#include <vector>

#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

//class Elements;
class FRSolver;
class Elements;
class Quads;
class Faces 
{
  friend class FRSolver;
  friend class Elements;
  friend class Quads;

  private:
    //const InputStruct *input = NULL;
    InputStruct *input = NULL;
    GeoStruct *geo = NULL;
    unsigned int nFpts, nDims, nVars;

    void apply_bcs();
    void apply_bcs_dU();
    //void compute_Fconv();
    //void compute_Fvisc();
    void rusanov_flux();
    void LDG_flux();
    void central_flux();
    void transform_flux();

  protected:
    mdvector<double> U, dU, Fconv, Fvisc, Fcomm, Fcomm_temp, Ucomm, P;
    mdvector<double> norm, jaco, coord;
    mdvector<int> outnorm;
    //std::vector<double> dA, waveSp;
    mdvector<double> dA, waveSp;
    mdvector<int> LDG_bias;

#ifdef _GPU
    mdvector_gpu<double> U_d, dU_d, Fconv_d, Fvisc_d, Fcomm_d, Fcomm_temp_d, Ucomm_d, P_d;
    mdvector_gpu<double> norm_d, jaco_d, coord_d;
    mdvector_gpu<int> outnorm_d;
    mdvector_gpu<double> dA_d, waveSp_d;
    mdvector_gpu<int> LDG_bias_d;
#endif

  public:
    //Faces(GeoStruct *geo, const InputStruct *input);
    Faces(GeoStruct *geo, InputStruct *input);
    void setup(unsigned int nDims, unsigned int nVars);
    void compute_common_U();
    void compute_common_F();
    void compute_Fconv();
    void compute_Fvisc();

    
};

#endif /* faces_hpp */
