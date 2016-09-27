#ifndef shockcapture_hpp
#define shockcapture_hpp

#include <memory>

#include "mdvector.hpp"
#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#include "input.hpp"
#include "geometry.hpp"
#include "elements.hpp"
#include "faces.hpp"

/* Sensor threshold:
 * After each iteration, the filter is applied recursively till each element has a senor value less than the threshold.
 */

class FRSolver;
class ShockCapture
{
  private:
    InputStruct* input;
    GeoStruct* geo;	
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    FRSolver* solver;
	unsigned int order;

    mdvector<double> Vander, VanderInv, Vander_d1;
    mdvector<double> Vander2D, VanderND, VanderNDInv;
    mdvector<double> Conc, oppS_1D, oppS, filt, filt2;
    mdvector<double> KS, U_spts, U_filt;
    double threshJ, normalTol;

#ifdef _GPU
    mdvector_gpu<double> oppS_d, KS_d, U_spts_d, filt_d, filt2_d, U_filt_d;
    mdvector_gpu<double> Vander2DInv_tr_d, Vander2D_tr_d;
    double max_sensor_d;
#endif

    void setup_vandermonde_matrices();
    void setup_concentration_matrix();
    double calc_expfilter_coeffs(int in_mode, int type);
    void setup_expfilter_matrix();
    void setup_threshold();
    void setup_oppS();

  public:
    mdvector<double> sensor; 
    mdvector<uint> sensor_bool;
    mdvector<double> squeeze_bool; 

#ifdef _GPU
    mdvector_gpu<double> sensor_d;
    mdvector_gpu<uint> sensor_bool_d;
    mdvector_gpu<double> squeeze_bool_d;

#endif
    void setup(InputStruct *input, FRSolver &solver);
    void apply_sensor();
    void apply_expfilter();
    void apply_expfilter_type2();

};

#endif /* shockcapture_hpp */