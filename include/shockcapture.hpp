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

    mdvector<double> Vander, VanderInv, Vander_d1, Vander2D_tr, Vander2DInv_tr, Conc, oppS_1D, oppS;
    mdvector<double> KS, U_spts;
    double threshJ, normalTol;

#ifdef _GPU
    mdvector_gpu<double> oppS_d, KS_d, U_spts_d, Vander2DInv_tr_d, Vander2D_tr_d;
    double max_sensor_d;
#endif

    void setup_vandermonde_matrices();
    void setup_concentration_matrix();
    void setup_threshold();
    void setup_oppS();
    void bring_to_square(uint ele, uint var, double Ulow, double Uhigh);

  public:
    mdvector<double> sensor; 
#ifdef _GPU
    mdvector_gpu<double> sensor_d;
#endif
    void setup(InputStruct *input, FRSolver &solver);
    void apply_sensor();
    void compute_Umodal();
    void compute_Unodal();
    void limiter();

};


#endif /* shockcapture_hpp */