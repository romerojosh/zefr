#ifndef filter_hpp
#define filter_hpp

#include <memory>

#include "mdvector.hpp"
#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#include "input.hpp"

class FRSolver;
class Filter
{
  private:
    InputStruct* input;
    FRSolver* solver;

    std::map<ELE_TYPE, mdvector<double>> KS, U_ini, U_filt;
    std::map<ELE_TYPE, double> threshJ;

#ifdef _GPU
    std::map<ELE_TYPE, mdvector_gpu<double>> KS_d, U_ini_d, U_filt_d;
    double max_sensor_d;
#endif
	
    void setup_threshold();

  public:
    std::map<ELE_TYPE, mdvector<double>> sensor;
    std::map<ELE_TYPE, mdvector<unsigned int>> sensor_bool;

#ifdef _GPU
    std::map<ELE_TYPE, mdvector_gpu<double>> sensor_d;
    std::map<ELE_TYPE, mdvector_gpu<unsigned int>> sensor_bool_d;
#endif

    void setup(InputStruct *input, FRSolver &solver);

    //! Apply the concentration sensor to the solution
    void apply_sensor();

    //! Apply exponential filter
    void apply_expfilter();
};

double calc_expfilter_coeffs(unsigned int P, unsigned int nModes, double alpha, double s, unsigned int nDims, int mode);

#endif /* filter_hpp */
