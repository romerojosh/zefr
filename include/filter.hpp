#ifndef filter_hpp
#define filter_hpp

#include <memory>

#include "mdvector.hpp"
#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#include "input.hpp"
#include "geometry.hpp"
#include "elements.hpp"
#include "faces.hpp"
#include "solver.hpp"

class Filter
{
  private:
    InputStruct* input;
    GeoStruct* geo;	
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    FRSolver* solver;
		unsigned int order;

    mdvector<double> Vander, Vander_d1, Conc, Fop;
    mdvector<unsigned int> reshapeOp, appendOp;
		double threshJ, DeltaHat;
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
		void setup_reshapeOp();
    void setup_DeltaHat();
    void setup_Fop();
    void setup_appendOp();
    double apply_sensor(unsigned int ele, unsigned int var);
    void apply_filter(unsigned int ele, unsigned int var);

  public:
		mdvector<double> sensor;
#ifdef _GPU
    mdvector_gpu<double> sensor_d;
#endif
  
    void setup(InputStruct *input, FRSolver &solver);
		void apply_sensor();
    void apply_filter();
};


#endif /* filter_hpp */