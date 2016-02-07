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
    InputStruct *input = NULL;
    GeoStruct *geo;	
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
		
		unsigned int order;
		mdvector<double> Vander, Vander_d1, Conc, Fop;
    mdvector<unsigned int> reshapeOp;
		double threshJ, Delta;
    mdvector<double> u, u_lines, uh_lines, KS_lines;
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
		void setup_reshapeOp();
    void apply_sensor_ele(unsigned int ele);

  public:
		mdvector<double> sensor;
#ifdef _GPU
    mdvector_gpu<double> sensor_d;
#endif
  
    void setup(InputStruct *input, FRSolver &solver);
		void apply_sensor();
};


#endif /* filter_hpp */