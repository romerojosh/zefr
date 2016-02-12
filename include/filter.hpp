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

/* Sensor threshold:
 * After each iteration, the filter is applied recursively till each element has a senor value less than the threshold.
 * 
 * Filter width:
 * For each inner iteration, the troubled cells are filtered using a fixed filter width.
 */


class Filter
{
  private:
    InputStruct* input;
    GeoStruct* geo;	
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    FRSolver* solver;
		unsigned int order;

    mdvector<double> Vander, Vander_d1, Conc;
    mdvector<unsigned int> reshapeOp;
    double threshJ;
    mdvector<unsigned int> appendOp;
    std::vector<mdvector<double>> Fop;
		std::vector<double> DeltaHat;
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
		void setup_reshapeOp();
    void setup_DeltaHat(unsigned int level);
    void setup_Fop(unsigned int level);
    void setup_appendOp();
    double apply_sensor(unsigned int ele, unsigned int var);
    unsigned int apply_filter(unsigned int ele, unsigned int var, unsigned int level);

  public:
		mdvector<double> sensor;
#ifdef _GPU
    mdvector_gpu<double> sensor_d;
#endif
  
    void setup(InputStruct *input, FRSolver &solver);
		void apply_sensor();
    unsigned int apply_filter(unsigned int level);
};


#endif /* filter_hpp */