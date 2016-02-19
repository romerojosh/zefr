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

/* Sensor threshold:
 * After each iteration, the filter is applied recursively till each element has a senor value less than the threshold.
 * 
 * Filter width:
 * For each inner iteration, the troubled cells are filtered using a fixed filter width.
 */

class FRSolver;
class Filter
{
  private:
    InputStruct* input;
    GeoStruct* geo;	
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    FRSolver* solver;
		unsigned int order;

    mdvector<double> Vander, VanderInv, Vander_d1, Conc, oppS_1D, oppS;
    mdvector<double> KS, U_spts;
    double threshJ, normalTol;
    std::vector<mdvector<double>> oppF_1D, oppF_spts, oppF_fpts;
    std::vector<double> DeltaHat;
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
    void setup_oppS();
    void setup_DeltaHat(unsigned int level);
    void setup_oppF_1D(unsigned int level);
    void setup_oppF(unsigned int level);

  public:
		mdvector<double> sensor; 
    void setup(InputStruct *input, FRSolver &solver);
		void apply_sensor();
    unsigned int apply_filter(unsigned int level);
};


#endif /* filter_hpp */