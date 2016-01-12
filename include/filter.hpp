#ifndef filter_hpp
#define filter_hpp

#include "mdvector.hpp"
#include "input.hpp"
#include "geometry.hpp"
#include "elements.hpp"
#include "faces.hpp"

class Filter
{
  private:
    InputStruct *input;
    GeoStruct *geo;	
    Elements *eles;
    Faces *faces;
		
		unsigned int order;
		mdvector<double> Vander, Vander_d1, Conc, Fop;
    mdvector<double> reshapeOp;
		double threshJ, Delta;
    mdvector<double> u, u_lines, uh_lines, KS_lines;
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
		void setup_reshapeOp();
    void apply_sensor_ele(unsigned int ele, unsigned int var);
		void apply_filter_1D();

    void U_to_faces();
    void U_from_faces();
	
  public:
		mdvector<double> sensor;
#ifdef _GPU
    mdvector_gpu<double> sensor_d;
#endif
  
    Filter(InputStruct *input = NULL, GeoStruct *geo = NULL, Elements *eles = NULL, Faces *faces = NULL);
		void setup();
		void apply_sensor();
		void apply_filter();
};


#endif /* filter_hpp */