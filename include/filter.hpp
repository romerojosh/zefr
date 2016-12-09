/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

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

    mdvector<double> Vander, VanderInv, Vander_d1;
    mdvector<double> Vander2D, VanderND, VanderNDInv;
    mdvector<double> Conc, oppS_1D, oppS, filt, filt2;
    mdvector<double> KS, U_spts, U_filt;
    double threshJ, normalTol;

    // Stuff for Kartikey's version
    std::vector<mdvector<double>> oppF_1D, oppF_spts, oppF_fpts;
    std::vector<double> DeltaHat;

#ifdef _GPU
    mdvector_gpu<double> oppS_d, KS_d, U_spts_d;
    mdvector_gpu<double> filt_d, filt2_d, U_filt_d;
    mdvector_gpu<double> Vander2DInv_tr_d, Vander2D_tr_d;
    std::vector<mdvector_gpu<double>> oppF_spts_d, oppF_fpts_d; // For Kartikey's version
    double max_sensor_d;
#endif
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
    double calc_expfilter_coeffs(int in_mode, int type);
    void setup_expfilter_matrix();
    void setup_threshold();
    void setup_oppS();

    // Functions for Kartikey's version
    void setup_DeltaHat(unsigned int level);
    void setup_oppF_1D(unsigned int level);
    void setup_oppF(unsigned int level);

  public:
    mdvector<double> sensor;
    mdvector<unsigned int> sensor_bool;

#ifdef _GPU
    mdvector_gpu<double> sensor_d;
    mdvector_gpu<unsigned int> sensor_bool_d;
#endif

    void setup(InputStruct *input, FRSolver &solver);

    //! Apply the concentration sensor to the solution
    void apply_sensor();

    //! Apply Kartikey's version of the filter
    unsigned int apply_filter(unsigned int level);

    //! Apply Abhishek's version of the filter
    void apply_expfilter();
    void apply_expfilter_type2();
};


#endif /* filter_hpp */
