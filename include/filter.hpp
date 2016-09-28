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

    mdvector<double> Vander, VanderInv, Vander_d1, Conc, oppS_1D, oppS;
    mdvector<double> KS, U_spts;
    double threshJ, normalTol;
    std::vector<mdvector<double>> oppF_1D, oppF_spts, oppF_fpts;
    std::vector<double> DeltaHat;

#ifdef _GPU
    mdvector_gpu<double> oppS_d, KS_d, U_spts_d;
    std::vector<mdvector_gpu<double>> oppF_spts_d, oppF_fpts_d;
    double max_sensor_d;
#endif
	
		void setup_vandermonde_matrices();
		void setup_concentration_matrix();
		void setup_threshold();
    void setup_oppS();
    void setup_DeltaHat(unsigned int level);
    void setup_oppF_1D(unsigned int level);
    void setup_oppF(unsigned int level);

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
