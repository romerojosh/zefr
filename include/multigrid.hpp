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

#ifndef multigrid_hpp
#define multigrid_hpp

#include <memory>
#include <vector>
#include <chrono>
#include <fstream>

#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class PMGrid
{
  private:
    InputStruct *input = NULL;
    int order, nLevels;
    std::vector<std::map<ELE_TYPE, mdvector<double>>> correctionsBT, sourcesBT, solutionsBT;
    std::vector<std::shared_ptr<FRSolver>> grids;

#ifdef _GPU
    std::vector<std::map<ELE_TYPE, mdvector_gpu<double>>> correctionsBT_d, sourcesBT_d, solutionsBT_d;
#endif

    void restrict_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);
    void prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector<double>> &correctionBT, FRSolver &grid_f);
    void prolong_U(FRSolver &grid_c, FRSolver &grid_f);
    void compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector<double>> &sourceBT);

    /* Overloaded methods for GPU */
#ifdef _GPU
    void compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT);
    void prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector_gpu<double>> &correctionBT, FRSolver &grid_f);
#endif

  public:
    void setup(InputStruct *input, FRSolver &solver, _mpi_comm comm_in);
    void cycle(FRSolver &solver, std::ofstream& histfile, std::chrono::high_resolution_clock::time_point t1);
    void v_cycle(FRSolver &solver, int fine_order);
};

#endif /* multigrid_hpp */
