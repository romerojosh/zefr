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

#ifndef solver_hpp
#define solver_hpp

#include <fstream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "filter.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
#endif

#ifdef _BUILD_LIB
#include "zefr.hpp"
#endif

class PMGrid;
#ifdef _BUILD_LIB
class Zefr;
#endif

class FRSolver
{
  friend class PMGrid;
  friend class Filter;
#ifdef _BUILD_LIB
  friend class Zefr;
#endif
  private:
    InputStruct *input = NULL;
    GeoStruct geo;
    int order;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    int current_iter = 0;
    int restart_iter = 0;
    double flow_time = 0.;
    unsigned int nStages;
    double CFL_ratio = 1;
    mdvector<double> rk_alpha, rk_beta;
    mdvector<double> dt;
    mdvector<double> U_ini;
    Filter filt;

    /* Implicit method parameters */
    unsigned int nCounter;
    unsigned int prev_color = 0;
    double SER_omg = 1;
    double SER_res[2] = {0};
#ifndef _NO_TNT
    std::vector<std::vector<JAMA::LU<double>>> LUptrs;
#endif

#ifdef _GPU
    mdvector_gpu<double> U_ini_d, dt_d, rk_alpha_d, rk_beta_d;
#endif

    _mpi_comm myComm;

    void initialize_U();
    void setup_views();
    void restart(std::string restart_file);
    void restart_pyfr(const std::string &restart_file);
    void setup_update();
    void setup_output();

#ifdef _GPU
    void solver_data_to_device();
#endif

    void compute_element_dt();

#ifdef _GPU
    // For moving grids
    MotionVars *motion_vars, *motion_vars_d;
#endif

  public:
    double res_max = 1;
    FRSolver(InputStruct *input, int order = -1);
    void setup(_mpi_comm comm_in);
    void compute_residual(unsigned int stage, unsigned int color = 0);
    void add_source(unsigned int stage, unsigned int startEle, unsigned int endEle);
#ifdef _CPU
    void update(const mdvector<double> &source = mdvector<double>());
#endif
#ifdef _GPU
    void update(const mdvector_gpu<double> &source = mdvector_gpu<double>());
#endif
    void write_solution(const std::string &_prefix);
    void write_solution_pyfr(const std::string &_prefix);
    void write_surfaces(const std::string &_prefix);
    void write_overset_boundary(const std::string &_prefix);
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::ofstream &f);
    void report_error(std::ofstream &f);
    void filter_solution();

    /* Routines for implicit method */
    void compute_LHS();
    void compute_LHS_LU(unsigned int startEle, unsigned int endEle, unsigned int color = 1);
    void compute_RHS(unsigned int color = 1);
#ifdef _CPU
    void compute_RHS_source(const mdvector<double> &source, unsigned int color = 1);
#endif
#ifdef _GPU
    void compute_RHS_source(const mdvector_gpu<double> &source, unsigned int color = 1);
#endif
    void compute_deltaU(unsigned int color = 1);
    void compute_U(unsigned int color = 1);
    void dFcdU_from_faces();
    void compute_SER_dt();
    void write_color();

#ifdef _BUILD_LIB
    Zefr *ZEFR;
#endif

    void move(double time);
};

#endif /* solver_hpp */
