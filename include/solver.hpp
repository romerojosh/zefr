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
    double prev_time = 0.;
    unsigned int nStages;
    double CFL_ratio = 1;
    mdvector<double> rk_alpha, rk_beta, rk_bhat, rk_c;
    mdvector<double> dt;
    mdvector<double> U_ini, U_til;
    mdvector<double> rk_err; //! RK Error Estimate
    Filter filt;

    /* --- Adaptive time-stepping stuff --- */
    double prev_err;        //! RK error estimate for previous step
    double expa, expb;

    /* --- Rigid-Body Motion --- */
    mdvector<double> v_ini, v_til;         //! Velocity of body CG
    mdvector<double> omega_ini, omega_til; //! Angular velocity of body
    mdvector<double> x_ini, x_til;         //! Grid node positions
    mdvector<double> q_ini, q_til; //! Grid rotation vector
    mdvector<double> qdot_ini, qdot_til; //! Grid rotation vector

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
    mdvector_gpu<double> U_til_d, rk_err_d;  //! Low-Storage Adaptive RK scheme

    mdvector_gpu<double> x_ini_d, x_til_d;
    mdvector_gpu<double> v_ini_d, v_til_d;
    mdvector_gpu<double> q_ini_d, q_til_d;
    mdvector_gpu<double> qdot_ini_d, omega_til_d;
#endif

    _mpi_comm myComm, worldComm;

    void initialize_U();
    void setup_views();
    void restart(std::string restart_file, unsigned restart_iter = 0);
    void restart_pyfr(std::string restart_file, unsigned restart_iter = 0);
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
    void setup(_mpi_comm comm_in, _mpi_comm comm_world = DEFAULT_COMM);
    void compute_residual(unsigned int stage, unsigned int color = 0);
    void add_source(unsigned int stage, unsigned int startEle, unsigned int endEle);
#ifdef _CPU
    void update(const mdvector<double> &source = mdvector<double>());

    //! Implicit Multi-Color Gauss-Seidel update loop
    void step_MCGS(const mdvector<double>& source = mdvector<double>());

    //! Standard explicit (diagonal) Runge-Kutta update loop
    void step_RK(const mdvector<double>& source = mdvector<double>());

    //! Special Low-Storage (2-register) Runge-Kutta update loop
    void step_LSRK(const mdvector<double>& source = mdvector<double>());
    void step_adaptive_LSRK(const mdvector<double>& source = mdvector<double>());
#endif
#ifdef _GPU
    void update(const mdvector_gpu<double> &source = mdvector_gpu<double>());
    void step_MCGS(const mdvector_gpu<double>& source = mdvector_gpu<double>());
    void step_RK(const mdvector_gpu<double>& source = mdvector_gpu<double>());
    void step_LSRK(const mdvector_gpu<double>& source = mdvector_gpu<double>());
    void step_adaptive_LSRK(const mdvector_gpu<double>& source = mdvector_gpu<double>());
#endif

    void write_solution(const std::string &_prefix);
    void write_solution_pyfr(const std::string &_prefix);
    void write_surfaces(const std::string &_prefix);
    void write_overset_boundary(const std::string &_prefix);
    void write_LHS(const std::string &_prefix);
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::ofstream &f);
    void report_error(std::ofstream &f);
#ifdef _GPU
    void report_gpu_mem_usage();
#endif
    void filter_solution();

    void compute_forces(std::array<double, 3>& force_conv, std::array<double, 3>& force_visc, std::ofstream* f);
    void compute_moments(std::array<double, 3>& tot_force, std::array<double, 3>& tot_moment);

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
    void rigid_body_update(unsigned int stage);
};

#endif /* solver_hpp */
