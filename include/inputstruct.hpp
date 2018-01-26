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

#ifndef inputstruct_hpp
#define inputstruct_hpp

#include <chrono>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _MPI
#include "mpi.h"
#endif

#include "mdvector.hpp"
#ifdef _GPU
#include "mdvector_gpu.h"
#endif

enum MOTION_TYPE {
  STATIC = 0,
  TEST1 = 1,
  TEST2 = 2,
  TEST3 = 3,
  CIRCULAR_TRANS = 4,
  RADIAL_VIBE = 5,
  RIGID_BODY = 10
};

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point tStart;
  std::chrono::high_resolution_clock::time_point tStop;
  double duration = 0;
  std::string prefix = "Execution Time = ";
public:

  Timer(void) {}

  Timer(const std::string &prefix) { this->prefix = prefix; }
  Timer(const char *prefix) { this->prefix = prefix; }

  void setPrefix(const std::string &prefix) { this->prefix = prefix; }
  void setPrefix(const char *prefix) { this->prefix = prefix; }

  void startTimer(void)
  {
    tStart = std::chrono::high_resolution_clock::now();
  }

  void stopTimer(void)
  {
    tStop = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>( tStop - tStart ).count();
    duration += (double).001*elapsed;
  }

  void resetTimer(void)
  {
    duration = 0;
    tStart = std::chrono::high_resolution_clock::now();
  }

  double getTime(void)
  {
    return .001*duration;
  }

  void showTime(int precision = 2)
  {
    int rank = 0;
#ifdef _MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    double seconds = .001 * duration;
    if (seconds > 60) {
      int minutes = floor(seconds/60);
      seconds = seconds - (minutes*60);
      std::cout << "Rank " << rank << ": " << prefix << minutes << "min ";
      std::cout << std::setprecision(precision) << seconds << "s" << std::endl;
    }
    else
    {
      std::cout << "Rank " << rank << ": " << prefix;
      std::cout << std::setprecision(precision) << seconds << "s" << std::endl;
    }
  }
};

/*! Structure to make useful simulation data easily available 
 *  from Python interface layer */
struct DataStruct
{
  double forces[6];

  int nfields;
  int nspts[4];  // By element type
  double* u_spts[4] = {NULL};
  double* du_spts[4] = {NULL};
#ifdef _GPU
  double* u_spts_d[4];
  double* du_spts_d[4];
#endif
};

struct InputStruct
{
  unsigned int equation, dt_type, CFL_type = 0, ic_type, nDims, nQpts1D, n_steps, order;
  unsigned int report_freq, write_freq, force_freq, turb_stat_freq, res_type, error_freq, test_case, err_field, res_field, FMG_vcycles;
  std::string output_prefix, meshfile, spt_type, dt_scheme, restart_file, mg_cycle;
  bool viscous, p_multi, restart, fix_vis, squeeze, source, grad_via_div, disable_nondim, adapt_dt;
  std::vector<unsigned int> mg_levels, mg_steps;
  unsigned int fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau;
  double AdvDiff_D, dt, res_tol, CFL, rel_fac;
  mdvector<double> AdvDiff_A, V_fs, norm_fs, V_wall, norm_wall;
  double T_gas, gamma, prandtl, mu, R, S;
  double rho_fs, u_fs, v_mag_fs, P_fs;
  double mach_fs, L_fs, T_fs, Re_fs, nx_fs, ny_fs, T_tot_fs, P_tot_fs;
  double mach_wall, T_wall, nx_wall, ny_wall, u_wall, v_wall;
  double T_ref, P_ref, rho_ref, mu_ref, time_ref, R_ref, c_sth, rt;
  double exps0, s_factor;
  unsigned int rank, nRanks, grank;
  unsigned int filt_on, sen_write, sen_norm, filt_maxLevels;
  unsigned int shockcapture, limiter, filt2on;
  double sen_Jfac, filt_gamma;
  double alpha, filtexp, nonlin_exp, filtexp2, alpha2;
  unsigned int iter = 0, initIter = 0;
  double time, tfinal;

  /* --- Adaptive time-stepping --- */
  unsigned int nStages;
  double pi_alpha, pi_beta;     //! PI-controller valuse
  double sfact, maxfac, minfac; //! delta-t adjustment factors
  double max_dt;
  double atol, rtol;

  /* --- I/O --- */
  short write_paraview, write_pyfr, plot_surfaces, plot_overset;
  unsigned int restart_iter, restart_type;
  std::string restart_case;
  bool catch_signals = false;
  bool write_LHS, write_RHS;
  bool tavg;  //! Collect time-averaged quantities
  unsigned int tavg_freq; //! Frequency at which to accumulate time-averaged quantities
  unsigned int write_tavg_freq; //! Frequency at which to dump time-averaged quantities

  /* --- Overset / Moving-Grid Variables --- */
  bool motion, overset, use_lgp, full_6dof;
  unsigned int gridID = 0;
  unsigned int nGrids, gridType, motion_type;
  std::vector<std::string> oversetGrids;  //! List of mesh files for each overset grid
  std::vector<int> gridTypes;             //! Overset grid types (background or geometry-containing)

  double moveAx, moveAy, moveAz;
  double moveFx, moveFy, moveFz;

  double rot_axis[3];
  double rot_angle;
  double xc[3], dxc[3], vc[3], dvc[3];

  double g;       //! Gravitational acceleration (default to 0 / no gravity)
  double v0[3];   //! Initial grid translational velocity
  double w0[3];   //! Initial grid angular velocity
  double mass;    //! Mass of moving body of interest
  double Imat[9]; //! Inertia tensor of input geometry of interest

  /* --- Additional Mesh Variables --- */
  std::map<std::string,std::string> meshBounds; //! Mapping from mesh-file names to Zefr BC's

  /* Implicit Parameters */
  bool implicit_method = false, implicit_steady = false;
  bool pseudo_time, adapt_dtau, FDA_Jacobian, KPF_Jacobian, freeze_Jacobian, remove_deltaU;
  double dtau, CFL_tau, dtau_ratio_max, dtau_growth_rate;
  unsigned int dtau_type, CFL_tau_type = 0;
  unsigned int iterative_method = 0, linear_solver = 0;
  unsigned int nColors, iterNM_max, report_NMconv_freq, iterBM_max, report_BMconv_freq;
  bool backsweep;
  double svd_omg, svd_cutoff;

  Timer waitTimer; /// DEBUGGING
#ifdef _GPU
  mdvector_gpu<double> AdvDiff_A_d, V_fs_d, norm_fs_d, V_wall_d, norm_wall_d;
#endif
};
#endif /* inputstruct_hpp */
