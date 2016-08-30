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

class Timer
{
private:
  std::chrono::high_resolution_clock::time_point tStart;
  std::chrono::high_resolution_clock::time_point tStop;
  double duration = 0;
  std::string prefix = "Execution Time = ";
public:

  Timer(void) {}

  Timer(std::string prefix) { this->prefix = prefix; }

  void setPrefix(std::string prefix) { this->prefix = prefix; }

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

struct InputStruct
{
  unsigned int equation, dt_type, CFL_type, ic_type, nDims, nQpts1D, n_steps, order, adapt_CFL;
  unsigned int report_freq, write_freq, force_freq, res_type, error_freq, test_case, err_field, FMG_vcycles;
  std::string output_prefix, meshfile, spt_type, dt_scheme, restart_file, mg_cycle;
  bool viscous, p_multi, restart, fix_vis, squeeze, serendipity, source;
  std::vector<unsigned int> mg_levels, mg_steps;
  std::string fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau;
  double AdvDiff_D, dt, res_tol, CFL, rel_fac, CFL_max, CFL_ratio;
  mdvector<double> AdvDiff_A, V_fs, norm_fs, V_wall, norm_wall;
  double T_gas, gamma, prandtl, mu, R, S;
  double rho_fs, u_fs, v_fs, P_fs;
  double mach_fs, L_fs, T_fs, Re_fs, nx_fs, ny_fs, T_tot_fs, P_tot_fs;
  double mach_wall, T_wall, nx_wall, ny_wall, u_wall, v_wall;
  double T_ref, P_ref, rho_ref, mu_ref, time_ref, R_ref, c_sth, rt;
  double exps0, s_factor;
  unsigned int rank, nRanks;
  unsigned int filt_on, sen_write, sen_norm, filt_maxLevels;
  double sen_Jfac, filt_gamma;
  double iter = 0, initIter = 0, time = 0, rkTime = 0;

  /* --- Overset / Moving-Grid Variables --- */
  bool motion, overset, use_lgp;
  unsigned int gridID = 0;
  unsigned int oversetMethod, nGrids, quad_order, motion_type;
  std::vector<std::string> oversetGrids;

  double moveAx, moveAy, moveAz;
  double moveFx, moveFy, moveFz;

  /* --- Additional Mesh Variables --- */
  std::map<std::string,std::string> meshBounds;

  /* Implicit Parameters */
  bool SER, inv_mode, stream_mode, backsweep, LU_pivot;
  unsigned int Jfreeze_freq, nColors, n_LHS_blocks;

  Timer waitTimer; /// DEBUGGING
#ifdef _GPU
  mdvector_gpu<double> AdvDiff_A_d, V_fs_d, norm_fs_d, V_wall_d, norm_wall_d;
#endif
};
#endif /* inputstruct_hpp */
