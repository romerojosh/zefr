#ifndef input_hpp
#define input_hpp

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mdvector.hpp"
#include "macros.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

enum EQN {AdvDiff = 0, EulerNS = 1};

struct InputStruct
{
  unsigned int equation, dt_type, ic_type, nDims, nQpts1D, n_steps, order, low_order, smooth_steps;
  unsigned int report_freq, write_freq, force_freq, res_type, error_freq, test_case, err_field;
  std::string output_prefix, meshfile, spt_type, dt_scheme, restart_file;
  bool viscous, p_multi, restart, fix_vis, squeeze, serendipity;
  std::string fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau; 
  double AdvDiff_D, dt, CFL, rel_fac;
  mdvector<double> AdvDiff_A, V_fs, norm_fs, V_wall, norm_wall;
  double T_gas, gamma, prandtl, mu, R, S;
  double rho_fs, u_fs, v_fs, P_fs;
  double mach_fs, L_fs, T_fs, Re_fs, nx_fs, ny_fs, T_tot_fs, P_tot_fs;
  double mach_wall, T_wall, nx_wall, ny_wall, u_wall, v_wall;
  double T_ref, P_ref, rho_ref, mu_ref, time_ref, R_ref, c_sth, rt;
  double exps0, s_factor;
  unsigned int rank, nRanks;

#ifdef _GPU
  mdvector_gpu<double> AdvDiff_A_d, V_fs_d, norm_fs_d, V_wall_d, norm_wall_d;
#endif
};

InputStruct read_input_file(std::string inputfile);
void apply_nondim(InputStruct &input);

/* Function to read parameter from input file. Throws exception if parameter 
 * is not found. */
template <typename T>
void read_param(std::ifstream &f, std::string name, T &var)
{
  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  std::string param;

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    if (param == name)
    {
      f >> var;
      return;
    }
  }

  ThrowException("Input parameter " + name + " not found!");

}

/* Function to read parameter from input file. Sets var to provided default
 * value if parameter is not found. */
template <typename T>
void read_param(std::ifstream &f, std::string name, T &var, T default_val)
{
  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  std::string param;

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    if (param == name)
    {
      f >> var;
      return;
    }
  }

  var = default_val;
}

#endif /* input_hpp */
