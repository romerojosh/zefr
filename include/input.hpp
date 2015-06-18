#ifndef input_hpp
#define input_hpp

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#include "macros.hpp"

struct InputStruct
{
  unsigned int dt_type, ic_type, nDims, nQpts1D, n_steps, order, low_order, report_freq, smooth_steps, write_freq;
  std::string output_prefix, meshfile, equation, spt_type, dt_scheme, restart_file;
  bool viscous, p_multi, restart, fix_vis;
  std::string fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau; 
  double AdvDiff_D, dt, CFL, rel_fac;
  std::array<double, 3> AdvDiff_A;
  double T_gas, gamma, prandtl, mu, R, S;
  double rho_fs, u_fs, v_fs, P_fs;
  double mach_fs, L_fs, T_fs, Re_fs, nx_fs, ny_fs;
  double mach_wall, T_wall, nx_wall, ny_wall, u_wall, v_wall;
  double T_ref, P_ref, rho_ref, mu_ref, time_ref, R_ref, c_sth, rt;
};

InputStruct read_input_file(std::string inputfile);
void apply_nondim(InputStruct &input);

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

#endif /* input_hpp */
