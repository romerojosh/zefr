#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "input.hpp"
#include "macros.hpp"

InputStruct read_input_file(std::string inputfile)
{
  std::ifstream f(inputfile);

  InputStruct input;
  std::string str;

  read_param(f, "nDims", input.nDims);
  read_param(f, "meshfile", input.meshfile);

  read_param(f, "order", input.order);
  read_param(f, "equation", str);

  if (str == "AdvDiff")
    input.equation = AdvDiff;
  else if (str == "EulerNS")
    input.equation = EulerNS;
  else
    ThrowException("Equation not recognized!");

  read_param(f, "viscous", input.viscous);

  read_param(f, "n_steps", input.n_steps);
  read_param(f, "dt_scheme", input.dt_scheme);
  read_param(f, "dt_type", input.dt_type);
  read_param(f, "dt", input.dt);
  read_param(f, "CFL", input.CFL);

  read_param(f, "restart", input.restart);
  read_param(f, "restart_file", input.restart_file);

  read_param(f, "p_multi", input.p_multi);
  read_param(f, "smooth_steps", input.smooth_steps);
  read_param(f, "rel_fac", input.rel_fac);
  read_param(f, "low_order", input.low_order);

  read_param(f, "output_prefix", input.output_prefix);
  read_param(f, "write_freq", input.write_freq);
  read_param(f, "report_freq", input.report_freq);
  read_param(f, "force_freq", input.force_freq);
  read_param(f, "nQpts1D", input.nQpts1D);
  read_param(f, "compute_error", input.compute_error);

  read_param(f, "fconv_type", input.fconv_type);
  read_param(f, "fvisc_type", input.fvisc_type);
  read_param(f, "rus_k", input.rus_k);
  read_param(f, "ldg_b", input.ldg_b);
  read_param(f, "ldg_tau", input.ldg_tau);
  read_param(f, "spt_type", input.spt_type);

  read_param(f, "ic_type", input.ic_type);

  input.AdvDiff_A.assign({3});
  read_param(f, "AdvDiff_Ax", input.AdvDiff_A(0));
  read_param(f, "AdvDiff_Ay", input.AdvDiff_A(1));
  read_param(f, "AdvDiff_D", input.AdvDiff_D);

  read_param(f, "T_gas", input.T_gas);
  read_param(f, "gamma", input.gamma);
  read_param(f, "R", input.R);
  read_param(f, "mu", input.mu);
  read_param(f, "prandtl", input.prandtl);
  read_param(f, "prandtl", input.S);

  read_param(f, "rho_fs", input.rho_fs);
  //read_param(f, "u_fs", input.u_fs);
  //read_param(f, "v_fs", input.v_fs);
  input.V_fs.assign({3});
  read_param(f, "u_fs", input.V_fs(0));
  read_param(f, "v_fs", input.V_fs(1));
  read_param(f, "P_fs", input.P_fs);


  read_param(f, "fix_vis", input.fix_vis);
  read_param(f, "mach_fs", input.mach_fs);
  read_param(f, "Re_fs", input.Re_fs);
  read_param(f, "L_fs", input.L_fs);
  read_param(f, "T_fs", input.T_fs);
  //read_param(f, "nx_fs", input.nx_fs);
  //read_param(f, "ny_fs", input.ny_fs);
  input.norm_fs.assign({3});
  read_param(f, "nx_fs", input.norm_fs(0));
  read_param(f, "ny_fs", input.norm_fs(1));

  read_param(f, "mach_wall", input.mach_wall);
  read_param(f, "T_wall", input.T_wall);
  //read_param(f, "nx_wall", input.nx_wall);
  //read_param(f, "ny_wall", input.ny_wall);
  input.norm_wall.assign({3});
  input.V_wall.assign({3});
  read_param(f, "nx_wall", input.norm_wall(0));
  read_param(f, "ny_wall", input.norm_wall(1));

  f.close();

  /* If running Navier-Stokes, nondimensionalize */
  if (input.viscous && input.equation == EulerNS)
    apply_nondim(input);

  return input;
}

void apply_nondim(InputStruct &input)
{
  /* Compute dimensional freestream quantities */
  double V_fs_mag = input.mach_fs * std::sqrt(input.gamma * input.R * input.T_fs);
  for (unsigned int dim = 0; dim < input.nDims; dim++)
    input.V_fs(dim) = V_fs_mag * input.norm_fs(dim);

  input.rho_fs = input.mu * input.Re_fs / (V_fs_mag * input.L_fs);
  input.P_fs = input.rho_fs * input.R * input.T_fs;

  /* If using Sutherland's law, update viscosity */
  if (!input.fix_vis)
  {
    input.mu = input.mu * std::pow(input.T_fs / input.T_gas, 1.5) * ((input.T_gas + input.S)/(input.T_fs + input.S));
  }

  /* Set reference quantities for nondimensionalization */
  input.T_ref = input.T_fs;
  input.rho_ref = input.rho_fs;
  input.P_ref = input.rho_fs * V_fs_mag * V_fs_mag;
  input.mu_ref = input.rho_fs * V_fs_mag * input.L_fs;
  input.R_ref = input.R * input.T_fs / (V_fs_mag * V_fs_mag);
  input.rt = input.T_gas * input.R / (V_fs_mag * V_fs_mag);
  input.c_sth = input.S / input.T_gas;

  /* Nondimensionalize freestream quantities */
  input.mu = input.mu/input.mu_ref;
  input.rho_fs = input.rho_fs/input.rho_ref;
  for (unsigned int n = 0; n < input.nDims; n++)
    input.V_fs(n) = input.V_fs(n) / V_fs_mag;
  input.P_fs = input.P_fs / input.P_ref;
  input.T_tot_fs = (input.T_fs / input.T_ref) * (1.0 + 0.5 * (input.gamma - 1.0) * input.mach_fs * input.mach_fs);
  input.P_tot_fs = input.P_fs * std::pow(1.0 + 0.5 * (input.gamma - 1.0) * input.mach_fs * input.mach_fs, input.gamma /
      (input.gamma - 1.0));

  /* Compute and nondimensionalize wall quantities */
  double V_wall_mag = input.mach_wall * std::sqrt(input.gamma * input.R * input.T_wall);
  for (unsigned int n = 0; n < input.nDims; n++)
    input.V_wall(n) = V_wall_mag * input.norm_wall(n) / V_fs_mag;

  input.T_wall = input.T_wall / input.T_ref;
}
