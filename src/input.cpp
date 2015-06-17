#include <fstream>
#include <stdexcept>
#include <string>

#include "input.hpp"
#include "macros.hpp"

InputStruct read_input_file(std::string inputfile)
{
  std::ifstream f(inputfile);

  InputStruct input;

  read_param(f, "nDims", input.nDims);
  read_param(f, "meshfile", input.meshfile);

  read_param(f, "order", input.order);
  read_param(f, "equation", input.equation);
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
  read_param(f, "nQpts1D", input.nQpts1D);

  read_param(f, "fconv_type", input.fconv_type);
  read_param(f, "fvisc_type", input.fvisc_type);
  read_param(f, "rus_k", input.rus_k);
  read_param(f, "ldg_b", input.ldg_b);
  read_param(f, "ldg_tau", input.ldg_tau);
  read_param(f, "spt_type", input.spt_type);

  read_param(f, "ic_type", input.ic_type);

  read_param(f, "AdvDiff_Ax", input.AdvDiff_A[0]);
  read_param(f, "AdvDiff_Ay", input.AdvDiff_A[1]);
  read_param(f, "AdvDiff_D", input.AdvDiff_D);

  read_param(f, "gamma", input.gamma);
  read_param(f, "rho_fs", input.rho_fs);
  read_param(f, "u_fs", input.u_fs);
  read_param(f, "v_fs", input.v_fs);
  read_param(f, "P_fs", input.P_fs);

  f.close();

  return input;
}
