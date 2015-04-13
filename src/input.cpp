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
  read_param(f, "dt", input.dt);

  read_param(f, "write_freq", input.write_freq);
  read_param(f, "report_freq", input.report_freq);

  read_param(f, "fconv_type", input.fconv_type);
  read_param(f, "fvisc_type", input.fvisc_type);
  read_param(f, "rus_k", input.rus_k);
  read_param(f, "ldg_b", input.ldg_b);
  read_param(f, "ldg_tau", input.ldg_tau);
  read_param(f, "spt_type", input.spt_type);

  read_param(f, "AdvDiff_Ax", input.AdvDiff_Ax);
  read_param(f, "AdvDiff_Ay", input.AdvDiff_Ay);
  read_param(f, "AdvDiff_D", input.AdvDiff_D);

  f.close();

  return input;
}
