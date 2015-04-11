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
  read_param(f, "dt_scheme", input.dt_scheme);
  read_param(f, "dt", input.dt);
  read_param(f, "equation", input.equation);
  read_param(f, "flux_type", input.flux_type);
  read_param(f, "spt_type", input.spt_type);
  read_param(f, "AdvDiff_Ax", input.AdvDiff_Ax);
  read_param(f, "AdvDiff_Ay", input.AdvDiff_Ay);

  f.close();

  return input;
}
