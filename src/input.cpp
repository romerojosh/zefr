#include <fstream>
#include <stdexcept>
#include <string>

#include "input.hpp"
#include "macros.hpp"

InputStruct read_input_file(std::string inputfile)
{
  std::ifstream f(inputfile);

  InputStruct input;

  read_param(f, "order", input.order);
  read_param(f, "equation", input.equation);
  read_param(f, "flux_type", input.flux_type);

  return input;
}
