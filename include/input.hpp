#ifndef input_hpp
#define input_hpp

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

#include "macros.hpp"

struct InputStruct
{
  unsigned int ic_type, nDims, nQpts1D, n_steps, order, report_freq, viscous, write_freq;
  std::string output_prefix, meshfile, equation, spt_type, dt_scheme;
  std::string fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau; 
  double AdvDiff_D, dt;
  std::array<double, 3> AdvDiff_A;
};

InputStruct read_input_file(std::string inputfile);

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
