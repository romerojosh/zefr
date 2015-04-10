#ifndef input_hpp
#define input_hpp

#include <fstream>
#include <stdexcept>
#include <string>

#include "macros.hpp"

struct InputStruct
{
  unsigned int order;
  std::string meshfile, equation, flux_type, spt_type;
  double AdvDiff_Ax, AdvDiff_Ay;
};

InputStruct read_input_file(std::string inputfile);

template <typename T>
void read_param(std::ifstream &f, std::string name, T &var)
{
  if (!f.is_open())
  {
    //throw std::runtime_error("Input file not open for reading!");
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
