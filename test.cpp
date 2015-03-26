#include <iostream>

#include "quads.hpp"
#include "input.hpp"


int main()
{
  auto input = read_input_file("input.txt");

  Quads eles(10,  1 , &input);
  eles.setup();

}
