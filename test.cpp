#include <iostream>

#include "faces.hpp"
#include "quads.hpp"
#include "input.hpp"


int main()
{
  auto input = read_input_file("input.txt");

  Quads eles(1,  4 , &input);
  Faces faces(8, &input);

  eles.associate_faces(&faces);
  eles.setup();

}
