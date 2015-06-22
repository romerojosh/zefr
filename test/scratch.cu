#include <iostream>

#include "cuda_runtime.h"

#include "mdvector.hpp"
#include "mdvector_gpu.h"


__global__
void test_access(mdvector_gpu<double> vec, double val)
{
  vec(1,1) = val;
  vec(2,1) = val;
  vec(1,2) = val;
}

int main()
{
  /* Create some data */
  mdvector<double> host_data({3,3},3.4,0);
  mdvector_gpu<double> dev_data;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      std::cout << host_data(i,j) << " ";
    std::cout << std::endl;
  }

  dev_data = host_data;

  test_access<<<1,1>>>(dev_data, 1.2);

  host_data = dev_data;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
      std::cout << host_data(i,j) << " ";
    std::cout << std::endl;
  }


  dev_data.free_data();

  return 0;
}
