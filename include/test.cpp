#include "mdvector.hpp"
#include <iostream>

int main()
{
  mdvector<int> a({100,100});
  mdvector<int> b({100,100});

  std::cout << a(1,100);


}
