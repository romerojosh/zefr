#ifndef points_hpp
#define points_hpp

#include <cmath>
#include <stdexcept>
#include <vector>

std::vector<double> Gauss_Legendre_pts(unsigned int P)
{
  std::vector<double> zeros(P,0.0);

  switch(P)
  {
    case 1:
      zeros = {0.0}; break;

    case 2:
      zeros = {-std::sqrt(1./3.),
                std::sqrt(1./3.)}; break;

    case 3:
      zeros = {-std::sqrt(3./5.), 
               0.0, 
               std::sqrt(3./5.)}; break;

    case 4:
      zeros = {-std::sqrt((3.+2.*std::sqrt(6./5.))/7.), 
               -std::sqrt((3.-2.*std::sqrt(6./5.))/7.),
                std::sqrt((3.-2.*std::sqrt(6./5.))/7.), 
                std::sqrt((3.+2.*std::sqrt(6./5.))/7.)}; break;

    case 5:
      zeros = {-(1./3.)*std::sqrt(5.+2.*std::sqrt(10./7.)), 
               -(1./3.)*std::sqrt(5.-2.*std::sqrt(10./7.)),
                0.0,
                (1./3.)*std::sqrt(5.-2.*std::sqrt(10./7.)), 
                (1./3.)*std::sqrt(5.+2.*std::sqrt(10./7.))}; break;

    default:
      throw std::runtime_error("Gauss_Legendre_pts supports P up to 5!");
    
  }

  return zeros;
}

std::vector<double> Shape_pts(unsigned int P)
{
  std::vector<double> pts(P+1,0.0);
  switch (P)
  {
    case 1:
      pts = {-1.0, 1.0}; break;

    case 2:
      pts = {-1.0, 0.0, 1.0}; break;

    case 3:
      pts = {-1.0, -1./3., 1./3., 1.0}; break;

    default:
      throw std::runtime_error("Shape_pts supports up to cubic edges only!");
  }

  return pts; 
}

#endif /* points_hpp */
