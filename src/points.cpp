#ifndef points_hpp
#define points_hpp

#include <cmath>
#include <vector>

#include "macros.hpp"

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
      ThrowException("Gauss_Legendre_pts supports P up to 5!");
    
  }

  return zeros;
}

std::vector<double> Gauss_Legendre_weights(unsigned int n)
{
  std::vector<double> weights(n,0.0);

  switch(n)
  {
    case 1:
      weights = {2.0}; break;

    case 2:
      weights = {1.0, 1.0}; break;

    case 3:
      weights = {5./9., 8./9., 5./9.}; break;

    case 4:
      weights = {1./36. * std::sqrt(18. - std::sqrt(30.)),
                 1./36. * std::sqrt(18. + std::sqrt(30.)),
                 1./36. * std::sqrt(18. + std::sqrt(30.)),
                 1./36. * std::sqrt(18. - std::sqrt(30.))}; break;

    case 5:
      weights = {1./900. * (322 + 13. * std::sqrt(70.)),
                 1./900. * (322 - 13. * std::sqrt(70.)),
                 128./225.,
                 1./900. * (322 - 13. * std::sqrt(70.)),
                 1./900. * (322 + 13. * std::sqrt(70.))}; break;

    default:
      ThrowException("Gauss_Legendre_weights supports up to 5 points!");

  }

  return weights;
}

std::vector<double> Shape_pts(unsigned int P)
{
  std::vector<double> nodes(P+1,0.0);

  double dx = 2.0/(double)P;

  nodes[0] = -1.0;
  for (unsigned int i = 1; i < P; i++)
    nodes[i] = nodes[i-1] + dx;

  nodes[P] = 1.0;

  return nodes; 
}


#endif /* points_hpp */
