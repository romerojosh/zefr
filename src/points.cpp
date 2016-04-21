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

    case 10:
      zeros = {-0.973906528517172, 
               -0.865063366688985, 
               -0.679409568299024, 
               -0.433895394129247, 
               -0.148874338981631, 
               0.148874338981631,
               0.433895394129247, 
               0.679409568299024, 
               0.865063366688985, 
               0.973906528517172}; break;

    default:
      ThrowException("Gauss_Legendre_pts supports P up to 5!");
    
  }

  return zeros;
}

std::vector<double> DFRsp_pts(unsigned int P, double z1)
{
  std::vector<double> zeros(P,0.0);

  switch(P)
  {
    case 4:
      zeros = {-std::sqrt((3. - 5. * z1*z1)/(5. - 15.* z1 * z1)), 
               -z1,
                z1, 
                std::sqrt((3. - 5. * z1*z1)/(5. - 15.* z1 * z1))};
      break;

    default:
      ThrowException("DFRsp_pts supports limited cases right now!");
    
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
      weights = {1./36. * (18. - std::sqrt(30.)),
                 1./36. * (18. + std::sqrt(30.)),
                 1./36. * (18. + std::sqrt(30.)),
                 1./36. * (18. - std::sqrt(30.))}; break;

    case 5:
      weights = {1./900. * (322 - 13. * std::sqrt(70.)),
                 1./900. * (322 + 13. * std::sqrt(70.)),
                 128./225.,
                 1./900. * (322 + 13. * std::sqrt(70.)),
                 1./900. * (322 - 13. * std::sqrt(70.))}; break;
    case 10:
      weights = {0.066671344308688, 
                 0.149451349150581,
                 0.219086362515982,
                 0.269266719309996, 
                 0.295524224714753, 
                 0.295524224714753, 
                 0.269266719309996, 
                 0.219086362515982, 
                 0.149451349150581,
                 0.066671344308688}; break;

    default:
      ThrowException("Gauss_Legendre_weights supports up to 5 points!");

  }

  return weights;
}

std::vector<double> Shape_pts(unsigned int P)
{
  std::vector<double> nodes(P+1, 0.0);

  double dx = 2.0/(double)P;

  nodes[0] = -1.0;
  for (unsigned int i = 1; i < P; i++)
    nodes[i] = nodes[i-1] + dx;

  nodes[P] = 1.0;

  return nodes; 
}


#endif /* points_hpp */
