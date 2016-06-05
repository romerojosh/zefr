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

    case 6:
      zeros = { -0.932469514203152,
                -0.661209386466264,
                -0.238619186083197,
                0.238619186083197,
                0.661209386466264,
                0.932469514203152}; break;
      
    case 7:
      zeros = { -0.949107912342758,
                -0.741531185599394,
                -0.405845151377397,
                0,
                0.405845151377397,
                0.741531185599394,
                0.949107912342758}; break;
      
    case 8:
      zeros = { -0.960289856497536,
                -0.796666477413627,
                -0.525532409916329,
                -0.183434642495650,
                0.183434642495650,
                0.525532409916329,
                0.796666477413627,
                0.960289856497536}; break;

    case 9:
      zeros = { -0.968160239507626,
                -0.836031107326636,
                -0.613371432700590,
                -0.324253423403809,
                0.0,
                0.324253423403809,
                0.613371432700590,
                0.836031107326636,    
                0.968160239507626}; break;

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
      ThrowException("Gauss_Legendre_pts supports P up to 10!");
    
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

    case 6:
      weights = {0.171324492379171,
                 0.360761573048139,
                 0.467913934572691,
                 0.467913934572691,
                 0.360761573048139,
                 0.171324492379171}; break;
      
    case 7:
      weights = {0.129484966168870,
                 0.279705391489277,
                 0.381830050505119,
                 0.417959183673469,
                 0.381830050505119,
                 0.279705391489277,
                 0.129484966168870}; break;
      
    case 8:
      weights = {0.101228536290377,
                 0.222381034453374,
                 0.313706645877887,
                 0.362683783378362,
                 0.362683783378362,
                 0.313706645877887,
                 0.222381034453374,
                 0.101228536290377}; break;
      
    case 9:
      weights = {0.081274388361575,
                 0.180648160694857,
                 0.260610696402936,
                 0.312347077040003,
                 0.330239355001260,
                 0.312347077040003,
                 0.260610696402936,
                 0.180648160694857,
                 0.081274388361575}; break;
      
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
      ThrowException("Gauss_Legendre_weights supports up to 10 points!");
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
