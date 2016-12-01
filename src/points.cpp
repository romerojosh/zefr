/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef points_hpp
#define points_hpp

#include <cmath>
#include <vector>

#include "macros.hpp"
#include "mdvector.hpp"

// TODO: Can get open-source scripts for generating quadrature points for arbitrary P>0
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

// TODO: Can get open-source scripts for generating quadrature weights for arbitrary P>0
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

mdvector<double> WS_Tri_pts(unsigned int P)
{
  unsigned int nPts = (P+1)*(P+2)/2;
  mdvector<double> pts({nPts, 2});

  switch(P)
  {
    case 1:
      pts(0,0) = -2./3.; pts(0,1) = -2./3.;
      pts(1,0) = 1./3.; pts(1,1) = -2./3.;
      pts(2,0) = -2./3.; pts(2,1) = 1./3.;
      break;

    case 2:
      pts(0,0) = -0.8168475729804400;
      pts(1,0) = -0.1081030181680711;
      pts(2,0) = 0.6336951459608803;
      pts(3,0) = -0.7837939636638578;
      pts(4,0) = -0.1081030181680712;
      pts(5,0) = -0.8168475729804400;

      pts(0,1) = -0.8168475729804400;
      pts(1,1) = -0.7837939636638578;
      pts(2,1) = -0.8168475729804400;
      pts(3,1) = -0.1081030181680710;
      pts(4,1) = -0.1081030181680710;
      pts(5,1) = 0.6336951459608803;
      break;

    case 3:
      pts(0,0) = -0.8888718946604130;
      pts(1,0) = -0.4089325765282142;
      pts(2,0) = 0.2684214954914466;
      pts(3,0) = 0.7777437893208270;
      pts(4,0) = -0.8594889189632320;
      pts(5,0) = -0.3333333333333333;
      pts(6,0) = 0.2684214954914465;
      pts(7,0) = -0.8594889189632320;
      pts(8,0) = -0.4089325765282144;
      pts(9,0) = -0.8888718946604140;

      pts(0,1) = -0.8888718946604140;
      pts(1,1) = -0.8594889189632320;
      pts(2,1) = -0.8594889189632320;
      pts(3,1) = -0.8888718946604140;
      pts(4,1) = -0.4089325765282145;
      pts(5,1) = -0.3333333333333333;
      pts(6,1) = -0.4089325765282145;
      pts(7,1) = 0.2684214954914466;
      pts(8,1) = 0.2684214954914466;
      pts(9,1) = 0.7777437893208271;
      break;

    case 4:
      pts(0,0) = -0.9282582446085320;
      pts(1,0) = -0.5969922362363997;
      pts(2,0) = -0.0513824244458427;
      pts(3,0) = 0.5023672622129675;
      pts(4,0) = 0.8565164892170650;
      pts(5,0) = -0.9053750259765680;
      pts(6,0) = -0.5165412084640661;
      pts(7,0) = 0.0330824169281324;
      pts(8,0) = 0.5023672622129675;
      pts(9,0) = -0.8972351511083140;
      pts(10,0) = -0.5165412084640662;
      pts(11,0) = -0.0513824244458428; 
      pts(12,0) = -0.9053750259765680;
      pts(13,0) = -0.5969922362363999;
      pts(14,0) = -0.9282582446085330;

      pts(0,1) = -0.9282582446085330;
      pts(1,1) = -0.9053750259765680;
      pts(2,1) = -0.8972351511083150;
      pts(3,1) = -0.9053750259765680;
      pts(4,1) = -0.9282582446085330;
      pts(5,1) = -0.5969922362363997;
      pts(6,1) = -0.5165412084640661;
      pts(7,1) = -0.5165412084640661;
      pts(8,1) = -0.5969922362363997;
      pts(9,1) = -0.0513824244458427;
      pts(10,1) = 0.0330824169281325;
      pts(11,1) = -0.0513824244458427;
      pts(12,1) = 0.5023672622129678;
      pts(13,1) = 0.5023672622129678;
      pts(14,1) = 0.8565164892170660;
      break;

    case 5:
      pts(0,0) = -0.9437740956346720;
      pts(1,0) = -0.7028683754582263;
      pts(2,0) = -0.2856074027686370;
      pts(3,0) = 0.2099578235502646;
      pts(4,0) = 0.6358019600569991;
      pts(5,0) = 0.8875481912693450;
      pts(6,0) = -0.9329335845987730;
      pts(7,0) = -0.6457218030613650;
      pts(8,0) = -0.1889828082651339;
      pts(9,0) = 0.2914436061227302;
      pts(10,0) = 0.6358019600569991;
      pts(11,0) = -0.9243504207816280;
      pts(12,0) = -0.6220343834697322;
      pts(13,0) = -0.1889828082651339;
      pts(14,0) = 0.2099578235502646;
      pts(15,0) = -0.9243504207816280;
      pts(16,0) = -0.6457218030613651;
      pts(17,0) = -0.2856074027686371;
      pts(18,0) = -0.9329335845987730;
      pts(19,0) = -0.7028683754582263;
      pts(20,0) = -0.9437740956346730;

      pts(0,1) = -0.9437740956346730;
      pts(1,1) = -0.9329335845987730;
      pts(2,1) = -0.9243504207816280;
      pts(3,1) = -0.9243504207816280;
      pts(4,1) = -0.9329335845987730;
      pts(5,1) = -0.9437740956346730;
      pts(6,1) = -0.7028683754582263;
      pts(7,1) = -0.6457218030613650;
      pts(8,1) = -0.6220343834697322;
      pts(9,1) = -0.6457218030613650;
      pts(10,1) = -0.7028683754582263;
      pts(11,1) = -0.2856074027686371;
      pts(12,1) = -0.1889828082651339;
      pts(13,1) = -0.1889828082651339;
      pts(14,1) = -0.2856074027686371;
      pts(15,1) = 0.2099578235502648;
      pts(16,1) = 0.2914436061227303;
      pts(17,1) = 0.2099578235502648;
      pts(18,1) = 0.6358019600569992;
      pts(19,1) = 0.6358019600569992;
      pts(20,1) = 0.8875481912693450;
      break;

    default:
      ThrowException("WS_Tri_pts undefined for provided order!");
  }

  return pts;
}


#endif /* points_hpp */
