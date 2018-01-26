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

/*!
 * \file funcs.cpp
 * \author J. Romero, Stanford University
 * \brief This file contains several miscellaneous functions used in the code.
 */

#include <cmath>

extern "C" {
#include "cblas.h"
}

#include "funcs.hpp"
#include "input.hpp"

//! Storing Gmsh to structured-IJK index mappings to avoid re-computing
static std::map<int, std::vector<int>> gmsh_maps_hex;
static std::map<int, std::vector<int>> gmsh_maps_quad;
static std::map<int, std::vector<int>> ijk_maps_quad;
static std::map<int, std::vector<int>> ijk_maps_hex;

static std::map<int, std::vector<int>> gmsh_maps_tet;
static std::map<int, std::vector<int>> gmsh_maps_tri;
static std::map<int, std::vector<int>> ijk_maps_tri;
static std::map<int, std::vector<int>> ijk_maps_tet;

static std::map<int, std::vector<int>> gmsh_maps_pri;
static std::map<int, std::vector<int>> ijk_maps_pri;

double compute_U_init(double x, double y, double z, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      
      val = std::sin(M_PI * x) * 
            std::sin(M_PI * y);
      
      //val =  std::sin(2 * M_PI * x/10.) + std::sin(2 * M_PI * y/10.);
      //val = std::exp(20.*(-x*x-y*y));
      //val =  step((x - input->AdvDiff_A(0) * t));
    }
    else if (input->nDims == 3)
    {
      
      val = std::sin(M_PI * x) *
            std::sin(M_PI * y) *
            std::sin(M_PI * z);
      
      //val =  std::sin(2 * M_PI * x) + std::sin(2 * M_PI * y) + std::sin(2 * M_PI * z);
    }
  }
  else if (input->equation == EulerNS)
  {
    if (input->test_case == 1 || input->test_case == 3)
    {
      double rho, Vx, Vy, P;
      if (input->test_case == 1)
      {
        double G = 5.0;
        double R = 1.;

        double f = (1.0 - x*x - y*y)/R;

        rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
                  M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
        Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f);
        Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
        P = std::pow(rho, input->gamma);
      }
      else if (input->test_case == 3)
      {
        /* Vincent et al. Isentropic Euler Vortex */
        double D = 20.0; // Domain size (e.g. [-D,D] x [-D,D] domain)
        double G = 13.5; // Vortex strength
        double M = 0.4;  // Mach number
        double R = 1.5;  // Vortex radius

        double omg = G / (2.0*M_PI*R);
        double f = std::exp((1.0 - x*x - y*y) / (2.0*R*R));

        rho = std::pow(1.0 - 0.5*(input->gamma-1.0)*omg*omg*M*M*R*R*f*f, 
            1.0/(input->gamma-1.0));
        Vx = -omg*f*y;
        Vy = 1.0 + omg*f*x;
        P = 1.0 / (input->gamma*M*M) * std::pow(rho, input->gamma); // Unit entropy
      }

      if (input->nDims == 2)
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * Vx; break;
          case 2:
            val = rho * Vy; break;
          case 3:
            val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
        }
      }
      else
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * Vx; break;
          case 2:
            val = rho * Vy; break;
          case 3:
            val = 0; break;
          case 4:
            val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
        }
      }
    }
    else if (input->test_case == 4)
    {
      /* Taylor-Green Vortex Test Case */
      if (input->nDims == 2)
        ThrowException("Taylor-Green Vortex test case for 3D cases only");

      double L = input->L_fs;
      double U0 = input->v_mag_fs;
      double Rho0 = input->rho_fs;
      double P0 = input->P_fs;

      double u =  U0 * sin(x/L) * cos(y/L) * cos(z/L);
      double v = -U0 * cos(x/L) * sin(y/L) * cos(z/L);

      double p = P0 + Rho0*U0*U0/16. * (cos(2*x/L) + cos(2*y/L)) * (cos(2*z/L) + 2);
      double rho = p / (input->R * input->T_fs);

      switch (var)
      {
        case 0:
          val = rho; break;
        case 1:
          val = rho * u; break;
        case 2:
          val = rho * v; break;
        case 3:
          val = 0.0; break;
        case 4:
          val = p/(input->gamma - 1.0) + 0.5 * rho * (u * u + v * v); break;
      }
    }
    else if (input->test_case == 2)
    {
      /* Couette flow test case */
      double gamma = input->gamma;
      double Pr = input->prandtl;
      double P = input->P_fs;
      double R = input->R;
      double Vw = input->V_wall(0);
      double Tw = input->T_wall;
      double cp = gamma * R / (gamma - 1);

      double rho = gamma / (gamma - 1) * (2 * P)/(2*cp*Tw + Pr*Vw*Vw * y * (1-y));
      double rho_avg = 4.*(P * std::sqrt(Pr*(Pr*Vw*Vw+8*cp*Tw))*std::log((std::sqrt(Pr*(Pr*Vw*Vw+8*cp*Tw))+Pr*Vw)/
            (std::sqrt(Pr*(Pr*Vw*Vw+8*cp*Tw))-Pr*Vw))*gamma)/(Pr*Vw*(Pr*Vw*Vw+8*cp*Tw)*(gamma-1));

      if (input->nDims == 2)
      {
        switch (var)
        {
          case 0:
            val = rho_avg; break;
          case 1:
            val = rho_avg * Vw; break;
          case 2:
            val = 0.0; break;
          case 3:
            val = P / (gamma - 1) + 0.5 * rho_avg * Vw * Vw; break;
        }
      }
      else
      {
        switch (var)
        {
          case 0:
            val = rho_avg; break;
          case 1:
            val = rho_avg * Vw; break;
          case 2:
            val = 0.0; break;
          case 3:
            val = 0.0; break;
          case 4:
            val = P / (gamma - 1) + 0.5 * rho_avg * Vw * Vw; break;
        }
      }

    }
    else
    {
      ThrowException("Unknown test case ID!");
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}

double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))* 
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t));
      
      //val =  std::sin(2 * M_PI * x/10.) + std::sin(2 * M_PI * y/10.);
      //val = std::exp(-x*x-y*y);
      //val =  step((x - input->AdvDiff_A(0) * t));
    }
    else if (input->nDims == 3)
    {
      
      val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
             std::sin(M_PI * (x - input->AdvDiff_A(0) * t))*
             std::sin(M_PI * (y - input->AdvDiff_A(1) * t))*
             std::sin(M_PI * (z - input->AdvDiff_A(2) * t));
      
//      val =  std::sin(2 * M_PI * x) + std::sin(2 * M_PI * y) + std::sin(2 * M_PI * z);
//      val =  std::sin(.2 * M_PI * x) * std::sin(.2 * M_PI * y) * std::sin(.2 * M_PI * z);
    }
  }
  else if (input->equation == EulerNS)
  {
    if (input->test_case == 1 || input->test_case == 3)
    {
      double rho, Vx, Vy, P;
      if (input->test_case == 1)
      {
        double G = 5.0;
        double R = 1.;

        double f = (1.0 - x*x - y*y)/R;

        rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
                  M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
        Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f);
        Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
        P = std::pow(rho, input->gamma);
      }
      else if (input->test_case == 3)
      {
        /* Vincent et al. Isentropic Euler Vortex */
        double D = 20.0; // Domain size (e.g. [-D,D] x [-D,D] domain)
        double G = 13.5; // Vortex strength
        double M = 0.4;  // Mach number
        double R = 1.5;  // Vortex radius

        double omg = G / (2.0*M_PI*R);
        double f = std::exp((1.0 - x*x - y*y) / (2.0*R*R));

        rho = std::pow(1.0 - 0.5*(input->gamma-1.0)*omg*omg*M*M*R*R*f*f, 
            1.0/(input->gamma-1.0));
        Vx = -omg*f*y;
        Vy = 1.0 + omg*f*x;
        P = 1.0 / (input->gamma*M*M) * std::pow(rho, input->gamma); // Unit entropy
      }

      if (input->nDims == 2)
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * Vx; break;
          case 2:
            val = rho * Vy; break;
          case 3:
            val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
        }
      }
      else
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * Vx; break;
          case 2:
            val = rho * Vy; break;
          case 3:
            val = 0; break;
          case 4:
            val = P/(input->gamma - 1.0) + 0.5 * rho * (Vx * Vx + Vy * Vy); break;
        }
      }
    }
    else if (input->test_case == 2)
    {
      /* Couette flow test case */
      double gamma = input->gamma;
      double Pr = input->prandtl;
      double P = input->P_fs;
      double R = input->R;
      double Vw = input->V_wall(0);
      double Tw = input->T_wall;
      double cp = gamma * R / (gamma - 1);

      double rho = gamma / (gamma - 1) * (2 * P)/(2*cp*Tw + Pr*Vw*Vw * y * (1-y));

      if (input->nDims == 2)
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * (Vw*y); break;
          case 2:
            val = 0.0; break;
          case 3:
            val = P / (gamma - 1) + 0.5 * (gamma / (gamma - 1) *
                                           2 * P / (2*cp*Tw + Pr * Vw*Vw * y * (1-y))) * Vw*Vw*y*y; break;
        }
      }
      else
      {
        switch (var)
        {
          case 0:
            val = rho; break;
          case 1:
            val = rho * (Vw*y); break;
          case 2:
            val = 0.0; break;
          case 3:
            val =  0.0; break;
          case 4:
            val = P / (gamma - 1) + 0.5 * (gamma / (gamma - 1) *
                                           2 * P / (2*cp*Tw + Pr * Vw*Vw * y * (1-y))) * Vw*Vw*y*y; break;
        }
      }
    }
    else if (input->test_case == 4)
    {
      /* Taylor-Green Vortex Test Case */
      ThrowException("Error formula for Taylor-Green Vortex test case not implemented");
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}

// TODO: Complete implementation of derivative for Euler vortex
double compute_dU_true(double x, double y, double z, double t, unsigned int var, 
    unsigned int dim, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      if (dim == 0)
      {
        val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
               std::sin(M_PI * (y - input->AdvDiff_A(1) * t)) *
               M_PI * std::cos(M_PI * (x - input->AdvDiff_A(0) * t));

      }
      else
      {
        val =  std::exp(-2. * input->AdvDiff_D * M_PI * M_PI * t) * 
               std::sin(M_PI * (x - input->AdvDiff_A(0) * t)) *
               M_PI * std::cos(M_PI * (y - input->AdvDiff_A(1) * t));
      }
    }
    else if (input->nDims == 3)
    {
      val = 0.0; // implement
    }
  }
  else
  {
    val = 0.0; // implement
  }

  return val;
}

double compute_source_term(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
  double val = 0.;
  if (input->equation == AdvDiff)
  {
    if (input->nDims == 2)
    {
      val =  -M_PI * (std::cos(M_PI * x) + M_PI * std::sin(M_PI * x) + 
             std::cos(M_PI * y) + M_PI * std::sin(M_PI * y));
    }
    else
    {
      val =  -M_PI * (std::cos(M_PI * x) + M_PI * std::sin(M_PI * x) + 
             std::cos(M_PI * y) + M_PI * std::sin(M_PI * y) + 
             std::cos(M_PI * z) + M_PI * std::sin(M_PI * z));
    }
  }
  else
  {
    ThrowException("No source defined for EulerNS!");
  }

  return val;
}

double get_cfl_limit_adv(int order)
{
  /* Upwinded */
  switch(order)
  {
    case 0:
      return 1.392;

    case 1:
      return 0.4642; 

    case 2:
      return 0.2351;

    case 3:
      return 0.1453;

    case 4:
      return 0.1000;

    case 5:
      return 0.07363;
      
    case 6:
      return 0.05678;
    
    case 7:
      return 0.04530;
    
    case 8:
      return 0.03709;
    
    case 9:
      return 0.03101;
      
    case 10:
      return 0.02635;

    default:
      ThrowException("CFL limit no available for this order!");
  }
}

double get_cfl_limit_diff(int order, double beta)
{
  /* Centered */
  if (beta == 0)
  {
    switch(order)
    {
      case 0:
        return 2.785;

      case 1:
        return 0.1740;

      case 2:
        return 0.04264;

      case 3:
        return 0.01580;

      case 4:
        return 0.007193;

      case 5:
        return 0.003730;

      case 6:
        return 0.002120;

      case 7:
        return 0.001292;

      case 8:
        return 0.0008314;

      case 9:
        return 0.0005586;

      case 10:
        return 0.0003890;

      default:
        ThrowException("CFL limit no available for this order!");
    }
  }

  /* Upwinded */
  else
  {
    switch(order)
    {
      case 0:
        return 0.6963;

      case 1:
        return 0.07736; 

      case 2:
        return 0.01878;

      case 3:
        return 0.006345;

      case 4:
        return 0.002664;

      case 5:
        return 0.001299;

      case 6:
        return 0.0007060;

      case 7:
        return 0.0004153;

      case 8:
        return 0.0002599;

      case 9:
        return 0.0001708;

      case 10:
        return 0.0001168;

      default:
        ThrowException("CFL limit no available for this order!");
    }
  }
}

// See Eigen's 'determinant.h' from 2014-9-18,
// https://bitbucket.org/eigen/eigen file Eigen/src/LU/determinant.h,
double det_3x3_part(const double* mat, int a, int b, int c)
{
  return mat[a] * (mat[3+b] * mat[6+c] - mat[3+c] * mat[6+b]);
}

double det_4x4_part(const double* mat, int j, int k, int m, int n)
{
  return (mat[j*4] * mat[k*4+1] - mat[k*4] * mat[j*4+1])
      * (mat[m*4+2] * mat[n*4+3] - mat[n*4+2] * mat[m*4+3]);
}

double det_2x2(const double* mat)
{
  return mat[0]*mat[3] - mat[1]*mat[2];
}

double det_3x3(const double* mat)
{
  return det_3x3_part(mat,0,1,2) - det_3x3_part(mat,1,0,2)
      + det_3x3_part(mat,2,0,1);
}

double det_4x4(const double* mat)
{
  return det_4x4_part(mat,0,1,2,3) - det_4x4_part(mat,0,2,1,3)
      + det_4x4_part(mat,0,3,1,2) + det_4x4_part(mat,1,2,0,3)
      - det_4x4_part(mat,1,3,0,2) + det_4x4_part(mat,2,3,0,1);
}

mdvector<double> adjoint(const mdvector<double> &mat)
{
  auto dims = mat.shape();

  if (dims[0] != dims[1])
    ThrowException("Adjoint only meaningful for square matrices.");

  mdvector<double> adj({dims[0],dims[1]});

  int signRow = -1;
  mdvector<double> Minor({dims[0]-1,dims[1]-1});
  for (int row = 0; row < dims[0]; row++)
  {
    signRow *= -1;
    int sign = -1*signRow;
    for (int col = 0; col < dims[1]; col++)
    {
      sign *= -1;
      // Setup the minor matrix (expanding along row, col)
      int i0 = 0;
      for (int i = 0; i < dims[0]; i++)
      {
        if (i == row) continue;
        int j0 = 0;
        for (int j = 0; j < dims[1]; j++)
        {
          if (j == col) continue;
          Minor(i0,j0) = mat(i,j);
          j0++;
        }
        i0++;
      }
      // Recall: adjoint is TRANSPOSE of cofactor matrix
      adj(col,row) = sign*determinant(Minor);
    }
  }

  return adj;
}


//! In-place adjoint function
void adjoint(const mdvector<double> &mat, mdvector<double> &adj)
{
  auto dims = mat.shape();

  if (dims[0] != dims[1])
    ThrowException("Adjoint only meaningful for square matrices.");

  adj.resize({dims[0],dims[1]});

  int signRow = -1;
  mdvector<double> Minor({dims[0]-1,dims[1]-1});
  for (int row = 0; row < dims[0]; row++)
  {
    signRow *= -1;
    int sign = -1*signRow;
    for (int col = 0; col < dims[1]; col++)
    {
      sign *= -1;
      // Setup the minor matrix (expanding along row, col)
      int i0 = 0;
      for (int i = 0; i < dims[0]; i++)
      {
        if (i == row) continue;
        int j0 = 0;
        for (int j = 0; j < dims[1]; j++)
        {
          if (j == col) continue;
          Minor(i0,j0) = mat(i,j);
          j0++;
        }
        i0++;
      }
      // Recall: adjoint is TRANSPOSE of cofactor matrix
      adj(col,row) = sign*determinant(Minor);
    }
  }
}

void adjoint_3x3(double *mat, double *adj)
{
  double a11 = mat[0], a12 = mat[1], a13 = mat[2];
  double a21 = mat[3], a22 = mat[4], a23 = mat[5];
  double a31 = mat[6], a32 = mat[7], a33 = mat[8];

  adj[0] = a22*a33 - a23*a32;
  adj[1] = a13*a32 - a12*a33;
  adj[2] = a12*a23 - a13*a22;

  adj[3] = a23*a31 - a21*a33;
  adj[4] = a11*a33 - a13*a31;
  adj[5] = a13*a21 - a11*a23;

  adj[6] = a21*a32 - a22*a31;
  adj[7] = a12*a31 - a11*a32;
  adj[8] = a11*a22 - a12*a21;
}

void adjoint_4x4(double *mat, double *adj)
{
  double a11 = mat[0],  a12 = mat[1],  a13 = mat[2],  a14 = mat[3];
  double a21 = mat[4],  a22 = mat[5],  a23 = mat[6],  a24 = mat[7];
  double a31 = mat[8],  a32 = mat[9],  a33 = mat[10], a34 = mat[11];
  double a41 = mat[12], a42 = mat[13], a43 = mat[14], a44 = mat[15];

  adj[0] = -a24*a33*a42 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 + a22*a33*a44;
  adj[1] =  a14*a33*a42 - a13*a34*a42 - a14*a32*a43 + a12*a34*a43 + a13*a32*a44 - a12*a33*a44;
  adj[2] = -a14*a23*a42 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 + a12*a23*a44;
  adj[3] =  a14*a23*a32 - a13*a24*a32 - a14*a22*a33 + a12*a24*a33 + a13*a22*a34 - a12*a23*a34;

  adj[4] =  a24*a33*a41 - a23*a34*a41 - a24*a31*a43 + a21*a34*a43 + a23*a31*a44 - a21*a33*a44;
  adj[5] = -a14*a33*a41 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 + a11*a33*a44;
  adj[6] =  a14*a23*a41 - a13*a24*a41 - a14*a21*a43 + a11*a24*a43 + a13*a21*a44 - a11*a23*a44;
  adj[7] = -a14*a23*a31 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 + a11*a23*a34;

  adj[8] = -a24*a32*a41 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 + a21*a32*a44;
  adj[9] =  a14*a32*a41 - a12*a34*a41 - a14*a31*a42 + a11*a34*a42 + a12*a31*a44 - a11*a32*a44;
  adj[10]= -a14*a22*a41 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 + a11*a22*a44;
  adj[11]=  a14*a22*a31 - a12*a24*a31 - a14*a21*a32 + a11*a24*a32 + a12*a21*a34 - a11*a22*a34;

  adj[12]=  a23*a32*a41 - a22*a33*a41 - a23*a31*a42 + a21*a33*a42 + a22*a31*a43 - a21*a32*a43;
  adj[13]= -a13*a32*a41 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 + a11*a32*a43;
  adj[14]=  a13*a22*a41 - a12*a23*a41 - a13*a21*a42 + a11*a23*a42 + a12*a21*a43 - a11*a22*a43;
  adj[15]= -a13*a22*a31 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33;
}

double determinant(const mdvector<double> &mat)
{
  auto dims = mat.shape();

  if (dims[0] != dims[1])
    ThrowException("Determinant only meaningful for square matrices.");

  if (dims[0] == 1)
    return mat(0,0);

  else if (dims[0] == 2)
    return mat(0,0)*mat(1,1) - mat(1,0)*mat(0,1);

  else if (dims[0] == 3)
    return det_3x3(mat.data());

  else if (dims[0] == 4)
    return det_4x4(mat.data());

  else
  {
    // Use minor-matrix recursion
    double Det = 0;
    int sign = -1;
    mdvector<double> Minor({dims[0]-1, dims[1]-1});
    for (int row = 0; row < dims[0]; row++)
    {
      sign *= -1;
      // Setup the minor matrix (expanding along first column)
      int i0 = 0;
      for (int i = 0; i < dims[0]; i++)
      {
        if (i == row) continue;
        for (int j = 1; j < dims[1]; j++)
        {
          Minor(i0,j-1) = mat(i,j);
        }
        i0++;
      }
      // Add in the minor's determinant
      Det += sign*determinant(Minor)*mat(row,0);
    }

    return Det;
  }
}

mdvector<double> getRotationMatrix(double axis[3], double angle)
{
  Vec3 Axis = Vec3(axis[0],axis[1],axis[2]);
  double mag = Axis.norm();
  double ax = Axis.x; ax /= mag;
  double ay = Axis.y; ay /= mag;
  double az = Axis.z; az /= mag;

  if (mag > 1e-8)
  {
    ax /= mag;
    ay /= mag;
    az /= mag;
  }

  mdvector<double> mat({3,3}, 0);

  double s = sin(angle);
  double c = cos(angle);
  double c1 = 1.-c;

  double axyc = ax*ay*c1;
  double axzc = ax*az*c1;
  double ayzc = ay*az*c1;

  mat(0,0) = c + ax*ax*c1;  mat(0,1) = axyc - az*s;  mat(0,2) = axzc + ay*s;
  mat(1,0) = axyc + az*s;   mat(1,1) = c + ay*ay*c1; mat(1,2) = ayzc - ax*s;
  mat(2,0) = axzc - ay*s;   mat(2,1) = ayzc + ax*s;  mat(2,2) = c + az*az*c1;

  return mat;
}

mdvector<double> getRotationMatrix(const Quat &q)
{
  Vec3 Axis = Vec3(q[1],q[2],q[3]);
  double mag = Axis.norm();
  double ax = Axis.x;
  double ay = Axis.y;
  double az = Axis.z;

  if (mag > 1e-8)
  {
    ax /= mag;
    ay /= mag;
    az /= mag;
  }

  mdvector<double> mat({3,3}, 0);

  double angle = 2*acos(q[0]/q.norm()); // Must have unit quaternion
  double s = sin(angle);
  double c = cos(angle);
  double c1 = 1.-c;

  double axyc = ax*ay*c1;
  double axzc = ax*az*c1;
  double ayzc = ay*az*c1;

  mat(0,0) = c + ax*ax*c1;  mat(0,1) = axyc - az*s;  mat(0,2) = axzc + ay*s;
  mat(1,0) = axyc + az*s;   mat(1,1) = c + ay*ay*c1; mat(1,2) = ayzc - ax*s;
  mat(2,0) = axzc - ay*s;   mat(2,1) = ayzc + ax*s;  mat(2,2) = c + az*az*c1;

  return mat;
}

mdvector<double> identityMatrix(unsigned int N)
{
  mdvector<double> mat({N,N}, 0.0);
  for (unsigned int i = 0; i < N; i++)
    mat(i,i) = 1;
  return mat;
}

std::vector<int> gmsh_to_structured_tri(unsigned int nNodes)
{
  if (ijk_maps_tri.count(nNodes))
    return ijk_maps_tri[nNodes];

  std::vector<int> gmsh_to_pyfr(nNodes,0);

  switch (nNodes)
  {
    case 3:
      gmsh_to_pyfr = {0,1,2};
      break;

    case 6:
      gmsh_to_pyfr = {0, 2, 5, 1, 4, 3};
      break;

    case 10:
      gmsh_to_pyfr = {0, 3, 9, 1, 2, 6, 8, 7, 4, 5};
      break;

    case 15:
      gmsh_to_pyfr = {0, 4, 14, 1, 2, 3, 8, 11, 13, 12, 9, 5, 6, 7, 10};
      break;

    case 21:
      gmsh_to_pyfr = {0, 5, 20, 1, 2, 3, 4, 10, 14, 17, 19, 18, 15,
                     11, 6, 7, 8, 9, 13, 16, 12};
      break;

    default:
      ThrowException("PyFR/Gmsh node map not implemented for this nNodes.\n"
         "See https://github.com/vincentlab/PyFR/blob/develop/pyfr/readers/nodemaps.py");
  }

  ijk_maps_tri[nNodes] = gmsh_to_pyfr;

  return gmsh_to_pyfr;
}

std::vector<int> gmsh_to_structured_tet(unsigned int nNodes)
{
  if (ijk_maps_tet.count(nNodes))
    return ijk_maps_tet[nNodes];

  std::vector<int> gmsh_to_pyfr(nNodes,0);

  switch (nNodes)
  {
    case 4:
      gmsh_to_pyfr = {0,1,2,3};
      break;

    case 10:
      gmsh_to_pyfr = {0, 2, 5, 9, 1, 4, 3, 6, 8, 7};
      break;

    case 20:
      gmsh_to_pyfr = {0, 3, 9, 19, 1, 2, 6, 8, 7, 4, 16, 10, 18, 15,
                     17, 12, 5, 11, 13, 14};
      break;

    case 35:
      gmsh_to_pyfr = {0, 4, 14, 34, 1, 2, 3, 8, 11, 13, 12, 9, 5, 31,
                     25, 15, 33, 30, 24, 32, 27, 18, 6, 10, 7, 16,
                     17, 26, 19, 28, 22, 29, 21, 23, 20};
      break;

    case 56:
      gmsh_to_pyfr = {0, 5, 20, 55, 1, 2, 3, 4, 10, 14, 17, 19, 18,
                     15, 11, 6, 52, 46, 36, 21, 54, 51, 45, 35, 53,
                     48, 39, 25, 7, 16, 9, 12, 13, 8, 22, 24, 47,
                     23, 38, 37, 26, 49, 33, 40, 43, 30, 50, 29, 34,
                     42, 32, 44, 27, 28, 31, 41};
      break;

    case 84:
      gmsh_to_pyfr = {0, 6, 27, 83, 1, 2, 3, 4, 5, 12, 17, 21, 24,
                      26, 25, 22, 18, 13, 7, 80, 74, 64, 49, 28, 82,
                      79, 73, 63, 48, 81, 76, 67, 53, 33, 8, 23, 11,
                      14, 19, 20, 16, 10, 9, 15, 29, 32, 75, 30, 31,
                      52, 66, 65, 50, 51, 34, 77, 46, 54, 68, 71, 61,
                      43, 39, 58, 78, 38, 47, 70, 57, 42, 45, 62, 72,
                      60, 35, 37, 44, 69, 36, 41, 40, 55, 59, 56};
      break;

    default:
      ThrowException("PyFR/Gmsh node map not implemented for this nNodes.\n"
         "See https://github.com/vincentlab/PyFR/blob/develop/pyfr/readers/nodemaps.py");
  }

  ijk_maps_tet[nNodes] = gmsh_to_pyfr;

  return gmsh_to_pyfr;
}


std::vector<int> gmsh_to_structured_pri(unsigned int nNodes)
{
  if (ijk_maps_pri.count(nNodes))
    return ijk_maps_pri[nNodes];

  std::vector<int> gmsh_to_ijk(nNodes,0);

  switch (nNodes)
  {
    case 6:
      gmsh_to_ijk = {0,1,2,3,4,5};
      break;

    case 18:
      gmsh_to_ijk = {0, 2, 5, 12, 14, 17, 1, 3, 6, 4, 8, 11, 13, 15,
                     16, 7, 9, 10};
      break;

    case 40:
      gmsh_to_ijk = {0, 3, 9, 30, 33, 39, 1, 2, 4, 7, 10, 20, 6, 8,
                     13, 23, 19, 29, 31, 32, 34, 37, 36, 38, 5, 35,
                     11, 12, 22, 21, 14, 24, 27, 17, 16, 18, 28, 26,
                     15, 25};
      break;

    case 75:
      gmsh_to_ijk = {0, 4, 14, 60, 64, 74, 1, 2, 3, 5, 9, 12, 15,
                     30, 45, 8, 11, 13, 19, 34, 49, 29, 44, 59, 61,
                     62, 63, 65, 69, 72, 68, 71, 73, 6, 10, 7, 66,
                     67, 70, 16, 18, 48, 46, 17, 33, 47, 31, 32, 20,
                     50, 57, 27, 35, 54, 42, 24, 39, 23, 28, 58, 53,
                     26, 43, 56, 38, 41, 21, 51, 36, 22, 52, 37, 25,
                     55, 40};
      break;

    case 126:
      gmsh_to_ijk = {0, 5, 20, 105, 110, 125, 1, 2, 3, 4, 6, 11,
                     15, 18, 21, 42, 63, 84, 10, 14, 17, 19, 26,
                     47, 68, 89, 41, 62, 83, 104, 106, 107, 108,
                     109, 111, 116, 120, 123, 115, 119, 122, 124,
                     7, 16, 9, 12, 13, 8, 112, 114, 121, 113, 118,
                     117, 22, 25, 88, 85, 23, 24, 46, 67, 87, 86,
                     64, 43, 44, 45, 66, 65, 27, 90, 102, 39, 48,
                     69, 95, 99, 81, 60, 36, 32, 53, 74, 78, 57,
                     31, 40, 103, 94, 35, 38, 61, 82, 101, 98, 73,
                     52, 56, 59, 80, 77, 28, 91, 49, 70, 30, 93,
                     51, 72, 37, 100, 58, 79, 29, 92, 50, 71, 34,
                     97, 55, 76, 33, 96, 54, 75};
      break;

    default:
      ThrowException("PyFR/Gmsh node map not implemented for this nNodes.\n"
         "See https://github.com/vincentlab/PyFR/blob/develop/pyfr/readers/nodemaps.py");
  }

  ijk_maps_pri[nNodes] = gmsh_to_ijk;

  return gmsh_to_ijk;
}

std::vector<int> structured_to_gmsh_pri(unsigned int nNodes)
{
  return reverse_map(gmsh_to_structured_pri(nNodes));
}

std::vector<int> gmsh_to_structured_quad(unsigned int nNodes)
{
  if (ijk_maps_quad.count(nNodes))
    return ijk_maps_quad[nNodes];

  std::vector<int> gmsh_to_ijk(nNodes,0);

  /* Lagrange Elements */
  if (nNodes != 8)
  {
    int nNodes1D = sqrt(nNodes);

    if (nNodes1D * nNodes1D != nNodes)
      ThrowException("nNodes must be a square number.");

    int nLevels = nNodes1D / 2;

    /* Set shape values via recursive strategy (from Flurry) */
    int node = 0;
    for (int i = 0; i < nLevels; i++)
    {
      /* Corner Nodes */
      int i2 = (nNodes1D - 1) - i;
      gmsh_to_ijk[node]     = i  + nNodes1D * i;
      gmsh_to_ijk[node + 1] = i2 + nNodes1D * i;
      gmsh_to_ijk[node + 2] = i2 + nNodes1D * i2;
      gmsh_to_ijk[node + 3] = i  + nNodes1D * i2;

      node += 4;

      int nEdgeNodes = nNodes1D - 2 * (i + 1);
      for (int j = 0; j < nEdgeNodes; j++)
      {
        gmsh_to_ijk[node + j]                = i+1+j  + nNodes1D * i;
        gmsh_to_ijk[node + nEdgeNodes + j]   = i2     + nNodes1D * (i+1+j);
        gmsh_to_ijk[node + 2*nEdgeNodes + j] = i2-1-j + nNodes1D * i2;
        gmsh_to_ijk[node + 3*nEdgeNodes + j] = i      + nNodes1D * (i2-1-j);
      }

      node += 4 * nEdgeNodes;
    }

    /* Add center node in odd case */
    if (nNodes1D % 2 != 0)
    {
      gmsh_to_ijk[nNodes - 1] = nNodes1D/2 + nNodes1D * (nNodes1D/2);
    }
  }

  ijk_maps_quad[nNodes] = gmsh_to_ijk;

  return gmsh_to_ijk;
}

std::vector<int> structured_to_gmsh_quad(unsigned int nNodes)
{
  if (gmsh_maps_quad.count(nNodes))
  {
    return gmsh_maps_quad[nNodes];
  }
  else
  {
    auto gmsh2ijk = gmsh_to_structured_quad(nNodes);

    gmsh_maps_quad[nNodes] = reverse_map(gmsh2ijk);

    return gmsh_maps_quad[nNodes];
  }
}

std::vector<int> structured_to_gmsh_hex(unsigned int nNodes)
{
  if (gmsh_maps_hex.count(nNodes))
  {
    return gmsh_maps_hex[nNodes];
  }
  else
  {
    auto gmsh2ijk = gmsh_to_structured_hex(nNodes);

    gmsh_maps_hex[nNodes] = reverse_map(gmsh2ijk);

    return gmsh_maps_hex[nNodes];
  }
}

std::vector<int> gmsh_to_structured_hex(unsigned int nNodes)
{
  if (ijk_maps_hex.count(nNodes))
    return ijk_maps_hex[nNodes];

  std::vector<int> gmsh_to_ijk(nNodes,0);

  int nSide = cbrt(nNodes);

  if (nSide*nSide*nSide != nNodes)
  {
    std::cout << "nNodes = " << nNodes << std::endl;
    ThrowException("For Lagrange hex of order N, must have (N+1)^3 shape points.");
  }

  std::vector<double> xlist(nSide);
  double dxi = 2./(nSide-1);

  for (int i=0; i<nSide; i++)
    xlist[i] = -1. + i*dxi;

  int nLevels = nSide / 2;
  int isOdd = nSide % 2;

  /* Recursion for all high-order Lagrange elements:
   * 8 corners, each edge's points, interior face points, volume points */
  int nPts = 0;
  for (int i = 0; i < nLevels; i++) {
    // Corners
    int i2 = (nSide-1) - i;
    gmsh_to_ijk[nPts+0] = i  + nSide * (i  + nSide * i);
    gmsh_to_ijk[nPts+1] = i2 + nSide * (i  + nSide * i);
    gmsh_to_ijk[nPts+2] = i2 + nSide * (i2 + nSide * i);
    gmsh_to_ijk[nPts+3] = i  + nSide * (i2 + nSide * i);
    gmsh_to_ijk[nPts+4] = i  + nSide * (i  + nSide * i2);
    gmsh_to_ijk[nPts+5] = i2 + nSide * (i  + nSide * i2);
    gmsh_to_ijk[nPts+6] = i2 + nSide * (i2 + nSide * i2);
    gmsh_to_ijk[nPts+7] = i  + nSide * (i2 + nSide * i2);
    nPts += 8;

    // Edges
    int nSide2 = nSide - 2 * (i+1);
    for (int j = 0; j < nSide2; j++) {
      // Edges around 'bottom'
      gmsh_to_ijk[nPts+0*nSide2+j] = i+1+j  + nSide * (i     + nSide * i);
      gmsh_to_ijk[nPts+3*nSide2+j] = i2     + nSide * (i+1+j + nSide * i);
      gmsh_to_ijk[nPts+5*nSide2+j] = i2-1-j + nSide * (i2    + nSide * i);
      gmsh_to_ijk[nPts+1*nSide2+j] = i      + nSide * (i+1+j + nSide * i);

      // 'Vertical' edges
      gmsh_to_ijk[nPts+2*nSide2+j] = i  + nSide * (i  + nSide * (i+1+j));
      gmsh_to_ijk[nPts+4*nSide2+j] = i2 + nSide * (i  + nSide * (i+1+j));
      gmsh_to_ijk[nPts+6*nSide2+j] = i2 + nSide * (i2 + nSide * (i+1+j));
      gmsh_to_ijk[nPts+7*nSide2+j] = i  + nSide * (i2 + nSide * (i+1+j));

      // Edges around 'top'
      gmsh_to_ijk[nPts+ 8*nSide2+j] = i+1+j  + nSide * (i     + nSide * i2);
      gmsh_to_ijk[nPts+10*nSide2+j] = i2     + nSide * (i+1+j + nSide * i2);
      gmsh_to_ijk[nPts+11*nSide2+j] = i2-1-j + nSide * (i2    + nSide * i2);
      gmsh_to_ijk[nPts+ 9*nSide2+j] = i      + nSide * (i+1+j + nSide * i2);
    }
    nPts += 12*nSide2;

    /* --- Faces [Use recursion from quadrilaterals] --- */

    int nLevels2 = nSide2 / 2;
    int isOdd2 = nSide2 % 2;

    // --- Bottom face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = j  + nSide * (j  + nSide * i);
      gmsh_to_ijk[nPts+1] = j  + nSide * (j2 + nSide * i);
      gmsh_to_ijk[nPts+2] = j2 + nSide * (j2 + nSide * i);
      gmsh_to_ijk[nPts+3] = j2 + nSide * (j  + nSide * i);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = j      + nSide * (j+1+k  + nSide * i);
        gmsh_to_ijk[nPts+1*nSide3+k] = j+1+k  + nSide * (j2     + nSide * i);
        gmsh_to_ijk[nPts+2*nSide3+k] = j2     + nSide * (j2-1-k + nSide * i);
        gmsh_to_ijk[nPts+3*nSide3+k] = j2-1-k + nSide * (j      + nSide * i);
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = nSide/2 +  nSide*(nSide/2) +  nSide*nSide*i;
      nPts += 1;
    }

    // --- Front face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = j  + nSide * (i + nSide * j);
      gmsh_to_ijk[nPts+1] = j2 + nSide * (i + nSide * j);
      gmsh_to_ijk[nPts+2] = j2 + nSide * (i + nSide * j2);
      gmsh_to_ijk[nPts+3] = j  + nSide * (i + nSide * j2);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = j+1+k  + nSide * (i + nSide * j);
        gmsh_to_ijk[nPts+1*nSide3+k] = j2     + nSide * (i + nSide * (j+1+k));
        gmsh_to_ijk[nPts+2*nSide3+k] = j2-1-k + nSide * (i + nSide * j2);
        gmsh_to_ijk[nPts+3*nSide3+k] = j      + nSide * (i + nSide * (j2-1-k));
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = nSide/2 + nSide*(i + nSide*(nSide/2));
      nPts += 1;
    }

    // --- Left face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = i + nSide * (j  + nSide * j);
      gmsh_to_ijk[nPts+1] = i + nSide * (j  + nSide * j2);
      gmsh_to_ijk[nPts+2] = i + nSide * (j2 + nSide * j2);
      gmsh_to_ijk[nPts+3] = i + nSide * (j2 + nSide * j);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = i + nSide * (j      + nSide * (j+1+k));
        gmsh_to_ijk[nPts+1*nSide3+k] = i + nSide * (j+1+k  + nSide * j2);
        gmsh_to_ijk[nPts+2*nSide3+k] = i + nSide * (j2     + nSide * (j2-1-k));
        gmsh_to_ijk[nPts+3*nSide3+k] = i + nSide * (j2-1-k + nSide * j);
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = i + nSide * (nSide/2 + nSide * (nSide/2));
      nPts += 1;
    }

    // --- Right face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = i2 + nSide * (j  + nSide * j);
      gmsh_to_ijk[nPts+1] = i2 + nSide * (j2 + nSide * j);
      gmsh_to_ijk[nPts+2] = i2 + nSide * (j2 + nSide * j2);
      gmsh_to_ijk[nPts+3] = i2 + nSide * (j  + nSide * j2);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = i2 + nSide * (j+1+k  + nSide * j);
        gmsh_to_ijk[nPts+1*nSide3+k] = i2 + nSide * (j2     + nSide * (j+1+k));
        gmsh_to_ijk[nPts+2*nSide3+k] = i2 + nSide * (j2-1-k + nSide * j2);
        gmsh_to_ijk[nPts+3*nSide3+k] = i2 + nSide * (j      + nSide * (j2-1-k));
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = i2 + nSide * (nSide/2 + nSide * (nSide/2));
      nPts += 1;
    }

    // --- Back face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = j2 + nSide * (i2 + nSide * j);
      gmsh_to_ijk[nPts+1] = j  + nSide * (i2 + nSide * j);
      gmsh_to_ijk[nPts+2] = j  + nSide * (i2 + nSide * j2);
      gmsh_to_ijk[nPts+3] = j2 + nSide * (i2 + nSide * j2);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = j2-1-k + nSide * (i2 + nSide*j);
        gmsh_to_ijk[nPts+1*nSide3+k] = j      + nSide * (i2 + nSide*(j+1+k));
        gmsh_to_ijk[nPts+2*nSide3+k] = j+1+k  + nSide * (i2 + nSide*j2);
        gmsh_to_ijk[nPts+3*nSide3+k] = j2     + nSide * (i2 + nSide*(j2-1-k));
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = nSide/2 + nSide * (i2 + nSide * (nSide/2));
      nPts += 1;
    }

    // --- Top face ---
    for (int j0 = 0; j0 < nLevels2; j0++) {
      // Corners
      int j = j0 + i + 1;
      int j2 = i + 1 + (nSide2-1) - j0;
      gmsh_to_ijk[nPts+0] = j  + nSide * (j  + nSide * i2);
      gmsh_to_ijk[nPts+1] = j2 + nSide * (j  + nSide * i2);
      gmsh_to_ijk[nPts+2] = j2 + nSide * (j2 + nSide * i2);
      gmsh_to_ijk[nPts+3] = j  + nSide * (j2 + nSide * i2);
      nPts += 4;

      // Edges: Bottom, right, top, left
      int nSide3 = nSide2 - 2 * (j0+1);
      for (int k = 0; k < nSide3; k++) {
        gmsh_to_ijk[nPts+0*nSide3+k] = j+1+k  + nSide * (j      + nSide * i2);
        gmsh_to_ijk[nPts+1*nSide3+k] = j2     + nSide * (j+1+k  + nSide * i2);
        gmsh_to_ijk[nPts+2*nSide3+k] = j2-1-k + nSide * (j2     + nSide * i2);
        gmsh_to_ijk[nPts+3*nSide3+k] = j      + nSide * (j2-1-k + nSide * i2);
      }
      nPts += 4*nSide3;
    }

    // Center node for even-ordered Lagrange quads (odd value of nSide)
    if (isOdd2) {
      gmsh_to_ijk[nPts] = nSide/2 + nSide * (nSide/2 +  nSide * i2);
      nPts += 1;
    }
  }

  // Center node for even-ordered Lagrange quads (odd value of nSide)
  if (isOdd) {
    gmsh_to_ijk[nNodes-1] = nSide/2 + nSide * (nSide/2 + nSide * (nSide/2));
  }

  ijk_maps_hex[nNodes] = gmsh_to_ijk;

  return gmsh_to_ijk;
}

std::vector<int> reverse_map(const std::vector<int> &map1)
{
  auto map2 = map1;
  for (int i = 0; i < map1.size(); i++)
    map2[map1[i]] = i;

  return map2;
}

std::vector<int> get_int_list(int N, int start)
{
  std::vector<int> list(N);
  for (int i = 0; i < N; i++)
    list[i] = start + i;

  return list;
}

std::vector<uint> get_int_list(uint N, uint start)
{
  std::vector<uint> list(N);
  for (uint i = 0; i < N; i++)
    list[i] = start + i;

  return list;
}

unsigned int tri_nodes_to_order(unsigned int nNodes)
{
  int P = 0;
  while (P < 20)
  {
    if ((P+1) * (P+2)/2 == nNodes)
      return P;
    P++;
  }

  ThrowException("Can't figure out triangle shape order!");
}

unsigned int tet_nodes_to_order(unsigned int nNodes)
{
  int P = 0;
  while (P < 20)
  {
    if ((P+1) * (P+2) * (P+3)/6 == nNodes)
      return P;
    P++;
  }
  ThrowException("Can't figure out tetrahedra shape order!");
}

mdvector<double> quat_mul(const mdvector<double> &p, const mdvector<double> &q)
{
  // Assuming real part is last value [ i, j, k, real ]
  mdvector<double> z({4}, 0.0);

  z(0) = p(3)*q(3) - p(0)*q(0) - p(1)*q(1) - p(2)*q(1);
  z(1) = p(3)*q(0) + p(0)*q(3) + p(1)*q(2) - p(2)*q(1);
  z(2) = p(3)*q(1) - p(0)*q(2) + p(1)*q(3) + p(2)*q(0);
  z(3) = p(3)*q(2) + p(0)*q(1) - p(1)*q(0) + p(2)*q(3);

  return z;
}

Quat Quat::conj(void)
{
  Quat p;
  p[0] = q[0];

  for (unsigned i = 1; i < 4; i++)
    p[i] = -q[i];

  return p;
}

Quat Quat::operator+(const Quat &p)
{
  Quat z;
  for (unsigned i = 0; i < 4; i++)
    z[i] = q[i] + p[i];
  return z;
}

Quat Quat::operator-(const Quat &p)
{
  Quat z;
  for (unsigned i = 0; i < 4; i++)
    z[i] = q[i] + p[i];
  return z;
}

void Quat::operator+=(const Quat &p)
{
  for (unsigned i = 0; i < 4; i++)
    q[i] += p[i];
}

void Quat::operator-=(const Quat &p)
{
  for (unsigned i = 0; i < 4; i++)
    q[i] -= p[i];
}

Quat Quat::cross(const Quat& p)
{
  // Vector op - Assuming we are ignoring the real component of the quaternion
  Quat z;
  z[1] = q[2]*p[3] - q[3]*p[2];
  z[2] = q[3]*p[1] - q[1]*p[3];
  z[3] = q[1]*p[2] - q[2]*p[1];
  return z;
}

Quat Quat::operator*(const Quat &p)
{
  Quat z;
  z[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3];
  z[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2];
  z[2] = q[0]*p[2] + q[2]*p[0] + q[3]*p[1] - q[1]*p[3];
  z[3] = q[0]*p[3] + q[3]*p[0] + q[1]*p[2] - q[2]*p[1];
  return z;
}

Quat Quat::operator*(const std::array<double,3> &p)
{
  Quat z;
  z[0] = -q[1]*p[1] - q[2]*p[2] - q[3]*p[3];
  z[1] = q[0]*p[1] + q[2]*p[3] - q[3]*p[2];
  z[2] = q[0]*p[2] + q[3]*p[1] - q[1]*p[3];
  z[3] = q[0]*p[3] + q[1]*p[2] - q[2]*p[1];
  return z;
}

Quat operator*(double a, const Quat &b)
{
  Quat c;
  for (unsigned int i = 0; i < 4; i++)
    c[i] = a*b[i];
  return c;
}

/* Simple string hash from web: http://www.cse.yorku.ca/~oz/hash.html */
unsigned long hash_str(const char *str)
{
  unsigned long hash = 5381;
  int c;

  while (c = *str++)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}
