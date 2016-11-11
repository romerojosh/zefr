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

#ifdef _OMP
#include "omp.h"
#endif

#include "funcs.hpp"
#include "input.hpp"
#include "funcs.hpp"

//! Storing Gmsh to structured-IJK index mappings to avoid re-computing
static std::map<int, std::vector<int>> gmsh_maps_hex;
static std::map<int, std::vector<int>> gmsh_maps_quad;
static std::map<int, std::vector<int>> ijk_maps_quad;
static std::map<int, std::vector<int>> ijk_maps_hex;

double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
   
  double val = 0.0;

  if (input->equation == AdvDiff || input->equation == Burgers)
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
    if (!input->viscous)
    {
      double G = 5.0;
      double R = 1.;

      double f = (1.0 - x*x - y*y)/R;

      double rho = std::pow(1.0 - (G * G * (input->gamma - 1.))/(8.0 * input->gamma * 
                M_PI * M_PI) * std::exp(f), 1.0/(input->gamma - 1.0)); 
      double Vx = 1.0 - G * y / (2.0*M_PI) * std::exp(0.5 * f);
      double Vy = 1.0 + G * x / (2.0*M_PI) * std::exp(0.5 * f);
      double P = std::pow(rho, input->gamma);

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
    else
    {
      val = 1.0246950765959597 * y;
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
      ThrowException("Under construction!");
    }
  }
  else if (input->equation == Burgers)
  {
    val = 0.0; // Just a placeholder value for now.
  }
  else if (input->equation == EulerNS)
  {
    if (!input->viscous)
    {
      val = 0.0; // Just a placeholder value for now.
    }
    else
    {
      if (dim == 0)
      {
        val = 0.0;
      }
      else
      {
        val = 1.0246950765959597;
      }
    }
  }
  else
  {
    ThrowException("Under construction!");
  }

  return val;
}

double compute_source_term(double x, double y, double z, double t, unsigned int var, const InputStruct *input)
{
  double val = 0.;
  if (input->equation == AdvDiff || input->equation == Burgers)
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

#ifdef _OMP
void omp_blocked_dgemm(CBLAS_ORDER mode, CBLAS_TRANSPOSE transA, 
    CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, double* A, int lda, 
    double* B, int ldb, double beta, double* C, int ldc)
{
  
#pragma omp parallel
  {
    int nThreads = omp_get_num_threads();
    int thread_idx = omp_get_thread_num();

    int block_size = N / nThreads;
    int start_idx = block_size * thread_idx;

    if (thread_idx == nThreads-1)
      block_size += N % (block_size);

    cblas_dgemm(mode, transA, transB, M, block_size, K, alpha, A, lda, 
        B + ldb * start_idx, ldb, beta, C + ldc * start_idx, ldc);
  }
}
#endif

//std::ostream& operator<<(std::ostream &os, const point &pt)
//{
//  os << "(x,y,z) = " << pt.x << ", " << pt.y << ", " << pt.z;
//  return os;
//}

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

std::vector<int> gmsh_to_structured_quad(unsigned int nNodes)
{
  if (ijk_maps_quad.count(nNodes))
    return ijk_maps_quad[nNodes];

  std::vector<int> gmsh_to_ijk(nNodes,0);

  /* Lagrange Elements (or linear serendipity) */
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

  /* 8-node Serendipity Element */
  else
  {
    gmsh_to_ijk[0] = 0; gmsh_to_ijk[1] = 2;  gmsh_to_ijk[2] = 7;
    gmsh_to_ijk[3] = 5; gmsh_to_ijk[4] = 1;  gmsh_to_ijk[5] = 3;
    gmsh_to_ijk[6] = 4; gmsh_to_ijk[7] = 6;
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
    map2[i] = findFirst(map1, i);

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
