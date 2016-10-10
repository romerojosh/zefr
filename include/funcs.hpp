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

#ifndef funcs_hpp
#define funcs_hpp

#include <cmath>

extern "C" {
#include "cblas.h"
}

#include "input.hpp"
#include "mdvector.hpp"

/* Computes solution at specified time and location */
double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input);
double compute_dU_true(double x, double y, double z, double t, unsigned int var, 
    unsigned int dim, const InputStruct *input);

/* Computes source term as specified time and location */
double compute_source_term(double x, double y, double z, double t, unsigned int var, const InputStruct *input);

/* Compute maximum CFL */
double get_cfl_limit_adv(int order);
double get_cfl_limit_diff(int order, double beta);

/* Blocked gemm operation using OpenMP */
#ifdef _OMP
void omp_blocked_dgemm(CBLAS_ORDER mode, CBLAS_TRANSPOSE transA, 
    CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, double* A, int lda, 
    double* B, int ldb, double beta, double* C, int ldc);
#endif

template <typename T>
struct square
{
  T operator()(const T &val1, const T &val2)
  {
    return (val1 + val2 * val2);
  }
};

template <typename T>
struct abs_sum
{
  T operator()(const T &val1, const T &val2)
  {
    return (val1 + std::abs(val2));
  }
};

/* Diagonal general matrix multiplication (C = alpha * A * diag(x) + beta * C) */
template <typename T>
void dgmm(int m, int n, double alpha, T* A, int lda, T* x, int incx, double beta, T* C, int ldc)
{
  for (unsigned int j = 0; j < n; j++)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      C[ldc*j + i] = alpha * A[lda*j + i] * x[j + incx] + beta * C[ldc*j + i];
    }
  }
}

/* General matrix multiplication (C = alpha * A * B + beta * C) */
template <typename T>
void gemm(int m, int n, int p, double alpha, T* A, int lda, T* B, int ldb, double beta, T* C, int ldc)
{
  for (unsigned int j = 0; j < n; j++)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      double val = 0;
      for (unsigned int k = 0; k < p; k++)
      {
        val += A[lda*k + i] * B[ldb*j + k];
      }
      C[ldc*j + i] = alpha * val + beta * C[ldc*j + i];
    }
  }
}

template <typename T> 
unsigned int step(const T& val) 
{
    return T(0) <= val;
}

//std::ostream& operator<<(std::ostream &os, const point &pt);

mdvector<double> adjoint(const mdvector<double> &mat);

double determinant(const mdvector<double> &mat);

template<typename T>
int findFirst(const std::vector<T>& vec, T val)
{
  for (int i = 0; i < vec.size(); i++)
    if (vec[i] == val)
      return i;

  return -1;
}

std::vector<int> reverse_map(const std::vector<int> &map1);

std::vector<int> get_int_list(int N, int start = 0);
std::vector<uint> get_int_list(uint N, uint start = 0);

//! Map a structured ijk-type index to the equivalent Gmsh node index
std::vector<int> structured_to_gmsh_quad(unsigned int nNodes);
std::vector<int> structured_to_gmsh_hex(unsigned int nNodes);

//! Map a Gmsh node index to the equivalent structured ijk-type index
std::vector<int> gmsh_to_structured_quad(unsigned int nNodes);
std::vector<int> gmsh_to_structured_hex(unsigned int nNodes);


template<typename T>
std::vector<uint> sort_ind(const std::vector<T> &data, const std::vector<uint> &inds)
{
  std::vector<std::pair<T,size_t>> vp;
  vp.reserve(inds.size());

  for (uint i = 0; i < inds.size(); i++)
    vp.push_back(make_pair(data[inds[i]], inds[i]));

  /* Sorting will put lower values [vp.first] ahead of larger ones, resolving
   * ties using the original index [vp.second] */
  std::sort(vp.begin(), vp.end());

  std::vector<uint> sorted_ind(inds.size());
  for (uint i = 0; i < vp.size(); i++)
    sorted_ind[i] = vp[i].second;

  return sorted_ind;
}

template<typename T>
std::vector<uint> sort_ind(const std::vector<T> &data)
{
  std::vector<std::pair<T,size_t>> vp;
  vp.reserve(data.size());

  for (uint i = 0; i < data.size(); i++)
    vp.push_back(make_pair(data[i], i));

  /* Sorting will put lower values [vp.first] ahead of larger ones, resolving
   * ties using the original index [vp.second] */
  std::sort(vp.begin(), vp.end());

  std::vector<uint> sorted_ind(data.size());
  for (uint i = 0; i < vp.size(); i++)
    sorted_ind[i] = vp[i].second;

  return sorted_ind;
}

template<typename T>
std::vector<uint>  fuzzysort_ind(mdvector<T> mat, const std::vector<uint> &inds, uint dim = 0, double tol = 1e-6)
{
  auto dims = mat.shape();
  std::vector<T> data(dims[1]);
  for (uint i = 0; i < dims[1]; i++)
    data[i] = mat(dim, i);

  auto ind = sort_ind(data, inds);

  uint j, i = 0;
  uint ix = ind[0];
  for (j = 1; j < ind.size(); j++)
  {
    uint jx = ind[j];
    if (data[jx] - data[ix] >= tol)
    {
      if (j - i > 1 && dim+1 < dims[0])
      {
        // Get the new indices to sort
        std::vector<uint> _inds(j-i);
        for (uint k = 0; k < j-i; k++)
          _inds[k] = ind[i+k];

        // Sort the duplicated by the next dimension and update 'ind'
        auto ind2 = fuzzysort_ind(mat,_inds,dim+1,tol);

        for (uint k = 0; k < j-i; k++)
          ind[i+k] = ind2[k];
      }
      i = j;
      ix = jx;
    }
  }

  if (i != j && dim+1 < dims[0])
  {
    std::vector<uint> _inds(j-i);
    for (uint k = 0; k < j-i; k++)
      _inds[k] = ind[i+k];

    // Sort the duplicated by the next dimension and update 'ind'
    auto ind2 = fuzzysort_ind(mat,_inds,dim+1,tol);

    for (uint k = 0; k < j-i; k++)
      ind[i+k] = ind2[k];
  }

  return ind;
}

template<typename T>
std::vector<uint> fuzzysort(const mdvector<T>& mat, uint dim = 0, double tol = 1e-6)
{
  auto dims = mat.shape();
  auto list = get_int_list(dims[1]);
  return fuzzysort_ind(mat, list, dim, tol);
}

#endif /* funcs_hpp */
