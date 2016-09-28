#ifndef funcs_hpp
#define funcs_hpp

#include <cmath>

#include "cblas.h"

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

#endif /* funcs_hpp */
