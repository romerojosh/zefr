#ifndef funcs_hpp
#define funcs_hpp

#include <cmath>

#include "cblas.h"

#include "input.hpp"
#include "mdvector.hpp"

/* Computes solution at specified time and location */
//double compute_U_true(double x, double y, double t, unsigned int var, const InputStruct *input);
double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input);
double compute_dU_true(double x, double y, double z, double t, unsigned int var, 
    unsigned int dim, const InputStruct *input);

double get_cfl_limit(int order);

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

template <typename T> 
unsigned int step(const T& val) 
{
    return T(0) <= val;
}

#endif /* funcs_hpp */
