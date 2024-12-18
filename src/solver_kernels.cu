#include <iostream>

#include "cuda_runtime.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#ifdef _MPI
#include "mpi.h"
#endif

#include "input.hpp"
#include "macros.hpp"
#include "mdvector_gpu.h"
#include "aux_kernels.h"
#include "solver_kernels.h"
#include "funcs_kernels.cu"

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

#define PI 3.141592653589793

__global__
void gaussJordanInv(int N, double** Aarray, int lda, double** Carray, int ldc, int batchCount)
{
  const unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int batch = blockDim.y * blockIdx.y + threadIdx.y; batch < batchCount; batch += gridDim.y * blockDim.y)
  {
    for (unsigned int j = 0; j < N; j++)
    { 
      for (unsigned int i = tidx; i < N; i += blockDim.x)
      {
        Carray[batch][i + j*lda] = (double) (i == j);
      }
    }

    for (unsigned int j = 0; j < N; j++)
    { 
      for (unsigned int i = tidx; i < N; i += blockDim.x)
      {
        if (i != j)
        {
          double fac =  Aarray[batch][i + j*lda] / Aarray[batch][j + j*lda];
          for (unsigned int k = 0; k < N; k++)
          {
            Aarray[batch][i + k*lda] -= fac * Aarray[batch][j + k*lda];
            Carray[batch][i + k*lda] -= fac * Carray[batch][j + k*lda];
          }
        }
      }
    }

    for (unsigned int j = 0; j < N; j++)
    { 
      for (unsigned int i = tidx; i < N; i += blockDim.x)
      {
        Carray[batch][i + j*lda] /= Aarray[batch][i + i*lda];
      }
    }

    __syncthreads(); /* To avoid divergence */
  }

}

void gaussJordanInv_wrapper(int N, double** Aarray, int lda, double** Carray, int ldc, int batchCount)
{
  dim3 threads(32, 6);
  dim3 blocks(1, std::min((batchCount + threads.y - 1)/threads.y, MAX_GRID_DIM));

  gaussJordanInv<<<blocks, threads>>>(N, Aarray, lda, Carray, ldc, batchCount);
}


__global__
void DgemvBatched_noAlpha_noBeta(const int M, const int N, const double** Aarray, int lda, const double** xarray, int incx,
    double** yarray, int incy, int batchCount)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= M)
    return;

  for (unsigned int batch = blockDim.y * blockIdx.y + threadIdx.y; batch < batchCount; batch += gridDim.y * blockDim.y)
  {
    double sum = 0.0;
    for (unsigned int j = 0; j < N; j++)
      sum += Aarray[batch][i + j*lda] * xarray[batch][j * incx];

    yarray[batch][i * incy] = sum;

    __syncthreads(); /* To avoid divergence */
  }
}

void DgemvBatched_wrapper(const int M, const int N, const double alpha, const double** Aarray, int lda, const double** xarray, int incx,
    const double beta, double** yarray, int incy, int batchCount)
{
  dim3 threads(32, 6);
  dim3 blocks((M + threads.x - 1)/threads.x, std::min((batchCount + threads.y - 1)/threads.y, MAX_GRID_DIM));

  if (alpha == 1.0 && beta == 0.0)
    DgemvBatched_noAlpha_noBeta<<<blocks, threads>>>(M, N, Aarray, lda, xarray, incx, yarray, incy, batchCount);
  else
    ThrowException("DgemvBatched not implemented for alpha and nonzero beta!");
}

template<unsigned int nVars, unsigned int nDims>
__global__
void add_source(mdvector_gpu<double> divF, const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int equation, 
    double flow_time, unsigned int stage, bool overset = false, const int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double x = coord_spts(spt, 0, ele);
    double y = coord_spts(spt, 1, ele);
    double z = 0;
    if (nDims == 3)
      z = coord_spts(spt, 2, ele);

    double jaco_det = jaco_det_spts(spt, ele);

    for (unsigned int n = 0; n < nVars; n++)
    {
      divF(stage, spt, n, ele) += compute_source_term_dev(x, y, z, flow_time, n, nDims, equation) * jaco_det;
    }
  }
}

void add_source_wrapper(mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  if (nDims == 2)
  {
    if (equation == AdvDiff)
      add_source<1, 2><<<blocks, threads>>>(divF, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage);
    else
      add_source<4, 2><<<blocks, threads>>>(divF, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, overset, iblank);
  }
  else
  {
    if (equation == AdvDiff)
      add_source<1, 3><<<blocks, threads>>>(divF, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage);
    else
      add_source<5, 3><<<blocks, threads>>>(divF, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, overset, iblank);
  }
}



template <unsigned int nVars>
__global__
void RK_update(mdvector_gpu<double> U_spts, const mdvector_gpu<double> U_ini, 
    const mdvector_gpu<double> divF, const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> dt_in, 
    const mdvector_gpu<double> rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int stage, unsigned int nStages, bool last_stage, bool overset = false, const int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  if (overset && iblank[ele] != 1)
      return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  //for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double fac = dt / jaco_det_spts(spt,ele);

    if (!last_stage)
    {
      double coeff = rk_coeff(stage);
      for (unsigned int var = 0; var < nVars; var ++)
        U_spts(spt, var, ele) = U_ini(spt, var, ele) - coeff *  
            fac * divF(stage, spt, var, ele);
    }
    else
    {
      double sum[nVars] = {0.0};

      for (unsigned int n = 0; n < nStages; n++)
      {
        double coeff = rk_coeff(n);
        for (unsigned int var = 0; var < nVars; var++)
        {
          sum[var] -= coeff * fac * divF(n, spt, var, ele);
        }
      }

      for (unsigned int var = 0; var < nVars; var++)
        U_spts(spt, var, ele) += sum[var];
    }
  }
}

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage, bool overset, int* iblank)
{
  //unsigned int threads = 128;
  //unsigned int blocks = (nEles + threads - 1)/ threads;
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  if (equation == AdvDiff)
  {
      RK_update<1><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      RK_update<4><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
    else
      RK_update<5><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage, overset, iblank);
  }
}

template <unsigned int nVars>
__global__
void RK_update_source(mdvector_gpu<double> U_spts, const mdvector_gpu<double> U_ini, 
    const mdvector_gpu<double> divF, const mdvector_gpu<double> source, const mdvector_gpu<double> jaco_det_spts, 
    const mdvector_gpu<double> dt_in, const mdvector_gpu<double> rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int stage, unsigned int nStages, 
    bool last_stage, bool overset = false, int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);

  if (ele >= nEles)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double fac = dt / jaco_det_spts(spt,ele);

    if (!last_stage)
    {
      double coeff = rk_coeff(stage);
      for (unsigned int var = 0; var < nVars; var ++)
        U_spts(spt, var, ele) = U_ini(spt, var, ele) - coeff *  
            fac * (divF(stage, spt, var, ele) + source(spt, var, ele));
    }
    else
    {
      double sum[nVars] = {0.0};;

      for (unsigned int n = 0; n < nStages; n++)
      {
        double coeff = rk_coeff(n);
        for (unsigned int var = 0; var < nVars; var++)
        {
          sum[var] -= coeff * fac * (divF(n, spt, var, ele) + source(spt, var, ele));
        }
      }

      for (unsigned int var = 0; var < nVars; var++)
        U_spts(spt, var, ele) += sum[var];

    }
  }
}

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage,
    bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
      RK_update_source<1><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage, overset, iblank);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      RK_update_source<4><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
    else
      RK_update_source<5><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage, overset, iblank);
  }
}

template <unsigned int nVars>
__global__
void LSRK_update(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_til,
    mdvector_gpu<double> rk_err, const mdvector_gpu<double> divF,
    const mdvector_gpu<double> jaco_det_spts, double dt, double ai, double bi,
    double bhi, unsigned int nSpts, unsigned int nEles, unsigned int stage,
    unsigned int nStages, bool adapt_dt, bool overset = false, int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  if (overset && iblank[ele] != 1)
      return;

  double fac = dt / jaco_det_spts(spt,ele);

  if (adapt_dt)
    for (unsigned int var = 0; var < nVars; var++)
      rk_err(spt, var, ele) -= (bi - bhi) * fac * divF(0, spt, var, ele);

  if (stage < nStages - 1)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      U_spts(spt, var, ele) = U_til(spt, var, ele) - ai * fac *
          divF(0, spt, var, ele);

      U_til(spt, var, ele) = U_spts(spt, var, ele) - (bi - ai) * fac *
          divF(0, spt, var, ele);
    }
  }
  else
  {
    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt, var, ele) = U_til(spt, var, ele) - bi * fac *
          divF(0, spt, var, ele);
  }
}

void LSRK_update_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, double dt,
    double ai, double bi, double bhi, unsigned int nSpts, unsigned int nEles,
    unsigned int nVars, unsigned int stage, unsigned int nStages, bool adapt_dt, 
    bool overset, int* iblank)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  switch (nVars)
  {
    case 1:
      LSRK_update<1><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, adapt_dt, 
          overset, iblank);
      break;

    case 4:
      LSRK_update<4><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, adapt_dt, 
          overset, iblank);
      break;

    case 5:
      LSRK_update<5><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, adapt_dt, 
          overset, iblank);
      break;

    default:
      std::string errs = "Update wrapper for nVars = " + std::to_string(nVars) +
          " no implemented.";
      ThrowException(errs.c_str());
  }
}

template <unsigned int nVars>
__global__
void LSRK_source_update(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_til,
    mdvector_gpu<double> rk_err, mdvector_gpu<double> divF,
    mdvector_gpu<double> source, mdvector_gpu<double> jaco_det_spts, double dt,
    double ai, double bi, double bhi, unsigned int nSpts, unsigned int nEles,
    unsigned int stage, unsigned int nStages, bool adapt_dt, bool overset = false,
    int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  if (overset && iblank[ele] != 1)
      return;

  double fac = dt / jaco_det_spts(spt,ele);

  if (adapt_dt)
    for (unsigned int var = 0; var < nVars; var ++)
      rk_err(spt, var, ele) -= (bi - bhi) * fac *
          (divF(0, spt, var, ele) + source(spt, var, ele));

  if (stage != nStages - 1)
  {
    for (unsigned int var = 0; var < nVars; var ++)
    {
      U_spts(spt, var, ele) = U_til(spt, var, ele) - ai * fac *
          (divF(0, spt, var, ele) + source(spt, var, ele));

      U_til(spt, var, ele) = U_spts(spt, var, ele) - (bi - ai) * fac *
          (divF(0, spt, var, ele) + source(spt, var, ele));
    }
  }
  else
  {
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, var, ele) = U_til(spt, var, ele) - bi * fac *
          (divF(0, spt, var, ele) + source(spt, var, ele));
  }
}

void LSRK_update_source_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source,
    mdvector_gpu<double> &jaco_det_spts, double dt, double ai, double bi,
    double bhi, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int stage, unsigned int nStages, bool adapt_dt, bool overset, 
    int* iblank)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  switch (nVars)
  {
    case 1:
      LSRK_source_update<1><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
          adapt_dt, overset, iblank);
      break;

    case 4:
      LSRK_source_update<4><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
          adapt_dt, overset, iblank);
      break;

    case 5:
      LSRK_source_update<5><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
          adapt_dt, overset, iblank);
      break;

    default:
      std::string errs = "Update wrapper for nVars = " + std::to_string(nVars) +
          " no implemented.";
      ThrowException(errs.c_str());
  }
}

template <unsigned int nVars>
__global__
void DIRK_update(mdvector_gpu<double> U_spts, const mdvector_gpu<double> divF, 
    const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> dt_in, 
    const mdvector_gpu<double> rk_coeff, unsigned int dt_type, unsigned int nSpts, 
    unsigned int nEles, unsigned int nStages)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  double dt = (dt_type != 2) ? dt_in(0) : dt_in(ele);

  //for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double fac = dt / jaco_det_spts(spt,ele);
    double sum[nVars] = {0.0};
    for (unsigned int stage = 0; stage < nStages; stage++)
    {
      double coeff = rk_coeff(stage);
      for (unsigned int var = 0; var < nVars; var++)
        sum[var] -= coeff * fac * divF(stage, spt, var, ele);
    }

    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt, var, ele) += sum[var];
  }
}

void DIRK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &divF, 
    mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    unsigned int nStages)
{
  //unsigned int threads = 128;
  //unsigned int blocks = (nEles + threads - 1)/ threads;
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  if (equation == AdvDiff)
    DIRK_update<1><<<blocks, threads>>>(U_spts, divF, jaco_det_spts, dt, 
        rk_coeff, dt_type, nSpts, nEles, nStages);
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      DIRK_update<4><<<blocks, threads>>>(U_spts, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, nStages);
    else
      DIRK_update<5><<<blocks, threads>>>(U_spts, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, nStages);
  }
}

template <unsigned int nVars>
__global__
void RK_error_update(mdvector_gpu<double> rk_err, const mdvector_gpu<double> divF, 
    const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> dt_in, 
    const mdvector_gpu<double> rk_beta, const mdvector_gpu<double> rk_bhat, 
    unsigned int nSpts, unsigned int nEles, unsigned int nStages)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  double dt = dt_in(0);

  //for (unsigned int spt = 0; spt < nSpts; spt++)
  {
    double fac = dt / jaco_det_spts(spt,ele);
    double sum[nVars] = {0.0};
    for (unsigned int stage = 0; stage < nStages; stage++)
    {
      double coeff = rk_beta(stage) - rk_bhat(stage);
      for (unsigned int var = 0; var < nVars; var++)
        sum[var] -= coeff * fac * divF(stage, spt, var, ele);
    }

    for (unsigned int var = 0; var < nVars; var++)
      rk_err(spt, var, ele) += sum[var];
  }
}

void RK_error_update_wrapper(mdvector_gpu<double> &rk_err, mdvector_gpu<double> &divF, 
    mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_beta, mdvector_gpu<double> &rk_bhat, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    unsigned int nStages)
{
  //unsigned int threads = 128;
  //unsigned int blocks = (nEles + threads - 1)/ threads;
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  if (equation == AdvDiff)
    RK_error_update<1><<<blocks, threads>>>(rk_err, divF, jaco_det_spts, dt, 
        rk_beta, rk_bhat, nSpts, nEles, nStages);
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      RK_error_update<4><<<blocks, threads>>>(rk_err, divF, jaco_det_spts, dt, 
          rk_beta, rk_bhat, nSpts, nEles, nStages);
    else
      RK_error_update<5><<<blocks, threads>>>(rk_err, divF, jaco_det_spts, dt, 
          rk_beta, rk_bhat, nSpts, nEles, nStages);
  }
}

template <unsigned int nVars>
__global__
void get_rk_error(mdvector_gpu<double> U_spts, const mdvector_gpu<double> U_ini,
    mdvector_gpu<double> rk_err, uint nSpts, uint nEles, double atol,
    double rtol, bool overset = false, int* iblank = NULL)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (ele >= nEles || spt >= nSpts)
    return;

  if (overset && iblank[ele] != 1)
  {
      for (unsigned int var = 0; var < nVars; var ++)
        rk_err(spt, var, ele) = 0.0;
    return;
  }

  for (unsigned int var = 0; var < nVars; var ++)
  {
    rk_err(spt, var, ele)  =  abs(rk_err(spt, var, ele));
    rk_err(spt, var, ele) /= atol + rtol *
        max( abs(U_spts(spt, var, ele)), abs(U_ini(spt, var, ele)) );
  }
}

double get_rk_error_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err, uint nSpts,
    uint nEles, uint nVars, double atol, double rtol, _mpi_comm comm_in,
    bool overset, int* iblank)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y -1)/threads.y);

  switch (nVars)
  {
    case 1:
      get_rk_error<1><<<blocks, threads>>>(U_spts, U_ini, rk_err, nSpts, nEles,
          atol, rtol, overset, iblank);
      break;

    case 4:
      get_rk_error<4><<<blocks, threads>>>(U_spts, U_ini, rk_err, nSpts, nEles,
          atol, rtol, overset, iblank);
      break;

    case 5:
      get_rk_error<5><<<blocks, threads>>>(U_spts, U_ini, rk_err, nSpts, nEles,
          atol, rtol, overset, iblank);
      break;

    default:
      ThrowException("Invalid value for nVars");
  }

  /* Get min dt using thrust (pretty slow) */
  thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(rk_err.data());
  thrust::device_ptr<double> max_ptr = thrust::max_element(err_ptr, err_ptr + nEles*nSpts*nVars);

#ifdef _MPI
  double max_err = max_ptr[0];
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, comm_in);
  err_ptr[0] = max_err;
#else
  err_ptr[0] = max_ptr[0];
#endif

  return err_ptr[0];
}

double set_adaptive_dt_wrapper(mdvector_gpu<double> &dt_in, double &dt_out,
    double expa, double expb, double minfac, double maxfac, double sfact,
    double max_dt, double max_err, double prev_err)
{
  // Determine the time step scaling factor and the new time step
  double fac = pow(max_err, -expa) * pow(prev_err, expb);
  fac = std::min(maxfac, std::max(minfac, sfact*fac));

  thrust::device_ptr<double> dt_ptr = thrust::device_pointer_cast(dt_in.data());
  dt_ptr[0] = std::min(fac*dt_ptr[0], max_dt);

  dt_out = dt_ptr[0]; // Set value on CPU for other uses

  return max_err;
}

template <unsigned int nDims>
__global__
void compute_element_dt(mdvector_gpu<double> dt, const mdvector_gpu<double> waveSp_gfpts, const mdvector_gpu<double> diffCo_gfpts,
    const mdvector_gpu<double> dA, const mdvector_gpu<int> fpt2gfpt, const mdvector_gpu<char> fpt2gfpt_slot, const mdvector_gpu<double> weights_fpts,
    const mdvector_gpu<double> vol, const mdvector_gpu<double> h_ref, unsigned int nFptsPerFace, double CFL, double beta, int order, int CFL_type,
    unsigned int nFpts, unsigned int nEles, unsigned int startEle, bool overset = false, const int* iblank = NULL)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles)
    return;

  unsigned int eleBT = ele + startEle;

  if (overset)
  {
    dt(ele) = 1e10;
    if (iblank[ele] != 1)
      return;
  }

  /* CFL-estimate used by Liang, Lohner, and others. Factor of 2 to be 
   * consistent with 1D CFL estimates. */
  if (CFL_type == 1)
  {
    double int_waveSp = 0.;  /* Edge/Face integrated wavespeed */
    for (unsigned int fpt = 0; fpt < nFpts; fpt++)
    {
      int gfpt = fpt2gfpt(fpt,eleBT);
      int slot = fpt2gfpt_slot(fpt,eleBT);

      int_waveSp += weights_fpts(fpt % nFptsPerFace) * waveSp_gfpts(gfpt) * dA(slot, gfpt);
    }

    dt(ele) = 2.0 * CFL * get_cfl_limit_adv_dev(order) * vol(ele) / int_waveSp;
  }

  /* CFL-estimate based on MacCormack for NS */
  else if (CFL_type == 2)
  {
    /* Compute inverse of timestep in each face */
    double dtinv[2*nDims] = {0};
    for (unsigned int face = 0; face < 2*nDims; face++)
    {
      for (unsigned int fpt = face * nFptsPerFace; fpt < (face+1) * nFptsPerFace; fpt++)
      {
        int gfpt = fpt2gfpt(fpt,eleBT);

        /* Compute inverse of timestep for each fpt */
        double dtinv_temp = waveSp_gfpts(gfpt) / (get_cfl_limit_adv_dev(order) * h_ref(fpt, ele)) + 
                            diffCo_gfpts(gfpt) / (get_cfl_limit_diff_dev(order, beta) * h_ref(fpt, ele) * h_ref(fpt, ele));
        dtinv[face] = max(dtinv[face], dtinv_temp);
      }
    }

    /* Find maximum in each dimension */
    if (nDims == 2)
    {
      dtinv[0] = max(dtinv[0], dtinv[2]);
      dtinv[1] = max(dtinv[1], dtinv[3]);

      dt(ele) = CFL / (dtinv[0] + dtinv[1]);
    }
    else
    {
      dtinv[0] = max(dtinv[0],dtinv[1]);
      dtinv[1] = max(dtinv[2],dtinv[3]);
      dtinv[2] = max(dtinv[4],dtinv[5]);

      /// NOTE: this seems ultra-conservative.  Need additional scaling factor?
      dt(ele) = CFL / (dtinv[0] + dtinv[1] + dtinv[2]); // 32 = empirically-found factor
    }
  }
}

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, 
    mdvector_gpu<double> &diffCo_gfpts, mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, 
    mdvector_gpu<char> &fpt2gfpt_slot, mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nFptsPerFace, double CFL, double beta, int order, 
    unsigned int CFL_type, unsigned int nFpts, unsigned int nEles, unsigned int nDims, 
    unsigned int startEle, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1) / threads;

  if (nDims == 2)
  {
    compute_element_dt<2><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, 
        fpt2gfpt_slot, weights_fpts, vol, h_ref, nFptsPerFace, CFL, beta, order, CFL_type, 
        nFpts, nEles, startEle);
  }
  else
  {
    compute_element_dt<3><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, 
        fpt2gfpt_slot, weights_fpts, vol, h_ref, nFptsPerFace, CFL, beta, order, CFL_type, 
        nFpts, nEles, startEle, overset, iblank);
  }
}


__global__
void apply_dt_LHS(mdvector_gpu<double> LHS, mdvector_gpu<double> dt,
    double coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles,
    unsigned int nVars)
{
  const unsigned int nVarsSpts2 = nVars * nSpts * nVars * nSpts;
  const unsigned int tidx = blockIdx.x * blockDim.x  + threadIdx.x;

  if (tidx >= nVarsSpts2)
    return;

  const unsigned int spti = tidx % nSpts;
  const unsigned int vari = (tidx / nSpts) % nVars;
  const unsigned int sptj = (tidx / (nVars * nSpts)) % nSpts;
  const unsigned int varj = (tidx / (nSpts * nVars * nSpts)) % nVars;

  const unsigned int tidy = blockIdx.y * blockDim.y  + threadIdx.y;
  for (unsigned int ele = tidy; ele < nEles; ele += gridDim.y * blockDim.y)
  {
    const double val = coeff * ((dt_type != 2) ? dt(0) : dt(ele));

    LHS(ele, varj, sptj, vari, spti) = val * LHS(ele, varj, sptj, vari, spti) 
      + (double) (spti == sptj && vari == varj);

    __syncthreads(); // To avoid divergence
  }
}

void apply_dt_LHS_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &dt,
    double coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars)
{
  dim3 threads(32, 6);
  dim3 blocks((nVars*nSpts*nVars*nSpts + threads.x - 1)/threads.x, 
      std::min((nEles + threads.y - 1) / threads.y, MAX_GRID_DIM));

  apply_dt_LHS<<<blocks, threads>>>(LHS, dt, coeff, dt_type, nSpts, nEles, nVars);
}

__global__
void compute_RHS(mdvector_gpu<double> divF, mdvector_gpu<double> jaco_det_spts, 
    mdvector_gpu<double> rk_coeff, mdvector_gpu<double> RHS, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int stage)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (spt >= nSpts || ele >= nEles)
    return;

  const double jaco_det = jaco_det_spts(spt, ele);
  for (unsigned int var = 0; var < nVars; var++)
  {
    RHS(ele, var, spt) = -rk_coeff(stage,0) * divF(0, spt, var, ele) / jaco_det;
    for (unsigned int s = 1; s <= stage; s++)
      RHS(ele, var, spt) -= rk_coeff(stage, s) * divF(s, spt, var, ele) / jaco_det;
  }
}

__global__
void apply_dt_RHS(mdvector_gpu<double> dt, mdvector_gpu<double> RHS, double coeff, 
    unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (spt >= nSpts || ele >= nEles)
    return;

  const double val = coeff * ((dt_type != 2) ? dt(0) : dt(ele));
  for (unsigned int var = 0; var < nVars; var++)
    RHS(ele, var, spt) *= val;
}

__global__
void apply_deltaU_RHS(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_ini, mdvector_gpu<double> RHS, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int var = 0; var < nVars; var++)
    RHS(ele, var, spt) -= (U_spts(spt, var, ele) - U_ini(spt, var, ele));
}

void compute_RHS_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_iniNM, 
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &dtau, mdvector_gpu<double> &rk_coeff, 
    mdvector_gpu<double> &RHS, double dtau_ratio, bool implicit_steady, bool pseudo_time, 
    bool remove_deltaU, unsigned int dt_type, unsigned int dtau_type, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int stage)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y - 1)/threads.y);

  compute_RHS<<<blocks, threads>>>(divF, jaco_det_spts, rk_coeff, RHS, nSpts, nEles, nVars, stage);

  if (!implicit_steady)
  {
    apply_dt_RHS<<<blocks, threads>>>(dt, RHS, 1, dt_type, nSpts, nEles, nVars);
    apply_deltaU_RHS<<<blocks, threads>>>(U_spts, U_ini, RHS, nSpts, nEles, nVars);
  }

  if (pseudo_time)
  {
    apply_dt_RHS<<<blocks, threads>>>(dtau, RHS, dtau_ratio, dtau_type, nSpts, nEles, nVars);
    if (!remove_deltaU)
      apply_deltaU_RHS<<<blocks, threads>>>(U_spts, U_iniNM, RHS, nSpts, nEles, nVars);
  }
}

__global__
void compute_U(mdvector_gpu<double> U_spts, mdvector_gpu<double> deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{  
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int spt = (blockDim.y * blockIdx.y + threadIdx.y);

  if (spt >= nSpts || ele >= nEles)
    return;

  for (unsigned int var = 0; var < nVars; var++)
    U_spts(spt, var, ele) += deltaU(ele, var, spt);
}

void compute_U_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars)
{
  dim3 threads(32, 4);
  dim3 blocks((nEles + threads.x - 1)/threads.x, (nSpts + threads.y - 1)/threads.y);

  compute_U<<<blocks, threads>>>(U_spts, deltaU, nSpts, nEles, nVars);
}

#ifdef _MPI
__global__
void pack_U(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    const mdview_gpu<double> U, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int var = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (i >= nFpts || var >= nVars)
    return;

  U_sbuffs(var, i) = U(0, var, fpts(i));
}

void pack_U_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &U, unsigned int nVars, int stream)
{
  unsigned int threads = 128;
  unsigned int blocks = (fpts.size() * nVars + threads - 1) / threads;

  if (stream == -1)
    pack_U<<<blocks,threads>>>(U_sbuffs, fpts, U, nVars, fpts.size());
  else
    pack_U<<<blocks,threads, 0, stream_handles[stream]>>>(U_sbuffs, fpts, U, nVars, fpts.size());
}

__global__
void unpack_U(const mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdview_gpu<double> U, unsigned int nVars, unsigned int nFpts, 
    bool overset = false, const int* iblank = NULL)
{
  const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int var = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (i >= nFpts || var >= nVars)
    return;

  unsigned int gfpt = fpts(i);

  if (overset && iblank[gfpt] != 1)
      return;

  U(1, var, gfpt) = U_rbuffs(var, i);
}

void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &U, unsigned int nVars, int stream, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (fpts.size() * nVars + threads - 1) / threads;

  if (stream == -1)
    unpack_U<<<blocks,threads>>>(U_rbuffs, fpts, U, nVars, fpts.size(), overset, iblank);
  else
    unpack_U<<<blocks,threads, 0, stream_handles[stream]>>>(U_rbuffs, fpts, U, nVars, 
      fpts.size(), overset, iblank);
}

template<unsigned int nDims>
__global__
void pack_dU(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    const mdview_gpu<double> dU, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    U_sbuffs(dim, var, i) = dU(0, dim, var, fpts(i));
  }
}

void pack_dU_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (stream == -1)
  {
    if (nDims == 2)
      pack_dU<2><<<blocks, threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
    else
      pack_dU<3><<<blocks, threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
  }
  else
  {
    if (nDims == 2)
      pack_dU<2><<<blocks, threads, 0, stream_handles[stream]>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
    else
      pack_dU<3><<<blocks, threads, 0, stream_handles[stream]>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
  }
}

template<unsigned int nDims>
__global__
void unpack_dU(const mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdview_gpu<double> dU, unsigned int nVars, unsigned int nFpts,
    bool overset = false, const int* iblank = NULL)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  if (overset)
    if (iblank[fpts(i)] != 1)
      return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dU(1, dim, var, fpts(i)) = U_rbuffs(dim, var, i);
  }
}

void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream, bool overset,
    int* iblank)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (stream == -1)
  {
    if (nDims == 2)
      unpack_dU<2><<<blocks, threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
    else 
      unpack_dU<3><<<blocks, threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size(),
        overset, iblank);
  }
  else
  {
    if (nDims == 2)
      unpack_dU<2><<<blocks, threads, 0, stream_handles[stream]>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
    else 
      unpack_dU<3><<<blocks, threads, 0, stream_handles[stream]>>>(U_rbuffs, fpts, dU, nVars, fpts.size(),
        overset, iblank);
  }
}

#endif

__global__
void compute_moments(mdview_gpu<double> U, mdview_gpu<double> dU, mdvector_gpu<double> P, mdvector_gpu<double> coord,
    mdvector_gpu<double> x_cg, mdvector_gpu<double> norm, mdvector_gpu<double> dA,
    mdvector_gpu<double> weights, mdvector_gpu<double> force, mdvector_gpu<double> moment,
    double gamma, double rt, double c_sth, double mu_in, bool viscous, bool fix_vis, int nDims,
    int nFaces, mdvector_gpu<int> faceIDs, mdvector_gpu<int> face2fpts, int nFptsPerFace)
{
  const unsigned int face = blockDim.x * blockIdx.x + threadIdx.x;

  if (face >= nFaces)
    return;

  const unsigned int fid = faceIDs(face);

  const int c1[3] = {1,2,0}; // For cross-products
  const int c2[3] = {2,0,1};

  double taun[3];
  double tot_force[3] = {0,0,0};
  double tot_moment[3] = {0,0,0};

  for (int fpt = 0; fpt < nFptsPerFace; fpt++) /// TODO: NO MORE FACE ORDERING?
  {
    double tmp_force[3] = {0,0,0};
    int gfpt = face2fpts(fpt, fid);

    /* Get pressure */
    double PL = P(0, gfpt);

    /* Sum inviscid force contributions */
    for (unsigned int dim = 0; dim < nDims; dim++)
      tmp_force[dim] = weights(fpt) * PL * norm(dim, gfpt) * dA(0, gfpt);

    if (viscous)
    {
      if (nDims == 2)
      {
        /* Setting variables for convenience */
        /* States */
        double rho = U(0, 0, gfpt);
        double momx = U(0, 1, gfpt);
        double momy = U(0, 2, gfpt);
        double e = U(0, 3, gfpt);

        double u = momx / rho;
        double v = momy / rho;
        double e_int = e / rho - 0.5 * (u*u + v*v);

        /* Gradients */
        double rho_dx = dU(0, 0, 0, gfpt);
        double momx_dx = dU(0, 0, 1, gfpt);
        double momy_dx = dU(0, 0, 2, gfpt);

        double rho_dy = dU(0, 1, 0, gfpt);
        double momx_dy = dU(0, 1, 1, gfpt);
        double momy_dy = dU(0, 1, 2, gfpt);

        /* Set viscosity */
        double mu;
        if (fix_vis)
        {
          mu = mu_in;
        }
        /* If desired, use Sutherland's law */
        else
        {
          double rt_ratio = (gamma - 1.0) * e_int / (rt);
          mu = mu_in * std::pow(rt_ratio,1.5) * (1. + c_sth) / (rt_ratio + c_sth);
        }

        double du_dx = (momx_dx - rho_dx * u) / rho;
        double du_dy = (momx_dy - rho_dy * u) / rho;

        double dv_dx = (momy_dx - rho_dx * v) / rho;
        double dv_dy = (momy_dy - rho_dy * v) / rho;

        double diag = (du_dx + dv_dy) / 3.0;

        double tauxx = 2.0 * mu * (du_dx - diag);
        double tauxy = mu * (du_dy + dv_dx);
        double tauyy = 2.0 * mu * (dv_dy - diag);

        /* Get viscous normal stress */
        taun[0] = tauxx * norm(0, gfpt) + tauxy * norm(1, gfpt);
        taun[1] = tauxy * norm(0, gfpt) + tauyy * norm(1, gfpt);

        for (unsigned int dim = 0; dim < nDims; dim++)
          tmp_force[dim] -= weights(fpt) * taun[dim] * dA(0, gfpt);
      }
      else if (nDims == 3)
      {
        /* Setting variables for convenience */
        /* States */
        double rho = U(0, 0, gfpt);
        double momx = U(0, 1, gfpt);
        double momy = U(0, 2, gfpt);
        double momz = U(0, 3, gfpt);
        double e = U(0, 4, gfpt);

        double u = momx / rho;
        double v = momy / rho;
        double w = momz / rho;
        double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

        /* Gradients */
        double rho_dx = dU(0, 0, 0, gfpt);
        double momx_dx = dU(0, 0, 1, gfpt);
        double momy_dx = dU(0, 0, 2, gfpt);
        double momz_dx = dU(0, 0, 3, gfpt);

        double rho_dy = dU(0, 1, 0, gfpt);
        double momx_dy = dU(0, 1, 1, gfpt);
        double momy_dy = dU(0, 1, 2, gfpt);
        double momz_dy = dU(0, 1, 3, gfpt);

        double rho_dz = dU(0, 2, 0, gfpt);
        double momx_dz = dU(0, 2, 1, gfpt);
        double momy_dz = dU(0, 2, 2, gfpt);
        double momz_dz = dU(0, 2, 3, gfpt);

        /* Set viscosity */
        double mu;
        if (fix_vis)
        {
          mu = mu_in;
        }
        /* If desired, use Sutherland's law */
        else
        {
          double rt_ratio = (gamma - 1.0) * e_int / (rt);
          mu = mu_in * std::pow(rt_ratio,1.5) * (1. + c_sth) / (rt_ratio + c_sth);
        }

        double du_dx = (momx_dx - rho_dx * u) / rho;
        double du_dy = (momx_dy - rho_dy * u) / rho;
        double du_dz = (momx_dz - rho_dz * u) / rho;

        double dv_dx = (momy_dx - rho_dx * v) / rho;
        double dv_dy = (momy_dy - rho_dy * v) / rho;
        double dv_dz = (momy_dz - rho_dz * v) / rho;

        double dw_dx = (momz_dx - rho_dx * w) / rho;
        double dw_dy = (momz_dy - rho_dy * w) / rho;
        double dw_dz = (momz_dz - rho_dz * w) / rho;

        double diag = (du_dx + dv_dy + dw_dz) / 3.0;

        double tauxx = 2.0 * mu * (du_dx - diag);
        double tauyy = 2.0 * mu * (dv_dy - diag);
        double tauzz = 2.0 * mu * (dw_dz - diag);
        double tauxy = mu * (du_dy + dv_dx);
        double tauxz = mu * (du_dz + dw_dx);
        double tauyz = mu * (dv_dz + dw_dy);

        /* Get viscous normal stress */
        taun[0] = tauxx * norm(0, gfpt) + tauxy * norm(1, gfpt) + tauxz * norm(2, gfpt);
        taun[1] = tauxy * norm(0, gfpt) + tauyy * norm(1, gfpt) + tauyz * norm(2, gfpt);
        taun[3] = tauxz * norm(0, gfpt) + tauyz * norm(1, gfpt) + tauzz * norm(2, gfpt);

        for (unsigned int dim = 0; dim < nDims; dim++)
          tmp_force[dim] -= weights(fpt) * taun[dim] * dA(0, gfpt);
      }
    }

    // Add fpt's contribution to total force and moment
    for (unsigned int d = 0; d < nDims; d++)
      tot_force[d] += tmp_force[d];

    if (nDims == 3)
    {
      for (unsigned int d = 0; d < nDims; d++)
        tot_moment[d] += (coord(c1[d], gfpt)-x_cg(c1[d])) * tmp_force[c2[d]] - (coord(c2[d], gfpt)-x_cg(c2[d])) * tmp_force[c1[d]];
    }
    else
    {
      // Only a 'z' component in 2D
      tot_moment[2] += (coord(0,gfpt)-x_cg(0)) * tmp_force[1] - (coord(1,gfpt)-x_cg(1)) * tmp_force[0];
    }
  }

  // Write final sum for face to global memory
  for (int d = 0; d < nDims; d++)
  {
    force(face,d) = tot_force[d];
    moment(face,d) = tot_moment[d];
  }
}

void compute_moments_wrapper(std::array<double,3> &tot_force, std::array<double,3> &tot_moment,
    mdview_gpu<double> &U_fpts, mdview_gpu<double> &dU_fpts, mdvector_gpu<double>& P_fpts, mdvector_gpu<double> &coord,
    mdvector_gpu<double> &x_cg, mdvector_gpu<double> &norm, mdvector_gpu<double> &dA,
    mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &force_face, mdvector_gpu<double> &moment_face,
    double gamma, double rt, double c_sth, double mu, bool viscous, bool fix_vis, int nVars, int nDims,
    int nFaces, mdvector_gpu<int> faceList, mdvector_gpu<int> face2fpts, int nFptsPerFace)
{
  int threads = 192;
  int blocks = (nFaces + threads - 1) / threads;

  if (nFaces == 0 || nVars < 4) // Only applicable to Euler / Navier-Stokes
    return;

  compute_moments<<<blocks, threads>>>(U_fpts,dU_fpts,P_fpts,coord,x_cg,norm,dA,weights_fpts,
      force_face,moment_face,gamma,rt,c_sth,mu,viscous,fix_vis,nDims,nFaces,faceList,face2fpts,nFptsPerFace);

  for (int d = 0; d < nDims; d++)
  {
    thrust::device_ptr<double> f_ptr = thrust::device_pointer_cast(force_face.data()+d*nFaces);
    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(moment_face.data()+d*nFaces);
    tot_force[d] = thrust::reduce(f_ptr, f_ptr+nFaces, 0.);
    tot_moment[d] = thrust::reduce(m_ptr, m_ptr+nFaces, 0.);
  }

  check_error();
}

__global__
void accumulate_time_averages(mdvector_gpu<double> tavg_acc, mdvector_gpu<double> tavg_prev,
    mdvector_gpu<double> tavg_curr, mdvector_gpu<double> U_spts, double prev_time,
    double curr_time, double gamma, int nSpts, int nVars, int nDims, int nEles)
{
  const int ele = (blockDim.x * blockIdx.x + threadIdx.x) % nEles;
  const int spt = (blockDim.x * blockIdx.x + threadIdx.x) / nEles;

  if (ele >= nEles || spt >= nSpts) return;

  double rho = U_spts(spt,0,ele);
  double rhoU = U_spts(spt,1,ele);
  double rhoV = U_spts(spt,2,ele);
  double rhoW = U_spts(spt,3,ele);
  double rhoE = U_spts(spt,4,ele);

  // Conservative Variables
  tavg_curr(spt,0,ele) = rho;
  tavg_curr(spt,1,ele) = rhoU;
  tavg_curr(spt,2,ele) = rhoV;
  tavg_curr(spt,3,ele) = rhoW;
  tavg_curr(spt,4,ele) = rhoE;

  // Velocities
  tavg_curr(spt,5,ele) = rhoU/rho;
  tavg_curr(spt,6,ele) = rhoV/rho;
  tavg_curr(spt,7,ele) = rhoW/rho;

  // Pressure
  tavg_curr(spt,8,ele) = (gamma-1.0) *
      (rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV + rhoW*rhoW) / rho);

  double fac = 0.5 * (curr_time - prev_time);
  for (int i = 0; i < nVars+nDims+1; i++)
    tavg_acc(spt, i, ele) += fac * (tavg_prev(spt,i,ele) + tavg_curr(spt,i,ele));
}

void accumulate_time_averages_wrapper(mdvector_gpu<double> &tavg_acc, mdvector_gpu<double> &tavg_prev,
    mdvector_gpu<double> &tavg_curr, mdvector_gpu<double> &U_spts, double prev_time,
    double curr_time, double gamma, int nSpts, int nVars, int nDims, int nEles)
{
  int threads = 192;
  int blocks = (nEles * nSpts + threads - 1) / threads;

  accumulate_time_averages<<<blocks,threads>>>(tavg_acc,tavg_prev,tavg_curr,U_spts,
      prev_time,curr_time,gamma,nSpts,nVars,nDims,nEles);
}

__global__
void move_grid(mdvector_gpu<double> coords, mdvector_gpu<double> coords_0, mdvector_gpu<double> Vg,
    MotionVars params, unsigned int nNodes, unsigned int nDims, int motion_type, double time, int gridID = 0)
{
  unsigned int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= nNodes)
    return;

  switch (motion_type)
  {
    case TEST1:
    {
      double t0 = 10;
      double Atx = 2;
      double Aty = 2;
      double DX = 5;
      double DY = 5;
      /// Taken from Kui, AIAA-2010-5031-661
      double x0 = coords_0(node,0); double y0 = coords_0(node,1);
      coords(node,0) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Atx*PI*time/t0);
      coords(node,1) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Aty*PI*time/t0);
      Vg(node,0) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Atx*PI*time/t0);
      Vg(node,1) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Aty*PI*time/t0);
      break;
    }
    case TEST2:
    {
      double t0 = 10.*sqrt(5.);
      double DX = 5;
      double DY = 5;
      double DZ = 5;
      double Atx = 4;
      double Aty = 8;
      double Atz = 4;
      if (nDims == 2)
      {
        /// Taken from Liang-Miyaji
        double x0 = coords_0(node,0); double y0 = coords_0(node,1);
        coords(node,0) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Atx*PI*time/t0);
        coords(node,1) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Aty*PI*time/t0);
        Vg(node,0) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Atx*PI*time/t0);
        Vg(node,1) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Aty*PI*time/t0);
      }
      else
      {
        /// Taken from Liang-Miyaji
        double x0 = coords_0(node,0); double y0 = coords_0(node,1); double z0 = coords_0(node,2);
        coords(node,0) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Atx*PI*time/t0);
        coords(node,1) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Aty*PI*time/t0);
        coords(node,2) = z0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Atz*PI*time/t0);
        Vg(node,0) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Atx*PI*time/t0);
        Vg(node,1) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Aty*PI*time/t0);
        Vg(node,2) = Atz*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Atz*PI*time/t0);
      }
      break;
    }
    case TEST3:
    {
      if (gridID==0)
      {
        /// Liangi-Miyaji with easily-modifiable domain width
        double t0 = 10.*sqrt(5.);
        double width = 5.;
        coords(node,0) = coords_0(node,0) + sin(PI*coords_0(node,0)/width)*sin(PI*coords_0(node,1)/width)*sin(4*PI*time/t0);
        coords(node,1) = coords_0(node,1) + sin(PI*coords_0(node,0)/width)*sin(PI*coords_0(node,1)/width)*sin(8*PI*time/t0);
        Vg(node,0) = 4.*PI/t0*sin(PI*coords_0(node,0)/width)*sin(PI*coords_0(node,1)/width)*cos(4*PI*time/t0);
        Vg(node,1) = 8.*PI/t0*sin(PI*coords_0(node,0)/width)*sin(PI*coords_0(node,1)/width)*cos(8*PI*time/t0);
      }
      break;
    }
    case CIRCULAR_TRANS:
    {
      /// Rigid oscillation in a circle
      if (gridID == 0)
      {
        double Ax = params.moveAx; // Amplitude  (m)
        double Ay = params.moveAy; // Amplitude  (m)
        double fx = params.moveFx; // Frequency  (Hz)
        double fy = params.moveFy; // Frequency  (Hz)
        coords(node,0) = coords_0(node,0) + Ax*sin(2.*PI*fx*time);
        coords(node,1) = coords_0(node,1) + Ay*(1-cos(2.*PI*fy*time));
        Vg(node,0) = 2.*PI*fx*Ax*cos(2.*PI*fx*time);
        Vg(node,1) = 2.*PI*fy*Ay*sin(2.*PI*fy*time);

        if (nDims == 3)
        {
          double Az = params.moveAz;
          double fz = params.moveFz;
          coords(node,2) = coords_0(node,2) - Az*sin(2.*PI*fz*time);
          Vg(node,2) = -2.*PI*fz*Az*cos(2.*PI*fz*time);
        }
      }
      break;
    }
  }
}

void move_grid_wrapper(mdvector_gpu<double> &coords,
    mdvector_gpu<double> &coords_0, mdvector_gpu<double> &Vg, MotionVars &params,
    unsigned int nNodes, unsigned int nDims, int motion_type, double time,
    int gridID)
{
  int threads = 128;
  int blocks = (nNodes + threads - 1) / threads;
  move_grid<<<blocks, threads>>>(coords, coords_0, Vg, params, nNodes, nDims,
      motion_type, time, gridID);
}


template<unsigned int nDims>
__global__
void estimate_point_positions_nodes(mdvector_gpu<double> coord_nodes,
    mdvector_gpu<double> vel_nodes, double dt, unsigned int nNodes)
{
  const int node = blockDim.x * blockIdx.x + threadIdx.x;

  if (node >= nNodes) return;

  for (int dim = 0; dim < nDims; dim++)
    coord_nodes(node, dim) += vel_nodes(node, dim) * dt;
}

template<unsigned int nDims>
__global__
void estimate_point_positions_spts(mdvector_gpu<double> coord_spts,
    mdvector_gpu<double> vel_spts, double dt, unsigned int nSpts, unsigned int nEles)
{
  const int ele = (blockDim.x * blockIdx.x + threadIdx.x) % nEles;
  const int spt = (blockDim.x * blockIdx.x + threadIdx.x) / nEles;

  if (spt >= nSpts) return;

  for (int dim = 0; dim < nDims; dim++)
    coord_spts(spt, dim, ele) += vel_spts(spt, dim, ele) * dt;
}

template<unsigned int nDims>
__global__
void estimate_point_positions_fpts(mdvector_gpu<double> coord_fpts,
    mdvector_gpu<double> vel_fpts, double dt, unsigned int nFpts)
{
  const int fpt = blockDim.x * blockIdx.x + threadIdx.x;

  if (fpt >= nFpts) return;

  for (int dim = 0; dim < nDims; dim++)
    coord_fpts(dim, fpt) += vel_fpts(dim, fpt) * dt;
}

void estimate_point_positions_nodes_wrapper(mdvector_gpu<double> &coord_nodes,
  mdvector_gpu<double> &vel_nodes,double dt, unsigned int nNodes, unsigned int nDims)
{
  int threads = 192;
  int blocks = (nNodes + threads - 1) / threads;

  if (nDims == 2)
    estimate_point_positions_nodes<2><<<blocks,threads>>>(coord_nodes,vel_nodes,dt,nNodes);
  else
    estimate_point_positions_nodes<3><<<blocks,threads>>>(coord_nodes,vel_nodes,dt,nNodes);

  check_error();
}

void estimate_point_positions_fpts_wrapper(mdvector_gpu<double> &coord_fpts,
  mdvector_gpu<double> &vel_fpts, double dt, unsigned int nFpts, unsigned int nDims)
{
  int threads = 192;
  int blocks = (nFpts + threads - 1) / threads;

  if (nDims == 2)
    estimate_point_positions_fpts<2><<<blocks,threads>>>(coord_fpts,vel_fpts,dt,nFpts);
  else
    estimate_point_positions_fpts<3><<<blocks,threads>>>(coord_fpts,vel_fpts,dt,nFpts);

  check_error();
}

void estimate_point_positions_spts_wrapper(mdvector_gpu<double> &coord_spts,
    mdvector_gpu<double> &vel_spts, double dt, unsigned int nSpts,
    unsigned int nEles, unsigned int nDims)
{
  int threads = 192;
  int blocks = (nSpts*nEles + threads - 1) / threads;

  if (nDims == 2)
    estimate_point_positions_spts<2><<<blocks,threads>>>(coord_spts,vel_spts,dt,nSpts,nEles);
  else
    estimate_point_positions_spts<3><<<blocks,threads>>>(coord_spts,vel_spts,dt,nSpts,nEles);

  check_error();
}

__global__
void unpack_fringe_u(mdvector_gpu<double> U_fringe,
    mdview_gpu<double> U, mdview_gpu<double> U_ldg, mdvector_gpu<unsigned int> fringe_fpts,
    mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars)
{
  const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int var = tot_ind % nVars;
  const unsigned int fpt = (tot_ind / nVars) % nFpts;
  const unsigned int face = tot_ind / (nFpts * nVars);

  if (face >= nFringe)
    return;

  const unsigned int gfpt = fringe_fpts(fpt, face);
  const unsigned int side = fringe_side(fpt, face);

  double val = U_fringe(face,fpt,var);
  U(side, var, gfpt) = val; //fpt, face, var); /// TODO: look into further
  U_ldg(side, var, gfpt) = val;
}

void unpack_fringe_u_wrapper(mdvector_gpu<double> &U_fringe,
    mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, mdvector_gpu<unsigned int> &fringe_fpts,
    mdvector_gpu<unsigned int> &fringe_side, unsigned int nFringe, unsigned int nFpts,
    unsigned int nVars, int stream)
{
  int threads = 192;
  int blocks = (nFringe * nFpts * nVars + threads - 1) / threads;

  if (stream == -1)
  {
    unpack_fringe_u<<<blocks, threads>>>(U_fringe, U, U_ldg, fringe_fpts,
        fringe_side, nFringe, nFpts, nVars);
  }
  else
  {
    unpack_fringe_u<<<blocks, threads, 0, stream_handles[stream]>>>(U_fringe, U,
        U_ldg, fringe_fpts, fringe_side, nFringe, nFpts, nVars);
  }

  check_error();
}

template <unsigned nDims>
__global__
void unpack_fringe_grad(mdvector_gpu<double> dU_fringe,
    mdview_gpu<double> dU, mdvector_gpu<unsigned int> fringe_fpts,
    mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars)
{
  const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int var = tot_ind % nVars;
  const unsigned int fpt = (tot_ind / nVars) % nFpts;
  const unsigned int face = tot_ind / (nFpts * nVars);

  if (fpt >= nFpts || face >= nFringe || var >= nVars)
    return;

  const unsigned int gfpt = fringe_fpts(fpt, face);
  const unsigned int side = fringe_side(fpt, face);

  for (unsigned int dim = 0; dim < nDims; dim++)
    dU(side, dim, var, gfpt) = dU_fringe(face, fpt, dim, var);
}

void unpack_fringe_grad_wrapper(mdvector_gpu<double> &dU_fringe,
    mdview_gpu<double> &dU, mdvector_gpu<unsigned int> &fringe_fpts,
    mdvector_gpu<unsigned int> &fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims, int stream)
{
  int threads  = 192;
  int blocks = (nFringe * nFpts * nVars + threads - 1) / threads;

  if (stream == -1)
  {
    if (nDims == 2)
      unpack_fringe_grad<2><<<blocks, threads>>>(dU_fringe, dU, fringe_fpts,
                                                 fringe_side, nFringe, nFpts, nVars);

    else if (nDims == 3)
      unpack_fringe_grad<3><<<blocks, threads>>>(dU_fringe, dU, fringe_fpts,
                                                 fringe_side, nFringe, nFpts, nVars);
  }
  else
  {
    if (nDims == 2)
      unpack_fringe_grad<2><<<blocks, threads, 0, stream_handles[stream]>>>
          (dU_fringe, dU, fringe_fpts, fringe_side, nFringe, nFpts, nVars);

    else if (nDims == 3)
      unpack_fringe_grad<3><<<blocks, threads, 0, stream_handles[stream]>>>
          (dU_fringe, dU, fringe_fpts, fringe_side, nFringe, nFpts, nVars);
  }

  check_error();
}

__global__
void unpack_unblank_u(mdvector_gpu<double> U_unblank,
    mdvector_gpu<double> U_spts, mdvector_gpu<int> cellIDs,
    unsigned int nCells, unsigned int nSpts, unsigned int nVars)
{
  const unsigned int tot_ind = (blockDim.x * blockIdx.x + threadIdx.x);
  const unsigned int var = tot_ind % nVars;
  const unsigned int spt = (tot_ind / nVars) % nSpts;
  const unsigned int ic = tot_ind / (nSpts * nVars);

  if (ic >= nCells)
    return;

  const unsigned int ele = cellIDs(ic);

  U_spts(spt, var, ele) = U_unblank(ic,spt,var);
}

void unpack_unblank_u_wrapper(mdvector_gpu<double> &U_unblank,
    mdvector_gpu<double>& U_spts, mdvector_gpu<int> &cellIDs, unsigned int nCells,
    unsigned int nSpts, unsigned int nVars, int stream)
{
  int threads = 192;
  int blocks = (nCells * nSpts * nVars + threads - 1) / threads;

  if (stream == -1)
  {
    unpack_unblank_u<<<blocks, threads>>>(U_unblank, U_spts, cellIDs, nCells, nSpts, nVars);
  }
  else
  {
    unpack_unblank_u<<<blocks, threads, 0, stream_handles[stream]>>>(U_unblank, U_spts,
        cellIDs, nCells, nSpts, nVars);
  }

  check_error();
}

template<int nDims>
__global__
void pack_fringe_coords(mdvector_gpu<unsigned int> fringe_fpts, mdvector_gpu<double> xyz,
    mdvector_gpu<double> coord_fpts, int nPts)
{
  const unsigned int pt = blockDim.x * blockIdx.x + threadIdx.x;

  if (pt >= nPts)
    return;

  int gfpt = fringe_fpts(pt);

  for (unsigned int d = 0; d < nDims; d++)
    xyz(pt,d) = coord_fpts(d,gfpt);
}

void pack_fringe_coords_wrapper(mdvector_gpu<unsigned int> &fringe_fpts, mdvector_gpu<double> &xyz,
    mdvector_gpu<double> &coord_fpts, int nPts, int nDims, int stream)
{
  int threads = 192;
  int blocks = (nPts + threads - 1) / threads;

  if (stream == -1)
  {
    if (nDims == 2)
      pack_fringe_coords<2><<<blocks, threads>>>(fringe_fpts,xyz,coord_fpts,nPts);
    else
      pack_fringe_coords<3><<<blocks, threads>>>(fringe_fpts,xyz,coord_fpts,nPts);
  }
  else
  {
    if (nDims == 2)
      pack_fringe_coords<2><<<blocks, threads, 0, stream_handles[stream]>>>(fringe_fpts,
          xyz,coord_fpts,nPts);
    else
      pack_fringe_coords<3><<<blocks, threads, 0, stream_handles[stream]>>>(fringe_fpts,
          xyz,coord_fpts,nPts);
  }

  check_error();
}

template<int nDims>
__global__
void pack_cell_coords(mdvector_gpu<int> cellIDs, mdvector_gpu<double> xyz,
    mdvector_gpu<double> coord_spts, int nCells, int nSpts)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (ele >= nCells)
    return;

  int ic = cellIDs(ele);

  for (unsigned int d = 0; d < nDims; d++)
    xyz(ele,spt,d) = coord_spts(spt,d,ic);
}

void pack_cell_coords_wrapper(mdvector_gpu<int> &cellIDs, mdvector_gpu<double> &xyz,
    mdvector_gpu<double> &coord_spts, int nCells, int nSpts, int nDims, int stream)
{
  int threads = 192;
  int blocks = (nCells * nSpts + threads - 1) / threads;

  if (stream == -1)
  {
    if (nDims == 2)
      pack_cell_coords<2><<<blocks, threads>>>(cellIDs,xyz,coord_spts,nCells,nSpts);
    else
      pack_cell_coords<3><<<blocks, threads>>>(cellIDs,xyz,coord_spts,nCells,nSpts);
  }
  else
  {
    if (nDims == 2)
      pack_cell_coords<2><<<blocks, threads, 0, stream_handles[stream]>>>(cellIDs,
          xyz,coord_spts,nCells,nSpts);
    else
      pack_cell_coords<3><<<blocks, threads, 0, stream_handles[stream]>>>(cellIDs,
          xyz,coord_spts,nCells,nSpts);
  }

  check_error();
}

/*! Evaluates the Lagrange function corresponding to the specified mode on xiGrid at location xi.
 *
 * \param xiGrid The grid of interpolation points. Sorted in domain [-1,1].
 * \param mode Mode of the Lagrange function. Defined such that function is 1 at xiGrid(mode)
 * zero at other grid points.
 * \param xi  Point of evaluation in domain [-1,1].
 *
 * \return Value of Lagrange function at xi.
 */
__device__ __forceinline__
double Lagrange_gpu(double* xiGrid, int npts, double xi, int mode)
{
  double val = 1.0;

  for (int i = 0; i < mode; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  for (int i = mode + 1; i < npts; i++)
    val *= (xi - xiGrid[i])/(xiGrid[mode] - xiGrid[i]);

  return val;
}

template<int nSpts1D>
__global__
void get_nodal_basis(double* rst_in, double* weights,
    double* xiGrid, int nFringe)
{
  const int nSpts = nSpts1D*nSpts1D*nSpts1D;
  const int idx = (blockDim.x * blockIdx.x + threadIdx.x);

  if (nSpts1D == 1)
  {
    for (int i = idx; i < nFringe * nSpts; i += gridDim.x * blockDim.x)
      weights[i] = 1.0;
    return;
  }

  __shared__ double xi[nSpts1D];

  if (threadIdx.x < nSpts1D)
    xi[threadIdx.x] = xiGrid[threadIdx.x];

  __syncthreads();

  for (int i = idx; i < nFringe * nSpts; i += gridDim.x * blockDim.x)
  {
    int spt = i % nSpts;
    int ipt = i / nSpts;

    if (ipt >= nFringe) continue;

    double rst[3];
    for (int d = 0; d < 3; d++)
      rst[d] = rst_in[3*ipt+d];

    int ispt = spt % nSpts1D;
    int jspt = (spt / nSpts1D) % nSpts1D;
    int kspt = spt / (nSpts1D*nSpts1D);
    weights[nSpts*ipt+spt] = Lagrange_gpu(xi,nSpts1D,rst[0],ispt) *
        Lagrange_gpu(xi,nSpts1D,rst[1],jspt) *
        Lagrange_gpu(xi,nSpts1D,rst[2],kspt);
  }
}

void get_nodal_basis_wrapper(int* cellIDs, double* rst, double* weights,
    double* xiGrid, int nFringe, int nSpts, int nSpts1D, int stream)
{
  int threads = 128;
  int blocks = min((nFringe * nSpts + threads - 1) / threads, MAX_GRID_DIM);
  int nbShare = nSpts1D*sizeof(double);

  if (stream == -1)
  {
    switch (nSpts1D)
    {
      case 1:
        get_nodal_basis<1><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 2:
        get_nodal_basis<2><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 3:
        get_nodal_basis<3><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 4:
        get_nodal_basis<4><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 5:
        get_nodal_basis<5><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      case 6:
        get_nodal_basis<6><<<blocks, threads, nbShare>>>(rst,weights,xiGrid,nFringe);
        break;
      default:
        ThrowException("nSpts1D case not implemented");
    }
  }
  else
  {
    switch (nSpts1D)
    {
      case 1:
        get_nodal_basis<1><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 2:
        get_nodal_basis<2><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 3:
        get_nodal_basis<3><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 4:
        get_nodal_basis<4><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 5:
        get_nodal_basis<5><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      case 6:
        get_nodal_basis<6><<<blocks, threads, nbShare, stream_handles[stream]>>>
            (rst,weights,xiGrid,nFringe);
        break;
      default:
        ThrowException("nSpts1D case not implemented");
    }
  }

  check_error();
}
