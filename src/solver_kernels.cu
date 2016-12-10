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

#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#ifdef _MPI
#include "mpi.h"
#endif

#include "input.hpp"
#include "macros.hpp"
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "funcs_kernels.cu"

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

#define PI 3.141592653589793

static const unsigned int MAX_GRID_DIM = 65535;

/* Create handles for default (0) and concurrent (1-16) streams */
static std::vector<cublasHandle_t> cublas_handles(17);
static std::vector<cudaStream_t> stream_handles(17);
static std::vector<cudaEvent_t> event_handles(2);

void initialize_cuda()
{
  cublasCreate(&cublas_handles[0]);
  stream_handles[0] = cudaStreamPerThread;
  cublasSetStream(cublas_handles[0], stream_handles[0]);


  for (int i = 1; i < 17; i++)
  {
    cublasCreate(&cublas_handles[i]);
    cudaStreamCreate(&stream_handles[i]);
    cublasSetStream(cublas_handles[i], stream_handles[i]);
  }

  cudaEventCreateWithFlags(&event_handles[0], cudaEventDisableTiming);
  cudaEventCreateWithFlags(&event_handles[1], cudaEventDisableTiming);
}

template <typename T>
void allocate_device_data(T* &device_data, unsigned int size)
{
  cudaMalloc((void**)&device_data, size*sizeof(T));
  check_error();
}

template void allocate_device_data<double>(double* &device_data, unsigned int size);
template void allocate_device_data<double*>(double** &device_data, unsigned int size);
template void allocate_device_data<unsigned int>(unsigned int* &device_data, unsigned int size);
template void allocate_device_data<int>(int* &device_data, unsigned int size);
template void allocate_device_data<MotionVars>(MotionVars* &device_data, unsigned int size);


template <typename T>
void free_device_data(T* &device_data)
{
  cudaFree(device_data);
  check_error();
}

template void free_device_data<double>(double* &device_data);
template void free_device_data<double*>(double** &device_data);
template void free_device_data<unsigned int>(unsigned int* &device_data);
template void free_device_data<int>(int* &device_data);
template void free_device_data<MotionVars>(MotionVars* &device_data);

template <typename T>
void copy_to_device(T* device_data, const T* host_data, unsigned int size, int stream)
{
  if (stream == -1)
  {
    cudaMemcpy(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
  }
  else 
  {
    cudaMemcpyAsync(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice, stream_handles[stream]);
  }
  check_error();
}


template void copy_to_device<double>(double* device_data, const double* host_data, unsigned int size, int stream);
template void copy_to_device<double*>(double** device_data, double* const* host_data, unsigned int size, int stream);
template void copy_to_device<unsigned int>(unsigned int* device_data, const unsigned int* host_data, unsigned int size, int stream);
template void copy_to_device<int>(int* device_data, const int* host_data,  unsigned int size, int stream);
template void copy_to_device<MotionVars>(MotionVars* device_data, const MotionVars* host_data,  unsigned int size, int stream);

template <typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size, int stream)
{
  check_error();
  if (stream == -1)
  {
    cudaMemcpy(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
  }
  else
  {
    cudaMemcpyAsync(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost, stream_handles[stream]);
  }
  check_error();
}

template void copy_from_device<double>(double* host_data, const double* device_data, unsigned int size, int stream);
template void copy_from_device<double*>(double** host_data, double* const* device_data, unsigned int size, int stream);
template void copy_from_device<unsigned int>(unsigned int* host_data, const unsigned int* device_data, unsigned int size, int stream);
template void copy_from_device<int>(int* host_data, const int* device_data, unsigned int size, int stream);

void sync_stream(unsigned int stream)
{
  cudaStreamSynchronize(stream_handles[stream]);
}

void event_record(unsigned int event, unsigned int stream)
{
  cudaEventRecord(event_handles[event], stream_handles[stream]);
}

void stream_wait_event(unsigned int stream, unsigned int event)
{
  cudaStreamWaitEvent(stream_handles[stream], event_handles[event], 0);
}

__global__
void copy_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (uint i = idx; i < size; i += gridDim.x * blockDim.x)
  {
    vec1(i) = vec2(i);
  }
}

void device_copy(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = min((size + threads - 1) /threads, MAX_GRID_DIM);
  copy_kernel<<<blocks, threads>>>(vec1, vec2, size);

  check_error();
}

__global__
void add_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  for (uint i = idx; i < size; i += gridDim.x * blockDim.x)
    vec1(i) += vec2(i);
}

void device_add(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = min((size + threads - 1) /threads, MAX_GRID_DIM);
  add_kernel<<<blocks, threads>>>(vec1, vec2, size);

  check_error();
}

__global__
void subtract_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  for (uint i = idx; i < size; i += gridDim.x * blockDim.x)
    vec1(i) -= vec2(i);
}

void device_subtract(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = min((size + threads - 1) /threads, MAX_GRID_DIM);
  subtract_kernel<<<blocks, threads>>>(vec1, vec2, size);

  check_error();
}

__global__
void fill_kernel(mdvector_gpu<double> vec, double val, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  for (uint i = idx; i < size; i += gridDim.x * blockDim.x)
    vec(i) = val;
}

void device_fill(mdvector_gpu<double> &vec, unsigned int size, double val)
{
  unsigned int threads = 192;
  unsigned int blocks = min((size + threads - 1) /threads, MAX_GRID_DIM);
  fill_kernel<<<blocks, threads>>>(vec, val, size);

  check_error();
}

void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream)
{
  cublasDgemm(cublas_handles[stream], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cublasDGEMM_transA_wrapper(int M, int N, int K, const double alpha, const double* A,
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream)
{
  cublasDgemm(cublas_handles[stream], CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cublasDGEMM_transB_wrapper(int M, int N, int K, const double alpha, const double* A,
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream)
{
  cublasDgemm(cublas_handles[stream], CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cublasDgemmBatched_wrapper(int M, int N, int K, const double alpha, const double** Aarray,
    int lda, const double** Barray, int ldb, const double beta, double** Carray, int ldc, int batchCount)
{
  cublasDgemmBatched(cublas_handles[0], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, Aarray, lda, Barray, 
      ldb, &beta, Carray, ldc, batchCount);
}

void cublasDgemv_wrapper(int M, int N, const double alpha, const double* A, int lda, const double* x, int incx, 
    const double beta, double *y, int incy, int stream)
{
  cublasDgemv(cublas_handles[stream], CUBLAS_OP_N, M, N, &alpha, A, lda, x, incx, &beta, y, incy); 
}

void cublasDgetrfBatched_wrapper(int N, double** Aarray, int lda, int* PivotArray, int* InfoArray, int batchCount)
{
  cublasDgetrfBatched(cublas_handles[0], N, Aarray, lda, PivotArray, InfoArray, batchCount);
}

void cublasDgetrsBatched_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray, 
    double** Barray, int ldb, int* info, int batchCount)
{
  cublasDgetrsBatched(cublas_handles[0], CUBLAS_OP_N, N, NRHS, Aarray, lda, PivotArray, Barray, ldb, info, batchCount);
}

void cublasDgetriBatched_wrapper(int N, const double** Aarray, int lda, int* PivotArray, double** Carray, int ldc, int* InfoArray, int batchCount)
{
  cublasDgetriBatched(cublas_handles[0], N, Aarray, lda, PivotArray, Carray, ldc, InfoArray, batchCount);
}

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
void cublasDgemvBatched_custom(const int M, const int N, const double alpha, const double** Aarray, int lda, const double** xarray, int incx,
    const double beta, double** yarray, int incy, int batchCount)
{
  const unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  for (unsigned int batch = blockDim.y * blockIdx.y + threadIdx.y; batch < batchCount; batch += gridDim.y * blockDim.y)
  {
    for (unsigned int i = tidx; i < M; i += blockDim.x)
    { 
      double sum = 0.0;

      for (unsigned int j = 0; j < N; j++)
      {
        sum += Aarray[batch][i + j*lda] * xarray[batch][j];
      }

      yarray[batch][i * incy] = sum;
    }

    __syncthreads(); /* To avoid divergence */
  }


}

void cublasDgemvBatched_wrapper(const int M, const int N, const double alpha, const double** Aarray, int lda, const double** xarray, int incx,
    const double beta, double** yarray, int incy, int batchCount)
{
  dim3 threads(32, 6);
  dim3 blocks(1, std::min((batchCount + threads.y - 1)/threads.y, MAX_GRID_DIM));

  cublasDgemvBatched_custom<<<blocks, threads>>>(M, N, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount);
}

template <unsigned int nVars>
__global__
void dFcdU_from_faces(mdvector_gpu<double> dFcdU_gfpts, mdvector_gpu<double> dFcdU_fpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, mdvector_gpu<unsigned int> gfpt2bnd, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nEles, unsigned int nFpts)
{
  const unsigned int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (fpt >= nFpts || ele >= nEles)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
    return;

  int slot = fpt2gfpt_slot(fpt,ele);
  int notslot = 1;
  if (slot == 1)
  {
    notslot = 0;
  }

  /* Add dFcdU on non-periodic boundaries */
  if (gfpt >= (int)nGfpts_int && gfpt < (int)(nGfpts_int + nGfpts_bnd))
  {
    unsigned int bnd_id = gfpt2bnd(gfpt - nGfpts_int);
    if (bnd_id != PERIODIC)
    {
      for (unsigned int nj = 0; nj < nVars; nj++) 
      {
        for (unsigned int ni = 0; ni < nVars; ni++) 
        {
          dFcdU_fpts(fpt, ele, ni, nj, 0) = dFcdU_gfpts(gfpt, ni, nj, slot, slot) + 
                                            dFcdU_gfpts(gfpt, ni, nj, notslot, slot);
        }
      }
    }
  }
  else
  {
    for (unsigned int nj = 0; nj < nVars; nj++) 
    {
      for (unsigned int ni = 0; ni < nVars; ni++) 
      {
        dFcdU_fpts(fpt, ele, ni, nj, 0) = dFcdU_gfpts(gfpt, ni, nj, slot, slot);
        dFcdU_fpts(fpt, ele, ni, nj, 1) = dFcdU_gfpts(gfpt, ni, nj, notslot, slot);
      }
    }
  }
}

void dFcdU_from_faces_wrapper(mdvector_gpu<double> &dFcdU_gfpts, mdvector_gpu<double> &dFcdU_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, mdvector_gpu<unsigned int> &gfpt2bnd, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nFpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    dFcdU_from_faces<1><<<blocks, threads>>>(dFcdU_gfpts, dFcdU_fpts, fpt2gfpt, fpt2gfpt_slot, gfpt2bnd,
        nGfpts_int, nGfpts_bnd, nEles, nFpts);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      dFcdU_from_faces<4><<<blocks, threads>>>(dFcdU_gfpts, dFcdU_fpts, fpt2gfpt, fpt2gfpt_slot, gfpt2bnd,
          nGfpts_int, nGfpts_bnd, nEles, nFpts);
    else
      dFcdU_from_faces<5><<<blocks, threads>>>(dFcdU_gfpts, dFcdU_fpts, fpt2gfpt, fpt2gfpt_slot, gfpt2bnd,
          nGfpts_int, nGfpts_bnd, nEles, nFpts);
  }
}

template <unsigned int nVars>
__global__
void RK_update(mdvector_gpu<double> U_spts, const mdvector_gpu<double> U_ini, 
    const mdvector_gpu<double> divF, const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> dt_in, 
    const mdvector_gpu<double> rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int stage, unsigned int nStages, bool last_stage, bool overset = false, const int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  double fac = dt / jaco_det_spts(spt,ele);

  if (!last_stage)
  {
    double coeff = rk_coeff(stage);
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, ele, var) = U_ini(spt, ele, var) - coeff *  
          fac * divF(spt, ele, var, stage);
  }
  else
  {
    double sum[nVars] = {0.0};

    for (unsigned int n = 0; n < nStages; n++)
    {
      double coeff = rk_coeff(n);
      for (unsigned int var = 0; var < nVars; var++)
      {
        sum[var] -= coeff * fac * divF(spt, ele, var, n);
      }
    }

    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt,ele,var) += sum[var];

  }
}

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
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
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  double fac = dt / jaco_det_spts(spt,ele);

  if (!last_stage)
  {
    double coeff = rk_coeff(stage);
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, ele, var) = U_ini(spt, ele, var) - coeff *  
          fac * (divF(spt, ele, var, stage) + source(spt, ele, var));
  }
  else
  {
    double sum[nVars] = {0.0};;

    for (unsigned int n = 0; n < nStages; n++)
    {
      double coeff = rk_coeff(n);
      for (unsigned int var = 0; var < nVars; var++)
      {
        sum[var] -= coeff * fac * (divF(spt, ele, var, n) + source(spt, ele, var));
      }
    }

    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt,ele,var) += sum[var];

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
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
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
    unsigned int nStages, bool overset = false, int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double fac = dt / jaco_det_spts(spt,ele);

  for (unsigned int var = 0; var < nVars; var++)
    rk_err(spt, ele, var) -= (bi - bhi) * fac * divF(spt, ele, var, 0);

  if (stage < nStages - 1)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      U_spts(spt, ele, var) = U_til(spt, ele, var) - ai * fac *
          divF(spt, ele, var, 0);

      U_til(spt, ele, var) = U_spts(spt, ele, var) - (bi - ai) * fac *
          divF(spt, ele, var, 0);
    }
  }
  else
  {
    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt, ele, var) = U_til(spt, ele, var) - bi * fac *
          divF(spt, ele, var, 0);
  }
}

void LSRK_update_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, double dt,
    double ai, double bi, double bhi, unsigned int nSpts, unsigned int nEles,
    unsigned int nVars, unsigned int stage, unsigned int nStages, bool overset,
    int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  switch (nVars)
  {
    case 1:
      LSRK_update<1><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, overset,
          iblank);
      break;

    case 4:
      LSRK_update<4><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, overset,
          iblank);
      break;

    case 5:
      LSRK_update<5><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages, overset,
          iblank);
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
    unsigned int stage, unsigned int nStages, bool overset = false,
    int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset && iblank[ele] != 1)
      return;

  double fac = dt / jaco_det_spts(spt,ele);

  for (unsigned int var = 0; var < nVars; var ++)
    rk_err(spt, ele, var) -= (bi - bhi) * fac *
        (divF(spt, ele, var, 0) + source(spt, ele, var));

  if (stage != nStages - 1)
  {
    for (unsigned int var = 0; var < nVars; var ++)
    {
      U_spts(spt, ele, var) = U_til(spt, ele, var) - ai * fac *
          (divF(spt, ele, var, 0) + source(spt, ele, var));

      U_til(spt, ele, var) = U_spts(spt, ele, var) - (bi - ai) * fac *
          (divF(spt, ele, var, 0) + source(spt, ele, var));
    }
  }
  else
  {
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, ele, var) = U_til(spt, ele, var) - bi * fac *
          (divF(spt, ele, var, 0) + source(spt, ele, var));
  }
}

void LSRK_update_source_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source,
    mdvector_gpu<double> &jaco_det_spts, double dt, double ai, double bi,
    double bhi, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int stage, unsigned int nStages, bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  switch (nVars)
  {
    case 1:
      LSRK_source_update<1><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
          overset, iblank);
      break;

    case 4:
      LSRK_source_update<4><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
          overset, iblank);
      break;

    case 5:
      LSRK_source_update<5><<<blocks, threads>>>(U_spts, U_til, rk_err, divF,
          source, jaco_det_spts, dt, ai, bi, bhi, nSpts, nEles, stage, nStages,
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
void get_rk_error(mdvector_gpu<double> U_spts, const mdvector_gpu<double> U_ini,
    mdvector_gpu<double> rk_err, uint nSpts, uint nEles, double atol,
    double rtol, bool overset = false, int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  if (overset && iblank[ele] != 1)
      return;

  for (unsigned int var = 0; var < nVars; var ++)
  {
    rk_err(spt, ele, var)  =  abs(rk_err(spt, ele, var));
    rk_err(spt, ele, var) /= atol + rtol *
        max( abs(U_spts(spt, ele, var)), abs(U_ini(spt, ele, var)) );
  }
}

double get_rk_error_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err, uint nSpts,
    uint nEles, uint nVars, double atol, double rtol, _mpi_comm comm_in,
    bool overset, int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

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

double set_adaptive_dt_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &dt_in, double &dt_out, uint nSpts, uint nEles,
    uint nVars, double atol, double rtol, double expa, double expb,
    double minfac, double maxfac, double sfact, double prev_err,
    _mpi_comm comm_in, bool overset, int* iblank)
{
  double max_err = get_rk_error_wrapper(U_spts, U_ini, rk_err, nSpts, nEles,
      nVars, atol, rtol, comm_in, overset, iblank);

  // Determine the time step scaling factor and the new time step
  double fac = pow(max_err, -expa) * pow(prev_err, expb);
  fac = std::min(maxfac, std::max(minfac, sfact*fac));

  thrust::device_ptr<double> dt_ptr = thrust::device_pointer_cast(dt_in.data());
  dt_ptr[0] *= fac;

  dt_out = dt_ptr[0]; // Set value on CPU for other uses

  return max_err;
}

template <unsigned int nDims>
__global__
void compute_element_dt(mdvector_gpu<double> dt, const mdvector_gpu<double> waveSp_gfpts, const mdvector_gpu<double> diffCo_gfpts,
    const mdvector_gpu<double> dA, const mdvector_gpu<int> fpt2gfpt, const mdvector_gpu<double> weights_spts,
    const mdvector_gpu<double> vol, const mdvector_gpu<double> h_ref, unsigned int nSpts1D, double CFL, double beta, int order, int CFL_type,
    unsigned int nFpts, unsigned int nEles, bool overset = false, const int* iblank = NULL)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles)
    return;

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
      /* Skip if on ghost edge. */
      int gfpt = fpt2gfpt(fpt,ele);
      if (gfpt == -1)
        continue;

      if (nDims == 2)
      {
        int_waveSp += weights_spts(fpt % nSpts1D) * waveSp_gfpts(gfpt) * dA(gfpt);
      }
      else
      {
        int idx = fpt % (nSpts1D * nSpts1D);
        int i = idx % nSpts1D;
        int j = idx / nSpts1D;

        int_waveSp += weights_spts(i) * weights_spts(j) * waveSp_gfpts(gfpt) * dA(gfpt);
      }
    }

    dt(ele) = 2.0 * CFL * get_cfl_limit_adv_dev(order) * vol(ele) / int_waveSp;
  }

  /* CFL-estimate based on MacCormack for NS */
  else if (CFL_type == 2)
  {
    int nFptsFace = (nDims == 2) ? nSpts1D : nSpts1D*nSpts1D;
    /* Compute inverse of timestep in each face */
    double dtinv[2*nDims] = {0};
    for (unsigned int face = 0; face < 2*nDims; face++)
    {
      for (unsigned int fpt = face * nFptsFace; fpt < (face+1) * nFptsFace; fpt++)
      {
        /* Skip if on ghost edge. */
        int gfpt = fpt2gfpt(fpt,ele);
        if (gfpt == -1)
          continue;

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

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, mdvector_gpu<double> &diffCo_gfpts,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<double> &weights_spts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nSpts1D, double CFL, double beta, int order, unsigned int dt_type, unsigned int CFL_type,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims, _mpi_comm comm_in, bool overset,
    int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (nEles + threads - 1) / threads;

  if (nDims == 2)
  {
    compute_element_dt<2><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, weights_spts, vol, h_ref,
        nSpts1D, CFL, beta, order, CFL_type, nFpts, nEles);
  }
  else
  {
    compute_element_dt<3><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, weights_spts, vol, h_ref,
        nSpts1D, CFL, beta, order, CFL_type, nFpts, nEles, overset, iblank);
  }

  if (dt_type == 1)
  {
    /* Get min dt using thrust (pretty slow) */
    thrust::device_ptr<double> dt_ptr = thrust::device_pointer_cast(dt.data());
    thrust::device_ptr<double> min_ptr = thrust::min_element(dt_ptr, dt_ptr + nEles);

#ifdef _MPI
    double min_dt = min_ptr[0];
    MPI_Allreduce(MPI_IN_PLACE, &min_dt, 1, MPI_DOUBLE, MPI_MIN, comm_in);
    dt_ptr[0] = min_dt;
#else
    dt_ptr[0] = min_ptr[0];
    //thrust::copy(min_ptr, min_ptr+1, dt_ptr);
#endif

  }
}

template<unsigned int nVars, unsigned int nDims>
__global__
void add_source(mdvector_gpu<double> divF_spts, const mdvector_gpu<double> jaco_det_spts, const mdvector_gpu<double> coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int equation, 
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle,
    bool overset = false, const int* iblank = NULL)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
    return;

  if (overset)
    if (iblank[ele] != 1)
      return;

  double x = coord_spts(spt, ele, 0);
  double y = coord_spts(spt, ele, 1);
  double z = 0;
  if (nDims == 3)
    z = coord_spts(spt, ele, 2);

  double jaco_det = jaco_det_spts(spt, ele);

  for (unsigned int n = 0; n < nVars; n++)
  {
    divF_spts(spt, ele, n, stage) += compute_source_term_dev(x, y, z, flow_time, n, nDims, equation) * jaco_det;
  }
}

void add_source_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle, bool overset,
    int* iblank)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  if (nDims == 2)
  {
    if (equation == AdvDiff || equation == Burgers)
      add_source<1, 2><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
    else
      add_source<4, 2><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle, overset, iblank);
  }
  else
  {
    if (equation == AdvDiff || equation == Burgers)
      add_source<1, 3><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
    else
      add_source<5, 3><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle, overset, iblank);
  }
}

__global__
void compute_RHS(mdvector_gpu<double> divF_spts, mdvector_gpu<double> jaco_det_spts, mdvector_gpu<double> dt_in, 
    mdvector_gpu<double> RHS, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
    return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  for (unsigned int n = 0; n < nVars; n++)
  {
    RHS(spt, n, ele) = -(dt * divF_spts(spt, ele, n, 0)) / jaco_det_spts(spt, ele);
  }
}

void compute_RHS_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &RHS, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  compute_RHS<<<blocks, threads>>>(divF_spts, jaco_det_spts, dt, RHS, dt_type, nSpts, nEles, nVars, startEle, endEle);
}

__global__
void compute_RHS_source(mdvector_gpu<double> divF_spts, mdvector_gpu<double> source, mdvector_gpu<double> jaco_det_spts, mdvector_gpu<double> dt_in, 
    mdvector_gpu<double> RHS, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
    return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  for (unsigned int n = 0; n < nVars; n++)
  {
    RHS(spt, n, ele) = -(dt * (divF_spts(spt, ele, n, 0) + source(spt, ele, n))) / jaco_det_spts(spt, ele);
  }
}

void compute_RHS_source_wrapper(mdvector_gpu<double> &divF_spts, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &RHS, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  compute_RHS_source<<<blocks, threads>>>(divF_spts, source, jaco_det_spts, dt, RHS, dt_type, nSpts, nEles, nVars, startEle, endEle);
}

__global__
void compute_U(mdvector_gpu<double> U_spts, mdvector_gpu<double> deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{  
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
    return;

  for (unsigned int n = 0; n < nVars; n++)
  {
    U_spts(spt, ele, n) += deltaU(spt, n, ele);
  }

}

void compute_U_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle)
{
  unsigned int threads = 128;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  compute_U<<<blocks, threads>>>(U_spts, deltaU, nSpts, nEles, nVars, startEle, endEle);
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

  U_sbuffs(i, var) = U(fpts(i), var, 0);
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

  if (overset)
    if (iblank[gfpt] != 1)
      return;

  U(gfpt, var, 1) = U_rbuffs(i, var);
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
    U_sbuffs(i, var, dim) = dU(fpts(i), var, dim, 0);
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
    dU(fpts(i), var, dim, 1) = U_rbuffs(i, var, dim);
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
    mdvector_gpu<double> norm, mdvector_gpu<double> &dA, mdvector_gpu<uint> fpt2bnd,
    mdvector_gpu<double> weights, mdvector_gpu<double> &force, mdvector_gpu<double> &moment,
    double gamma, double rt, double c_sth, double mu_in, bool viscous, bool fix_vis, int nDims, int start_fpt,
    int nFaces, int nFptsPerFace)
{
  const unsigned int face = blockDim.x * blockIdx.x + threadIdx.x;

  if (face >= nFaces)
    return;

  const unsigned int face_start = face * nFptsPerFace + start_fpt;
  const unsigned int bnd_start = face * nFptsPerFace;

  const int c1[3] = {1,2,0}; // For cross-products
  const int c2[3] = {2,0,1};

  double taun[3];
  double tot_force[3] = {0,0,0};
  double tot_moment[3] = {0,0,0};

  unsigned int bnd_id = fpt2bnd(bnd_start);

  switch (bnd_id)
  {
    // All wall boundary conditions
    case SLIP_WALL_G:
    case SLIP_WALL_P:
    case ISOTHERMAL_NOSLIP_G:
    case ISOTHERMAL_NOSLIP_P:
    case ISOTHERMAL_NOSLIP_MOVING_G:
    case ISOTHERMAL_NOSLIP_MOVING_P:
    case ADIABATIC_NOSLIP_G:
    case ADIABATIC_NOSLIP_P:
    case ADIABATIC_NOSLIP_MOVING_G:
    case ADIABATIC_NOSLIP_MOVING_P:
    {
      for (int fpt = 0; fpt < nFptsPerFace; fpt++)
      {
        double tmp_force[3] = {0,0,0};
        int gfpt = face_start + fpt;
//        int lfpt = bnd_start + fpt;

        /* Get pressure */
        double PL = P(gfpt, 0);

        /* Sum inviscid force contributions */
        for (unsigned int dim = 0; dim < nDims; dim++)
          tmp_force[dim] = weights(fpt) * PL * norm(gfpt, dim, 0) * dA(gfpt);

        if (viscous)
        {
          if (nDims == 2)
          {
            /* Setting variables for convenience */
            /* States */
            double rho = U(gfpt, 0, 0);
            double momx = U(gfpt, 1, 0);
            double momy = U(gfpt, 2, 0);
            double e = U(gfpt, 3, 0);

            double u = momx / rho;
            double v = momy / rho;
            double e_int = e / rho - 0.5 * (u*u + v*v);

            /* Gradients */
            double rho_dx = dU(gfpt, 0, 0, 0);
            double momx_dx = dU(gfpt, 1, 0, 0);
            double momy_dx = dU(gfpt, 2, 0, 0);

            double rho_dy = dU(gfpt, 0, 1, 0);
            double momx_dy = dU(gfpt, 1, 1, 0);
            double momy_dy = dU(gfpt, 2, 1, 0);

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
            taun[0] = tauxx * norm(gfpt, 0, 0) + tauxy * norm(gfpt, 1, 0);
            taun[1] = tauxy * norm(gfpt, 0, 0) + tauyy * norm(gfpt, 1, 0);

            for (unsigned int dim = 0; dim < nDims; dim++)
              tmp_force[dim] -= weights(fpt) * taun[dim] * dA(gfpt);
          }
          else if (nDims == 3)
          {
            /* Setting variables for convenience */
            /* States */
            double rho = U(gfpt, 0, 0);
            double momx = U(gfpt, 1, 0);
            double momy = U(gfpt, 2, 0);
            double momz = U(gfpt, 3, 0);
            double e = U(gfpt, 4, 0);

            double u = momx / rho;
            double v = momy / rho;
            double w = momz / rho;
            double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

            /* Gradients */
            double rho_dx = dU(gfpt, 0, 0, 0);
            double momx_dx = dU(gfpt, 1, 0, 0);
            double momy_dx = dU(gfpt, 2, 0, 0);
            double momz_dx = dU(gfpt, 3, 0, 0);

            double rho_dy = dU(gfpt, 0, 1, 0);
            double momx_dy = dU(gfpt, 1, 1, 0);
            double momy_dy = dU(gfpt, 2, 1, 0);
            double momz_dy = dU(gfpt, 3, 1, 0);

            double rho_dz = dU(gfpt, 0, 2, 0);
            double momx_dz = dU(gfpt, 1, 2, 0);
            double momy_dz = dU(gfpt, 2, 2, 0);
            double momz_dz = dU(gfpt, 3, 2, 0);

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
            taun[0] = tauxx * norm(gfpt, 0, 0) + tauxy * norm(gfpt, 1, 0) + tauxz * norm(gfpt, 2, 0);
            taun[1] = tauxy * norm(gfpt, 0, 0) + tauyy * norm(gfpt, 1, 0) + tauyz * norm(gfpt, 2, 0);
            taun[3] = tauxz * norm(gfpt, 0, 0) + tauyz * norm(gfpt, 1, 0) + tauzz * norm(gfpt, 2, 0);

            for (unsigned int dim = 0; dim < nDims; dim++)
              tmp_force[dim] -= weights(fpt) * taun[dim] * dA(gfpt);
          }
        }

        // Add fpt's contribution to total force and moment
        for (unsigned int d = 0; d < nDims; d++)
          tot_force[d] += tmp_force[d];

        if (nDims == 3)
        {
          for (unsigned int d = 0; d < nDims; d++)
            tot_moment[d] += coord(gfpt,c1[d]) * tmp_force[c2[d]] - coord(gfpt,c2[d]) * tmp_force[c1[d]];
        }
        else
        {
          // Only a 'z' component in 2D
          tot_moment[2] += coord(gfpt,0) * tmp_force[1] - coord(gfpt,1) * tmp_force[0];
        }
      }

      // Write final sum for face to global memory
      for (int d = 0; d < nDims; d++)
      {
        force(face,d) = tot_force[d];
        moment(face,d) = tot_moment[d];
      }

      break;
    }

    default:
      // Not a wall boundary - ignore
      break;
  }
}

void compute_moments_wrapper(std::array<double,3> &tot_force, std::array<double,3> &tot_moment,
    mdview_gpu<double> &U_fpts, mdview_gpu<double> &dU_fpts, mdvector_gpu<double>& P_fpts, mdvector_gpu<double> &coord,
    mdvector_gpu<double> &norm, mdvector_gpu<double> &dA, mdvector_gpu<uint> &fpt2bnd,
    mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &force_face, mdvector_gpu<double> &moment_face,
    double gamma, double rt, double c_sth, double mu, bool viscous, bool fix_vis, int nVars, int nDims, int start_fpt, int nFaces, int nFptsPerFace)
{
  int threads = 192;
  int blocks = (nFaces + threads - 1) / threads;

  if (nVars < 4) // Only applicable to Euler / Navier-Stokes
    return;

  compute_moments<<<blocks, threads>>>(U_fpts,dU_fpts,P_fpts,coord,norm,dA,fpt2bnd,weights_fpts,
      force_face,moment_face,gamma,rt,c_sth,mu,viscous,fix_vis,nDims,start_fpt,nFaces,nFptsPerFace);

  for (int d = 0; d < nDims; d++)
  {
    thrust::device_ptr<double> f_ptr = thrust::device_pointer_cast(force_face.data()+d*nFaces);
    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(moment_face.data()+d*nFaces);
    tot_force[d] = thrust::reduce(f_ptr, f_ptr+nFaces, 0.);
    tot_moment[d] = thrust::reduce(m_ptr, m_ptr+nFaces, 0.);
  }
}

__global__
void move_grid(mdvector_gpu<double> coords, mdvector_gpu<double> coords_0, mdvector_gpu<double> Vg,
    MotionVars *params, unsigned int nNodes, unsigned int nDims, int motion_type, double time, int gridID = 0)
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
      double DX = 5;/// 0.5 * input->periodicDX; /// TODO
      double DY = 5;/// 0.5 * input->periodicDY; /// TODO
      /// Taken from Kui, AIAA-2010-5031-661
      double x0 = coords_0(0,node); double y0 = coords_0(1,node);
      coords(0,node) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Atx*PI*time/t0);
      coords(1,node) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Aty*PI*time/t0);
      Vg(0,node) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Atx*PI*time/t0);
      Vg(1,node) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Aty*PI*time/t0);
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
        double x0 = coords_0(0,node); double y0 = coords_0(1,node);
        coords(0,node) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Atx*PI*time/t0);
        coords(1,node) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(Aty*PI*time/t0);
        Vg(0,node) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Atx*PI*time/t0);
        Vg(1,node) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*cos(Aty*PI*time/t0);
      }
      else
      {
        /// Taken from Liang-Miyaji
        double x0 = coords_0(0,node); double y0 = coords_0(1,node); double z0 = coords_0(2,node);
        coords(0,node) = x0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Atx*PI*time/t0);
        coords(1,node) = y0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Aty*PI*time/t0);
        coords(2,node) = z0 + sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*sin(Atz*PI*time/t0);
        Vg(0,node) = Atx*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Atx*PI*time/t0);
        Vg(1,node) = Aty*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Aty*PI*time/t0);
        Vg(2,node) = Atz*PI/t0*sin(PI*x0/DX)*sin(PI*y0/DY)*sin(PI*z0/DZ)*cos(Atz*PI*time/t0);
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
        coords(0,node) = coords_0(0,node) + sin(PI*coords_0(0,node)/width)*sin(PI*coords_0(1,node)/width)*sin(4*PI*time/t0);
        coords(1,node) = coords_0(1,node) + sin(PI*coords_0(0,node)/width)*sin(PI*coords_0(1,node)/width)*sin(8*PI*time/t0);
        Vg(0,node) = 4.*PI/t0*sin(PI*coords_0(0,node)/width)*sin(PI*coords_0(1,node)/width)*cos(4*PI*time/t0);
        Vg(1,node) = 8.*PI/t0*sin(PI*coords_0(0,node)/width)*sin(PI*coords_0(1,node)/width)*cos(8*PI*time/t0);
      }
      break;
    }
    case CIRCULAR_TRANS:
    {
      /// Rigid oscillation in a circle
      if (gridID == 0)
      {
        double Ax = params->moveAx; // Amplitude  (m)
        double Ay = params->moveAy; // Amplitude  (m)
        double fx = params->moveFx; // Frequency  (Hz)
        double fy = params->moveFy; // Frequency  (Hz)
        coords(0,node) = coords_0(0,node) + Ax*sin(2.*PI*fx*time);
        coords(1,node) = coords_0(1,node) + Ay*(1-cos(2.*PI*fy*time));
        Vg(0,node) = 2.*PI*fx*Ax*cos(2.*PI*fx*time);
        Vg(1,node) = 2.*PI*fy*Ay*sin(2.*PI*fy*time);
      }
      break;
    }
    case RADIAL_VIBE:
    {
      /// Radial Expansion / Contraction
      if (gridID == 0) {
        double Ar = 0;///input->moveAr; /// TODO
        double Fr = 0;///input->moveFr; /// TODO
        double r = 1;///rv0(node,0) + Ar*(1. - cos(2.*pi*Fr*time)); /// TODO
        double rdot = 2.*PI*Ar*Fr*sin(2.*PI*Fr*time);
        double theta = 1;///rv0(node,1); /// TODO
        double psi = 1;///rv0(node,2); /// TODO
        coords(0,node) = r*sin(psi)*cos(theta);
        coords(1,node) = r*sin(psi)*sin(theta);
        coords(2,node) = r*cos(psi);
        Vg(0,node) = rdot*sin(psi)*cos(theta);
        Vg(1,node) = rdot*sin(psi)*sin(theta);
        Vg(2,node) = rdot*cos(psi);
      }
      break;
    }
  }
}

void move_grid_wrapper(mdvector_gpu<double> &coords,
    mdvector_gpu<double> &coords_0, mdvector_gpu<double> &Vg, MotionVars *params,
    unsigned int nNodes, unsigned int nDims, int motion_type, double time,
    int gridID)
{
  int threads = 128;
  int blocks = (nNodes + threads - 1) / threads;
  move_grid<<<blocks, threads>>>(coords, coords_0, Vg, params, nNodes, nDims,
      motion_type, time, gridID);
}
