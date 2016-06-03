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

static const unsigned int MAX_GRID_DIM = 65535;

/* Create handles for default (0) and concurrent (1-16) streams */
static std::vector<cublasHandle_t> cublas_handles(17);
static std::vector<cudaStream_t> stream_handles(17);

void check_error()
{
#ifndef _NO_CUDA_ERROR
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    ThrowException(cudaGetErrorString(err));
  }
#endif
}

void start_cublas()
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

template <typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size, int stream)
{
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

__global__
void copy_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{

  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  vec1(idx) = vec2(idx);

}

void device_copy(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = (size + threads - 1) /threads;
  copy_kernel<<<blocks, threads>>>(vec1, vec2, size);
}

__global__
void add_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  vec1(idx) += vec2(idx);
}

void device_add(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = (size + threads - 1) /threads;
  add_kernel<<<blocks, threads>>>(vec1, vec2, size);
}

__global__
void subtract_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  vec1(idx) -= vec2(idx);
}
void device_subtract(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = (size + threads - 1) /threads;
  subtract_kernel<<<blocks, threads>>>(vec1, vec2, size);
}

void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream)
{
  cublasDgemm(cublas_handles[stream], CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
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
void U_to_faces(mdvector_gpu<double> U_fpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> Ucomm, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nEles, unsigned int nFpts, bool viscous, unsigned int startEle, unsigned int endEle)
{  
  const unsigned int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts + startEle;

  if (fpt >= nFpts || ele >= endEle)
    return;

  int gfpt = fpt2gfpt(fpt,ele);
  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
  {
    if (viscous) // if viscous, put extrapolated solution into Ucomm
    {
      for (unsigned int var = 0; var < nVars; var++)
        Ucomm(fpt, ele, var) = U_fpts(fpt, ele, var);
    }
    return;
  }

  int slot = fpt2gfpt_slot(fpt,ele);

  for (unsigned int var = 0; var < nVars; var++)
    U_gfpts(gfpt, var, slot) = U_fpts(fpt, ele, var);

}

void U_to_faces_wrapper(mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &U_gfpts, 
    mdvector_gpu<double> &Ucomm, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation, 
    bool viscous, unsigned int startEle, unsigned int endEle)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nFpts * (endEle - startEle)) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    U_to_faces<1><<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, viscous, startEle, endEle);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      U_to_faces<4><<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, viscous, startEle, endEle);
    else if (nDims == 3)
      U_to_faces<5><<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, viscous, startEle, endEle);
  }
}

template <unsigned int nVars>
__global__
void U_from_faces(mdvector_gpu<double> Ucomm_gfpts, mdvector_gpu<double> Ucomm_fpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nEles, unsigned int nFpts, unsigned int startEle, unsigned int endEle)
{
  const unsigned int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts + startEle;

  if (fpt >= nFpts || ele >= endEle)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
    return;

  int slot = fpt2gfpt_slot(fpt,ele);

  for (unsigned int var = 0; var < nVars; var++)
    Ucomm_fpts(fpt, ele, var) = Ucomm_gfpts(gfpt, var, slot);

}

void U_from_faces_wrapper(mdvector_gpu<double> &Ucomm_gfpts, mdvector_gpu<double> &Ucomm_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation,
    unsigned int startEle, unsigned int endEle)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nFpts * (endEle - startEle)) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
    U_from_faces<1><<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, startEle, endEle);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      U_from_faces<4><<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, startEle, endEle);
    else
      U_from_faces<5><<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, startEle, endEle);
  }

}

template <unsigned int nVars, unsigned int nDims>
__global__
void dU_to_faces(mdvector_gpu<double> dU_fpts, mdvector_gpu<double> dU_gfpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nEles, unsigned int nFpts)
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

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    for (unsigned int var = 0; var < nVars; var++)
    {
      dU_gfpts(gfpt, var, dim, slot) = dU_fpts(fpt, ele, var, dim);
    }
  }

}

void dU_to_faces_wrapper(mdvector_gpu<double> &dU_fpts, mdvector_gpu<double> &dU_gfpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nFpts * nEles) + threads - 1)/ threads;
  

  if (equation == AdvDiff || equation == Burgers)
  {
    if (nDims == 2)
      dU_to_faces<1, 2><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
    else
      dU_to_faces<1, 3><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      dU_to_faces<4, 2><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
    else
      dU_to_faces<5, 3><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
  }
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
  unsigned int threads= 192;
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
void RK_update(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_ini, 
    mdvector_gpu<double> divF, mdvector_gpu<double> jaco_det_spts, mdvector_gpu<double> dt_in, 
    mdvector_gpu<double> rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int stage, unsigned int nStages, bool last_stage)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
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
    double sum[nVars];
    for (unsigned int var = 0; var < nVars; var++)
      sum[var] = 0.;

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
    unsigned int nStages, bool last_stage)
{
  unsigned int threads = 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
      RK_update<1><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      RK_update<4><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
    else
      RK_update<5><<<blocks, threads>>>(U_spts, U_ini, divF, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
  }
}

template <unsigned int nVars>
__global__
void RK_update_source(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_ini, 
    mdvector_gpu<double> divF, mdvector_gpu<double> source, mdvector_gpu<double> jaco_det_spts, 
    mdvector_gpu<double> dt_in, mdvector_gpu<double> rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int stage, unsigned int nStages, 
    bool last_stage)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
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
    double sum[nVars];
    for (unsigned int var = 0; var < nVars; var++)
      sum[var] = 0.;

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
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage)
{
  unsigned int threads = 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff || equation == Burgers)
  {
      RK_update_source<1><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      RK_update_source<4><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
    else
      RK_update_source<5><<<blocks, threads>>>(U_spts, U_ini, divF, source, jaco_det_spts, dt, 
          rk_coeff, dt_type, nSpts, nEles, stage, nStages, last_stage);
  }
}

template <unsigned int nDims>
__global__
void compute_element_dt(mdvector_gpu<double> dt, mdvector_gpu<double> waveSp_gfpts, mdvector_gpu<double> diffCo_gfpts,
    mdvector_gpu<double> dA, mdvector_gpu<int> fpt2gfpt, mdvector_gpu<double> weights_spts,
    mdvector_gpu<double> vol, mdvector_gpu<double> h_ref, unsigned int nSpts1D, double CFL, double beta, int order, int CFL_type,
    unsigned int nFpts, unsigned int nEles)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles)
    return;

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
    /* Compute inverse of timestep in each face */
    double dtinv[2*nDims] = {0};
    for (unsigned int face = 0; face < 2*nDims; face++)
    {
      for (unsigned int fpt = face * nSpts1D; fpt < (face+1) * nSpts1D; fpt++)
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
    dtinv[0] = max(dtinv[0], dtinv[2]);
    dtinv[1] = max(dtinv[1], dtinv[3]);

    dt(ele) = CFL / (dtinv[0] + dtinv[1]);
  }
}

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, mdvector_gpu<double> &diffCo_gfpts,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<double> &weights_spts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nSpts1D, double CFL, double beta, int order, unsigned int dt_type, unsigned int CFL_type,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1) / threads;

  if (nDims == 2)
  {
    compute_element_dt<2><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, weights_spts, vol, h_ref,
        nSpts1D, CFL, beta, order, CFL_type, nFpts, nEles);
  }
  else
  {
    compute_element_dt<3><<<blocks, threads>>>(dt, waveSp_gfpts, diffCo_gfpts, dA, fpt2gfpt, weights_spts, vol, h_ref,
        nSpts1D, CFL, beta, order, CFL_type, nFpts, nEles);
  }

  if (dt_type == 1)
  {
    /* Get min dt using thrust (pretty slow) */
    thrust::device_ptr<double> dt_ptr = thrust::device_pointer_cast(dt.data());
    thrust::device_ptr<double> min_ptr = thrust::min_element(dt_ptr, dt_ptr + nEles);

#ifdef _MPI
    double min_dt = min_ptr[0];
    MPI_Allreduce(MPI_IN_PLACE, &min_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    dt_ptr[0] = min_dt;
#else
    dt_ptr[0] = min_ptr[0];
    //thrust::copy(min_ptr, min_ptr+1, dt_ptr);
#endif

  }
}

template<unsigned int nVars, unsigned int nDims>
__global__
void add_source(mdvector_gpu<double> divF_spts, mdvector_gpu<double> jaco_det_spts, mdvector_gpu<double> coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int equation, 
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts + startEle;

  if (spt >= nSpts || ele >= endEle)
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
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle)
{
  unsigned int threads = 192;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  if (nDims == 2)
  {
    if (equation == AdvDiff || equation == Burgers)
      add_source<1, 2><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
    else
      add_source<4, 2><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
  }
  else
  {
    if (equation == AdvDiff || equation == Burgers)
      add_source<1, 3><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
    else
      add_source<5, 3><<<blocks, threads>>>(divF_spts, jaco_det_spts, coord_spts, nSpts, nEles, equation,
          flow_time, stage, startEle, endEle);
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
  unsigned int threads = 192;
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
  unsigned int threads = 192;
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
  unsigned int threads = 192;
  unsigned int blocks = (nSpts * (endEle - startEle) + threads - 1)/ threads;

  compute_U<<<blocks, threads>>>(U_spts, deltaU, nSpts, nEles, nVars, startEle, endEle);
}

#ifdef _MPI
__global__
void pack_U(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> U, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int var = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (i >= nFpts || var >= nVars)
    return;

  U_sbuffs(i, var) = U(fpts(i), var, 0);
}

void pack_U_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars, int stream)
{
  unsigned int threads = 192;
  unsigned int blocks = (fpts.size() * nVars + threads - 1) / threads;

  if (stream == -1)
    pack_U<<<blocks,threads>>>(U_sbuffs, fpts, U, nVars, fpts.size());
  else
    pack_U<<<blocks,threads, 0, stream_handles[stream]>>>(U_sbuffs, fpts, U, nVars, fpts.size());
}

__global__
void unpack_U(mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> U, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int var = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (i >= nFpts || var >= nVars)
    return;

  U(fpts(i), var, 1) = U_rbuffs(i, var);
}

void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars, int stream)
{
  unsigned int threads = 192;
  unsigned int blocks = (fpts.size() * nVars + threads - 1) / threads;

  if (stream == -1)
    unpack_U<<<blocks,threads>>>(U_rbuffs, fpts, U, nVars, fpts.size());
  else
    unpack_U<<<blocks,threads, 0, stream_handles[stream]>>>(U_rbuffs, fpts, U, nVars, fpts.size());
}

template<unsigned int nDims>
__global__
void pack_dU(mdvector_gpu<double> U_sbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> dU, unsigned int nVars, unsigned int nFpts)
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
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (nDims == 2)
    pack_dU<2><<<blocks,threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
  else
    pack_dU<3><<<blocks,threads>>>(U_sbuffs, fpts, dU, nVars, fpts.size());
}

template<unsigned int nDims>
__global__
void unpack_dU(mdvector_gpu<double> U_rbuffs, mdvector_gpu<unsigned int> fpts, 
    mdvector_gpu<double> dU, unsigned int nVars, unsigned int nFpts)
{
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int var = blockDim.y * blockIdx.y + threadIdx.y;

  if (i >= nFpts || var >= nVars)
    return;

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dU(fpts(i), var, dim, 1) = U_rbuffs(i, var, dim);
  }
}

void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims)
{
  dim3 threads(32,4);
  dim3 blocks((fpts.size() + threads.x - 1)/threads.x, (nVars + threads.y - 1)/threads.y);

  if (nDims == 2)
    unpack_dU<2><<<blocks,threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
  else 
    unpack_dU<3><<<blocks,threads>>>(U_rbuffs, fpts, dU, nVars, fpts.size());
}

#endif
