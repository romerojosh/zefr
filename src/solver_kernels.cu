#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "macros.hpp"
#include "mdvector_gpu.h"
#include "solver_kernels.h"

void check_error()
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    ThrowException(cudaGetErrorString(err));
  }
}

static cublasHandle_t cublas_handle;
void start_cublas()
{
  cublasCreate(&cublas_handle);
}

template <typename T>
void allocate_device_data(T* &device_data, unsigned int size)
{
  cudaMalloc((void**)&device_data, size*sizeof(T));
  check_error();
}

template void allocate_device_data<double>(double* &device_data, unsigned int size);
template void allocate_device_data<unsigned int>(unsigned int* &device_data, unsigned int size);
template void allocate_device_data<int>(int* &device_data, unsigned int size);


template <typename T>
void free_device_data(T* &device_data)
{
  cudaFree(device_data);
  check_error();
}

template void free_device_data<double>(double* &device_data);
template void free_device_data<unsigned int>(unsigned int* &device_data);
template void free_device_data<int>(int* &device_data);

template <typename T>
void copy_to_device(T* device_data, const T* host_data, unsigned int size)
{
  cudaMemcpy(device_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice);
  check_error();
}

template void copy_to_device<double>(double* device_data, const double* host_data, unsigned int size);
template void copy_to_device<unsigned int>(unsigned int* device_data, const unsigned int* host_data, unsigned int size);
template void copy_to_device<int>(int* device_data, const int* host_data, unsigned int size);

template <typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size)
{
  cudaMemcpy(host_data, device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
  check_error();
}

template void copy_from_device<double>(double* host_data, const double* device_data, unsigned int size);
template void copy_from_device<unsigned int>(unsigned int* host_data, const unsigned int* device_data, unsigned int size);
template void copy_from_device<int>(int* host_data, const int* device_data, unsigned int size);

void cublasDGEMM_wrapper(int M, int N, int K, const double *alpha, const double* A, int lda, const double* B, int ldb,
    const double* beta, double *C, int ldc)
{
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__
void test_access(mdvector_gpu<double> vec, double val)
{
  vec(1,1) = val;
  vec(2,1) = val;
  vec(1,2) = val;
}

void test_access_wrapper(mdvector_gpu<double> vec, double val)
{
  test_access<<<1,1>>>(vec, val);
}

__global__
void U_to_faces(mdvector_gpu<double> U_fpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> Ucomm, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, bool viscous)
{  
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int var = blockDim.z * blockIdx.z + threadIdx.z;

  if (fpt >= nFpts || ele >= nEles || var >= nVars)
    return;

  int gfpt = fpt2gfpt(fpt,ele);
  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
  {
    if (viscous) // if viscous, put extrapolated solution into Ucomm
      Ucomm(fpt, ele, var) = U_fpts(fpt, ele, var);

    return;
  }

  int slot = fpt2gfpt_slot(fpt,ele);

  U_gfpts(slot, var, gfpt) = U_fpts(fpt, ele, var);

}

void U_to_faces_wrapper(mdvector_gpu<double> U_fpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> Ucomm, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, bool viscous)
{
  dim3 threads(16,16,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/threads.y, (nVars + threads.z - 1)/threads.z);

  U_to_faces<<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nVars, nEles, nFpts, viscous);
}

__global__
void U_from_faces(mdvector_gpu<double> Ucomm_gfpts, mdvector_gpu<double> Ucomm_fpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int var = blockDim.z * blockIdx.z + threadIdx.z;

  if (fpt >= nFpts || ele >= nEles || var >= nVars)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
    return;

  int slot = fpt2gfpt_slot(fpt,ele);

  Ucomm_fpts(fpt, ele, var) = Ucomm_gfpts(slot, var, gfpt);

}

void U_from_faces_wrapper(mdvector_gpu<double> Ucomm_gfpts, mdvector_gpu<double> Ucomm_fpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts)
{
  dim3 threads(16,16,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/threads.y, (nVars + threads.z - 1)/threads.z);

  U_from_faces<<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nVars, nEles, nFpts);
}

__global__
void dU_to_faces(mdvector_gpu<double> dU_fpts, mdvector_gpu<double> dU_gfpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims)
{
  const unsigned int fpt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int var = blockDim.z * blockIdx.z + threadIdx.z;

  if (fpt >= nFpts || ele >= nEles || var >= nVars)
    return;

  int gfpt = fpt2gfpt(fpt,ele);

  /* Check if flux point is on ghost edge */
  if (gfpt == -1)
    return;

  int slot = fpt2gfpt_slot(fpt,ele);

  for (unsigned int dim = 0; dim < nDims; dim++)
  {
    dU_gfpts(slot, var, dim, gfpt) = dU_fpts(fpt, ele, var, dim);
  }

}

void dU_to_faces_wrapper(mdvector_gpu<double> dU_fpts, mdvector_gpu<double> dU_gfpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims)
{
  dim3 threads(16,16,4);
  dim3 blocks((nFpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/threads.y, (nVars + threads.z - 1)/threads.z);

  dU_to_faces<<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nVars, nEles, nFpts, nDims);
}
