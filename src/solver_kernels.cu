#include <iostream>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "input.hpp"
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

__global__
void copy_kernel(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{

  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx >= size)
    return;

  vec1(idx) = vec2(idx);

}

void device_copy(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
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

void device_add(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
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
void device_subtract(mdvector_gpu<double> vec1, mdvector_gpu<double> vec2, unsigned int size)
{
  unsigned int threads = 192;
  unsigned int blocks = (size + threads - 1) /threads;
  subtract_kernel<<<blocks, threads>>>(vec1, vec2, size);
}

void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc)
{
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <unsigned int nVars>
__global__
void U_to_faces(mdvector_gpu<double> U_fpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> Ucomm, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nEles, unsigned int nFpts, bool viscous)
{  
  const unsigned int fpt = (blockDim.x * blockIdx.x + threadIdx.x) % nFpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nFpts;

  if (fpt >= nFpts || ele >= nEles)
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
    bool viscous)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nFpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
    U_to_faces<1><<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, viscous);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      U_to_faces<4><<<blocks, threads>>>(U_fpts, U_gfpts, Ucomm, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts, viscous);
    else if (nDims == 3)
      ThrowException("Under Construction!");
  }
}

template <unsigned int nVars>
__global__
void U_from_faces(mdvector_gpu<double> Ucomm_gfpts, mdvector_gpu<double> Ucomm_fpts, mdvector_gpu<int> fpt2gfpt, 
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

  for (unsigned int var = 0; var < nVars; var++)
    Ucomm_fpts(fpt, ele, var) = Ucomm_gfpts(gfpt, var, slot);

}

void U_from_faces_wrapper(mdvector_gpu<double> &Ucomm_gfpts, mdvector_gpu<double> &Ucomm_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nFpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
    U_from_faces<1><<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      U_from_faces<4><<<blocks, threads>>>(Ucomm_gfpts, Ucomm_fpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
    else
      ThrowException("Under Construction");
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
  

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      dU_to_faces<1, 2><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
    else
      ThrowException("Under Construction");
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      dU_to_faces<4, 2><<<blocks, threads>>>(dU_fpts, dU_gfpts, fpt2gfpt, fpt2gfpt_slot, nEles, nFpts);
    else
      ThrowException("Under Construction");
  }
}

template <unsigned int nVars, unsigned int nDims>
__global__
void compute_divF(mdvector_gpu<double> divF, mdvector_gpu<double> dF_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int stage)
{
  const unsigned int spt = (blockDim.x * blockIdx.x + threadIdx.x) % nSpts;
  const unsigned int ele = (blockDim.x * blockIdx.x + threadIdx.x) / nSpts;

  if (spt >= nSpts || ele >= nEles)
    return;

  double sum[nVars];

  for (unsigned int var = 0; var < nVars; var++)
    sum[var] = 0.0;

  for (unsigned int dim = 0; dim < nDims; dim++)
    for (unsigned int var = 0; var < nVars; var++)
      sum[var] += dF_spts(spt, ele, var, dim);

  for (unsigned int var = 0; var < nVars; var++)
    divF(spt, ele, var, stage) = sum[var];


}

void compute_divF_wrapper(mdvector_gpu<double> &divF, mdvector_gpu<double> &dF_spts, 
    unsigned int nSpts, unsigned int nVars, unsigned int nEles, unsigned int nDims,
    unsigned int equation, unsigned int stage)
{
  unsigned int threads= 192;
  unsigned int blocks = ((nSpts * nEles) + threads - 1)/ threads;

  if (equation == AdvDiff)
  {
    if (nDims == 2)
      compute_divF<1,2><<<blocks, threads>>>(divF, dF_spts, nSpts, nEles, stage);
    else
      ThrowException("Under construction!");
  }
  else if (equation == EulerNS)
  {
    if (nDims == 2)
      compute_divF<4,2><<<blocks, threads>>>(divF, dF_spts, nSpts, nEles, stage);
    else
      ThrowException("Under construction!");
  }
}

template <unsigned int nVars>
__global__
void RK_update(mdvector_gpu<double> U_spts, mdvector_gpu<double> U_ini, 
    mdvector_gpu<double> divF, mdvector_gpu<double> jaco_det_spts, mdvector_gpu<double> dt_in, 
    mdvector_gpu<double> rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int stage, unsigned int nStages, bool last_stage)
{
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  double jaco_det = jaco_det_spts(spt,ele);

  if (!last_stage)
  {
    double coeff = rk_coeff(stage);
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, ele, var) = U_ini(spt, ele, var) - coeff * dt / 
          jaco_det * divF(spt, ele, var, stage);
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
        sum[var] -= coeff * dt / jaco_det * divF(spt, ele, var, n);
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
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/
      threads.y);

  if (equation == AdvDiff)
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
      ThrowException("Under Construction");
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
  const unsigned int spt = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int ele = blockDim.y * blockIdx.y + threadIdx.y;

  if (spt >= nSpts || ele >= nEles)
    return;

  double dt;
  if (dt_type != 2)
    dt = dt_in(0);
  else
    dt = dt_in(ele);

  double jaco_det = jaco_det_spts(spt,ele);

  if (!last_stage)
  {
    double coeff = rk_coeff(stage);
    for (unsigned int var = 0; var < nVars; var ++)
      U_spts(spt, ele, var) = U_ini(spt, ele, var) - coeff * dt / 
          jaco_det * (divF(spt, ele, var, stage) + source(spt, ele, var));
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
        sum[var] -= coeff * dt / jaco_det * (divF(spt, ele, var, n) + source(spt, ele, var));
      }
    }

    for (unsigned int var = 0; var < nVars; var++)
      U_spts(spt,ele,var) += sum[var];

  }
}

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage)
{
  dim3 threads(16,12);
  dim3 blocks((nSpts + threads.x - 1)/threads.x, (nEles + threads.y - 1)/
      threads.y);

  if (equation == AdvDiff)
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
      ThrowException("Under Construction");
  }
}

__device__
double get_cfl_limit_dev(int order)
{
  switch(order)
  {
    case 0:
      return 1.393;

    case 1:
      return 0.464; 

    case 2:
      return 0.235;

    case 3:
      return 0.139;

    case 4:
      return 0.100;

    case 5:
      return 0.068;
  }
}


__global__
void compute_element_dt(mdvector_gpu<double> dt, mdvector_gpu<double> waveSp_gfpts, 
    mdvector_gpu<double> dA, mdvector_gpu<int> fpt2gfpt, double CFL, int order, 
    unsigned int nFpts, unsigned int nEles)
{
  const unsigned int ele = blockDim.x * blockIdx.x + threadIdx.x;

  if (ele >= nEles)
    return;

  double waveSp_max = 0.0;

  /* Compute maximum wavespeed */
  for (unsigned int fpt = 0; fpt <nFpts; fpt++)
  {
    /* Skip if on ghost edge. */
    int gfpt = fpt2gfpt(fpt,ele);
    if (gfpt == -1)
      continue;

    double waveSp = waveSp_gfpts(gfpt) / dA(gfpt);

    waveSp_max = max(waveSp, waveSp_max);
  }

  /* Note: CFL is applied to parent space element with width 2 */
  dt(ele) = (CFL) * get_cfl_limit_dev(order) * (2.0 / (waveSp_max+1.e-10));
}

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, double CFL, int order, 
    unsigned int dt_type, unsigned int nFpts, unsigned int nEles)
{
  unsigned int threads = 192;
  unsigned int blocks = (nEles + threads - 1) / threads;

  compute_element_dt<<<blocks, threads>>>(dt, waveSp, dA, fpt2gfpt, CFL, order, 
      nFpts, nEles);

  if (dt_type == 1)
  {
    /* Get min dt using thrust (pretty slow) */
    thrust::device_ptr<double> dt_ptr = thrust::device_pointer_cast(dt.data());
    thrust::device_ptr<double> min_ptr = thrust::min_element(dt_ptr, dt_ptr + nEles);
    //dt_ptr[0] = min_ptr[0];
    thrust::copy(min_ptr, min_ptr+1, dt_ptr);
  }

}
