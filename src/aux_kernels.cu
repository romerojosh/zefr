#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "aux_kernels.h"
#include "macros.hpp"
#include "mdvector_gpu.h"

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

  for (int i = 0; i < N_EVENTS; i++)
    cudaEventCreateWithFlags(&event_handles[i], cudaEventDisableTiming);
}

cudaEvent_t* get_event_handle(int event)
{
  return &event_handles[event];
}

cudaStream_t* get_stream_handle(int stream)
{
  return &stream_handles[stream];
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
template void allocate_device_data<char>(char* &device_data, unsigned int size);
template void allocate_device_data<signed char>(signed char* &device_data, unsigned int size);
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
template void free_device_data<char>(char* &device_data);
template void free_device_data<signed char>(signed char* &device_data);
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
template void copy_to_device<char>(char* device_data, const char* host_data,  unsigned int size, int stream);
template void copy_to_device<signed char>(signed char* device_data, const signed char* host_data,  unsigned int size, int stream);
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
template void copy_from_device<char>(char* host_data, const char* device_data, unsigned int size, int stream);
template void copy_from_device<signed char>(signed char* host_data, const signed char* device_data, unsigned int size, int stream);

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

void event_record_wait_pair(unsigned int event, unsigned int stream_rec, unsigned int stream_wait)
{
  cudaEventRecord(event_handles[event], stream_handles[stream_rec]);
  cudaStreamWaitEvent(stream_handles[stream_wait], event_handles[event], 0);
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
  unsigned int threads = 128;
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
  unsigned int threads = 128;
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
  unsigned int threads = 128;
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
  unsigned int threads = 128;
  unsigned int blocks = min((size + threads - 1) /threads, MAX_GRID_DIM);
  fill_kernel<<<blocks, threads>>>(vec, val, size);

  check_error();
}

double device_min(mdvector_gpu<double> &vec, unsigned int size)
{
  /* Get min using thrust (pretty slow) */
  thrust::device_ptr<double> vec_ptr = thrust::device_pointer_cast(vec.data());
  thrust::device_ptr<double> min_ptr = thrust::min_element(vec_ptr, vec_ptr + size);
  check_error();
  return min_ptr[0];
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

void cublasDgetrsBatched_trans_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray,
    double** Barray, int ldb, int* info, int batchCount)
{
  cublasDgetrsBatched(cublas_handles[0], CUBLAS_OP_T, N, NRHS, Aarray, lda, PivotArray, Barray, ldb, info, batchCount);
}

void cublasDgetriBatched_wrapper(int N, const double** Aarray, int lda, int* PivotArray, double** Carray, int ldc, int* InfoArray, int batchCount)
{
  cublasDgetriBatched(cublas_handles[0], N, Aarray, lda, PivotArray, Carray, ldc, InfoArray, batchCount);
}
