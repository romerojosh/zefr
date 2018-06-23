#ifndef aux_kernels_h
#define aux_kernels_h

#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

//! For ease of access to moving-grid params in CUDA
struct MotionVars
{
  double moveAx, moveAy, moveAz;
  double moveFx, moveFy, moveFz;
};

template<typename T>
class mdvector_gpu;
template<typename T>
class mdview_gpu;

static const unsigned int MAX_GRID_DIM = 65535;

#define N_EVENTS 6
/* Create handles for default (0) and concurrent (1-16) streams */
static std::vector<cublasHandle_t> cublas_handles(17);
static std::vector<cudaStream_t> stream_handles(17);
static std::vector<cudaEvent_t> event_handles(N_EVENTS);

void initialize_cuda();

cudaEvent_t* get_event_handle(int event);
cudaStream_t* get_stream_handle(int stream);

/* Wrappers for alloc/free GPU memory */
template<typename T>
void allocate_device_data(T* &device_data, unsigned int size);
template<typename T>
void free_device_data(T* &device_data);

/* Wrappers for copy to and from GPU memory */
template<typename T>
void copy_to_device(T* device_data, const T* host_data, unsigned int size, int stream = -1);
template<typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size, int stream = -1);

void device_copy(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);
void device_add(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);
void device_subtract(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);
void device_fill(mdvector_gpu<double> &vec, unsigned int size, double val = 0.);
double device_min(mdvector_gpu<double> &vec, unsigned int size);

void sync_stream(unsigned int stream);
void event_record(unsigned int event, unsigned int stream);
void stream_wait_event(unsigned int stream, unsigned int event);
void event_record_wait_pair(unsigned int event, unsigned int stream_rec, unsigned int stream_wait);

/* Wrapper for cublas DGEMM */
void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A,
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream = 0);

// cublasDGEMM with transposed 'A'
void cublasDGEMM_transA_wrapper(int M, int N, int K, const double alpha, const double* A,
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream = 0);

// cublasDGEMM with transposed 'B'
void cublasDGEMM_transB_wrapper(int M, int N, int K, const double alpha, const double* A,
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream = 0);

void cublasDgemmBatched_wrapper(int M, int N, int K, const double alpha, const double** Aarray,
    int lda, const double** Barray, int ldb, const double beta, double** Carray, int ldc, int batchCount);

void cublasDgemv_wrapper(int M, int N, const double alpha, const double* A, int lda, const double* x, int incx,
    const double beta, double *y, int incy, int stream = 0);

void cublasDgetrfBatched_wrapper(int N, double** Aarray, int lda, int* PivotArray, int* InfoArray, int batchCount);
void cublasDgetrsBatched_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray,
    double** Barray, int ldb, int* info, int batchCount);

// cublas batched DGETRS with transposed 'A'
void cublasDgetrsBatched_trans_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray,
    double** Barray, int ldb, int* info, int batchCount);

void cublasDgetriBatched_wrapper(int N, const double** Aarray, int lda, int* PivotArray, double** Carray, int ldc, int* InfoArray, int batchCount);

void gaussJordanInv_wrapper(int N, double** Aarray, int lda, double** Carray, int ldc, int batchCount);

void DgemvBatched_wrapper(const int M, const int N, const double alpha, const double** Aarray, int lda, const double** xarray, int incx,
    const double beta, double** yarray, int incy, int batchCount);

#endif /* aux_kernels_h */
