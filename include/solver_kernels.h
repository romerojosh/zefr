#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H


template<typename T>
class mdvector_gpu;

void check_error();

void start_cublas();

/* Wrappers for alloc/free GPU memory */
template<typename T>
void allocate_device_data(T* &device_data, unsigned int size);
template<typename T>
void free_device_data(T* &device_data);

/* Wrappers for copy to and from GPU memory */
template<typename T>
void copy_to_device(T* device_data, const T* host_data, unsigned int size);
template<typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size);

void test_access_wrapper(mdvector_gpu<double> vec, double val);

/* Wrapper for cublas DGEMM */
void cublasDGEMM_wrapper(int M, int N, int K, const double* alpha, const double* A, int lda, const double* B, int ldb,
    const double* beta, double *C, int ldc);

/* Wrappers for custom kernels */
void U_to_faces_wrapper(mdvector_gpu<double> U_fpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> Ucomm, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, bool viscous);

void U_from_faces_wrapper(mdvector_gpu<double> Ucomm_gfpts, mdvector_gpu<double> Ucomm_fpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts);

void dU_to_faces_wrapper(mdvector_gpu<double> dU_fpts, mdvector_gpu<double> dU_gfpts, mdvector_gpu<int> fpt2gfpt, 
    mdvector_gpu<int> fpt2gfpt_slot, unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims);


#endif /* SOLVER_KERNELS_H */
