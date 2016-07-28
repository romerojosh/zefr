#ifndef solver_kernels_h
#define solver_kernels_h

#ifdef _MPI
#define _mpi_comm MPI_Comm
#include "mpi.h"
#else
#define _mpi_comm int
#endif

template<typename T>
class mdvector_gpu;

/* TODO: Move these general operators to a different file (aux_kernels.h/cu) */
void check_error();

void start_cublas();

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

void sync_stream(unsigned int stream);

/* Wrapper for cublas DGEMM */
void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream = 0);

void cublasDgemmBatched_wrapper(int M, int N, int K, const double alpha, const double** Aarray,
    int lda, const double** Barray, int ldb, const double beta, double** Carray, int ldc, int batchCount);

void cublasDgemv_wrapper(int M, int N, const double alpha, const double* A, int lda, const double* x, int incx, 
    const double beta, double *y, int incy, int stream = 0); 

void cublasDgetrfBatched_wrapper(int N, double** Aarray, int lda, int* PivotArray, int* InfoArray, int batchCount);
void cublasDgetrsBatched_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray, 
    double** Barray, int ldb, int* info, int batchCount);

void cublasDgetriBatched_wrapper(int N, const double** Aarray, int lda, int* PivotArray, double** Carray, int ldc, int* InfoArray, int batchCount);

void gaussJordanInv_wrapper(int N, double** Aarray, int lda, double** Carray, int ldc, int batchCount);

void cublasDgemvBatched_wrapper(const int M, const int N, const double alpha, const double** Aarray, int lda, const double** xarray, int incx,
    const double beta, double** yarray, int incy, int batchCount);

/* Wrappers for custom kernels */
void U_to_faces_wrapper(mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &U_gfpts, 
    mdvector_gpu<double> &Ucomm, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation, 
    bool viscous, unsigned int startEle, unsigned int endEle);

void U_from_faces_wrapper(mdvector_gpu<double> &Ucomm_gfpts, mdvector_gpu<double> &Ucomm_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation,
    unsigned int startEle, unsigned int endEle);

void dU_to_faces_wrapper(mdvector_gpu<double> &dU_fpts, mdvector_gpu<double> &dU_gfpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation);

void dFcdU_from_faces_wrapper(mdvector_gpu<double> &dFcdU_gfpts, mdvector_gpu<double> &dFcdU_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, mdvector_gpu<unsigned int> &gfpt2bnd, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation);

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage);

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage);

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, mdvector_gpu<double> &diffCo_gfpts,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<double> &weights_spts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nSpts1D, double CFL, double beta, int order, unsigned int dt_type, unsigned int CFL_type,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims, _mpi_comm comm_in);

void add_source_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle);

void compute_RHS_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt_in,
    mdvector_gpu<double> &b, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle);

void compute_RHS_source_wrapper(mdvector_gpu<double> &divF_spts, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt_in, mdvector_gpu<double> &b, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle);

void compute_U_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int startEle, unsigned int endEle);

#ifdef _MPI
void pack_U_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars, int stream = -1);
void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &U, unsigned int nVars, int stream = -1);
void pack_dU_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims);
void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdvector_gpu<double> &dU, unsigned int nVars, unsigned int nDims);
#endif


#endif /* solver_kernels_h */
