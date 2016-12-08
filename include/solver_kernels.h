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
template<typename T>
class mdview_gpu;

//! For ease of access to moving-grid params in CUDA
struct MotionVars
{
  double moveAx, moveAy, moveAz;
  double moveFx, moveFy, moveFz;
};

/* TODO: Move these general operators to a different file (aux_kernels.h/cu) */
#define check_error() \
{ \
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) \
  { \
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << std::endl; \
    ThrowException(cudaGetErrorString(err)); \
  } \
}

void initialize_cuda();

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

void sync_stream(unsigned int stream);
void event_record(unsigned int event, unsigned int stream);
void stream_wait_event(unsigned int event, unsigned int stream);

/* Wrapper for cublas DGEMM */
void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc, unsigned int stream = 0);

// cublasDGEMM with transposed 'A'
void cublasDGEMM_transA_wrapper(int M, int N, int K, const double alpha, const double* A,
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
void dFcdU_from_faces_wrapper(mdvector_gpu<double> &dFcdU_gfpts, mdvector_gpu<double> &dFcdU_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<char> &fpt2gfpt_slot, mdvector_gpu<char> &gfpt2bnd, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation);

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage, bool overset = false, int* iblank = NULL);

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage, 
    bool overset = false, int* iblank = NULL);

void LSRK_update_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double>& rk_err,
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, double dt,
    double ai, double bi, double bhi, unsigned int nSpts, unsigned int nEles,
    unsigned int nVars, unsigned int stage, unsigned int nStages,
    bool overset = false, int* iblank = NULL);

void LSRK_update_source_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source,
    mdvector_gpu<double> &jaco_det_spts, double dt, double ai, double bi,
    double bhi, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int stage, unsigned int nStages, bool overset = false,
    int* iblank = NULL);

double get_rk_error_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err, uint nSpts,
    uint nEles, uint nVars, double atol, double rtol, _mpi_comm comm_in,
    bool overset = false, int* iblank = NULL);

double set_adaptive_dt_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &dt_in, double& dt_out, uint nSpts, uint nEles,
    uint nVars, double atol, double rtol, double expa, double expb,
    double minfac, double maxfac, double sfact, double prev_err,
    _mpi_comm comm_in, bool overset = false, int* iblank = NULL);

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, mdvector_gpu<double> &diffCo_gfpts,
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nFptsPerFace, double CFL, double beta, int order, unsigned int dt_type, unsigned int CFL_type,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims, _mpi_comm comm_in,
    bool overset = false, int* iblank = NULL);

void add_source_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage, unsigned int startEle, unsigned int endEle,
    bool overset = false, int* iblank = NULL);

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
    mdview_gpu<double> &U, unsigned int nVars, int stream = -1);
void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &U, unsigned int nVars, int stream = -1, bool overset = false,
    int* iblank = NULL);
void pack_dU_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream = -1);
void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream = -1, bool overset = false,
    int* iblank = NULL);
#endif

void move_grid_wrapper(mdvector_gpu<double> &coords,
    mdvector_gpu<double>& coords_0, mdvector_gpu<double> &Vg, MotionVars *params,
    unsigned int nNodes, unsigned int nDims, int motion_type, double time,
    int gridID = 0);

#endif /* solver_kernels_h */
