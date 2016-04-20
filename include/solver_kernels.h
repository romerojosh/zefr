#ifndef solver_kernels_h
#define solver_kernels_h


template<typename T>
class mdvector_gpu;
template<typename T>
class spmatrix_gpu;

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
void copy_to_device(T* device_data, const T* host_data, unsigned int size);
template<typename T>
void copy_from_device(T* host_data, const T* device_data, unsigned int size);

void device_copy(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);
void device_add(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);
void device_subtract(mdvector_gpu<double> &vec1, mdvector_gpu<double> &vec2, unsigned int size);

/* Wrapper for cublas DGEMM */
void cublasDGEMM_wrapper(int M, int N, int K, const double alpha, const double* A, 
    int lda, const double* B, int ldb, const double beta, double *C, int ldc);

void cublasDgetrfBatched_wrapper(int N, double** Aarray, int lda, int* PivotArray, int* InfoArray, int batchSize);
void cublasDgetrsBatched_wrapper(int N, int NRHS, const double** Aarray, int lda, const int* PivotArray, 
    double** Barray, int ldb, int* info, int batchSize);

/* Wrappers for custom kernels */
void U_to_faces_wrapper(mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &U_gfpts, 
    mdvector_gpu<double> &Ucomm, mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, 
    unsigned int nVars, unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation, 
    bool viscous);

void U_from_faces_wrapper(mdvector_gpu<double> &Ucomm_gfpts, mdvector_gpu<double> &Ucomm_fpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation);

void dU_to_faces_wrapper(mdvector_gpu<double> &dU_fpts, mdvector_gpu<double> &dU_gfpts, 
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<int> &fpt2gfpt_slot, unsigned int nVars, 
    unsigned int nEles, unsigned int nFpts, unsigned int nDims, unsigned int equation);

void compute_divF_wrapper(mdvector_gpu<double> &divF, mdvector_gpu<double> &dF_spts, 
    unsigned int nSpts, unsigned int nVars, unsigned int nEles, unsigned int nDims,
    unsigned int equation, unsigned int stage);

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage);

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage);

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, double CFL, int order, 
    unsigned int dt_type, unsigned int nFpts, unsigned int nEles);

void add_source_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage);

void compute_RHS_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt_in,
    mdvector_gpu<double> &b, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

void compute_RHS_source_wrapper(mdvector_gpu<double> &divF_spts, mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt_in,
    mdvector_gpu<double> &b, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

void compute_deltaU_globalLHS_wrapper(spmatrix_gpu<double> &A, mdvector_gpu<double> &deltaU,  mdvector_gpu<double> &b, bool &GMRES_conv);

void compute_U_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &RHS, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

#endif /* solver_kernels_h */
