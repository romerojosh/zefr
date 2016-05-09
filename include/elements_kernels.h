#ifndef elements_kernels_h
#define elements_kernels_h

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_Fconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, mdvector_gpu<double> &AdvDiff_A, unsigned int startEle,
    unsigned int endEle);

void compute_Fconv_spts_Burgers_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, unsigned int startEle, unsigned int endEle);

void compute_Fconv_spts_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma, unsigned int startEle, unsigned int endEle);

void compute_Fvisc_spts_AdvDiff_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &dU_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, double AdvDiff_D);

void compute_Fvisc_spts_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis);

/* Element flux derivative kernel wrappers (Implicit Method) */
void compute_dFdUconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &dFdUconv_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, 
    mdvector_gpu<double> &AdvDiff_A);

void compute_dFdUconv_spts_Burgers_wrapper(mdvector_gpu<double> &dFdUconv_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims);

void compute_dFdUconv_spts_EulerNS_wrapper(mdvector_gpu<double> &dFdUconv_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma);

void add_scaled_oppD_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims);

void add_scaled_oppDiv_wrapper(mdvector_gpu<double> &LHS_tempSF, mdvector_gpu<double> &oppDiv_fpts, 
    mdvector_gpu<double> &C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles);

void add_scaled_oppDiv_times_oppE_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, mdvector_gpu<double> oppE,
    mdvector_gpu<double> C, unsigned int nSpts, unsigned int nFpts, unsigned int nVars, 
    unsigned int nEles);

void finalize_LHS_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int dt_type);

/* Element transformation kernel wrappers */
void transform_dU_quad_wrapper(mdvector_gpu<double> &dU_spts, 
    mdvector_gpu<double> &jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int equation);

void transform_dU_hexa_wrapper(mdvector_gpu<double> &dU_spts, 
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int equation);

void transform_flux_quad_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation, unsigned int startEle, unsigned int endEle);

void transform_flux_hexa_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation);

/* Element transformation kernel wrappers (Implicit Method) */
void transform_dFdU_quad_wrapper(mdvector_gpu<double> &dFdU_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation);

/* Additional wrappers */
void compute_Uavg_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &Uavg, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &weights_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, int order);

void poly_squeeze_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims);

#endif /* elements_kernels_h */
