#ifndef elements_kernels_h
#define elements_kernels_h

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_Fconv_spts_AdvDiff_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles, 
    unsigned int nDims, mdvector_gpu<double> &AdvDiff_A);

void compute_Fconv_spts_2D_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts,  unsigned int nEles, 
    double gamma);

void compute_Fconv_spts_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts, unsigned int nEles,
    unsigned int nDims, double gamma);

void compute_Fvisc_spts_2D_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, 
    unsigned int nSpts, unsigned int nEles, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis);

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
    unsigned int equation);

void transform_flux_hexa_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation);

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
