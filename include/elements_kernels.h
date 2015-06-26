#ifndef elements_kernels_h
#define elements_kernels_h

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_Fconv_spts_2D_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, unsigned int nSpts,  unsigned int nEles, 
    double gamma);

void compute_Fvisc_spts_2D_EulerNS_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts, 
    unsigned int nSpts, unsigned int nEles, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis);

void transform_dU_quad_wrapper(mdvector_gpu<double> &dU_spts, 
    mdvector_gpu<double> &jaco_spts, mdvector_gpu<double> &jaco_det_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int equation);

void transform_flux_quad_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims,
    unsigned int equation);

#endif /* elements_kernels_h */
