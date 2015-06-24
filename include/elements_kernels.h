#ifndef ELEMENTS_KERNELS_H
#define ELEMENTS_KERNELS_H

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_Fconv_spts_2D_EulerNS_wrapper(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> U_spts, unsigned int nSpts,  unsigned int nEles, 
    double gamma);

void transform_flux_quad_wrapper(mdvector_gpu<double> F_spts, 
    mdvector_gpu<double> jaco_spts, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars);

#endif /* ELEMENTS_KERNELS_H */
