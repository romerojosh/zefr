#ifndef ELEMENTS_KERNELS_H
#define ELEMENTS_KERNELS_H

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_Fconv_2D_EulerNS_wrapper(mdvector_gpu<double> F_spts_d, mdvector_gpu<double> U_spts_d, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, double gamma);

#endif /* ELEMENTS_KERNELS_H */
