#ifndef FACES_KERNELS_H
#define FACES_KERNELS_H

#include "mdvector_gpu.h"

void compute_Fconv_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> F_gfpts, mdvector_gpu<double> U_gfpts, mdvector_gpu<double> P_gfpts, 
    unsigned int nFpts, double gamma);

#endif /* FACES_KERNELS_H */
