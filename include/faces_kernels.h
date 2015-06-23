#ifndef FACES_KERNELS_H
#define FACES_KERNELS_H

#include "mdvector_gpu.h"

void compute_Fconv_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> F_fpts, mdvector_gpu<double> U_fpts, mdvector_gpu<double> P, 
    unsigned int nFpts, double gamma);

#endif /* FACES_KERNELS_H */
