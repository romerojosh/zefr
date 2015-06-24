#ifndef faces_kernels_h
#define faces_kernels_h

#include "mdvector_gpu.h"

void compute_Fconv_fpts_2D_EulerNS_wrapper(mdvector_gpu<double> F_gfpts, 
    mdvector_gpu<double> U_gfpts, mdvector_gpu<double> P_gfpts, unsigned int nFpts, 
    double gamma);

void apply_bcs_wrapper(mdvector_gpu<double> U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nVars, unsigned int nDims, double rho_fs, mdvector_gpu<double> V_fs, 
    double P_fs, double gamma, double R_ref, double T_tot_fs, double P_tot_fs, double T_wall, 
    mdvector_gpu<double> V_wall, mdvector_gpu<double> norm_fs, mdvector_gpu<double> norm, 
    mdvector_gpu<unsigned int> gfpt2bnd, mdvector_gpu<unsigned int> per_fpt_list); 

void rusanov_flux_wrapper(mdvector_gpu<double> U, mdvector_gpu<double> Fconv, 
    mdvector_gpu<double> Fcomm, mdvector_gpu<double> P, mdvector_gpu<double> norm,
    mdvector_gpu<int>outnorm, mdvector_gpu<double> waveSp, double gamma, double rus_k,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims);

void compute_common_U_LDG_wrapper(mdvector_gpu<double> U, mdvector_gpu<double> Ucomm, 
    mdvector_gpu<double> norm, double beta, unsigned int nFpts, unsigned int nVars);

void transform_flux_faces_wrapper(mdvector_gpu<double> Fcomm, mdvector_gpu<double> dA, 
    unsigned int nFpts, unsigned int nVars);

#endif /* faces_kernels_h */
