#ifndef faces_kernels_h
#define faces_kernels_h

#include "mdvector_gpu.h"

/* Face boundary conditions kernel wrappers */
void apply_bcs_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, double rho_fs, 
    mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, mdvector_gpu<double> &V_wall, mdvector_gpu<double> &Vg,
    mdvector_gpu<double> &norm_fs,  mdvector_gpu<double> &norm, mdvector_gpu<char> &gfpt2bnd,
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias, unsigned int equation,
    bool motion);

void apply_bcs_dU_wrapper(mdview_gpu<double> &dU, mdview_gpu<double> &U, mdvector_gpu<double> &norm, 
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, 
    unsigned int nDims, mdvector_gpu<char> &gfpt2bnd, unsigned int equation);

/* Face boundary conditions kernel wrappers (Implicit Method) */
void apply_bcs_dFdU_wrapper(mdview_gpu<double> &U, mdvector_gpu<double> &dUbdU, mdvector_gpu<double> &ddUbddU,
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, 
    bool viscous, double rho_fs, mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_wall, 
    mdvector_gpu<double> &V_wall, mdvector_gpu<double> &norm, mdvector_gpu<char> &gfpt2bnd, unsigned int equation);

/* Face common value kernel wrappers */
void compute_common_U_LDG_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars,
    unsigned int nDims, unsigned int equation, mdvector_gpu<char> &LDG_bias, unsigned int startFpt, unsigned int endFpt,
    mdvector_gpu<char> &flip_beta, bool overset = false, int* iblank = NULL);

void compute_common_F_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, mdview_gpu<double> &dU,
    mdview_gpu<double> &Fcomm, mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, 
    mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, mdvector_gpu<double> &diffCo,
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias,  mdvector_gpu<double> &dA, mdvector_gpu<double>& Vg, double AdvDiff_D, double gamma, double rus_k, 
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, unsigned int nFpts_int, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, 
    bool viscous, mdvector_gpu<char> &flip_beta, bool motion, bool overset = false, int* iblank = NULL);

/* Face common value kernel wrappers (Implicit Method) */
void compute_common_dFdU_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &dU, mdview_gpu<double> &dFcdU, 
    mdview_gpu<double> &dUcdU, mdview_gpu<double> &dFcddU, mdvector_gpu<double> &dUbdU, mdvector_gpu<double> &ddUbddU, 
    mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias, mdvector_gpu<double> &dA, double AdvDiff_D, double gamma, 
    double rus_k, double mu, double prandtl, double beta, double tau, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int startFpt, unsigned int endFpt, bool viscous, mdvector_gpu<char> &flip_beta);

#endif /* faces_kernels_h */
