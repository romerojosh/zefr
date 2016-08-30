#ifndef faces_kernels_h
#define faces_kernels_h

#include "mdvector_gpu.h"

/* Face flux kernel wrappers */
void compute_Fconv_fpts_AdvDiff_wrapper(mdvector_gpu<double> &F, 
    mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nDims,
    mdvector_gpu<double> &AdvDiff_A, unsigned int startFpt,
    unsigned int endFpt, bool overset = false, int* iblank = NULL);

void compute_Fconv_fpts_Burgers_wrapper(mdvector_gpu<double> &F, 
    mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nDims,
    unsigned int startFpt, unsigned int endFpt, bool overset = false,
    int* iblank = NULL);

void compute_Fconv_fpts_EulerNS_wrapper(mdvector_gpu<double> &F_gfpts, 
    mdvector_gpu<double> &U_gfpts, mdvector_gpu<double> &P_gfpts, 
    unsigned int nFpts, unsigned int nDims, double gamma,
    unsigned int startFpt, unsigned int endFpt, bool overset = false, 
    int* iblank = NULL);

void compute_Fvisc_fpts_AdvDiff_wrapper(mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &dU, unsigned int nFpts, unsigned int nDims, 
    double AdvDiff_D, unsigned int startFpt, unsigned int endFpt,
    bool overset = false, int* iblank = NULL);

void compute_Fvisc_fpts_EulerNS_wrapper(mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &U, mdvector_gpu<double> &dU, unsigned int nFpts, unsigned int nDims, double gamma, 
        double prandtl, double mu_in, double c_sth, double rt, bool fix_vis,
        unsigned int startFpt, unsigned int endFpt, bool overset = false, 
        int* iblank = NULL);

/* Face flux derivative kernel wrappers (Implicit Method) */
void compute_dFdUconv_fpts_AdvDiff_wrapper(mdvector_gpu<double> &dFdUconv, 
    unsigned int nFpts, unsigned int nDims, mdvector_gpu<double> &AdvDiff_A, 
    unsigned int startFpt, unsigned int endFpt);

void compute_dFdUconv_fpts_Burgers_wrapper(mdvector_gpu<double> &dFdUconv, 
    mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nDims,
    unsigned int startFpt, unsigned int endFpt);

void compute_dFdUconv_fpts_EulerNS_wrapper(mdvector_gpu<double> &dFdUconv, 
    mdvector_gpu<double> &U,unsigned int nFpts, unsigned int nDims, double gamma,
    unsigned int startFpt, unsigned int endFpt);

/* Face boundary conditions kernel wrappers */
void apply_bcs_wrapper(mdvector_gpu<double> &U, unsigned int nFpts, unsigned int nGfpts_int, 
    unsigned int nGfpts_bnd, unsigned int nVars, unsigned int nDims, double rho_fs, 
    mdvector_gpu<double> &V_fs, double P_fs, double gamma, double R_ref, double T_tot_fs, 
    double P_tot_fs, double T_wall, mdvector_gpu<double> &V_wall, mdvector_gpu<double> &norm_fs, 
    mdvector_gpu<double> &norm, mdvector_gpu<unsigned int> &gfpt2bnd, mdvector_gpu<unsigned int> &per_fpt_list,
    mdvector_gpu<int> &LDG_bias, unsigned int equation);

void apply_bcs_dU_wrapper(mdvector_gpu<double> &dU, mdvector_gpu<double> &U, mdvector_gpu<double> &norm, 
    unsigned int nFpts, unsigned int nGfpts_int, unsigned int nGfpts_bnd, unsigned int nVars, 
    unsigned int nDims, mdvector_gpu<unsigned int> &gfpt2bnd, mdvector_gpu<unsigned int> &per_fpt_list);

/* Face boundary conditions kernel wrappers (Implicit Method) */
void apply_bcs_dFdU_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &dFdUconv, mdvector_gpu<double> &dFdUvisc,
    mdvector_gpu<double> &dUcdU, mdvector_gpu<double> &dFddUvisc, unsigned int nGfpts_int, unsigned int nGfpts_bnd, 
    unsigned int nVars, unsigned int nDims, double rho_fs, mdvector_gpu<double> &V_fs, double P_fs, double gamma, 
    mdvector_gpu<double> &norm, mdvector_gpu<unsigned int> &gfpt2bnd, unsigned int equation, bool viscous);

/* Face common value kernel wrappers */
void rusanov_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fconv, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, 
    mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<int> &LDG_bias,  mdvector_gpu<double> &dA, mdvector_gpu<double> Vg, double gamma, double rus_k, unsigned int nFpts, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt, bool motion, 
    bool overset = false, int* iblank = NULL);

void roe_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fconv, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &norm,
    mdvector_gpu<double> &waveSp, mdvector_gpu<double> &dA, double gamma, double rus_k, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt,
    bool overset = false, int* iblank = NULL);

void compute_common_U_LDG_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars,
    unsigned int nDims, mdvector_gpu<int> &LDG_bias, unsigned int startFpt, unsigned int endFpt,
    bool overset = false, int* iblank = NULL);

void LDG_flux_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &Fvisc, 
    mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &Fcomm_temp, mdvector_gpu<double> &norm, mdvector_gpu<double> &diffCo,
    mdvector_gpu<int> &LDG_bias, mdvector_gpu<double> &dA, double AdvDiff_D, double gamma, double mu, double prandtl, 
    double beta, double tau, unsigned int nFpts, unsigned int nVars, unsigned int nDims, unsigned int equation,
    unsigned int startFpt, unsigned int endFpt, bool overset = false, int* iblank = NULL);

/* Face common value kernel wrappers (Implicit Method) */
void rusanov_dFcdU_wrapper(mdvector_gpu<double> &U, mdvector_gpu<double> &dFdUconv, 
    mdvector_gpu<double> &dFcdU, mdvector_gpu<double> &P, mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<int> &LDG_bias, double gamma, double rus_k, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt);

/* Face transformation kernel wrappers */
void transform_flux_faces_wrapper(mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &dA, 
    unsigned int nFpts, unsigned int nVar, bool overset = false, int* iblank = NULL);

/* Face transformation kernel wrappers (Implicit Method) */
void transform_dFcdU_faces_wrapper(mdvector_gpu<double> &dFcdU, mdvector_gpu<double> &dA, 
    unsigned int nFpts, unsigned int nVars);

void unpack_fringe_u_wrapper(mdvector_gpu<double> &U_fringe, mdvector_gpu<double> &U,
    mdvector_gpu<unsigned int> fringe_fpts, mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars);

void unpack_fringe_grad_wrapper(mdvector_gpu<double> &dU_fringe, mdvector_gpu<double> &dU,
    mdvector_gpu<unsigned int> fringe_fpts, mdvector_gpu<unsigned int> fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims);

#endif /* faces_kernels_h */
