/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

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

/* Face common value kernel wrappers */
void compute_common_U_LDG_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &Ucomm, 
    mdvector_gpu<double> &norm, double beta, unsigned int nFpts, unsigned int nVars,
    unsigned int nDims, unsigned int equation, mdvector_gpu<char> &LDG_bias, unsigned int startFpt, unsigned int endFpt,
    bool overset = false, int* iblank = NULL);

void common_U_to_F_wrapper(mdview_gpu<double> &Fcomm, mdview_gpu<double> &Ucomm, mdvector_gpu<double> &norm, 
    mdvector_gpu<double> &dA, unsigned int nFpts, unsigned int nVars, unsigned int nDims, unsigned int equation,
    unsigned int startFpt, unsigned int endFpt, unsigned int dim);

void compute_common_F_wrapper(mdview_gpu<double> &U, mdview_gpu<double> &U_ldg, mdview_gpu<double> &dU,
    mdview_gpu<double> &Fcomm, mdvector_gpu<double> &P, mdvector_gpu<double> &AdvDiff_A, 
    mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, mdvector_gpu<double> &diffCo,
    mdvector_gpu<char> &rus_bias, mdvector_gpu<char> &LDG_bias,  mdvector_gpu<double> &dA, mdvector_gpu<double>& Vg, double AdvDiff_D, double gamma, double rus_k, 
    double mu, double prandtl, double rt, double c_sth, bool fix_vis, double beta, double tau, unsigned int nFpts, unsigned int nFpts_int, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int fconv_type, unsigned int fvisc_type, unsigned int startFpt, unsigned int endFpt, 
    bool viscous, bool motion, bool overset = false, int* iblank = NULL);

/* Face common value kernel wrappers (Implicit Method) */
void rusanov_dFcdU_wrapper(mdview_gpu<double> &U, mdvector_gpu<double> &dFdUconv, 
    mdvector_gpu<double> &dFcdU, mdvector_gpu<double> &P, mdvector_gpu<double> &norm, mdvector_gpu<double> &waveSp, 
    mdvector_gpu<char> &rus_bias, double gamma, double rus_k, unsigned int nFpts, unsigned int nVars, 
    unsigned int nDims, unsigned int equation, unsigned int startFpt, unsigned int endFpt);


#endif /* faces_kernels_h */
