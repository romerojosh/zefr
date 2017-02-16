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

#ifndef elements_kernels_h
#define elements_kernels_h

#include "mdvector_gpu.h"

/* Element flux kernel wrappers */
void compute_F_wrapper(mdvector_gpu<double> &F_spts, 
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts,  mdvector_gpu<double> &inv_jaco_spts,
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nEles, unsigned int nDims, 
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous,
    bool grad_via_div, unsigned int startEle, unsigned int endEle, bool overset = false, int* iblank = NULL,
    bool motion = false);

void compute_unit_advF_wrapper(mdvector_gpu<double>& F_spts, mdvector_gpu<double>& U_spts, mdvector_gpu<double>& inv_jaco_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, unsigned int equation, unsigned int startEle, 
    unsigned int endEle, unsigned int dim);

void transform_gradF_quad_wrapper(mdvector_gpu<double> &divF_spts,
    mdvector_gpu<double> &dF_spts, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &grid_vel_spts, mdvector_gpu<double> &dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    unsigned int equation, bool overset = false, int* iblank = NULL);

void transform_gradF_hexa_wrapper(mdvector_gpu<double> &divF_spts,
    mdvector_gpu<double> &dF_spts, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &grid_vel_spts, mdvector_gpu<double> &dU_spts,
    unsigned int nSpts, unsigned int nEles, unsigned int stage,
    unsigned int equation, bool overset = false, int* iblank = NULL);

void extrapolate_Fn_wrapper(mdvector_gpu<double>& oppE,
    mdvector_gpu<double>& F_spts, mdvector_gpu<double>& tempF_fpts,
    mdvector_gpu<double>& dFn_fpts, mdvector_gpu<double>& norm,
    mdvector_gpu<double>& dA, mdvector_gpu<int>& fpt2gfpt,
    mdvector_gpu<char>& fpt2slot, unsigned int nSpts, unsigned int nFpts,
    unsigned int nEles, unsigned int nDims, unsigned int nVars, bool motion);

/* Additional wrappers */
void compute_Uavg_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &Uavg, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &weights_spts, mdvector_gpu<double> &vol, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, int order);

void poly_squeeze_wrapper(mdvector_gpu<double> &U_spts, 
    mdvector_gpu<double> &U_fpts, mdvector_gpu<double> &Uavg, 
    double gamma, double exps0, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nEles, unsigned int nVars,
    unsigned int nDims);

//! Update point coordinates or velocities for moving grids
void update_coords_wrapper(mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &g_nodes,  mdvector_gpu<double> &shape_spts,
    mdvector_gpu<double> &shape_fpts, mdvector_gpu<double> &coord_spts,
    mdvector_gpu<double> &coord_fpts, mdvector_gpu<double> &coord_faces,
    mdvector_gpu<int> &ele2node, mdvector_gpu<int>& fpt2gfpt, unsigned int nSpts,
    unsigned int nFpts, unsigned int nNodes, unsigned int nEles,
    unsigned int nDims);

void update_h_ref_wrapper(mdvector_gpu<double>& h_ref,
    mdvector_gpu<double>& coord_fpts, unsigned int nEles, unsigned int nFpts, unsigned int nPts1D,
    unsigned int nDims);

void calc_transforms_wrapper(mdvector_gpu<double> &nodes, mdvector_gpu<double> &jaco_spts,
    mdvector_gpu<double> &jaco_fpts, mdvector_gpu<double> &inv_jaco_spts,
    mdvector_gpu<double> &inv_jaco_fpts, mdvector_gpu<double> &jaco_det_spts,
    mdvector_gpu<double> &dshape_spts, mdvector_gpu<double> &dshape_fpts,
    int nSpts, int nFpts, int nNodes, int nEles, int nDims);

void calc_normals_wrapper(mdvector_gpu<double> &norm, mdvector_gpu<double> &dA,
    mdvector_gpu<double> &inv_jaco, mdvector_gpu<double> &tnorm,
    mdvector_gpu<int> &fpt2gfpt, mdvector_gpu<char> &fpt2slot, int nFpts,
    int nEles, int nDims);

//! For overset grid interpolation
void pack_donor_u_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars);

//! For overset grid interpolation
void pack_donor_grad_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars, unsigned int nDims);

#endif /* elements_kernels_h */
