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
void compute_F_wrapper(mdvector_gpu<double> &F_spts, mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &dU_spts, mdvector_gpu<double>& grid_vel_spts,  mdvector_gpu<double> &inv_jaco_spts,
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, unsigned int nEles, unsigned int nDims,
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu_in, double c_sth, double rt, bool fix_vis, bool viscous,
    bool grad_via_div, bool overset = false, int* iblank = NULL,
    bool motion = false);

void common_U_to_F_wrapper(mdvector_gpu<double> &Fcomm, mdvector_gpu<double> &Ucomm, mdvector_gpu<double> &norm_fpts, 
    mdvector_gpu<double> &dA_fpts, unsigned int nEles, unsigned int nFpts, unsigned int nVars, unsigned int nDims, unsigned int equation,
    unsigned int dim);

void compute_unit_advF_wrapper(mdvector_gpu<double>& F_spts, mdvector_gpu<double>& U_spts, mdvector_gpu<double>& inv_jaco_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nDims, unsigned int equation, unsigned int dim);

/* Residual Jacobian kernel wrappers (Implicit) */
void compute_inv_KPF_Jac_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppD_spts1D, 
    mdvector_gpu<double> oppDivE_spts1D, mdvector_gpu<double> dFdU_spts, mdvector_gpu<double> dFcdU, 
    unsigned int nSpts1D, unsigned int nVars, unsigned int nEles, unsigned int nDims);

void compute_visc_KPF_Jac_grad_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD_spts1D, 
    mdvector_gpu<double> &oppDivE_spts1D, mdvector_gpu<double> &oppDE_spts1D, mdvector_gpu<double> &dUcdU, 
    mdvector_gpu<double> &dFddU_spts, mdvector_gpu<double> &dFcddU, mdvector_gpu<double> &inv_jaco_spts, 
    mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts1D, unsigned int nSpts, unsigned int nVars, 
    unsigned int nEles, unsigned int nDims);

void compute_inv_Jac_spts_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &dFdU_spts, unsigned int nSpts, unsigned int nVars, unsigned int nEles,
    unsigned int nDims);

void compute_inv_Jac_fpts_wrapper(mdvector_gpu<double> LHS, mdvector_gpu<double> oppDiv_fpts, 
    mdvector_gpu<double> oppE, mdvector_gpu<double> dFcdU, unsigned int nSpts, unsigned int nFpts, 
    unsigned int nVars, unsigned int nEles);

void compute_visc_Jac_grad_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppD, 
    mdvector_gpu<double> &oppDiv_fpts, mdvector_gpu<double> &oppD_fpts, mdvector_gpu<double> &oppE, 
    mdvector_gpu<double> &dUcdU, mdvector_gpu<double> &dFddU_spts, mdvector_gpu<double> &dFcddU, 
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &jaco_det_spts, unsigned int nSpts, 
    unsigned int nFpts, unsigned int nVars, unsigned int nEles, unsigned int nDims);

void compute_visc_Jac_gradN_fpts_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &oppDiv_fpts, 
    mdvector_gpu<double> &oppD_fpts, mdvector_gpu<double> &oppE, mdvector_gpu<double> &dUcdU, 
    mdvector_gpu<double> &dFcddU, mdview_gpu<double> &inv_jacoN_spts, mdview_gpu<double> &jacoN_det_spts, 
    mdvector_gpu<unsigned int> &eleID, mdvector_gpu<int> &ele2eleN, mdvector_gpu<int> &face2faceN, 
    mdvector_gpu<int> &fpt2fptN, unsigned int startEle, unsigned int nSpts, unsigned int nFpts, 
    unsigned int nFptsPerFace, unsigned int nVars, unsigned int nFaces, unsigned int nEles, unsigned int nDims);

void scale_Jac_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &jaco_det_spts, 
    unsigned int nSpts, unsigned int nVars, unsigned int nEles);


/* Element dFdU kernel wrappers (Implicit) */
void compute_dFdU_wrapper(mdvector_gpu<double> &dFdU_spts, mdvector_gpu<double> &dFddU_spts,
    mdvector_gpu<double> &U_spts, mdvector_gpu<double> &dU_spts,
    mdvector_gpu<double> &inv_jaco_spts, unsigned int nSpts, unsigned int nEles, unsigned int nDims,
    unsigned int equation, mdvector_gpu<double> &AdvDiff_A, double AdvDiff_D, double gamma,
    double prandtl, double mu, bool viscous);

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

//! Copy updated node positions from geo to ele-based storage
void copy_coords_ele_wrapper(mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &g_nodes, mdvector_gpu<int> &ele2node,
    unsigned int nNodes, unsigned int nEles, unsigned int nDims);

//! Update point coordinates or velocities for moving grids
void update_coords_wrapper(mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &g_nodes,  mdvector_gpu<double> &shape_spts,
    mdvector_gpu<double> &shape_fpts, mdvector_gpu<double> &coord_spts,
    mdvector_gpu<double> &coord_fpts, mdvector_gpu<double> &coord_faces,
    mdvector_gpu<int> &ele2node, mdvector_gpu<int>& fpt2gfpt, unsigned int nSpts,
    unsigned int nFpts, unsigned int nNodes, unsigned int nEles,
    unsigned int nDims);

//! Update mesh nodes based on rigid-body rotation
void update_nodes_rigid_wrapper(mdvector_gpu<double> &nodes_init, mdvector_gpu<double> &nodes,
    mdvector_gpu<double> &Rmat, mdvector_gpu<double> &x_cg, unsigned int nNodes, unsigned int nDims);

//! Update transforms & normals based on rigid-body motion
void update_transforms_rigid_wrapper(mdvector_gpu<double>& jaco_spts_init, mdvector_gpu<double>& jaco_spts,
    mdvector_gpu<double> &inv_jaco_spts, mdvector_gpu<double> &norm_init,
    mdvector_gpu<double> &norm, mdvector_gpu<double> &Rmat, unsigned int nSpts,
    unsigned int nFpts, unsigned int nEles, unsigned int nDims, bool need_inv);

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

//! For moving-overset unblanking, estimate positions at end of time step
void estimate_point_positions_wrapper(mdvector_gpu<double> &coord_nodes,
  mdvector_gpu<double> &coord_spts, mdvector_gpu<double> &coord_fpts,
  mdvector_gpu<double> &vel_nodes, mdvector_gpu<double> &vel_spts,
  mdvector_gpu<double> &vel_fpts, double dt, unsigned int nNodes,
  unsigned int nSpts, unsigned int nFpts, unsigned int nEles, unsigned int nDims);

//! For overset grid interpolation
void pack_donor_u_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars);

//! For overset grid interpolation
void pack_donor_grad_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_donors, int* donorIDs, int nDonors,
    unsigned int nSpts, unsigned int nVars, unsigned int nDims);

#endif /* elements_kernels_h */
