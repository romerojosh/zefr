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

#ifndef elements_hpp
#define elements_hpp

#include <memory>
#include <string>
#include <vector>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#ifdef _BUILD_LIB
#include "zefr.hpp"
#endif

class FRSolver;
class PMGrid;
class Elements
{
  friend class FRSolver;
  friend class PMGrid;
  friend class Filter;
#ifdef _BUILD_LIB
  friend class Zefr;
#endif
  protected:
    InputStruct *input = NULL;
    GeoStruct *geo = NULL;
    ELE_TYPE etype;

    /* Geometric Parameters */
    unsigned int order, shape_order;
    unsigned int nEles, nDims, nVars;
    unsigned int nSpts, nFpts, nSpts1D, nPpts, nQpts, nFptsPerFace;
    unsigned int nFaces, nNodes;
    unsigned int nSubelements, nNodesPerSubelement;
    mdvector<double> coord_spts, coord_fpts, coord_ppts, coord_qpts;

    mdvector<double> loc_spts, loc_fpts, loc_ppts, loc_nodes, loc_qpts;
    mdvector<unsigned int> idx_spts, idx_fpts, idx_ppts, idx_nodes, idx_qpts;
    std::vector<double> loc_spts_1D, loc_nodes_1D, loc_qpts_1D, loc_DFR_1D;
    mdvector<double> tnorm; 
    mdvector<double> shape_spts, shape_fpts, shape_ppts, shape_qpts;
    mdvector<double> dshape_spts, dshape_fpts, dshape_ppts, dshape_qpts;
    mdvector<double> jaco_spts, jaco_det_spts, inv_jaco_spts;
    mdvector<double> jaco_ppts, jaco_qpts, jaco_det_qpts;
    mdvector<double> jaco_fpts, jaco_det_fpts, inv_jaco_fpts;
    mdvector<double> nodes;
    mdvector<double> vol;
    mdvector<double> weights_spts, weights_fpts, weights_qpts;
    mdvector<double> h_ref;
    mdvector<double> vandDB, inv_vandDB, vandRT, inv_vandRT;

    /* Moving-Grid related structures */
    mdvector<double> grid_vel_nodes, grid_vel_spts, grid_vel_fpts, grid_vel_ppts;
    mdvector<double> dF_spts, dUr_spts;
    mdvector<double> oppD0, oppE_Fn;
    mdvector<double> dFn_fpts, tempF_fpts;

    /* Element solution structures */
    mdvector<double> oppE, oppD, oppD_fpts, oppDiv_fpts;
    mdvector<double> oppE_ppts, oppE_qpts;
    mdvector<double> U_spts, U_fpts, U_ppts, U_qpts, Uavg, U_ini, U_til;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> Fcomm, Ucomm;
    mdvector<double> dU_spts, dU_fpts, dU_qpts, divF_spts;

    /* Multigrid operators */
    mdvector<double> oppPro, oppRes;

    mdvector<double> dt, rk_err;

    /* Element structures for implicit method */
    mdvector<double> LHS, LHSInv;  // Element local matrices for implicit system
    mdvector<int> LU_pivots, LU_info; 
#ifndef _NO_TNT
    std::vector<std::vector<JAMA::LU<double>>> LUptrs;
#endif
    mdvector<double*> LHS_ptrs, RHS_ptrs, LHSInv_ptrs, LHS_subptrs, LHS_tempSF_subptrs, oppE_ptrs, deltaU_ptrs; 
    mdvector<double> dFdU_spts, dFddU_spts;
    mdvector<double> dFcdU_fpts, dUcdU_fpts, dFcddU_fpts;
    mdvector<double> deltaU;
    mdvector<double> RHS;

    std::vector<mdvector<double>> LHSs, LHSInvs;

    mdvector<double> Cvisc0, CviscN, CdFddU0;
    mdvector<double> CtempSS, CtempFS, CtempFS2;
    mdvector<double> CtempSF;
    mdvector<double> CtempFSN, CtempFSN2;

    _mpi_comm myComm;

    mdvector<double> U_donors, dU_donors;

    /* Output data structures */
    mdvector<unsigned int> ppt_connect;

#ifdef _GPU
    /* GPU data */
    mdvector_gpu<double> oppE_d, oppD_d, oppD_fpts_d, oppDiv_fpts_d;
    mdvector_gpu<double> oppE_ppts_d, oppE_qpts_d;
    mdvector_gpu<double> U_spts_d, U_fpts_d, U_ppts_d, U_qpts_d, Uavg_d;
    mdvector_gpu<double> F_spts_d, F_fpts_d;
    mdvector_gpu<double> Fcomm_d, Ucomm_d;
    mdvector_gpu<double> dU_spts_d, dU_fpts_d, divF_spts_d;
    mdvector_gpu<double> jaco_spts_d, inv_jaco_spts_d, jaco_det_spts_d;
    mdvector_gpu<double> jaco_fpts_d, inv_jaco_fpts_d;
    mdvector_gpu<double> vol_d;
    mdvector_gpu<double> weights_spts_d, weights_fpts_d;
    mdvector_gpu<double> h_ref_d;

    /* Motion Related */
    mdvector_gpu<double> grid_vel_nodes_d, grid_vel_spts_d, grid_vel_fpts_d, grid_vel_ppts_d;
    mdvector_gpu<double> nodes_d, shape_spts_d, shape_fpts_d, dshape_spts_d, dshape_fpts_d;
    mdvector_gpu<double> tnorm_d;
    mdvector_gpu<double> dF_spts_d, dUr_spts_d;
    mdvector_gpu<double> oppD0_d, oppE_Fn_d;
    mdvector_gpu<double> dFn_fpts_d, tempF_fpts_d;

    /* Multigrid operators */
    mdvector_gpu<double> oppPro_d, oppRes_d;

    /* Element structures for implicit method */
    mdvector_gpu<double> LHS_d, LHSInv_d;
    mdvector_gpu<int> LU_pivots_d, LU_info_d; 
    mdvector_gpu<double*> LHS_ptrs_d, LHSInv_ptrs_d, RHS_ptrs_d, LHS_subptrs_d, LHS_tempSF_subptrs_d, oppE_ptrs_d, deltaU_ptrs_d; 
    mdvector_gpu<double> dFcdU_fpts_d, dFdU_spts_d;
    mdvector_gpu<double> deltaU_d;
    mdvector_gpu<double> RHS_d;

    mdvector_gpu<double> U_donors_d, dU_donors_d;
#endif

    void set_coords(std::shared_ptr<Faces> faces);
    void set_shape();
    void setup_FR();
    void setup_aux();

    virtual void set_locs() = 0;
    virtual void set_normals(std::shared_ptr<Faces> faces) = 0;
    virtual mdvector<double> calc_shape(unsigned int shape_order,
                             const std::vector<double> &loc) = 0;
    virtual mdvector<double> calc_d_shape(unsigned int shape_order,
                             const std::vector<double> &loc) = 0;

    virtual double calc_nodal_basis(unsigned int spt,
                   const std::vector<double> &loc) = 0;
    virtual double calc_nodal_basis(unsigned int spt, double *loc) = 0;
    virtual double calc_d_nodal_basis_spts(unsigned int spt,
                   const std::vector<double> &loc, unsigned int dim) = 0;
    virtual double calc_d_nodal_basis_fpts(unsigned int fpt,
                   const std::vector<double> &loc, unsigned int dim) = 0;

    virtual double calc_d_nodal_basis_fr(unsigned int spt,
                   const std::vector<double>& loc, unsigned int dim) = 0;

  public:
    void setup(std::shared_ptr<Faces> faces, _mpi_comm comm_in);
    virtual void setup_PMG(int pro_order, int res_order) = 0;
    virtual void setup_ppt_connectivity() = 0;
    void initialize_U();
    void extrapolate_U(unsigned int startEle, unsigned int endEle);
    void extrapolate_dU(unsigned int startEle, unsigned int endEle);
    void compute_dU(unsigned int startEle, unsigned int endEle);
    void compute_dU_spts(unsigned int startEle, unsigned int endEle);
    void compute_dU_fpts(unsigned int startEle, unsigned int endEle);
    void compute_dU_spts_via_divF(unsigned int startEle, unsigned int endEle, unsigned int dim);
    void compute_dU_fpts_via_divF(unsigned int startEle, unsigned int endEle, unsigned int dim);
    void compute_divF(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_divF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_divF_fpts(unsigned int stage, unsigned int startEle, unsigned int endEle);

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void compute_F(unsigned int startEle, unsigned int endEle);

    void compute_F(unsigned int startEle, unsigned int endEle);

    template<unsigned int nVars, unsigned int nDims>
    void compute_unit_advF(unsigned int startEle, unsigned int endEle, unsigned int dim);

    void compute_unit_advF(unsigned int startEle, unsigned int endEle, unsigned int dim);

    //! Calculate geometric transforms
    void calc_transforms(std::shared_ptr<Faces> faces);

    //! Calculate inverse of geo transforms for a set of points
    void set_inverse_transforms(const mdvector<double> &jaco,
         mdvector<double> &inv_jaco, mdvector<double> &jaco_det,
         unsigned int nPts, unsigned int nDims);

    /* Routines for implicit method */
#ifdef _CPU
    void compute_localLHS(mdvector<double> &dt, unsigned int startEle, unsigned int endEle, unsigned int color = 1);
#endif
#ifdef _GPU
    void compute_localLHS(mdvector_gpu<double> &dt_d, unsigned int startEle, unsigned int endEle, unsigned int color = 1);
#endif
    void compute_dFdUconv();
    void compute_dFdUvisc();
    void compute_dFddUvisc();
    virtual void transform_dFdU() = 0;

    /* Polynomial squeeze methods */
    void compute_Uavg();
    void poly_squeeze();
    void poly_squeeze_ppts();

    /* Motion-related functions */

    /* Functions required for overset interfacing */
    int get_nSpts(void) { return (int)nSpts; }
    int get_nFpts(void) { return (int)nFpts; }
    bool getRefLoc(int ele, double* xyz, double* rst);
    void get_interp_weights(double* rst, double* weights, int& nweights, int buffSize);

    std::vector<double> getBoundingBox(int ele);

#ifdef _GPU
    void donor_u_from_device(int* donorIDs, int nDonors);
    void donor_grad_from_device(int* donorIDs, int nDonors);
#endif

    void move(std::shared_ptr<Faces> faces);
    void update_point_coords(std::shared_ptr<Faces> faces);
    void update_grid_velocities(std::shared_ptr<Faces> faces);
    void compute_gradF_spts(unsigned int startEle, unsigned int endEle);
    void transform_gradF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_dU0(unsigned int startEle, unsigned int endEle);

    //! Extrapolated discontinuous normal flux to get delta-Fn at fpts
    void extrapolate_Fn(unsigned int startEle, unsigned int endEle, std::shared_ptr<Faces> faces);

    //! 'Standard' (non-DFR) correction procedure
    void correct_divF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void get_grid_velocity_ppts(void);
    void update_plot_point_coords();
};

#endif /* elements_hpp */
