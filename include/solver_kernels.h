#ifndef solver_kernels_h
#define solver_kernels_h

#include "aux_kernels.h"

#ifdef _MPI
#define _mpi_comm MPI_Comm
#include "mpi.h"
#else
#define _mpi_comm int
#endif

/* Wrappers for custom kernels */
void add_source_wrapper(mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &coord_spts, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    double flow_time, unsigned int stage, bool overset = false, int* iblank = NULL);

void RK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars, unsigned int nDims, unsigned int equation, unsigned int stage, 
    unsigned int nStages, bool last_stage, bool overset = false, int* iblank = NULL);

void RK_update_source_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_ini, 
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &rk_coeff, unsigned int dt_type, 
    unsigned int nSpts, unsigned int nEles, unsigned int nVars, unsigned int nDims, 
    unsigned int equation, unsigned int stage, unsigned int nStages, bool last_stage, 
    bool overset = false, int* iblank = NULL);

void LSRK_update_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double>& rk_err,
    mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, double dt,
    double ai, double bi, double bhi, unsigned int nSpts, unsigned int nEles,
    unsigned int nVars, unsigned int stage, unsigned int nStages, bool adapt_dt,
    bool overset = false, int* iblank = NULL);

void LSRK_update_source_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_til, mdvector_gpu<double> &rk_err,
    mdvector_gpu<double> &divF, const mdvector_gpu<double> &source,
    mdvector_gpu<double> &jaco_det_spts, double dt, double ai, double bi,
    double bhi, unsigned int nSpts, unsigned int nEles, unsigned int nVars,
    unsigned int stage, unsigned int nStages, bool adapt_dt, bool overset = false,
    int* iblank = NULL);

void DIRK_update_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &divF, 
    mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_coeff, unsigned int dt_type, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    unsigned int nStages);

void RK_error_update_wrapper(mdvector_gpu<double> &rk_err, mdvector_gpu<double> &divF, 
    mdvector_gpu<double> &jaco_det_spts, mdvector_gpu<double> &dt, 
    mdvector_gpu<double> &rk_beta, mdvector_gpu<double> &rk_bhat, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, unsigned int equation, 
    unsigned int nStages);

double get_rk_error_wrapper(mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &rk_err, uint nSpts,
    uint nEles, uint nVars, double atol, double rtol, _mpi_comm comm_in,
    bool overset = false, int* iblank = NULL);

double set_adaptive_dt_wrapper(mdvector_gpu<double> &dt_in, double& dt_out, double expa, double expb,
    double minfac, double maxfac, double sfact, double max_dt, double max_err, double prev_err);

void compute_element_dt_wrapper(mdvector_gpu<double> &dt, mdvector_gpu<double> &waveSp_gfpts, 
    mdvector_gpu<double> &diffCo_gfpts, mdvector_gpu<double> &dA, mdvector_gpu<int> &fpt2gfpt, 
    mdvector_gpu<char> &fpt2gfpt_slot, mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &vol, 
    mdvector_gpu<double> &h_ref, unsigned int nFptsPerFace, double CFL, double beta, int order, 
    unsigned int CFL_type, unsigned int nFpts, unsigned int nEles, unsigned int nDims, 
    unsigned int startEle, bool overset = false, int* iblank = NULL);


void apply_dt_LHS_wrapper(mdvector_gpu<double> &LHS, mdvector_gpu<double> &dt,
    double coeff, unsigned int dt_type, unsigned int nSpts, unsigned int nEles, 
    unsigned int nVars);

void compute_RHS_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &U_iniNM, 
    mdvector_gpu<double> &U_ini, mdvector_gpu<double> &divF, mdvector_gpu<double> &jaco_det_spts, 
    mdvector_gpu<double> &dt, mdvector_gpu<double> &dtau, mdvector_gpu<double> &rk_coeff, 
    mdvector_gpu<double> &RHS, double dtau_ratio, bool implicit_steady, bool pseudo_time, 
    bool remove_deltaU, unsigned int dt_type, unsigned int dtau_type, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int stage);

void compute_U_wrapper(mdvector_gpu<double> &U_spts, mdvector_gpu<double> &deltaU, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

#ifdef _MPI
void pack_U_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &U, unsigned int nVars, int stream = -1);
void unpack_U_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &U, unsigned int nVars, int stream = -1, bool overset = false,
    int* iblank = NULL);
void pack_dU_wrapper(mdvector_gpu<double> &U_sbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream = -1);
void unpack_dU_wrapper(mdvector_gpu<double> &U_rbuffs, mdvector_gpu<unsigned int> &fpts, 
    mdview_gpu<double> &dU, unsigned int nVars, unsigned int nDims, int stream = -1, bool overset = false,
    int* iblank = NULL);
#endif

void compute_moments_wrapper(std::array<double,3> &tot_force, std::array<double,3> &tot_moment,
    mdview_gpu<double> &U_fpts, mdview_gpu<double> &dU_fpts, mdvector_gpu<double> &P_fpts,
    mdvector_gpu<double> &coord, mdvector_gpu<double>& x_cg, mdvector_gpu<double> &norm,
    mdvector_gpu<double> &dA, mdvector_gpu<double> &weights_fpts, mdvector_gpu<double> &force_face,
    mdvector_gpu<double> &moment_face, double gamma, double rt, double c_sth, double mu,
    bool viscous, bool fix_vis, int nVars, int nDims, int nFaces, mdvector_gpu<int> faceList,
    mdvector_gpu<int> face2fpts, int nFptsPerFace);

void accumulate_time_averages_wrapper(mdvector_gpu<double> &tavg_acc, mdvector_gpu<double> &tavg_prev,
    mdvector_gpu<double> &tavg_curr, mdvector_gpu<double> &U_spts, double prev_time, double curr_time,
    double gamma, int nSpts, int nVars, int nDims, int nEles);

void move_grid_wrapper(mdvector_gpu<double> &coords,
    mdvector_gpu<double>& coords_0, mdvector_gpu<double> &Vg, MotionVars &params,
    unsigned int nNodes, unsigned int nDims, int motion_type, double time,
    int gridID = 0);

void estimate_point_positions_nodes_wrapper(mdvector_gpu<double> &coord_nodes,
  mdvector_gpu<double> &vel_nodes,double dt, unsigned int nNodes, unsigned int nDims);

void estimate_point_positions_fpts_wrapper(mdvector_gpu<double> &coord_fpts,
  mdvector_gpu<double> &vel_fpts, double dt, unsigned int nFpts, unsigned int nDims);

void estimate_point_positions_spts_wrapper(mdvector_gpu<double> &coord_spts,
    mdvector_gpu<double> &vel_spts, double dt, unsigned int nSpts,
    unsigned int nEles, unsigned int nDims);

void unpack_fringe_u_wrapper(mdvector_gpu<double> &U_fringe, mdview_gpu<double> &U, mdview_gpu<double> &U_ldg,
    mdvector_gpu<unsigned int>& fringe_fpts, mdvector_gpu<unsigned int>& fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, int stream = -1);

void unpack_fringe_grad_wrapper(mdvector_gpu<double> &dU_fringe, mdview_gpu<double> &dU,
    mdvector_gpu<unsigned int>& fringe_fpts, mdvector_gpu<unsigned int>& fringe_side, unsigned int nFringe,
    unsigned int nFpts, unsigned int nVars, unsigned int nDims, int stream = -1);

void unpack_unblank_u_wrapper(mdvector_gpu<double> &U_unblank,
    mdvector_gpu<double> &U_spts, mdvector_gpu<int> &cellIDs, unsigned int nCells,
    unsigned int nSpts, unsigned int nVars, int stream = -1);

void pack_fringe_coords_wrapper(mdvector_gpu<unsigned int> &fringe_fpts, mdvector_gpu<double> &xyz,
    mdvector_gpu<double> &coord_fpts, int nPts, int nDims, int stream = -1);

void pack_cell_coords_wrapper(mdvector_gpu<int> &cellIDs, mdvector_gpu<double> &xyz,
    mdvector_gpu<double> &coord_spts, int nCells, int nSpts, int nDims, int stream = -1);

void get_nodal_basis_wrapper(int* cellIDs, double* rst, double* weights,
    double* xiGrid, int nFringe, int nSpts, int nSpts1D, int stream = -1);

#endif /* solver_kernels_h */
