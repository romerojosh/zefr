#ifndef elements_hpp
#define elements_hpp

#include <memory>
#include <string>
#include <vector>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"

#ifdef _CPU
#include <Eigen/Dense>
#endif

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
    unsigned int elesObjID;

    void init(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order);

    /* Geometric Parameters */
    unsigned int order;
    unsigned int startEle, endEle;
    unsigned int nEles, nElesPad, nDims, nVars;
    unsigned int nSpts, nFpts, nSpts1D, nPpts, nQpts, nFptsPerFace;
    unsigned int nFaces, nNodes;
    unsigned int nSubelements, nNodesPerSubelement;
    mdvector<double> coord_spts, coord_fpts, coord_ppts, coord_qpts;
    std::vector<unsigned int> nFpts_face;

    mdvector<double> loc_spts, loc_fpts, loc_ppts, loc_nodes, loc_qpts;
    mdvector<unsigned int> idx_spts, idx_fpts, idx_ppts, idx_nodes, idx_qpts;
    std::vector<double> loc_spts_1D, loc_nodes_1D, loc_qpts_1D, loc_DFR_1D;
    mdvector<double> tnorm, tdA;
    mdvector<double> shape_spts, shape_fpts, shape_ppts, shape_qpts;
    mdvector<double> dshape_spts, dshape_fpts, dshape_ppts, dshape_qpts;
    mdvector<double> jaco_spts, jaco_det_spts, inv_jaco_spts;
    mdvector<double> jaco_ppts, jaco_qpts, jaco_det_qpts;
    mdvector<double> jaco_fpts, jaco_det_fpts, inv_jaco_fpts;
    mdvector<double> nodes;
    mdvector<double> vol;
    mdvector<double> weights_spts, weights_fpts, weights_qpts;
    mdvector<double> h_ref;
    mdvector<double> vand, inv_vand, vandRT, inv_vandRT;

    /* Moving-Grid related structures */
    mdvector<double> grid_vel_nodes, grid_vel_spts, grid_vel_fpts, grid_vel_ppts;
    mdvector<double> dF_spts, dUr_spts;
    mdvector<double> dFn_fpts, tempF_fpts;

    mdvector<double> inv_jaco_spts_init;

    /* Performance hacks for Hexas::calc_shape and calc_d_shape, and others */
    unsigned int nNdSide;
    std::vector<double> xlist;
    std::vector<double> lag_i, lag_j, lag_k;
    std::vector<double> dlag_i, dlag_j, dlag_k;
    std::vector<int> ijk2gmsh;
    mdvector<double> tmp_coords, tmp_shape, tmp_dshape;
    mdvector<double> tmp_S;

    /* Element solution structures */
    mdvector<double> oppE, oppD, oppD_fpts, oppDiv, oppDiv_fpts;
    unsigned long oppE_id = 0, oppD_id = 0, oppD_fpts_id = 0, oppDiv_id = 0, oppDiv_fpts_id = 0; // IDs for GiMMiK
    mdvector<double> oppE_ppts, oppE_qpts, oppRestart;
    mdvector<double> U_spts, U_fpts, U_ppts, U_qpts, Uavg, U_ini, U_til;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> Fcomm, Ucomm;
    mdvector<double> dU_spts, dU_fpts, dU_qpts, divF_spts;
    mdvector<double> oppCorr; // For FR tris/tets

    /* Multigrid operators */
    mdvector<double> oppPro, oppRes;

    /* Filter operators */
    mdvector<double> oppS_1D, oppS, oppF;

    mdvector<double> dt, rk_err;

    /* Element structures for implicit Jacobian computation */
    mdvector<double> oppD_spts1D, oppDE_spts1D, oppDivE_spts1D;
    mdvector<double> dFdU_spts, dFddU_spts;
    mdvector<double> dFcdU, dUcdU, dFcddU;
    mdvector<double> ddUdUc;

    /* Temporary structures for implicit method (CPU) */
    mdvector<double> Cvisc0, CviscN, CdFddU0, CdFcddU0;
    mdvector<double> CtempSF, CtempD, CtempFS, CtempFSN;

    /* Views for viscous implicit Jacobian */
    mdview<double> inv_jacoN_spts, jacoN_det_spts;

    /* Element structures for implicit method */
    mdvector<double> dtau, LHS, LHSinv, RHS, deltaU, U_iniNM;
#ifdef _CPU
    unsigned int svd_rank;
    std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>> LU_ptrs;
    std::vector<Eigen::JacobiSVD<Eigen::MatrixXd>> SVD_ptrs;
    mdvector<double> LHSinvD, LHSinvS, LHSU, LHSV;
#endif
#ifdef _GPU
    mdvector<double*> LHS_ptrs, LHSinv_ptrs, RHS_ptrs, deltaU_ptrs;
    mdvector<int> LHS_info;
#endif

    _mpi_comm myComm;

    mdvector<double> U_donors, dU_donors;
    unsigned int nDonors = 0;
    int *donorIDs_d = NULL;
    std::vector<int> donorIDs;

    /* Output data structures */
    mdvector<unsigned int> ppt_connect;

    /* Averaging and Statistics */
    mdvector<double> tavg_acc, tavg_prev, tavg_curr;

#ifdef _GPU
    /* GPU data */
    mdvector_gpu<double> oppE_d, oppD_d, oppD_fpts_d, oppDiv_d, oppDiv_fpts_d;
    mdvector_gpu<double> oppE_ppts_d, oppE_qpts_d;
    mdvector_gpu<double> U_spts_d, U_fpts_d, U_ppts_d, U_qpts_d, Uavg_d, U_ini_d, U_til_d;
    mdvector_gpu<double> F_spts_d, F_fpts_d;
    mdvector_gpu<double> Fcomm_d, Ucomm_d;
    mdvector_gpu<double> dU_spts_d, dU_fpts_d, divF_spts_d;
    mdvector_gpu<double> jaco_spts_d, inv_jaco_spts_d, jaco_det_spts_d;
    mdvector_gpu<double> jaco_fpts_d, inv_jaco_fpts_d;
    mdvector_gpu<double> vol_d;
    mdvector_gpu<double> weights_spts_d, weights_fpts_d;
    mdvector_gpu<double> h_ref_d;
    mdvector_gpu<double> coord_spts_d, coord_fpts_d;

    /* Motion Related */
    mdvector_gpu<double> grid_vel_nodes_d, grid_vel_spts_d, grid_vel_fpts_d, grid_vel_ppts_d;
    mdvector_gpu<double> nodes_d, shape_spts_d, shape_fpts_d, dshape_spts_d, dshape_fpts_d;
    mdvector_gpu<double> tnorm_d;
    mdvector_gpu<double> dF_spts_d, dUr_spts_d;
    mdvector_gpu<double> dFn_fpts_d, tempF_fpts_d;

    mdvector_gpu<double> jaco_spts_init_d, inv_jaco_spts_init_d;

    /* Multigrid operators */
    mdvector_gpu<double> oppPro_d, oppRes_d;

    /* Filter operators */
    mdvector_gpu<double> oppS_d, oppF_d;

    mdvector_gpu<double> dt_d, rk_err_d;

    /* Element structures for implicit Jacobian computation */
    mdvector_gpu<double> oppD_spts1D_d, oppDE_spts1D_d, oppDivE_spts1D_d;
    mdvector_gpu<double> dFdU_spts_d, dFddU_spts_d;
    mdvector_gpu<double> dFcdU_d, dUcdU_d, dFcddU_d;
    mdvector_gpu<double> ddUdUc_d;

    /* Views for viscous implicit Jacobian */
    mdview_gpu<double> inv_jacoN_spts_d, jacoN_det_spts_d;

    /* Element structures for implicit method */
    mdvector_gpu<double> dtau_d, LHS_d, LHSinv_d, RHS_d, deltaU_d, U_iniNM_d;
    mdvector_gpu<double*> LHS_ptrs_d, LHSinv_ptrs_d, RHS_ptrs_d, deltaU_ptrs_d;
    mdvector_gpu<int> LHS_info_d;

    /* Overset Related */
    mdvector_gpu<double> U_donors_d, dU_donors_d;
    mdvector_gpu<double> U_unblank_d;
    mdvector_gpu<int> unblankIDs_d;
    mdvector_gpu<double> cellCoords_d;
    mdvector_gpu<double> loc_spts_1D_d;

    /* Averaging and Statisctics */
    mdvector_gpu<double> tavg_acc_d, tavg_prev_d, tavg_curr_d;
#endif

    void set_coords(std::shared_ptr<Faces> faces);
    void set_shape();
    void setup_FR();
    void setup_aux();

    virtual void set_locs() = 0;
    virtual void set_normals(std::shared_ptr<Faces> faces) = 0;

    virtual void set_oppRestart(unsigned int order_restart, bool use_shape = false) = 0;
    virtual void set_vandermonde_mats() = 0;

    virtual void calc_shape(mdvector<double> &shape_val, const double* loc) = 0;
    virtual void calc_d_shape(mdvector<double> &dshape_val, const double* loc) = 0;

    virtual double calc_nodal_basis(unsigned int spt,
                   const std::vector<double> &loc) = 0;
    virtual double calc_nodal_basis(unsigned int spt, double *loc) = 0;
    virtual void calc_nodal_basis(double *loc, double* basis) = 0;
    virtual double calc_d_nodal_basis_spts(unsigned int spt,
                   const std::vector<double> &loc, unsigned int dim) = 0;
    virtual double calc_d_nodal_basis_fpts(unsigned int fpt,
                   const std::vector<double> &loc, unsigned int dim) = 0;

    virtual double calc_d_nodal_basis_fr(unsigned int spt,
                   const std::vector<double>& loc, unsigned int dim) = 0;

    virtual void modify_sensor() = 0;

    virtual mdvector<double> get_face_nodes(unsigned int face, unsigned int P) = 0;
    virtual mdvector<double> get_face_weights(unsigned int face, unsigned int P) = 0;

    virtual void project_face_point(int face, const double* loc, double* ploc) = 0;

    virtual double calc_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc) = 0;
    virtual double calc_orthonormal_basis(unsigned int mode, const double *loc) = 0;

    virtual double rst_max_lim(int dim, double* rst) = 0;
    virtual double rst_min_lim(int dim, double* rst) = 0;

  public:
    void setup(std::shared_ptr<Faces> faces, _mpi_comm comm_in);
    void setup_filter();
    virtual void setup_PMG(int pro_order, int res_order) = 0;
    virtual void setup_ppt_connectivity() = 0;
    void initialize_U();

    void extrapolate_U();
    void extrapolate_dU();
    void compute_dU();
    void compute_dU_spts();
    void compute_dU_fpts();
    void compute_dU_spts_via_divF(unsigned int dim);
    void compute_dU_fpts_via_divF(unsigned int dim);
    void compute_divF(unsigned int stage);
    void compute_divF_spts(unsigned int stage);
    void compute_divF_fpts(unsigned int stage);
    void add_source(unsigned int stage, double flow_time);

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void compute_F();

    void compute_F();

    //! Calculate geometric transforms
    void calc_transforms(std::shared_ptr<Faces> faces);

    //! Calculate inverse of geo transforms for a set of points
    void set_inverse_transforms(const mdvector<double> &jaco,
         mdvector<double> &inv_jaco, mdvector<double> &jaco_det,
         unsigned int nPts, unsigned int nDims);

    /* Routines for implicit method */
    void setup_ddUdUc();
    void compute_local_dRdU();

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void compute_dFdU();

    void compute_dFdU();
    void compute_KPF_dFcdU_gradN();

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
    void getBoundingBox(int ele, double bbox[6]);

#ifdef _GPU
    void unblank_u_to_device(int *cellIDs, int nCells, double* data);

    void get_cell_coords(int* cellIDs, int nCells, int* nPtsCell, double* xyz);

    void get_interp_weights_gpu(int* cellIDs, int nFringe, double* rst, double* weights);
#endif

    void move(std::shared_ptr<Faces> faces);
    void update_point_coords(std::shared_ptr<Faces> faces);
    void update_grid_velocities(std::shared_ptr<Faces> faces);

    void get_grid_velocity_ppts(void);
    void update_plot_point_coords();
};

#endif /* elements_hpp */
