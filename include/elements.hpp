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

class FRSolver;
class PMGrid;
class Elements
{
  friend class FRSolver;
  friend class PMGrid;
	friend class Filter;
  protected:
    InputStruct *input = NULL;
    GeoStruct *geo = NULL;

    /* Geometric Parameters */
    unsigned int order, shape_order;
    unsigned int nEles, nDims, nVars;
    unsigned int nSpts, nFpts, nSpts1D, nPpts, nQpts;
    unsigned int nFaces, nNodes;
    unsigned int nSubelements, nNodesPerSubelement;

    mdvector<double> loc_spts, loc_fpts, loc_ppts, loc_nodes, loc_qpts;
    mdvector<unsigned int> idx_spts, idx_fpts, idx_ppts, idx_nodes, idx_qpts;
    std::vector<double> loc_spts_1D, loc_nodes_1D, loc_qpts_1D, loc_DFR_1D;
    mdvector<double> tnorm; 
    mdvector<double> shape_spts, shape_fpts, shape_ppts, shape_qpts;
    mdvector<double> dshape_spts, dshape_fpts, dshape_ppts, dshape_qpts;
    mdvector<double> jaco_spts, jaco_det_spts, inv_jaco_spts;
    mdvector<double> jaco_ppts, jaco_qpts, jaco_det_qpts;
    mdvector<double> vol;
    mdvector<double> weights_spts;
    mdvector<double> h_ref;
    std::vector<double> weights_qpts;

    /* Element solution structures */
    mdvector<double> oppE, oppD, oppD_fpts, oppDiv_fpts;
    mdvector<double> oppE_ppts, oppE_qpts;
    mdvector<double> U_spts, U_fpts, U_ppts, U_qpts, Uavg;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> Fcomm, Ucomm;
    mdvector<double> dU_spts, dU_fpts, dU_qpts, divF_spts;

    /* Multigrid operators */
    mdvector<double> oppPro, oppRes;

    /* Element structures for implicit method */
    mdvector<double> LHS, LHSInv;  // Element local matrices for implicit system
    mdvector<int> LU_pivots, LU_info; 
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


#ifdef _GPU
    /* GPU data */
    mdvector_gpu<double> oppE_d, oppD_d, oppD_fpts_d, oppDiv_fpts_d;
    mdvector_gpu<double> oppE_ppts_d, oppE_qpts_d;
    mdvector_gpu<double> U_spts_d, U_fpts_d, U_ppts_d, U_qpts_d, Uavg_d;
    mdvector_gpu<double> F_spts_d, F_fpts_d;
    mdvector_gpu<double> Fcomm_d, Ucomm_d;
    mdvector_gpu<double> dU_spts_d, dU_fpts_d, divF_spts_d;
    mdvector_gpu<double> jaco_spts_d, inv_jaco_spts_d, jaco_det_spts_d;
    mdvector_gpu<double> vol_d;
    mdvector_gpu<double> weights_spts_d;
    mdvector_gpu<double> h_ref_d;

    /* Multigrid operators */
    mdvector_gpu<double> oppPro_d, oppRes_d;

    /* Element structures for implicit method */
    mdvector_gpu<double> LHS_d, LHSInv_d;
    mdvector_gpu<int> LU_pivots_d, LU_info_d; 
    mdvector_gpu<double*> LHS_ptrs_d, LHSInv_ptrs_d, RHS_ptrs_d, LHS_subptrs_d, LHS_tempSF_subptrs_d, oppE_ptrs_d, deltaU_ptrs_d; 
    mdvector_gpu<double> dFcdU_fpts_d, dFdU_spts_d;
    mdvector_gpu<double> deltaU_d;
    mdvector_gpu<double> RHS_d;
#endif

    void set_coords(std::shared_ptr<Faces> faces);
    void set_shape();
    void setup_FR();
    void setup_aux();

    virtual void set_locs() = 0;
    virtual void set_transforms(std::shared_ptr<Faces> faces) = 0;
    virtual void set_normals(std::shared_ptr<Faces> faces) = 0;
    virtual mdvector<double> calc_shape(unsigned int shape_order,
                            std::vector<double> &loc) = 0;
    virtual mdvector<double> calc_d_shape(unsigned int shape_order,
                            std::vector<double> &loc) = 0;

    virtual double calc_nodal_basis(unsigned int spt, std::vector<double> &loc) = 0;
    virtual double calc_d_nodal_basis_spts(unsigned int spt, std::vector<double> &loc, 
                                   unsigned int dim) = 0;
    virtual double calc_d_nodal_basis_fpts(unsigned int fpt, std::vector<double> &loc, 
                                   unsigned int dim) = 0;


  public:
    void setup(std::shared_ptr<Faces> faces);
    virtual void setup_PMG(int pro_order, int res_order) = 0;
    void extrapolate_U(unsigned int startEle, unsigned int endEle);
    void extrapolate_dU(unsigned int startEle, unsigned int endEle);
    void compute_dU(unsigned int startEle, unsigned int endEle);
    void compute_divF(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_divF_spts(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_divF_fpts(unsigned int stage, unsigned int startEle, unsigned int endEle);
    void compute_Fconv(unsigned int startEle, unsigned int endEle);
    void compute_Fvisc(unsigned int startEle, unsigned int endEle);
    virtual void transform_flux(unsigned int startEle, unsigned int endEle) = 0;
    virtual void transform_dU(unsigned int startEle, unsigned int endEle) = 0;

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
};

#endif /* elements_hpp */
