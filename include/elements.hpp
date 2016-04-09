#ifndef elements_hpp
#define elements_hpp

#include <memory>
#include <string>
#include <vector>

#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"
#include "spmatrix.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#include "spmatrix_gpu.h"
#endif

class FRSolver;
class PMGrid;
class Elements
{
  friend class FRSolver;
  friend class PMGrid;
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
    mdvector<double> weights_spts;
    std::vector<double> weights_qpts;

    /* Element solution structures */
    mdvector<double> oppE, oppD, oppD_fpts;
    mdvector<double> oppE_ppts, oppE_qpts;
    mdvector<double> U_spts, U_fpts, U_ppts, U_qpts, Uavg;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> Fconv_spts, Fvisc_spts;
    mdvector<double> Fcomm, Ucomm;
    mdvector<double> dU_spts, dU_fpts, dU_qpts, dF_spts, divF_spts;

    /* Multigrid operators */
    mdvector<double> oppPro, oppRes;

    /* Element structures for implicit method */
    spmatrix<double> GLHS; // Sparse matrix for global implicit system
    mdvector<double> LHS;  // Element local matrices for implicit system
    mdvector<double> dFdUconv_spts, dFdUvisc_spts, dFddUvisc_spts;
    mdvector<double> dFndUconv_fpts, dFndUvisc_fpts, dFnddUvisc_fpts, beta_Ucomm_fpts, taun_fpts;
    mdvector<double> deltaU;
    mdvector<double> RHS;

#ifdef _GPU
    /* GPU data */
    mdvector_gpu<double> oppE_d, oppD_d, oppD_fpts_d;
    mdvector_gpu<double> oppE_ppts_d, oppE_qpts_d;
    mdvector_gpu<double> U_spts_d, U_fpts_d, U_ppts_d, U_qpts_d, Uavg_d;
    mdvector_gpu<double> F_spts_d, F_fpts_d;
    mdvector_gpu<double> Fconv_spts_d, Fvisc_spts_d;
    mdvector_gpu<double> Fcomm_d, Ucomm_d;
    mdvector_gpu<double> dU_spts_d, dU_fpts_d, dF_spts_d, divF_spts_d;
    mdvector_gpu<double> jaco_spts_d, inv_jaco_spts_d, jaco_det_spts_d;
    mdvector_gpu<double> weights_spts_d;

    /* Multigrid operators */
    mdvector_gpu<double> oppPro_d, oppRes_d;

    /* Element structures for implicit method */
    spmatrix_gpu<double> GLHS_d;
    mdvector_gpu<double> deltaU_d;
    mdvector_gpu<double> RHS_d;
#endif

    void set_coords(std::shared_ptr<Faces> faces);
    void set_shape();
    void setup_FR();
    void setup_aux();

    virtual void setup_PMG() = 0;
    virtual void set_locs() = 0;
    virtual void set_transforms(std::shared_ptr<Faces> faces) = 0;
    virtual void set_normals(std::shared_ptr<Faces> faces) = 0;
    virtual double calc_shape(unsigned int shape_order, unsigned int idx,
                            std::vector<double> &loc) = 0;
    virtual double calc_d_shape(unsigned int shape_order, unsigned int idx,
                            std::vector<double> &loc, unsigned int dim) = 0;

    virtual double calc_nodal_basis(unsigned int spt, std::vector<double> &loc) = 0;
    virtual double calc_d_nodal_basis_spts(unsigned int spt, std::vector<double> &loc, 
                                   unsigned int dim) = 0;
    virtual double calc_d_nodal_basis_fpts(unsigned int fpt, std::vector<double> &loc, 
                                   unsigned int dim) = 0;


  public:
    void setup(std::shared_ptr<Faces> faces);
    void extrapolate_U();
    void extrapolate_dU();
    void compute_dU();
    void compute_dF();
    void compute_divF(unsigned int stage);
    void compute_Fconv();
    void compute_Fvisc();
    virtual void transform_flux() = 0;
    virtual void transform_dU() = 0;

    /* Routines for implicit method */
    void compute_globalLHS(mdvector<double> &dt);
    void compute_localLHS(mdvector<double> &dt);
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
