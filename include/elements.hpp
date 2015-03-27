#ifndef elements_hpp
#define elements_hpp

#include <string>

#include "faces.hpp"
#include "input.hpp"
#include "mdvector.hpp"

class Elements
{
  protected:
    const InputStruct *input = NULL;
    Faces *faces = NULL;

    /* Geometric Parameters */
    unsigned int order, shape_order;
    unsigned int nEles, nDims, nVars;
    unsigned int nSpts, nFpts, nSpts1D;
    unsigned int nFaces, nNodes;
    mdvector<int> nd2gnd;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> coord_nodes, coord_spts, coord_fpts;

    mdvector<double> loc_spts, loc_fpts, loc_nodes;
    mdvector<double> idx_spts, idx_fpts, idx_nodes;
    std::vector<double> loc_spts_1D, loc_nodes_1D;
    mdvector<double> tnorm; //, norm, dA;
    mdvector<double> shape_spts, shape_fpts;
    mdvector<double> dshape_spts, dshape_fpts;
    mdvector<double> jaco_spts, jaco_det_spts;
    //mdvector<double> jaco_fpts;

    /* Element solution structures */
    mdvector<double> oppE, oppD, oppD_fpts;
    mdvector<double> U_spts, U_fpts;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> dU_spts, dF_spts, divF_spts;


    virtual void set_locs() = 0;
    virtual void set_shape() = 0;
    virtual void set_transforms() = 0;
    virtual void set_normals() = 0;

    void initialize_U();
    virtual void setup_FR() = 0;

  public:
    void associate_faces(Faces *faces);
    void setup();
    void FR_cycle();
    const mdvector<double>& get_divF() const; 
    const mdvector<double>& get_U() const;
    

};

#endif /* elements_hpp */
