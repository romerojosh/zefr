#ifndef elements_hpp
#define elements_hpp

#include <string>

#include "input.hpp"
#include "mdvector.hpp"

class Elements
{
  private:
    const InputStruct *input = NULL;
    std::string ele_type;
    unsigned int order, shape_order;
    unsigned int nEles, nDims, nVars;
    unsigned int nSpts, nFpts, nFptsPerFace, nSpts1D;
    unsigned int nFaces, nNodes;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> oppE, oppD, oppD_fpts;
    mdvector<double> U_spts, U_fpts;
    mdvector<double> F_spts, F_fpts;
    mdvector<double> dU_spts, dF_spts, divF_spts;
    mdvector<double> loc_spts, loc_fpts, loc_nodes;

    void set_locs();

    void initialize_U();
    void setup_FR();

    void extrapolate_U();
    /* Note: Going to create ele2fpt and slot structure like FR2D. gfpt=-1 means no comm. */
    void U_to_faces();

    /* Viscous Stuff */
    void U_from_faces();
    void compute_dU();
    void dU_to_faces();

    /* Note: These will be additive, Fvisc will use F_spts += */
    void compute_Fconv_spts();
    void compute_Fvisc_spts();

    /* Note: Do I have to transform dU? */
    void transform_F();

    void F_from_faces();

    void compute_dF();
    void compute_divF();

  public:
    Elements(unsigned int nEles, std::string ele_type, unsigned int shape_order, 
             const InputStruct *input, unsigned int P = -1);
    void setup();
    void FR_cycle();
    const mdvector<double>& get_divF() const; 
    const mdvector<double>& get_U() const;
    

};

#endif /* elements_hpp */
