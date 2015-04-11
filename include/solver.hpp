#ifndef solver_hpp
#define solver_hpp

#include <memory>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"

class Elements;
class FRSolver
{
  private:
    const InputStruct *input = NULL;
    GeoStruct geo;
    unsigned int order;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;

    unsigned int nStages;
    std::vector<double> rk_alpha, rk_beta;
    mdvector<double> divF;
    mdvector<double> U_ini;

    void initialize_U();
    void setup_update();

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
    void compute_divF(unsigned int stage);


  public:
    FRSolver(const InputStruct *input, unsigned int order = -1);
    void setup();
    void compute_residual(unsigned int stage);
    void update();

};

#endif /* solver_hpp */
