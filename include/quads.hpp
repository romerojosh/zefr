#ifndef quads_hpp
#define quads_hpp

#include <memory>
#include <string>
#include <vector>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "solver.hpp"

class Quads: public Elements 
{
  private:
    void set_locs();
    void set_normals(std::shared_ptr<Faces> faces);

    mdvector<double> calc_shape(unsigned int shape_order,
                                const std::vector<double> &loc);

    mdvector<double> calc_d_shape(unsigned int shape_order,
                                  const std::vector<double> &loc);

    double calc_nodal_basis(unsigned int spt,
                            const std::vector<double> &loc);
    double calc_nodal_basis(unsigned int spt, double *loc);
    double calc_d_nodal_basis_spts(unsigned int spt,
                                   const std::vector<double> &loc,
                                   unsigned int dim);
    double calc_d_nodal_basis_fpts(unsigned int fpt,
                                   const std::vector<double> &loc,
                                   unsigned int dim);

  public:
    Quads(GeoStruct *geo, InputStruct *input, int order = -1);
    void transform_dU(unsigned int startEle, unsigned int endEle);
    void transform_flux(unsigned int startEle, unsigned int endEle);

    void setup_PMG(int pro_order, int res_order);

    /* Routines for implicit method */
    void transform_dFdU();
    double calc_d_nodal_basis_fr(unsigned int spt, const std::vector<double>& loc, unsigned int dim);
};

#endif /* quads_hpp */
