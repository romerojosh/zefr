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
    void set_transforms(std::shared_ptr<Faces> faces);
    void set_normals(std::shared_ptr<Faces> faces);
    void setup_PMG();

    double calc_shape(unsigned int shape_order, unsigned int idx,
                      std::vector<double> &loc);

    double calc_d_shape(unsigned int shape_order, unsigned int idx,
                       std::vector<double> &loc, unsigned int dim);

    double calc_nodal_basis(unsigned int spt, std::vector<double> &loc);
    double calc_d_nodal_basis_spts(unsigned int spt, std::vector<double> &loc, 
                                   unsigned int dim);
    double calc_d_nodal_basis_fpts(unsigned int fpt, std::vector<double> &loc, 
                                   unsigned int dim);

  public:
    //Quads(GeoStruct *geo, const InputStruct *input, int order = -1);
    Quads(GeoStruct *geo, InputStruct *input, int order = -1);
    void transform_dU();
    void transform_flux();

};

#endif /* quads_hpp */
