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
    void set_oppRestart(unsigned int order_restart, bool use_shape = false);
    void set_vandermonde_mats();

    void calc_shape(mdvector<double> &shape_val, const double* loc);
    void calc_d_shape(mdvector<double> &dshap_val, const double* loc);

    double calc_nodal_basis(unsigned int spt,
                            const std::vector<double> &loc);
    double calc_nodal_basis(unsigned int spt, double *loc);
    void calc_nodal_basis(double *loc, double* basis);
    double calc_d_nodal_basis_spts(unsigned int spt,
                                   const std::vector<double> &loc,
                                   unsigned int dim);
    double calc_d_nodal_basis_fpts(unsigned int fpt,
                                   const std::vector<double> &loc,
                                   unsigned int dim);

    void modify_sensor();

    mdvector<double> get_face_nodes(unsigned int face, unsigned int P);
    mdvector<double> get_face_weights(unsigned int face, unsigned int P);

    void project_face_point(int face, const double* loc, double* ploc);

    double calc_nodal_face_basis(unsigned int face, unsigned int pt, const double *loc);

    double calc_orthonormal_basis(unsigned int mode, const double *loc);

    double rst_max_lim(int dim, double *rst);
    double rst_min_lim(int dim, double *rst);

  public:
    Quads(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order = -1);

    void setup_PMG(int pro_order, int res_order);
    void setup_ppt_connectivity();

    double calc_d_nodal_basis_fr(unsigned int spt, const std::vector<double>& loc, unsigned int dim);

};

#endif /* quads_hpp */
