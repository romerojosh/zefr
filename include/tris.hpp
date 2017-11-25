/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef tris_hpp
#define tris_hpp

#include <memory>
#include <string>
#include <vector>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "solver.hpp"

class Tris: public Elements 
{
  private:
    void set_locs();
    void set_normals(std::shared_ptr<Faces> faces);
    void set_oppRestart(unsigned int order_restart, bool use_shape = false);
    void set_vandermonde_mats();

    void calc_shape(mdvector<double> &shape_val, const double* loc);
    void calc_d_shape(mdvector<double> &dshape_val, const double* loc);

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

    mdvector<double> get_face_nodes(unsigned int P);
    mdvector<double> get_face_weights(unsigned int P);

    void project_face_point(int face, const double* loc, double* ploc);

    double calc_nodal_face_basis(unsigned int pt, double *loc);

    double calc_orthonormal_basis(unsigned int mode, double *loc);

  public:
    Tris(GeoStruct *geo, InputStruct *input, unsigned int elesObjID, unsigned int startEle, unsigned int endEle, int order = -1);

    void setup_PMG(int pro_order, int res_order);
    void setup_ppt_connectivity();

    double calc_d_nodal_basis_fr(unsigned int spt, const std::vector<double>& loc, unsigned int dim);

    void modify_sensor();
};

#endif /* tris_hpp */
