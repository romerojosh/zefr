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
    void set_vandermonde_mats();

  public:
    Tris(GeoStruct *geo, InputStruct *input, int order = -1);

    void setup_PMG(int pro_order, int res_order);
    void setup_ppt_connectivity();

    /* Routines for implicit method */
    void transform_dFdU();
    double calc_d_nodal_basis_fr(unsigned int spt, const std::vector<double>& loc, unsigned int dim);
};

#endif /* tris_hpp */