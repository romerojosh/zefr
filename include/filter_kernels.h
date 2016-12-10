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

#include "mdvector_gpu.h"

void normalize_data_wrapper(mdvector_gpu<double> &U_spts, double normalTol, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars);

void compute_max_sensor_wrapper_ka(mdvector_gpu<double> &KS, mdvector_gpu<double> &sensor,
    unsigned int order, double &max_sensor, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

void copy_filtered_solution_wrapper_ka(mdvector_gpu<double> &U_spts_filt, mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars);

void compute_max_sensor_wrapper(mdvector_gpu<double> &KS, mdvector_gpu<double> &sensor,
    unsigned int order, double &max_sensor, mdvector_gpu<uint> &sensor_bool, double threshJ, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars, unsigned int nDims, double Q);

void copy_filtered_solution_wrapper(mdvector_gpu<double> &U_spts_filt, mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, int type);

void limiter_wrapper(uint nEles, uint nFaces, uint nVars, double threshJ, mdvector_gpu<int> &ele_adj_d,
  mdvector_gpu<double> &sensor_d, mdvector_gpu<double> &Umodal_d);
