/* Copyright (C) 2017 Aerospace Computing Laboratory (ACL).
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

#include "integrator.hpp"

#include "cblas.h"

void ODEIntegrator::add_equation(std::shared_ptr<ODEquation> eq)
{
  equations.push_back(eq);
}

void ODEIntegrator::eval_function(void)
{

}

void ODEIntegrator::do_step(void)
{
  for (auto &eq : equations)
    eq->eval_residual();
}


void ODEquation::add_registers(int r1, int r2, double fac)
{
  cblas_daxpy(reg_size, fac, regs[r1], 1, regs[r2], 1);
}
