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

#include <iostream>
#include <vector>
#include <memory>

class ODEquation
{
private:
  unsigned int reg_size;
  std::vector<double*> regs;

public:
  virtual void eval_residual(void) =0;

  virtual void eval_error(void) =0;

  virtual void assign_registers(int nreg) =0;

  //! r2 += fac*r1
  void add_registers(int r1, int r2, double fac);
};

class ODEIntegrator
{
protected:
  std::vector<std::shared_ptr<ODEquation>> equations;

  unsigned int n_steps = 0;
  unsigned int n_rejected = 0;

  int time_order;
  int n_stages;

  double time;

public:
  void add_equation(std::shared_ptr<ODEquation> eq);

  void eval_function(void);

  void do_step(void);

  void run_to_time(double T);

  int get_integrator_order(void) { return time_order; }
  int get_num_stages(void) { return n_stages; }
};


class RKStepper : public ODEIntegrator
{
private:

public:

};
