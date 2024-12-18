#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#include "input.hpp"
#include "macros.hpp"

double pi = 3.141592653589793;

std::map<std::string,int> bcStr2Num = {
  {"none", NONE},
  {"fluid", NONE},
  {"periodic", PERIODIC},
  {"char", CHAR},
  {"farfield", SUP_IN},
  {"inlet_sup", SUP_IN},
  {"outlet_sup", SUP_OUT},
  {"wall_slip", SLIP_WALL},
  {"wall_ns_iso", ISOTHERMAL_NOSLIP},
  {"wall_ns_iso_move", ISOTHERMAL_NOSLIP_MOVING},
  {"wall_ns_adi", ADIABATIC_NOSLIP},
  {"wall_ns_adi_move", ADIABATIC_NOSLIP_MOVING},
  {"overset", OVERSET},
  {"symmetry", SYMMETRY},
  {"wall_closure", WALL_CLOSURE},
  {"overset_closure", OVERSET_CLOSURE},
};

InputStruct read_input_file(std::string inputfile)
{
  std::ifstream f(inputfile);

  InputStruct input;
  std::string str;

  read_param(f, "nDims", input.nDims);
  read_param(f, "meshfile", input.meshfile);

  // Get mesh boundaries and boundary conditions, then convert to lowercase
  std::map<std::string, std::string> meshBndTmp;
  read_map(f, "mesh_bound", meshBndTmp);
  for (auto& B:meshBndTmp) {
    std::string tmp1, tmp2;
    tmp1 = B.first; tmp2 = B.second;
    std::transform(tmp1.begin(), tmp1.end(), tmp1.begin(), ::tolower);
    std::transform(tmp2.begin(), tmp2.end(), tmp2.begin(), ::tolower);
    input.meshBounds[tmp1] = tmp2;
  }

  read_param(f, "order", input.order);
  read_param(f, "equation", str, std::string("EulerNS"));

  if (str == "AdvDiff")
    input.equation = AdvDiff;
  else if (str == "EulerNS")
    input.equation = EulerNS;
  else
    ThrowException("Equation not recognized!");

  read_param(f, "viscous", input.viscous);
  read_param(f, "disable_nondim", input.disable_nondim, false);
  read_param(f, "source", input.source, false);
  read_param(f, "squeeze", input.squeeze, false);
  read_param(f, "s_factor", input.s_factor, 0.0);

  read_param(f, "n_steps", input.n_steps);
  read_param(f, "tfinal", input.tfinal, 1e15);
  read_param(f, "res_tol", input.res_tol, 0.0);
  read_param(f, "res_field", input.res_field, (unsigned int) 0);
  read_param(f, "dt_scheme", input.dt_scheme);
  if (input.dt_scheme != "Steady")
    read_param(f, "dt", input.dt);
  read_param(f, "dt_type", input.dt_type, (unsigned int) 0);
  if (input.dt_type != 0)
  {
    read_param(f, "CFL", input.CFL, 1.0);
    read_param(f, "CFL_type", input.CFL_type, (unsigned int) 1);
  }

  if (input.dt_scheme == "Euler")
    input.nStages = 1;
  else if (input.dt_scheme == "RK44")
    input.nStages = 4;
  else if (input.dt_scheme == "RK54")
    input.nStages = 5;
  else if (input.dt_scheme == "RKj")
    input.nStages = 4;
  else if (input.dt_scheme == "Steady")
  {
    input.nStages = 1;
    input.implicit_method = true;
    input.implicit_steady = true;
  }
  else if (input.dt_scheme == "DIRK34")
  {
    input.nStages = 3;
    input.implicit_method = true;
  }
  else if (input.dt_scheme == "ESDIRK43")
  {
    input.nStages = 4;
    input.implicit_method = true;
  }
  else if (input.dt_scheme == "ESDIRK64")
  {
    input.nStages = 6;
    input.implicit_method = true;
  }
  else
    ThrowException("Unknown dt_scheme");

  read_param(f, "adapt_dt", input.adapt_dt, false);
  if (input.adapt_dt)
  {
    // NOTE: to reduce time step size (generally speaking), reduce atol and rtol
    read_param(f, "err_atol", input.atol, 0.00001);
    read_param(f, "err_rtol", input.rtol, 0.00001);
    read_param(f, "pi_alpha", input.pi_alpha, 0.7);
    read_param(f, "pi_beta", input.pi_beta, 0.4);

    read_param(f, "safety_factor", input.sfact, 0.8);
    read_param(f, "max_factor", input.maxfac, 2.5);
    read_param(f, "min_factor", input.minfac, 0.3);

    read_param(f, "max_dt", input.max_dt, 100.);
  }

  read_param(f, "restart", input.restart, false);
  read_param(f, "restart_file", input.restart_file, std::string(""));
  read_param(f, "restart_case", input.restart_case, std::string(""));
  read_param(f, "restart_type", input.restart_type, (unsigned int)0);
  read_param(f, "restart_iter", input.restart_iter, (unsigned int)0);
  read_param(f, "restart_npart", input.restart_npart, -1);

  read_param(f, "mg_cycle", input.mg_cycle, std::string("V"));
  read_param(f, "FMG_vcycles", input.FMG_vcycles, (unsigned int) 1);
  read_param(f, "p_multi", input.p_multi, false);
  read_param(f, "rel_fac", input.rel_fac, 1.0);
  read_param_vec(f, "mg_levels", input.mg_levels);
  read_param_vec(f, "mg_steps", input.mg_steps);

  read_param(f, "catch_signals", input.catch_signals, false);
  read_param(f, "output_prefix", input.output_prefix);
  read_param(f, "write_paraview", input.write_paraview, (short)1);
  read_param(f, "write_pyfr", input.write_pyfr, (short)0);
  read_param(f, "plot_surfaces", input.plot_surfaces, (short)0);
  read_param(f, "plot_overset", input.plot_overset, (short)0);
  read_param(f, "write_LHS", input.write_LHS, false);
  read_param(f, "write_RHS", input.write_RHS, false);
  read_param(f, "write_freq", input.write_freq);
  read_param(f, "report_freq", input.report_freq);
  read_param(f, "res_type", input.res_type);
  read_param(f, "force_freq", input.force_freq, (unsigned int) 0);
  read_param(f, "error_freq", input.error_freq, (unsigned int) 0);
  read_param(f, "turb_stat_freq", input.turb_stat_freq, (unsigned int) 0);
  read_param(f, "write_tavg_freq", input.write_tavg_freq, (unsigned int) 0);
  read_param(f, "tavg_freq", input.tavg_freq, (unsigned int) 100);
  if (input.write_tavg_freq > 0 && input.tavg_freq > 0)
  {
    input.tavg = true;
  }
  else
  {
    input.write_tavg_freq = 0;
    input.tavg_freq = 0;
    input.tavg = false;
  }

  read_param(f, "test_case", input.test_case, (unsigned int) 0);
  read_param(f, "err_field", input.err_field, (unsigned int) 0);
  read_param(f, "nQpts1D", input.nQpts1D, (unsigned int)5);
  if (input.error_freq == 0 and input.nQpts1D != 0)
    input.nQpts1D = 0;  // if no error computation, no need for quadrature alloc

  read_param(f, "fconv_type", str, std::string("Rusanov"));
  if (str == "Rusanov")
    input.fconv_type = Rusanov;
  else
    ThrowException("Convective interface flux not recognized!");

  read_param(f, "fvisc_type", str, std::string("LDG"));
  if (str == "LDG")
    input.fvisc_type = LDG;
  else
    ThrowException("Viscous interface flux not recognized!");

  read_param(f, "rus_k", input.rus_k, 0.0);
  read_param(f, "ldg_b", input.ldg_b, 0.5);
  read_param(f, "ldg_tau", input.ldg_tau, 1.0);
  read_param(f, "spt_type", input.spt_type, std::string("Legendre"));

  read_param(f, "ic_type", input.ic_type);

  input.AdvDiff_A.assign({3});
  read_param(f, "AdvDiff_Ax", input.AdvDiff_A(0), 1.0);
  read_param(f, "AdvDiff_Ay", input.AdvDiff_A(1), 1.0);
  read_param(f, "AdvDiff_Az", input.AdvDiff_A(2), 1.0);
  if (input.viscous)
    read_param(f, "AdvDiff_D", input.AdvDiff_D, 0.1);
  else
    input.AdvDiff_D = 0;

  read_param(f, "T_gas", input.T_gas, 291.15);
  read_param(f, "gamma", input.gamma, 1.4);
  read_param(f, "R", input.R, 286.9);
  read_param(f, "prandtl", input.prandtl, 0.72);
  read_param(f, "S", input.S, 120.0);

  read_param(f, "rho_fs", input.rho_fs, 1.4);
  read_param(f, "P_fs", input.P_fs, 1.0);

  read_param(f, "fix_vis", input.fix_vis, false);
  read_param(f, "mach_fs", input.mach_fs, 0.2);
  read_param(f, "Re_fs", input.Re_fs, 200.0);
  read_param(f, "L_fs", input.L_fs, 1.0);
  read_param(f, "T_fs", input.T_fs, 300.0);
  input.norm_fs.assign({3});
  read_param(f, "nx_fs", input.norm_fs(0), 1.0);
  read_param(f, "ny_fs", input.norm_fs(1), 0.0);
  read_param(f, "nz_fs", input.norm_fs(2), 0.0);

  read_param(f, "mach_wall", input.mach_wall, 0.0);
  read_param(f, "T_wall", input.T_wall, 300.0);
  input.norm_wall.assign({3});
  read_param(f, "nx_wall", input.norm_wall(0), 1.0);
  read_param(f, "ny_wall", input.norm_wall(1), 0.0);
  
  read_param(f, "filt_on", input.filt_on, (unsigned int) 0);
  read_param(f, "sen_write", input.sen_write, (unsigned int) 1);
  read_param(f, "sen_norm", input.sen_norm, (unsigned int) 1);
  read_param(f, "sen_Jfac", input.sen_Jfac, 1.0);
  read_param(f, "alpha", input.alpha, 1.0);
  read_param(f, "filtexp", input.filtexp, 2.0);
  read_param(f, "nonlin_exp", input.nonlin_exp, 2.0);

  if (input.filt_on && input.order <= 1)
  {
    std::cout << "WARNING: requesting filtering yet P <= 1. Filtering will not be used." << std::endl;
    input.filt_on = 0;
  }

  /* Implicit parameters */
  if (input.implicit_method)
  {
    /* Compute residual Jacobian using finite difference approximation */
    read_param(f, "FDA_Jacobian", input.FDA_Jacobian, false);

    /* Form Jacobians based on Kronecker product formulation */
    // Note: only valid for quad and hex elements
    read_param(f, "KPF_Jacobian", input.KPF_Jacobian, false);

    /* Freeze Jacobian across all stages in unsteady simulation */
    if (input.implicit_steady)
      input.freeze_Jacobian = false;
    else
      read_param(f, "freeze_Jacobian", input.freeze_Jacobian, true);

    /* Pseudo Timestepping */
    read_param(f, "pseudo_time", input.pseudo_time, false);
    if (input.pseudo_time)
    {
      read_param(f, "dtau", input.dtau, 1.0);
      read_param(f, "dtau_type", input.dtau_type, (unsigned int) 0);
      if (input.dtau_type != 0)
      {
        read_param(f, "CFL_tau", input.CFL_tau, 1.0);
        read_param(f, "CFL_tau_type", input.CFL_tau_type, (unsigned int) 1);
      }

      // Note: Removing deltaU sometimes speeds up convergence
      read_param(f, "remove_deltaU", input.remove_deltaU, false);

      // Exponential growth rate
      read_param(f, "adapt_dtau", input.adapt_dtau, false);
      if (input.adapt_dtau)
      {
        read_param(f, "dtau_ratio_max", input.dtau_ratio_max, 1.0);
        read_param(f, "dtau_growth_rate", input.dtau_growth_rate, 0.0);
      }
    }

    /* Newton Steps */
    if (input.implicit_steady)
    {
      read_param(f, "iterNM_max", input.iterNM_max, (unsigned int) 0);
      if (input.iterNM_max > 0) input.n_steps = input.iterNM_max;
    }
    else
      read_param(f, "iterNM_max", input.iterNM_max, (unsigned int) 1);
    read_param(f, "report_NMconv_freq", input.report_NMconv_freq, (unsigned int) 0);

    /* Block iterative method */
    read_param(f, "iterative_method", str, std::string("JAC"));
    read_param(f, "iterBM_max", input.iterBM_max, (unsigned int) 100);
    read_param(f, "backsweep", input.backsweep, false);
    read_param(f, "report_BMconv_freq", input.report_BMconv_freq, (unsigned int) 0);
    if (str == "JAC")
      input.iterative_method = JAC;
    else if (str == "MCGS")
    {
      input.iterative_method = MCGS;
      read_param(f, "nColors", input.nColors, (unsigned int) 1);
    }
    else
      ThrowException("Iterative method not recognized!");

    if (input.report_NMconv_freq > 0 && input.report_BMconv_freq == 0)
      input.report_BMconv_freq = input.iterBM_max;

    /* Block Linear Solver */
    read_param(f, "linear_solver", str, std::string("LU"));
    if (str == "LU")
      input.linear_solver = LU;
    else if (str == "INV")
      input.linear_solver = INV;
    else if (str == "SVD")
    {
      input.linear_solver = SVD;
      read_param(f, "svd_omg", input.svd_omg, 0.5);
      read_param(f, "svd_cutoff", input.svd_cutoff, 1.0);
    }
    else
      ThrowException("Linear solver not recognized!");

    /* Restrictions */
    if (input.equation == EulerNS && input.viscous && !input.fix_vis)
      ThrowException("Viscous implicit Jacobians with Sutherland's law not available!");
#ifdef _CPU
    if (input.KPF_Jacobian)
      ThrowException("Implicit KPF Jacobians only implemented on GPU!");
#endif
#ifdef _GPU
    if (input.FDA_Jacobian)
      ThrowException("Implicit FDA Jacobians only implemented on CPU!");
#endif
  }

  read_param(f, "overset", input.overset, false);
  read_param_vec(f, "overset_grids", input.oversetGrids);
  read_param_vec(f, "overset_grid_type", input.gridTypes);
  if (input.gridTypes.size() < input.oversetGrids.size())
    input.gridTypes.assign(input.oversetGrids.size(), 1);

  read_param(f, "motion", input.motion, false);
  read_param(f, "motion_type", input.motion_type, (int)STATIC);
  if (!input.motion) input.motion_type = STATIC;

  switch(input.motion_type)
  {
    case STATIC:
      input.motion = false;
      break;

    case TEST1:
    case TEST2:
    case TEST3:
      break;

    case CIRCULAR_TRANS:
    {
      read_param(f, "moveAx", input.moveAx);
      read_param(f, "moveAy", input.moveAy);
      read_param(f, "moveFx", input.moveFx);
      read_param(f, "moveFy", input.moveFy);
      if (input.nDims == 3)
      {
        read_param(f, "moveAz", input.moveAz, 0.0);
        read_param(f, "moveFz", input.moveFz, 0.0);
      }

      break;
    }

    case RIGID_BODY:
    {
      if (input.nDims != 3)
        ThrowException("Rigid-Body motion implemented in 3D only");

      // Acceleration due to gravity [NOTE: assumed along -z axis]
      read_param(f, "g", input.g, 0.);

      // Do fully dynamic 6DOF simulation by computing forces & moments on body
      read_param(f, "full_6dof", input.full_6dof, false);

      // Initial translational velocity
      read_param(f, "vx0", input.v0[0], 0.);
      read_param(f, "vy0", input.v0[1], 0.);
      read_param(f, "vz0", input.v0[2], 0.);

      // Initial angular velocity
      read_param(f, "wx0", input.w0[0], 0.);
      read_param(f, "wy0", input.w0[1], 0.);
      read_param(f, "wz0", input.w0[2], 0.);

      // Mass of moving body
      read_param(f, "mass", input.mass);

      // Diagonal components of inertia tensor [in initial 'world' coordinates]
      read_param(f, "Ixx", input.Imat[0]);
      read_param(f, "Iyy", input.Imat[4]);
      read_param(f, "Izz", input.Imat[8]);

      // Off-diagonal components
      read_param(f, "Ixy", input.Imat[1], 0.);
      read_param(f, "Ixz", input.Imat[2], 0.);
      read_param(f, "Iyz", input.Imat[5], 0.);

      // Apply symmetry
      input.Imat[3] = input.Imat[1];
      input.Imat[6] = input.Imat[2];
      input.Imat[7] = input.Imat[5];

      break;
    }

    default:
    {
      std::stringstream ss("Error: Given motion_type not supported: ");
      ss << std::to_string(input.motion_type) << std::endl;
      ThrowException(ss.str().c_str());
    }
  }

  f.close();

  /* If running Euler or Navier-Stokes, similarity parameters to calculate remaining quantities */
  if (input.equation == EulerNS)
  {
    input.v_mag_fs = input.mach_fs * std::sqrt(input.gamma * input.P_fs / input.rho_fs); // mach * c
    input.V_fs.assign({3});

    // Normalize freestream normal vector
    double magn = 0.;
    for (int i = 0; i < 3; i++)
      magn += input.norm_fs(i) * input.norm_fs(i);
    magn = std::sqrt(magn);
    for (int i = 0; i < 3; i++)
      input.norm_fs(i) /= magn;

    if (input.viscous)
    {
      apply_nondim(input);
    }
    else
    {
      for (int i = 0; i < 3; i++)
        input.V_fs(i) = input.norm_fs(i) * input.v_mag_fs;
    }
  }

  /* If using polynomial squeezing, set entropy bound */
  /* NOTES: This bound seems to play a large role in convergence. */
  if (input.squeeze)
  {
    input.exps0 = input.s_factor * (input.P_fs / std::pow(input.rho_fs, input.gamma));
  }

  return input;
}

void apply_nondim(InputStruct &input)
{
  input.V_wall.assign({3});

  if (input.disable_nondim) 
  { 
    /* Run with dimensional quantities from input file:
     * ++ Re, Ma, rho, L, p, gamma, Prandtl specified
     * ++ Remaining parameters calculated for consistency */
    input.T_fs = input.P_fs / (input.rho_fs * input.R); // Ideal gas law -> T
    input.mu = input.rho_fs * input.v_mag_fs * input.L_fs  / input.Re_fs; // Re -> mu

    input.R_ref = input.R;

    input.T_tot_fs = input.T_fs * (1.0 + 0.5*(input.gamma - 1.0)*input.mach_fs*input.mach_fs);
    input.P_tot_fs = input.P_fs * std::pow(1.0 + 0.5*(input.gamma - 1.0)*input.mach_fs*input.mach_fs,
        input.gamma / (input.gamma - 1.0));

    for (unsigned int dim = 0; dim < input.nDims; dim++)
      input.V_fs(dim) = input.v_mag_fs * input.norm_fs(dim);

    double V_wall_mag = input.mach_wall * std::sqrt(input.gamma * input.R * input.T_wall);
    for (unsigned int n = 0; n < input.nDims; n++)
      input.V_wall(n) = V_wall_mag * input.norm_wall(n) / input.v_mag_fs;
    
    input.fix_vis = 1;

    return;
  }

  /* Compute dimensional freestream quantities */

  double V_fs_mag = input.mach_fs * std::sqrt(input.gamma * input.R * input.T_fs);
  for (unsigned int dim = 0; dim < input.nDims; dim++)
    input.V_fs(dim) = V_fs_mag * input.norm_fs(dim);

  input.mu = (input.rho_fs * V_fs_mag * input.L_fs) / input.Re_fs;

  /* If using Sutherland's law, update viscosity */
  if (!input.fix_vis)
    input.mu = input.mu * std::pow(input.T_fs / input.T_gas, 1.5) * ((input.T_gas + input.S)/
        (input.T_fs + input.S));

  input.rho_fs = input.mu * input.Re_fs / (V_fs_mag * input.L_fs);
  input.P_fs = input.rho_fs * input.R * input.T_fs;
  input.rt = input.T_gas * input.R / (V_fs_mag * V_fs_mag);
  input.c_sth = input.S / input.T_gas;

  /* -- Set reference quantities for nondimensionalization -- */
  input.T_ref = input.T_fs;
  input.rho_ref = input.rho_fs;
  input.P_ref = input.rho_fs * V_fs_mag * V_fs_mag;
  input.mu_ref = input.rho_fs * V_fs_mag;
  input.R_ref = input.R * input.T_fs / (V_fs_mag * V_fs_mag);

  /* Nondimensionalize freestream quantities */
  input.mu = input.mu/input.mu_ref;
  input.rho_fs = input.rho_fs/input.rho_ref;
  for (unsigned int n = 0; n < input.nDims; n++)
    input.V_fs(n) = input.V_fs(n) / V_fs_mag;
  input.P_fs = input.P_fs / input.P_ref;
  input.T_tot_fs = (input.T_fs / input.T_ref) * (1.0 + 0.5 * (input.gamma - 1.0) * 
      input.mach_fs * input.mach_fs);
  input.P_tot_fs = input.P_fs * std::pow(1.0 + 0.5 * (input.gamma - 1.0) * input.mach_fs * 
      input.mach_fs, input.gamma / (input.gamma - 1.0));

  /* Compute and nondimensionalize wall quantities */
  double V_wall_mag = input.mach_wall * std::sqrt(input.gamma * input.R * input.T_wall);
  for (unsigned int n = 0; n < input.nDims; n++)
    input.V_wall(n) = V_wall_mag * input.norm_wall(n) / V_fs_mag;

  input.T_wall = input.T_wall / input.T_ref;

  input.v_mag_fs = 1.0;
  input.T_fs = 1.0;
  input.R = input.R_ref;
}

//! Apply any changes to input parameters from external wrapping layer
void initialize_inputs(InputStruct &input)
{
  switch (input.nStages)
  {
    case 1:
      input.dt_scheme = "Euler"; break;
    case 4:
      input.dt_scheme = "RK44"; break;
    case 5:
      input.dt_scheme = "RK54"; break;
  }

  input.implicit_method = false;
  input.implicit_steady = false;
  input.dt_type = 0;

  input.initIter = input.iter;

  if (input.viscous && input.equation == EulerNS)
    apply_nondim(input);
}
