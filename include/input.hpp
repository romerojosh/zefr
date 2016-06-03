#ifndef input_hpp
#define input_hpp
//#define _CPU
//#define _MPI
#include <array>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "mdvector.hpp"
#include "macros.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

enum EQN {AdvDiff = 0, EulerNS = 1, Burgers = 2};

/*! Enumeration for original, mesh-file-defined face type */
enum FACE_TYPE {
  HOLE_FACE = -1,
  INTERNAL  = 0,
  BOUNDARY  = 1,
  MPI_FACE  = 2,
  OVER_FACE = 3
};

/*! Enumeration for original, mesh-file-defined node type */
enum NODE_TYPE {
  NORMAL_NODE = 0,
  OVERSET_NODE = 1,
  BOUNDARY_NODE = 2
};

/*! Enumeration for mesh (either create cartesian mesh or read from file) */
enum meshType {
  READ_MESH   = 0,
  CREATE_MESH = 1,
  OVERSET_MESH = 2
};

enum EQUATION {
  ADVECTION_DIFFUSION = 0,
  NAVIER_STOKES       = 1
};

/*! Enumeration for all available boundary conditions */
enum BC_TYPE {
  NONE = 0,
  PERIODIC,
  CHAR,
  SUP_IN,
  SUP_OUT,
  SUB_IN,
  SUB_OUT,
  SUB_IN_CHAR,
  SUB_OUT_CHAR,
  SLIP_WALL_P,
  SLIP_WALL_G,
  ISOTHERMAL_NOSLIP_P,
  ISOTHERMAL_NOSLIP_G,
  ADIABATIC_NOSLIP_P,
  ADIABATIC_NOSLIP_G,
  ISOTHERMAL_NOSLIP_MOVING_P,
  ISOTHERMAL_NOSLIP_MOVING_G,
  ADIABATIC_NOSLIP_MOVING_P,
  ADIABATIC_NOSLIP_MOVING_G,
  OVERSET,
  SYMMETRY_P,
  SYMMETRY_G
};

extern std::map<std::string,int> bcStr2Num;

struct InputStruct
{
  unsigned int equation, dt_type, CFL_type, ic_type, nDims, nQpts1D, n_steps, order, low_order, smooth_steps, p_smooth_steps, c_smooth_steps, f_smooth_steps, adapt_CFL;
  unsigned int report_freq, write_freq, force_freq, res_type, error_freq, test_case, err_field, FMG_vcycles;
  std::string output_prefix, meshfile, spt_type, dt_scheme, restart_file, mg_cycle;
  bool viscous, p_multi, restart, fix_vis, squeeze, serendipity, source;
  std::string fconv_type, fvisc_type;
  double rus_k, ldg_b, ldg_tau; 
  double AdvDiff_D, dt, res_tol, CFL, rel_fac, CFL_max, CFL_ratio;
  mdvector<double> AdvDiff_A, V_fs, norm_fs, V_wall, norm_wall;
  double T_gas, gamma, prandtl, mu, R, S;
  double rho_fs, u_fs, v_fs, P_fs;
  double mach_fs, L_fs, T_fs, Re_fs, nx_fs, ny_fs, T_tot_fs, P_tot_fs;
  double mach_wall, T_wall, nx_wall, ny_wall, u_wall, v_wall;
  double T_ref, P_ref, rho_ref, mu_ref, time_ref, R_ref, c_sth, rt;
  double exps0, s_factor;
  unsigned int rank, nRanks;
  unsigned int filt_on, sen_write, sen_norm, filt_maxLevels;
  double sen_Jfac, filt_gamma;
  double iter, initIter, time, rkTime;

  /* --- Overset / Moving-Grid Variables --- */
  bool motion, overset, use_lgp;
  unsigned int oversetMethod, nGrids, quad_order, motion_type;
  std::vector<std::string> oversetGrids;

  double moveAx, moveAy, moveFx, moveFy;


  /* --- Additional Mesh Variables --- */
  std::map<std::string,std::string> meshBounds;

  /* Implicit Parameters */
  bool SER, inv_mode, stream_mode, backsweep, LU_pivot;
  unsigned int Jfreeze_freq, nColors, n_LHS_blocks;

#ifdef _GPU
  mdvector_gpu<double> AdvDiff_A_d, V_fs_d, norm_fs_d, V_wall_d, norm_wall_d;
#endif
};

InputStruct read_input_file(std::string inputfile);
void apply_nondim(InputStruct &input);

/* Function to read parameter from input file. Throws exception if parameter 
 * is not found. */
template <typename T>
void read_param(std::ifstream &f, std::string name, T &var)
{
  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  std::string param;

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    if (param == name)
    {
      f >> var;
      return;
    }
  }

  ThrowException("Input parameter " + name + " not found!");

}

/* Function to read parameter from input file. Sets var to provided default
 * value if parameter is not found. */
template <typename T>
void read_param(std::ifstream &f, std::string name, T &var, T default_val)
{
  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  std::string param;

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    if (param == name)
    {
      f >> var;
      return;
    }
  }

  var = default_val;
}

template<typename T>
void read_vector(std::ifstream &f, std::string name, unsigned int &nVars, std::vector<T> &vars)
{
  std::string str, optKey;

  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  // Rewind to the start of the file
  f.seekg(0, f.beg);

  // Search for the given option string
  while (std::getline(f, str)) {
    // Remove any leading whitespace & see if first word is the input option
    std::stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(name)==0) {
      if (!(ss >> nVars)) {
        // This could happen if, for example, trying to assign a string to a double
        cerr << "WARNING: Unable to read number of entries for vector option " << name << endl;
        string errMsg = "Required option not set: " + name;
        ThrowException(errMsg.c_str());
      }

      vars.resize(nVars);
      for (unsigned int i = 0; i < nVars; i++) {
        if (!(ss >> vars[i])) {
          std::cerr << "WARNING: Unable to assign all values to vector option " << name << std::endl;
          std::string errMsg = "Required option not set: " + name;
          ThrowException(errMsg.c_str())
        }
      }

      return;
    }
  }

  // Option was not found; throw error & exit
  std::string errMsg = "Required option not found: " + name;
  ThrowException(errMsg.c_str());
}

/*! Read in a map of type <T,U> from input file; each entry prefaced by optName
 *  i.e. 'mesh_bound  airfoil  wall_ns_adi'
 */
template<typename T, typename U>
void read_map(std::ifstream &f, std::string optName, std::map<T,U> &opt) {
  std::string str, optKey;
  T tmpT;
  U tmpU;
  bool found;

  if (!f.is_open())
  {
    ThrowException("Input file not open for reading!");
  }

  // Rewind to the start of the file
  f.clear();
  f.seekg(0,f.beg);

  // Search for the given option string
  while (std::getline(f, str))
  {
    // Remove any leading whitespace & see if first word is the input option
    std::stringstream ss;
    ss.str(str);
    ss >> optKey;
    if (optKey.compare(optName) == 0)
    {
      found = true;
      if (!(ss >> tmpT >> tmpU))
      {
        // This could happen if, for example, trying to assign a string to a double
        std::cerr << "WARNING: Unable to assign value to option " << optName << std::endl;
        std::string errMsg = "Required option not set: " + optName;
        ThrowException(errMsg.c_str());
      }

      opt[tmpT] = tmpU;
      optKey = "";
    }
  }

  if (!found)
  {
    // Option was not found; throw error & exit
    std::string errMsg = "Required option not found: " + optName;
    ThrowException(errMsg.c_str());
  }
}

#endif /* input_hpp */
