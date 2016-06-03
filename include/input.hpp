#ifndef input_hpp
#define input_hpp

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
  NONE = -1,
  PERIODIC = 0,
  CHAR = 1,
  CHAR_PYFR = 2,
  SUP_IN = 3,
  SUP_OUT = 4,
  SUB_IN = 5,
  SUB_OUT = 6,
  SUB_IN_CHAR = 7,
  SUB_OUT_CHAR = 8,
  SLIP_WALL = 9,
  ISOTHERMAL_NOSLIP = 10,
  ADIABATIC_NOSLIP = 11,
  ISOTHERMAL_NOSLIP_MOVING = 12,
  ADIABATIC_NOSLIP_MOVING = 13,
  OVERSET = 14,
  SYMMETRY = 15
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

  /* Implicit Parameters */
  bool SER, inv_mode, stream_mode, backsweep, LU_pivot;
  unsigned int Jfreeze_freq, nColors, n_LHS_blocks;

  /* --- Additional Mesh Variables --- */
  std::map<std::string,std::string> meshBounds;

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
