#ifndef input_hpp
#define input_hpp

#include <array>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "inputstruct.hpp"
#include "macros.hpp"

#ifdef _MPI // This is kinda hacky, but it works and keeps things simple
#define _mpi_comm MPI_Comm
#else
#define _mpi_comm int
#endif

extern double pi;

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
  SYMMETRY_G,
  WALL_CLOSURE,
  OVERSET_CLOSURE
};

extern std::map<std::string,int> bcStr2Num;

/*! Useful 3D point object with simple geometric functions */
struct point
{
  double x, y, z;

  point() {
    x = 0;
    y = 0;
    z = 0;
  }

  point (double _x, double _y, double _z) {
    x = _x;
    y = _y;
    z = _z;
  }

  point(double* pt, int nDims=3) {
    x = pt[0];
    y = pt[1];
    if (nDims==3)
      z = pt[2];
    else
      z = 0;
  }

  void zero() {
    x = 0;
    y = 0;
    z = 0;
  }

  double& operator[](int ind) {
    switch(ind) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        std::cout << "ind = " << ind << ": " << std::flush;
        ThrowException("Invalid index for point struct.");
    }
  }

  double operator[](int ind) const {
    switch(ind) {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        std::cout << "ind = " << ind << ": " << std::flush;
        ThrowException("Invalid index for point struct.");
    }
  }

  point operator=(double* a) {
    struct point pt;
    pt.x = a[0];
    pt.y = a[1];
    pt.z = a[2];
    return pt;
  }

  point operator-(point b) {
    struct point c;
    c.x = x - b.x;
    c.y = y - b.y;
    c.z = z - b.z;
    return c;
  }

  point operator+(point b) {
    struct point c;
    c.x = x + b.x;
    c.y = y + b.y;
    c.z = z + b.z;
    return c;
  }

  point operator/(point b) {
    struct point c;
    c.x = x / b.x;
    c.y = y / b.y;
    c.z = z / b.z;
    return c;
  }

  point& operator+=(point b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }

  point& operator-=(point b) {
    x -= b.x;
    y -= b.y;
    z -= b.z;
    return *this;
  }

  point& operator+=(double* b) {
    x += b[0];
    y += b[1];
    z += b[2];
    return *this;
  }

  point& operator-=(double* b) {
    x -= b[0];
    y -= b[1];
    z -= b[2];
    return *this;
  }

  point& operator/=(double a) {
    x /= a;
    y /= a;
    z /= a;
    return *this;
  }

  point& operator*=(double a) {
    x *= a;
    y *= a;
    z *= a;
    return *this;
  }

  double operator*(point b) {
    return x*b.x + y*b.y + z*b.z;
  }

  void abs(void) {
    x = std::abs(x);
    y = std::abs(y);
    z = std::abs(z);
  }

  double norm(void) {
    return std::sqrt(x*x+y*y+z*z);
  }

  point cross(point b) {
    point v;
    v.x = y*b.z - z*b.y;
    v.y = z*b.x - x*b.z;
    v.z = x*b.y - y*b.x;
    return v;
  }

};

//! For clearer notation when a vector is implied, rather than a point
typedef struct point Vec3;

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

/* Function to read vector of parameters from input file. */
template <typename T>
void read_param_vec(std::ifstream &f, std::string name, std::vector<T> &vec)
{
  if (vec.size() != 0)
  {
    ThrowException("Trying to assign input parameters to a vector that has data!");
  }

  std::string param;

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    if (param == name)
    {
      std::string line;
      std::getline(f, line);
      std::stringstream ss(line);

      T val;
      while (ss >> val)
      {
        vec.push_back(val);
      }

      return;
    }
  }
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
