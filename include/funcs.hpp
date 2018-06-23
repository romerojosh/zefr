#ifndef funcs_hpp
#define funcs_hpp

#include <cmath>
#include <array>

extern "C" {
#include "cblas.h"
}

#include "input.hpp"
#include "mdvector.hpp"

//! Quaternion object [q0 + q1 i + q2 j + q3 k]
struct Quat
{
private:
  double q[4] = {0,0,0,0};

public:
  Quat(void) {}

  Quat(double a, double b, double c, double d)
  {
    q[0] = a; q[1] = b; q[2] = c; q[3] = d;
  }

  void operator=(const Quat &p)
  {
    q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = p[3];
  }

  double& operator[](unsigned i) { return q[i]; }
  double  operator[](unsigned i) const { return q[i]; }

  Quat operator*(const Quat &p);
  Quat operator*(const std::array<double,3> &p);

  Quat operator+(const Quat &p);
  Quat operator-(const Quat &p);

  void operator+=(const Quat &p);
  void operator-=(const Quat &p);

  Quat cross(const Quat &p);

  Quat conj(void);

  double norm(void) const
  {
    return std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  }

  void normalize(void)
  {
    double N = this->norm();
    for (unsigned i = 0; i < 4; i++)
      q[i] /= N;
  }
};

Quat operator*(double a, const Quat &b);

/* Computes solution at specified time and location */
double compute_U_init(double x, double y, double z, unsigned int var, const InputStruct *input);
double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input);
double compute_dU_true(double x, double y, double z, double t, unsigned int var, 
    unsigned int dim, const InputStruct *input);

/* Computes source term as specified time and location */
double compute_source_term(double x, double y, double z, double t, unsigned int var, const InputStruct *input);

/* Compute maximum CFL */
double get_cfl_limit_adv(int order);
double get_cfl_limit_diff(int order, double beta);

unsigned int tri_nodes_to_order(unsigned int nNodes);
unsigned int tet_nodes_to_order(unsigned int nNodes);
unsigned int pri_nodes_to_order(unsigned int nNodes);
unsigned int npts_to_order(ELE_TYPE etype, unsigned int npts);

/* Blocked gemm operation using OpenMP */
#ifdef _OMP
void omp_blocked_dgemm(CBLAS_ORDER mode, CBLAS_TRANSPOSE transA, 
    CBLAS_TRANSPOSE transB, int M, int N, int K, double alpha, double* A, int lda, 
    double* B, int ldb, double beta, double* C, int ldc);
#endif

template <typename T>
struct square
{
  T operator()(const T &val1, const T &val2)
  {
    return (val1 + val2 * val2);
  }
};

template <typename T>
struct abs_sum
{
  T operator()(const T &val1, const T &val2)
  {
    return (val1 + std::abs(val2));
  }
};


/* Diagonal general matrix multiplication (C = alpha * A * diag(x) + beta * C) */
/*
template <typename T>
void dgmm(int m, int n, double alpha, T* A, int lda, T* x, int incx, double beta, T* C, int ldc)
{
  for (unsigned int i = 0; i < m; i++)
  {
    for (unsigned int j = 0; j < n; j++)
    {
      C[ldc*i + j] = alpha * A[lda*i + j] * x[j + incx] + beta * C[ldc*i + j];
    }
  }
}
*/

/* Diagonal general matrix multiplication (C = alpha * A * diag(x) + beta * C) */
template <typename T>
void dgmm(int m, int n, double alpha, T* A, int lda, T* x, int incx, double beta, T* C, int ldc)
{
  for (unsigned int j = 0; j < n; j++)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      C[ldc*j + i] = alpha * A[lda*j + i] * x[j + incx] + beta * C[ldc*j + i];
    }
  }
}

/* General matrix multiplication (C = alpha * A * B + beta * C) */
template <typename T>
void gemm(int m, int n, int p, double alpha, T* A, int lda, T* B, int ldb, double beta, T* C, int ldc)
{
  for (unsigned int j = 0; j < n; j++)
  {
    for (unsigned int i = 0; i < m; i++)
    {
      double val = 0;
      for (unsigned int k = 0; k < p; k++)
      {
        val += A[lda*k + i] * B[ldb*j + k];
      }
      C[ldc*j + i] = alpha * val + beta * C[ldc*j + i];
    }
  }
}

template <typename T> 
unsigned int step(const T& val) 
{
    return T(0) <= val;
}

//std::ostream& operator<<(std::ostream &os, const point &pt);

double det_2x2(const double* mat);
double det_3x3(const double* mat);
double det_4x4(const double* mat);

mdvector<double> adjoint(const mdvector<double> &mat);

//! In-place matrix adjoint
void adjoint(const mdvector<double> &mat, mdvector<double> &adj);

void adjoint_3x3(double *mat, double *adj);
void adjoint_4x4(double *mat, double *adj);

double determinant(const mdvector<double> &mat);

mdvector<double> getRotationMatrix(double axis[3], double angle);
mdvector<double> getRotationMatrix(const Quat &q);

mdvector<double> identityMatrix(unsigned int N);

template<typename T>
int findFirst(const std::vector<T>& vec, T val)
{
  for (int i = 0; i < vec.size(); i++)
    if (vec[i] == val)
      return i;

  return -1;
}

std::vector<int> reverse_map(const std::vector<int> &map1);

std::vector<int> get_int_list(int N, int start = 0);
std::vector<uint> get_int_list(uint N, uint start = 0);

//! Map a structured ijk-type index to the equivalent Gmsh node index
std::vector<int> structured_to_gmsh_quad(unsigned int nNodes);
std::vector<int> structured_to_gmsh_hex(unsigned int nNodes);

//! Map a Gmsh node index to the equivalent structured ijk-type index
std::vector<int> gmsh_to_structured_quad(unsigned int nNodes);
std::vector<int> gmsh_to_structured_hex(unsigned int nNodes);

std::vector<int> gmsh_to_structured_tri(unsigned int nNodes);
std::vector<int> gmsh_to_structured_tet(unsigned int nNodes);

std::vector<int> gmsh_to_structured_pri(unsigned int nNodes);
std::vector<int> structured_to_gmsh_pri(unsigned int nNodes);

template<typename T>
std::vector<uint> sort_ind(const std::vector<T> &data, const std::vector<uint> &inds)
{
  std::vector<std::pair<T,size_t>> vp;
  vp.reserve(inds.size());

  for (uint i = 0; i < inds.size(); i++)
    vp.push_back(make_pair(data[inds[i]], inds[i]));

  /* Sorting will put lower values [vp.first] ahead of larger ones, resolving
   * ties using the original index [vp.second] */
  std::sort(vp.begin(), vp.end());

  std::vector<uint> sorted_ind(inds.size());
  for (uint i = 0; i < vp.size(); i++)
    sorted_ind[i] = vp[i].second;

  return sorted_ind;
}

template<typename T>
std::vector<uint> sort_ind(const std::vector<T> &data)
{
  std::vector<std::pair<T,size_t>> vp;
  vp.reserve(data.size());

  for (uint i = 0; i < data.size(); i++)
    vp.push_back(make_pair(data[i], i));

  /* Sorting will put lower values [vp.first] ahead of larger ones, resolving
   * ties using the original index [vp.second] */
  std::sort(vp.begin(), vp.end());

  std::vector<uint> sorted_ind(data.size());
  for (uint i = 0; i < vp.size(); i++)
    sorted_ind[i] = vp[i].second;

  return sorted_ind;
}

template<typename T>
void fuzzysort_ind(const mdvector<T> &mat, uint *inds, uint ninds, uint nDims, uint dim = 0, double tol = 1e-6)
{
  std::sort(inds, inds + ninds, [&](uint a, uint b) { return mat(dim, a) < mat(dim, b); } );

  uint j, i = 0;
  uint ix = inds[0];
  for (j = 1; j < ninds; j++)
  {
    uint jx = inds[j];
    if (mat(dim,jx) - mat(dim,ix) >= tol)
    {
      // Sort the duplicated by the next dimension and update 'ind'
      if (j - i > 1 && dim+1 < nDims)
        fuzzysort_ind(mat,inds+i,j-i,nDims,dim+1,tol);
      i = j;
      ix = jx;
    }
  }

  // Sort the duplicated by the next dimension and update 'ind'
  if (i != j && dim+1 < nDims)
    fuzzysort_ind(mat,inds+i,j-i,nDims,dim+1,tol);
}

template<typename T>
std::vector<uint> fuzzysort(const mdvector<T>& mat, uint dim = 0, double tol = 1e-8)
{
  auto dims = mat.shape();
  auto list = get_int_list(dims[1]);

  fuzzysort_ind(mat, list.data(), list.size(), dims[0], dim, tol);

  return list;
}

template<typename T>
void fuzzysort_ind_row(const mdvector<T> &mat, uint *inds, uint ninds, uint nDims, uint dim = 0, double tol = 1e-6)
{
  std::sort(inds, inds + ninds, [&](uint a, uint b) { return mat(a, dim) < mat(b, dim); } );

  uint j, i = 0;
  uint ix = inds[0];
  for (j = 1; j < ninds; j++)
  {
    uint jx = inds[j];
    if (mat(jx,dim) - mat(ix,dim) >= tol)
    {
      // Sort the duplicated by the next dimension and update 'ind'
      if (j - i > 1 && dim+1 < nDims)
        fuzzysort_ind_row(mat,inds+i,j-i,nDims,dim+1,tol);
      i = j;
      ix = jx;
    }
  }

  // Sort the duplicated by the next dimension and update 'ind'
  if (i != j && dim+1 < nDims)
    fuzzysort_ind_row(mat,inds+i,j-i,nDims,dim+1,tol);
}
template<typename T>
std::vector<uint> fuzzysort_row(const mdvector<T>& mat, uint dim = 0, double tol = 1e-8)
{
  auto dims = mat.shape();
  auto list = get_int_list(dims[0]);

  fuzzysort_ind_row(mat, list.data(), list.size(), dims[1], dim, tol);

  return list;
}

/* Helper function to write raw binary to file */
template<typename T>
void binary_write(std::ostream &f, T value)
{
  f.write((char *) &value, sizeof(T));
}

/* Helper function to read raw binary to from */
template<typename T>
void binary_read(std::istream &f, T &value)
{
  f.read((char *) &value, sizeof(T));
}

unsigned long hash_str(const char *str);

template <typename T>
void write_opp(mdvector<T> op, std::string str, unsigned long id, int m, int n)
{
//#ifdef _DUMP_OPS
  std::ofstream f(str + ".txt");

  f << id << std::endl;

  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      f << std::setprecision(16) << op(j + i*n) << " ";
    }
    f << std::endl;
  }
//#endif
}

#endif /* funcs_hpp */
