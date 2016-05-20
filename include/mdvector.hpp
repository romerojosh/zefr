#ifndef mdvector_hpp
#define mdvector_hpp

/*! mdvector.hpp 
 * \brief Template class for multidimensional vector implementation. i
 * Currently uses column-major formatting.
 *
 * \author Josh Romero, Stanford University
 *
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"

template<typename T>
class mdvector_gpu;
#endif

template <typename T>
class mdvector
{
  private:
    int ndims = 0;
    unsigned int nvals = 0;
    std::array<unsigned int,4> dims; 
    std::array<unsigned int,4> strides;
    std::vector<T> values;
#ifndef _NO_TNT
    std::shared_ptr<JAMA::LU<double>> LUptr;
#endif

  public:
    //! Constructors
    mdvector();
    mdvector(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0);

    //! Setup operator
    void assign(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0); // 

    //! Push back operator (for compatibility)
    void push_back(T value); 

    //! Fill operator
    void fill(T value);

    //! Size operator
    size_t size() const;

    //! Method to return vector shape
    std::array<unsigned int,4> shape(void) const;

    //! Method to return starting data pointer
    T* data();
    
    //! Method to return max element
    T max_val() const;
    
    //! Method to return min element
    T min_val() const;
    
    //! Method to calculate LU factors
    void calc_LU();
    
    //! Method to solve L U x = B for x
    void solve(mdvector<T>& x, const mdvector<T>& B) const;

    void add_dim_0(unsigned int ind, const T& val);
    void add_dim_1(unsigned int ind, const T& val);
    void add_dim_2(unsigned int ind, const T& val);
    void add_dim_3(unsigned int ind, const T& val);

    //! Method to return number of values (with padding)
    unsigned int get_nvals() const;

    //! Method to return pointer to strides (for GPU)
    const unsigned int* strides_ptr() const;

    //! Overloaded methods to access data
    T operator()(unsigned int idx0) const;
    T& operator()(unsigned int idx0);
    T operator()(unsigned int idx0, unsigned int idx1) const;
    T& operator()(unsigned int idx0, unsigned int idx1);
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    T operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2) const;
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);
    T operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3) const;

#ifdef _GPU
    //! Assignment (copy from GPU)
    mdvector<T>& operator= (mdvector_gpu<T> &vec);
#endif

};

template <typename T>
mdvector<T>::mdvector(){};

// KA: Padding not used as yet
// KA: Edge case - dims = {}
template <typename T>
mdvector<T>::mdvector(std::vector<unsigned int> dims, T value, unsigned int padding)
{
  ndims = (int)dims.size();
  
  //assert(ndims <= 4);
  
  nvals = 1;
  unsigned int i = 0;

  for (auto &d : dims)
  {
    nvals *= (d + padding);
    strides[i] = d + padding;
    this->dims[i] = d;
    i++;
  }

  values.assign(nvals, (T)value);
}

// KA: Padding not used as yet
// KA: Edge case - dims = {}
template <typename T>
void mdvector<T>::assign(std::vector<unsigned int> dims, T value, unsigned int padding)
{
  ndims = (int)dims.size();
  
  //assert(ndims <= 4);
  
  nvals = 1;
  unsigned int i = 0;

  for (auto &d : dims)
  {
    nvals *= (d + padding);
    strides[i] = d + padding;
    this->dims[i] = d;
    i++;
  }

  values.assign(nvals, (T)value);
}

template <typename T>
void mdvector<T>::fill(T value)
{
  std::fill(values.begin(), values.end(), value);
}

template <typename T>
size_t mdvector<T>::size() const
{
  return values.size();
}

// TODO: Must update dims and strides
template <typename T>
void mdvector<T>::push_back(T value)
{
  values.push_back(value);
  nvals++;
}

template <typename T>
std::array<unsigned int,4> mdvector<T>::shape(void) const
{
  return dims;
}

template <typename T>
T* mdvector<T>::data(void)
{
  return values.data();
}

template <typename T>
T mdvector<T>::max_val(void) const
{
  return *std::max_element(values.begin(), values.end());
}

template <typename T>
T mdvector<T>::min_val(void) const
{
  return *std::min_element(values.begin(), values.end());
}

template <typename T>
void mdvector<T>::calc_LU()
{
#ifndef _NO_TNT
  // Copy mdvector into TNT object
  unsigned int m = dims[0], n = dims[1];
  TNT::Array2D<double> A(m, n);
  for (unsigned int j = 0; j < n; j++)
    for (unsigned int i = 0; i < m; i++)
      A[i][j] = (*this)(i,j);
      
  // Calculate and store LU object
  LUptr = std::make_shared<JAMA::LU<double>>(A);
#endif
}

template <typename T>
void mdvector<T>::solve(mdvector<T>& x, const mdvector<T>& B) const
{
#ifndef _NO_TNT
  // Copy mdvector into TNT object
  std::array<unsigned int, 4> B_dims = B.shape();
  unsigned int n = B_dims[0], p = B_dims[1];
  TNT::Array2D<double> BArr(n, p);
  for (unsigned int j = 0; j < p; j++)
    for (unsigned int i = 0; i < n; i++)
      BArr[i][j] = B(i,j);
  
  // Solve for x
  TNT::Array2D<double> xArr = LUptr->solve(BArr);
      
  // Convert back to mdvector format
  for (unsigned int j = 0; j < p; j++)
    for (unsigned int i = 0; i < n; i++)
      x(i,j) = xArr[i][j];
#endif
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0) 
{
  //assert(ndims == 1);
  return values[idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0)const
{
  //assert(ndims == 1);
  return values[idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1) 
{
  //assert(ndims == 2);
  return values[idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1) const
{
  //assert(ndims == 2);
  return values[idx1 * strides[0] + idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  //assert(ndims == 3);
  return values[(idx2 * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) const
{
  //assert(ndims == 3);
  return values[(idx2 * strides[1] + idx1) * strides[0] + idx0];
}
template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  //assert(ndims == 4);
  return values[((idx3 * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) const
{
  //assert(ndims == 4);
  return values[((idx3 * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template<typename T>
void mdvector<T>::add_dim_0(unsigned int ind, const T& val)
{
  /* Insert new 'row' of memory */
  unsigned int stride0 = dims[0]*dims[1]*dims[2];
  unsigned int stride1 = dims[0]*dims[1];
  unsigned int stride2 = dims[0];
  unsigned int rowSize = 1;
  unsigned int offset = ind*rowSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    for (int j = dims[2]-1; j >= 0; j--) {
      for (int k = dims[1]-1; k >= 0; k--) {
        auto it = values.begin() + i*stride0 + j*stride1 + k*stride2 + offset;
        values.insert(it, rowSize, val);
      }
    }
  }

  dims[0]++;
  strides[0]++;
  nvals = values.size();
}

template<typename T>
void mdvector<T>::add_dim_1(unsigned int ind, const T &val)
{
  /* Insert new 'column' of memory */
  unsigned int stride0 = dims[0]*dims[1]*dims[2];
  unsigned int stride1 = dims[0]*dims[1];
  unsigned int colSize = dims[0];
  unsigned int offset = ind*colSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    for (int j = dims[2]-1; j >= 0; j--) {
      auto it = values.begin() + i*stride0 + j*stride1 + offset;
      values.insert(it, colSize, val);
    }
  }

  dims[1]++;
  strides[1]++;
  nvals = values.size();
}

template<typename T>
void mdvector<T>::add_dim_2(unsigned int ind, const T &val)
{
  /* Insert new 'page' of memory */
  unsigned int stride   = dims[0]*dims[1]*dims[2];
  unsigned int pageSize = dims[0]*dims[1];
  unsigned int offset = ind*pageSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    auto it = values.begin() + i*stride + offset;
    values.insert(it, pageSize, val);
  }

  dims[2]++;
  strides[2]++;
  nvals = values.size();
}

template<typename T>
void mdvector<T>::add_dim_3(unsigned int ind, const T &val)
{
  /* Insert new 'book' of memory */
  unsigned int bookSize = dims[0]*dims[1]*dims[2];
  auto it = values.begin() + bookSize*ind;
  values.insert(it, bookSize, val);
  dims[3]++;
  strides[3]++;
  nvals = values.size();
}

template <typename T>
unsigned int mdvector<T>::get_nvals() const
{
  return nvals;
}

template <typename T>
const unsigned int* mdvector<T>::strides_ptr() const
{
  return strides.data();
}

#ifdef _GPU
/* NOTE: Currently assumes GPU data to copy is same size! */
template <typename T>
mdvector<T>&  mdvector<T>::operator= (mdvector_gpu<T> &vec)
{
  //cudaMemcpy(values.data(), vec.data(), nvals*sizeof(T), cudaMemcpyDeviceToHost);
  copy_from_device(values.data(), vec.data(), nvals);

  return *this;
}
#endif


#endif /* mdvector_hpp */
