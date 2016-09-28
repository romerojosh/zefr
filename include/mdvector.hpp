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

#ifndef mdvector_hpp
#define mdvector_hpp

/*! mdvector.hpp 
 * \brief Template class for multidimensional vector implementation. 
 * Currently uses column-major formatting.
 *
 * \author Josh Romero, Stanford University
 *
 */

#ifdef _CPU
static const unsigned int CACHE_LINE_SIZE = 64;
#endif
#ifdef _GPU
static const unsigned int CACHE_LINE_SIZE = 128;
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

#include "macros.hpp"

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "cuda_runtime.h"

template<typename T>
class mdvector_gpu;
#endif

template <typename T>
class mdvector
{
  private:
    int nDims = 0;
    size_t size_ = 0;  // Size of true vector
    size_t max_size_ = 0;  // Size of allocation
    std::array<unsigned int,6> dims; 
    std::array<unsigned int,6> strides;
    std::vector<T> values;
    T* values_ptr; // For pinned memory only!

#ifndef _NO_TNT
    std::shared_ptr<JAMA::LU<double>> LUptr;
#endif

    bool pinned = false;

  public:
    //! Constructors
    mdvector();
    ~mdvector();
    mdvector(std::vector<unsigned int> dims, T value = 0, bool pad = false, bool pinned = false);

    //! Setup operator
    void assign(std::vector<unsigned int> dims, T value = 0, bool pad = false, bool pinned = false);

    //! Push back operator (for compatibility)
    void push_back(T value); 

    //! Fill operator
    void fill(T value);

    //! Size operator
    size_t size() const;
    //
    //! Method to return number of values (with padding)
    size_t max_size() const;

    //! Method to return vector shape
    std::array<unsigned int,6> shape(void) const;

    //! Method to return vector strides
    std::array<unsigned int,6> get_strides(void) const;

    //! Method to get leading dimension
    unsigned int ldim() const; 

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

    //! Methods to add to dimensions of an mdvector
    void add_dim_0(int ind, const T& val);
    void add_dim_1(int ind, const T& val);
    void add_dim_2(int ind, const T& val);
    void add_dim_3(int ind, const T& val);

    //! Methods to remove from dimensions of an mdvector
    void remove_dim_0(unsigned int ind);
    void remove_dim_1(unsigned int ind);
    void remove_dim_2(unsigned int ind);
    void remove_dim_3(unsigned int ind);

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
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4);
    T operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4) const;
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4, unsigned int idx5);
    T operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4, unsigned int idx5) const;

#ifdef _GPU
    //! Assignment (copy from GPU)
    mdvector<T>& operator= (mdvector_gpu<T> &vec);
#endif

};

template <typename T>
mdvector<T>::mdvector(){};

template <typename T>
mdvector<T>::mdvector(std::vector<unsigned int> dims, T value, bool pad, bool pinned)
{
  this->assign(dims, value, pad, pinned);
}

template <typename T>
mdvector<T>::~mdvector()
{
#ifdef _GPU
  if (pinned)
  {
    cudaFreeHost(values_ptr);
  }
#endif
}

template <typename T>
void mdvector<T>::assign(std::vector<unsigned int> dims, T value, bool pad, bool pinned)
{
  nDims = (int)dims.size();

  size_ = 1;
  max_size_ = 1;


  for (unsigned int i = 0; i < nDims; i++)
  {
    strides[i] = dims[i];

    /* Pad leading dimension to cache line boundary */
    if (i == 0 and nDims > 1 and pad)
    {
      strides[i] += dims[i] % (CACHE_LINE_SIZE / sizeof(T));
    }

    size_ *= dims[i];
    max_size_ *= strides[i];

    this->dims[i] = dims[i];
  }

#ifdef _GPU
  this->pinned = pinned;
#endif
  if (!this->pinned)
    values.assign(max_size_, (T)value);
  else
  {
#ifdef _GPU
    auto status = cudaMallocHost(&values_ptr, max_size_ * sizeof(T));
    if (status != cudaSuccess)
      ThrowException("cudaMemcpy from device to host failed!");
#endif

#ifdef _CPU
    ThrowException("CPU code should not be able to allocate pinned memory. Something's wrong!");
#endif
  }
}

template <typename T>
void mdvector<T>::fill(T value)
{
  if (!pinned)
    std::fill(values.begin(), values.end(), value);
  else
    std::fill(values_ptr, values_ptr + max_size_, value);
}

template <typename T>
size_t mdvector<T>::size() const
{
  return size_;
}

template <typename T>
size_t mdvector<T>::max_size() const
{
  return max_size_;
}

template <typename T>
void mdvector<T>::push_back(T value) /* Only valid for 1D arrays! */
{
  if (!pinned)
  {
    values.push_back(value);
    size_++;
    max_size_++;
  }
  else
  {
    ThrowException("Cannot push back to pinned mdvector!");
  }
}

template <typename T>
std::array<unsigned int,6> mdvector<T>::shape(void) const
{
  return dims;
}

template <typename T>
std::array<unsigned int,6> mdvector<T>::get_strides(void) const
{
  return strides;
}

template <typename T>
unsigned int mdvector<T>::ldim() const
{
  return strides[0];
}

template <typename T>
T* mdvector<T>::data(void)
{
  if (!pinned)
    return values.data();
  else
    return values_ptr;
}

template <typename T>
T mdvector<T>::max_val(void) const
{
  if (!pinned)
    return *std::max_element(values.begin(), values.end());
  else
    return *std::max_element(values_ptr, values_ptr + max_size_);
}

template <typename T>
T mdvector<T>::min_val(void) const
{
  if (!pinned)
    return *std::min_element(values.begin(), values.end());
  else
    return *std::min_element(values_ptr, values_ptr + max_size_);
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
  auto B_dims = B.shape();
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
void mdvector<T>::add_dim_0(int ind, const T& val)
{
  if (ind == -1) ind = dims[0]; // Add to end by default

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
  size_ = values.size();
}

template<typename T>
void mdvector<T>::add_dim_1(int ind, const T &val)
{
  if (ind == -1) ind = dims[1]; // Add to end by default

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
  size_ = values.size();
}

template<typename T>
void mdvector<T>::add_dim_2(int ind, const T &val)
{
  if (ind == -1) ind = dims[2]; // Add to end by default

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
  size_ = values.size();
}

template<typename T>
void mdvector<T>::add_dim_3(int ind, const T &val)
{
  if (ind == -1) ind = dims[3]; // Add to end by default

  /* Insert new 'book' of memory */
  unsigned int bookSize = dims[0]*dims[1]*dims[2];
  auto it = values.begin() + bookSize*ind;
  values.insert(it, bookSize, val);
  dims[3]++;
  strides[3]++;
  size_ = values.size();
}

template<typename T>
void mdvector<T>::remove_dim_0(unsigned int ind)
{
  /* Remove 'row' of memory */
  unsigned int stride0 = dims[0]*dims[1]*dims[2];
  unsigned int stride1 = dims[0]*dims[1];
  unsigned int stride2 = dims[0];
  unsigned int rowSize = 1;
  unsigned int offset = ind*rowSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    for (int j = dims[2]-1; j >= 0; j--) {
      for (int k = dims[1]-1; k >= 0; k--) {
        auto it = values.begin() + i*stride0 + j*stride1 + k*stride2 + offset;
        values.erase(it, it+rowSize);
      }
    }
  }

  dims[0]--;
  strides[0]--;
  size_ = values.size();
}

template<typename T>
void mdvector<T>::remove_dim_1(unsigned int ind)
{
  /* Remove 'column' of memory */
  unsigned int stride0 = dims[0]*dims[1]*dims[2];
  unsigned int stride1 = dims[0]*dims[1];
  unsigned int colSize = dims[0];
  unsigned int offset = ind*colSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    for (int j = dims[2]-1; j >= 0; j--) {
      auto it = values.begin() + i*stride0 + j*stride1 + offset;
      values.erase(it, it+colSize);
    }
  }

  dims[1]--;
  strides[1]--;
  size_ = values.size();
}

template<typename T>
void mdvector<T>::remove_dim_2(unsigned int ind)
{
  /* Remove 'page' of memory */
  unsigned int stride   = dims[0]*dims[1]*dims[2];
  unsigned int pageSize = dims[0]*dims[1];
  unsigned int offset = ind*pageSize;
  for (int i = dims[3]-1; i >= 0; i--) {
    auto it = values.begin() + i*stride + offset;
    values.erase(it, it+pageSize);
  }

  dims[2]--;
  strides[2]--;
  size_ = values.size();
}

template<typename T>
void mdvector<T>::remove_dim_3(unsigned int ind)
{
  /* Remove 'book' of memory */
  unsigned int bookSize = dims[0]*dims[1]*dims[2];
  auto it = values.begin() + bookSize*ind;
  values.erase(it, it+bookSize);
  dims[3]--;
  strides[3]--;
  size_ = values.size();
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) 
{
  //assert(ndims == 5);
  return values[(((idx4 * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) const
{
  //assert(ndims == 5);
  return values[(((idx4 * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) 
{
  //assert(ndims == 6);
  return values[((((idx5 * strides[4] + idx4) * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) const
{
  //assert(ndims == 6);
  return values[((((idx5 * strides[4] + idx4) * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
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
  if (!pinned)
  {
    copy_from_device(values.data(), vec.data(), this->max_size());
    //auto status = cudaMemcpy(values.data(), vec.data(), max_size_ * sizeof(T), cudaMemcpyDeviceToHost);
    //if (status != cudaSuccess)
    //  ThrowException("cudaMemcpy from device to host failed!");
  }
  else
  {
    copy_from_device(values_ptr, vec.data(), this->max_size());
    //auto status = cudaMemcpy(values_ptr, vec.data(), max_size_ * sizeof(T), cudaMemcpyDeviceToHost);
    //if (status != cudaSuccess)
    //  ThrowException("cudaMemcpy from device to host failed!");
  }

  return *this;
}
#endif

#endif /* mdvector_hpp */
