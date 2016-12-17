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
    mdvector(const mdvector<T> &orig);
    mdvector<T>& operator= (const mdvector<T> &orig);
    mdvector(std::vector<unsigned int> dims, T value = T(), bool pinned = false);

    //! Setup operator
    void assign(std::vector<unsigned int> dims, T value = T(), bool pinned = false);

    //! Setup operator (set size if needed, w/o initializing data)
    void resize(std::vector<unsigned int> dims);

    //! Push back operator (for compatibility)
    void push_back(T value); 

    //! Fill operator
    void fill(T value);

    //! Size operator
    size_t size() const;
    //
    //! Method to return number of values
    size_t max_size() const;

    int get_nDims(void) { return nDims; }

    //! Method to return vector shape
    std::array<unsigned int,6> shape(void) const;

    //! Method to return vector strides
    unsigned int get_dim(unsigned int dim) const;

    //! Method to return vector strides
    unsigned int get_stride(unsigned int dim) const;

    //! Method to get leading dimension
    unsigned int ldim() const; 

    //! Method to return starting data pointer
    T* data();

    const T* data() const;
    
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

    //! Method to return pointer to dim and strides (for GPU)
    const unsigned int* dims_ptr() const;
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
mdvector<T>::mdvector(std::vector<unsigned int> dims, T value, bool pinned)
{
  this->assign(dims, value, pinned);
}

template <typename T>
mdvector<T>::mdvector(const mdvector<T> &orig)
{
  this->nDims = orig.nDims;
  this->size_ = orig.size_;
  this->max_size_ = orig.max_size_;
  this->dims = orig.dims;
  this->strides = orig.strides;
  this->values = orig.values;

  this->pinned = orig.pinned;

  if (!this->pinned)
    this->values_ptr = this->values.data();
  else
    ThrowException("Unsupported copy of pinned mdvector detected! Don't do this!");
}

template <typename T>
mdvector<T>& mdvector<T>::operator= (const mdvector<T> &orig)
{
  this->nDims = orig.nDims;
  this->size_ = orig.size_;
  this->max_size_ = orig.max_size_;
  this->dims = orig.dims;
  this->strides = orig.strides;
  this->values = orig.values;

  this->pinned = orig.pinned;

  if (!this->pinned)
    this->values_ptr = this->values.data();
  else
    ThrowException("Unsupported copy of pinned mdvector detected! Don't do this!");

  return *this;
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
void mdvector<T>::assign(std::vector<unsigned int> dims, T value, bool pinned)
{
  nDims = (int)dims.size();

  size_ = 1;
  max_size_ = 1;


  for (unsigned int i = 0; i < nDims; i++)
  {
    if (i > 0)
    {
      strides[i-1] = 1;
      for (unsigned int j = 0; j < i; j++)
        strides[i-1] *= dims[j];
    }

    size_ *= dims[i];
    max_size_ *= dims[i];

    this->dims[i] = dims[i];
  }

#ifdef _GPU
  this->pinned = pinned;
#endif
  if (!this->pinned)
  {
    values.assign(max_size_, (T)value);
    values_ptr = values.data();
  }
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
void mdvector<T>::resize(std::vector<unsigned int> dims)
{
  nDims = (int)dims.size();

  size_ = 1;
  max_size_ = 1;

  for (unsigned int i = 0; i < nDims; i++)
  {
    strides[i] = dims[i];

    size_ *= dims[i];
    max_size_ *= strides[i];

    this->dims[i] = dims[i];
  }

  if (this->pinned)
  {
    ThrowException("Should not be calling mat.resize() on pinned memory. Something's wrong!");
  }
  else if (max_size_ != values.size())
  {
    values.resize(max_size_);
    values_ptr = values.data();
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
    values_ptr = values.data();
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
unsigned int mdvector<T>::get_dim(unsigned int dim) const
{
  return dims[dim];
}

template <typename T>
unsigned int mdvector<T>::get_stride(unsigned int dim) const
{
  return strides[dim];
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
const T* mdvector<T>::data(void) const
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
  return values_ptr[idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0) const
{
  //assert(ndims == 1);
  return values_ptr[idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1) 
{
  //assert(ndims == 2);
  return values_ptr[idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1) const
{
  //assert(ndims == 2);
  return values_ptr[idx1 * strides[0] + idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  //assert(ndims == 3);
  return values_ptr[idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) const
{
  //assert(ndims == 3);
  return values_ptr[idx2 * strides[1] + idx1 * strides[0] + idx0];
}
template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  //assert(ndims == 4);
  return values_ptr[idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) const
{
  //assert(ndims == 4);
  return values_ptr[idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) 
{
  //assert(ndims == 5);
  return values[idx4 * strides[3] + idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) const
{
  //assert(ndims == 5);
  return values[idx4 * strides[3] + idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) 
{
  //assert(ndims == 6);
  return values[idx5 * strides[4] + idx4 * strides[3] + idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template <typename T>
T mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) const
{
  //assert(ndims == 6);
  return values[idx5 * strides[4] + idx4 * strides[3] + idx3 * strides[2] + idx2 * strides[1] + idx1 * strides[0] + idx0];
}

template<typename T>
void mdvector<T>::add_dim_0(int ind, const T& val)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::add_dim_1(int ind, const T &val)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::add_dim_2(int ind, const T &val)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::add_dim_3(int ind, const T &val)
{
  if (!pinned)
  {
    if (ind == -1) ind = dims[3]; // Add to end by default

    /* Insert new 'book' of memory */
    unsigned int bookSize = dims[0]*dims[1]*dims[2];
    auto it = values.begin() + bookSize*ind;
    values.insert(it, bookSize, val);
    dims[3]++;
    strides[3]++;
    size_ = values.size();
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::remove_dim_0(unsigned int ind)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::remove_dim_1(unsigned int ind)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::remove_dim_2(unsigned int ind)
{
  if (!pinned)
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
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template<typename T>
void mdvector<T>::remove_dim_3(unsigned int ind)
{
  if (!pinned)
  {
    /* Remove 'book' of memory */
    unsigned int bookSize = dims[0]*dims[1]*dims[2];
    auto it = values.begin() + bookSize*ind;
    values.erase(it, it+bookSize);
    dims[3]--;
    strides[3]--;
    size_ = values.size();
    values_ptr = values.data();
  }
  else
  {
    ThrowException("Cannot modify dimensions of pinned mdvector!");
  }
}

template <typename T>
const unsigned int* mdvector<T>::dims_ptr() const
{
  return dims.data();
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

/* mdview class to allow indirect data access from faces */
// NOTE: Assumes index in last place used for slot always, and thus
// applies to strided access into base_ptrs/strides vectors. 
template <typename T>
class mdview
{
  private:
    mdvector<T*> base_ptrs;
    mdvector<unsigned int> strides;
    unsigned int base_stride;

  public:
    mdview(){}

    mdview(mdvector<T*> &base_ptrs, mdvector<unsigned int>& strides, unsigned int base_stride)
    {
      this->base_ptrs = base_ptrs;
      this->strides = strides;
      this->base_stride = base_stride;
    }

    void assign(mdvector<T*> &base_ptrs, mdvector<unsigned int>& strides, unsigned int base_stride)
    {
      this->base_ptrs = base_ptrs;
      this->strides = strides;
      this->base_stride = base_stride;
    }

    T& operator() (unsigned int idx0) 
    {
      return *(base_ptrs(idx0));
    }

    T operator() (unsigned int idx0) const
    {
      return *(base_ptrs(idx0));
    }

    T& operator() (unsigned int idx0, unsigned int idx1) 
    {
      return *(base_ptrs(idx0 + base_stride * idx1));
    }

    T operator() (unsigned int idx0, unsigned int idx1) const
    {
      return *(base_ptrs(idx0 + base_stride * idx1));
    }

    T& operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
    {
      return *(base_ptrs(idx0 + base_stride * idx2) + strides(idx0 + base_stride * idx2, 0) * idx1);
    }

    T operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) const
    {
      return *(base_ptrs(idx0 + base_stride * idx2) + strides(idx0 + base_stride * idx2, 0) * idx1);
    }

    T& operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
        unsigned int idx3) 
    {
      return *(base_ptrs(idx0 + base_stride * idx3) + strides(idx0 + base_stride * idx3, 0) * 
          idx1 + strides(idx0 + base_stride * idx3, 1) * idx2);
    }

    T operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
        unsigned int idx3) const
    {
      return *(base_ptrs(idx0 + base_stride * idx3) + strides(idx0 + base_stride * idx3, 0) * 
          idx1 + strides(idx0 + base_stride * idx3, 1) * idx2);
    }

};


#endif /* mdvector_hpp */
