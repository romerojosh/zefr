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

#ifndef mdvector_gpu_hpp
#define mdvector_gpu_hpp

/*! mdvector_gpu.hpp 
 * \brief Template class for multidimensional vector implementation onGPU. 
 * Currently uses column-major formatting.
 *
 * \author Josh Romero, Stanford University
 *
 */

#include <vector>

#include "macros.hpp"
#include "mdvector.hpp"
#include "cuda_runtime.h"
#include "solver_kernels.h"

/* Texture (ldg) load template function */
template<typename T>
__device__ __forceinline__ T ldg(const T* ptr)
{
// TODO: Using ldg instruction slows down compute_F kernel quite a bit. Figure out why before re-enabling.
//#if __CUDA_ARCH__ >= 350
//  return __ldg(ptr);
//#else
  return *ptr;
//#endif
}

template<>
__device__ __forceinline__ double* ldg<double*>(double* const*  ptr)
{
  return *ptr;
}

template<typename T>
class mdvector;

template <typename T>
class mdvector_gpu
{
  private:
    int nDims = 0;
    size_t size_ = 0;
    size_t max_size_ = 0;
    unsigned int* strides;
    unsigned int strides_h[6];
    unsigned int dims_h[6];
    unsigned int ldim_;
    T* values = NULL;
    bool allocated = false;

  public:
    mdvector_gpu();
    ~mdvector_gpu();

    void free_data();

    void assign(std::vector<unsigned> dims, T* values, int stream = -1);

    //! Allocate memory & dimensions of vec w/o copying values
    void set_size(mdvector<T>& vec);

    void set_size(std::vector<unsigned> dims);

    //! Assignment (copy from host)
    mdvector_gpu<T>& operator= (mdvector<T>& vec);

    //! Method to return number of values (with padding)
    size_t max_size() const;

    size_t size() const;

    //! Method to return leading dimension
    unsigned int ldim() const;

    //! Method to return starting data pointer
    __host__ __device__
    T* data();

    //! Overloaded methods to access data
    __device__ __forceinline__
    T& operator()(unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx0) const;
    __device__ __forceinline__
    T& operator()(unsigned int idx1, unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx1, unsigned int idx0) const;
    __device__ __forceinline__
    T& operator()(unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx2, unsigned int idx1, unsigned int idx0) const;
    __device__ __forceinline__
    T& operator()(unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0) const;
    __device__ __forceinline__
    T& operator()(unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0) const;
    __device__ __forceinline__
    T& operator()(unsigned int idx5, unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __device__ __forceinline__
    T operator()(unsigned int idx5, unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0) const;

    __host__ __forceinline__
    T* get_ptr(unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx1, unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx5, unsigned int idx4, unsigned int idx3, unsigned int idx2, unsigned int idx1, unsigned int idx0);

    unsigned int get_dim(unsigned int dim) { return dims_h[dim]; };

    unsigned int get_stride(unsigned int dim) { return strides_h[dim]; };
};

template <typename T>
mdvector_gpu<T>::mdvector_gpu(){};

template <typename T>
mdvector_gpu<T>::~mdvector_gpu()
{
  /* NOTE: Cannot free here. Kernel will delete object copy! */
  /*
  if (allocated)
  {
    cudaFree(values);
    cudaFree(dims);
    cudaFree(strides);
  }
  */
}

template <typename T>
void mdvector_gpu<T>::free_data()
{
  if (allocated)
  {
    free_device_data(values);
    free_device_data(strides);

    allocated = false;
  }
}

template <typename T>
void mdvector_gpu<T>::assign(std::vector<unsigned> dims, T* vec, int stream)
{
  nDims = dims.size();
  size_ = 1;
  for (auto &dim : dims)
    size_ *= dim;

  if (allocated && max_size_ != size_)
    free_data();

  if(!allocated)
  {
    max_size_ = size_;
    allocate_device_data(values, max_size_);
    allocate_device_data(strides, 6);

    strides_h[0] = 1;

    for (int i = 0; i < nDims; i++)
    {
      strides_h[i] = 1;
      for (unsigned int j = nDims - i; j < nDims; j++)
        strides_h[i] *= dims[j];

      dims_h[i] = dims[i];
    }

    ldim_ = strides_h[nDims-1];

    allocated = true;
  }

  copy_to_device(strides, strides_h, 6, stream);
  copy_to_device(values, vec, size_, stream);
}

template <typename T>
void mdvector_gpu<T>::set_size(mdvector<T>& vec)
{
  if (allocated && max_size_ != vec.max_size())
    free_data();

  if(!allocated)
  {
    nDims = vec.get_nDims();
    size_ = vec.size();
    max_size_ = vec.max_size();
    ldim_ = vec.ldim();
    allocate_device_data(values, max_size_);
    allocate_device_data(strides, 6);

    copy_to_device(strides, vec.strides_ptr(), 6);

    std::copy(vec.strides_ptr(), vec.strides_ptr()+6, strides_h);
    std::copy(vec.dims_ptr(), vec.dims_ptr()+nDims, dims_h);

    allocated = true;
  }
}

template <typename T>
void mdvector_gpu<T>::set_size(std::vector<unsigned int> dims)
{
  nDims = (int)dims.size();

  size_ = 1;
  unsigned int new_size = 1;

  strides_h[0] = 1;

  for (int i = 0; i < nDims; i++)
  {
    strides_h[i] = 1;
    for (unsigned int j = nDims - i; j < nDims; j++)
      strides_h[i] *= dims[j];

    size_ *= dims[i];
    new_size *= dims[i];

    dims_h[i] = dims[i];
  }

  if (allocated && max_size_ != new_size)
    free_data();

  if(!allocated)
  {
    size_ = new_size;
    max_size_ = new_size;
    ldim_ = strides_h[nDims-1];
    allocate_device_data(values, max_size_);
    allocate_device_data(strides, 6);

    copy_to_device(strides, strides_h, 6);

    allocated = true;
  }
}

template <typename T>
mdvector_gpu<T>& mdvector_gpu<T>::operator= (mdvector<T>& vec)
{
  if(!allocated)
  {
    size_ = vec.size();
    max_size_ = vec.max_size();
    ldim_ = vec.ldim();
    allocate_device_data(values, max_size_);
    allocate_device_data(strides, 6);

    copy_to_device(strides, vec.strides_ptr(), 6);

    std::copy(vec.dims_ptr(), vec.dims_ptr() + 6, dims_h);
    std::copy(vec.strides_ptr(), vec.strides_ptr() + 6, strides_h);

    allocated = true;
  }

  /* Copy values to GPU */
  copy_to_device(values, vec.data(), max_size_);

  return *this;
}


template <typename T>
size_t mdvector_gpu<T>::max_size() const
{
  return max_size_; 
}

template <typename T>
size_t mdvector_gpu<T>::size() const
{
  return size_;
}

template <typename T>
unsigned int mdvector_gpu<T>::ldim() const
{
  return ldim_;
}

template <typename T>
__host__ __device__
T* mdvector_gpu<T>::data(void)
{
  return values;
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0) 
{
  return values[idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx0) const
{
  return ldg(values + idx0);
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx1, unsigned int idx0) 
{
  return values[idx1 * strides[1] + idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx1, unsigned int idx0) const
{
  return ldg(values + idx1 * strides[1] + idx0);
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx2, unsigned int idx1, unsigned int idx0) 
{
  return values[idx2 * strides[2] + idx1 * strides[1] + idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx2, unsigned int idx1, unsigned int idx0) const
{
  return ldg(values + idx2 * strides[2] + idx1 * strides[1] + idx0);
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx3, unsigned int idx2, unsigned int idx1, 
    unsigned int idx0) 
{
  return values[idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx3, unsigned int idx2, unsigned int idx1, 
    unsigned int idx0) const
{
  return ldg(values + idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0);
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx4, unsigned int idx3, unsigned int idx2, 
    unsigned int idx1, unsigned int idx0) 
{
  return values[idx4 * strides[4] + idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx4, unsigned int idx3, unsigned int idx2, 
    unsigned int idx1, unsigned int idx0) const
{
  return ldg(values + idx4 * strides[4] + idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0);
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx5, unsigned int idx4, unsigned int idx3, 
    unsigned int idx2, unsigned int idx1, unsigned int idx0) 
{
  return values[idx5 * strides[5] + idx4 * strides[4] + idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0];
}

template <typename T>
__device__ __forceinline__
T mdvector_gpu<T>::operator() (unsigned int idx5, unsigned int idx4, unsigned int idx3, 
    unsigned int idx2, unsigned int idx1, unsigned int idx0) const
{
  return ldg(values + idx5 * strides[5] + idx4 * strides[4] + idx3 * strides[3] + idx2 * strides[2] + idx1 * strides[1] + idx0);
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0) 
{
  return values + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx1, unsigned int idx0) 
{
  return values + idx1 * strides_h[1] + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx2, unsigned int idx1, unsigned int idx0) 
{
  return values + idx2 * strides_h[2] + idx1 * strides_h[1] + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx3, unsigned int idx2, unsigned int idx1, 
    unsigned int idx0) 
{
  return values + idx3 * strides_h[3] + idx2 * strides_h[2] + idx1 * strides_h[1] + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx4, unsigned int idx3, unsigned int idx2, 
    unsigned int idx1, unsigned int idx0) 
{
  return values + idx4 * strides_h[4] + idx3 * strides_h[3] + idx2 * strides_h[2] + idx1 * strides_h[1] + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx5, unsigned int idx4, unsigned int idx3, 
    unsigned int idx2, unsigned int idx1, unsigned int idx0) 
{
  return values + idx5 * strides_h[5] + idx4 * strides_h[4] + idx3 * strides_h[3] + idx2 * strides_h[2] + idx1 * strides_h[1] + idx0;
}

/* mdview_gpu class to allow indirect data access from faces */
// NOTE: Assumes index in last place used for slot always, and thus
// applies to strided access into base_ptrs/strides vectors. 
template <typename T>
class mdview_gpu
{
  private:
    mdvector_gpu<T*> base_ptrs;
    mdvector_gpu<unsigned int> strides;
    unsigned int* base_stride;

  public:
    mdview_gpu(){}

    mdview_gpu(mdvector<T*> &base_ptrs_h, mdvector<unsigned int>& strides_h, unsigned int base_stride_h)
    {
      base_ptrs = base_ptrs_h;
      strides = strides_h;

      allocate_device_data(base_stride, 1);
      copy_to_device(base_stride, &base_stride_h, 1);
    }

    void assign(mdvector<T*> &base_ptrs_h, mdvector<unsigned int>& strides_h, unsigned int base_stride_h)
    {
      base_ptrs = base_ptrs_h;
      strides = strides_h;

      allocate_device_data(base_stride, 1);
      copy_to_device(base_stride, &base_stride_h, 1);
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx0) 
    {
      return *(base_ptrs(idx0));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx0) const
    {
      return ldg(base_ptrs(idx0));
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx1, unsigned int idx0) 
    {
      return *(base_ptrs(idx0 + *(base_stride) * idx1));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx1, unsigned int idx0) const
    {
      return ldg(base_ptrs(idx0 + *(base_stride) * idx1));
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx2, unsigned int idx1, unsigned int idx0) 
    {
      return *(base_ptrs(idx0 + *(base_stride) * idx2) + strides(idx0 + *(base_stride) * idx2) * idx1);
    }

    __device__ __forceinline__
    T operator() (unsigned int idx2, unsigned int idx1, unsigned int idx0) const
    {
      return ldg(base_ptrs(idx0 + *(base_stride) * idx2) + strides(idx0 + *(base_stride) * idx2) * idx1);
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx3, unsigned int idx2, unsigned int idx1, 
        unsigned int idx0) 
    {
      return *(base_ptrs(idx0 + *(base_stride) * idx3) + strides(0, idx0 + *(base_stride) * idx3) * 
          idx1 + strides(1, idx0 + *(base_stride) * idx3) * idx2);
    }

    __device__ __forceinline__
    T operator() (unsigned int idx3, unsigned int idx2, unsigned int idx1, 
        unsigned int idx0) const
    {
      return ldg(base_ptrs(idx0 + *(base_stride) * idx3) + strides(0, idx0 + *(base_stride) * idx3) * 
          idx1 + strides(1, idx0 + *(base_stride) * idx3) * idx2);
    }

};

#endif /* mdvector_gpu_hpp */
