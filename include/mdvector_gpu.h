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
    unsigned int* strides_h;
    unsigned int ldim_;
    T* values = NULL;
    bool allocated = false;

  public:
    mdvector_gpu();
    ~mdvector_gpu();

    void free_data();

    //! Allocated memory & dimensions of vec w/o copying values
    void set_size(mdvector<T>& vec);

    //! Assignment (copy from host)
    mdvector_gpu<T>& operator= (mdvector<T>& vec);

    //! Method to return number of values (with padding)
    size_t max_size() const;

    size_t size() const;

    //! Method to return leading dimension
    unsigned int ldim() const;

    //! Method to return starting data pointer
    T* data();

    //! Overloaded methods to access data
    __device__ __forceinline__
    T& operator()(unsigned int idx0);
    __device__ __forceinline__
    T& operator()(unsigned int idx0, unsigned int idx1);
    __device__ __forceinline__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    __device__ __forceinline__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);
    __device__ __forceinline__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4);
    __device__ __forceinline__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4, unsigned int idx5);

    __host__ __forceinline__
    T* get_ptr(unsigned int idx0);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx0, unsigned int idx1);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4);
    __host__ __forceinline__
    T* get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3, unsigned int idx4, unsigned int idx5);

    unsigned int get_dim(unsigned int dim) { return strides_h[dim]; };

    __device__ __forceinline__
    unsigned int get_stride(unsigned int dim) { return strides[dim]; };
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
    delete[] strides_h;

    allocated = false;
  }
}

template <typename T>
void mdvector_gpu<T>::set_size(mdvector<T>& vec)
{
  if (allocated && max_size_ != vec.max_size())
    free_data();

  if(!allocated)
  {
    size_ = vec.size();
    max_size_ = vec.max_size();
    ldim_ = vec.ldim();
    allocate_device_data(values, max_size_);
    allocate_device_data(strides, 6);
    strides_h = new unsigned int[6];

    copy_to_device(strides, vec.strides_ptr(), 6);
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
    strides_h = new unsigned int[6];

    copy_to_device(strides, vec.strides_ptr(), 6);

    std::copy(vec.strides_ptr(), vec.strides_ptr()+6, strides_h);

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
T* mdvector_gpu<T>::data(void)
{
  return values;
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0) 
{
  //assert(ndims == 1);
  return values[idx0];
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1) 
{
  //assert(ndims == 2);
  return values[idx1 * strides[0] + idx0];
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  //assert(ndims == 3);
  return values[(idx2 * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  //assert(ndims == 4);
  return values[((idx3 * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) 
{
  //assert(ndims == 5);
  return values[(((idx4 * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
__device__ __forceinline__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) 
{
  //assert(ndims == 6);
  return values[((((idx5 * strides[4] + idx4) * strides[3] + idx3) * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0) 
{
  return values + idx0;
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0, unsigned int idx1) 
{
  return values + (idx1 * strides_h[0] + idx0);
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  return values + ((idx2 * strides_h[1] + idx1) * strides_h[0] + idx0);
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  return values + (((idx3 * strides_h[2] + idx2) * strides_h[1] + idx1) * strides_h[0] + idx0);
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4) 
{
  return values + ((((idx4 * strides_h[3] + idx3) * strides_h[2] + idx2) * strides_h[1] + idx1) * strides_h[0] + idx0);
}

template <typename T>
__host__ __forceinline__
T* mdvector_gpu<T>::get_ptr(unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3, unsigned int idx4, unsigned int idx5) 
{
  return values + (((((idx5 * strides_h[4] + idx4) * strides_h[3] + idx3) * strides_h[2] + idx2) * strides_h[1] + idx1) * strides_h[0] + idx0);
}

template <typename T>
class mdview_gpu
{
  private:
    mdvector_gpu<T*> ptr_map;

  public:
    mdview_gpu(){}

    mdview_gpu(mdvector<T*> &ptr_map_h)
    {
      ptr_map = ptr_map_h;
    }

    void assign(mdvector<T*> &ptr_map_h)
    {
      ptr_map = ptr_map_h;
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx0) 
    {
      return *(ptr_map(idx0));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx0) const
    {
      return *(ptr_map(idx0));
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx0, unsigned int idx1) 
    {
      return *(ptr_map(idx0, idx1));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx0, unsigned int idx1) const
    {
      return *(ptr_map(idx0, idx1));
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
    {
      return *(ptr_map(idx0, idx1, idx2));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) const
    {
      return *(ptr_map(idx0, idx1, idx2));
    }

    __device__ __forceinline__
    T& operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
        unsigned int idx3) 
    {
      return *(ptr_map(idx0, idx1, idx2, idx3));
    }

    __device__ __forceinline__
    T operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
        unsigned int idx3) const
    {
      return *(ptr_map(idx0, idx1, idx2, idx3));
    }

};

#endif /* mdvector_gpu_hpp */
