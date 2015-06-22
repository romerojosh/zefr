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

#include "mdvector.hpp"
#include "cuda_runtime.h"
#include "solver_kernels.h"

template<typename T>
class mdvector;

template <typename T>
class mdvector_gpu
{
  private:
    int ndims = 0;
    int nvals;
    unsigned int* dims; 
    unsigned int* strides;
    T* values;
    bool allocated = false;

  public:
    mdvector_gpu();
    ~mdvector_gpu();

    void free_data();

    //! Assignment (copy from host)
    mdvector_gpu<T>& operator= (mdvector<T>& vec);

    //! Method to return starting data pointer
    T* data();

    //! Overloaded methods to access data
    __device__
    T& operator()(unsigned int idx0, unsigned int idx1);
    __device__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    __device__
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);

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
    /*
    cudaFree(values);
    cudaFree(dims);
    cudaFree(strides);
    */
    free_device_data(values);
    free_device_data(dims);
    free_device_data(strides);
  }
}

template <typename T>
mdvector_gpu<T>& mdvector_gpu<T>::operator= (mdvector<T>& vec)
{
  if(!allocated)
  {
    nvals = vec.get_nvals();
    /*
    cudaMalloc(&values, nvals*sizeof(T));
    cudaMalloc(&strides, 4*sizeof(unsigned int));
    cudaMalloc(&dims, 4*sizeof(unsigned int));
    */
    allocate_device_data(values, nvals*sizeof(T));
    allocate_device_data(strides, 4*sizeof(T));
    allocate_device_data(dims, 4*sizeof(T));
    allocated = true;
  }

  /* Copy values to GPU */
  //cudaMemcpy(values, vec.data(), nvals*sizeof(T), cudaMemcpyHostToDevice);
  copy_to_device(values, vec.data(), nvals*sizeof(T));

  /* Copy strides to GPU (always size 4 for now!) */
  //cudaMemcpy(strides, vec.strides_ptr(), 4*sizeof(unsigned int), cudaMemcpyHostToDevice);
  copy_to_device(strides, vec.strides_ptr(), 4*sizeof(unsigned int));

  return *this;
}

template <typename T>
T* mdvector_gpu<T>::data(void)
{
  return values;
}

template <typename T>
__device__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1) 
{
  //assert(ndims == 2);
  return values[idx1 * strides[0] + idx0];
}

template <typename T>
__device__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  //assert(ndims == 3);
  return values[(idx2 * strides[1] + idx1) * strides[0] + idx0];
}

template <typename T>
__device__
T& mdvector_gpu<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  //assert(ndims == 4);
  return values[((idx3 * strides[2] + idx2) * strides[1] + idx1) * strides[0] + idx0];
}

#endif /* mdvector_gpu_hpp */
