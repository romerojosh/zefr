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
    unsigned int nvals;
    std::array<unsigned int,4> dims; 
    std::array<unsigned int,4> strides;
    std::vector<T> values;

  public:
    //! Constructors
    mdvector();
    mdvector(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0);

    //! Setup operator
    void assign(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0);

    //! Push back operator (for compatibility)
    void push_back(T value);

    //! Fill operator
    void fill(T value);

    //! Method to return vector shape
    std::array<unsigned int,4> shape(void) const;

    //! Method to return starting data pointer
    T* data();

    //! Method to return number of values (with padding)
    unsigned int get_nvals() const;

    //! Method to return pointer to strides (for GPU)
    const unsigned int* strides_ptr() const;

    //! Overloaded methods to access data
    T operator()(unsigned int idx0) const;
    T& operator()(unsigned int idx0);
    T& operator()(unsigned int idx0, unsigned int idx1);
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);

#ifdef _GPU
    //! Assignment (copy from GPU)
    mdvector<T>& operator= (mdvector_gpu<T> &vec);
#endif

};

template <typename T>
mdvector<T>::mdvector(){};

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
void mdvector<T>::push_back(T value)
{
  values.push_back(value);
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
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
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
