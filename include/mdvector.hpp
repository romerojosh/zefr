#ifndef mdvector_hpp
#define mdvector_hpp

/*! mdvector.hpp 
 * \brief Template class for multidimensional vector implementation.
 *
 * \author Josh Romero, Stanford University
 *
 */

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

template <typename T>
class mdvector;
template <typename T>
std::ostream& operator<<(std::ostream &os, const mdvector<T> &vec);

template <typename T>
class mdvector
{
  private:
    int ndims = 0;
    std::array<unsigned int,4> dims; 
    std::array<unsigned int,4> strides;
    std::vector<T> values;

  public:
    //! Constructors
    mdvector();
    mdvector(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0);

    //! Setup operator
    void assign(std::vector<unsigned int> dims, T value = 0, unsigned int padding = 0);

    //! Method to return vector shape
    std::array<unsigned int,4> shape(void) const;

    //! Method to return starting data pointer
    const T* data();

    //! Overloaded methods to access data
    T& operator()(unsigned int idx0, unsigned int idx1);
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2);
    T& operator()(unsigned int idx0, unsigned int idx1, unsigned int idx2, unsigned int idx3);

    //! Assignment
    mdvector<T>& operator= (const mdvector<T> &vec);
    
    //! Utility function for print
    friend std::ostream& operator<<<>(std::ostream &os, const mdvector<T> &vec);

};

template <typename T>
mdvector<T>::mdvector(){};

template <typename T>
mdvector<T>::mdvector(std::vector<unsigned int> dims, T value, unsigned int padding)
{
  ndims = (int)dims.size();
  
  assert(ndims <= 4);
  
  unsigned int nvals = 1;
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
  mdvector<T> vec(dims, value, padding);
  this->ndims = vec.ndims;
  this->values = vec.values;
  this->dims = vec.dims;
  this->strides = vec.strides;
}

template <typename T>
std::array<unsigned int,4> mdvector<T>::shape(void) const
{
  return dims;
}

template <typename T>
const T* mdvector<T>::data(void)
{
  return values.data();
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1) 
{
  assert(ndims == 2);
  return values[idx0 * strides[1] + idx1];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2) 
{
  assert(ndims == 3);
  return values[(idx0 * strides[1] + idx1) * strides[2] + idx2];
}

template <typename T>
T& mdvector<T>::operator() (unsigned int idx0, unsigned int idx1, unsigned int idx2, 
    unsigned int idx3) 
{
  assert(ndims == 4);
  return values[((idx0 * strides[1] + idx1) * strides[2] + idx2) * strides[3] + idx3];
}

template <typename T>
mdvector<T>&  mdvector<T>::operator= (const mdvector<T> &vec)
{
  this->values = vec.values;
  this->dims = vec.dims;
  this->strides = vec.strides;
}

template<typename T>
std::ostream& operator<<(std::ostream &os, const mdvector<T> &vec)
{
  if (vec.ndims > 2)
    std::runtime_error("mdvector printing supports up to 2D vectors.");

  for (unsigned int i = 0; i < vec.dims[0]; i++)
  {
    for (unsigned int j = 0; j < vec.dims[1]; j++)
      os << vec.values[i*vec.strides[1] + j] << ' ';

    os << std::endl;
  }
  
  return os;
}
#endif /* mdvector_hpp */
