#ifndef spmatrix_gpu_h
#define spmatrix_gpu_h

#include "solver_kernels.h"
#include "spmatrix.hpp"

/* spmatrix_gpu - Sparse matrix class on GPU device. 
 * Represents a sparse matrix in CSR format on the device, transfered from host.
 */
template<typename T>
class spmatrix;

template <typename T>
class spmatrix_gpu
{
  private:
    int* row_ptr; /* Vector for row-pointer (CSR), i index (COO) */
    int* col_idx; /* Vector for column index (CSR), j index (COO) */
    T* vals; 
    int nNonzeros = 0;
    int nRows = 0;
    bool allocated = false;

  public:
    /* Method to free allocated GPU data */
    void free_data();

    /* Overloaded assignment operator to transfer values */
    spmatrix_gpu<T>& operator= (spmatrix<T>& mat);

    /* Method to get CSR value vector */
    T* getVals(); 

    /* Method to get CSR row pointer vector */
    int* getRowPtr(); 

    /* Method to get CSR column index vector */
    int* getColIdx(); 

    /* Method to get number of rows */
    int getNrows(); 

    /* Method to get number of nonzero entries */
    int getNnonzeros(); 

};

template <typename T>
void spmatrix_gpu<T>::free_data()
{
  if (allocated)
  {
    free_device_data(vals);
    free_device_data(col_idx);
    free_device_data(row_ptr);

    allocated = false;
  }
}

template <typename T>
spmatrix_gpu<T>& spmatrix_gpu<T>::operator=(spmatrix<T>& mat)
{
  nNonzeros = mat.getNnonzeros();
  nRows= mat.getNrows();

  if(!allocated)
  {
    allocate_device_data(vals, nNonzeros);
    allocate_device_data(col_idx, nNonzeros);
    allocate_device_data(row_ptr, nRows + 1);

    allocated = true;
  }

  /* Copy values to GPU */
  copy_to_device(vals, mat.getVals().data(), nNonzeros);
  copy_to_device(col_idx, mat.getColIdx().data(), nNonzeros);
  copy_to_device(row_ptr, mat.getRowPtr().data(), nRows + 1);

  return *this;
}


template <typename T>
T* spmatrix_gpu<T>::getVals()
{
  return vals;
}

template <typename T>
int* spmatrix_gpu<T>::getRowPtr()
{
  return row_ptr;
}


template <typename T>
int* spmatrix_gpu<T>::getColIdx()
{
  return col_idx;
}

template <typename T>
int spmatrix_gpu<T>::getNrows()
{
  return nRows;
}

template <typename T>
int spmatrix_gpu<T>::getNnonzeros()
{
  return nNonzeros;
}
#endif /* spmatrix_gpu_h */
