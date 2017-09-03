%module zefr

// -----------------------------------------------------------------------------
// Header files required by any of the following C++ code
// -----------------------------------------------------------------------------
%header
%{
#include <numpy/arrayobject.h>
#ifdef _MPI
#include <mpi.h>
#define _mpi_comm MPI_Comm
#define DEFAULT_COMM MPI_COMM_WORLD
#else
#define _mpi_comm int
#define DEFAULT_COMM 0
#endif

#include "zefr_interface.hpp"
%}

%init
%{
  import_array();
%}

// -----------------------------------------------------------------------------
// Header files and other declarations to be parsed as SWIG input
// -----------------------------------------------------------------------------

#ifdef _MPI
// MPI SWIG interface file & MPI_Comm to Python Comm typemap
%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm,MPI_Comm);
#endif

%include "inputstruct.hpp"
%include "zefr.hpp"
%include "zefr_interface.hpp"

// <-- Additional C++ declations [anything that would normally go in a header]

// -----------------------------------------------------------------------------
// Additional functions which have been declared, but not defined (including
// definition in other source files which will be linked in later)
// -----------------------------------------------------------------------------

%inline
%{
// <-- Additional C++ definitions [anything that would normally go in a .cpp]
%}

// --------------------------------------------------------
// FUNCTIONS TO CONVERT POINTERS TO NUMPY ARRAYS (AND BACK)
// --------------------------------------------------------
%inline
%{
PyObject* ptrToArray(float* data, int n)
{
  npy_intp dims[1] = {n};
  return PyArray_SimpleNewFromData(1,dims,NPY_FLOAT,(void*)data);
}

PyObject* ptrToArray(double* data, int n)
{
  npy_intp dims[1] = {n};
  return PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,(void*)data);
}

PyObject* ptrToArray(int* data, int n)
{
  npy_intp dims[1] = {n};
  return PyArray_SimpleNewFromData(1,dims,NPY_INT,(void*)data);
}

PyObject* ptrToArray(unsigned int* data, int n)
{
  npy_intp dims[1] = {n};
  return PyArray_SimpleNewFromData(1,dims,NPY_UINT,(void*)data);
}

double* arrayToDblPtr(PyObject* arr)
{
  return (double *)(((PyArrayObject *)arr)->data);
}

float* arrayToFloatPtr(PyObject* arr)
{
  return (float *)(((PyArrayObject *)arr)->data);
}

int* arrayToIntPtr(PyObject* arr)
{
  return (int *)(((PyArrayObject *)arr)->data);
}

unsigned int* arrayToUintPtr(PyObject* arr)
{
  return (unsigned int *)(((PyArrayObject *)arr)->data);
}
%}


// -----------------------------------------------------------------------------
// Additional Python functions to add to module
// [can use any functions/variables declared above]
// -----------------------------------------------------------------------------

%pythoncode
%{

%}
