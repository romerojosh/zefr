%module zefr

%include "typemaps.i"
%include "global.i"

// -----------------------------------------------------------------------------
// Header files required by any of the following C++ code
// -----------------------------------------------------------------------------
%header
%{
#ifdef _MPI
#include <mpi.h>
#define _mpi_comm MPI_Comm
#define DEFAULT_COMM MPI_COMM_WORLD
#else
#define _mpi_comm int
#define DEFAULT_COMM 0
#endif

#include "zefrPyGlobals.h"
#include "zefr_interface.hpp"

//extern int zefrInit(const char *fileName, int comm);
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

// ---------------------------------------------
// FUNCTIONS TO CONVERT POINTERS TO NUMPY ARRAYS 
// ---------------------------------------------
%inline %{
void convert_to_npdarray(double* data, int _size)
{
  dataSizePy = _size;
  dataPy = data;
}

void convert_to_npiarray(int* data, int _size)
{
  dataSizePy = _size;
  idataPy = data;
}
%}


// -----------------------------------------------------------------------------
// Additional Python functions to add to module
// [can use any functions/variables declared above]
// -----------------------------------------------------------------------------

%pythoncode
%{
# Python functions here
%}

%include "zefrPyGlobals.h"

%init
%{
  import_array();
%}
