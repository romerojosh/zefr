%module zefr

// -----------------------------------------------------------------------------
// Header files required by any of the following C++ code
// -----------------------------------------------------------------------------
%header
%{
#include "zefr.hpp"
%}

// -----------------------------------------------------------------------------
// Header files and other declarations to be parsed as SWIG input
// -----------------------------------------------------------------------------
%include "inputstruct.hpp"
%include "zefr.hpp"

// <-- Additional C++ declations [anything that would normally go in a header]

// -----------------------------------------------------------------------------
// Additional functions which have been declared, but not defined (including
// definition in other source files which will be linked in later)
// -----------------------------------------------------------------------------

%inline
%{
// <-- Additional C++ definitions [anything that would normally go in a .cpp]
%}

// -----------------------------------------------------------------------------
// Additional Python functions to add to module
// [can use any functions/variables declared above]
// -----------------------------------------------------------------------------

%pythoncode
%{
# Python functions here
%}
