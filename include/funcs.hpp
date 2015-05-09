#ifndef funcs_hpp
#define funcs_hpp

#include "input.hpp"

/* Computes solution at specified time and location */
double compute_U_true(double x, double y, double t, unsigned int var, const InputStruct *input);
double compute_U_true(double x, double y, double z, double t, unsigned int var, const InputStruct *input);
                     
#endif /* funcs_hpp */
