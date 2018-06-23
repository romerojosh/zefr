#include "mdvector_gpu.h"

void normalize_data_wrapper(mdvector_gpu<double> &U_spts, double normalTol, unsigned int nSpts,
    unsigned int nEles, unsigned int nVars);

void compute_sensor_wrapper(mdvector_gpu<double>& KS, mdvector_gpu<double>& sensor,
    double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, 
    unsigned int nDims, unsigned int nSptsKS, double Q, double epsilon);

void copy_filtered_solution_wrapper(mdvector_gpu<double> &U_spts_filt, mdvector_gpu<double> &U_spts,
    mdvector_gpu<double> &sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars);
