#include "mdvector_gpu.h"

void normalize_data_wrapper(mdvector_gpu<double> U_spts, double normalTol, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars);

void compute_max_sensor_wrapper(mdvector_gpu<double> KS, mdvector_gpu<double> sensor, 
    unsigned int order, unsigned int nSpts, unsigned int nEles, unsigned int nVars);
