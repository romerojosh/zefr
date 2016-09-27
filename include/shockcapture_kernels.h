#include "mdvector_gpu.h"

void normalize_data_wrapper(mdvector_gpu<double>& U_spts, double normalTol, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars);

void compute_max_sensor_wrapper(mdvector_gpu<double>& KS, mdvector_gpu<double>& sensor, 
    unsigned int order, double& max_sensor, mdvector_gpu<uint>& sensor_bool, double threshJ, unsigned int nSpts, 
    unsigned int nEles, unsigned int nVars, unsigned int nDims, double Q);

void copy_filtered_solution_wrapper(mdvector_gpu<double>& U_spts_filt, mdvector_gpu<double>& U_spts, 
    mdvector_gpu<double>& sensor, double threshJ, unsigned int nSpts, unsigned int nEles, unsigned int nVars, int type);

void limiter_wrapper(uint nEles, uint nFaces, uint nVars, double threshJ, mdvector_gpu<int> ele_adj_d, 
	mdvector_gpu<double> sensor_d, mdvector_gpu<double>& Umodal_d);

// void compute_primitive_wrapper(uint nSpts, uint nEles, uint nVars, mdvector_gpu<double>& U_spts, mdvector_gpu<double>& U_prim);
