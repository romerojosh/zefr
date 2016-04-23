#ifndef solver_hpp
#define solver_hpp

#include <fstream>
#include <chrono>
#include <memory>
#include <string>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "filter.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class PMGrid;

class FRSolver
{
  friend class PMGrid;
  friend class Filter;
  
  private:
    InputStruct *input = NULL;
    GeoStruct geo;
    int order;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    int current_iter = 0;
    double flow_time = 0.;
    unsigned int nStages;
    mdvector<double> rk_alpha, rk_beta;
    mdvector<double> dt;
    mdvector<double> U_ini, U_avg;
    Filter filt;

    /* Additioanl "Finite-Volume" variables */
    bool FV_mode;
    mdvector<int> FV_parts; 
    std::vector<std::vector<double>> FV_vols;

#ifdef _GPU
    mdvector_gpu<double> U_ini_d, dt_d, rk_alpha_d, rk_beta_d;
#endif

    void initialize_U();
    void restart(std::string restart_file);
    void setup_update();
    void setup_output();
    void setup_h_levels();

#ifdef _GPU
    void solver_data_to_device();
#endif


    /* Routines to communicate data between faces and elements */
    void U_to_faces();
    void U_from_faces();
    void dU_to_faces();
    void F_from_faces();

    void compute_element_dt();

  public:
    FRSolver(InputStruct *input, int order = -1, bool FV_mode = false);
    void setup();
    void compute_residual(unsigned int stage);
    void add_source(unsigned int stage);
#ifdef _CPU
    void update(const mdvector<double> &source = mdvector<double>());
#endif
#ifdef _GPU
    void update(const mdvector_gpu<double> &source = mdvector_gpu<double>());
#endif
    void write_solution(const std::string &prefix);
    void write_partition_file();
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::ofstream &f);
    void report_error(std::ofstream &f);
    void filter_solution();

};

#endif /* solver_hpp */
