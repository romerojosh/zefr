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

    /* Additional "Finite-Volume" variables */
    bool FV_mode;
    mdvector<int> FV_ele2part; 
    std::vector<mdvector<int>> FV_part2eles;
    std::vector<mdvector<double>> FV_vols;
    mdvector<double> dt_part;

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

    void compute_element_dt(int FV_level = 0);

  public:
    FRSolver(InputStruct *input, int order = -1, bool FV_mode = false);
    void setup();
    void compute_residual(unsigned int stage, int FV_level = 0);
    void add_source(unsigned int stage);
    void accumulate_partition_U(int FV_level);
    void accumulate_partition_divF(unsigned int stage, int FV_level);
    void accumulate_partition_divF_weighted(unsigned int stage, int FV_level);
#ifdef _CPU
    void update(const mdvector<double> &source = mdvector<double>());
    void update_FV(int FV_level, const mdvector<double> &source = mdvector<double>());
#endif
#ifdef _GPU
    void update(const mdvector_gpu<double> &source = mdvector_gpu<double>());
    void update_FV(const mdvector_gpu<double> &source = mdvector_gpu<double>(), int FV_level);
#endif
    void write_solution(const std::string &prefix);
    void write_partition_file();
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1, int FV_level = 0);
    void report_forces(std::ofstream &f);
    void report_error(std::ofstream &f);
    void filter_solution();

};

#endif /* solver_hpp */
