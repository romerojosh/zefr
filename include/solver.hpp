#ifndef solver_hpp
#define solver_hpp

#include <fstream>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "spmatrix.hpp"
#include "filter.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#include "spmatrix_gpu.h"
#endif

#ifndef _NO_TNT
#include "tnt.h"
#include <jama_lu.h>
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
    int restart_iter = 0;
    double flow_time = 0.;
    unsigned int nStages;
    double CFL_ratio = 1;
    mdvector<double> rk_alpha, rk_beta;
    mdvector<double> dt;
    mdvector<double> U_ini;
    Filter filt;

    /* Implicit method parameters */
    unsigned int nCounter;
    unsigned int prev_color;
    double SER_omg = 1;
    double SER_res[2] = {0};
    bool GMRES_conv;
#ifndef _NO_TNT
    //std::vector<std::shared_ptr<JAMA::LU<double>>> LUptrs;
    //std::vector<std::vector<std::shared_ptr<JAMA::LU<double>>>> LUptrs;
    std::vector<std::vector<JAMA::LU<double>>> LUptrs;
#endif

#ifdef _GPU
    mdvector_gpu<double> U_ini_d, dt_d, rk_alpha_d, rk_beta_d;
#endif

    void initialize_U();
    void restart(std::string restart_file);
    void setup_update();
    void setup_output();

#ifdef _GPU
    void solver_data_to_device();
#endif

    /* Routines to communicate data between faces and elements */
    void U_to_faces(unsigned int startEle, unsigned int endEle);
    void U_from_faces(unsigned int startEle, unsigned int endEle);
    void dU_to_faces(unsigned int startEle, unsigned int endEle);
    void F_from_faces(unsigned int startEle, unsigned int endEle);

    void compute_element_dt();

  public:
    double res_max = 1;
    FRSolver(InputStruct *input, int order = -1);
    void setup();
    void compute_residual(unsigned int stage, unsigned int color = 0);
    void add_source(unsigned int stage, unsigned int startEle, unsigned int endEle);
#ifdef _CPU
    void update(const mdvector<double> &source = mdvector<double>());
#endif
#ifdef _GPU
    void update(const mdvector_gpu<double> &source = mdvector_gpu<double>());
#endif
    void write_solution(const std::string &prefix);
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::ofstream &f);
    void report_error(std::ofstream &f);
    void filter_solution();

    /* Routines for implicit method */
    void compute_LHS();
    void compute_LHS_LU(unsigned int color = 1);
    void compute_RHS(unsigned int color = 1);
#ifdef _CPU
    void compute_RHS_source(const mdvector<double> &source, unsigned int color = 1);
#endif
#ifdef _GPU
    void compute_RHS_source(const mdvector_gpu<double> &source, unsigned int color = 1);
#endif
    void compute_deltaU(unsigned int color = 1);
    void compute_U(unsigned int color = 1);
    void dFcdU_from_faces();
    void compute_SER_dt();
    void write_color();
};

#endif /* solver_hpp */
