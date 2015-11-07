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

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class PMGrid;
class FRSolver
{
  friend class PMGrid;

  private:
    InputStruct *input = NULL;
    GeoStruct geo;
    int order;
    int restart_iter = 0;
    double flow_time = 0.;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;

    unsigned int nStages;
    mdvector<double> rk_alpha, rk_beta;
    mdvector<double> dt;
    mdvector<double> U_ini;

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
    void U_to_faces();
    void U_from_faces();
    void dU_to_faces();
    void F_from_faces();

    void compute_element_dt();

  public:
    FRSolver(InputStruct *input, int order = -1);
    void setup();
    void compute_residual(unsigned int stage);
    void update();
    void update_with_source(mdvector<double> &source);
#ifdef _GPU
    void update_with_source(mdvector_gpu<double> &source);
#endif
    void write_solution(std::string prefix, unsigned int nIter);
    void write_solution_vtk(std::string prefix, unsigned int nIter);
    void report_residuals(std::ofstream &f, unsigned int iter, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::string prefix, std::ofstream &f, unsigned int iter);
    void report_error(std::ofstream &f, unsigned int iter);


};

#endif /* solver_hpp */
