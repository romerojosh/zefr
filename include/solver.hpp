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
    //const InputStruct *input = NULL;
    InputStruct *input = NULL;
    GeoStruct geo;
    int order;
    int restart_iter = 0;
    double flow_time = 0.;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;

    unsigned int nStages;
    //std::vector<double> rk_alpha, rk_beta;
    mdvector<double> rk_alpha, rk_beta;
    mdvector<double> divF;
    mdvector<double> dt;
    mdvector<double> U_ini;

#ifdef _GPU
    mdvector_gpu<double> divF_d, U_ini_d, dt_d, rk_alpha_d, rk_beta_d;
#endif

    void initialize_U();
    void restart(std::string restart_file);
    void setup_update();
    void setup_output();

#ifdef _GPU
    void solver_data_to_device();
#endif

    void extrapolate_U();
    /* Note: Going to create ele2fpt and slot structure like FR2D. gfpt=-1 means no comm. */
    void U_to_faces();

    /* Viscous Stuff */
    void U_from_faces();
    void compute_dU();
    void extrapolate_dU();
    void dU_to_faces();

    /* Note: These will be additive, Fvisc will use F_spts += */
    void compute_Fconv_spts();
    void compute_Fvisc_spts();

    /* Note: Do I have to transform dU? */
    void transform_F();

    void F_from_faces();

    void compute_dF();
    void compute_divF(unsigned int stage);

    void compute_element_dt();

  public:
    //FRSolver(const InputStruct *input, int order = -1);
    FRSolver(InputStruct *input, int order = -1);
    void setup();
    void compute_residual(unsigned int stage);
    void update();
    void update_with_source(mdvector<double> &source);
#ifdef _GPU
    void update_with_source(mdvector_gpu<double> &source);
#endif
    void write_solution(std::string prefix, unsigned int nIter);
    void report_residuals(std::ofstream &f, unsigned int iter, std::chrono::high_resolution_clock::time_point t1);
    void report_forces(std::string prefix, std::ofstream &f, unsigned int iter);
    void report_error(std::ofstream &f, unsigned int iter);

};

#endif /* solver_hpp */
