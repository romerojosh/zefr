#ifndef solver_hpp
#define solver_hpp

#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "elements.hpp"
#include "faces.hpp"
#include "geometry.hpp"
#include "input.hpp"
#include "filter.hpp"

#ifdef _CPU
#include <Eigen/Dense>
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#ifdef _BUILD_LIB
#include "zefr.hpp"
#endif

class PMGrid;
#ifdef _BUILD_LIB
class Zefr;
#endif

class FRSolver
{
  friend class PMGrid;
  friend class Filter;
#ifdef _BUILD_LIB
  friend class Zefr;
#endif
  private:
    InputStruct *input = NULL;
    GeoStruct geo;
    int order;
    std::shared_ptr<Elements> eles;
    std::shared_ptr<Faces> faces;
    
    std::vector<std::shared_ptr<Elements>> elesObjs;
    std::map<ELE_TYPE, std::shared_ptr<Elements>> elesObjsBT;

    int current_iter = 0;
    int restart_iter = 0;
    double flow_time = 0.;
    double prev_time = 0.;
    double restart_time = 0.;
    double grid_time = 0.;
    mdvector<double> rk_alpha, rk_beta, rk_bhat, rk_c;
    Filter filt;

    /* --- Adaptive time-stepping stuff --- */
    double prev_err;        //! RK error estimate for previous step
    double expa, expb;
    unsigned int rejected_steps = 0;

    /* --- Rigid-Body Motion --- */
    mdvector<double> x_ini, x_til;         //! Position of body CG
    mdvector<double> v_ini, v_til;         //! Velocity of body CG
    mdvector<double> omega_ini, omega_til; //! Angular velocity of body
    mdvector<double> nodes_ini, nodes_til; //! Grid node positions
    mdvector<double> q_ini, q_til; //! Grid rotation vector
    mdvector<double> qdot_ini, qdot_til; //! Grid rotation vector

    /* --- Averaging & Statistics --- */
    double tavg_prev_time = 0.;

    /* Implicit method parameters */
#ifdef _CPU
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> 
      MatrixXdRM;                       // Eigen matrix type used for LHS
#endif
    std::vector<std::vector<std::shared_ptr<Elements>>> 
      elesObjsBC;                       // elesObj by color
    mdvector<unsigned int> ele2elesObj; // Map from ele to elesObj
    unsigned int startStage = 0;        // Starting stage for DIRK
    double dtau_ratio = 1.0;            // Ratio of dtau / dt
    unsigned int nCounter;              // Number of sweeps in block iteration
    int prev_color = 0;                 // Previous color
    unsigned int report_NMconv_freq;    // Report frequency for Newton's Method
    std::ofstream conv_file;            // File used for convergence output
    std::chrono::high_resolution_clock::time_point 
      conv_timer;                       // Convergence timer

    /* Viscous implicit jacoN data for MPI boundaries */
#ifdef _MPI
    mdvector<double> inv_jacoN_spts_mpibnd, jacoN_det_spts_mpibnd;
#ifdef _GPU
    mdvector_gpu<double> inv_jacoN_spts_mpibnd_d, jacoN_det_spts_mpibnd_d;
#endif
#endif

#ifdef _GPU
    mdvector_gpu<double> rk_alpha_d, rk_beta_d, rk_bhat_d;

    mdvector_gpu<double> nodes_ini_d, nodes_til_d;
    mdvector_gpu<double> x_ini_d, x_til_d;
    mdvector_gpu<double> v_ini_d, v_til_d;
    mdvector_gpu<double> q_ini_d, q_til_d;
    mdvector_gpu<double> qdot_ini_d, qdot_til_d;

    mdvector_gpu<double> force_d, moment_d; //! Force / Moment *per boundary face* for reduction op
#endif

    _mpi_comm myComm, worldComm;

    void setup_views();
    void restart_paraview(std::string restart_case, unsigned int restart_iter, int npart_restart);
    void restart_pyfr(std::string restart_case, unsigned restart_iter);
    void process_restart_stats(const std::string &stats_str);
    void setup_update();
    void setup_output();
    void create_elesObj(ELE_TYPE etype, unsigned int elesObjID, unsigned int startEle, unsigned int endEle);
    void orient_fpts();

#ifdef _GPU
    void solver_data_to_device();
#endif

    void compute_element_dt(bool pseudo_flag = false);

#ifdef _GPU
    // For moving grids, to pass parameters to CUDA kernels more easily
    MotionVars motion_vars;
#endif

    /* Routines for implicit method */
    void set_fpt_adjacency();
    void setup_jacoN_views();
    void compute_LHS(unsigned int stage);
    void compute_dRdU();
    void compute_LHS_LU();
    void compute_LHS_inverse();
    void compute_LHS_SVD();
    void compute_RHS(unsigned int stage, int color = -1);
    void compute_deltaU(int color = -1);
    void compute_U(int color = -1);

  public:
    double res_max = std::numeric_limits<double>::max();
    FRSolver(InputStruct *input, int order = -1);
    void setup(_mpi_comm comm_in, _mpi_comm comm_world = DEFAULT_COMM);
    void restart_solution(void);
    void compute_residual(unsigned int stage, int color = -1);
    void compute_residual_start(unsigned int stage, int color = -1);
    void compute_residual_mid(unsigned int stage, int color = -1);
    void compute_residual_finish(unsigned int stage, int color = -1);
    void add_source(unsigned int stage, unsigned int startEle, unsigned int endEle);

    void step_RK_stage_start(int stage);
    void step_RK_stage_mid(int stage);
    void step_LSRK_stage_start(int stage);
#ifdef _CPU
    void update(const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());
    void step_adaptive(const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());

    //! Standard explicit (diagonal) Runge-Kutta update loop
    void step_RK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());
    void step_RK_stage(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());
    void step_RK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());
    void step_LSRK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());

    //! Special Low-Storage (2-register) Runge-Kutta update loop
    void step_LSRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());

    //! Implicit Steady State update loop
    void step_Steady(unsigned int stage, unsigned int iterNM, const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());

    //! Diagonally Implicit Runge-Kutta update loop
    void step_DIRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT = std::map<ELE_TYPE, mdvector<double>>());
#endif
#ifdef _GPU
    void update(const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_adaptive(const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_RK(const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_RK_stage(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_RK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_LSRK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_LSRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_Steady(unsigned int stage, unsigned int iterNM, const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
    void step_DIRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &source = std::map<ELE_TYPE, mdvector_gpu<double>>());
#endif

    double adapt_dt(void);

    void accumulate_time_averages(void);

    void write_solution(const std::string &_prefix);
    void write_solution_pyfr(const std::string &_prefix);
    void write_surfaces(const std::string &_prefix);
    void write_overset_boundary(const std::string &_prefix);
    void write_LHS(const std::string &_prefix);
    void write_RHS(const std::string &_prefix);
    void write_color();
    void write_averages(const std::string &_prefix);
    void report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1);
    void report_RHS(unsigned int stage, unsigned int iterNM, unsigned int iter);
    void report_forces(std::ofstream &f, double* tot_force = NULL);
    void report_error(std::ofstream &f);
    void report_turbulent_stats(std::ofstream &f);
#ifdef _GPU
    void report_gpu_mem_usage();
#endif
    void set_conv_file(std::chrono::high_resolution_clock::time_point t1);
    double get_current_time(void);
    void filter_solution();

    void compute_moments(std::array<double, 3>& tot_force, std::array<double, 3>& tot_moment, std::ofstream* cp_file = NULL);

#ifdef _BUILD_LIB
    Zefr *ZEFR;
#endif

    void init_grid_motion(double time);
    void move(double time, bool update_iblank = false);
    void move_grid_now(double time);
    void move_grid_next(double time);
    void rigid_body_update(int stage);

    /* Wrappers for the callback functions for overset interpolation */
    void overset_u_send(void);
    void overset_u_recv(void);
    void overset_grad_send(void);
    void overset_grad_recv(void);
};

#endif /* solver_hpp */
