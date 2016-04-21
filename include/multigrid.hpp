#ifndef multigrid_hpp
#define multigrid_hpp

#include <memory>
#include <vector>
#include <chrono>
#include <fstream>

#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class PMGrid
{
  private:
    InputStruct *input = NULL;
    int order;
    std::vector<mdvector<double>> corrections, sources, solutions;
    std::vector<std::shared_ptr<FRSolver>> grids;

#ifdef _GPU
    std::vector<mdvector_gpu<double>> corrections_d, sources_d, solutions_d;
#endif

    void restrict_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);
    void prolong_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);
    void prolong_err(FRSolver &grid_c, mdvector<double> &correction_c, FRSolver &grid_f);
    void prolong_U(FRSolver &grid_c, FRSolver &grid_f);
    void compute_source_term(FRSolver &grid, mdvector<double> &source);

    /* Overloaded methods for GPU */
#ifdef _GPU
    void compute_source_term(FRSolver &grid, mdvector_gpu<double> &source);
    void prolong_err(FRSolver &grid_c, mdvector_gpu<double> &correction_c, FRSolver &grid_f);
#endif

  public:
    void setup(InputStruct *input, FRSolver &solver);
    void cycle(FRSolver &solver, std::ofstream& histfile, std::chrono::high_resolution_clock::time_point t1);
    void v_cycle(FRSolver &solver, int fine_order);
};

#endif /* multigrid_hpp */
