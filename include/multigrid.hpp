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
    int order, nLevels;
    std::vector<std::map<ELE_TYPE, mdvector<double>>> correctionsBT, sourcesBT, solutionsBT;
    std::vector<std::shared_ptr<FRSolver>> grids;

#ifdef _GPU
    std::vector<std::map<ELE_TYPE, mdvector_gpu<double>>> correctionsBT_d, sourcesBT_d, solutionsBT_d;
#endif

    void restrict_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);
    void prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector<double>> &correctionBT, FRSolver &grid_f);
    void prolong_U(FRSolver &grid_c, FRSolver &grid_f);
    void compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector<double>> &sourceBT);

    /* Overloaded methods for GPU */
#ifdef _GPU
    void compute_source_term(FRSolver &grid, std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT);
    void prolong_err(FRSolver &grid_c, std::map<ELE_TYPE, mdvector_gpu<double>> &correctionBT, FRSolver &grid_f);
#endif

  public:
    void setup(InputStruct *input, FRSolver &solver, _mpi_comm comm_in);
    void cycle(FRSolver &solver, std::ofstream& histfile, std::chrono::high_resolution_clock::time_point t1);
    void v_cycle(FRSolver &solver, int fine_order);
};

#endif /* multigrid_hpp */
