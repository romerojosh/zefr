#ifndef multigrid_hpp
#define multigrid_hpp

#include <memory>
#include <vector>

#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class PMGrid
{
  private:
    //const InputStruct *input = NULL;
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
    void compute_source_term(FRSolver &grid, mdvector<double> &source);

    /* Overloaded methods for GPU */
#ifdef _GPU
    void compute_source_term(FRSolver &grid, mdvector_gpu<double> &source);
    void prolong_err(FRSolver &grid_c, mdvector_gpu<double> &correction_c, FRSolver &grid_f);
#endif

  public:
    //void setup(int order, const InputStruct *input, FRSolver &solver);
    void setup(int order, InputStruct *input, FRSolver &solver);
    void cycle(FRSolver &solver);


};

#endif /* multigrid_hpp */
