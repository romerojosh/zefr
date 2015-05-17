#ifndef multigrid_hpp
#define multigrid_hpp

#include <memory>
#include <vector>

#include "input.hpp"
#include "mdvector.hpp"
#include "solver.hpp"

class PMGrid
{
  private:
    const InputStruct *input = NULL;
    int order;
    std::vector<mdvector<double>> corrections;
    std::vector<std::shared_ptr<FRSolver>> grids;

    void restrict_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);
    void prolong_pmg(FRSolver &grid_fine, FRSolver &grid_coarse);

  public:
    void setup(int order, const InputStruct *input);
    void cycle(FRSolver &solver);


};

#endif /* multigrid_hpp */
