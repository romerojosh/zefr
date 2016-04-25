#ifndef faces_hpp
#define faces_hpp

#include <string>
#include <vector>

#include "geometry.hpp"
#include "input.hpp"
#include "mdvector.hpp"

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

class FRSolver;
class Elements;
class Quads;
class Hexas;
class Faces 
{
  friend class FRSolver;
  friend class Elements;
  friend class Quads;
  friend class Hexas;
	friend class Filter;

  private:
    InputStruct *input = NULL;
    GeoStruct *geo = NULL;
    unsigned int nFpts, nDims, nVars;

    void apply_bcs();
    void apply_bcs_dU();
    void rusanov_flux(unsigned int startFpt, unsigned int endFpt);
    void roe_flux(unsigned int startFpt, unsigned int endFpt);
    void LDG_flux(unsigned int startFpt, unsigned int endFpt);
    void central_flux();
    void transform_flux();

    /* Routines for implicit method */
    void apply_bcs_dFdU();
    void rusanov_dFcdU(unsigned int startFpt, unsigned int endFpt);
    void roe_dFcdU(unsigned int startFpt, unsigned int endFpt);
    void LDG_dFcdU(unsigned int startFpt, unsigned int endFpt);
    void transform_dFcdU();

#ifdef _MPI
    void send_U_data();
    void recv_U_data();
    void send_dU_data();
    void recv_dU_data();
#endif

  protected:
    mdvector<double> U, dU, Fconv, Fvisc, Fcomm, Fcomm_temp, Ucomm, P;
    mdvector<double> norm, jaco, coord;
    mdvector<int> outnorm;
    mdvector<double> dA, waveSp, diffCo;
    mdvector<int> LDG_bias;
    mdvector<int> bc_bias;

    /* Structures for implicit method */
    mdvector<double> dFdUconv, dFdUvisc, dFddUvisc; 
    mdvector<double> dFcdU, dUcdU, dFcddU;
    mdvector<double> dFndUL_temp, dFndUR_temp, dFnddUL_temp, dFnddUR_temp;
    mdvector<double> dFcdU_temp, dFcddU_temp;

#ifdef _MPI
    /* Send and receive buffers to MPI communication. Keyed by paired rank. */
    std::map<unsigned int, mdvector<double>> U_sbuffs, U_rbuffs;

    /* Vector to store request handles for non-blocking comms. */
    std::vector<MPI_Request> sreqs, rreqs;
#endif

#ifdef _GPU
    mdvector_gpu<double> U_d, dU_d, Fconv_d, Fvisc_d, Fcomm_d, Fcomm_temp_d, Ucomm_d, P_d;
    mdvector_gpu<double> norm_d, jaco_d, coord_d;
    mdvector_gpu<int> outnorm_d;
    mdvector_gpu<double> dA_d, waveSp_d;
    mdvector_gpu<int> LDG_bias_d;
    mdvector_gpu<int> bc_bias_d;

#ifdef _MPI
    std::map<unsigned int, mdvector_gpu<double>> U_sbuffs_d, U_rbuffs_d;
#endif
#endif

  public:
    Faces(GeoStruct *geo, InputStruct *input);
    void setup(unsigned int nDims, unsigned int nVars);
    void compute_common_U(unsigned int startFpt, unsigned int endFpt);
    void compute_common_F(unsigned int startFpt, unsigned int endFpt);
    void compute_Fconv(unsigned int startFpt, unsigned int endFpt);
    void compute_Fvisc(unsigned int startFpt, unsigned int endFpt);
    
    /* Routines for implicit method */
    void compute_dFdUconv(unsigned int startFpt, unsigned int endFpt);
    void compute_dFdUvisc(unsigned int startFpt, unsigned int endFpt);
    void compute_dFddUvisc(unsigned int startFpt, unsigned int endFpt);
    void compute_dUcdU(unsigned int startFpt, unsigned int endFpt);
    void compute_dFcdU(unsigned int startFpt, unsigned int endFpt);
};

#endif /* faces_hpp */
