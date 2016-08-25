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

#ifdef _BUILD_LIB
#include "zefr.hpp"
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
#ifdef _BUILD_LIB
  friend class Zefr;
#endif

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

    void send_U_data_blocked(void);
    void recv_U_data_blocked(int mpiFaceID);

    void mpi_prod(void);
#endif

  protected:
    mdvector<double> U, dU, Fconv, Fvisc, Fcomm, Fcomm_temp, Ucomm, P;
    mdvector<double> norm, jaco, coord;
    mdvector<double> dA, waveSp, diffCo;
    mdvector<int> LDG_bias;
    std::vector<double> FL, FR, WL, WR;

    mdvector<double> Vg;  //! Grid velocity

    /* Structures for implicit method */
    bool CPU_flag; // Temporary flag for global implicit method
    mdvector<double> dFdUconv, dFdUvisc, dFddUvisc; 
    mdvector<double> dFcdU, dUcdU, dFcddU;
    mdvector<double> dFndUL_temp, dFndUR_temp, dFnddUL_temp, dFnddUR_temp;
    mdvector<double> dFcdU_temp, dFcddU_temp;
    mdvector<double> dFdURconv, dUcdUR, dFdURvisc, dFddURvisc;
    mdvector<double> dURdUL, ddURddUL;

    _mpi_comm myComm;
#ifdef _MPI
    /* Send and receive buffers to MPI communication. Keyed by paired rank. */
    std::map<unsigned int, mdvector<double>> U_sbuffs, U_rbuffs;

    /* Vector to store request handles for non-blocking comms. */
    std::vector<MPI_Request> sreqs, rreqs;
    int nsends, nrecvs;

    /// JACOB'S ADDITIONS FOR TESTING NEW COMMUNICATION STRATEGY
    std::vector<MPI_Request> sends, recvs;
    std::vector<MPI_Status> sstatuses, rstatuses;
    std::array<MPI_Status,2> new_statuses;
    int n_reqs;
    MPI_Status status;
    std::vector<mdvector<double>> buffUR, buffUL;
    std::array<std::vector<int>, 5> rot_permute;

    mdvector<double> tmpOversetU; /// TODO: find better way?
    mdvector<double> tmpOversetdU; /// TODO: find better way?
#endif

#ifdef _GPU
    mdvector_gpu<double> U_d, dU_d, Fconv_d, Fvisc_d, Fcomm_d, Fcomm_temp_d, Ucomm_d, P_d;
    mdvector_gpu<double> norm_d, jaco_d, coord_d;
    mdvector_gpu<double> dA_d, waveSp_d, diffCo_d;
    mdvector_gpu<int> LDG_bias_d;

    /* Structures for implicit method */
    mdvector_gpu<double> dFdUconv_d, dFdUvisc_d, dFddUvisc_d;
    mdvector_gpu<double> dFcdU_d, dUcdU_d;

#ifdef _MPI
    std::map<unsigned int, mdvector_gpu<double>> U_sbuffs_d, U_rbuffs_d;
#endif
#endif

#ifdef _GPU
    mdvector_gpu<double> U_fringe_d, dU_fringe_d;
    mdvector_gpu<unsigned int> fringe_fpts_d, fringe_side_d;

    mdvector<double> U_fringe, dU_fringe;
    mdvector<unsigned int> fringe_fpts, fringe_side;
#endif

  public:
    Faces(GeoStruct *geo, InputStruct *input, _mpi_comm comm_in);
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

    //! Get location of right-state U data for a flux point
    void get_U_index(int faceID, int fpt, int& ind, int& stride);
    double& get_u_fpt(int faceID, int fpt, int var);
    double& get_grad_fpt(int faceID, int fpt, int var, int dim);

#ifdef _GPU
    void fringe_u_to_device(int* fringeIDs, int nFringe);
    void fringe_grad_to_device(int nFringe);
#endif
};

#endif /* faces_hpp */
