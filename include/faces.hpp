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

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void rusanov_flux(unsigned int startFpt, unsigned int endFpt);

    template<unsigned int nVars>
    void compute_common_U(unsigned int startFpt, unsigned int endFpt);

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void LDG_flux(unsigned int startFpt, unsigned int endFpt);

    /* Routines for implicit method */
    void apply_bcs_dFdU();

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void rusanov_dFdU(unsigned int startFpt, unsigned int endFpt);

    template<unsigned int nVars, unsigned int nDims, unsigned int equation>
    void LDG_dFdU(unsigned int startFpt, unsigned int endFpt);

#ifdef _MPI
    void send_U_data();
    void recv_U_data();
    void send_dU_data();
    void recv_dU_data();
#endif

  protected:
    mdvector<double> P;
    mdview<double> U, dU, Fcomm, Ucomm, U_ldg;
    mdvector<double> U_bnd, U_bnd_ldg, dU_bnd, Fcomm_bnd, Ucomm_bnd;
    mdvector<double> norm, coord;
    mdvector<double> dA, waveSp, diffCo;
    mdvector<char> rus_bias, LDG_bias;

    mdvector<double> Vg;  //! Grid velocity

    /* Structures for implicit method */
    mdview<double> dUcdU, dFcdU, dFcddU;
    mdvector<double> dUcdU_bnd, dFcdU_bnd, dFcddU_bnd;
    mdvector<double> dUbdU, ddUbddU;

    /* Moving-Grid Variables */
    mdvector<double> norm_init;

    _mpi_comm myComm;
#ifdef _MPI
    mdview<double> U_mpi, dU_mpi;

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
#endif

#ifdef _GPU
    mdvector_gpu<double> P_d;
    mdview_gpu<double> U_d, U_ldg_d, dU_d, Fcomm_d, Ucomm_d;
    mdvector_gpu<double> U_bnd_d, U_bnd_ldg_d, dU_bnd_d, Fcomm_bnd_d, Ucomm_bnd_d;
    mdvector_gpu<double> norm_d, jaco_d, coord_d;
    mdvector_gpu<double> dA_d, waveSp_d, diffCo_d;
    mdvector_gpu<char> rus_bias_d, LDG_bias_d;

    mdvector_gpu<double> Vg_d;

    /* Structures for implicit method */
    mdview_gpu<double> dUcdU_d, dFcdU_d, dFcddU_d;
    mdvector_gpu<double> dUcdU_bnd_d, dFcdU_bnd_d, dFcddU_bnd_d;
    mdvector_gpu<double> dUbdU_d, ddUbddU_d;

    /* Moving-Grid Vars */
    mdvector_gpu<double> norm_init_d;

#ifdef _MPI
    std::map<unsigned int, mdvector_gpu<double>> U_sbuffs_d, U_rbuffs_d;
#endif
#endif

#ifdef _GPU
    std::map<ELE_TYPE, mdvector_gpu<double>> U_fringe_d, dU_fringe_d;
    std::map<ELE_TYPE, mdvector_gpu<unsigned int>> fringe_fpts_d, fringe_side_d;

    std::map<ELE_TYPE, mdvector<double>> U_fringe, dU_fringe;
    std::map<ELE_TYPE, mdvector<unsigned int>> fringe_fpts, fringe_side;

    std::map<ELE_TYPE, unsigned int> nfringe_type;

    mdvector_gpu<unsigned int> fringeGFpts_d;
    mdvector_gpu<double> fringeCoords_d;
    mdvector<unsigned int> fringeGFpts;
#endif

  public:
    Faces(GeoStruct *geo, InputStruct *input, _mpi_comm comm_in);
    void setup(unsigned int nDims, unsigned int nVars);
    void compute_common_U(unsigned int startFpt, unsigned int endFpt);
    void compute_common_F(unsigned int startFpt, unsigned int endFpt);
    
    /* Routines for implicit method */
    void compute_common_dFdU(unsigned int startFpt, unsigned int endFpt);

    //! Get location of right-state U data for a flux point
    void get_U_index(int faceID, int fpt, int& ind, int& stride);
    double& get_u_fpt(int faceID, int fpt, int var);
    double& get_grad_fpt(int faceID, int fpt, int var, int dim);

#if defined(_GPU) && defined(_BUILD_LIB)
    void fringe_u_to_device(int* fringeIDs, int nFringe);
    void fringe_u_to_device(int* fringeIDs, int nFringe, double* data);
    void fringe_grad_to_device(int* fringeIDs, int nFringe);
    void fringe_grad_to_device(int* fringeIDs, int nFringe, double* data);
    void get_face_coords(int* fringeIDs, int nFringe, int* nPtsFace, double* xyz);
#endif
};

#endif /* faces_hpp */
