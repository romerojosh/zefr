#ifndef geometry_hpp
#define geometry_hpp

#include <map>
#include <string>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "input.hpp"
#include "mdvector.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

#define HOLE 0
#define FRINGE -1
#define NORMAL 1

#ifdef _MPI
#include "mpi.h"
#endif

//! Struct for reading/storing face connectivity from PyFR mesh format
typedef struct {
  char c_type[5];
  int ic;
  short loc_f;
  char tag; // Optional tag
} face_con;

struct GeoStruct
{
  /* Maps to organize geometry data by element type */
  std::set<ELE_TYPE> ele_set; // Set of element types discovered in mesh
  mdvector<int> ele_types; // List of element types in mesh
  std::set<ELE_TYPE> face_set;
  mdvector<int> face_types; // List of face types in mesh
  std::map<ELE_TYPE, int> nFacesBT;    //! # of faces per face type
  mdvector<int> nNode_face;   //! List of # of nodes per face, by type
  std::map<ELE_TYPE, mdvector<unsigned int>> eleID;  // unique ID of element across all element types
  mdvector<unsigned int> eleIDg; // Unique global ele ID from Gmsh file
  mdvector<int> gID2ele;         // Map from global ele ID to rank- and type-local ele ID
  mdvector<ELE_TYPE> gEtype;     // Map from global ele ID to rank- and type-local ele ID
  std::map<ELE_TYPE, mdvector<unsigned int>> faceID; // unique ID of face across all face types
  mdvector<unsigned int> faceID_type;
  mdvector<unsigned int> eleID_type;
  mdvector<ELE_TYPE> eleType;  // element type by unique ID;
  mdvector<ELE_TYPE> faceType; // face type by unique ID
  std::map<ELE_TYPE, unsigned int> nElesBT;
  std::map<ELE_TYPE, unsigned int> nFacesPerEleBT;
  std::map<ELE_TYPE, unsigned int> nNodesPerEleBT;
  std::map<ELE_TYPE, unsigned int> nNodesPerFaceBT;
  std::map<ELE_TYPE, unsigned int> nCornerNodesBT;
  std::map<ELE_TYPE, unsigned int> nFptsPerFaceBT;
  std::map<ELE_TYPE, unsigned int> nFptsPerEleBT;
  std::map<ELE_TYPE, mdvector<double>> weights_fpts;
  std::map<ELE_TYPE, mdvector<int>> ele2nodesBT;
  std::map<ELE_TYPE, std::vector<ELE_TYPE>> eleFaceTypesBT;
  std::map<ELE_TYPE, std::vector<std::vector<unsigned int>>> face_nodesBT;
  std::map<ELE_TYPE, mdvector<int>> fpt2gfptBT;
  std::map<ELE_TYPE, mdvector<char>> fpt2gfpt_slotBT;


  unsigned int nEles = 0; 
  unsigned int nBnds = 0;
  unsigned int nDims, nNodes, nFaces;
  std::map<ELE_TYPE, unsigned int> nFptsPerFace, nNodesPerFace;
  unsigned int nGfpts, nGfpts_int, nGfpts_bnd;
  unsigned int nGfpts_mpi = 0;
  bool per_bnd_flag = false;

  std::map<ELE_TYPE, unsigned int> nNdFaceCurved; /// # of nodes per face, including edge and interior nodes
  std::map<ELE_TYPE, mdvector<int>> faceNodesCurved;

  /* Connectivity Data */
  mdvector<int> face2eles, face2eles_idx;
  std::map<ELE_TYPE, mdvector<int>> ele2face, face2nodes;
  std::vector<int*> c2v_ptr, f2v_ptr, c2f_ptr;
  std::vector<int> nf_ptr, nv_ptr, nc_ptr, ncf_ptr;

  std::vector<unsigned int> bnd_ids;  //! List of boundary conditions for each boundary
  mdvector<char> gfpt2bnd;
  std::map<std::vector<unsigned int>, int> bnd_faces;
  std::map<std::vector<unsigned int>, ELE_TYPE> bnd_face_type;
  std::map<std::vector<unsigned int>, std::vector<unsigned int>> per_bnd_pairs;
  std::map<ELE_TYPE, mdvector<double>> ele_nodes;
  mdvector<double> coord_nodes;

  mdvector<double> grid_vel_nodes, coords_init;

  /* Connectivity Data for Implicit */
  mdvector<int> ele2eleN, face2faceN, fpt2fptN;
#ifdef _GPU
  std::map<ELE_TYPE, mdvector_gpu<unsigned int>> eleID_d;
  std::map<ELE_TYPE, mdvector_gpu<unsigned int>> faceID_d;
  std::map<ELE_TYPE, mdvector_gpu<double>> weights_fpts_d;
  mdvector_gpu<int> ele2eleN_d, face2faceN_d, fpt2fptN_d;
#endif

  /* Color Data for MCGS */
  unsigned int nColors;
  std::map<ELE_TYPE, mdvector<int>> ele2colorBT;
  std::map<ELE_TYPE, std::vector<unsigned int>> nElesPerColorBT;
  std::map<ELE_TYPE, std::vector<unsigned int>> rangePerColorBT;

  unsigned int nBounds;               //! Number of distinct mesh boundary regions
  std::map<unsigned int,int> bcIdMap; //! Map from Gmsh boundary ID to Flurry BC ID
  std::vector<std::string> bcNames;   //! Name of each boundary given in mesh file
  std::vector<std::string> bcGlobal;  //! Global list of all boundaries in mesh
  std::vector<unsigned int> bcType;   //! Boundary condition for each boundary face
  std::map<std::vector<unsigned int>, unsigned int> face2bnd;
  std::vector<std::vector<unsigned int>> boundFaces; //! List of face IDs for each mesh-defined boundary
  std::vector<std::vector<ELE_TYPE>> boundFaceTypes; //! List of face types for each boundary face
  mdvector<int> wallFaces;

  //! --- New additions for PyFR format (consider re-organizing) ---
  mdvector<face_con> face_list;
  std::vector<std::vector<face_con>> bound_faces;
#ifdef _MPI
  std::map<int,std::vector<face_con>> mpi_conn;
  std::vector<int> send_ranks;
#endif
  std::string mesh_uuid, config, stats;

  std::map<ELE_TYPE, std::vector<int>> pyfr2zefr_face, zefr2pyfr_face;

  _mpi_comm myComm;
#ifdef _MPI
  std::map<std::vector<unsigned int>, std::set<int>> mpi_faces;
  std::vector<std::vector<unsigned int>> per_mpi_faces;
  std::unordered_map<unsigned int, unsigned int> node_map_p2g, node_map_g2p;
  std::map<unsigned int, mdvector<unsigned int>> fpt_buffer_map, face_buffer_map;

  unsigned int nMpiFaces;
  std::vector<int> procR, faceID_R, gIC_R, mpiLocF, mpiRotR, mpiLocF_R, mpiPeriodic;
#endif
  mdvector<char> flip_beta;

#ifdef _GPU
    std::map<ELE_TYPE, mdvector_gpu<int>> fpt2gfptBT_d;
    std::map<ELE_TYPE, mdvector_gpu<int>> ele2nodesBT_d;
    std::map<ELE_TYPE, mdvector_gpu<char>> fpt2gfpt_slotBT_d;
    mdvector_gpu<char> gfpt2bnd_d;
    mdvector_gpu<double> coords_init_d, coord_nodes_d, grid_vel_nodes_d;
    mdvector_gpu<char> flip_beta_d;
    mdvector_gpu<int> faceType_d;
#ifdef _MPI
  std::map<unsigned int, mdvector_gpu<unsigned int>> fpt_buffer_map_d;
#endif
#endif

  /* --- Motion-Related Variables --- */
  mdvector<double> vel_nodes;  //! Grid velocity at all mesh nodes

  double mass;
  mdvector<double> Imat; //! Inertia tensor in global coords
  mdvector<double> Jmat; //! Inertia tensor in body coords
  mdvector<double> Jinv; //! Inverse of inertia tensor in body coords
  mdvector<double> x_cg, vel_cg; //! Position and linear velocity of body frame
  mdvector<double> tmp_x_cg, tmp_Rmat; //! Temporary data structures for unblank procedure
  mdvector<double> dx_cg;
  mdvector<double> q, qdot; //! Rotation quaternion of body frame (and derivative)
  mdvector<double> Rmat;    //! Matrix form of rotation quaternion
  mdvector<double> Wmat;    //! Matrix form of omega cross-product
  mdvector<double> dRmat;   //! Combination of current and previous rotation to update from previous time step
  mdvector<double> omega;   //! Angular velocity of body in body-frame coordinates
  std::array<double,3> omega_res;
  mdvector<double> qdot_res, q_res; //! Residual for rotation quaternion update eqns.
  mdvector<double> x_res, v_res; //! Residuals of rigid-body dynamics for RK time-stepping

  /* --- Overset-Related Variables --- */

  InputStruct *input;

  unsigned int nBndFaces, nIntFaces, nOverFaces;
  std::vector<std::vector<unsigned int>> bndPts;   //! List of points on each boundary
  std::vector<std::vector<unsigned int>> faceList; //! Ordered list of faces matching c2f / f2c
  std::map<std::vector<unsigned int>, unsigned int> nodes_to_face; //! Map from face nodes to face ID
  std::vector<int> fpt2face; //! fpt index to face index
  std::map<ELE_TYPE, mdvector<int>> face2fpts; //! Face index to fpt indices

  std::vector<int> iblank_node; //! iblank values for nodes, cells, faces
  mdvector<int> iblank_cell, iblank_face;

  std::vector<int> bndFaces, mpiFaces; //! Current list of all boundar & MPI faces
  std::set<int> overFaces;  //! Ordered list of all current overset faces
  std::vector<int> overFaceList, wallFaceList;

  int nWall, nOver; //! Number of nodes on wall & overset boundaries
  std::vector<int> wallNodes, overNodes; //! Wall & overset boundary node lists

  mdvector<int> linear_tag; //! Tag for whether an element can be considered linear

#ifdef _GPU
  mdvector_gpu<int> iblank_fpts_d, iblank_cell_d;
  mdvector_gpu<int> iblank_face_d; /// TEMP / DEBUGGING - RETHINK LATER
  mdvector<int> iblank_fpts;

  mdvector_gpu<double> x_cg_d, vel_cg_d, q_d, qdot_d, Rmat_d, omega_d;
  mdvector_gpu<double> dx_cg_d, dRmat_d, Wmat_d;

#endif

  // Additional vars for computing forces/moments on mixed grids
  std::map<ELE_TYPE, mdvector<int>> wallFacesBT;

#ifdef _GPU
  std::map<ELE_TYPE, mdvector_gpu<int>> face2fpts_d;
  std::map<ELE_TYPE, mdvector_gpu<int>> wallFacesBT_d;
#endif

  unsigned int nGrids;  //! Number of distinct overset grids
  int nProcsGrid;       //! Number of MPI processes assigned to current (overset) grid block
  unsigned int gridID;  //! Which (overset) grid block is this process handling
  int gridRank;         //! MPI rank of process *within* the grid block [0 to nprocPerGrid-1]
  int rank;
  int nproc;
};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims, _mpi_comm comm_in);
void setup_etypes(GeoStruct &geo);
void load_mesh_data_gmsh(InputStruct *input, GeoStruct &geo);
void load_mesh_data_pyfr(InputStruct *input, GeoStruct &geo);
void read_boundary_ids(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_boundary_faces(std::ifstream &f, GeoStruct &geo);
void set_face_nodes(GeoStruct &geo);
void couple_periodic_bnds(GeoStruct &geo);
void setup_global_fpts(InputStruct *input, GeoStruct &geo, unsigned int order);
void setup_global_fpts_pyfr(InputStruct *input, GeoStruct &geo, unsigned int order);
void setup_flip_beta(InputStruct *input, GeoStruct &geo);
void pair_periodic_gfpts(GeoStruct &geo);

/* Methods for Implicit */
void set_ele_adjacency(GeoStruct &geo);
void setup_element_colors(InputStruct *input, GeoStruct &geo);
void shuffle_data_by_color(GeoStruct &geo);

#ifdef _MPI
void partition_geometry(InputStruct *input, GeoStruct &geo);
#endif

void move_grid(InputStruct *input, GeoStruct &geo, double time);

#endif /* geometry_hpp */
