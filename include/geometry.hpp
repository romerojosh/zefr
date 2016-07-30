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

struct GeoStruct
{
    unsigned int nEles = 0; 
    unsigned int nBnds = 0;
    unsigned int nDims, nNodes, nFaces, shape_order, nFacesPerEle, nNodesPerEle, nNodesPerFace, nFptsPerFace;
    unsigned int nCornerNodes, nGfpts, nGfpts_int, nGfpts_bnd;
    bool per_bnd_flag = false;

    /* Connectivity Data */
    mdvector<int> ele2nodes, ele2face, face2nodes, face2eles, face2eles_idx;

    std::vector<unsigned int> bnd_ids;  //! List of boundary conditions for each boundary
    std::vector<unsigned int> ele_color_range, ele_color_nEles;
    mdvector<unsigned int> gfpt2bnd, per_fpt_list;
    std::map<std::vector<unsigned int>, int> bnd_faces, per_bnd_rot;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> per_bnd_pairs, face2ordered;
    std::unordered_map<unsigned int, unsigned int> per_fpt_pairs, per_node_pairs;
    mdvector<unsigned int> ppt_connect;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> coord_nodes, coord_spts, coord_fpts, coord_ppts, coord_qpts;
    mdvector<unsigned int> face_nodes;
    mdvector<int> ele_adj;

    unsigned int nColors;
    mdvector<unsigned int> ele_color;

    unsigned int nBounds;               //! Number of distinct mesh boundary regions
    std::map<unsigned int,int> bcIdMap; //! Map from Gmsh boundary ID to Flurry BC ID
    std::vector<std::string> bcNames;   //! Name of each boundary given in mesh file
    std::vector<unsigned int> bcType;   //! Boundary condition for each boundary face

    _mpi_comm myComm;
#ifdef _MPI
    unsigned int nGfpts_mpi;
    std::map<std::vector<unsigned int>, std::set<int>> mpi_faces;
    std::unordered_map<unsigned int, unsigned int> node_map_p2g, node_map_g2p;
    std::map<unsigned int, mdvector<unsigned int>> fpt_buffer_map;
    std::map<unsigned int, MPI_Datatype> mpi_types;

    unsigned int nMpiFaces;
    std::vector<int> procR, faceID_R, gIC_R, mpiLocF, mpiRotR, mpiLocF_R, mpiPeriodic;
#endif

#ifdef _GPU
    mdvector_gpu<int> fpt2gfpt_d, fpt2gfpt_slot_d;
    mdvector_gpu<unsigned int> gfpt2bnd_d, per_fpt_list_d;
    mdvector_gpu<double> coord_spts_d;
#ifdef _MPI
    std::map<unsigned int, mdvector_gpu<unsigned int>> fpt_buffer_map_d;
#endif
#endif

    /* --- Motion-Related Variables --- */
    mdvector<double> vel_nodes;  //! Grid velocity at all mesh nodes

    /* --- Overset-Related Variables --- */

    InputStruct *input;

    unsigned int nBndFaces, nIntFaces, nOverFaces;
    std::vector<std::vector<unsigned int>> bndPts;   //! List of points on each boundary
    std::vector<std::vector<unsigned int>> faceList; //! Ordered list of faces matching ele2face / face2eles
    std::map<std::vector<unsigned int>, unsigned int> nodes_to_face; //! Map from face nodes to face ID
    std::vector<int> fpt2face; //! fpt index to face index
    mdvector<int> face2fpts; //! Face index to fpt indices

    std::vector<int> iblank_node, iblank_face; //! iblank values for nodes, cells, faces
    mdvector<int> iblank_cell;

    std::vector<int> bndFaces, mpiFaces; //! Current list of all boundar & MPI faces
    std::set<int> overFaces;  //! Ordered list of all current overset faces
    std::vector<int> overFaceList;

    int nWall, nOver; //! Number of nodes on wall & overset boundaries
    std::vector<int> wallNodes, overNodes; //! Wall & overset boundary node lists

#ifdef _GPU
    mdvector_gpu<int> iblank_fpts_d, iblank_cell_d;
    mdvector<int> iblank_fpts;

    std::map<unsigned int,unsigned int> fpt2side; // For all overset fringe points
#endif

    unsigned int nGrids;  //! Number of distinct overset grids
    int nProcsGrid;       //! Number of MPI processes assigned to current (overset) grid block
    unsigned int gridID;  //! Which (overset) grid block is this process handling
    int gridRank;         //! MPI rank of process *within* the grid block [0 to nprocPerGrid-1]
    int rank;
    int nproc;

public:

};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims, _mpi_comm comm_in);
void load_mesh_data(InputStruct *input, GeoStruct &geo);
void read_boundary_ids(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_boundary_faces(std::ifstream &f, GeoStruct &geo);
void set_face_nodes(GeoStruct &geo);
void set_ele_adjacency(GeoStruct &geo);
void couple_periodic_bnds(GeoStruct &geo);
void setup_global_fpts(InputStruct *input, GeoStruct &geo, unsigned int order);
void pair_periodic_gfpts(GeoStruct &geo);
void setup_element_colors(InputStruct *input, GeoStruct &geo);
void shuffle_data_by_color(GeoStruct &geo);

#ifdef _MPI
void partition_geometry(InputStruct *input, GeoStruct &geo);
#endif

#endif /* geometry_hpp */
