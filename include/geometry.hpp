/* Copyright (C) 2016 Aerospace Computing Laboratory (ACL).
 * See AUTHORS for contributors to this source code.
 *
 * This file is part of ZEFR.
 *
 * ZEFR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ZEFR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ZEFR.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    unsigned int nEles = 0; 
    unsigned int nBnds = 0;
    unsigned int nDims, nNodes, nFaces, shape_order, nFacesPerEle, nNodesPerEle, nNodesPerFace, nFptsPerFace;
    unsigned int nCornerNodes, nGfpts, nGfpts_int, nGfpts_bnd;
    unsigned int nGfpts_mpi = 0;
    bool per_bnd_flag = false;

    /* Connectivity Data */
    mdvector<int> ele2nodes, ele2face, face2nodes, face2eles, face2eles_idx;

    std::vector<unsigned int> bnd_ids;  //! List of boundary conditions for each boundary
    std::vector<unsigned int> ele_color_range, ele_color_nEles;
    mdvector<unsigned int> per_fpt_list;
    mdvector<char> gfpt2bnd;
    std::map<std::vector<unsigned int>, int> bnd_faces, per_bnd_rot;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> per_bnd_pairs, face2ordered;
    std::unordered_map<unsigned int, unsigned int> per_fpt_pairs, per_node_pairs;
    mdvector<unsigned int> ppt_connect;
    mdvector<int> fpt2gfpt;
    mdvector<char> fpt2gfpt_slot;
    mdvector<double> ele_nodes, coord_nodes, coord_spts, coord_fpts, coord_ppts, coord_qpts;
    mdvector<unsigned int> face_nodes;
    mdvector<int> ele_adj;

    mdvector<double> grid_vel_nodes, coords_init;

    unsigned int nColors;
    mdvector<unsigned int> ele_color;

    unsigned int nBounds;               //! Number of distinct mesh boundary regions
    std::map<unsigned int,int> bcIdMap; //! Map from Gmsh boundary ID to Flurry BC ID
    std::vector<std::string> bcNames;   //! Name of each boundary given in mesh file
    std::vector<unsigned int> bcType;   //! Boundary condition for each boundary face
    std::map<std::vector<unsigned int>, unsigned int> face2bnd;
    std::vector<std::vector<unsigned int>> boundFaces; //! List of face IDs for each mesh-defined boundary

    //! --- New additions for PyFR format (consider re-organizing) ---
    mdvector<face_con> face_list;
    std::vector<std::vector<face_con>> bound_faces;
#ifdef _MPI
    std::map<int,std::vector<face_con>> mpi_conn;
    std::vector<int> send_ranks;
#endif
    std::string mesh_uuid, config, stats;

    std::vector<int> pyfr2zefr_face, zefr2pyfr_face;

    _mpi_comm myComm;
#ifdef _MPI
    std::map<std::vector<unsigned int>, std::set<int>> mpi_faces;
    std::unordered_map<unsigned int, unsigned int> node_map_p2g, node_map_g2p;
    std::map<unsigned int, mdvector<unsigned int>> fpt_buffer_map;
    std::map<unsigned int, MPI_Datatype> mpi_types;

    unsigned int nMpiFaces;
    std::vector<int> procR, faceID_R, gIC_R, mpiLocF, mpiRotR, mpiLocF_R, mpiPeriodic;
#endif

#ifdef _GPU
    mdvector_gpu<int> fpt2gfpt_d;
    mdvector_gpu<char> fpt2gfpt_slot_d;
    mdvector_gpu<unsigned int> per_fpt_list_d;
    mdvector_gpu<char> gfpt2bnd_d;
    mdvector_gpu<double> coord_spts_d, coord_fpts_d;
    mdvector_gpu<double> coords_init_d, coord_nodes_d, grid_vel_nodes_d;
    mdvector_gpu<int> ele2nodes_d;
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
///    mdvector<int> c2f, f2c, c2c;            //! Cell-to-face and face-to-cell conncectivity
    std::vector<std::vector<unsigned int>> faceList; //! Ordered list of faces matching c2f / f2c
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
#endif

    unsigned int nGrids;  //! Number of distinct overset grids
    int nProcsGrid;       //! Number of MPI processes assigned to current (overset) grid block
    unsigned int gridID;  //! Which (overset) grid block is this process handling
    int gridRank;         //! MPI rank of process *within* the grid block [0 to nprocPerGrid-1]
    int rank;
    int nproc;
};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims, _mpi_comm comm_in);
void load_mesh_data_gmsh(InputStruct *input, GeoStruct &geo);
void load_mesh_data_pyfr(InputStruct *input, GeoStruct &geo);
void read_boundary_ids(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_boundary_faces(std::ifstream &f, GeoStruct &geo);
void set_face_nodes(GeoStruct &geo);
void set_ele_adjacency(GeoStruct &geo);
void couple_periodic_bnds(GeoStruct &geo);
void setup_global_fpts(InputStruct *input, GeoStruct &geo, unsigned int order);
void setup_global_fpts_pyfr(InputStruct *input, GeoStruct &geo, unsigned int order);
void pair_periodic_gfpts(GeoStruct &geo);
void setup_element_colors(InputStruct *input, GeoStruct &geo);
void shuffle_data_by_color(GeoStruct &geo);

#ifdef _MPI
void partition_geometry(InputStruct *input, GeoStruct &geo);
#endif

void move_grid(InputStruct *input, GeoStruct &geo, double time);

#endif /* geometry_hpp */
