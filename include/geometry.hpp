#ifndef geometry_hpp
#define geometry_hpp

#include <map>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

#include "input.hpp"
#include "mdvector.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

struct GeoStruct
{
    unsigned int nEles = 0; 
    unsigned int nBnds = 0;
    unsigned int nDims, nNodes, shape_order, nFacesPerEle, nNodesPerEle, nNodesPerFace;
    unsigned int nCornerNodes, nGfpts, nGfpts_int, nGfpts_bnd;
    bool per_bnd_flag = false;
    std::vector<unsigned int> bnd_ids;
    mdvector<unsigned int> gfpt2bnd, per_fpt_list;
    std::map<std::vector<unsigned int>, unsigned int> bnd_faces, per_bnd_rot;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> per_bnd_pairs, face2ordered;
    std::unordered_map<unsigned int, unsigned int> per_fpt_pairs, per_node_pairs;
    mdvector<unsigned int> nd2gnd, ppt_connect;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> coord_nodes, coord_spts, coord_fpts, coord_ppts, coord_qpts;
    mdvector<unsigned int> face_nodes;

#ifdef _MPI
    unsigned int nGfpts_mpi;
    std::map<std::vector<unsigned int>, std::set<int>> mpi_faces;
    std::unordered_map<unsigned int, unsigned int> node_map_p2g, node_map_g2p;
    std::map<unsigned int, mdvector<unsigned int>> fpt_buffer_map;
#endif

#ifdef _GPU
    mdvector_gpu<int> fpt2gfpt_d, fpt2gfpt_slot_d;
    mdvector_gpu<unsigned int> gfpt2bnd_d, per_fpt_list_d;
    mdvector_gpu<double> coord_spts_d;
#ifdef _MPI
    std::map<unsigned int, mdvector_gpu<unsigned int>> fpt_buffer_map_d;
#endif
#endif
};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims);
void load_mesh_data(InputStruct *input, GeoStruct &geo);
void read_boundary_ids(std::ifstream &f, GeoStruct &geo);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo, InputStruct *input);
void read_boundary_faces(std::ifstream &f, GeoStruct &geo);
void set_face_nodes(GeoStruct &geo);
void couple_periodic_bnds(GeoStruct &geo);
void setup_global_fpts(GeoStruct &geo, unsigned int order);
void pair_periodic_gfpts(GeoStruct &geo);

#ifdef _MPI
void partition_geometry(GeoStruct &geo);
#endif

#endif /* geometry_hpp */
