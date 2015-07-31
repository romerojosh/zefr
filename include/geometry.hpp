#ifndef geometry_hpp
#define geometry_hpp

#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include "mdvector.hpp"

#ifdef _GPU
#include "mdvector_gpu.h"
#endif

struct GeoStruct
{
    unsigned int nEles = 0; 
    unsigned int nBnds = 0;
    unsigned int nDims, nNodes, shape_order, nFacesPerEle, nNodesPerEle, nNodesPerFace, nGfpts, nGfpts_int;
    bool per_bnd_flag = false;
    std::vector<unsigned int> bnd_ids;
    mdvector<unsigned int> gfpt2bnd, per_fpt_list;
    std::map<std::vector<unsigned int>, unsigned int> bnd_faces, per_bnd_rot;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> per_bnd_pairs, face2ordered;
    std::unordered_map<unsigned int, unsigned int> per_fpt_pairs, per_node_pairs;
    mdvector<unsigned int> nd2gnd, ppt_connect;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> coord_nodes, coord_spts, coord_fpts, coord_ppts, coord_qpts;

#ifdef _GPU
    mdvector_gpu<int> fpt2gfpt_d, fpt2gfpt_slot_d;
    mdvector_gpu<unsigned int> gfpt2bnd_d, per_fpt_list_d;
#endif
};

GeoStruct process_mesh(std::string meshfile, unsigned int order, unsigned int nDims);
void load_mesh_data(std::string meshfile, GeoStruct &geo);
void read_boundary_ids(std::ifstream &f, GeoStruct &geo);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo);
void read_boundary_faces(std::ifstream &f, GeoStruct &geo);
void couple_periodic_bnds(GeoStruct &geo);
void setup_global_fpts(GeoStruct &geo, unsigned int order);
void pair_periodic_gfpts(GeoStruct &geo);

#endif /* geometry_hpp */
