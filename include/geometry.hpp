#ifndef geometry_hpp
#define geometry_hpp

#include <string>

#include "mdvector.hpp"

struct GeoStruct
{
    unsigned int nEles = 0; 
    unsigned int nBnds = 0;
    unsigned int nDims, nNodes, shape_order, nNodesPerEle, nGfpts;
    mdvector<unsigned int> nd2gnd, ppt_connect;
    mdvector<int> fpt2gfpt, fpt2gfpt_slot;
    mdvector<double> coord_nodes, coord_spts, coord_fpts, coord_ppts;
};

GeoStruct process_mesh(std::string meshfile, unsigned int order, unsigned int nDims);
void load_mesh_data(std::string meshfile, GeoStruct &geo);
void read_node_coords(std::ifstream &f, GeoStruct &geo);
void read_element_connectivity(std::ifstream &f, GeoStruct &geo);
void setup_global_fpts(GeoStruct &geo, unsigned int order);

#endif /* geometry_hpp */