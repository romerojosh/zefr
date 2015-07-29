#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "geometry.hpp"
#include "macros.hpp"
#include "mdvector.hpp"

GeoStruct process_mesh(std::string meshfile, unsigned int order, unsigned int nDims)
{
  GeoStruct geo;
  geo.nDims = nDims;

  load_mesh_data(meshfile, geo);

  setup_global_fpts(geo, order);

  return geo;

}

void load_mesh_data(std::string meshfile, GeoStruct &geo)
{
  std::ifstream f(meshfile);

  if (!f.is_open())
    ThrowException("Could not open specified mesh file!");

  std::string param;

  /* Process file information */
  while (f >> param)
  {
    /* Load boundary tags (must be before $Elements) */
    if (param == "$PhysicalNames")
      read_boundary_ids(f, geo);
    /* Load node coordinate data */
    if (param == "$Nodes")
      read_node_coords(f, geo);


    /* Load element connectivity data */
    if (param == "$Elements")
    {
      read_element_connectivity(f, geo);
      read_boundary_faces(f, geo);
    }
  }

  f.close();
}

void read_boundary_ids(std::ifstream &f, GeoStruct &geo)
{
  unsigned int nBndIds;
  f >> nBndIds;
  
  /* This is oversized to use gmsh boundary tag indices directy (1-indexed) */
  geo.bnd_ids.assign(nBndIds+1,0);
  std::string bnd_id;
  for (unsigned int n = 1; n < nBndIds+1; n++)
  {
    unsigned int val;
    f >> val >> val >> bnd_id;

    /* Check boundary tag and set appropriate index */
    if (bnd_id == "\"PERIODIC\"")
    {
      geo.bnd_ids[val] = 1;
      geo.per_bnd_flag = true;
    }
    else if (bnd_id == "\"FARFIELD\"" || bnd_id == "\"INLET_SUP\"")
    {
      geo.bnd_ids[val] = 2;
    }
    else if (bnd_id == "\"OUTLET_SUP\"")
    {
      geo.bnd_ids[val] = 3;
    }
    else if (bnd_id == "\"INLET_SUB\"")
    {
      geo.bnd_ids[val] = 4;
    }
    else if (bnd_id == "\"OUTLET_SUB\"")
    {
      geo.bnd_ids[val] = 5;
    }
    else if (bnd_id == "\"CHAR\"")
    {
      geo.bnd_ids[val] = 6;
    }
    else if (bnd_id == "\"SYMMETRY\"")
    {
      geo.bnd_ids[val] = 7;
    }
    else if (bnd_id == "\"WALL_SLIP\"")
    {
      geo.bnd_ids[val] = 8;
    }
    else if (bnd_id == "\"WALL_NS_ISO\"")
    {
      geo.bnd_ids[val] = 9;
    }
    else if (bnd_id == "\"WALL_NS_ISO_MOVE\"")
    {
      geo.bnd_ids[val] = 10;
    }
    else if (bnd_id == "\"WALL_NS_ADI\"")
    {
      geo.bnd_ids[val] = 11;
    }
    else if (bnd_id == "\"WALL_NS_ADI_MOVE\"")
    {
      geo.bnd_ids[val] = 12;
    }
    else if (bnd_id == "\"FLUID\"")
    {
    }
    else
    {
      ThrowException("Boundary type not recognized!");
    }
  }
}

void read_node_coords(std::ifstream &f, GeoStruct &geo)
{
      f >> geo.nNodes;
      geo.coord_nodes.assign({geo.nNodes, geo.nDims});
      for (unsigned int node = 0; node < geo.nNodes; node++)
      {
        unsigned int vint;
        double vdouble;
        f >> vint;
        if (geo.nDims == 2)
          f >> geo.coord_nodes(node,0) >> geo.coord_nodes(node,1) >> vdouble;
        else if (geo.nDims == 3)
          f >> geo.coord_nodes(node,0) >> geo.coord_nodes(node,1) >> geo.coord_nodes(node,2);
      }
 
}

void read_element_connectivity(std::ifstream &f, GeoStruct &geo)
{ 
  /* Get total number of elements and boundaries */
  unsigned int nElesBnds;
  f >> nElesBnds;
  auto pos = f.tellg();

  /* Determine number of elements */
  for (unsigned int n = 0; n < nElesBnds; n++)
  {
    unsigned int val;
    std::string line;
    f >> val >> val;
    if (geo.nDims == 2)
    {
      geo.nFacesPerEle = 4; geo.nNodesPerFace = 2;

      if (val == 2 || val == 3)
      {
        geo.nEles++; 
        geo.shape_order = 1; geo.nNodesPerEle = 4;
      }
      else if (val == 9 || val == 10)
      {
        geo.nEles++; 
        geo.shape_order = 2; geo.nNodesPerEle = 8;
      }
      else if (val == 1 || val == 8)
      {
        geo.nBnds++;
      }
      else
      {
        ThrowException("Inconsistent Element/Face type detected!");
      }
    }
    else if (geo.nDims == 3)
    {
      geo.nFacesPerEle = 6; geo.nNodesPerFace = 4;
      if (val == 2 || val == 3)
      {
        geo.nBnds++;
      }
      else if (val == 5 || val == 6)
      {
        geo.nEles++;
        geo.shape_order = 1; geo.nNodesPerEle = 8;
      }
    }
    std::getline(f,line);
  }

  f.seekg(pos);

  /* Allocate memory for element connectivity */
  geo.nd2gnd.assign({geo.nNodesPerEle, geo.nEles});

  /* Read element connectivity (skip boundaries in this loop) */
  unsigned int ele = 0;
  for (unsigned int n = 0; n < nElesBnds; n++)
  {
    unsigned int vint, ele_type;
    f >> vint >> ele_type;

    unsigned int nTags;
    f >> nTags; 

    for (unsigned int n = 0; n < nTags ; n++)
      f >> vint;

    if (geo.nDims == 2)
    {
      switch(ele_type)
      {
        case 1: /* 2-node Line (skip) */
          f >> vint >> vint;
          break;

        case 2: /* 3-node Triangle */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele);
          geo.nd2gnd(3,ele) = geo.nd2gnd(2,ele); 
          ele++; break;

        case 3: /* 4-node Quadrilateral */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          ele++; break;

        case 8: /* 3-node Line (skip) */
          f >> vint >> vint >> vint;
          break;

        case 9: /* 6-node Triangle */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele); 
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(7,ele);
          geo.nd2gnd(3,ele) = geo.nd2gnd(2,ele); geo.nd2gnd(6,ele) = geo.nd2gnd(2,ele);
          ele++; break;

        case 10: /* 9-node Quadilateral (read as 8-node) */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(6,ele) >> geo.nd2gnd(7,ele);
          f >> vint;
          ele++; break;

        default:
          ThrowException("Unrecognized element type detected!"); break;

      }
    }
    else
    {
      //ThrowException("3D not implemented yet!");

      switch(ele_type)
      {
        case 2: /* 3-node Triangle (skip)*/
          f >> vint >> vint >> vint;
          break;

        case 3: /* 4-node Quadrilateral (skip)*/
          f >> vint >> vint >> vint >> vint;
          break;

        case 5: /* 8-node Hexahedral */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(6,ele) >> geo.nd2gnd(7,ele);
          ele++; break;

        case 6: /* 6-node Prism */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele);
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(6,ele);
          geo.nd2gnd(3,ele) = geo.nd2gnd(2,ele);
          geo.nd2gnd(7,ele) = geo.nd2gnd(6,ele);
          ele++; break;

        default:
          ThrowException("Unrecognized element type detected!"); break;
      }

    }
  }

  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    //std::cout << ele << " ";
    for (unsigned int n = 0; n < geo.nNodesPerEle; n++)
    {
      //std::cout << geo.nd2gnd(n,ele) << " ";
      geo.nd2gnd(n,ele)--;
    }
    //std::cout << std::endl;
  }

  /* Rewind file */
  f.seekg(pos);
}

void read_boundary_faces(std::ifstream &f, GeoStruct &geo)
{
  if (geo.nDims == 2)
  {
    //std::unordered_map<std::array<unsigned int, 2>, unsigned int> bnd_faces;
    std::vector<unsigned int> face(geo.nNodesPerFace,0);
    for (unsigned int n = 0; n < (geo.nEles + geo.nBnds); n++)
    {
      unsigned int vint, ele_type, bnd_id, nTags;
      std::string line;
      f >> vint >> ele_type;

      /* Get boundary id and face nodes */
      switch (ele_type)
      {
        case 1: /* 2-node line */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          f >> face[0] >> face[1]; break;

        case 8: /* 3-node Line */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          f >> face[0] >> face[1] >> vint; break;

        default:
          std::getline(f,line); continue; break;
      }

      face[0]--; face[1]--;

      /* Sort for consistency and add to map*/
      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = geo.bnd_ids[bnd_id];
    }
  }
  else if (geo.nDims == 3)
  {
    std::vector<unsigned int> face(geo.nNodesPerFace,0);
    for (unsigned int n = 0; n < (geo.nEles + geo.nBnds); n++)
    {
      unsigned int vint, ele_type, bnd_id, nTags;
      std::string line;
      f >> vint >> ele_type;

      /* Get boundary id and face nodes */
      switch (ele_type)
      {
        case 2: /* 3-node Triangle */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          f >> face[0] >> face[1] >> face[2]; 
          face[3] = face[2]; break;

        case 3: /* 4-node Quadrilateral */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          f >> face[0] >> face[1] >> face[2] >> face[3]; break;

        default:
          std::getline(f,line); continue; break;
      }

      face[0]--; face[1]--; face[2]--; face[3]--;

      /* Sort for consistency and add to map*/
      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = geo.bnd_ids[bnd_id];
    }
 
    //ThrowException("3D not implemented yet!");

  }

  if (geo.per_bnd_flag)
    couple_periodic_bnds(geo);
}

void couple_periodic_bnds(GeoStruct &geo)
{
  mdvector<double> coords_face1, coords_face2;
  coords_face1.assign({geo.nNodesPerFace,geo.nDims});
  coords_face2.assign({geo.nNodesPerFace,geo.nDims});
  /* Loop over boundary faces */
  for (auto &bnd_face : geo.bnd_faces)
  {
    unsigned int bnd_id = bnd_face.second;
    auto face = bnd_face.first;
    
    /* Check if face is periodic */
    if (bnd_id == 1)
    {
      /* Get face node coordinates */
      for (unsigned int node = 0; node < geo.nNodesPerFace; node++)
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
          coords_face1(node, dim) = geo.coord_nodes(face[node], dim);

      /* Compute centroid location */
      std::vector<double> c1(geo.nDims, 0.0);
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        for (unsigned int node = 0; node < geo.nNodesPerFace; node++)
        {
          c1[dim] += coords_face1(node, dim);
        }

        c1[dim] /= geo.nNodesPerFace;
      }

      /* Search for face to couple */
      for(auto &bnd_face2 : geo.bnd_faces)
      {
        auto face2 = bnd_face2.first;
        if (face2 == face)
          continue;

        /* Get face node coordinates */
        for (unsigned int node = 0; node < geo.nNodesPerFace; node++)
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            coords_face2(node, dim) = geo.coord_nodes(face2[node], dim);

        /* Compute centroid location */
        std::vector<double> c2(geo.nDims, 0.0);
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          for (unsigned int node = 0; node < geo.nNodesPerFace; node++)
          {
            c2[dim] += coords_face2(node, dim);
          }

          c2[dim] /= geo.nNodesPerFace;
        }


        /* Compare centroid locations to couple faces */
        unsigned int count = 0;
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          if (std::abs(c1[dim] - c2[dim]) < 1.e-6)
          {
            bool onPlane = true;
            double coord = coords_face1(0,dim);

            for (unsigned int node = 1; node < geo.nNodesPerFace; node++)
            {
              if (std::abs(coord - coords_face1(node, dim)) > 1e-6)
                onPlane = false;
            }

            if (!onPlane)
              count++;
          }
        }

        if (count == geo.nDims - 1)
        {
          geo.per_bnd_pairs[face] = face2;
        }
      }
    }
  }

  /*
  std::cout << "per_bnd_pairs" << std::endl;
  for (auto &pair : geo.per_bnd_pairs)
  {
    std::cout << pair.first[0] << " " << pair.first[1] << " -> " << pair.second[0] << " " << pair.second[1] << std::endl;
  }
  */
}

void setup_global_fpts(GeoStruct &geo, unsigned int order)
{
  /* Form set of unique faces */
  if (geo.nDims == 2 || geo.nDims == 3)
  {
    unsigned int nFptsPerFace = (order + 1);
    if (geo.nDims == 3)
      nFptsPerFace *= (order + 1);

    std::map<std::vector<unsigned int>, std::vector<unsigned int>> face_fpts;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> bndface2fpts;
    std::vector<std::vector<int>> ele2fpts(geo.nEles);
    std::vector<std::vector<int>> ele2fpts_slot(geo.nEles);

    std::vector<unsigned int> face(geo.nNodesPerFace,0);

    /* Define node indices for faces */
    std::vector<std::vector<unsigned int>> faces_nodes(geo.nFacesPerEle);

    for (auto &nodes : faces_nodes)
    {
      nodes.assign(geo.nNodesPerFace,0);
    }

    if (geo.nDims == 2)
    {
      /* Face 0: Bottom */
      faces_nodes[0][0] = 0; faces_nodes[0][1] = 1;

      /* Face 1: Right */
      faces_nodes[1][0] = 1; faces_nodes[1][1] = 2;

      /* Face 2: Top */
      faces_nodes[2][0] = 2; faces_nodes[2][1] = 3;

      /* Face 3: Left */
      faces_nodes[3][0] = 3; faces_nodes[3][1] = 0;

    }
    else if (geo.nDims == 3)
    {
      /* Face 0: Bottom */
      faces_nodes[0][0] = 0; faces_nodes[0][1] = 1;
      faces_nodes[0][2] = 2; faces_nodes[0][3] = 3;

      /* Face 1: Top */
      faces_nodes[1][0] = 4; faces_nodes[1][1] = 5;
      faces_nodes[1][2] = 6; faces_nodes[1][3] = 7;

      /* Face 2: Left */
      faces_nodes[2][0] = 3; faces_nodes[2][1] = 0;
      faces_nodes[2][2] = 4; faces_nodes[2][3] = 7;

      /* Face 3: Right */
      faces_nodes[3][0] = 2; faces_nodes[3][1] = 1;
      faces_nodes[3][2] = 5; faces_nodes[3][3] = 6;

      /* Face 4: Front */
      faces_nodes[4][0] = 0; faces_nodes[4][1] = 1;
      faces_nodes[4][2] = 5; faces_nodes[4][3] = 4;

      /* Face 5: Back */
      faces_nodes[5][0] = 7; faces_nodes[5][1] = 6;
      faces_nodes[5][2] = 3; faces_nodes[5][3] = 2;
    }

    /* Determine number of interior global flux points */
    std::set<std::vector<unsigned int>> unique_faces;
    geo.nGfpts_int = 0;

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
      {
        for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
        {
          face[i] = geo.nd2gnd(faces_nodes[n][i], ele);
        }

        /* Check if face is collapsed */
        if (face[geo.nNodesPerFace - 2] == face[geo.nNodesPerFace - 1])
          continue;

        std::sort(face.begin(), face.end());

        /* Check if face is not on boundary and not previously encountered */
        if (!unique_faces.count(face) && !geo.bnd_faces.count(face))
          geo.nGfpts_int += nFptsPerFace;

        unique_faces.insert(face);
      }
    }

    /* Initialize global flux point indicies (to place boundary fpts at end of global fpt data structure) */
    unsigned int gfpt = 0; unsigned int gfpt_bnd = geo.nGfpts_int;

    /* Begin loop through faces */
    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      ele2fpts[ele].assign(geo.nFacesPerEle * nFptsPerFace, -1);
      ele2fpts_slot[ele].assign(geo.nFacesPerEle * nFptsPerFace, -1);

      for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
      {
        /* Get face nodes and sort for consistency */
        for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
        {
          face[i] = geo.nd2gnd(faces_nodes[n][i], ele);
        }

        /* Check if face is collapsed */
        if (face[geo.nNodesPerFace - 2] == face[geo.nNodesPerFace - 1])
          continue;

        std::sort(face.begin(), face.end());

        /* Check if face has been encountered previously */
        std::vector<unsigned int> fpts(nFptsPerFace,0);
        if(!face_fpts.count(face))
        {
          /* Check if face is on boundary */
          if (geo.bnd_faces.count(face))
          {
            unsigned int bnd_id = geo.bnd_faces[face];
            for (auto &fpt : fpts)
            {
              geo.gfpt2bnd.push_back(bnd_id);
              fpt = gfpt_bnd;
              gfpt_bnd++;
            }

            bndface2fpts[face] = fpts;
            
          }
          else
          {
            for (auto &fpt : fpts)
            {
              fpt = gfpt;
              gfpt++;
            }
          }

          face_fpts[face] = fpts;

          for (unsigned int i = 0; i < nFptsPerFace; i++)
          {
            ele2fpts[ele][n*nFptsPerFace + i] = fpts[i];
            ele2fpts_slot[ele][n*nFptsPerFace + i] = 0;
          }
        }
        else
        {
          auto fpts = face_fpts[face];
          for (unsigned int i = 0; i < nFptsPerFace; i++)
          {
            ele2fpts[ele][n*nFptsPerFace + i] = fpts[nFptsPerFace-1-i];
            ele2fpts_slot[ele][n*nFptsPerFace + i] = 1;
          }
        }
      }
    }

    /* Pair up periodic flux points if needed */ 
    if (geo.per_bnd_flag)
    {
      /* Creating simple vector of flux point pairs to replace map, since it
       * cannot be used for GPU */
      geo.per_fpt_list.assign({gfpt_bnd - geo.nGfpts_int}, 0);

      for (auto &bnd_face1 : bndface2fpts)
      {
        auto face1 = bnd_face1.first;
        auto fpts1 = bnd_face1.second;

        /* If boundary face is not periodic, skip this pairing */
        if (geo.bnd_faces[face1] != 1)
          continue;

        auto face2 = geo.per_bnd_pairs[face1];
        auto fpts2 = bndface2fpts[face2];

        for (unsigned int i = 0; i < nFptsPerFace; i++)
        {
          geo.per_fpt_pairs[fpts1[i]] = fpts2[nFptsPerFace - 1 - i];
          geo.per_fpt_list(fpts1[i] - geo.nGfpts_int) = fpts2[nFptsPerFace - 1 - i];
        }
      } 

    }

    for (unsigned int i = 0; i < gfpt_bnd - geo.nGfpts_int; i++)
    {
      std::cout << i + geo.nGfpts_int << " " << geo.per_fpt_list(i) << std::endl;
    }

    /* Populate data structures */
    geo.nGfpts = gfpt_bnd;
    std::cout << geo.nGfpts << std::endl;

    geo.fpt2gfpt.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});
    geo.fpt2gfpt_slot.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEle * nFptsPerFace; fpt++)
      {
        geo.fpt2gfpt(fpt,ele) = ele2fpts[ele][fpt];
        geo.fpt2gfpt_slot(fpt,ele) = ele2fpts_slot[ele][fpt];
      }
    }

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      std::cout << ele;
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEle * nFptsPerFace; fpt++)
      {
        std::cout << " " << geo.fpt2gfpt(fpt, ele);
      }
      std::cout << std::endl;
    }
  }
  else
  {
    ThrowException("nDims is not valid!");
  }


}

