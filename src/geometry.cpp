#include <algorithm>
#include <array>
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
  std::string param;

  /*
  while (f >> param)
  {
    if (param == "$Elements")
    {
      unsigned int val;
      f >> val >> val >> val;

      if (val == 2 || val == 3)
      {
        geo.nDims = 2; geo.shape_order = 1; geo.nNodesPerEle = 4;
      }
      else
      {
        ThrowException("3D not implemented yet!");
      }
    }
  }
*/

  /* Return to begining of file */
  //f.clear();
  //f.seekg(0, f.beg);

  /* Process file information */
  while (f >> param)
  {
    /* Load node coordinate data */
    if (param == "$Nodes")
      read_node_coords(f, geo);

    /* Load element connectivity data */
    if (param == "$Elements")
      read_element_connectivity(f, geo);
  }

  f.close();
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
  unsigned int nElesBnds;

  /* Get total number of elements and boundaries */
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
      if (val == 2 || val == 3)
        geo.nEles++;
      else if (val == 1)
        geo.nBnds++;
      else
        ThrowException("Inconsistent Element/Face type detected!");
    }
    else if (geo.nDims == 3)
    {
        ThrowException("3D not implemented yet!");
    }
    std::getline(f,line);
  }

  f.seekg(pos);


  geo.nd2gnd.assign({geo.nEles, geo.nNodesPerEle});

  unsigned int ele = 0;
  for (unsigned int n = 0; n < nElesBnds; n++)
  {
    unsigned int vint, ele_type;
    f >> vint >> ele_type;

    for (unsigned int n = 0; n < 3 ; n++)
      f >> vint;

    switch(ele_type)
    {
      case 1:
        f >> vint >> vint;
        break;
      case 2: /* 3-node Triangle */
        f >> geo.nd2gnd(ele,0) >> geo.nd2gnd(ele,1) >> geo.nd2gnd(ele,2);
        geo.nd2gnd(ele,3) = geo.nd2gnd(ele,2); 
        ele++; break;

      case 3: /* 4-node Quadrilateral */
        f >> geo.nd2gnd(ele,0) >> geo.nd2gnd(ele,1) >> geo.nd2gnd(ele,2) >> geo.nd2gnd(ele,3);
        ele++; break;
      default:
        ThrowException("Unrecognized element type detected!"); break;
    }
  }

  for (unsigned int ele = 0; ele < geo.nEles; ele++)
    for (unsigned int n = 0; n < geo.nNodesPerEle; n++)
      geo.nd2gnd(ele,n)--;

}

void setup_global_fpts(GeoStruct &geo, unsigned int order)
{
  /* Form set of unique faces */
  if (geo.nDims == 2)
  {
    unsigned int nFptsPerFace = order + 1;
    std::set<std::array<unsigned int, 2>> unique_faces;
    std::map<std::array<unsigned int, 2>, std::vector<unsigned int>> face_fpts;
    std::vector<std::vector<int>> ele2fpts(geo.nEles);
    std::vector<std::vector<int>> ele2fpts_slot(geo.nEles);

    /* Initialize global flux point index */
    unsigned int gfpt = 0;

    /* Begin loop through faces */
    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      std::array<unsigned int,2> face;
      ele2fpts[ele].assign(4*nFptsPerFace, -1);
      ele2fpts_slot[ele].assign(4*nFptsPerFace, -1);

      for (unsigned int n = 0; n < 4; n++)
      {
        face[0] = geo.nd2gnd(ele,n);
        face[1] = geo.nd2gnd(ele,(n+1)%4);
        std::sort(face.begin(), face.end());

        /* Check if face is collapsed */
        if (face[0] == face[1])
          continue;

        /* Check if face exists in unique set */
        if(!unique_faces.count(face))
        {
          std::vector<unsigned int> fpts(nFptsPerFace,0);
          for (auto &fpt : fpts)
          {
            fpt = gfpt;
            gfpt++;
          }

          face_fpts[face] = fpts;

          for (unsigned int i = 0; i < nFptsPerFace; i++)
          {
            ele2fpts[ele][n*nFptsPerFace + i] = fpts[i];
            ele2fpts_slot[ele][n*nFptsPerFace + i] = 0;
          }

          unique_faces.insert(face);
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

    /* Populate data structures */
    geo.nGfpts = gfpt;
    geo.fpt2gfpt.assign({geo.nEles, 4*nFptsPerFace});
    geo.fpt2gfpt_slot.assign({geo.nEles, 4*nFptsPerFace});

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < 4*nFptsPerFace; fpt++)
      {
        geo.fpt2gfpt(ele,fpt) = ele2fpts[ele][fpt];
        geo.fpt2gfpt_slot(ele,fpt) = ele2fpts_slot[ele][fpt];
      }
    }
  }
  else
  {
    ThrowException("3D not implemented!");
  }


}
