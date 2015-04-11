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
      geo.bnd_ids[n] = 1;
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
      if (val == 2 || val == 3)
      {
        geo.nEles++; 
        geo.shape_order = 1; geo.nNodesPerEle = 4;
      }
      else if (val == 1)
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
        ThrowException("3D not implemented yet!");
    }
    std::getline(f,line);
  }

  f.seekg(pos);

  /* Allocate memory for element connectivity */
  geo.nd2gnd.assign({geo.nEles, geo.nNodesPerEle});

  /* Read element connectivity (skip boundaries in this loop) */
  unsigned int ele = 0;
  for (unsigned int n = 0; n < nElesBnds; n++)
  {
    unsigned int vint, ele_type;
    f >> vint >> ele_type;

    for (unsigned int n = 0; n < 3 ; n++)
      f >> vint;

    if (geo.nDims == 2)
    {
      switch(ele_type)
      {
        case 1: /* 2-node Line (skip) */
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
    else
    {
      ThrowException("3D not implemented yet!");
    }
  }

  for (unsigned int ele = 0; ele < geo.nEles; ele++)
    for (unsigned int n = 0; n < geo.nNodesPerEle; n++)
      geo.nd2gnd(ele,n)--;

  /* Rewind file */
  f.seekg(pos);
}

void read_boundary_faces(std::ifstream &f, GeoStruct &geo)
{
  if (geo.nDims == 2)
  {
    //std::unordered_map<std::array<unsigned int, 2>, unsigned int> bnd_faces;
    std::vector<unsigned int> face(2,0);
    for (unsigned int n = 0; n < (geo.nEles + geo.nBnds); n++)
    {
      unsigned int vint, ele_type, bnd_id;
      std::string line;
      f >> vint >> ele_type;

      /* Get boundary id and face nodes */
      switch (ele_type)
      {
        case 1: /* 2-node line */
          f >> bnd_id;
          f >> vint >> vint;
          f >> face[0] >> face[1]; break;

        default:
          std::getline(f,line); continue; break;
      }

      /* Sort for consistency and add to map*/
      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = geo.bnd_ids[bnd_id];
    }
  }
  else
  {
    ThrowException("3D not implemented yet!");
  }
}

void setup_global_fpts(GeoStruct &geo, unsigned int order)
{
  /* Form set of unique faces */
  if (geo.nDims == 2)
  {
    unsigned int nFptsPerFace = order + 1;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> face_fpts;
    std::vector<std::vector<int>> ele2fpts(geo.nEles);
    std::vector<std::vector<int>> ele2fpts_slot(geo.nEles);

    std::vector<unsigned int> face(2,0);

    /* Determine number of interior global flux points */
    std::set<std::vector<unsigned int>> unique_faces;
    geo.nGfpts_int = 0;

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int n = 0; n < 4; n++)
      {
        face[0] = geo.nd2gnd(ele,n);
        face[1] = geo.nd2gnd(ele,(n+1)%4);
        std::sort(face.begin(), face.end());

        /* Check if face is collapsed */
        if (face[0] == face[1])
          continue;

        /* Check if face is not on boundary and not previously encountered*/
        if (!unique_faces.count(face) && !geo.bnd_faces.count(face))
          geo.nGfpts_int += nFptsPerFace;

        unique_faces.insert(face);
      }
    }

    /* Initialize global flux point index */
    unsigned int gfpt = 0; unsigned int gfpt_bnd = geo.nGfpts_int;

    /* Begin loop through faces */
    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      ele2fpts[ele].assign(4*nFptsPerFace, -1);
      ele2fpts_slot[ele].assign(4*nFptsPerFace, -1);

      for (unsigned int n = 0; n < 4; n++)
      {
        /* Get face nodes and sort for consistency */
        face[0] = geo.nd2gnd(ele,n);
        face[1] = geo.nd2gnd(ele,(n+1)%4);
        std::sort(face.begin(), face.end());

        /* Check if face is collapsed */
        if (face[0] == face[1])
          continue;

        /* Check if face has been encountered previously */
        std::vector<unsigned int> fpts(nFptsPerFace,0);
        if(!face_fpts.count(face))
        {
          if (geo.bnd_faces.count(face))
          {
            for (auto &fpt : fpts)
            {
              fpt = gfpt_bnd;
              gfpt_bnd++;
            }
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

    /* Populate data structures */
    geo.nGfpts = (gfpt + gfpt_bnd);
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
