#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

#ifdef _MPI
#include <unistd.h>
#include "mpi.h"
#include "metis.h"
#endif

#include "geometry.hpp"
#include "macros.hpp"
#include "mdvector.hpp"

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims)
{
  GeoStruct geo;
  geo.nDims = nDims;

  load_mesh_data(input, geo);

#ifdef _MPI
  partition_geometry(geo);
#endif

  setup_global_fpts(input, geo, order);

  if (input->dt_scheme == "LUSGS" or input->dt_scheme == "LUJac")
  {
    setup_element_colors(input, geo);
  }


  return geo;

}

void load_mesh_data(InputStruct *input, GeoStruct &geo)
{
  std::ifstream f(input->meshfile);

  if (!f.is_open())
    ThrowException("Could not open specified mesh file!");

  std::string param;

  /* Process file information */
  /* Load boundary tags */
  while (f >> param)
  {
    if (param == "$PhysicalNames")
    {
      read_boundary_ids(f, geo); break;
    }
  }

  if (f.eof()) ThrowException("Meshfile missing $PhysicalNames tag");

  f.clear();
  f.seekg(0, f.beg);

  /* Load node coordinate data */
  while (f >> param)
  {
    if (param == "$Nodes")
    {
      read_node_coords(f, geo); break;
    }
  }

  if (f.eof()) ThrowException("Meshfile missing $Nodes tag");

  f.clear();
  f.seekg(0, f.beg);

  while (f >> param)
  {
    /* Load element connectivity data */
    if (param == "$Elements")
    {
      read_element_connectivity(f, geo, input);
      read_boundary_faces(f, geo);
      break;
    }
  }

  if (f.eof()) ThrowException("Meshfile missing $Elements tag");

  set_face_nodes(geo);

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
      ThrowException("Boundary type " + bnd_id + " not recognized!");
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

void read_element_connectivity(std::ifstream &f, GeoStruct &geo, InputStruct *input)
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
      geo.nCornerNodes = 4;

      switch(val)
      {
        /* Linear quad/tri */
        case 2:
        case 3:
          geo.nEles++; 
          geo.shape_order = 1; geo.nNodesPerEle = 4; break;

        /* Biquadratic quad/tri */
        case 9:
        case 10:
          geo.nEles++; 
          geo.shape_order = 2; 
          if (input->serendipity)
            geo.nNodesPerEle = 8;
          else
            geo.nNodesPerEle = 9;
          break;

        /* Bicubic quad */
        case 36:
          geo.nEles++; 
          geo.shape_order = 3; geo.nNodesPerEle = 16; break;

        /* Biquartic quad */
        case 37:
          geo.nEles++; 
          geo.shape_order = 4; geo.nNodesPerEle = 25; break;

        /* Biquintic quad */
        case 38:
          geo.nEles++; 
          geo.shape_order = 5; geo.nNodesPerEle = 36; break;

        case 1:
        case 8:
        case 26:
        case 27:
        case 28:
          geo.nBnds++; break;

        default:
          ThrowException("Inconsistent Element/Face type detected! Is nDims set correctly?");
      }
    }
    else if (geo.nDims == 3)
    {
      geo.nFacesPerEle = 6; geo.nNodesPerFace = 4;
      geo.nCornerNodes = 8;

      switch(val)
      {
        /* Trilinear Hex/Tet/Prism/Pyramid */
        case 4:
        case 5:
        case 6:
        case 7:
          geo.nEles++;
          geo.shape_order = 1; geo.nNodesPerEle = 8; break;

        case 11:
        case 12:
          geo.nEles++;
          geo.shape_order = 2; geo.nNodesPerEle = 20; break;

        case 2:
        case 3:
        case 9:
        case 10:
          geo.nBnds++; break;

        default:
          ThrowException("Inconsistent Element/Face type detected! Is nDims set correctly?");
      }
    }
    std::getline(f,line);
  }

  f.seekg(pos);

  /* Allocate memory for element connectivity */
  geo.nd2gnd.assign({geo.nNodesPerEle, geo.nEles});

  /* Read element connectivity (skip boundaries in this loop) */
  unsigned int ele = 0;
  std::string line;
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
        case 8: /* 3-node Line (skip) */
        case 26: /* 4-node Line (skip) */
        case 27: /* 5-node Line (skip) */
        case 28: /* 6-node Line (skip) */
          std::getline(f,line); break;

        case 2: /* 3-node Triangle */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele);
          geo.nd2gnd(3,ele) = geo.nd2gnd(2,ele); 
          ele++; break;

        case 3: /* 4-node Quadrilateral */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          ele++; break;

        case 9: /* 6-node Triangle */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele); 
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(7,ele);
          geo.nd2gnd(3,ele) = geo.nd2gnd(2,ele); geo.nd2gnd(6,ele) = geo.nd2gnd(2,ele);

          if (!input->serendipity)
          {
            //TODO set geo.nd2gnd(8,ele) to centroid
            ThrowException("Biquadratic quad to triangles not implemented yet! Set serendipity = 1!");
          }

          ele++; break;

        case 10: /* 9-node Quadilateral */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(6,ele) >> geo.nd2gnd(7,ele);
          if (!input->serendipity)
            f >> geo.nd2gnd(8,ele);
          else
            f >> vint;
          ele++; break;

        case 36: /* 16-node Quadilateral */
          for (int n = 0; n < 16; n++)
            f >> geo.nd2gnd(n, ele);
          ele++; break;

        case 37: /* 25-node Quadilateral */
          for (int n = 0; n < 25; n++)
            f >> geo.nd2gnd(n, ele);
          ele++; break;

        case 38: /* 36-node Quadilateral */
          for (int n = 0; n < 36; n++)
            f >> geo.nd2gnd(n, ele);
          ele++; break;

        default:
          ThrowException("Unrecognized element type detected!"); break;

      }
    }
    else
    {
      switch(ele_type)
      {
        case 2: /* 3-node Triangle (skip)*/
        case 3: /* 4-node Quadrilateral (skip)*/
        case 9: /* 6-node Triangle (skip) */
        case 10: /* 9-node Quadrilateral (skip) */
          std::getline(f,line); break;

        case 4: /* 4-node Tetrahedral */
        {
          /* Selecting collapsed nodes per Hesthaven's thesis. Works for non-periodic
           * fully tetrahedral meshes. */
          std::vector<unsigned int> nodes(4,0);

          for (unsigned int i = 0; i < 4; i++)
          {
            f >> nodes[i];
          }

          /* Locate minimum node index and position */
          auto it_min = std::min_element(nodes.begin(), nodes.end());
          auto min_node = *it_min;

          unsigned int min_pos = 0;
          for (unsigned int i = 0; i < 4; i++)
          {
            if (nodes[i] == min_node)
              min_pos = i;
          }

          /* Set minimum node to "top" collapsed node */
          geo.nd2gnd(4, ele) = min_node;
          geo.nd2gnd(5, ele) = min_node;
          geo.nd2gnd(6, ele) = min_node;
          geo.nd2gnd(7, ele) = min_node;

          nodes.erase(it_min);

          /* Find next minimum node */
          it_min = std::min_element(nodes.begin(), nodes.end());
          min_node = *it_min;


          /* Rotate base nodes so that second minimum is "bottom" collapsed node. 
           * Reorder for CCW orientation if needed */
          while (nodes[2] != min_node)
          {
            std::rotate(nodes.begin(), nodes.begin() + 1, nodes.end());
          }
          if (min_pos == 0 || min_pos == 2)
          {
            geo.nd2gnd(0, ele) = nodes[1];
            geo.nd2gnd(1, ele) = nodes[0];
            geo.nd2gnd(2, ele) = nodes[2];
            geo.nd2gnd(3, ele) = nodes[2];
          }
          else if (min_pos == 1 || min_pos == 3)
          {
            geo.nd2gnd(0, ele) = nodes[0];
            geo.nd2gnd(1, ele) = nodes[1];
            geo.nd2gnd(2, ele) = nodes[2];
            geo.nd2gnd(3, ele) = nodes[2];
          }

          ele++; break;
        }

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

        case 7: /* 5-node Pyramid */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1, ele) >> geo.nd2gnd(2, ele);
          f >> geo.nd2gnd(3, ele) >> geo.nd2gnd(4,ele);
          geo.nd2gnd(5, ele) = geo.nd2gnd(4, ele);
          geo.nd2gnd(6, ele) = geo.nd2gnd(4, ele);
          geo.nd2gnd(7, ele) = geo.nd2gnd(4, ele);
          ele++; break;

        case 11: /* 10-node Tetrahedron (read as collapsed 20-node serendipity) */
        {
          /* Selecting collapsed nodes per Hesthaven's thesis. Works for non-periodic
           * fully tetrahedral meshes. */
          std::vector<unsigned int> nodes(10,0);
          std::vector<unsigned int> verts(4,0);

          for (unsigned int i = 0; i < 10; i++)
          {
            f >> nodes[i];
            //nodes[i] = i;
          }

          for (unsigned int i = 0; i < 4; i++)
          {
            verts[i] = nodes[i];
          }

          /* Locate minimum vertex index and position */
          auto it_min = std::min_element(verts.begin(), verts.end());
          auto min_vert = *it_min;

          unsigned int min_pos = 0;
          for (unsigned int i = 0; i < 4; i++)
          {
            if (verts[i] == min_vert)
              min_pos = i;
          }

          std::cout << ele << " " << min_pos << std::endl;

          /* Set minimum node to "top" collapsed node */
          geo.nd2gnd(4, ele) = min_vert; geo.nd2gnd(5, ele) = min_vert;
          geo.nd2gnd(6, ele) = min_vert; geo.nd2gnd(7, ele) = min_vert;
          geo.nd2gnd(16,ele) =  min_vert; geo.nd2gnd(17,ele) = min_vert;
          geo.nd2gnd(18,ele) =  min_vert; geo.nd2gnd(19,ele) = min_vert;

          verts.erase(it_min);

          /* Get bottom and middle edge vertices, based on min_pos */
          std::vector<unsigned int> bverts(3,0);
          std::vector<unsigned int> mverts(3,0);
          if (min_pos == 0)
          {
            bverts = {nodes[5], nodes[8], nodes[9]};
            mverts = {nodes[4], nodes[6], nodes[7]};
          }
          else if (min_pos == 1)
          {
            bverts = {nodes[6], nodes[8], nodes[7]};
            mverts = {nodes[4], nodes[5], nodes[9]};
          }
          else if (min_pos == 2)
          {
            bverts = {nodes[4], nodes[9], nodes[7]};
            mverts = {nodes[6], nodes[5], nodes[8]};
          }
          else
          {
            bverts = {nodes[4], nodes[5], nodes[6]};
            mverts = {nodes[7], nodes[9], nodes[8]};
          }

          /* Find next minimum vertex */
          it_min = std::min_element(verts.begin(), verts.end());
          min_vert = *it_min;


          /* Rotate base nodes so that second minimum is "bottom" collapsed node. 
           * Reorder for CCW orientation if needed */
          while (verts[2] != min_vert)
          {
            std::rotate(verts.begin(), verts.begin() + 1, verts.end());
            std::rotate(bverts.begin(), bverts.begin() + 1, bverts.end());
            std::rotate(mverts.begin(), mverts.begin() + 1, mverts.end());
          }
          if (min_pos == 0 || min_pos == 2)
          {
            geo.nd2gnd(0, ele) = verts[1];
            geo.nd2gnd(1, ele) = verts[0];
            geo.nd2gnd(2, ele) = verts[2];
            geo.nd2gnd(3, ele) = verts[2];

            geo.nd2gnd(8,ele) =  bverts[0]; geo.nd2gnd(9,ele) = bverts[2];
            geo.nd2gnd(10,ele) = verts[2]; geo.nd2gnd(11,ele) = bverts[1];

            geo.nd2gnd(12,ele) =  mverts[1]; geo.nd2gnd(13,ele) = mverts[0];
            geo.nd2gnd(14,ele) =  mverts[2]; geo.nd2gnd(15,ele) = mverts[2];
          }
          else if (min_pos == 1 || min_pos == 3)
          {
            geo.nd2gnd(0, ele) = verts[0];
            geo.nd2gnd(1, ele) = verts[1];
            geo.nd2gnd(2, ele) = verts[2];
            geo.nd2gnd(3, ele) = verts[2];

            geo.nd2gnd(8,ele) =  bverts[0]; geo.nd2gnd(9,ele) = bverts[1];
            geo.nd2gnd(10,ele) = verts[2]; geo.nd2gnd(11,ele) = bverts[2];

            geo.nd2gnd(12,ele) =  mverts[0]; geo.nd2gnd(13,ele) = mverts[1];
            geo.nd2gnd(14,ele) =  mverts[2]; geo.nd2gnd(15,ele) = mverts[2];
          }

          ele++; break;
        }

        case 12: /* Triquadratic Hex (read as 20-node serendipity) */
          f >> geo.nd2gnd(0,ele) >> geo.nd2gnd(1,ele) >> geo.nd2gnd(2,ele) >> geo.nd2gnd(3,ele);
          f >> geo.nd2gnd(4,ele) >> geo.nd2gnd(5,ele) >> geo.nd2gnd(6,ele) >> geo.nd2gnd(7,ele);
          f >> geo.nd2gnd(8,ele) >> geo.nd2gnd(11,ele) >> geo.nd2gnd(12,ele) >> geo.nd2gnd(9,ele);
          f >> geo.nd2gnd(13,ele) >> geo.nd2gnd(10,ele) >> geo.nd2gnd(14,ele) >> geo.nd2gnd(15,ele);
          f >> geo.nd2gnd(16,ele) >> geo.nd2gnd(19,ele) >> geo.nd2gnd(17,ele) >> geo.nd2gnd(18,ele);
          std::getline(f,line); ele++; break;


        default:
          ThrowException("Unrecognized element type detected!"); break;
      }

    }
  }

  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    for (unsigned int n = 0; n < geo.nNodesPerEle; n++)
    {
      geo.nd2gnd(n,ele)--;
    }
  }

  /* Rewind file */
  f.seekg(pos);
}

void read_boundary_faces(std::ifstream &f, GeoStruct &geo)
{
  if (geo.nDims == 2)
  {
    std::vector<unsigned int> face(geo.nNodesPerFace,0);
    for (unsigned int n = 0; n < (geo.nEles + geo.nBnds); n++)
    {
      unsigned int vint, ele_type, bnd_id, nTags;
      std::string line;
      f >> vint >> ele_type;

      /* Get boundary id and face nodes */
      f >> nTags;
      f >> bnd_id;

      for (unsigned int i = 0; i < nTags - 1; i++)
        f >> vint;

      switch (ele_type)
      {
        case 1: /* 2-node line */
        case 8: /* 3-node Line */
        case 26: /* 4-node Line (skip) */
        case 27: /* 5-node Line (skip) */
        case 28: /* 6-node Line (skip) */
          f >> face[0] >> face[1]; 
          std::getline(f,line); break;

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
    std::vector<unsigned int> face;
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

          
          face.assign(3,0);
          f >> face[0] >> face[1] >> face[2]; 
          break;

        case 3: /* 4-node Quadrilateral */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          face.assign(4,0);
          f >> face[0] >> face[1] >> face[2] >> face[3]; break;

        case 9: /* 6-node Triangle */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          face.assign(3,0);
          f >> face[0] >> face[1] >> face[2]; 
          std::getline(f,line);
          break;


        case 10: /* 9-node Quadrilateral */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          face.assign(4,0);
          f >> face[0] >> face[1] >> face[2] >> face[3];
          std::getline(f,line);
          break;


        default:
          std::getline(f,line); continue; break;
      }

      for (auto &node : face)
        node--;

      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = geo.bnd_ids[bnd_id];
      
    }

  }

}

void set_face_nodes(GeoStruct &geo)
{
  /* Define node indices for faces */
  geo.face_nodes.assign({geo.nFacesPerEle, geo.nNodesPerFace}, 0);


  if (geo.nDims == 2)
  {
    /* Face 0: Bottom */
    geo.face_nodes(0, 0) = 0; geo.face_nodes(0, 1) = 1;

    /* Face 1: Right */
    geo.face_nodes(1, 0) = 1; geo.face_nodes(1, 1) = 2;

    /* Face 2: Top */
    geo.face_nodes(2, 0) = 2; geo.face_nodes(2, 1) = 3;

    /* Face 3: Left */
    geo.face_nodes(3, 0) = 3; geo.face_nodes(3, 1) = 0;

  }
  else if (geo.nDims == 3)
  {
    /* Face 0: Bottom */
    geo.face_nodes(0, 0) = 0; geo.face_nodes(0, 1) = 1;
    geo.face_nodes(0, 2) = 2; geo.face_nodes(0, 3) = 3;

    /* Face 1: Top */
    geo.face_nodes(1, 0) = 5; geo.face_nodes(1, 1) = 4;
    geo.face_nodes(1, 2) = 7; geo.face_nodes(1, 3) = 6;

    /* Face 2: Left */
    geo.face_nodes(2, 0) = 0; geo.face_nodes(2, 1) = 3;
    geo.face_nodes(2, 2) = 7; geo.face_nodes(2, 3) = 4;

    /* Face 3: Right */
    geo.face_nodes(3, 0) = 2; geo.face_nodes(3, 1) = 1;
    geo.face_nodes(3, 2) = 5; geo.face_nodes(3, 3) = 6;

    /* Face 4: Front */
    geo.face_nodes(4, 0) = 1; geo.face_nodes(4, 1) = 0;
    geo.face_nodes(4, 2) = 4; geo.face_nodes(4, 3) = 5;

    /* Face 5: Back */
    geo.face_nodes(5, 0) = 3; geo.face_nodes(5, 1) = 2;
    geo.face_nodes(5, 2) = 6; geo.face_nodes(5, 3) = 7;
  }
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
    auto face1 = bnd_face.first;
    unsigned int nNodesPerFace = (unsigned int) face1.size();
    
    /* Check if face is periodic */
    if (bnd_id == 1)
    {
      /* Get face node coordinates */
      for (unsigned int node = 0; node < nNodesPerFace; node++)
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
          coords_face1(node, dim) = geo.coord_nodes(face1[node], dim);

      /* Compute centroid location */
      std::vector<double> c1(geo.nDims, 0.0);
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        for (unsigned int node = 0; node < nNodesPerFace; node++)
        {
          c1[dim] += coords_face1(node, dim);
        }

        c1[dim] /= nNodesPerFace;
      }

      /* Search for face to couple */
      bool paired = false;
      for(auto &bnd_face2 : geo.bnd_faces)
      {
        auto face2 = bnd_face2.first;
        if (face2 == face1 || face2.size() != nNodesPerFace)
          continue;

        /* Get face node coordinates */
        for (unsigned int node = 0; node < nNodesPerFace; node++)
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            coords_face2(node, dim) = geo.coord_nodes(face2[node], dim);

        /* Compute centroid location */
        std::vector<double> c2(geo.nDims, 0.0);
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          for (unsigned int node = 0; node < nNodesPerFace; node++)
          {
            c2[dim] += coords_face2(node, dim);
          }

          c2[dim] /= nNodesPerFace;
        }


        /* Compare centroid locations to couple faces */
        unsigned int count = 0;
        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          if (std::abs(c1[dim] - c2[dim]) < 1.e-6)
          {
            bool onPlane = true;
            double coord = coords_face1(0,dim);

            for (unsigned int node = 1; node < nNodesPerFace; node++)
            {
              if (std::abs(coord - coords_face1(node, dim)) > 1.e-6)
                onPlane = false;
            }

            if (!onPlane)
              count++;
          }
        }

        if (count == geo.nDims - 1)
        {
          paired = true;
          geo.per_bnd_pairs[face1] = face2;

          /* Fill in map coupling nodes */
          for (unsigned int node1 = 0; node1 < nNodesPerFace; node1++)
          {
            unsigned int per_node = -1;

            for (unsigned int node2 = 0; node2 < nNodesPerFace; node2++)
            {
              unsigned int count2 = 0;
              for (unsigned int dim = 0; dim < geo.nDims; dim++)
              {
                if (std::abs(coords_face1(node1, dim) - coords_face2(node2,dim)) < 1e-6)
                  count2++;
              }

              if (count2++ == geo.nDims - 1)
              {
                per_node = node2; break;
              }
            }

            geo.per_node_pairs[face1[node1]] = face2[per_node];
          }

          auto face1_ordered = geo.face2ordered[face1];
          for (auto &i: face1_ordered)
            i = geo.per_node_pairs[i];
          auto face2_ordered = geo.face2ordered[face2];

          /* Determine rotation using ordered faces*/
          unsigned int rot = 0;
          if (geo.nDims == 3)
          {
            while (face1_ordered[rot] != face2_ordered[0])
            {
              rot++;
            }
          }
          else
          {
            rot = 4; /* Rotation for 2D edge only */
          }

          geo.per_bnd_rot[face1] = rot;
          break;
        }
      }

      if (!paired)
        ThrowException("Unpaired periodic face detected. Check your mesh.");
    }
  }
}

void setup_global_fpts(InputStruct *input, GeoStruct &geo, unsigned int order)
{
  /* Form set of unique faces */
  if (geo.nDims == 2 || geo.nDims == 3)
  {
    unsigned int nFptsPerFace = (order + 1);
    unsigned int nFpts1D = (order + 1);
    if (geo.nDims == 3)
      nFptsPerFace *= (order + 1);

    unsigned int nVars = 1;
    if (input->equation == EulerNS)
      nVars = geo.nDims + 2;


    geo.nFptsPerFace = nFptsPerFace;

    std::map<std::vector<unsigned int>, std::vector<unsigned int>> face_fpts;
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> bndface2fpts;
    std::vector<std::vector<int>> ele2fpts(geo.nEles);
    std::vector<std::vector<int>> ele2fpts_slot(geo.nEles);
    std::map<std::vector<unsigned int>, std::vector<unsigned int>> face2eles;

    std::vector<unsigned int> face(geo.nNodesPerFace,0);

    /* Determine number of interior global flux points */
    std::set<std::vector<unsigned int>> unique_faces;
    geo.nGfpts_int = 0; geo.nGfpts_bnd = 0;

#ifdef _MPI
    geo.nGfpts_mpi = 0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
      {
        face.assign(geo.nNodesPerFace, 0);

        for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
        {
          face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele);
        }

        /* Check if face is collapsed */
        std::set<unsigned int> nodes;
        for (auto node : face)
          nodes.insert(node);

        if (nodes.size() <= geo.nDims - 1) /* Fully collapsed face. Assign no fpts. */
        {
          continue;
        }
        else if (nodes.size() == 3) /* Triangular collapsed face. Must tread carefully... */
        {
          face.assign(nodes.begin(), nodes.end());
        }

        std::sort(face.begin(), face.end());
        face2eles[face].push_back(ele);


        /* Check if face is has not been previously encountered */
        if (!unique_faces.count(face))
        {
            if (geo.bnd_faces.count(face))
            {
              geo.nGfpts_bnd += nFptsPerFace;
            }
#ifdef _MPI
            else if (geo.mpi_faces.count(face))
            {
              geo.nGfpts_mpi += nFptsPerFace;
            }
#endif
            else
            {
              geo.nGfpts_int += nFptsPerFace;
            }
        }

        unique_faces.insert(face);
      }
    }

    /* Initialize global flux point indicies (to place boundary fpts at end of global fpt data structure) */
    unsigned int gfpt = 0; unsigned int gfpt_bnd = geo.nGfpts_int;

#ifdef _MPI
    unsigned int gfpt_mpi = geo.nGfpts_int + geo.nGfpts_bnd;
    std::set<std::vector<unsigned int>> mpi_faces_to_process;
#endif


    /* Begin loop through faces */
    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      ele2fpts[ele].assign(geo.nFacesPerEle * nFptsPerFace, -1);
      ele2fpts_slot[ele].assign(geo.nFacesPerEle * nFptsPerFace, -1);

      for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
      {
        face.assign(geo.nNodesPerFace, 0);

        /* Get face nodes and sort for consistency */
        for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
        {
          face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele);
        }

        auto face_ordered = face;

        /* Check if face is collapsed */
        std::set<unsigned int> nodes;
        for (auto node : face)
          nodes.insert(node);

        if (nodes.size() <= geo.nDims - 1) /* Fully collapsed face. Assign no fpts. */
        {
          continue;
        }
        else if (nodes.size() == 3) /* Triangular collapsed face. Must tread carefully... */
        {
          face.assign(nodes.begin(), nodes.end());
        }

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
#ifdef _MPI
          /* Check if face is on MPI boundary */
          else if (geo.mpi_faces.count(face))
          {
            /* Add face to set to process later. */
            mpi_faces_to_process.insert(face);

            for (auto &fpt : fpts)
            {
              fpt = gfpt_mpi;
              gfpt_mpi++;
            }
          }
#endif
          else
          {
            for (auto &fpt : fpts)
            {
              fpt = gfpt;
              gfpt++;
            }
          }

          face_fpts[face] = fpts;
          geo.face2ordered[face] = face_ordered;

          for (unsigned int i = 0; i < nFptsPerFace; i++)
          {
            ele2fpts[ele][n*nFptsPerFace + i] = fpts[i];
            ele2fpts_slot[ele][n*nFptsPerFace + i] = 0;
          }
        }
        /* If face has already been encountered, must assign existing global flux points */
        else
        {
          auto fpts = face_fpts[face];
          auto face0_ordered = geo.face2ordered[face];
          
          /* Determine rotation using ordered faces*/
          unsigned int rot = 0;
          if (geo.nDims == 3)
          {
            while (face_ordered[rot] != face0_ordered[0])
            {
              rot++;
            }
          }
          else
          {
            rot = 4;
          }

          /* Based on rotation, couple flux points */
          switch (rot)
          {
            case 0:
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                for (unsigned int j = 0; j < nFpts1D; j++)
                {
                  ele2fpts[ele][n*nFptsPerFace + i + j*nFpts1D] = fpts[i * nFpts1D + j];
                }
              } break;

            case 1:
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                for (unsigned int j = 0; j < nFpts1D; j++)
                {
                  ele2fpts[ele][n*nFptsPerFace + i + j*nFpts1D] = fpts[nFpts1D - i + j * nFpts1D - 1];
                }
              } break;

            case 2:
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                for (unsigned int j = 0; j < nFpts1D; j++)
                {
                  ele2fpts[ele][n*nFptsPerFace + i + j*nFpts1D] = fpts[nFptsPerFace - i * nFpts1D - j - 1];
                }
              } break;

            case 3:
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                for (unsigned int j = 0; j < nFpts1D; j++)
                {
                  ele2fpts[ele][n*nFptsPerFace + i + j*nFpts1D] = fpts[nFptsPerFace - (j+1) * nFpts1D + i];
                }
              } break;

            case 4:
              for (unsigned int i = 0; i < nFptsPerFace; i++)
              {
                  ele2fpts[ele][n*nFptsPerFace + i] = fpts[nFptsPerFace - i - 1];
              } break;
          }
          
          for (unsigned int i = 0; i < nFptsPerFace; i++)
          {
            ele2fpts_slot[ele][n*nFptsPerFace + i] = 1;
          }
        }
      }
    }

    /* Process MPI faces, if needed */
#ifdef _MPI
    for (const auto &face : mpi_faces_to_process)
    {
      auto ranks = geo.mpi_faces[face];
      int sendRank = *std::min_element(ranks.begin(), ranks.end());
      int recvRank = *std::max_element(ranks.begin(), ranks.end());

      /* Additional note: Deadlock is avoided due to consistent global ordering of mpi_faces map */


      /* If partition has minimum (of 2) ranks assigned to this face, use its flux point order. Send
       * information to other rank. */
      if (rank == sendRank)
      {
        auto fpts = face_fpts[face];
        auto face_ordered = geo.face2ordered[face];

        /* Convert face_ordered node indexing to global indexing */
        for (auto &nd : face_ordered)
          nd = geo.node_map_p2g[nd];

        /* Append flux points to fpt_buffer_map in existing order */
        for (auto fpt : fpts)
          geo.fpt_buffer_map[recvRank].push_back(fpt);

        /* Send ordered face to paired rank */
        MPI_Send(face_ordered.data(), geo.nNodesPerFace, MPI_INT, recvRank, 0, MPI_COMM_WORLD);
      }
      else if (rank == recvRank)
      {
        auto fpts = face_fpts[face];
        auto face_ordered = geo.face2ordered[face];
        auto face_ordered_mpi = face_ordered;

        /* Receive ordered face from paired rank */
        MPI_Status temp;
        MPI_Recv(face_ordered_mpi.data(), geo.nNodesPerFace, MPI_INT, sendRank, 0, MPI_COMM_WORLD, &temp);

        /* Convert received face_ordered node indexing to partition local indexing */
        for (auto &nd : face_ordered_mpi)
          nd = geo.node_map_g2p[nd];

        /* Determine rotation using ordered faces*/
        unsigned int rot = 0;
        if (geo.nDims == 3)
        {
          while (face_ordered[rot] != face_ordered_mpi[0])
          //while (face_ordered_mpi[rot] != face_ordered[0])
          {
            rot++;
          }
        }
        else
        {
          rot = 4;
        }

        /* Based on rotation, append flux points to fpt_buffer_map (to be consistent with paired rank fpt order) */
        switch (rot)
        {
          case 0:
            for (unsigned int j = 0; j < nFpts1D; j++)
            {
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                 geo.fpt_buffer_map[sendRank].push_back(fpts[i * nFpts1D + j]);
              }
            } break;

          case 1:
            for (unsigned int j = 0; j < nFpts1D; j++)
            {
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                geo.fpt_buffer_map[sendRank].push_back(fpts[nFpts1D - i + j * nFpts1D - 1]);
              }
            } break;

          case 2:
            for (unsigned int j = 0; j < nFpts1D; j++)
            {
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                geo.fpt_buffer_map[sendRank].push_back(fpts[nFptsPerFace - i * nFpts1D - j - 1]);
              }
            } break;

          case 3:
            for (unsigned int j = 0; j < nFpts1D; j++)
            {
              for (unsigned int i = 0; i < nFpts1D; i++)
              {
                geo.fpt_buffer_map[sendRank].push_back(fpts[nFptsPerFace - (j+1) * nFpts1D + i]);
              }
            } break;

          case 4:
            for (unsigned int i = 0; i < nFptsPerFace; i++)
            {
              geo.fpt_buffer_map[sendRank].push_back(fpts[nFptsPerFace - i - 1]);
            } break;
        }

      }
      else
      {
        ThrowException("Error in mpi_faces. Neither rank is this rank.");
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Create MPI Derived type for sends/receives */
    for (auto &entry : geo.fpt_buffer_map)
    {
      unsigned int sendRank = entry.first;
      auto fpts = entry.second;
      std::vector<int> block_len(nVars * fpts.size(), 1);
      std::vector<int> disp(nVars * fpts.size(), 0);
      
      for (unsigned int var = 0; var < nVars; var++)
      {
        for (unsigned int i = 0; i < fpts.size(); i++)
          disp[i + var * fpts.size()] = fpts(i) + var * gfpt_mpi;
      }
      
      MPI_Type_indexed(nVars * fpts.size(), block_len.data(), disp.data(), MPI_DOUBLE, &geo.mpi_types[sendRank]); 

      MPI_Type_commit(&geo.mpi_types[sendRank]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

#endif

    // TODO: For periodic MPI, move this up the code, define which faces are periodic first.
    if (geo.per_bnd_flag)
      couple_periodic_bnds(geo);

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
        auto face1_ordered = geo.face2ordered[face1];

        /* If boundary face is not periodic, skip this pairing */
        if (geo.bnd_faces[face1] != 1)
          continue;

        auto face2 = geo.per_bnd_pairs[face1];
        auto fpts2 = bndface2fpts[face2];
        auto face2_ordered = geo.face2ordered[face2];

        face2eles[face1].push_back(face2eles[face2][0]);

        /* Determine rotation using ordered faces*/
        unsigned int rot = 0;
        if (geo.nDims == 3)
        {
          rot = geo.per_bnd_rot[face2];
        }
        else
        {
          rot = 4;
        }

        /* Based on rotation, couple flux points */
        switch (rot)
        {
          case 0:
            for (unsigned int i = 0; i < nFpts1D; i++)
            {
              for (unsigned int j = 0; j < nFpts1D; j++)
              {
                geo.per_fpt_pairs[fpts1[i + j*nFpts1D]] = fpts2[i * nFpts1D + j];
                geo.per_fpt_list(fpts1[i + j*nFpts1D] - geo.nGfpts_int) = fpts2[i * nFpts1D + j];
              }
            } break;

          case 1:
            for (unsigned int i = 0; i < nFpts1D; i++)
            {
              for (unsigned int j = 0; j < nFpts1D; j++)
              {
                geo.per_fpt_pairs[fpts1[i + j*nFpts1D]] = fpts2[nFpts1D - i - 1 + j*nFpts1D];
                geo.per_fpt_list(fpts1[i + j*nFpts1D] - geo.nGfpts_int) = fpts2[nFpts1D - i - 1 + j*nFpts1D];
              }
            } break;

          case 2:
            for (unsigned int i = 0; i < nFpts1D; i++)
            {
              for (unsigned int j = 0; j < nFpts1D; j++)
              {
                geo.per_fpt_pairs[fpts1[i + j*nFpts1D]] = fpts2[nFptsPerFace - 1 - i * nFpts1D - j];
                geo.per_fpt_list(fpts1[i + j*nFpts1D] - geo.nGfpts_int) = fpts2[nFptsPerFace - 1 - i * nFpts1D - j];
              }
            } break;

          case 3:
            for (unsigned int i = 0; i < nFpts1D; i++)
            {
              for (unsigned int j = 0; j < nFpts1D; j++)
              {
                geo.per_fpt_pairs[fpts1[i + j*nFpts1D]] = fpts2[nFptsPerFace - nFpts1D * (j+1) + i];
                geo.per_fpt_list(fpts1[i + j*nFpts1D] - geo.nGfpts_int) = fpts2[nFptsPerFace - nFpts1D * (j+1) + i];
              }
            } break;

          case 4:
            for (unsigned int i = 0; i < nFptsPerFace; i++)
            {
                geo.per_fpt_pairs[fpts1[i]] = fpts2[nFptsPerFace - i - 1];
                geo.per_fpt_list(fpts1[i] - geo.nGfpts_int) = fpts2[nFptsPerFace - i - 1];
            } break;
        }
      } 
    }

    /* Populate data structures */
#ifdef _MPI
    geo.nGfpts = gfpt_mpi;
#else
    geo.nGfpts = gfpt_bnd;
#endif

    geo.fpt2gfpt.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});
    geo.fpt2gfpt_slot.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});
    geo.ele_adj.assign({geo.nFacesPerEle, geo.nEles});

    for (unsigned int ele = 0; ele < geo.nEles; ele++)
    {
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEle * nFptsPerFace; fpt++)
      {
        geo.fpt2gfpt(fpt,ele) = ele2fpts[ele][fpt];
        geo.fpt2gfpt_slot(fpt,ele) = ele2fpts_slot[ele][fpt];
      }

      for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
      {
        face.assign(geo.nNodesPerFace, 0);

        for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
        {
          face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele);
        }

        std::sort(face.begin(), face.end());

        if (face2eles[face].empty() or face2eles[face].back() == ele)
          geo.ele_adj(n, ele) = -1;
        else
          geo.ele_adj(n, ele) = face2eles[face].back();

        face2eles[face].pop_back();

      }
    }

  }
  else
  {
    ThrowException("nDims is not valid!");
  }

}

void setup_element_colors(InputStruct *input, GeoStruct &geo)
{
  /* Setup element colors */
  geo.ele_color.assign({geo.nEles});
  if (input->dt_scheme == "LUJac")
  {
    geo.nColors = 1;
    geo.ele_color.fill(1);
  }
  else if (input->dt_scheme == "LUSGS")
  {
    geo.nColors = 2;
    geo.ele_color(0) = 1;
    std::queue<unsigned int> Q;
    Q.push(0);
    while (!Q.empty())
    {
      unsigned int ele1 = Q.front();
      Q.pop();

      /* Determine opposite color */
      unsigned int color = geo.ele_color(ele1);
      if (color == 1)
      {
        color = 2;
      }
      else
      {
        color = 1;
      }

      /* Color neighbors */
      for (unsigned int face = 0; face < geo.nFacesPerEle; face++)
      {
        int ele2 = geo.ele_adj(face, ele1);
        if (ele2 != -1 && geo.ele_color(ele2) == 0)
        {
          geo.ele_color(ele2) = color;
          Q.push(ele2);
        }
      }
    }
  }

#ifdef _MPI
  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  /* In parallel, determine consistent coloring between partitions (easy method for 2 colors) */
  int repeat = 1;
  int flipRank = nRanks;

  if (rank == 0)
    flipRank = 0;

  /* Repeat process until colors are consistent */
  while (repeat)
  {
    repeat = 0;
    for (int sendRank = 0; sendRank < nRanks; sendRank++)
    {
      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == sendRank)
      {
        std::set<int> processed;
        std::vector<unsigned int> face(geo.nNodesPerFace, 0);
        int other_color, other_flipRank;

        /* Loop over elements and seek out MPI boundaries */
        for (unsigned int ele1 = 0; ele1 < geo.nEles; ele1++)
        {
          for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
          {
            int ele2 = geo.ele_adj(n, ele1);

            /* Check if adjacent element is outside partition */
            if (ele2 == -1)
            {
              /* Get face nodes and sort for consistency */
              for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
              {
                face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele1);
              }
              
              std::sort(face.begin(), face.end());
              
              /* Check if face is at an MPI interface */
              if (geo.mpi_faces.count(face))
              {
                auto face_ranks = geo.mpi_faces[face];
                auto rank1 = *std::min_element(face_ranks.begin(), face_ranks.end());
                auto rank2 = *std::max_element(face_ranks.begin(), face_ranks.end());
                auto recvRank = (rank1 == rank) ? rank2 : rank1;

                if (processed.count(recvRank))
                  continue;

                processed.insert(recvRank);

                for (auto &nd : face)
                  nd = geo.node_map_p2g[nd];

                int flag = 0;

                /* Send face information to adjacent element */
                MPI_Send(&flag, 1, MPI_INT, recvRank, 0, MPI_COMM_WORLD);
                MPI_Send(face.data(), geo.nNodesPerFace, MPI_INT, recvRank, 0, MPI_COMM_WORLD);
                MPI_Send(&geo.ele_color(ele1), 1, MPI_INT, recvRank, 0, MPI_COMM_WORLD);
                MPI_Send(&flipRank, 1, MPI_INT, recvRank, 0, MPI_COMM_WORLD);

                /* Receive adjacent elemnt information */
                MPI_Recv(&other_color, 1, MPI_INT, recvRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&other_flipRank, 1, MPI_INT, recvRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* If adjacent element is in a partition previous flipped by a lesser rank, flip this partition */
                if (geo.ele_color(ele1) == (unsigned int) other_color and flipRank > other_flipRank)
                {
                  flipRank = other_flipRank;
                  repeat = 1;
                  for (unsigned int ele = 0; ele < geo.nEles; ele++)
                  {
                    geo.ele_color(ele) = (geo.ele_color(ele) == 1) ? 2 : 1;
                  }
                }

                flipRank = std::min(flipRank, other_flipRank);
                continue;
              }
            }
          }
        }

        /* Send termination flag to remaining unprocessed ranks */
        for (int n = 0; n < nRanks; n++)
        {
          if (!processed.count(n) and n != rank)
          {
            int flag = 1;
            MPI_Send(&flag, 1, MPI_INT, n, 0, MPI_COMM_WORLD);
          }
        }
      }
      else
      {
        std::vector<unsigned int> face(geo.nNodesPerFace, 0), other_face(geo.nNodesPerFace, 0);
        int flag;
        int other_color, other_flipRank;

        /* Check if terminated */
        MPI_Recv(&flag, 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (flag)
          continue;

        /* Receive data from adjancent element */
        MPI_Recv(other_face.data(), geo.nNodesPerFace, MPI_INT, sendRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&other_color, 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&other_flipRank, 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (auto &nd : other_face)
          nd = geo.node_map_g2p[nd];

        /* Search for corresponding interface */
        for (unsigned int ele1 = 0; ele1 < geo.nEles; ele1++)
        {
          for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
          {
            int ele2 = geo.ele_adj(n, ele1);

            /* Check if adjacent element is on boundary */
            if (ele2 == -1)
            {
              /* Get face nodes and sort for consistency */
              for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
              {
                face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele1);
              }
              
              std::sort(face.begin(), face.end());

              /* Found matched face! */
              if (other_face == face)
              {
                MPI_Send(&geo.ele_color(ele1), 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD);
                MPI_Send(&other_flipRank, 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD);

                /* Apply similar logic as case above */
                if (geo.ele_color(ele1) == (unsigned int) other_color and flipRank > other_flipRank)
                {
                  flipRank = other_flipRank;
                  repeat = 1;
                  for (unsigned int ele = 0; ele < geo.nEles; ele++)
                  {
                    geo.ele_color(ele) = (geo.ele_color(ele) == 1) ? 2 : 1;
                  }
                }

                flipRank = std::min(flipRank, other_flipRank);
                break;
              }
            }
          }
          if (other_face == face)
            break;
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    /* If any flips have occured, repeat process */
    MPI_Allreduce(MPI_IN_PLACE, &repeat, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  /* Reorganize required geometry data by color */
  std::vector<std::vector<unsigned int>> color2eles(geo.nColors);
  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    color2eles[geo.ele_color(ele) - 1].push_back(ele);
  }

  /* TODO: Consider an in-place permutation to save memory */
  auto nd2gnd_temp = geo.nd2gnd;
  auto fpt2gfpt_temp = geo.fpt2gfpt;
  auto fpt2gfpt_slot_temp = geo.fpt2gfpt_slot;

  unsigned int ele1 = 0;
  for (unsigned int color = 1; color <= geo.nColors; color++)
  {
    for (unsigned int i = 0; i < color2eles[color - 1].size(); i++)
    {
      unsigned int ele2 = color2eles[color - 1][i];

      for (unsigned int node = 0; node < geo.nNodesPerEle; node++)
      {
        geo.nd2gnd(node, ele1) = nd2gnd_temp(node, ele2);
      }

      for (unsigned int fpt = 0; fpt < geo.nFptsPerFace * geo.nFacesPerEle; fpt++) 
      {
        geo.fpt2gfpt(fpt, ele1) = fpt2gfpt_temp(fpt, ele2);
        geo.fpt2gfpt_slot(fpt, ele1) = fpt2gfpt_slot_temp(fpt, ele2);
      }

      geo.ele_color(ele1) = color;

      ele1++;
    }
  }

  /* Setup element color ranges */
  geo.ele_color_range.assign(geo.nColors + 1, 0);
  geo.ele_color_range[1] = color2eles[0].size(); 
  for (unsigned int color = 2; color <= geo.nColors; color++)
  {
    geo.ele_color_range[color] = geo.ele_color_range[color - 1] + color2eles[color-1].size(); 
  }
}

#ifdef _MPI
void partition_geometry(GeoStruct &geo)
{
  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  /* Setup METIS */
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  

  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  options[METIS_OPTION_DBGLVL] = 0;
  options[METIS_OPTION_CONTIG] = 1;
  options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;

  /* Form eptr and eind arrays */
  std::vector<int> eptr(geo.nEles + 1); 
  std::vector<int> eind(geo.nEles * geo.nCornerNodes); 
  std::vector<int> vwgt(geo.nEles, 1);
  std::set<unsigned int> nodes;
  std::vector<unsigned int> face;

  int n = 0;
  eptr[0] = 0;
  for (unsigned int i = 0; i < geo.nEles; i++)
  {
    for (unsigned int j = 0; j < geo.nCornerNodes;  j++)
    {
      eind[j + n] = geo.nd2gnd(j, i);
      nodes.insert(geo.nd2gnd(j,i));
    } 

    /* Check for collapsed edge (not fully general yet)*/
    if (nodes.size() < geo.nCornerNodes)
    {
      n += geo.nCornerNodes - 1;
    }
    else
    {
      n += geo.nCornerNodes;
    }
    eptr[i + 1] = n;
    nodes.clear();

    /* Loop over faces and search for boundaries */
    for (unsigned int k = 0; k < geo.nFacesPerEle; k++)
    {
      face.assign(geo.nNodesPerFace, 0);

      for (unsigned int nd = 0; nd < geo.nNodesPerFace; nd++)
      {
        face[nd] = geo.nd2gnd(geo.face_nodes(k, nd), i);
      }

      if (geo.bnd_faces.count(face))
      {
        auto bnd_id = geo.bnd_faces[face];

        switch (bnd_id)
        {
          case 6:
            vwgt[i] += 0; break;
          case 7:
          case 8:
          case 9:
          case 10:
          case 11:
          case 12:
            vwgt[i] += 0; break;

          default:
            vwgt[i] += 0; break;
        }
        vwgt[i]++;
      }
    }


  }

  int objval;
  std::vector<int> epart(geo.nEles, 0);
  std::vector<int> npart(geo.nNodes);
  /* TODO: Should just not call this entire function if nRanks == 1 */
  if (nRanks > 1) 
  {
    int nNodesPerFace = geo.nNodesPerFace; // TODO: What should this be?
    int nEles = geo.nEles;
    int nNodes = geo.nNodes;

    METIS_PartMeshDual(&nEles, &nNodes, eptr.data(), eind.data(), vwgt.data(), 
        nullptr, &nNodesPerFace, &nRanks, nullptr, options, &objval, epart.data(), 
        npart.data());  
  }

  /* Obtain list of elements on this partition */
  std::vector<unsigned int> myEles;
  for (unsigned int ele = 0; ele < geo.nEles; ele++) 
  {
    if (epart[ele] == rank) 
      myEles.push_back(ele);
  }

  /* Collect map of *ALL* MPI interfaces from METIS partition data */
  //std::vector<unsigned int> face(geo.nNodesPerFace);
  std::map<std::vector<unsigned int>, std::set<int>> face2ranks;    
  std::map<std::vector<unsigned int>, std::set<int>> mpi_faces_glob;

  /* Iterate over faces of complete mesh */
  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    int face_rank = epart[ele];
    for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
    {
      face.assign(geo.nNodesPerFace, 0);
      for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
      {
        face[i] = geo.nd2gnd(geo.face_nodes(n, i), ele);
      }

      /* Sort for consistency */
      std::sort(face.begin(), face.end());
      
      face2ranks[face].insert(face_rank);

      /* If two ranks assigned to same face, add to map of "MPI" faces */
      if (face2ranks[face].size() == 2)
      {
        mpi_faces_glob[face] = face2ranks[face];  
      }
    }
  }

  /* Reduce connectivity to contain only partition local elements */
  auto nd2gnd_glob = geo.nd2gnd;
  geo.nd2gnd.assign({geo.nNodesPerEle, (unsigned int) myEles.size()},0);
  for (unsigned int ele = 0; ele < myEles.size(); ele++)
  {
    for (unsigned int nd = 0; nd < geo.nNodesPerEle; nd++)
    {
      geo.nd2gnd(nd, ele) = nd2gnd_glob(nd, myEles[ele]);
    }
  }

  /* Obtain set of unique nodes on this partition */
  std::set<unsigned int> uniqueNodes;
  for (unsigned int ele = 0; ele < myEles.size(); ele++)
  {
    for (unsigned int nd = 0; nd < geo.nNodesPerEle; nd++)
    {
      uniqueNodes.insert(geo.nd2gnd(nd, ele));
    }
  }

  /* Reduce node coordinate data to contain only partition local nodes */
  auto coord_nodes_glob = geo.coord_nodes;
  geo.coord_nodes.assign({(unsigned int) uniqueNodes.size(), geo.nDims}, 0);
  unsigned int idx = 0;
  for (unsigned int nd: uniqueNodes)
  {
    geo.node_map_g2p[nd] = idx; /* Map of global node idx to partition node idx */
    geo.node_map_p2g[idx] = nd; /* Map of parition node idx to global node idx */

    for (unsigned int dim = 0; dim < geo.nDims; dim++)
      geo.coord_nodes(idx, dim) = coord_nodes_glob(nd, dim);

    idx++;
  }

  /* Renumber connectivity data using partition local indexing (via geo.node_map_g2p)*/
  for (unsigned int ele = 0; ele < myEles.size(); ele++)
    for (unsigned int nd = 0; nd < geo.nNodesPerEle; nd++)
      geo.nd2gnd(nd, ele) = geo.node_map_g2p[geo.nd2gnd(nd, ele)];

  /* Reduce boundary faces data to contain only faces on local partition. Also
   * reindex via geo.node_map_g2p */
  auto bnd_faces_glob = geo.bnd_faces;
  geo.bnd_faces.clear();
  
  /* Iterate over all boundary faces */
  for (auto entry : bnd_faces_glob)
  {
    std::vector<unsigned int> bnd_face = entry.first;
    int bnd_id = entry.second;

    /* If all nodes are on this partition, keep face data */
    bool myFace = true;
    for (auto nd : bnd_face)
    {
      if (!uniqueNodes.count(nd))
      {
        myFace = false;
      }
    }

    if (myFace)
    {
      /* Renumber nodes and store */
      for (auto &nd : bnd_face)
      {
        nd = geo.node_map_g2p[nd];
      }
      geo.bnd_faces[bnd_face] = bnd_id;
    }
  }

  /* Reduce MPI face data to contain only MPI faces for local partition. Also
   * reindex via geo.node_map_g2p. */
  for (auto entry : mpi_faces_glob)
  {
    std::vector<unsigned int> mpi_face = entry.first;
    auto face_ranks = entry.second;

    /* If all nodes are on this partition, keep face data */
    bool myFace = true;
    for (auto nd : mpi_face)
    {
      if (!uniqueNodes.count(nd))
      {
        myFace = false;
      }
    }

    if (myFace)
    {
      /* Renumber nodes and store */
      for (auto &nd : mpi_face)
      {
        nd = geo.node_map_g2p[nd];
      }
      geo.mpi_faces[mpi_face] = face_ranks;
    }
  }
  
  /* Set number of nodes/elements to partition local */
  geo.nNodes = (unsigned int) uniqueNodes.size();
  geo.nEles = (unsigned int) myEles.size();

}
#endif

