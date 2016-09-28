#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _MPI
#include <unistd.h>
#include "mpi.h"
#include "metis.h"
#endif

#include "funcs.hpp"
#include "geometry.hpp"
#include "macros.hpp"
#include "mdvector.hpp"

#ifndef _NO_HDF5
#include "H5Cpp.h"
#ifndef _H5_NO_NAMESPACE
using namespace H5;
#endif
#endif

enum MESH_FORMAT {
  GMSH, PYFR
};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims)
{
  GeoStruct geo;
  geo.nDims = nDims;
  geo.input = input;

  int format;

  if (input->meshfile.find(".msh") != std::string::npos)
    format = GMSH;
  else if (input->meshfile.find(".pyfr") != std::string::npos)
    format = PYFR;
  else
    ThrowException("Unrecognized mesh format - expecting *.msh or *.pyfrm.");

  if (format == GMSH)
  {
    load_mesh_data_gmsh(input, geo);

    if (input->dt_scheme == "MCGS")
    {
      setup_element_colors(input, geo);
    }

#ifdef _MPI
    partition_geometry(input, geo);
#endif

    setup_global_fpts(input, geo, order);

    if (input->dt_scheme == "MCGS")
    {
      shuffle_data_by_color(geo);
    }
  }

  else if (format == PYFR)
  {
    load_mesh_data_pyfr(input, geo);

    setup_global_fpts_pyfr(input, geo, order);
  }

  return geo;
}


//typedef struct {
//  std::vector<std::string> obj_names;
//  std::vector<hid_t> obj_ids;
//} h5obj_data;

///** Operator function for H5_Iterate */
//herr_t file_info(hid_t loc_id, const char *name, void *opdata)
//{
//  h5obj_data *od = (h5obj_data*)opdata;

//  od->obj_names.push_back(std::string(name));
//  od->obj_ids.push_back(loc_id);

//  return 0;
//}

void load_mesh_data_pyfr(InputStruct *input, GeoStruct &geo)
{
  std::cout << "****************************************" << std::endl;

  H5File file(input->meshfile, H5F_ACC_RDONLY);

  // Load the names of all datasets into a vector so we can more easily
  // process each one

  std::vector<std::string> dsNames;

  ssize_t nObjs = file.getNumObjs();

  for (int i  = 0; i < nObjs; i++)
  {
    if (file.getObjTypeByIdx(i) == H5G_DATASET)
    {
      dsNames.push_back(file.getObjnameByIdx(i));
      //std::cout << "DataSet name " << i << ": " << dsNames.back() << std::endl;
    }
  }

  // Read the mesh ID, config, and stats strings
  DataSet dset = file.openDataSet("mesh_uuid");
  DataType dtype = dset.getDataType();
  DataSpace dspace(H5S_SCALAR);
  dset.read(geo.mesh_uuid, dtype, dspace);
  dset.close();

  // Oops - only for solution files.  my bad. move this later
//  dset = file.openDataSet("config");
//  dset.read(geo.config, dtype, dspace);
//  dset.close();

//  dset = file.openDataSet("stats");
//  dset.read(geo.stats, dtype, dspace);
//  dset.close();


  // Figure out # of dimensions
  for (auto &name : dsNames)
  {
    /// TODO: add support for reading triangles, tets, etc.
    /// (nEles += _, read into tmp array, duplicate node 2 like with Gmsh)
    if (name.find("spt_quad") == std::string::npos &&
        name.find("spt_hex") == std::string::npos)
      continue;

    auto DS = file.openDataSet(name);
    auto ds = DS.getSpace();

    std::vector<hsize_t> dims(3);
    hsize_t max_dims = 3;
    int ds_rank = ds.getSimpleExtentDims(dims.data(), &max_dims);

    if (ds_rank != 3 or DS.getTypeClass() != H5T_FLOAT)
      ThrowException("Cannot read element nodes from PyFR mesh file - wrong data type.");

    geo.nDims = dims[2];
    geo.nEles = dims[1];
    geo.nNodesPerEle = dims[0];
    if (dims[0] == 8)
    {
      input->serendipity = 1;
      geo.shape_order = 2;
    }
    else
      geo.shape_order = std::sqrt(dims[0]) - 1;

    geo.ele_nodes.assign({geo.nNodesPerEle, geo.nEles, geo.nDims}, 0);

    DS.read(geo.ele_nodes.data(), PredType::NATIVE_DOUBLE);

    /// TODO: change Gmsh reading / geo setup so this isn't needed
    geo.coord_nodes.assign({geo.nDims, geo.nEles*geo.nNodesPerEle});
    geo.ele2nodes.assign({geo.nNodesPerEle, geo.nEles});

    int gnd = 0;
    for (int ele = 0; ele < geo.nEles; ele++)
    {
      for (int nd = 0; nd < geo.nNodesPerEle; nd++)
      {
        for (int d = 0; d < geo.nDims; d++)
          geo.coord_nodes(gnd, d) = geo.ele_nodes(nd, ele, d);
        geo.ele2nodes(nd, ele) = gnd;
        gnd++;
      }
    }
  }

  if (geo.nDims == 3)
  {
    geo.pyfr2zefr_face = {0,4,3,5,2,1};
    geo.zefr2pyfr_face = {0,5,4,2,1,3};
  }
  else if (geo.nDims == 2)
  {
    geo.pyfr2zefr_face = {0,1,2,3};
    geo.zefr2pyfr_face = {0,1,2,3};
  }


  // ----- Read element / face connectivity -----

  for (auto &name : dsNames)
  {
    if (name.substr(0,4) == "con_")
    {
      auto DS = file.openDataSet(name);
      auto ds = DS.getSpace();

      std::vector<hsize_t> dims(3);
      hsize_t max_dims = 3;
      int ds_rank = ds.getSimpleExtentDims(dims.data(), &max_dims);

      if (ds_rank != 2 or DS.getTypeClass() != H5T_COMPOUND)
        ThrowException("Cannot read internal connectivity from PyFR mesh file - expecting compound data type of rank 2.");

      hid_t char4_t = H5Tcopy(H5T_C_S1);
      H5Tset_size(char4_t, 5); // 4 chars + null-termination character

      CompType fcon_t(sizeof(face_con));
      fcon_t.insertMember("f0", HOFFSET(face_con, c_type), char4_t);
      fcon_t.insertMember("f1", HOFFSET(face_con, ic), PredType::NATIVE_INT);
      fcon_t.insertMember("f2", HOFFSET(face_con, loc_f), PredType::NATIVE_SHORT);

      geo.nFaces = dims[1];

      geo.face_list.resize(2*geo.nFaces);
      DS.read(geo.face_list.data(), fcon_t);

//      for (int f = 0; f < geo.nFaces; f++)
//      {
//        auto f1 = temp_conn[f];
//        auto f2 = temp_conn[geo.nFaces+f];
//        std::cout << f1.f_type << ", " << f1.ic << ", " << f1.loc_f << " | ";
//        std::cout << f2.f_type << ", " << f2.ic << ", " << f2.loc_f << std::endl;
//      }

      break;
    }
  }

  // Setup Zefr-style ele-to-face connectivity
  geo.nFacesPerEle = (geo.nDims == 3) ? 6 : 4;
  geo.ele2face.assign({geo.nFacesPerEle, geo.nEles});
  for (int f = 0; f < geo.nFaces; f++)
  {
    auto &f1 = geo.face_list[f];
    auto &f2 = geo.face_list[geo.nFaces+f];
    f1.loc_f = geo.pyfr2zefr_face[f1.loc_f];
    f2.loc_f = geo.pyfr2zefr_face[f2.loc_f];
    geo.ele2face(f1.loc_f, f1.ic) = f;
    geo.ele2face(f2.loc_f, f2.ic) = f;
  }

  // ----- Read boundary conditions -----

  geo.nBounds = 0;
  geo.bound_faces.resize(0);
  for (auto &name : dsNames)
  {
    if (name.substr(0,5) == "bcon_")
    {
      std::string bcName = name;
      size_t ind = bcName.find("_");
      bcName.erase(bcName.begin(),bcName.begin() + ind+1);
      ind = bcName.find("_p");
      bcName.erase(bcName.begin()+ind,bcName.end());

      std::cout << "Reading boundary " << bcName << std::endl; /// DEBUGGING

      // Convert to lowercase to make matching easier
      std::transform(bcName.begin(), bcName.end(), bcName.begin(), ::tolower);

      // First, map mesh boundary to boundary name in input file
      if (!input->meshBounds.count(bcName))
      {
        std::string errS = "Unrecognized mesh boundary: \"" + bcName + "\"\n";
        errS += "Boundary names in input file must match those in mesh file.";
        ThrowException(errS.c_str());
      }

      // Map the mesh boundary name to the input-file-specified Zefr boundary condition
      std::string bcStr = input->meshBounds[bcName];

      // Next, check that the requested boundary condition exists
      if (!bcStr2Num.count(bcStr))
      {
        std::string errS = "Unrecognized boundary condition: \"" + bcStr + "\"";
        ThrowException(errS.c_str());
      }

      geo.bnd_ids.push_back(bcStr2Num[bcStr]);
      geo.bcNames.push_back(bcName);
      geo.bcIdMap[geo.nBounds] = geo.nBounds; // Map Gmsh bcid to ZEFR bound index
      if (geo.bnd_ids.back() == PERIODIC) geo.per_bnd_flag = true; /// TODO: not support?

      auto DS = file.openDataSet(name);
      auto ds = DS.getSpace();

      int ds_rank = ds.getSimpleExtentNdims();
      std::vector<hsize_t> dims(ds_rank);
      ds.getSimpleExtentDims(dims.data());

      if (ds_rank != 1 or DS.getTypeClass() != H5T_COMPOUND)
        ThrowException("Cannot read boundary condition from PyFR mesh file - expecting compound data type of rank 1.");

      hid_t char4_t = H5Tcopy(H5T_C_S1);
      H5Tset_size(char4_t, 5); // 4 chars + null-termination character

      CompType fcon_t(sizeof(face_con));
      fcon_t.insertMember("f0", HOFFSET(face_con, c_type), char4_t);
      fcon_t.insertMember("f1", HOFFSET(face_con, ic), PredType::NATIVE_INT);
      fcon_t.insertMember("f2", HOFFSET(face_con, loc_f), PredType::NATIVE_SHORT);

      geo.bound_faces.emplace_back();
      geo.bound_faces[geo.nBounds].resize(dims[0]);
      DS.read(geo.bound_faces[geo.nBounds].data(), fcon_t);

      for (auto &face : geo.bound_faces[geo.nBounds])
        face.loc_f = geo.pyfr2zefr_face[face.loc_f];

      geo.nBounds++;
    }
  }

  std::cout << "-----------------------" << std::endl;
  std::cout << "Done reading PyFR mesh." << std::endl;
  std::cout << "-----------------------" << std::endl;

  /* To load:
  bcon_bcwalllower_p0      Dataset {8}
  bcon_bcwallupper_p0      Dataset {8}
  con_p0                   Dataset {2, 81}
  mesh_uuid                Dataset {SCALAR}
  spt_quad_p0              Dataset {4, 37, 2}
  spt_tri_p0               Dataset {3, 10, 2}
  */

  /// TODO: load by rank (look for '_p[rank]')
  /// TODO: Load MPI face connectivity ( '_p[myrank][rank2]' )
}

void load_mesh_data_gmsh(InputStruct *input, GeoStruct &geo)
{
  std::ifstream f(input->meshfile);

  if (!f.is_open())
    ThrowException("Could not open specified mesh file!");

  std::string param;

  /* Process file information */
  /* Load boundary tags */
  read_boundary_ids(f, geo, input);

  /* Load node coordinate data */
  read_node_coords(f, geo);


  /* Load element connectivity data */
  read_element_connectivity(f, geo, input);
  read_boundary_faces(f, geo);

  set_face_nodes(geo);

  set_ele_adjacency(geo);

  f.close();

}

void read_boundary_ids(std::ifstream &f, GeoStruct &geo, InputStruct *input)
{
  /* Move cursor to $PhysicalNames */
  f.clear();
  f.seekg(0, f.beg);

  std::string str;
  while(1)
  {
    std::getline(f, str);
    if (str.find("$PhysicalNames") != std::string::npos) break;
    if (f.eof()) ThrowException("Meshfile missing $PhysicalNames tag");
  }

  unsigned int nBndIds;
  f >> nBndIds;
  std::getline(f, str); // Clear remainder of line

  /* New boundary-reading format taken from Flurry++ */
  geo.nBounds = 0;
  for (unsigned int i = 0; i < nBndIds; i++)
  {
    std::string bcStr, bcName;
    std::stringstream ss;
    int bcdim, bcid;

    std::getline(f,str);
    ss << str;
    ss >> bcdim >> bcid >> bcStr;

    // Remove quotation marks from around boundary condition
    size_t ind = bcStr.find("\"");
    while (ind!=std::string::npos)
    {
      bcStr.erase(ind,1);
      ind = bcStr.find("\"");
    }
    bcName = bcStr;

    // Convert to lowercase to match Flurry's boundary condition strings
    std::transform(bcStr.begin(), bcStr.end(), bcStr.begin(), ::tolower);

    // First, map mesh boundary to boundary name in input file
    if (!input->meshBounds.count(bcStr))
    {
      std::string errS = "Unrecognized mesh boundary: \"" + bcStr + "\"\n";
      errS += "Boundary names in input file must match those in mesh file.";
      ThrowException(errS.c_str());
    }

    // Map the Gmsh PhysicalName to the input-file-specified Flurry boundary condition
    bcStr = input->meshBounds[bcStr];

    // Next, check that the requested boundary condition exists
    if (!bcStr2Num.count(bcStr))
    {
      std::string errS = "Unrecognized boundary condition: \"" + bcStr + "\"";
      ThrowException(errS.c_str());
    }

    if (bcStr.compare("fluid")==0)
    {
      if (bcdim != input->nDims)
        ThrowException("nDims in mesh file does not match input-specified nDims.");
      geo.bcIdMap[bcid] = -1;
    }
    else
    {
      geo.bnd_ids.push_back(bcStr2Num[bcStr]);
      geo.bcNames.push_back(bcName);
      geo.bcIdMap[bcid] = geo.nBounds; // Map Gmsh bcid to ZEFR bound index
      if (geo.bnd_ids.back() == PERIODIC) geo.per_bnd_flag = true;
      geo.nBounds++;
    }
  }

  geo.boundFaces.resize(geo.nBounds);
}

void read_node_coords(std::ifstream &f, GeoStruct &geo)
{
  /* Move cursor to $Nodes */
  f.clear();
  f.seekg(0, f.beg);

  std::string str;
  while(1)
  {
    std::getline(f, str);
    if (str.find("$Nodes") != std::string::npos) break;
    if (f.eof()) ThrowException("Meshfile missing $Nodes tag");
  }

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
  /* Move cursor to $Elements */
  f.clear();
  f.seekg(0, f.beg);

  std::string str;
  while(1)
  {
    std::getline(f, str);
    if (str.find("$Elements") != std::string::npos) break;
    if (f.eof()) ThrowException("Meshfile missing $Elements tag");
  }

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
  geo.ele2nodes.assign({geo.nNodesPerEle, geo.nEles});

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
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele);
          geo.ele2nodes(3,ele) = geo.ele2nodes(2,ele);
          ele++; break;

        case 3: /* 4-node Quadrilateral */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele) >> geo.ele2nodes(3,ele);
          ele++; break;

        case 9: /* 6-node Triangle */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele);
          f >> geo.ele2nodes(4,ele) >> geo.ele2nodes(5,ele) >> geo.ele2nodes(7,ele);
          geo.ele2nodes(3,ele) = geo.ele2nodes(2,ele); geo.ele2nodes(6,ele) = geo.ele2nodes(2,ele);

          if (!input->serendipity)
          {
            //TODO set geo.nd2gnd(8,ele) to centroid
            ThrowException("Biquadratic quad to triangles not implemented yet! Set serendipity = 1!");
          }

          ele++; break;

        case 10: /* 9-node Quadilateral */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele) >> geo.ele2nodes(3,ele);
          f >> geo.ele2nodes(4,ele) >> geo.ele2nodes(5,ele) >> geo.ele2nodes(6,ele) >> geo.ele2nodes(7,ele);
          if (!input->serendipity)
            f >> geo.ele2nodes(8,ele);
          else
            f >> vint;
          ele++; break;

        case 36: /* 16-node Quadilateral */
          for (int n = 0; n < 16; n++)
            f >> geo.ele2nodes(n, ele);
          ele++; break;

        case 37: /* 25-node Quadilateral */
          for (int n = 0; n < 25; n++)
            f >> geo.ele2nodes(n, ele);
          ele++; break;

        case 38: /* 36-node Quadilateral */
          for (int n = 0; n < 36; n++)
            f >> geo.ele2nodes(n, ele);
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
          geo.ele2nodes(4, ele) = min_node;
          geo.ele2nodes(5, ele) = min_node;
          geo.ele2nodes(6, ele) = min_node;
          geo.ele2nodes(7, ele) = min_node;

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
            geo.ele2nodes(0, ele) = nodes[1];
            geo.ele2nodes(1, ele) = nodes[0];
            geo.ele2nodes(2, ele) = nodes[2];
            geo.ele2nodes(3, ele) = nodes[2];
          }
          else if (min_pos == 1 || min_pos == 3)
          {
            geo.ele2nodes(0, ele) = nodes[0];
            geo.ele2nodes(1, ele) = nodes[1];
            geo.ele2nodes(2, ele) = nodes[2];
            geo.ele2nodes(3, ele) = nodes[2];
          }

          ele++; break;
        }

        case 5: /* 8-node Hexahedral */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele) >> geo.ele2nodes(3,ele);
          f >> geo.ele2nodes(4,ele) >> geo.ele2nodes(5,ele) >> geo.ele2nodes(6,ele) >> geo.ele2nodes(7,ele);
          ele++; break;

        case 6: /* 6-node Prism */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele);
          f >> geo.ele2nodes(4,ele) >> geo.ele2nodes(5,ele) >> geo.ele2nodes(6,ele);
          geo.ele2nodes(3,ele) = geo.ele2nodes(2,ele);
          geo.ele2nodes(7,ele) = geo.ele2nodes(6,ele);
          ele++; break;

        case 7: /* 5-node Pyramid */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1, ele) >> geo.ele2nodes(2, ele);
          f >> geo.ele2nodes(3, ele) >> geo.ele2nodes(4,ele);
          geo.ele2nodes(5, ele) = geo.ele2nodes(4, ele);
          geo.ele2nodes(6, ele) = geo.ele2nodes(4, ele);
          geo.ele2nodes(7, ele) = geo.ele2nodes(4, ele);
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
          geo.ele2nodes(4, ele) = min_vert; geo.ele2nodes(5, ele) = min_vert;
          geo.ele2nodes(6, ele) = min_vert; geo.ele2nodes(7, ele) = min_vert;
          geo.ele2nodes(16,ele) =  min_vert; geo.ele2nodes(17,ele) = min_vert;
          geo.ele2nodes(18,ele) =  min_vert; geo.ele2nodes(19,ele) = min_vert;

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
            geo.ele2nodes(0, ele) = verts[1];
            geo.ele2nodes(1, ele) = verts[0];
            geo.ele2nodes(2, ele) = verts[2];
            geo.ele2nodes(3, ele) = verts[2];

            geo.ele2nodes(8,ele) =  bverts[0]; geo.ele2nodes(9,ele) = bverts[2];
            geo.ele2nodes(10,ele) = verts[2]; geo.ele2nodes(11,ele) = bverts[1];

            geo.ele2nodes(12,ele) =  mverts[1]; geo.ele2nodes(13,ele) = mverts[0];
            geo.ele2nodes(14,ele) =  mverts[2]; geo.ele2nodes(15,ele) = mverts[2];
          }
          else if (min_pos == 1 || min_pos == 3)
          {
            geo.ele2nodes(0, ele) = verts[0];
            geo.ele2nodes(1, ele) = verts[1];
            geo.ele2nodes(2, ele) = verts[2];
            geo.ele2nodes(3, ele) = verts[2];

            geo.ele2nodes(8,ele) =  bverts[0]; geo.ele2nodes(9,ele) = bverts[1];
            geo.ele2nodes(10,ele) = verts[2]; geo.ele2nodes(11,ele) = bverts[2];

            geo.ele2nodes(12,ele) =  mverts[0]; geo.ele2nodes(13,ele) = mverts[1];
            geo.ele2nodes(14,ele) =  mverts[2]; geo.ele2nodes(15,ele) = mverts[2];
          }

          ele++; break;
        }

        case 12: /* Triquadratic Hex (read as 20-node serendipity) */
          f >> geo.ele2nodes(0,ele) >> geo.ele2nodes(1,ele) >> geo.ele2nodes(2,ele) >> geo.ele2nodes(3,ele);
          f >> geo.ele2nodes(4,ele) >> geo.ele2nodes(5,ele) >> geo.ele2nodes(6,ele) >> geo.ele2nodes(7,ele);
          f >> geo.ele2nodes(8,ele) >> geo.ele2nodes(11,ele) >> geo.ele2nodes(12,ele) >> geo.ele2nodes(9,ele);
          f >> geo.ele2nodes(13,ele) >> geo.ele2nodes(10,ele) >> geo.ele2nodes(14,ele) >> geo.ele2nodes(15,ele);
          f >> geo.ele2nodes(16,ele) >> geo.ele2nodes(19,ele) >> geo.ele2nodes(17,ele) >> geo.ele2nodes(18,ele);
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
      geo.ele2nodes(n,ele)--;
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

      /* Map to ZEFR boundary */
      bnd_id = geo.bcIdMap[bnd_id];
      int bcType = geo.bnd_ids[bnd_id];

      /* Sort for consistency and add to map*/
      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = bcType;
      geo.face2bnd[face] = bnd_id;
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

      /* Map to ZEFR boundary */
      bnd_id = geo.bcIdMap[bnd_id];
      int bcType = geo.bnd_ids[bnd_id];

      std::sort(face.begin(), face.end());
      geo.bnd_faces[face] = bcType;
      geo.face2bnd[face] = bnd_id;
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

void set_ele_adjacency(GeoStruct &geo)
{
  std::map<std::vector<unsigned int>, std::vector<unsigned int>> face2eles;
  std::vector<unsigned int> face(geo.nNodesPerFace,0);

  /* Fill face2eles structure */
  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
    {
      face.assign(geo.nNodesPerFace, 0);

      for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
      {
        face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
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
    }
  }

  /* Generate element adjacency */
  geo.ele_adj.assign({geo.nFacesPerEle, geo.nEles});
  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    for (unsigned int n = 0; n < geo.nFacesPerEle; n++)
    {
      face.assign(geo.nNodesPerFace, 0);

      for (unsigned int i = 0; i < geo.nNodesPerFace; i++)
      {
        face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
      }

      std::sort(face.begin(), face.end());

      if (face2eles.count(face))
      {        
        if (face2eles[face].empty() or face2eles[face].back() == ele)
          geo.ele_adj(n, ele) = -1;
        else
          geo.ele_adj(n, ele) = face2eles[face].back();

        face2eles[face].pop_back();
      }
      else
      {
        geo.ele_adj(n, ele) = -1;
      }
    }
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
    if (bnd_id == PERIODIC)
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
          face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
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

    geo.nFaces = 0;
    geo.faceList.resize(0);

    geo.ele2face.assign({geo.nFacesPerEle, geo.nEles}, -1);
    geo.face2eles.assign({2, unique_faces.size()}, -1);

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
          face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
        }

        auto face_ordered = face;

        /* Check if face is collapsed */
        std::set<unsigned int> nodes;
        for (auto node : face)
          nodes.insert(node);

        if (nodes.size() <= geo.nDims - 1) /* Fully collapsed face. Assign no fpts. */
        {
          //geo.c2f(ele, n) = -1;
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
          geo.ele2face(n, ele) = geo.nFaces;
          geo.face2eles(0, geo.nFaces) = ele;

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

            int bnd = geo.face2bnd[face];
            geo.boundFaces[bnd].push_back(geo.nFaces);
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
          /* Otherwise, internal face */
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

          geo.faceList.push_back(face);
          geo.nodes_to_face[face] = geo.nFaces;
          geo.nFaces++;
        }
        /* If face has already been encountered, must assign existing global flux points */
        else
        {
          int ff = geo.nodes_to_face[face];
          geo.ele2face(n, ele) = ff;
          geo.face2eles(1, ff) = ele;

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
        if (geo.bnd_faces[face1] != PERIODIC)
          continue;

        auto face2 = geo.per_bnd_pairs[face1];
        auto fpts2 = bndface2fpts[face2];
        auto face2_ordered = geo.face2ordered[face2];

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
          face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
        }

        std::sort(face.begin(), face.end());

      }
    }

  }
  else
  {
    ThrowException("nDims is not valid!");
  }

}

void setup_global_fpts_pyfr(InputStruct *input, GeoStruct &geo, unsigned int order)
{
  if (input->rank == 0)
    std::cout << "Setting up global flux point connectivity..." << std::endl;

  /* Form set of unique faces */
  if (geo.nDims != 2 && geo.nDims != 3)
    ThrowException("Improper value for nDims - should be 2 or 3.");

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

  std::vector<unsigned int> face(geo.nNodesPerFace,0);

  /* Determine number of interior global flux points */
  std::set<std::vector<unsigned int>> unique_faces;
  geo.nGfpts_int = geo.face_list.size() * nFptsPerFace;

  /* Determine total number of boundary faces and flux points */
  geo.nBndFaces = 0;
  for (auto &vec : geo.bound_faces)
  {
    geo.nBndFaces += vec.size();
  }
  geo.nGfpts_bnd = geo.nBndFaces * nFptsPerFace;

#ifdef _MPI
  /* Determine total number of MPI flux points */
  geo.nGfpts_mpi = geo.mpiface_list.size() * nFptsPerFace;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  geo.fpt2gfpt.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});
  geo.fpt2gfpt_slot.assign({geo.nFacesPerEle * nFptsPerFace, geo.nEles});

  // --- Handy map to grab the nodes making up each face ---
  std::map<int,mdvector<int>> ct2fv;
  ct2fv[HEX].assign({6, 4});
  ct2fv[QUAD].assign({4, 2});

  // Ordering: Bottom, Top, Left, Right, Front, Back
  std::vector<int> tmp = {0,1,2,3, 5,4,7,6, 0,3,7,4, 2,1,5,6, 1,0,4,5, 3,2,6,7};
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 4; j++)
      ct2fv[HEX](i,j) = tmp[i*4+j];

  for (int i = 0; i < 4; i++)
  {
    ct2fv[QUAD](i,0) = i;
    ct2fv[QUAD](i,1) = (i==3) ? 0 : i+1;
  }

  /* Based on rotation, couple flux points */
  mdvector<int> ROT({5, nFptsPerFace});
  for (uint i = 0; i < nFpts1D; i++)
  {
    if (geo.nDims == 3)
    {
      for (uint j = 0; j < nFpts1D; j++)
      {
        ROT(0, i + j*nFpts1D) = i*nFpts1D + j;
        ROT(1, i + j*nFpts1D) = nFpts1D - i + j*nFpts1D - 1;
        ROT(2, i + j*nFpts1D) = nFptsPerFace - i * nFpts1D - j - 1;
        ROT(3, i + j*nFpts1D) = nFptsPerFace - (j+1) * nFpts1D + i;
      }
    }
    else
    {
      ROT(4, i) = nFptsPerFace - i - 1;
    }
  }

  // Counter for total global flux points so far
  unsigned int gfpt = 0;

  // Begin by looping over all internal / periodic faces
  for (int ff = 0; ff < geo.nFaces; ff++)
  {
    auto faceL = geo.face_list[ff];
    auto faceR = geo.face_list[ff+geo.nFaces];
    int eL = faceL.ic;
    int eR = faceR.ic;

    // Determine relative rotation using ordered faces-vertex lists
    unsigned int rot = 0;
    if (geo.nDims == 3)
    {
      // Find relative rotation ('rot tag')
      int nd = ct2fv[HEX](faceL.loc_f, 0);
      double x = geo.ele_nodes(nd, eL, 0);
      double y = geo.ele_nodes(nd, eL, 1);
      double z = (geo.nDims == 2) ? 0. : geo.ele_nodes(nd, eL, 2);
      point pt1(x,y,z);

      nd = ct2fv[HEX](faceR.loc_f, 0);
      x = geo.ele_nodes(nd, eR, 0);
      y = geo.ele_nodes(nd, eR, 1);
      z = (geo.nDims == 2) ? 0. : geo.ele_nodes(nd, eR, 2);
      point pt2(x,y,z);

      while (point(pt2-pt1).norm() > 1e-9)
      {
        rot++;

        nd = ct2fv[HEX](faceR.loc_f, rot);
        pt2.x = geo.ele_nodes(nd, eR, 0);
        pt2.y = geo.ele_nodes(nd, eR, 1);
        pt2.z = (geo.nDims == 2) ? 0. : geo.ele_nodes(nd, eR, 2);
      }
    }
    else
    {
      rot = 4;
    }

    // Setup global flux point IDs within left/right eles
    for (int fpt = 0; fpt < nFptsPerFace; fpt++)
    {
      geo.fpt2gfpt(faceL.loc_f*nFptsPerFace + fpt, eL) = gfpt;
      geo.fpt2gfpt(faceR.loc_f*nFptsPerFace + ROT(rot,fpt), eR) = gfpt;
      geo.fpt2gfpt_slot(faceL.loc_f*nFptsPerFace + fpt, eL) = 0;
      geo.fpt2gfpt_slot(faceR.loc_f*nFptsPerFace + ROT(rot,fpt), eR) = 1;
      gfpt++;
    }
  }

  // Counter of total boundary flux points so far
  unsigned int gfpt_bnd = 0;

  // Setup boundary-face flux points
  geo.gfpt2bnd.assign({geo.nGfpts_bnd});
  for (uint bnd = 0; bnd < geo.nBounds; bnd++)
  {
    for (auto &face : geo.bound_faces[bnd])
    {
      int ele = face.ic;
      int n = face.loc_f;
      for (int fpt = 0; fpt < nFptsPerFace; fpt++)
      {
        geo.fpt2gfpt(n*nFptsPerFace + fpt, ele) = gfpt;
        geo.fpt2gfpt_slot(n*nFptsPerFace + fpt, ele) = 0;
        geo.gfpt2bnd(gfpt_bnd) = geo.bnd_ids[bnd];
        gfpt_bnd++;
        gfpt++;
      }
    }
  }


//    /* Check if face is on boundary */
//    if (geo.bnd_faces.count(face))
//    {
//      unsigned int bnd_id = geo.bnd_faces[face];
//      for (auto &fpt : fpts)
//      {
//        geo.gfpt2bnd.push_back(bnd_id);
//        fpt = gfpt_bnd;
//        gfpt_bnd++;
//      }

//      bndface2fpts[face] = fpts;

//      int bnd = geo.face2bnd[face];
//      geo.boundFaces[bnd].push_back(geo.nFaces);
//    }
//#ifdef _MPI
//    /* Check if face is on MPI boundary */
//    else if (geo.mpi_faces.count(face))
//    {
//      /* Add face to set to process later. */
//      mpi_faces_to_process.insert(face);

//      for (auto &fpt : fpts)
//      {
//        fpt = gfpt_mpi;
//        gfpt_mpi++;
//      }
//    }
//#endif
}

void setup_element_colors(InputStruct *input, GeoStruct &geo)
{
  /* Setup element colors */
  geo.ele_color.assign({geo.nEles});
  if (input->nColors == 1)
  {
    geo.nColors = 1;
    geo.ele_color.fill(1);
  }
  else
  {
    geo.nColors = input->nColors;
    std::vector<bool> used(geo.nColors, false);
    std::vector<unsigned int> counts(geo.nColors, 0);
    std::queue<unsigned int> eleQ;
    geo.ele_color.fill(0);
    geo.ele_color(0) = 0;

    eleQ.push(0);

    /* Loop over elements and assign colors using greedy algorithm */
    while (!eleQ.empty())
    {
      unsigned int ele = eleQ.front();
      eleQ.pop();

      if (geo.ele_color(ele) != 0)
        continue;

      for (unsigned int face = 0; face < geo.nFacesPerEle; face++)
      {
        int eleN = geo.ele_adj(face, ele);

        if (eleN != -1 && geo.ele_color(eleN) == 0)
        {
          eleQ.push(eleN);
        }

        if (eleN == -1)
          continue;

        unsigned int colorN = geo.ele_color(eleN);

        /* Record if neighbor is using a given color */
        if (colorN != 0)
          used[colorN - 1] = true;
      }

      unsigned int color = 0;
      unsigned int min_count = 0;
      unsigned int min_color_all = 1;
      unsigned int min_count_all = counts[0];

      /* Set current element color to color unused by neighbors with minimum count in domain */
      for (unsigned int c = 0; c < geo.nColors; c++)
      {
        if (!used[c] and color == 0)
        {
          color = c + 1;
          min_count = counts[c];
        }
        else if (!used[c])
        {
          if (counts[c] < min_count)
          {
            color = c + 1;
            min_count = counts[c];
          }
        }

        if (counts[c] < min_count_all)
        {
          min_count_all = counts[c];
          min_color_all = c + 1;
        }
      }

      if (color == 0)
      {
        ThrowException("Could not color graph with number of colors provided. Increase nColors!");
      }

      geo.ele_color(ele) = color;
      counts[color-1]++;
      used.assign(geo.nColors, false);
    }
  }
}

void shuffle_data_by_color(GeoStruct &geo)
{
  /* Reorganize required geometry data by color */
  std::vector<std::vector<unsigned int>> color2eles(geo.nColors);
  for (unsigned int ele = 0; ele < geo.nEles; ele++)
  {
    color2eles[geo.ele_color(ele) - 1].push_back(ele);
  }

  /* TODO: Consider an in-place permutation to save memory */
  auto nd2gnd_temp = geo.ele2nodes;
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
        geo.ele2nodes(node, ele1) = nd2gnd_temp(node, ele2);
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
  geo.ele_color_nEles.assign(geo.nColors, 0);
  geo.ele_color_range.assign(geo.nColors + 1, 0);

  geo.ele_color_range[1] = color2eles[0].size(); 
  geo.ele_color_nEles[0] = color2eles[0].size(); 
  for (unsigned int color = 2; color <= geo.nColors; color++)
  {
    geo.ele_color_range[color] = geo.ele_color_range[color - 1] + color2eles[color-1].size(); 
    geo.ele_color_nEles[color - 1] = geo.ele_color_range[color] - geo.ele_color_range[color-1];
  }

  /* Print out color distribution */
  std::cout << "color distribution: ";
  for (unsigned int color = 1; color <= geo.nColors; color++)
  {
    std::cout<< geo.ele_color_nEles[color - 1] << " ";
  }
  std::cout << std::endl;
}

#ifdef _MPI
void partition_geometry(InputStruct *input, GeoStruct &geo)
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
      eind[j + n] = geo.ele2nodes(j, i);
      nodes.insert(geo.ele2nodes(j,i));
    } 

    /* Check for collapsed edge (not fully general yet)*/
    /* TODO: This is hardcoded for first order collapsed triangles, but seems to work. Generalize for
     * better partitioning. */
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
    /*
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
    */


  }

  int objval;
  std::vector<int> epart(geo.nEles, 0);
  std::vector<int> npart(geo.nNodes);
  /* TODO: Should just not call this entire function if nRanks == 1 */
  if (nRanks > 1) 
  {
    int nNodesPerFace = geo.nNodesPerFace; // TODO: Related to previous TODO
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
        face[i] = geo.ele2nodes(geo.face_nodes(n, i), ele);
      }

      /* Check if face is collapsed */
      std::set<unsigned int> nodes;
      for (auto node : face)
        nodes.insert(node);

      if (nodes.size() <= geo.nDims - 1) /* Fully collapsed face. Assign no fpts. */
      {
        //geo.c2f(ele, n) = -1;
        continue;
      }
      else if (nodes.size() == 3) /* Triangular collapsed face. Must tread carefully... */
      {
        face.assign(nodes.begin(), nodes.end());
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
  auto nd2gnd_glob = geo.ele2nodes;
  geo.ele2nodes.assign({geo.nNodesPerEle, (unsigned int) myEles.size()},0);
  for (unsigned int ele = 0; ele < myEles.size(); ele++)
  {
    for (unsigned int nd = 0; nd < geo.nNodesPerEle; nd++)
    {
      geo.ele2nodes(nd, ele) = nd2gnd_glob(nd, myEles[ele]);
    }
  }

  if (input->dt_scheme == "MCGS")
  {
    /* Reduce color data to only contain partition local elements */
    auto ele_color_glob = geo.ele_color;
    geo.ele_color.assign({(unsigned int) myEles.size()}, 0);
    for (unsigned int ele = 0; ele < myEles.size(); ele++)
    {
      geo.ele_color(ele) = ele_color_glob(myEles[ele]);
    }
  }

  /* Obtain set of unique nodes on this partition */
  std::set<unsigned int> uniqueNodes;
  for (unsigned int ele = 0; ele < myEles.size(); ele++)
  {
    for (unsigned int nd = 0; nd < geo.nNodesPerEle; nd++)
    {
      uniqueNodes.insert(geo.ele2nodes(nd, ele));
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
      geo.ele2nodes(nd, ele) = geo.node_map_g2p[geo.ele2nodes(nd, ele)];

  /* Reduce boundary faces data to contain only faces on local partition. Also
   * reindex via geo.node_map_g2p */
  auto bnd_faces_glob = geo.bnd_faces;
  auto face2bnd_glob = geo.face2bnd;
  geo.bnd_faces.clear();
  geo.face2bnd.clear();

  /* Iterate over all boundary faces */
  for (auto entry : bnd_faces_glob)
  {
    std::vector<unsigned int> bnd_face = entry.first;
    int bcType = entry.second;

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
      geo.bnd_faces[bnd_face] = bcType;
    }
  }

  for (auto entry : face2bnd_glob)
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
      geo.face2bnd[bnd_face] = bnd_id;
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

/*! ==== Overset-Related Functionality ==== */

#ifdef _MPI
void splitGridProcs(const MPI_Comm &Comm_World, MPI_Comm &Comm_Grid, InputStruct *input, GeoStruct &geo)
{
  
  // Split the processes among the overset grids such that they are roughly balanced

  // --- Read Number of Elements in Each Grid --- 

  std::vector<int> nElesGrid(input->nGrids);
  int nElesTotal = 0;

  for (unsigned int i=0; i<input->nGrids; i++)
  {
    std::ifstream meshFile;
    std::string str;
    std::string fileName = input->oversetGrids[i];

    meshFile.open(fileName.c_str());
    if (!meshFile.is_open())
      ThrowException("Unable to open mesh file.");

    // Move cursor to $Elements
    meshFile.clear();
    meshFile.seekg(0, ios::beg);
    while(1)
    {
      getline(meshFile,str);
      if (str.find("$Elements")!=string::npos) break;
      if(meshFile.eof()) ThrowException("$Elements tag not found in Gmsh file!");
    }

    // Read total number of interior + boundary elements
    meshFile >> nElesGrid[i];
    meshFile.close();

    nElesTotal += nElesGrid[i];
  }

  // --- Balance the processes across the grids --- 

  geo.nProcGrid.resize(input->nGrids);
  for (unsigned int i=0; i<input->nGrids; i++)
  {
    double eleRatio = (double)nElesGrid[i]/nElesTotal;
    geo.nProcGrid[i] = round(eleRatio*input->nRanks);
  }

  // --- Get the final gridID for this rank --- 

  int g = 0;
  int procSum = geo.nProcGrid[0];
  while (procSum < input->rank+1 && g < input->nGrids-1)
  {
    g++;
    procSum += geo.nProcGrid[g];
  }
  geo.gridID = g;

  // --- Split MPI Processes Based Upon gridID: Create MPI_Comm for each grid --- 

  MPI_Comm_split(Comm_World, geo.gridID, input->rank, &Comm_Grid);

  MPI_Comm_rank(Comm_Grid,&geo.gridRank);
  MPI_Comm_size(Comm_Grid,&geo.nProcsGrid);

  geo.gridIdList.resize(input->nRanks);
  MPI_Allgather(&geo.gridID,1,MPI_INT,geo.gridIdList.data(),1,MPI_INT,MPI_COMM_WORLD);
  
}
#endif
