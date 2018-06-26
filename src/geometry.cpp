#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <regex>
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
#include "points.hpp"

#ifndef _NO_HDF5
#include "H5Cpp.h"
#ifndef _H5_NO_NAMESPACE
using namespace H5;
#endif
#endif

enum MESH_FORMAT {
  GMSH, PYFR
};

GeoStruct process_mesh(InputStruct *input, unsigned int order, int nDims, _mpi_comm comm_in)
{
  GeoStruct geo;
  geo.nDims = nDims;
  geo.input = input;
  geo.myComm = comm_in;
  geo.gridID = input->gridID;
  geo.rank = input->rank;
  geo.gridRank = input->rank;

  int format;

  if (input->meshfile.find(".msh") != std::string::npos)
    format = GMSH;
  else if (input->meshfile.find(".pyfr") != std::string::npos)
    format = PYFR;
  else
    ThrowException("Unrecognized mesh format - expecting *.msh or *.pyfrm.");

  setup_etypes(geo);

  if (format == GMSH)
    load_mesh_data_gmsh(input, geo);
  else
    load_mesh_data_pyfr(input, geo);

  if (input->iterative_method == MCGS)
    setup_element_colors(input, geo);

#ifdef _MPI
  if (format == GMSH)
    partition_geometry(input, geo);
#endif

  if (input->iterative_method == MCGS)
    shuffle_data_by_color(geo);

  if (format == GMSH)
  {
    setup_global_fpts(input, geo, order);
  }
  else
  {
    setup_global_fpts_pyfr(input, geo, order);
  }

  setup_flip_beta(input, geo);

  if (input->implicit_method && input->viscous)
    set_ele_adjacency(geo);

  if (input->overset)
  {
    geo.iblank_node.assign(geo.nNodes, NORMAL);
    geo.iblank_cell.assign({geo.nEles}, NORMAL);
    geo.iblank_face.assign({geo.nFaces}, NORMAL);
  }

  geo.x_cg.assign({3});
  if (input->motion)
  {
    geo.coords_init = geo.coord_nodes;
    geo.grid_vel_nodes.assign({geo.nNodes, geo.nDims}, 0.0);

    geo.vel_cg.assign({3});
    geo.Rmat.assign({3,3});

    // Initialize Rmat to identity matrix
    for (int d = 0; d < 3; d++)
      geo.Rmat(d,d) = 1;
  }

  if ( nDims == 3 && (input->motion_type == RIGID_BODY) )
  {
    geo.q.assign({4});
    geo.qdot.assign({4});
    geo.omega.assign({3});
    geo.mass = input->mass;

    geo.Wmat.assign({3,3});
    geo.dRmat.assign({3,3});

    geo.q(0) = 1.;  // Initialize to unit quaternion of no rotation

    for (int d = 0; d < 3; d++)
      geo.dRmat(d,d) = 1;

    // Initial translational & angular velocity
    for (int d = 0; d < 3; d++)
    {
      if (geo.gridID == 0)
      {
        geo.omega(d) = input->w0[d];      // Global coords
        geo.qdot(d+1) = 0.5*input->w0[d]; // Body coords
        geo.vel_cg(d) = input->v0[d];
      }
    }

    Quat q(geo.q(0),geo.q(1),geo.q(2),geo.q(3));
    Quat qdot(geo.qdot(0),geo.qdot(1),geo.qdot(2),geo.qdot(3));
    Quat omega = 2*q.conj()*qdot;
    mdvector<double> W({3,3}, 0.);
    int c1[3] = {1,2,0}; // Cross-product index maps
    int c2[3] = {2,0,1};
    for (int i = 0; i < 3; i++)
    {
      W(i,c2[i]) =  omega[c1[i]+1];
      W(i,c1[i]) = -omega[c2[i]+1];
    }

    geo.Wmat.fill(0.);
    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        for (unsigned int k = 0; k < 3; k++)
          geo.Wmat(i,j) += geo.Rmat(i,k) * W(k,j);

    // Inertia tensor for body [Using initial coords as static body coord sys]
    geo.Jmat.assign({3,3});
    geo.Jmat(0,0) = input->Imat[0];
    geo.Jmat(1,1) = input->Imat[4];
    geo.Jmat(2,2) = input->Imat[8];

    // Off-diagonal components
    geo.Jmat(0,1) = input->Imat[1];
    geo.Jmat(0,2) = input->Imat[2];
    geo.Jmat(1,2) = input->Imat[5];

    // Apply symmetry
    geo.Jmat(1,0) = geo.Jmat(0,1);
    geo.Jmat(2,0) = geo.Jmat(0,2);
    geo.Jmat(2,1) = geo.Jmat(1,2);

    // Calculate inverse
    double det = determinant(geo.Jmat);
    geo.Jinv = adjoint(geo.Jmat);
    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        geo.Jinv(i,j) /= det;

    if (input->dt_scheme != "RK54")
    {
      // Assign storage for stage residuals
      geo.q_res.assign({input->nStages,4});
      geo.qdot_res.assign({input->nStages,4});
      geo.x_res.assign({input->nStages,3});
      geo.v_res.assign({input->nStages,3});
    }
  }

  /* Setup wall face list for computing forces / moments [by ftype] */
  for (int i = 0; i < geo.wallFaceList.size(); i++)
  {
    int fid = geo.wallFaceList[i];
    auto ftype = geo.faceType(fid);
    geo.wallFacesBT[ftype].push_back(fid);
  }

  return geo;
}

void setup_etypes(GeoStruct &geo)
{
  geo.nFacesPerEleBT[TRI] = 3;
  geo.nFacesPerEleBT[QUAD] = 4;
  geo.nFacesPerEleBT[TET] = 4;
  geo.nFacesPerEleBT[PRI] = 5;
  geo.nFacesPerEleBT[HEX] = 6;

  geo.nNodesPerFaceBT[LINE] = 2;
  geo.nNodesPerFaceBT[TRI] = 3;
  geo.nNodesPerFaceBT[QUAD] = 4;

  std::vector<ELE_TYPE> types = {TRI,QUAD,TET,PRI,HEX};

  /* Define node indices for faces */
  for (auto etype : types)
    geo.face_nodesBT[etype].resize(geo.nFacesPerEleBT[etype]);

  // -------------- Quads --------------

  /* Face 0: Bottom */
  geo.face_nodesBT[QUAD][0] = {0, 1};

  /* Face 1: Right */
  geo.face_nodesBT[QUAD][1] = {1, 2};

  /* Face 2: Top */
  geo.face_nodesBT[QUAD][2] = {2, 3};

  /* Face 3: Left */
  geo.face_nodesBT[QUAD][3] = {3, 0};

  geo.eleFaceTypesBT[QUAD] = {LINE, LINE, LINE, LINE};

  // -------------- Tris --------------

  /* Face 0: Bottom */
  geo.face_nodesBT[TRI][0] = {0, 1};

  /* Face 1: Hypotnuse */
  geo.face_nodesBT[TRI][1] = {1, 2};

  /* Face 2: Left */
  geo.face_nodesBT[TRI][2] = {2, 0};

  geo.eleFaceTypesBT[TRI] = {LINE, LINE, LINE};

  // -------------- Hexas --------------

  /* Face 0: Bottom */
  geo.face_nodesBT[HEX][0] = {0, 1, 2, 3};

  /* Face 1: Top */
  geo.face_nodesBT[HEX][1] = {5, 4, 7, 6};

  /* Face 2: Left */
  geo.face_nodesBT[HEX][2] = {0, 3, 7, 4};

  /* Face 3: Right */
  geo.face_nodesBT[HEX][3] = {2, 1, 5, 6};

  /* Face 4: Front */
  geo.face_nodesBT[HEX][4] = {1, 0, 4, 5};

  /* Face 5: Back */
  geo.face_nodesBT[HEX][5] = {3, 2, 6, 7};

  geo.eleFaceTypesBT[HEX] = {QUAD, QUAD, QUAD, QUAD, QUAD, QUAD};

  // -------------- Tets --------------

  /* Face 0: Rear (xi-eta plane) */
  geo.face_nodesBT[TET][0] = {0, 1, 2};

  /* Face 1: Bottom (xi-zeta plane) */
  geo.face_nodesBT[TET][1] = {0, 3, 1};

  /* Face 2: Left (eta-zeta plane) */
  geo.face_nodesBT[TET][2] = {0, 2, 3};

  /* Face 3: Angled */
  geo.face_nodesBT[TET][3] = {1, 2, 3};

  geo.eleFaceTypesBT[TET] = {TRI, TRI, TRI, TRI};

  // -------------- Prisms --------------

  /* Face 0: Bottom */
  geo.face_nodesBT[PRI][0] = {0,1,2};

  /* Face 1: Top */
  geo.face_nodesBT[PRI][1] = {3,4,5};

  /* Face 2: Front */
  geo.face_nodesBT[PRI][2] = {0,3,4,1};

  /* Face 3: Angled Face */
  geo.face_nodesBT[PRI][3] = {1,4,5,2};

  /* Face 4: Left */
  geo.face_nodesBT[PRI][4] = {0,2,5,3};

  geo.eleFaceTypesBT[PRI] = {TRI, TRI, QUAD, QUAD, QUAD};
}

void load_mesh_data_gmsh(InputStruct *input, GeoStruct &geo)
{
  std::ifstream f(input->meshfile);

  if (!f.is_open())
    ThrowException("Could not open specified mesh file!");

  if (input->rank == 0)
    std::cout << "Reading mesh file " << input->meshfile << std::endl;

  /* Process file information */
  /* Load boundary tags */
  read_boundary_ids(f, geo, input);

  /* Load node coordinate data */
  read_node_coords(f, geo);

  /* Load element connectivity data */
  read_element_connectivity(f, geo, input);
  read_boundary_faces(f, geo);


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

  // Read boundary condition names from Gmsh and match to input file
  geo.bcGlobal.resize(nBndIds);
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

    geo.bcGlobal[i] = bcName;

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
  geo.boundFaceTypes.resize(geo.nBounds);
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
      geo.nFacesPerEleBT[QUAD] = 4; geo.nCornerNodesBT[QUAD] = 4;
      geo.nFacesPerEleBT[TRI] = 3; geo.nCornerNodesBT[TRI] = 3;

      switch(val)
      {
        /* Linear quad/tri */
        case 2:
          geo.ele_set.insert(TRI);
          geo.nEles++;
          geo.nElesBT[TRI]++;
          geo.nNdFaceCurved[LINE] = 2;
          geo.nNodesPerEleBT[TRI] = 3; break;

        case 3:
          geo.ele_set.insert(QUAD);
          geo.nEles++;
          geo.nElesBT[QUAD]++;
          geo.nNdFaceCurved[LINE] = 2;
          geo.nNodesPerEleBT[QUAD] = 4; break;

        /* Biquadratic quad/tri */
        case 9:
          geo.ele_set.insert(TRI);
          geo.nEles++;
          geo.nElesBT[TRI]++;
          geo.nNdFaceCurved[LINE] = 3;
          geo.nNodesPerEleBT[TRI] = 6; break;

        case 10:
          geo.ele_set.insert(QUAD);
          geo.nEles++;
          geo.nElesBT[QUAD]++;
          geo.nNdFaceCurved[LINE] = 3;
          geo.nNodesPerEleBT[QUAD] = 9; break;

        /* Bicubic quad */
        case 36:
          geo.ele_set.insert(QUAD);
          geo.nEles++;
          geo.nElesBT[QUAD]++;
          geo.nNdFaceCurved[LINE] = 4;
          geo.nNodesPerEleBT[QUAD] = 16; break;

        /* Biquartic quad */
        case 37:
          geo.ele_set.insert(QUAD);
          geo.nEles++;
          geo.nElesBT[QUAD]++;
          geo.nNdFaceCurved[LINE] = 5;
          geo.nNodesPerEleBT[QUAD] = 25; break;

        /* Biquintic quad */
        case 38:
          geo.ele_set.insert(QUAD);
          geo.nEles++;
          geo.nElesBT[QUAD]++;
          geo.nNdFaceCurved[LINE] = 6;
          geo.nNodesPerEleBT[QUAD] = 36; break;

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
      geo.nFacesPerEleBT[HEX] = 6; geo.nCornerNodesBT[HEX] = 8;
      geo.nFacesPerEleBT[TET] = 4; geo.nCornerNodesBT[TET] = 4;
      geo.nFacesPerEleBT[PRI] = 5; geo.nCornerNodesBT[PRI] = 6;

      switch(val)
      {
        /* Linear Tet */
        case 4:
          geo.ele_set.insert(TET);
          geo.nEles++;
          geo.nElesBT[TET]++;
          geo.nNdFaceCurved[TRI] = 3;
          geo.nNodesPerEleBT[TET] = 4; break;

        /* Trilinear Hex */
        case 5:
          geo.ele_set.insert(HEX);
          geo.nEles++;
          geo.nElesBT[HEX]++;
          geo.nNdFaceCurved[QUAD] = 4;
          geo.nNodesPerEleBT[HEX] = 8; break;

        /* Linear Prism */
        case 6:
          geo.ele_set.insert(PRI);
          geo.nEles++;
          geo.nElesBT[PRI]++;
          geo.nNdFaceCurved[TRI] = 3;
          geo.nNdFaceCurved[QUAD] = 4;
          geo.nNodesPerEleBT[PRI] = 6; break;

        /* Quadratic Tet */
        case 11:
          geo.ele_set.insert(TET);
          geo.nEles++;
          geo.nElesBT[TET]++;
          geo.nNdFaceCurved[TRI] = 4;
          geo.nNodesPerEleBT[TET] = 10; break;

        /* Triquadratic Hex */
        case 12:
          geo.ele_set.insert(HEX);
          geo.nEles++;
          geo.nElesBT[HEX]++;
          geo.nNdFaceCurved[QUAD] = 9;
          geo.nNodesPerEleBT[HEX] = 27;
          break;

        /* Quadratic Prism */
        case 13:
          geo.ele_set.insert(PRI);
          geo.nEles++;
          geo.nElesBT[PRI]++;
          geo.nNdFaceCurved[TRI] = 6;
          geo.nNdFaceCurved[QUAD] = 9;
          geo.nNodesPerEleBT[PRI] = 18; break;

        /* Cubic Prism */
        case 90:
          geo.ele_set.insert(PRI);
          geo.nEles++;
          geo.nElesBT[PRI]++;
          geo.nNdFaceCurved[TRI] = 10;
          geo.nNdFaceCurved[QUAD] = 16;
          geo.nNodesPerEleBT[PRI] = 40; break;

        /* Tricubic Hex */
        case 92:
          geo.ele_set.insert(HEX);
          geo.nEles++;
          geo.nElesBT[HEX]++;
          geo.nNdFaceCurved[QUAD] = 16;
          geo.nNodesPerEleBT[HEX] = 64;
          break;

        /* Triquartic Hex */
        case 93:
          geo.ele_set.insert(HEX);
          geo.nEles++;
          geo.nElesBT[HEX]++;
          geo.nNdFaceCurved[QUAD] = 25;
          geo.nNodesPerEleBT[HEX] = 125;
          break;

        /* Triquintic Hex */
        case 94:
          geo.ele_set.insert(HEX);
          geo.nEles++;
          geo.nElesBT[HEX]++;
          geo.nNdFaceCurved[QUAD] = 36;
          geo.nNodesPerEleBT[HEX] = 216;
          break;

        case 2:
        case 3:
        case 9:
        case 10: 
        case 16: 
        case 36: 
        case 37: 
        case 38: 
          geo.nBnds++; break;

        default:
          ThrowException("Inconsistent Element/Face type detected! Is nDims set correctly?");
      }
    }
    std::getline(f,line);
  }

  if (geo.ele_set.count(TRI) || geo.ele_set.count(QUAD))
  {
    geo.face_set.insert(LINE);
  }

  if (geo.ele_set.count(HEX) || geo.ele_set.count(PRI) || geo.ele_set.count(PYR))
  {
    geo.face_set.insert(QUAD);
  }

  if (geo.ele_set.count(TET) || geo.ele_set.count(PRI) || geo.ele_set.count(PYR))
  {
    geo.face_set.insert(TRI);
  }

  geo.face_types.assign({(uint)geo.face_set.size()});
  geo.nNode_face.assign({(uint)geo.face_set.size()});

  f.seekg(pos);

  /* Allocate memory for element connectivity */
  for (auto etype : geo.ele_set) 
  {
    geo.ele2nodesBT[etype].assign({geo.nElesBT[etype], geo.nNodesPerEleBT[etype]});
    geo.eleID[etype].assign({geo.nElesBT[etype]});
    geo.nElesBT[etype] = 0; // zero out ele by type count to use for indexing
  }

  int n = 0;
  for (auto ftype : geo.face_set)
  {
    geo.face_types(n) = ftype;
    geo.nNode_face(n) = geo.nNdFaceCurved[ftype];
    n++;
  }

  /* Read element connectivity (skip boundaries in this loop) */
  //unsigned int ele = 0;
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
          break;

        /* Process triangular elements */
        case 2: /* 3-node Triangle */
        case 9: /* 6-node Triangle */
        {
          unsigned int ele = geo.nElesBT[TRI];
          for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[TRI]; nd++)
          {
            f >> geo.ele2nodesBT[TRI](ele,nd);
          }
          geo.nElesBT[TRI]++; break;
        }

        /* Process quadrilateral elements */
        case 3: /* 4-node Quadrilateral */
        case 10: /* 9-node Quadilateral */
        case 36: /* 16-node Quadilateral */
        case 37: /* 25-node Quadilateral */
        case 38: /* 36-node Quadilateral */
        {
          unsigned int ele = geo.nElesBT[QUAD];
          for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[QUAD]; nd++)
          {
            f >> geo.ele2nodesBT[QUAD](ele,nd);
          }
          geo.nElesBT[QUAD]++;
          break;
        }

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
        case 16: // Quadratic (Serendipity) quad
        case 36: // Cubic quad
        case 37: // Quartic quad
        case 38: // Quintic quad
          break;

        /* Process tetrahedral elements */
        case 4:  /* Linear Tet */
        case 11:  /* Quadratic Tet */
        {
          unsigned int ele = geo.nElesBT[TET];
          for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[TET]; nd++)
          {
            f >> geo.ele2nodesBT[TET](ele,nd);
          }
          geo.nElesBT[TET]++; break;
        }

        /* Process hexahedral elements */
        case 5: /* 8-node Hexahedral */
        case 12: /* Triquadratic Hex */
        case 92: /* Cubic Hex */
        case 93: /* Quartic Hex */
        case 94: /* Quintic Hex */
        {
          unsigned int ele = geo.nElesBT[HEX];
          for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[HEX]; nd++)
          {
            f >> geo.ele2nodesBT[HEX](ele,nd);
          }
          geo.nElesBT[HEX]++; break;
        }

          /* Process tetrahedral elements */
          case 6:  /* Linear Pri */
          case 13:  /* Quadratic Pri */
          case 90:  /* Quadratic Pri */
          {
            unsigned int ele = geo.nElesBT[PRI];
            for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[PRI]; nd++)
            {
              f >> geo.ele2nodesBT[PRI](ele,nd);
            }
            geo.nElesBT[PRI]++; break;
          }

        default:
          ThrowException("Unrecognized element type detected!"); break;
      }

    }

    std::getline(f,line);
  }

  /* Correct node values to be 0-indexed. Also assign global eleIDs. */

  unsigned int eleID = 0;
  geo.eleID_type.assign({geo.nEles}, -1);
  geo.eleType.assign({geo.nEles});
  for (auto etype : geo.ele_set)
  {
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (unsigned int n = 0; n < geo.nNodesPerEleBT[etype]; n++)
      {
        geo.ele2nodesBT[etype](ele,n)--;
      }
      geo.eleID[etype](ele) = eleID;
      geo.eleType(eleID) = etype;
      geo.eleID_type(eleID) = ele;

      eleID++;
    }
  }

  geo.gEtype = geo.eleType;

  /* Setup face-node maps for easier processing */
  set_face_nodes(geo);

  /* Rewind file */
  f.seekg(pos);
}

void read_boundary_faces(std::ifstream &f, GeoStruct &geo)
{
  if (geo.nDims == 2)
  {
    std::vector<unsigned int> face(2, 0);
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
        case 26: /* 4-node Line */
        case 27: /* 5-node Line */
        case 28: /* 6-node Line */
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
      geo.bnd_face_type[face] = LINE;
    }
  }
  else if (geo.nDims == 3)
  {
    std::vector<unsigned int> face;
    for (unsigned int n = 0; n < (geo.nEles + geo.nBnds); n++)
    {
      unsigned int vint, ele_type, bnd_id, nTags;
      ELE_TYPE ftype;
      std::string line;
      f >> vint >> ele_type;

      /* Get boundary id and face nodes */
      switch (ele_type)
      {
        case 2: /* 3-node Triangle */
        case 9: /* 6-node Triangle */
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;
          
          face.assign(3,0);
          f >> face[0] >> face[1] >> face[2];
          std::getline(f,line);
          ftype = TRI;
          break;

        case 3: /* 4-node Quadrilateral */
        case 10: /* 9-node Quadrilateral */
        case 16: // Quadratic (Serendipity) quad
        case 36: // Cubic quad
        case 37: // Quartic quad
        case 38: // Quintic quad
          f >> nTags;
          f >> bnd_id;

          for (unsigned int i = 0; i < nTags - 1; i++)
            f >> vint;

          face.assign(4,0);
          f >> face[0] >> face[1] >> face[2] >> face[3];
          std::getline(f,line);
          ftype = QUAD;
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
      geo.bnd_face_type[face] = ftype;
    }

  }

  if (geo.per_bnd_flag)
  {
    couple_periodic_bnds(geo);
  }
}

void set_face_nodes(GeoStruct &geo)
{
  for (auto etype : geo.ele_set)
  {
    // Get the shape node indices comprising each face of every element type
    if (etype == TET)
    {
      geo.faceNodesCurved[etype].assign({4, geo.nNdFaceCurved[TRI]});

      std::vector<std::vector<int>> nodes(4);
      switch (geo.nNdFaceCurved[TRI])
      {
        case 3:
          nodes[0] = {0,1,2};
          nodes[1] = {0,3,1};
          nodes[2] = {0,2,3};
          nodes[3] = {1,3,2};
          break;

        case 6:
          nodes[0] = {0,1,2,4,5,6};
          nodes[1] = {0,3,1,7,9,4};
          nodes[2] = {0,2,3,6,8,7};
          nodes[3] = {1,3,2,9,8,5};
          break;

        case 10:
          nodes[0] = {0,1,2,4,5,6,7,8,9,16};
          nodes[1] = {0,3,1,11,10,14,15,5,4};
          nodes[2] = {0,2,3,11,10,12,13,8,9};
          nodes[3] = {1,3,2,15,14,12,13,7,6};
          break;

        case 16:
          nodes[0] = {0,1,2,4,5,6,7,8,9,10,11,12,22,24,23};
          nodes[1] = {0,3,1,15,14,13,19,20,21,6,5,4,25,27,26};
          nodes[2] = {0,2,3,12,11,10,18,17,16,13,14,15,27,30,29};
          nodes[3] = {1,3,2,21,20,19,16,17,18,9,8,7,32,31,33};
          break;
      }

      for (int f = 0; f < 4; f++)
        for (int i = 0; i < geo.nNdFaceCurved[TRI]; i++)
          geo.faceNodesCurved[TET](f, i) = nodes[f][i];
    }
    else if (etype == HEX)
    {
      geo.faceNodesCurved[etype].assign({6, geo.nNdFaceCurved[QUAD]});
      auto ijk2hex = structured_to_gmsh_hex(geo.nNodesPerEleBT[etype]);
      auto ij2quad = structured_to_gmsh_quad(geo.nNdFaceCurved[QUAD]);
      int nSide = sqrt(geo.nNdFaceCurved[QUAD]);

      /* Face 0: Bottom: -z */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](0,ij2quad[j*nSide+i]) = ijk2hex[j*nSide+i];

      /* Face 1: Top: +z (Reverse Orientation) */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](1,ij2quad[j*nSide+i]) = ijk2hex[i*nSide+j+nSide*nSide*(nSide-1)];

      /* Face 2: Left: -x */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](2,ij2quad[j*nSide+i]) = ijk2hex[j*(nSide*nSide)+i*nSide];

      /* Face 2: Right: +x */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](3,ij2quad[j*nSide+i]) = ijk2hex[i*(nSide*nSide)+j*nSide+(nSide-1)];

      /* Face 2: Front: -y */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](4,ij2quad[j*nSide+i]) = ijk2hex[nSide-i-1+j*(nSide*nSide)];

      /* Face 2: Back: +y */
      for (int j = 0; j < nSide; j++)
        for (int i = 0; i < nSide; i++)
          geo.faceNodesCurved[etype](5,ij2quad[j*nSide+i]) = ijk2hex[(j+1)*(nSide*nSide)-nSide+i];
    }
    else if (etype == QUAD)
    {
      geo.faceNodesCurved[etype].assign({4, geo.nNdFaceCurved[LINE]});
      auto ij2quad = structured_to_gmsh_quad(geo.nNdFaceCurved[QUAD]);
      int nSide = sqrt(geo.nNdFaceCurved[QUAD]);

      /* Face 0: Bottom */
      for (int i = 0; i < nSide; i++)
        geo.faceNodesCurved[etype](0,i) = ij2quad[i];

      /* Face 1: Right */
      for (int i = 0; i < nSide; i++)
        geo.faceNodesCurved[etype](1,i) = ij2quad[i*nSide-1];

      /* Face 2: Top */
      for (int i = 0; i < nSide; i++)
        geo.faceNodesCurved[etype](2,i) = ij2quad[nSide*nSide-i-1];

      /* Face 3: Left */
      for (int i = 0; i < nSide; i++)
        geo.faceNodesCurved[etype](3,i) = ij2quad[nSide*(nSide-i-1)];
    }
    else if (etype == TRI)
    {
      geo.faceNodesCurved[etype].assign({3, geo.nNdFaceCurved[LINE]});

      switch (geo.nNodesPerEleBT[etype])
      {
        case 3:
          /* Face 0: Bottom */
          geo.faceNodesCurved[etype](0,0) = 0;
          geo.faceNodesCurved[etype](0,1) = 1;

          /* Face 1: Hypotenuse */
          geo.faceNodesCurved[etype](1,0) = 1;
          geo.faceNodesCurved[etype](1,1) = 2;

          /* Face 2: Left */
          geo.faceNodesCurved[etype](2,0) = 2;
          geo.faceNodesCurved[etype](2,1) = 0;
          break;

        case 6:
          /* Face 0: Bottom */
          geo.faceNodesCurved[etype](0,0) = 0;
          geo.faceNodesCurved[etype](0,1) = 3;
          geo.faceNodesCurved[etype](0,2) = 1;

          /* Face 1: Hypotenuse */
          geo.faceNodesCurved[etype](2,0) = 2;
          geo.faceNodesCurved[etype](2,1) = 4;
          geo.faceNodesCurved[etype](2,2) = 5;

          /* Face 2: Left */
          geo.faceNodesCurved[etype](2,0) = 2;
          geo.faceNodesCurved[etype](2,1) = 5;
          geo.faceNodesCurved[etype](2,2) = 0;
          break;

        default:
          ThrowException("Triangle face nodes not implemented for this order.");
      }
    }
    else if (etype == PRI)
    {
      uint maxN = std::max(geo.nNdFaceCurved[TRI], geo.nNdFaceCurved[QUAD]);
      geo.faceNodesCurved[etype].assign({6, maxN});

      std::vector<std::vector<int>> nodes(6);
      switch (geo.nNdFaceCurved[TRI])
      {
        case 3:
          nodes[0] = {0,1,2};
          nodes[1] = {3,5,4};
          break;

        case 6:
          nodes[0] = {0,1,2,6,8,7};
          nodes[1] = {3,5,4,13,14,12};
          break;

        case 10:
          nodes[0] = {0,1,2,6,7,12,13,9,8,24};
          nodes[1] = {3,5,4,20,21,23,22,19,8,25};
          break;
      }

      switch (geo.nNdFaceCurved[QUAD])
      {
        case 4:
          nodes[2] = {0,3,4,1};
          nodes[3] = {0,2,5,3};
          nodes[4] = {1,4,5,2};
          break;

        case 9:
          nodes[2] = {0,3,4,1,9,12,10,6,15};
          nodes[3] = {0,2,5,3,7,11,13,9,16};
          nodes[4] = {1,4,5,2,10,14,11,7,17};
          break;

        case 16:
          nodes[2] = {0,3,4,1,10,11,18,19,15,14,7,6,26,29,28,27};
          nodes[3] = {0,2,5,3,8,9,16,17,21,20,11,10,30,33,32,31};
          nodes[4] = {1,4,5,2,14,15,22,23,17,16,13,12,27,28,29,26};
          break;
      }

      for (int f = 0; f < 2; f++)
        for (int i = 0; i < geo.nNdFaceCurved[TRI]; i++)
          geo.faceNodesCurved[PRI](f, i) = nodes[f][i];

      for (int f = 2; f < 5; f++)
        for (int i = 0; i < geo.nNdFaceCurved[QUAD]; i++)
          geo.faceNodesCurved[PRI](f, i) = nodes[f][i];
    }
  }
}

void couple_periodic_bnds(GeoStruct &geo)
{
  mdvector<double> coords_face1, coords_face2;

  /* Loop over boundary faces */
  for (auto &bnd_face : geo.bnd_faces)
  {
    unsigned int bnd_id = bnd_face.second;
    auto face1 = bnd_face.first;
    unsigned int nNodesPerFace = (unsigned int) face1.size();

    coords_face1.assign({nNodesPerFace,geo.nDims});

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
        auto bnd_id2 = bnd_face2.second;

        if (face2 == face1 || face2.size() != nNodesPerFace || bnd_id2 != PERIODIC)
          continue;

        coords_face2.assign({nNodesPerFace,geo.nDims});

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
  if (geo.nDims != 2 && geo.nDims != 3)
    ThrowException("Improper value for nDims - should be 2 or 3.");

  std::map<ELE_TYPE, unsigned int> nFptsPerFace;

  for (auto ftype : geo.face_set)
  {
    switch (ftype)
    {
      case LINE:
      {
#ifdef _RT_TETS
        nFptsPerFace[ftype] = order + 2; // triangles require P+2 flux points on faces
        auto wts = Gauss_Legendre_pts(order + 2);
        geo.weights_fpts[ftype].assign({wts.size()});
        for (int i = 0; i < wts.size(); i++) geo.weights_fpts[ftype](i) = wts[i];
#else
        nFptsPerFace[ftype] = order + 1;
        auto wts = Gauss_Legendre_pts(order + 1);
        geo.weights_fpts[ftype].assign({(uint)wts.size()});
        for (int i = 0; i < wts.size(); i++) geo.weights_fpts[ftype](i) = wts[i];
#endif
        break;
      }

      case TRI:
      {
#ifdef _RT_TETS
        nFptsPerFace[ftype] = (order + 2) * (order + 3) / 2;
        geo.weights_fpts[ftype] = WS_Tri_weights(order+1);
#else
        nFptsPerFace[ftype] = (order + 1) * (order + 2) / 2;
        geo.weights_fpts[ftype] = WS_Tri_weights(order);
#endif
        break;
      }

      case QUAD:
      {
        nFptsPerFace[ftype] = (order + 1) * (order + 1);

        auto weights_1D = Gauss_Legendre_weights(order+1);

        geo.weights_fpts[ftype].assign({(order + 1) * (order + 1)});
        for (int i = 0; i < order+1; i++)
          for (int j = 0; j < order+1; j++)
            geo.weights_fpts[ftype](i*(order+1)+j) = weights_1D[i] * weights_1D[j];

        break;
      }

      default:
        ThrowException("Unknown face type encountered [Error in PyFR mesh reading?]");
    }
  }

  geo.nFptsPerFace = nFptsPerFace;

  std::map<std::vector<unsigned int>, std::vector<unsigned int>> face_fpts;
  std::map<std::vector<unsigned int>, std::vector<unsigned int>> bndface2fpts;
  std::map<ELE_TYPE, std::vector<std::vector<int>>> ele2fptsBT;
  std::map<ELE_TYPE, std::vector<std::vector<int>>> ele2fpts_slotBT;

  /* Determine number of interior global flux points */
  std::map<ELE_TYPE, std::set<std::vector<unsigned int>>> unique_faces;
  geo.nGfpts_int = 0; geo.nGfpts_bnd = 0;

#ifdef _MPI
  geo.nGfpts_mpi = 0;
  int rank;
  MPI_Comm_rank(geo.myComm, &rank);
#endif

  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;

    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      int F = 0;
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        auto ftype = geo.eleFaceTypesBT[etype][F];
        face.assign(face_nodes.size(), 0);

        for (unsigned int i = 0; i < face_nodes.size(); i++)
        {
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);
        }

        std::sort(face.begin(), face.end());

        if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC)
        {
          if (unique_faces[ftype].count(geo.per_bnd_pairs[face]))
            face = geo.per_bnd_pairs[face];
        }

        /* Check if face is has not been previously encountered */
        if (!unique_faces[ftype].count(face))
        {
            if (geo.bnd_faces.count(face) and geo.bnd_faces[face] != PERIODIC)
            {
              geo.nGfpts_bnd += nFptsPerFace[ftype];
            }
#ifdef _MPI
            else if (geo.mpi_faces.count(face))
            {
              geo.nGfpts_mpi += nFptsPerFace[ftype];
            }
#endif
            else
            {
              geo.nGfpts_int += nFptsPerFace[ftype];
            }
        }

        unique_faces[ftype].insert(face);
        F++;
      }
    }
  }

  /* Initialize global flux point indicies (to place boundary fpts at end of global fpt data structure) */
  unsigned int gfpt = 0; unsigned int gfpt_bnd = geo.nGfpts_int;

#ifdef _MPI
  unsigned int gfpt_mpi = geo.nGfpts_int + geo.nGfpts_bnd;
  std::set<std::vector<unsigned int>> mpi_faces_to_process;
#endif

  geo.nFaces = 0;
  geo.nIntFaces = 0;
  geo.nBndFaces = 0;
  geo.faceList.resize(0);

  /* Additional Connectivity Arrays */
#ifdef _MPI
  geo.mpiFaces.resize(0);
  geo.procR.resize(0);
  geo.mpiLocF.resize(0);
#endif

  geo.nGfpts = geo.nGfpts_int + geo.nGfpts_bnd;
#ifdef _MPI
  geo.nGfpts += geo.nGfpts_mpi;
#endif
  geo.fpt2face.assign(geo.nGfpts, -1);

  for (auto ftype : geo.face_set)
  {
    geo.face2fpts[ftype].assign({nFptsPerFace[ftype], (unsigned)unique_faces[ftype].size()}, -1);
    geo.faceID[ftype].assign({(uint)unique_faces[ftype].size()}, -1);
    geo.nFacesBT[ftype] = 0;
  }

  uint nf = 0;
  for (auto flist : unique_faces)
    nf += flist.second.size();

  geo.faceID_type.assign({nf}, -1);
  geo.faceType.assign({nf});
  geo.face2eles.assign({(unsigned)nf, 2}, -1);

  for (auto etype : geo.ele_set)
  {
    geo.nFptsPerEleBT[etype] = 0;
    for (auto ftype : geo.eleFaceTypesBT[etype])
      geo.nFptsPerEleBT[etype] += geo.nFptsPerFace[ftype];
  }  

  for (auto etype : geo.ele_set)
  {
    geo.ele2face[etype].assign({geo.nElesBT[etype], geo.nFacesPerEleBT[etype]}, -1);
  }

//#ifdef _BUILD_LIB
  for (auto ftype : geo.face_set)
  {
    geo.face2nodes[ftype].assign({(unsigned)unique_faces[ftype].size(), geo.nNdFaceCurved[ftype]}, -1);
  }
//#endif

  std::set<int> overPts, wallPts;
  std::set<unsigned int> overFaces, wallFaces;

  /* Begin loop through faces */
  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;
    ele2fptsBT[etype].resize(geo.nElesBT[etype]);
    ele2fpts_slotBT[etype].resize(geo.nElesBT[etype]);

    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      ele2fptsBT[etype][ele].assign(geo.nFptsPerEleBT[etype], -1);
      ele2fpts_slotBT[etype][ele].assign(geo.nFptsPerEleBT[etype], -1);

      unsigned int f0 = 0;
      unsigned int n = 0;
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        auto ftype = geo.eleFaceTypesBT[etype][n];
        face.assign(face_nodes.size(), 0);

        for (unsigned int i = 0; i < face_nodes.size(); i++)
        {
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);
        }

        std::sort(face.begin(), face.end());

        /* Check if face has been encountered previously */
        std::vector<unsigned int> fpts(nFptsPerFace[ftype],0);
        
        if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC)
        {
          if (face_fpts.count(geo.per_bnd_pairs[face]))
            face = geo.per_bnd_pairs[face];
        }

        if(!face_fpts.count(face))
        {
          geo.ele2face[etype](ele, n) = geo.nFaces;
          geo.face2eles(geo.nFaces, 0) = geo.eleID[etype](ele);

          for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
            geo.face2nodes[ftype](geo.nFacesBT[ftype], j) = geo.ele2nodesBT[etype](ele, geo.faceNodesCurved[etype](n,j));

          /* Check if face is on boundary */
          if (geo.bnd_faces.count(face) and geo.bnd_faces[face] != PERIODIC)
          {
            unsigned int bcType = geo.bnd_faces[face];
            for (auto &fpt : fpts)
            {
              geo.gfpt2bnd.push_back(bcType);
              fpt = gfpt_bnd;
              gfpt_bnd++;
            }

            bndface2fpts[face] = fpts;
            
            /* Create lists of all wall- and overset-boundary nodes */
            if (bcType == OVERSET)
            {
              for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
                overPts.insert(geo.face2nodes[ftype](geo.nFacesBT[ftype], j));
              overFaces.insert(geo.nFaces);
            }
            else if (bcType == SLIP_WALL ||
                     bcType == ISOTHERMAL_NOSLIP ||
                     bcType == ISOTHERMAL_NOSLIP_MOVING ||
                     bcType == ADIABATIC_NOSLIP ||
                     bcType == ADIABATIC_NOSLIP_MOVING ||
                     bcType == SYMMETRY)
            {
              for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
                wallPts.insert(geo.face2nodes[ftype](geo.nFacesBT[ftype], j));
              wallFaces.insert(geo.nFaces);
            }

            int bnd = geo.face2bnd[face];
            geo.boundFaces[bnd].push_back(geo.nFaces);
            geo.boundFaceTypes[bnd].push_back(ftype);
            geo.nBndFaces++;
          }
#ifdef _MPI
          /* Check if face is on MPI boundary */
          else if (geo.mpi_faces.count(face))
          {
            /* Add face to set to process later. */
            mpi_faces_to_process.insert(face);

            // Additional MPI face connectivity data
            geo.mpiFaces.push_back(geo.nFaces);
            auto procs = geo.mpi_faces[face];
            int p;
            for (auto &proc:procs)
              if (proc != rank) p = proc;
            geo.procR.push_back(p);
            geo.mpiLocF.push_back(n);

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

          for (unsigned int i = 0; i < nFptsPerFace[ftype]; i++)
          {
            ele2fptsBT[etype][ele][f0 + i] = fpts[i];
            ele2fpts_slotBT[etype][ele][f0 + i] = 0;
          }

          for (auto &fpt : fpts)
          {
            geo.fpt2face[fpt] = geo.nFaces;
          }

          for (int j = 0; j < nFptsPerFace[ftype]; j++)
            geo.face2fpts[ftype](j, geo.nFacesBT[ftype]) = fpts[j];

          geo.faceList.push_back(face);
          geo.nodes_to_face[face] = geo.nFaces;
          geo.faceType(geo.nFaces) = ftype;
          geo.faceID[ftype](geo.nFacesBT[ftype]) = geo.nFaces;
          geo.faceID_type(geo.nFaces) = geo.nFacesBT[ftype];
          geo.nFaces++;
          geo.nFacesBT[ftype]++;
        }
        /* If face has already been encountered, must assign existing global flux points */
        else
        {
          int ff = geo.nodes_to_face[face];
          geo.ele2face[etype](ele, n) = ff;
          geo.face2eles(ff, 1) = geo.eleID[etype](ele);

          auto fpts = face_fpts[face];
          
          /* Associate existing flux points with this face in reverse order (works as is for 2D, oriented later for 3D cases) */
          for (unsigned int i = 0; i < nFptsPerFace[ftype]; i++)
          {
            ele2fptsBT[etype][ele][f0 + i] = fpts[nFptsPerFace[ftype] - i - 1];
            ele2fpts_slotBT[etype][ele][f0 + i] = 1;
          }

          geo.nIntFaces++;
        }

        f0 += nFptsPerFace[ftype]; // Equivalent to n*nFptsPerFace if only 1 face type
        n++;
      }
    }
  }

  geo.nWall = wallPts.size();
  geo.nOver = overPts.size();
  geo.wallNodes.resize(0);
  geo.overNodes.resize(0);
  geo.overFaceList.resize(0);
  geo.overFaceList.reserve(overFaces.size());
  geo.wallFaceList.resize(0);
  geo.wallFaceList.reserve(wallFaces.size());
  geo.wallNodes.reserve(geo.nWall);
  geo.overNodes.reserve(geo.nOver);
  for (auto &pt:wallPts) geo.wallNodes.push_back(pt);
  for (auto &pt:overPts) geo.overNodes.push_back(pt);
  for (auto &fid:overFaces)
    geo.overFaceList.push_back(fid);
  for (auto &fid:wallFaces)
    geo.wallFaceList.push_back(fid);

  /* Process MPI faces, if needed */
#ifdef _MPI
  geo.nMpiFaces = geo.mpiFaces.size();
  geo.faceID_R.resize(geo.nMpiFaces);
  geo.mpiRotR.resize(geo.nMpiFaces);

  /* Loop over MPI faces (skipping periodic MPI faces in this loop) */
  for (const auto &face : mpi_faces_to_process)
  {
    if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC) continue;
    auto ranks = geo.mpi_faces[face];
    int sendRank = *std::min_element(ranks.begin(), ranks.end());
    int recvRank = *std::max_element(ranks.begin(), ranks.end());

    int faceID = geo.nodes_to_face[face];
    int ff = findFirst(geo.mpiFaces, faceID);

    /* Additional note: Deadlock is avoided due to consistent global ordering of mpi_faces map */

    if (rank == sendRank)
    {
      auto fpts = face_fpts[face];

      /* Append flux points to fpt_buffer_map in existing order */
      for (auto fpt : fpts)
        geo.fpt_buffer_map[recvRank].push_back(fpt);

      /* Append face to face_buffer_map */
      geo.face_buffer_map[recvRank].push_back(faceID);

      /* Send ordered face to paired rank */
      MPI_Status temp;
      MPI_Send(&faceID, 1, MPI_INT, recvRank, 0, geo.myComm);
      MPI_Recv(&geo.faceID_R[ff], 1, MPI_INT, recvRank, 0, geo.myComm, &temp);
    }
    else if (rank == recvRank)
    {
      auto fpts = face_fpts[face];

      MPI_Status temp;
      MPI_Recv(&geo.faceID_R[ff], 1, MPI_INT, sendRank, 0, geo.myComm, &temp);
      MPI_Send(&faceID, 1, MPI_INT, sendRank, 0, geo.myComm);

      /* Append existing flux points with this face in reverse order (works as is for 2D, oriented later for 3D cases) */
      for (unsigned int i = 0; i < fpts.size(); i++)
        geo.fpt_buffer_map[sendRank].push_back(fpts[fpts.size() - i - 1]);

      /* Append face to face_buffer_map */
      geo.face_buffer_map[sendRank].push_back(faceID);
    }
    else
    {
      ThrowException("Error in mpi_faces. Neither rank is this rank.");
    }
  }

  if (geo.per_bnd_flag)
  {
    /* Process periodic faces from ordered vector for consistency between ranks */
    for (const auto &face : geo.per_mpi_faces)
    {
      auto ranks = geo.mpi_faces[face];
      int sendRank = *std::min_element(ranks.begin(), ranks.end());
      int recvRank = *std::max_element(ranks.begin(), ranks.end());

      int faceID = geo.nodes_to_face[face];
      int ff = findFirst(geo.mpiFaces, faceID);

      /* Additional note: Deadlock is avoided due to consistent global ordering of mpi_faces map */

      if (rank == sendRank)
      {
        auto fpts = face_fpts[face];

        /* Append flux points to fpt_buffer_map in existing order */
        for (auto fpt : fpts)
          geo.fpt_buffer_map[recvRank].push_back(fpt);

        /* Append face to face_buffer_map */
        geo.face_buffer_map[recvRank].push_back(faceID);

        /* Send ordered face to paired rank */
        MPI_Status temp;
        MPI_Send(&faceID, 1, MPI_INT, recvRank, 0, geo.myComm);
        MPI_Recv(&geo.faceID_R[ff], 1, MPI_INT, recvRank, 0, geo.myComm, &temp);
      }
      else if (rank == recvRank)
      {
        auto fpts = face_fpts[face];

        MPI_Status temp;
        MPI_Recv(&geo.faceID_R[ff], 1, MPI_INT, sendRank, 0, geo.myComm, &temp);
        MPI_Send(&faceID, 1, MPI_INT, sendRank, 0, geo.myComm);

        /* Append existing flux points with this face in reverse order (works as is for 2D, oriented later for 3D cases) */
        for (unsigned int i = 0; i < fpts.size(); i++)
          geo.fpt_buffer_map[sendRank].push_back(fpts[fpts.size() - i - 1]);

        /* Append face to face_buffer_map */
        geo.face_buffer_map[sendRank].push_back(faceID);
      }
      else
      {
        ThrowException("Error in mpi_faces. Neither rank is this rank.");
      }
    }
  }

  MPI_Barrier(geo.myComm);

#endif

  /* Populate data structures */
#ifdef _MPI
  geo.nGfpts = gfpt_mpi;
#else
  geo.nGfpts = gfpt_bnd;
#endif

  for (auto etype : geo.ele_set)
  {
    geo.fpt2gfptBT[etype].assign({geo.nFptsPerEleBT[etype], geo.nElesBT[etype]});
    geo.fpt2gfpt_slotBT[etype].assign({geo.nFptsPerEleBT[etype], geo.nElesBT[etype]});

    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (unsigned int fpt = 0; fpt < geo.nFptsPerEleBT[etype]; fpt++)
      {
        geo.fpt2gfptBT[etype](fpt,ele) = ele2fptsBT[etype][ele][fpt];
        geo.fpt2gfpt_slotBT[etype](fpt,ele) = ele2fpts_slotBT[etype][ele][fpt];
      }
    }
  }
}

void setup_element_colors(InputStruct *input, GeoStruct &geo)
{
  /* Sizing ele2eleN for possible element type with greatest number of faces */
  mdvector<int> ele2eleN;
  if (geo.nDims == 2)
    ele2eleN.assign({geo.nFacesPerEleBT[QUAD], geo.nEles}, -1);
  else
    ele2eleN.assign({geo.nFacesPerEleBT[HEX], geo.nEles}, -1);

  /* Construct face2eles */
  std::map<std::vector<unsigned int>, std::vector<unsigned int>> face2eles;
  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        face.assign(face_nodes.size(), 0);
        for (unsigned int i = 0; i < face_nodes.size(); i++)
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);

        /* Sort for consistency */
        std::sort(face.begin(), face.end());

        face2eles[face].push_back(geo.eleID[etype](ele));

        /* Special case for periodic boundary */
        if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC)
          face2eles[geo.per_bnd_pairs[face]].push_back(geo.eleID[etype](ele));
      }
    }
  }

  /* Generate element adjacency (element to elements connectivity) */
  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      unsigned int n = 0;
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        face.assign(face_nodes.size(), 0);
        for (unsigned int i = 0; i < face_nodes.size(); i++)
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);

        /* Sort for consistency */
        std::sort(face.begin(), face.end());

        if (face2eles[face].size() == 1)
          ele2eleN(n, geo.eleID[etype](ele)) = -1;
        else
          ele2eleN(n, geo.eleID[etype](ele)) = (ele != face2eles[face][0]) ? face2eles[face][0] : face2eles[face][1];

        n++;
      }
    }
  }

  /* Setup element colors */
  for (auto etype : geo.ele_set)
    geo.ele2colorBT[etype].assign({geo.nElesBT[etype]}, -1);

  if (input->nColors == 1)
  {
    geo.nColors = 1;
    for (auto etype : geo.ele_set)
      geo.ele2colorBT[etype].fill(0);
  }
  else
  {
    geo.nColors = input->nColors;
    std::vector<bool> used(geo.nColors, false);
    std::vector<unsigned int> counts(geo.nColors, 0);
    std::queue<unsigned int> eleQ;

    eleQ.push(0);

    /* Loop over elements and assign colors using greedy algorithm */
    while (!eleQ.empty())
    {
      unsigned int ele = eleQ.front();
      auto etype = geo.eleType(ele);
      eleQ.pop();

      if (geo.ele2colorBT[etype](ele - geo.eleID[etype](0)) != -1) continue;

      for (unsigned int face = 0; face < geo.nFacesPerEleBT[etype]; face++)
      {
        int eleN = ele2eleN(face, ele);
        if (eleN == -1) continue;

        ELE_TYPE etypeN = geo.eleType(eleN);
        int colorN = geo.ele2colorBT[etypeN](eleN - geo.eleID[etypeN](0));

        /* Add element to queue if neighbor has no color */
        if (colorN == -1)
          eleQ.push(eleN);

        /* Record if neighbor is using a given color */
        else
          used[colorN] = true;
      }

      int color = -1;
      unsigned int min_count = 0;
      unsigned int min_color_all = 0;
      unsigned int min_count_all = counts[0];

      /* Set current element color to color unused by neighbors with minimum count in domain */
      for (unsigned int c = 0; c < geo.nColors; c++)
      {
        if (!used[c] and color == -1)
        {
          color = c;
          min_count = counts[c];
        }
        else if (!used[c])
        {
          if (counts[c] < min_count)
          {
            color = c;
            min_count = counts[c];
          }
        }

        if (counts[c] < min_count_all)
        {
          min_count_all = counts[c];
          min_color_all = c;
        }
      }

      if (color == -1)
        ThrowException("Could not color graph with number of colors provided. Increase nColors!");

      geo.ele2colorBT[etype](ele - geo.eleID[etype](0)) = color;
      counts[color]++;
      used.assign(geo.nColors, false);
    }
  }
}

void shuffle_data_by_color(GeoStruct &geo)
{
  /* Reorganize elements by color */
  std::map<ELE_TYPE, std::vector<std::vector<unsigned int>>> color2elesBT;
  for (auto etype : geo.ele_set)
  {
    color2elesBT[etype].resize(geo.nColors);
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
      color2elesBT[etype][geo.ele2colorBT[etype](ele)].push_back(ele);
  }

  /* Reorganize geometry data */
  // TODO: Consider an in-place permutation to save memory
  unsigned int ele_new = 0;
  for (auto etype : geo.ele_set)
  {
    auto eleID_temp = geo.eleID[etype];
    auto ele2colorBT_temp = geo.ele2colorBT[etype];
    auto ele2nodesBT_temp = geo.ele2nodesBT[etype];
    for (unsigned int color = 0; color < geo.nColors; color++)
      for (unsigned int i = 0; i < color2elesBT[etype][color].size(); i++)
      {
        unsigned int eleBT_new = ele_new - eleID_temp(0);
        unsigned int eleBT = color2elesBT[etype][color][i];
        unsigned int ele = eleID_temp(eleBT);

        geo.eleID[etype](eleBT_new) = ele_new;
        geo.ele2colorBT[etype](eleBT_new) = ele2colorBT_temp(eleBT);

        for (unsigned int node = 0; node < geo.nNodesPerEleBT[etype]; node++)
          geo.ele2nodesBT[etype](eleBT_new, node) = ele2nodesBT_temp(eleBT, node);

        ele_new++;
      }
  }

  /* Setup element color ranges */
  for (auto etype : geo.ele_set)
  {
    geo.nElesPerColorBT[etype].assign(geo.nColors, 0);
    geo.rangePerColorBT[etype].assign(geo.nColors + 1, 0);

    geo.rangePerColorBT[etype][1] = color2elesBT[etype][0].size();
    geo.nElesPerColorBT[etype][0] = color2elesBT[etype][0].size();
    for (unsigned int color = 1; color < geo.nColors; color++)
    {
      geo.rangePerColorBT[etype][color+1] = geo.rangePerColorBT[etype][color] + color2elesBT[etype][color].size();
      geo.nElesPerColorBT[etype][color] = geo.rangePerColorBT[etype][color+1] - geo.rangePerColorBT[etype][color];
    }
  }

  /* Print out color distribution */
#ifdef _MPI
  std::cout << "Rank " << geo.rank << " ";
#endif
  for (auto etype : geo.ele_set)
  {
    if (geo.ele_set.size() > 1) std::cout << "Ele Type " << etype << " ";
    std::cout << "Color Distribution:";

    for (unsigned int color = 0; color < geo.nColors; color++)
      std::cout << " " << geo.nElesPerColorBT[etype][color];
    std::cout << std::endl;
  }
}

void set_ele_adjacency(GeoStruct &geo)
{
  /* Sizing ele2eleN and face2faceN */
  if (geo.nDims == 2)
  {
    geo.ele2eleN.assign({geo.nFacesPerEleBT[QUAD], geo.nEles}, -1);
    geo.face2faceN.assign({geo.nFacesPerEleBT[QUAD], geo.nEles}, -1);
  }
  else
  {
    geo.ele2eleN.assign({geo.nFacesPerEleBT[HEX], geo.nEles}, -1);
    geo.face2faceN.assign({geo.nFacesPerEleBT[HEX], geo.nEles}, -1);
  }

  /* Construct ele2eleN and face2faceN connectivity */
  for (unsigned int faceID = 0; faceID < geo.nFaces; faceID++)
  {
    int eleID = geo.face2eles(faceID, 0);
    int eleNID = geo.face2eles(faceID, 1);
    if (eleNID == -1) continue;

    auto etype = geo.eleType(eleID);

    unsigned int face = 0;
    while (geo.ele2face[etype](eleID, face) != faceID) face++;
    geo.ele2eleN(face, eleID) = eleNID;

    unsigned int faceN = 0;
    while (geo.ele2face[etype](eleNID, faceN) != faceID) faceN++;
    geo.ele2eleN(faceN, eleNID) = eleID;

    geo.face2faceN(face, eleID) = faceN;
    geo.face2faceN(faceN, eleNID) = face;
  }

#ifdef _MPI
  /* Send ele2eleN and face2faceN connectivity on MPI faces */
  std::map<unsigned int, mdvector<unsigned int>> face_sbuffs, face_rbuffs;
  std::vector<MPI_Request> rreqs(geo.face_buffer_map.size());
  std::vector<MPI_Request> sreqs(geo.face_buffer_map.size());
  unsigned int idx = 0;
  for (const auto &entry : geo.face_buffer_map)
  {
    unsigned int rankN = entry.first;
    const auto &faceIDs = entry.second;

    /* Stage nonblocking receives */
    face_rbuffs[rankN].assign({2, (unsigned int) faceIDs.size()});
    MPI_Irecv(face_rbuffs[rankN].data(), 2 * (unsigned int) faceIDs.size(), MPI_UNSIGNED,
        rankN, 0, geo.myComm, &rreqs[idx]);

    /* Pack buffer of ele and face adjacency */
    face_sbuffs[rankN].assign({2, (unsigned int) faceIDs.size()});
    for (unsigned int i = 0; i < faceIDs.size(); i++)
    {
      unsigned int faceID = faceIDs(i);
      unsigned int eleID = geo.face2eles(faceID, 0);
      auto etype = geo.eleType(eleID);
      unsigned int face = 0; while (geo.ele2face[etype](eleID, face) != (int)faceID) face++;
      face_sbuffs[rankN](0, i) = eleID;
      face_sbuffs[rankN](1, i) = face;
    }

    /* Send buffer to paired rank */
    MPI_Isend(face_sbuffs[rankN].data(), 2 * (unsigned int) faceIDs.size(), MPI_UNSIGNED,
        rankN, 0, geo.myComm, &sreqs[idx]);
    idx++;
  }
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

  /* Unpack buffer into ele2eleN and face2faceN */
  for (const auto &entry : geo.face_buffer_map)
  {
    unsigned int rankN = entry.first;
    const auto &faceIDs = entry.second;
    for (unsigned int i = 0; i < faceIDs.size(); i++)
    {
      unsigned int faceID = faceIDs(i);
      unsigned int eleID = geo.face2eles(faceID, 0);
      auto etype = geo.eleType(eleID);
      unsigned int face = 0; while (geo.ele2face[etype](eleID, face) != (int)faceID) face++;
      geo.ele2eleN(face, eleID) = face_rbuffs[rankN](0, i);
      geo.face2faceN(face, eleID) = face_rbuffs[rankN](1, i);
    }
  }
#endif
}

#ifdef _MPI
void partition_geometry(InputStruct *input, GeoStruct &geo)
{
  int rank, nRanks;
  MPI_Comm_rank(geo.myComm, &rank);
  MPI_Comm_size(geo.myComm, &nRanks);

  /* Setup METIS */
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);

  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
  options[METIS_OPTION_DBGLVL] = 0;
  options[METIS_OPTION_CONTIG] = 1;
  options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;
  options[METIS_OPTION_NCUTS] = 5;

  /* Form eptr and eind arrays */
  std::vector<int> eptr(geo.nEles + 1);

  unsigned int nInds = 0;
  for (auto etype : geo.ele_set)
    nInds += geo.nElesBT[etype] * geo.nCornerNodesBT[etype];

  std::vector<int> eind(nInds);
  std::vector<int> vwgt(geo.nEles, 1);
  std::set<unsigned int> nodes;
  std::vector<unsigned int> face;

  int n = 0;
  eptr[0] = 0;

  unsigned int ele = 0;
  for (auto etype : geo.ele_set)
  {
    for (unsigned int i = 0; i < geo.nElesBT[etype]; i++)
    {
      for (unsigned int j = 0; j < geo.nCornerNodesBT[etype];  j++)
      {
        eind[j + n] = geo.ele2nodesBT[etype](i, j);
      } 

      n += geo.nCornerNodesBT[etype];
      eptr[ele + 1] = n;
      ele++;
    }
  }

  int objval;
  std::vector<int> epart(geo.nEles, 0);
  std::vector<int> npart(geo.nNodes);
  /// TODO: Should just not call this entire function if nRanks == 1
  if (nRanks > 1) 
  {
    int ncommon = geo.nDims;
    int ne = geo.nEles;
    int nn = geo.nNodes;

    METIS_PartMeshDual(&ne, &nn, eptr.data(), eind.data(), vwgt.data(), 
        nullptr, &ncommon, &nRanks, nullptr, options, &objval, epart.data(), 
        npart.data());  
  }

  /* Obtain list of elements on this partition */
  std::vector<unsigned int> myEles;
  std::map<ELE_TYPE, std::vector<unsigned int>> myElesBT;
  geo.gID2ele.assign({geo.nEles}, -1);

  for (auto etype : geo.ele_set)
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++) 
    {
      unsigned int gid = geo.eleID[etype](ele);
      if (epart[gid] == rank)
      {
        geo.gID2ele(gid) = myElesBT[etype].size();
        geo.eleIDg.push_back(geo.eleID[etype](ele));
        myElesBT[etype].push_back(ele);
      }
    }

  /* Collect map of *ALL* MPI interfaces from METIS partition data */
  std::map<std::vector<unsigned int>, std::set<int>> face2ranks;    
  std::map<std::vector<unsigned int>, std::set<int>> mpi_faces_glob;

  /* Iterate over faces of complete mesh */
  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;

    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      int face_rank = epart[geo.eleID[etype](ele)];
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        face.assign(face_nodes.size(), 0);

        for (unsigned int i = 0; i < face_nodes.size(); i++)
        {
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);
        }

        /* Sort for consistency */
        std::sort(face.begin(), face.end());

        if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC)
        {
          if (face2ranks.count(geo.per_bnd_pairs[face]))
            face = geo.per_bnd_pairs[face];
        }
        
        face2ranks[face].insert(face_rank);

        /* If two ranks assigned to same face, add to map of "MPI" faces */
        if (face2ranks[face].size() == 2)
        {
          mpi_faces_glob[face] = face2ranks[face];
          if (geo.bnd_faces.count(face) and geo.bnd_faces[face] == PERIODIC)
            mpi_faces_glob[geo.per_bnd_pairs[face]] = face2ranks[face];
        }
      }
    }
  }

  /* Set number of elements to partition local */
  geo.nEles = 0;
  for (auto etype : geo.ele_set)
  {
    geo.nElesBT[etype] = myElesBT[etype].size();
    geo.nEles += (unsigned int) myElesBT[etype].size();

    // remove etype if no elements of type in this partition
    if (geo.nElesBT[etype] == 0) geo.ele_set.erase(etype);
  }

  /* Reduce connectivity to contain only partition local elements. Also reindex eleIDs and eleTypes. */
  unsigned int eleID = 0;
  geo.eleType.assign({geo.nEles});
  for (auto etype : geo.ele_set)
  {
    auto ele2nodes_glob = geo.ele2nodesBT[etype];
    geo.ele2nodesBT[etype].assign({(unsigned int) myElesBT[etype].size(), geo.nNodesPerEleBT[etype]}, 0);
    geo.eleID[etype].assign({(unsigned int) myElesBT[etype].size()}, 0);

    for (unsigned int ele = 0; ele < myElesBT[etype].size(); ele++)
    {
      for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
      {
        geo.ele2nodesBT[etype](ele, nd) = ele2nodes_glob(myElesBT[etype][ele], nd);
      }
      geo.eleID[etype](ele) = eleID;
      geo.eleType(eleID) = etype;
      eleID++;
    }
  }

  if (input->iterative_method == MCGS)
  {
    /* Reduce color data to only contain partition local elements */
    auto ele2colorBT_glob = geo.ele2colorBT;
    for (auto etype : geo.ele_set)
    {
      geo.ele2colorBT[etype].assign({(unsigned int) myElesBT[etype].size()}, -1);
      for (unsigned int ele = 0; ele < myElesBT[etype].size(); ele++)
        geo.ele2colorBT[etype](ele) = ele2colorBT_glob[etype](myElesBT[etype][ele]);
    }
  }

  /* Obtain set of unique nodes on this partition. Set number of nodes. */
  std::set<unsigned int> uniqueNodes;
  for (auto etype : geo.ele_set)
  {
    for (unsigned int ele = 0; ele < myElesBT[etype].size(); ele++)
    {
      for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
      {
        uniqueNodes.insert(geo.ele2nodesBT[etype](ele, nd));
      }
    }
  }

  geo.nNodes = (unsigned int) uniqueNodes.size();

  /* Obtain set of unique faces on this partition. */
  std::set<std::vector<unsigned int>> uniqueFaces;
  for (auto etype : geo.ele_set)
  {
    std::vector<unsigned int> face;

    for (unsigned int ele = 0; ele < myElesBT[etype].size(); ele++)
    {
      for (auto face_nodes : geo.face_nodesBT[etype])
      {
        face.assign(face_nodes.size(), 0);

        for (unsigned int i = 0; i < face_nodes.size(); i++)
        {
          face[i] = geo.ele2nodesBT[etype](ele, face_nodes[i]);
        }

        /* Sort for consistency */
        std::sort(face.begin(), face.end());

        uniqueFaces.insert(face);
      }
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
  for (auto etype : geo.ele_set)
    for (unsigned int ele = 0; ele < myElesBT[etype].size(); ele++)
      for (unsigned int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
        geo.ele2nodesBT[etype](ele, nd) = geo.node_map_g2p[geo.ele2nodesBT[etype](ele, nd)];

  /* Reduce boundary faces data to contain only faces on local partition. Also
   * reindex via geo.node_map_g2p */
  auto bnd_faces_glob = geo.bnd_faces;
  auto face2bnd_glob = geo.face2bnd;
  geo.bnd_faces.clear();
  geo.face2bnd.clear();
  geo.nBnds = 0;

  /* Iterate over all boundary faces */
  for (auto entry : bnd_faces_glob)
  {
    std::vector<unsigned int> bnd_face = entry.first;
    int bcType = entry.second;


    if (uniqueFaces.count(bnd_face))
    {
      /* Renumber nodes and store */
      for (auto &nd : bnd_face)
      {
        nd = geo.node_map_g2p[nd];
      }
      geo.bnd_faces[bnd_face] = bcType;
      geo.nBnds++;
    }
  }
  
  for (auto entry : face2bnd_glob)
  {
    std::vector<unsigned int> bnd_face = entry.first;
    int bnd_id = entry.second;

    if (uniqueFaces.count(bnd_face))
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

    if (uniqueFaces.count(mpi_face))
    {
      /* Renumber nodes and store */
      for (auto &nd : mpi_face)
      {
        nd = geo.node_map_g2p[nd];
      }
      geo.mpi_faces[mpi_face] = face_ranks;
    }
  }

  if (geo.per_bnd_flag)
  {
    /* Process periodic boundaries exist*/
    std::set<std::vector<unsigned int>> processed;
    auto per_bnd_pairs_glob = geo.per_bnd_pairs;
    geo.per_bnd_pairs.clear();

    /* Iterate over all boundary faces */
    for (auto entry : per_bnd_pairs_glob)
    {
      std::vector<unsigned int> face1 = entry.first;
      std::vector<unsigned int> face2 = entry.second;

      /* If all nodes are on this partition, keep face data */
      bool myFace1 = uniqueFaces.count(face1);
      
      bool myFace2 = uniqueFaces.count(face2);

      /* Renumber nodes */
      for (auto &nd : face1)
        nd = geo.node_map_g2p[nd];
      for (auto &nd : face2)
        nd = geo.node_map_g2p[nd];

      /* Periodic boundary fully within partition, no MPI needed */
      if (myFace1 and myFace2)
      {
        geo.per_bnd_pairs[face1] = face2;
      }
      /* Otherwise, possible MPI boundary. Store ordered vector of faces to iterate over later */
      else if (myFace1 and !processed.count(face1))
      {
        geo.per_mpi_faces.push_back(face1);
        processed.insert(face1); // to avoid duplicates
      }
      else if (myFace2 and !processed.count(face2))
      {
        geo.per_mpi_faces.push_back(face2);
        processed.insert(face2);
      }
    }
  }

  std::cout << "Rank " << input->rank << ": nEles = " << geo.nEles;
  std::cout << ", nMpiFaces = " << geo.mpi_faces.size() << std::endl;

  if (input->rank == 0)
    std::cout << "Total # MPI Faces: " << mpi_faces_glob.size() << std::endl;
}
#endif

void load_mesh_data_pyfr(InputStruct *input, GeoStruct &geo)
{
  H5File file(input->meshfile, H5F_ACC_RDONLY);

  ssize_t nObjs = file.getNumObjs();

  // ---- Create compound data type for reading face connectivity ----

  hid_t char4_t = H5Tcopy(H5T_C_S1);
  H5Tset_size(char4_t, 5); // 4 chars + null-termination character

  // NOTE: HDF5 will segfault if you try to use this CompType more than once
  CompType fcon_type(sizeof(face_con));
  fcon_type.insertMember("f0", HOFFSET(face_con, c_type), char4_t);
  fcon_type.insertMember("f1", HOFFSET(face_con, ic), PredType::NATIVE_INT);
  fcon_type.insertMember("f2", HOFFSET(face_con, loc_f), PredType::NATIVE_SHORT);

  // Load the names of all datasets into a vector so we can more easily
  // process each one

  std::vector<std::string> dsNames;
  for (int i  = 0; i < nObjs; i++)
  {
    if (file.getObjTypeByIdx(i) == H5G_DATASET)
    {
      dsNames.push_back(file.getObjnameByIdx(i));
    }
  }

  // Read the mesh ID string
  DataSet dset = file.openDataSet("mesh_uuid");
  DataType dtype = dset.getDataType();
  DataSpace dspace(H5S_SCALAR);
  dset.read(geo.mesh_uuid, dtype, dspace);
  dset.close();

  int max_rank = 0;
  for (std::string name : dsNames)
  {
    size_t ind = name.find("con_p");
    if (ind == std::string::npos)
      continue;

    auto str = name.substr(5,name.length());
    ind = str.find("p");

    if (str.find("p") != std::string::npos)
      continue;

    std::stringstream ss(name.substr(5,name.length()));
    int _rank; ss >> _rank;
    max_rank = std::max(_rank, max_rank);
  }

  if (max_rank != input->nRanks-1)
  {
    std::cout << "max rank in mesh = " << max_rank << ", nRanks = " << input->nRanks << std::endl;
    ThrowException("Wrong number of ranks - MPI size should match # of mesh partitions.");
  }

  const std::string quadcon = "spt_quad_p" + std::to_string(input->rank);
  const std::string tricon = "spt_tri_p" + std::to_string(input->rank);
  const std::string tetcon = "spt_tet_p" + std::to_string(input->rank);
  const std::string pricon = "spt_pri_p" + std::to_string(input->rank);
  const std::string pyrcon = "spt_pyr_p" + std::to_string(input->rank);
  const std::string hexcon = "spt_hex_p" + std::to_string(input->rank);

  std::map<std::string, ELE_TYPE> ele_str_map;
  ele_str_map["tri"]  = TRI;
  ele_str_map["quad"] = QUAD;
  ele_str_map["tet"]  = TET;
  ele_str_map["pri"]  = PRI;
  ele_str_map["pyr"]  = PYR;
  ele_str_map["hex"]  = HEX;

  // Figure out # of dimensions
  geo.nEles = 0;
  for (auto &name : dsNames)
  {    
    ELE_TYPE etype;

    if (name == tricon)
      etype = TRI;
    else if (name == quadcon)
      etype = QUAD;
    else if (name == tetcon)
      etype = TET;
    else if (name == pricon)
      etype = PRI;
    else if(name == pyrcon)
      etype = PYR;
    else if(name == hexcon)
      etype = HEX;
    else
      continue;

    geo.ele_set.insert(etype); // Add to discovered element dtypes

    auto DS = file.openDataSet(name);
    auto ds = DS.getSpace();

    hsize_t dims[3];
    int ds_rank = ds.getSimpleExtentDims(dims);

    if (ds_rank != 3 or DS.getTypeClass() != H5T_FLOAT)
      ThrowException("Cannot read element nodes from PyFR mesh file - wrong data type.");

    geo.nDims = dims[2];

    geo.nNodesPerEleBT[etype] = dims[0];
    geo.nNodes = geo.nEles * geo.nNodesPerEleBT[etype];
    geo.nElesBT[etype] = dims[1];
    geo.eleID[etype].assign({(uint)dims[1]});

    geo.nEles += dims[1];

    if (etype == HEX)
    {
      int nSide = cbrt(geo.nNodesPerEleBT[etype]);
      geo.nNdFaceCurved[QUAD] = nSide*nSide;
    }
    else if (etype == QUAD)
    {
      geo.nNdFaceCurved[LINE] = sqrt(geo.nNodesPerEleBT[etype]);
    }
    else if (etype == TRI)
    {
      switch (geo.nNodesPerEleBT[etype])
      {
        case 3:
          geo.nNdFaceCurved[LINE] = 2; break;

        case 6:
          geo.nNdFaceCurved[LINE] = 3; break;

        case 10:
          geo.nNdFaceCurved[LINE] = 4; break;

        default:
          ThrowException("Shape orders > 3 not implemented for triangle grids");
      }
    }
    else if (etype == TET)
    {
      switch (geo.nNodesPerEleBT[etype])
      {
        case 4:
          geo.nNdFaceCurved[TRI] = 3; break;

        case 10:
          geo.nNdFaceCurved[TRI] = 6; break;

        case 20:
          geo.nNdFaceCurved[TRI] = 10; break;

        default:
          ThrowException("Shape orders > 3 not implemented for tet grids");
      }
    }
    else if (etype == PRI)
    {
      switch (geo.nNodesPerEleBT[etype])
      {
        case 6:
          geo.nNdFaceCurved[TRI] = 3;
          geo.nNdFaceCurved[QUAD] = 4;
          break;

        case 18:
          geo.nNdFaceCurved[TRI] = 6;
          geo.nNdFaceCurved[QUAD] = 9;
          break;

        case 40:
          geo.nNdFaceCurved[TRI] = 10;
          geo.nNdFaceCurved[QUAD] = 16;
          break;

        default:
          ThrowException("Shape orders > 3 not implemented for prism grids");
      }
    }

    mdvector<double> tmp_nodes({(unsigned)dims[0],(unsigned)dims[1],(unsigned)dims[2]});

    DS.read(tmp_nodes.data(), PredType::NATIVE_DOUBLE);

    /// TODO: change Gmsh reading / geo setup so this isn't needed
    geo.ele_nodes[etype].assign({geo.nNodesPerEleBT[etype], geo.nDims, geo.nElesBT[etype]});
    geo.ele2nodesBT[etype].assign({geo.nElesBT[etype], geo.nNodesPerEleBT[etype]});

    std::vector<int> ndmap;
    switch (etype)
    {
      case TRI:
        ndmap = gmsh_to_structured_tri(geo.nNodesPerEleBT[etype]); break;
      case QUAD:
        ndmap = gmsh_to_structured_quad(geo.nNodesPerEleBT[etype]); break;
      case HEX:
        ndmap = gmsh_to_structured_hex(geo.nNodesPerEleBT[etype]); break;
      case TET:
        ndmap = gmsh_to_structured_tet(geo.nNodesPerEleBT[etype]); break;
      case PRI:
        ndmap = gmsh_to_structured_pri(geo.nNodesPerEleBT[etype]); break;
    }

    // Re-order nodes to ZEFR format
    for (int ele = 0; ele < geo.nElesBT[etype]; ele++)
      for (int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
        for (int d = 0; d < geo.nDims; d++)
          geo.ele_nodes[etype](nd, d, ele) = tmp_nodes(ndmap[nd], ele, d);
  }

  int eleID = 0;
  geo.eleType.assign({geo.nEles});
  for (auto etype : geo.ele_set)
    for (int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      geo.eleID[etype](ele) = eleID;
      geo.eleType(eleID) = etype;
      eleID++;
    }

  // Collect total array of all nodes in mesh
  geo.nNodes = 0;
  for (auto etype : geo.ele_set)
    geo.nNodes += geo.nElesBT[etype] * geo.nNodesPerEleBT[etype];

  mdvector<double> temp_coords({geo.nNodes, geo.nDims});

  int gnd = 0;
  for (auto etype : geo.ele_set)
  {
    for (int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
      {
        for (int d = 0; d < geo.nDims; d++)
          temp_coords(gnd, d) = geo.ele_nodes[etype](nd, d, ele);

        geo.ele2nodesBT[etype](ele, nd) = gnd;
        gnd++;
      }
    }
  }

  /* --- Create 'proper' c2v connectivity (Remove duplicates) --- */

  auto sortind = fuzzysort_row(temp_coords);

  // Setup map to new node listing
  double tol = 1e-10;
  unsigned int idx = sortind[0];
  unsigned int n_nodes = 0;
  point pti = point(&temp_coords(idx,0), geo.nDims);
  std::vector<int> nodemap(geo.nNodes, -1);
  nodemap[idx] = n_nodes;

  for (int i = 1; i < geo.nNodes; i++)
  {
    idx = sortind[i];
    point ptj = point(&temp_coords(idx,0), geo.nDims);
    if (point(ptj-pti).norm() > tol)
    {
      // Increment our counters
      pti = ptj;
      n_nodes++;
    }

    nodemap[idx] = n_nodes;
  }
  n_nodes += 1; // final index -> total #

  // Setup geo structures with new node list
  geo.coord_nodes.assign({n_nodes, geo.nDims});
  for (int i = 0; i < geo.nNodes; i++)
  {
    for (int d = 0; d < geo.nDims; d++)
      geo.coord_nodes(nodemap[i], d) = temp_coords(i, d);
  }

  for (auto etype : geo.ele_set)
  {
    for (int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (int nd = 0; nd < geo.nNodesPerEleBT[etype]; nd++)
      {
        geo.ele2nodesBT[etype](ele, nd) = nodemap[geo.ele2nodesBT[etype](ele, nd)];
      }
    }
  }

  geo.nNodes = n_nodes;

  if (geo.ele_set.count(TRI) || geo.ele_set.count(QUAD))
  {
    geo.face_set.insert(LINE);
  }

  if (geo.ele_set.count(HEX) || geo.ele_set.count(PRI) || geo.ele_set.count(PYR))
  {
    geo.face_set.insert(QUAD);
  }

  if (geo.ele_set.count(TET) || geo.ele_set.count(PRI) || geo.ele_set.count(PYR))
  {
    geo.face_set.insert(TRI);
  }

  geo.face_types.assign({(uint)geo.face_set.size()});
  geo.nNode_face.assign({(uint)geo.face_set.size()});

  int n = 0;
  for (auto ftype : geo.face_set)
  {
    geo.face_types(n) = ftype;
    geo.nNode_face(n) = geo.nNdFaceCurved[ftype];
    n++;
  }

  // ----- Read interior element / face connectivity -----

  geo.pyfr2zefr_face[TRI] = {0,1,2};
  geo.zefr2pyfr_face[TRI] = {0,1,2};

  geo.pyfr2zefr_face[QUAD] = {0,1,2,3};
  geo.zefr2pyfr_face[QUAD] = {0,1,2,3};

  geo.pyfr2zefr_face[HEX] = {0,4,3,5,2,1};
  geo.zefr2pyfr_face[HEX] = {0,5,4,2,1,3};

  geo.pyfr2zefr_face[TET] = {0,1,2,3};
  geo.zefr2pyfr_face[TET] = {0,1,2,3};

  geo.pyfr2zefr_face[PRI] = {0,1,2,4,3};
  geo.zefr2pyfr_face[PRI] = {0,1,2,4,3};

  for (auto ftype : geo.face_set)
    geo.nFacesBT[ftype] = 0;

  std::string pcon = "con_p" + std::to_string(input->rank);
  for (auto &name : dsNames)
  {
    if (name != pcon)
      continue;

    auto DS = file.openDataSet(name);
    auto ds = DS.getSpace();

    hsize_t dims[2];
    int ds_rank = ds.getSimpleExtentDims(dims);

    if (ds_rank != 2 or DS.getTypeClass() != H5T_COMPOUND)
      ThrowException("Cannot read internal connectivity from PyFR mesh file - expecting compound data type of rank 2.");

    geo.nIntFaces = dims[1];

    CompType fcon_t = fcon_type; // NOTE: HDF5 segfaults if you try to re-use a CompType in more than one read

    geo.face_list.assign({2, geo.nIntFaces});
    DS.read(geo.face_list.data(), fcon_t);

    for (int f = 0; f < geo.nIntFaces; f++)
    {
      auto &f1 = geo.face_list(0,f);
      auto &f2 = geo.face_list(1,f);
      ELE_TYPE etype1 = ele_str_map[std::string(f1.c_type)];
      ELE_TYPE etype2 = ele_str_map[std::string(f2.c_type)];
      f1.loc_f = geo.pyfr2zefr_face[etype1][f1.loc_f];
      f2.loc_f = geo.pyfr2zefr_face[etype2][f2.loc_f];
      ELE_TYPE ftype = geo.eleFaceTypesBT[etype1][f1.loc_f];
      geo.nFacesBT[ftype]++;
    }

    break;
  }

  // ----- Read boundary conditions -----

  geo.nBounds = 0;
  geo.nBndFaces = 0;
  geo.bound_faces.resize(0);
  geo.boundFaces.resize(0);
  geo.bcGlobal.resize(0);
  std::string bcon_p = "_p" + std::to_string(input->rank);
  size_t len = bcon_p.length();

  // Gather global list of BC names
  std::set<std::string> nameList;
  for (auto &name : dsNames)
  {
    if (name.substr(0,5) == "bcon_")
    {
      std::string bcname = name.substr(5,name.size());
      uint bclen = bcname.find_last_of("_p") - 1;
      bcname = bcname.substr(0, bclen);
      nameList.insert(bcname);
    }
  }

  geo.bcGlobal.reserve(nameList.size());
  for (auto &name : nameList)
    geo.bcGlobal.push_back(name);

  // Get boundary faces for this partition
  for (auto &name : dsNames)
  {
    if (name.substr(0,5) == "bcon_" and
        name.substr(name.length() - len, len) == bcon_p)
    {
      std::string bcName = name.substr(5, name.length()-len-5);

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
      geo.bcIdMap[geo.nBounds] = geo.nBounds; // Map PyFR bcid to ZEFR bound index
      if (geo.bnd_ids.back() == PERIODIC) geo.per_bnd_flag = true;

      auto DS = file.openDataSet(name);
      auto ds = DS.getSpace();

      if (ds.getSimpleExtentNdims() != 1 or DS.getTypeClass() != H5T_COMPOUND)
        ThrowException("Cannot read boundary condition from PyFR mesh file - expecting compound data type of rank 1.");

      hsize_t dims[1];
      ds.getSimpleExtentDims(dims);

      CompType fcon_t = fcon_type; // NOTE: HDF5 segfaults if you try to re-use a CompType in more than one read

      geo.bound_faces.emplace_back();
      geo.bound_faces[geo.nBounds].resize(dims[0]);
      DS.read(geo.bound_faces[geo.nBounds].data(), fcon_t);

      for (auto &face : geo.bound_faces[geo.nBounds])
      {
        ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
        face.loc_f = geo.pyfr2zefr_face[etype][face.loc_f];
        ELE_TYPE ftype = geo.eleFaceTypesBT[etype][face.loc_f];
        geo.nFacesBT[ftype]++;
      }

      geo.boundFaces.push_back(get_int_list(dims[0], geo.nIntFaces + geo.nBndFaces));

      geo.nBndFaces += dims[0];
      geo.nBounds++;
    }
  }

#ifdef _MPI
  geo.nMpiFaces = 0;
  // Read MPI boundaries for this rank
  pcon += "p";
  geo.send_ranks.resize(0);
  for (auto &name : dsNames)
  {
    if (name.substr(0,pcon.length()) != pcon)
      continue;

    // Get the opposite rank
    int rank2;
    std::stringstream ss(name.substr(pcon.length(),name.length()-pcon.length()));
    ss >> rank2;
    geo.send_ranks.push_back(rank2);

    // Read the connectivity for this MPI boundary
    auto DS = file.openDataSet(name);
    auto ds = DS.getSpace();

    int ds_rank = ds.getSimpleExtentNdims();
    std::vector<hsize_t> dims(ds_rank);
    ds.getSimpleExtentDims(dims.data());

    if (ds_rank != 1 or DS.getTypeClass() != H5T_COMPOUND)
      ThrowException("Cannot read MPI connectivity from PyFR mesh file - expecting compound data type of rank 1.");

    CompType fcon_t = fcon_type;

    geo.mpi_conn[rank2].resize(dims[0]);
    DS.read(geo.mpi_conn[rank2].data(), fcon_t);

    for (auto &face : geo.mpi_conn[rank2])
    {
      ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
      face.loc_f = geo.pyfr2zefr_face[etype][face.loc_f];
      ELE_TYPE ftype = geo.eleFaceTypesBT[etype][face.loc_f];
      geo.nFacesBT[ftype]++;
    }

    geo.nMpiFaces += dims[0];
  }

  std::sort(geo.send_ranks.begin(), geo.send_ranks.end());
#endif

  geo.nFaces = geo.nIntFaces + geo.nBndFaces;
#ifdef _MPI
  geo.nFaces += geo.nMpiFaces;
#endif

  // Assign face types to all faces: grouped by internal, boudnary, MPI

  geo.faceType.resize({geo.nFaces});

  // Finish setting up ele-to-face / face-to-ele connectivity

  set_face_nodes(geo);

  for (auto etype : geo.ele_set)
  {
    geo.ele2face[etype].assign({geo.nElesBT[etype], geo.nFacesPerEleBT[etype]});
    geo.nElesBT[etype] = 0;
  }

  geo.eleID_type.assign({geo.nEles}, -1);
  for (int ele = 0; ele < geo.nEles; ele++)
    geo.eleID_type(ele) = geo.nElesBT[geo.eleType(ele)]++;

  geo.face2eles.assign({geo.nFaces, 2}, -1);
  geo.faceID_type.assign({geo.nFaces}, -1);
  for (auto ftype : geo.face_set)
  {
    geo.face2nodes[ftype].assign({(uint)geo.nFacesBT[ftype], (uint)geo.nNdFaceCurved[ftype]}, -1);
    geo.faceID[ftype].assign({(uint)geo.nFacesBT[ftype]}, -1);
    geo.nFacesBT[ftype] = 0;
  }

  for (int f = 0; f < geo.nIntFaces; f++)
  {
    auto &f1 = geo.face_list(0,f);
    auto &f2 = geo.face_list(1,f);
    ELE_TYPE etype1 = ele_str_map[std::string(f1.c_type)];
    ELE_TYPE etype2 = ele_str_map[std::string(f2.c_type)];
    ELE_TYPE ftype = geo.eleFaceTypesBT[etype1][f1.loc_f];
    geo.faceType(f) = ftype;
    geo.ele2face[etype1](f1.ic, f1.loc_f) = f;
    geo.ele2face[etype2](f2.ic, f2.loc_f) = f;
    geo.face2eles(f, 0) = f1.ic;
    geo.face2eles(f, 1) = f2.ic;

    // Connectivity only used in conjunction with TIOGA
    for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
      geo.face2nodes[ftype](geo.nFacesBT[ftype],j) = geo.ele2nodesBT[etype1](f1.ic, geo.faceNodesCurved[etype1](f1.loc_f, j));


    geo.faceID[ftype](geo.nFacesBT[ftype]) = f;
    geo.faceID_type(f) = geo.nFacesBT[ftype];
    geo.nFacesBT[ftype]++;
  }

  int fid = geo.nIntFaces;
  for (int bnd = 0; bnd < geo.nBounds; bnd++)
  {
    for (int ff = 0; ff < geo.bound_faces[bnd].size(); ff++)
    {
      auto f = geo.bound_faces[bnd][ff];
      ELE_TYPE etype = ele_str_map[std::string(f.c_type)];
      ELE_TYPE ftype = geo.eleFaceTypesBT[etype][f.loc_f];
      geo.faceType(fid) = ftype;
      geo.ele2face[etype](f.ic, f.loc_f) = fid;
      geo.face2eles(fid, 0) = f.ic;

      // Connectivity only used in conjunction with TIOGA
      for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
        geo.face2nodes[ftype](geo.nFacesBT[ftype],j) = geo.ele2nodesBT[etype](f.ic, geo.faceNodesCurved[etype](f.loc_f,j));

      geo.faceID[ftype](geo.nFacesBT[ftype]) = fid;
      geo.faceID_type(fid) = geo.nFacesBT[ftype];
      geo.nFacesBT[ftype]++;

      fid++;
    }
  }

#ifdef _MPI
  fid = geo.nIntFaces + geo.nBndFaces;
  geo.mpiFaces.resize(0);
  for (auto &p : geo.send_ranks)
  {
    for (int ff = 0; ff < geo.mpi_conn[p].size(); ff++)
    {
      auto f = geo.mpi_conn[p][ff];
      ELE_TYPE etype = ele_str_map[std::string(f.c_type)];
      ELE_TYPE ftype = geo.eleFaceTypesBT[etype][f.loc_f];
      geo.faceType(fid) = ftype;
      geo.ele2face[etype](f.ic, f.loc_f) = fid;
      geo.face2eles(fid, 0) = f.ic;

      // Connectivity only used in conjunction with TIOGA
      for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
        geo.face2nodes[ftype](geo.nFacesBT[ftype],j) = geo.ele2nodesBT[etype](f.ic, geo.faceNodesCurved[etype](f.loc_f,j));

      geo.mpiFaces.push_back(fid);
      geo.faceID[ftype](geo.nFacesBT[ftype]) = fid;
      geo.faceID_type(fid) = geo.nFacesBT[ftype];
      geo.nFacesBT[ftype]++;

      fid++;
    }
  }

  std::cout << "Rank " << input->rank << ": nEles = " << geo.nEles << ", nMpiFaces = " << geo.nMpiFaces << std::endl;
#endif
}

void setup_global_fpts_pyfr(InputStruct *input, GeoStruct &geo, unsigned int order)
{
  if (input->rank == 0)
    std::cout << "Setting up global flux point connectivity..." << std::endl;

  /* Form set of unique faces */
  if (geo.nDims != 2 && geo.nDims != 3)
    ThrowException("Improper value for nDims - should be 2 or 3.");

  std::map<ELE_TYPE, unsigned int> nFptsPerFace;

  for (auto ftype : geo.face_set)
  {
    switch (ftype)
    {
      case LINE:
      {
#ifdef _RT_TETS
        nFptsPerFace[ftype] = order + 2; // triangles require P+2 flux points on faces
        auto wts = Gauss_Legendre_pts(order + 2);
        geo.weights_fpts[ftype].assign({wts.size()});
        for (int i = 0; i < wts.size(); i++) geo.weights_fpts[ftype](i) = wts[i];
#else
        nFptsPerFace[ftype] = order + 1;
        auto wts = Gauss_Legendre_pts(order + 1);
        geo.weights_fpts[ftype].assign({(uint)wts.size()});
        for (int i = 0; i < wts.size(); i++) geo.weights_fpts[ftype](i) = wts[i];
#endif
        break;
      }

      case TRI:
      {
#ifdef _RT_TETS
        nFptsPerFace[ftype] = (order + 2) * (order + 3) / 2;
        geo.weights_fpts[ftype] = WS_Tri_weights(order+1);
#else
        nFptsPerFace[ftype] = (order + 1) * (order + 2) / 2;
        geo.weights_fpts[ftype] = WS_Tri_weights(order);
#endif
        break;
      }

      case QUAD:
      {
        nFptsPerFace[ftype] = (order + 1) * (order + 1);

        auto weights_1D = Gauss_Legendre_weights(order+1);

        geo.weights_fpts[ftype].assign({(order + 1) * (order + 1)});
        for (int i = 0; i < order+1; i++)
          for (int j = 0; j < order+1; j++)
            geo.weights_fpts[ftype](i*(order+1)+j) = weights_1D[i] * weights_1D[j];

        break;
      }

      default:
        ThrowException("Unknown face type encountered [Error in PyFR mesh reading?]");
    }
  }

  geo.nFptsPerFace = nFptsPerFace;

  /* Determine number of interior global flux points */

  geo.nGfpts_int = 0;
  for (int ff = 0; ff < geo.nIntFaces; ff++)
  {
    geo.nGfpts_int += nFptsPerFace[geo.faceType(ff)];
  }

  /* Determine total number of boundary faces and flux points */

  geo.nGfpts_bnd = 0;
  for (int ff = 0; ff < geo.nBndFaces; ff++)
  {
    geo.nGfpts_bnd += nFptsPerFace[geo.faceType(geo.nIntFaces+ff)];
  }

  geo.nGfpts = geo.nGfpts_int + geo.nGfpts_bnd;

#ifdef _MPI
  /* Determine total number of MPI flux points */
  geo.nGfpts_mpi = 0;
  for (int ff = 0; ff < geo.nMpiFaces; ff++)
  {
    geo.nGfpts_mpi += nFptsPerFace[geo.faceType(geo.nIntFaces+geo.nBndFaces+ff)];
  }

  int rank;
  MPI_Comm_rank(geo.myComm, &rank);

  geo.nGfpts += geo.nGfpts_mpi;
#endif

  /* Determine number of faces per face type */
  for (auto ftype : geo.face_set)
  {
    geo.face2fpts[ftype].assign({(uint)nFptsPerFace[ftype], (uint)geo.nFacesBT[ftype]}, -1);
  }

  for (auto etype : geo.ele_set)
  {
    geo.nFptsPerEleBT[etype] = 0;
    for (auto ftype : geo.eleFaceTypesBT[etype])
      geo.nFptsPerEleBT[etype] += geo.nFptsPerFace[ftype];

    geo.fpt2gfptBT[etype].assign({geo.nFptsPerEleBT[etype], geo.nElesBT[etype]});
    geo.fpt2gfpt_slotBT[etype].assign({geo.nFptsPerEleBT[etype], geo.nElesBT[etype]});
  }

  geo.fpt2face.assign(geo.nGfpts, -1);

  std::map<std::string, ELE_TYPE> ele_str_map;
  ele_str_map["tri"]  = TRI;
  ele_str_map["quad"] = QUAD;
  ele_str_map["tet"]  = TET;
  ele_str_map["pri"]  = PRI;
  ele_str_map["hex"]  = HEX;

  // Counter for total global flux points so far
  unsigned int gfpt = 0;

  // Begin by looping over all internal / periodic faces
  for (int ff = 0; ff < geo.nIntFaces; ff++)
  {
    auto faceL = geo.face_list(0,ff);
    auto faceR = geo.face_list(1,ff);
    int eL = faceL.ic;
    int eR = faceR.ic;
    ELE_TYPE etypeL = ele_str_map[std::string(faceL.c_type)];
    ELE_TYPE etypeR = ele_str_map[std::string(faceR.c_type)];
    ELE_TYPE ftype = geo.eleFaceTypesBT[etypeL][faceL.loc_f];

    int fpt0L = 0;
    for (int i = 0; i < faceL.loc_f; i++)
      fpt0L += geo.nFptsPerFace[geo.eleFaceTypesBT[etypeL][i]];

    int fpt0R = 0;
    for (int i = 0; i < faceR.loc_f; i++)
      fpt0R += geo.nFptsPerFace[geo.eleFaceTypesBT[etypeR][i]];

    // Setup global flux point IDs within left/right eles
    int nfpts = nFptsPerFace[ftype];
    for (int fpt = 0; fpt < nfpts; fpt++)
    {
      geo.fpt2gfptBT[etypeL](fpt0L + fpt, eL) = gfpt;
      geo.fpt2gfptBT[etypeR](fpt0R + nfpts - fpt - 1, eR) = gfpt;
      geo.fpt2gfpt_slotBT[etypeL](fpt0L + fpt, eL) = 0;
      geo.fpt2gfpt_slotBT[etypeR](fpt0R + nfpts - fpt - 1, eR) = 1;

      geo.face2fpts[ftype](fpt, geo.faceID_type(ff)) = gfpt;
      geo.fpt2face[gfpt] = ff;
      gfpt++;
    }
  }

  /* -- Stuff required for overset grids -- */
  std::set<int> overPts, wallPts;
  geo.overFaceList.resize(0);
  geo.wallFaceList.resize(0);

  // Counter of total boundary flux points so far
  unsigned int gfpt_bnd = 0;

  // Setup boundary-face flux points
  geo.gfpt2bnd.assign({geo.nGfpts_bnd});
  uint fid = geo.nIntFaces;
  for (uint bnd = 0; bnd < geo.nBounds; bnd++)
  {
    for (auto &face : geo.bound_faces[bnd])
    {
      int ele = face.ic;
      int n = face.loc_f;
      ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
      ELE_TYPE ftype = geo.faceType(fid);

      int fpt0 = 0;
      for (int i = 0; i < n; i++)
        fpt0 += nFptsPerFace[geo.eleFaceTypesBT[etype][i]];

      // Setup the flux points for the face
      for (int fpt = 0; fpt < nFptsPerFace[ftype]; fpt++)
      {
        geo.fpt2gfptBT[etype](fpt0 + fpt, ele) = gfpt;
        geo.fpt2gfpt_slotBT[etype](fpt0 + fpt, ele) = 0;

        geo.gfpt2bnd(gfpt_bnd) = geo.bnd_ids[bnd];
        geo.face2fpts[ftype](fpt, geo.faceID_type(fid)) = gfpt;
        geo.fpt2face[gfpt] = fid;
        gfpt_bnd++;
        gfpt++;
      }

      // Create lists of all wall- and overset-boundary nodes
      int bcType = geo.bnd_ids[bnd];
      if (bcType == OVERSET)
      {
        for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
          overPts.insert(geo.ele2nodesBT[etype](ele, geo.faceNodesCurved[etype](n, j)));

        geo.overFaceList.push_back(fid);
      }
      else if (bcType == SLIP_WALL ||
               bcType == ISOTHERMAL_NOSLIP ||
               bcType == ISOTHERMAL_NOSLIP_MOVING ||
               bcType == ADIABATIC_NOSLIP ||
               bcType == ADIABATIC_NOSLIP_MOVING ||
               bcType == SYMMETRY)
      {
        for (int j = 0; j < geo.nNdFaceCurved[ftype]; j++)
          wallPts.insert(geo.ele2nodesBT[etype](ele, geo.faceNodesCurved[etype](n, j)));

        geo.wallFaceList.push_back(fid);
      }

      fid++;
    }
  }

  geo.nWall = wallPts.size();
  geo.nOver = overPts.size();
  geo.wallNodes.resize(0);
  geo.overNodes.resize(0);
  geo.wallNodes.reserve(geo.nWall);
  geo.overNodes.reserve(geo.nOver);
  for (auto &pt:wallPts) geo.wallNodes.push_back(pt);
  for (auto &pt:overPts) geo.overNodes.push_back(pt);
  std::sort(geo.overFaceList.begin(),geo.overFaceList.end());
  geo.overFaceList.erase( std::unique(geo.overFaceList.begin(),geo.overFaceList.end()), geo.overFaceList.end() );
  std::sort(geo.wallFaceList.begin(),geo.wallFaceList.end());
  geo.wallFaceList.erase( std::unique(geo.wallFaceList.begin(),geo.wallFaceList.end()), geo.wallFaceList.end() );

#ifdef _MPI
  unsigned int mpiFace = geo.nIntFaces + geo.nBndFaces;
  unsigned int mface = 0;
  geo.faceID_R.resize(geo.nMpiFaces);
  geo.mpiFaces.resize(geo.nMpiFaces);
  geo.procR.resize(geo.nMpiFaces);
  for (auto &p2 : geo.send_ranks)
  {
    unsigned int nFaces = geo.mpi_conn[p2].size();

    mdvector<double> face_pt({geo.nDims, 2*nFaces});

    if (rank < p2)
    {
      if (geo.nDims == 3)
      {
        // Send face node '0' to 'right' rank
        for (int ff = 0; ff < nFaces; ff++)
        {
          auto face = geo.mpi_conn[p2][ff];
          int ic = face.ic;
          int f = face.loc_f;

          ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
          ELE_TYPE ftype = geo.eleFaceTypesBT[etype][f];

          // 'First' node of face
          for (int d = 0; d < geo.nDims; d++)
          {
            face_pt(d,ff) = geo.ele_nodes[etype](geo.face_nodesBT[etype][f][0],d,ic);
            face_pt(d,ff+nFaces) = 0.;
          }

          // Centroid of face
          for (int n = 0; n < geo.nNodesPerFaceBT[ftype]; n++)
            for (int d = 0; d < geo.nDims; d++)
              face_pt(d,ff+nFaces) += geo.ele_nodes[etype](geo.face_nodesBT[etype][f][n],d,ic) / geo.nNodesPerFaceBT[ftype];
        }

      }

      // Setup MPI flux points
      for (int ff = 0; ff < nFaces; ff++)
      {
        auto face = geo.mpi_conn[p2][ff];
        int ele = face.ic;
        int n = face.loc_f;

        ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
        ELE_TYPE ftype = geo.eleFaceTypesBT[etype][n];

        int fpt0 = 0;
        for (int i = 0; i < n; i++)
          fpt0 += nFptsPerFace[geo.eleFaceTypesBT[etype][i]];

        for (int fpt = 0; fpt < nFptsPerFace[ftype]; fpt++)
        {
          geo.fpt2gfptBT[etype](fpt0 + fpt, ele) = gfpt;
          geo.fpt2gfpt_slotBT[etype](fpt0 + fpt, ele) = 0;

          geo.fpt_buffer_map[p2].push_back(gfpt);
          geo.face2fpts[ftype](fpt, geo.faceID_type(mpiFace)) = gfpt;
          geo.fpt2face[gfpt] = mpiFace;
          gfpt++;
        }

        geo.face_buffer_map[p2].push_back(mpiFace);

        MPI_Status status;
        MPI_Send(&mpiFace, 1, MPI_INT, p2, 0, geo.myComm);
        MPI_Recv(&geo.faceID_R[mface], 1, MPI_INT, p2, 0, geo.myComm, &status);
        geo.mpiFaces[mface] = mpiFace;
        geo.procR[mface] = p2;
        mpiFace++;
        mface++;
      }
    }
    else // 'right' side of MPI boundary
    {
      // Setup MPI flux points
      std::vector<int> fpts;
      for (int ff = 0; ff < nFaces; ff++)
      {
        auto face = geo.mpi_conn[p2][ff];
        int ele = face.ic;
        int n = face.loc_f;

        ELE_TYPE etype = ele_str_map[std::string(face.c_type)];
        ELE_TYPE ftype = geo.eleFaceTypesBT[etype][n];

        int fpt0 = 0;
        for (int i = 0; i < n; i++)
          fpt0 += nFptsPerFace[geo.eleFaceTypesBT[etype][i]];

        fpts.resize(nFptsPerFace[ftype]);

        for (int fpt = 0; fpt < nFptsPerFace[ftype]; fpt++)
        {
          geo.fpt2gfptBT[etype](fpt0 + fpt, ele) = gfpt;
          geo.fpt2gfpt_slotBT[etype](fpt0 + fpt, ele) = 0;
          geo.face2fpts[ftype](fpt, geo.faceID_type(mpiFace)) = gfpt;
          geo.fpt2face[gfpt] = mpiFace;
          fpts[fpt] = gfpt;
          gfpt++;
        }

        for (int fpt = 0; fpt < nFptsPerFace[ftype]; fpt++)
        {
          geo.fpt_buffer_map[p2].push_back(fpts[nFptsPerFace[ftype] - fpt - 1]);
        }

        geo.face_buffer_map[p2].push_back(mpiFace);

        MPI_Status status;
        MPI_Recv(&geo.faceID_R[mface], 1, MPI_INT, p2, 0, geo.myComm, &status);
        MPI_Send(&mpiFace, 1, MPI_INT, p2, 0, geo.myComm);
        geo.mpiFaces[mface] = mpiFace;
        geo.procR[mface] = p2;
        mpiFace++;
        mface++;
      }
    }
  }

  if (mface != geo.nMpiFaces)
    ThrowException("Unused MPI Faces exist within PyFR mesh!");

  MPI_Barrier(geo.myComm);
#endif
}

void setup_flip_beta(InputStruct *input, GeoStruct &geo)
{
  geo.flip_beta.assign({geo.nGfpts}, 1);

  /* Only apply flipping trick if beta = +-0.5 */
  if (std::abs(input->ldg_b) != 0.5) return;

#ifdef _MPI
  for (const auto &entry : geo.fpt_buffer_map)
  {
    int pairedRank = entry.first;
    const auto &fpts = entry.second;

    char val = 0;
    if ((input->rank + pairedRank) % 2)
    {
      val = (input->rank > pairedRank) ? 1 : -1;
    }
    else
    {
      val = (input->rank < pairedRank) ? 1 : -1;
    }

    for (unsigned int i = 0; i < fpts.size(); i++)
      geo.flip_beta(fpts(i)) = val;
  }
#endif

}

void move_grid(InputStruct *input, GeoStruct &geo, double time)
{
  uint nNodes = geo.nNodes;

  switch (input->motion_type)
  {
    case TEST1:
    {
      double t0 = 10;
      double Atx = 2;
      double Aty = 2;
      double DX = 5;
      double DY = 5;

      for (int node = 0; node < nNodes; node++)
      {
        /// Taken from Kui, AIAA-2010-5031-661
        double x0 = geo.coords_init(node,0); double y0 = geo.coords_init(node,1);
        geo.coord_nodes(node,0) = x0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(Atx*pi*time/t0);
        geo.coord_nodes(node,1) = y0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(Aty*pi*time/t0);
        geo.grid_vel_nodes(node,0) = Atx*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*cos(Atx*pi*time/t0);
        geo.grid_vel_nodes(node,1) = Aty*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*cos(Aty*pi*time/t0);
      }
      break;
    }
    case TEST2:
    {
      double t0 = 10.*sqrt(5.);
      double DX = 5;
      double DY = 5;
      double DZ = 5;
      double Atx = 4;
      double Aty = 8;
      double Atz = 4;
      if (geo.nDims == 2)
      {
        for (uint node = 0; node < nNodes; node++)
        {
          /// Taken from Liang-Miyaji
          double x0 = geo.coords_init(node,0); double y0 = geo.coords_init(node,1);
          geo.coord_nodes(node,0) = x0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(Atx*pi*time/t0);
          geo.coord_nodes(node,1) = y0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(Aty*pi*time/t0);
          geo.grid_vel_nodes(node,0) = Atx*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*cos(Atx*pi*time/t0);
          geo.grid_vel_nodes(node,1) = Aty*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*cos(Aty*pi*time/t0);
        }
      }
      else
      {
        for (uint node = 0; node < nNodes; node++)
        {
          /// Taken from Liang-Miyaji
          double x0 = geo.coords_init(node,0); double y0 = geo.coords_init(node,1); double z0 = geo.coords_init(node,2);
          geo.coord_nodes(node,0) = x0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*sin(Atx*pi*time/t0);
          geo.coord_nodes(node,1) = y0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*sin(Aty*pi*time/t0);
          geo.coord_nodes(node,2) = z0 + sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*sin(Atz*pi*time/t0);
          geo.grid_vel_nodes(node,0) = Atx*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*cos(Atx*pi*time/t0);
          geo.grid_vel_nodes(node,1) = Aty*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*cos(Aty*pi*time/t0);
          geo.grid_vel_nodes(node,2) = Atz*pi/t0*sin(pi*x0/DX)*sin(pi*y0/DY)*sin(pi*z0/DZ)*cos(Atz*pi*time/t0);
        }
      }
      break;
    }
    case TEST3:
    {
      if (geo.gridID==0)
      {
        /// Liangi-Miyaji with easily-modifiable domain width
        double t0 = 10.*sqrt(5.);
        double width = 5.;
        for (uint node = 0; node < nNodes; node++)
        {
          geo.coord_nodes(node,0) = geo.coords_init(node,0) + sin(pi*geo.coords_init(node,0)/width)*sin(pi*geo.coords_init(node,1)/width)*sin(4*pi*time/t0);
          geo.coord_nodes(node,1) = geo.coords_init(node,1) + sin(pi*geo.coords_init(node,0)/width)*sin(pi*geo.coords_init(node,1)/width)*sin(8*pi*time/t0);
          geo.grid_vel_nodes(node,0) = 4.*pi/t0*sin(pi*geo.coords_init(node,0)/width)*sin(pi*geo.coords_init(node,1)/width)*cos(4*pi*time/t0);
          geo.grid_vel_nodes(node,1) = 8.*pi/t0*sin(pi*geo.coords_init(node,0)/width)*sin(pi*geo.coords_init(node,1)/width)*cos(8*pi*time/t0);
        }
      }
      break;
    }
    case CIRCULAR_TRANS:
    {
      /// Rigid oscillation in a circle
      if (geo.gridID == 0)
      {
        double Ax = input->moveAx; // Amplitude  (m)
        double Ay = input->moveAy; // Amplitude  (m)
        double fx = input->moveFx; // Frequency  (Hz)
        double fy = input->moveFy; // Frequency  (Hz)

        double Az = input->moveAz;
        double fz = input->moveFz;

        for (uint node = 0; node < nNodes ; node++)
        {
          geo.coord_nodes(node,0) = geo.coords_init(node,0) + Ax*sin(2.*pi*fx*time);
          geo.coord_nodes(node,1) = geo.coords_init(node,1) + Ay*(1-cos(2.*pi*fy*time));
          geo.grid_vel_nodes(node,0) = 2.*pi*fx*Ax*cos(2.*pi*fx*time);
          geo.grid_vel_nodes(node,1) = 2.*pi*fy*Ay*sin(2.*pi*fy*time);
          if (geo.nDims == 3)
          {
            geo.coord_nodes(node,2) = geo.coords_init(node,2) - Az*sin(2.*pi*fz*time);
            geo.grid_vel_nodes(node,2) = -2.*pi*fz*Az*cos(2.*pi*fz*time);
          }
        }
      }
      break;
    }
    case RIGID_BODY:
    {
      /// 6 DOF Rotation / Translation: Handled separately in eles->move()
      break;
    }
  }
}

