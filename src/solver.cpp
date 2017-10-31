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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <queue>
#include <vector>

extern "C" {
#include "cblas.h"
}

#include "elements.hpp"
#include "faces.hpp"
#include "funcs.hpp"
#include "geometry.hpp"
#include "hexas.hpp"
#include "points.hpp"
#include "polynomials.hpp"
#include "quads.hpp"
#include "input.hpp"
#include "mdvector.hpp"
#include "tets.hpp"
#include "tris.hpp"
#include "solver.hpp"

#include <Eigen/Dense>

#ifdef _MPI
#include "mpi.h"
#endif

#ifdef _GPU
#include "mdvector_gpu.h"
#include "solver_kernels.h"
#include "elements_kernels.h"
#include "cublas_v2.h"
#endif

#include "H5Cpp.h"
#ifndef _H5_NO_NAMESPACE
using namespace H5;
#endif
#ifdef _MPI
  #ifdef H5_HAVE_PARALLEL
//    #define _USE_H5_PARALLEL
  #endif
#endif

/*! NOTE: size of HDF5 attributes limited to 64kb, including header data
 *  Empirically determined max of 65471 8-byte ints
 *  (I'm rounding down to a nice number, though, just for safety) */
#define MAX_H5_ATTR_SIZE 65000

//! Function to use in iterateAttrs to grab all attribute names on an object
void attrOp(H5Location &loc, const H5std_string attr_name, void *op_data)
{
  std::vector<std::string> *names = static_cast<std::vector<std::string>*>(op_data);
  names->push_back(attr_name);
}

FRSolver::FRSolver(InputStruct *input, int order)
{
  this->input = input;
  if (order == -1)
    this->order = input->order;
  else
    this->order = order;

}

void FRSolver::setup(_mpi_comm comm_in, _mpi_comm comm_world)
{
  myComm = comm_in;
#ifdef _MPI
  worldComm = comm_world;
#endif

  if (input->rank == 0) std::cout << "Reading mesh: " << input->meshfile << std::endl;
  geo = process_mesh(input, order, input->nDims, myComm);

  if (input->rank == 0) std::cout << "Setting up timestepping..." << std::endl;
  setup_update();  

  if (input->rank == 0) std::cout << "Setting up elements and faces..." << std::endl;

  /* Create eles objects */
  unsigned int elesObjID = 0;
  ele2elesObj.assign({geo.nEles});
  if (input->iterative_method == MCGS)
  {
    for (auto etype : geo.ele_set)
      for (unsigned int color = 0; color < geo.nColors; color++)
      {
        unsigned int startEle = geo.rangePerColorBT[etype][color];
        unsigned int endEle = geo.rangePerColorBT[etype][color+1];
        create_elesObj(etype, elesObjID, startEle, endEle);

        /* Create connectivity from global ele to elesObjID */
        for (unsigned int ele = startEle; ele < endEle; ele++)
        {
          unsigned int eleID = geo.eleID[etype](ele);
          ele2elesObj(eleID) = elesObjID;
        }
        elesObjID++;
      }

    /* Organize eles objects by color */
    elesObjsBC.resize(geo.nColors);
    for (unsigned int color = 0; color < geo.nColors; color++)
      for (unsigned int etypeID = 0; etypeID < geo.ele_set.size(); etypeID++)
        elesObjsBC[color].push_back(elesObjs[etypeID*geo.nColors + color]);
  }

  else
  {
    for (auto etype : geo.ele_set)
    {
      unsigned int startEle = 0;
      unsigned int endEle = geo.nElesBT[etype];
      create_elesObj(etype, elesObjID, startEle, endEle);

      /* Create connectivity from global ele to elesObjID */
      for (unsigned int ele = startEle; ele < endEle; ele++)
      {
        unsigned int eleID = geo.eleID[etype](ele);
        ele2elesObj(eleID) = elesObjID;
      }
      elesObjID++;
    }
  }

  eles = elesObjs[0];

  faces = std::make_shared<Faces>(&geo, input, myComm);

  faces->setup(geo.nDims, elesObjs[0]->nVars);

  /* Partial element setup for flux point orientation */
  for (auto e : elesObjs)
  {
    e->set_locs();
    e->set_shape();
    e->set_coords(faces);
  }

  orient_fpts();

  /* Complete element setup */
  for (auto e : elesObjs)
  {
    e->setup(faces, myComm);
  }

  /* Setup fpt adjacency for viscous implicit Jacobians */
  if (input->implicit_method && input->viscous)
  {
    if (input->rank == 0) std::cout << "Setting up fpt adjacency..." << std::endl;
    set_fpt_adjacency();
  }

  /* Write Low Rank Approximation rank */
#ifdef _CPU
  if (input->linear_solver == SVD)
  {
    std::cout << "Low Rank Approximation rank:";
    for (auto e : elesObjs)
      std::cout << " " << e->svd_rank;
    std::cout << std::endl;
  }
#endif

  if (input->rank == 0) std::cout << "Setting up output..." << std::endl;
  setup_output();

  if (input->rank == 0) std::cout << "Initializing solution..." << std::endl;
  //initialize_U();
  for (auto e : elesObjs)
    e->initialize_U();

  if (input->filt_on)
  {
    if (input->rank == 0) std::cout << "Setting up filter..." << std::endl;
    filt.setup(input, *this);
  }

#ifdef _GPU
  if (input->rank == 0) std::cout << "Setting up data on GPU(s)..." << std::endl;
  solver_data_to_device();
#endif

  setup_views(); // Note: This function allocates addtional GPU memory for views

  if (input->tavg)
  {
    tavg_prev_time = flow_time;
    accumulate_time_averages();
  }

  if (input->implicit_method && input->viscous)
  {
    /* Setup jacoN views for viscous implicit Jacobians */
    if (input->rank == 0) std::cout << "Setting up jacoN views..." << std::endl;
    setup_jacoN_views();

    /* Setup ddUdUc for viscous KPF implicit Jacobians */
    if (input->KPF_Jacobian)
    {
      if (input->rank == 0) std::cout << "Setting up KPF Jacobian operators..." << std::endl;
      for (auto e : elesObjs)
        e->setup_ddUdUc();

#ifdef _GPU
      for (auto e : elesObjs)
        e->ddUdUc_d = e->ddUdUc;
#endif
    }
  }

#ifdef _GPU
  report_gpu_mem_usage();
#endif
}

void FRSolver::restart_solution(void)
{
  if (input->restart_type == 0)
  {
    if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_file + " ..." << std::endl;

    // Backwards compatibility: full filename given
    if (input->restart_file.find(".vtu")  != std::string::npos or
        input->restart_file.find(".pvtu") != std::string::npos)
    {
      restart(input->restart_file);
    }
    else if (input->restart_file.find(".pyfr") != std::string::npos)
    {
      restart_pyfr(input->restart_file);
    }
    else
      ThrowException("Unknown file type for restart file.");
  }
  else
  {
    if (input->rank == 0) std::cout << "Restarting solution from " + input->restart_case + "_" + std::to_string(input->restart_iter) + " ..." << std::endl;

    // New version: Use case name + iteration number to find file
    // [Overset compatible]
    if (input->restart_type == 1) // ParaView
      restart(input->restart_case, input->restart_iter);
    else if (input->restart_type == 2) // PyFR
      restart_pyfr(input->restart_case, input->restart_iter);
  }

#ifdef _GPU
  for (auto e : elesObjs)
   e->U_spts_d = e->U_spts;
#endif

  if (input->tavg)
  {
    tavg_prev_time = flow_time;
    accumulate_time_averages();
  }
  
  // Update grid to current status based on restart file if needed
  if (input->motion)
  {
    if (input->motion_type == RIGID_BODY)
    {
      Quat q(geo.q(0),geo.q(1),geo.q(2),geo.q(3));
      if (geo.gridID == 0)
        geo.Rmat = getRotationMatrix(q);
      else
        geo.Rmat = identityMatrix(3);

      // Wmat = Matrix to get velocity due to rotation
      // W = 'Spin' of omega (cross-product in matrix form)
      int c1[3] = {1,2,0}; // Cross-product index maps
      int c2[3] = {2,0,1};
      mdvector<double> W({3,3}, 0.);
      for (int i = 0; i < 3; i++)
      {
        W(i,c2[i]) =  geo.omega(c1[i]);
        W(i,c1[i]) = -geo.omega(c2[i]);
      }
      geo.Wmat.assign({3,3}, 0.);
      for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 3; j++)
          for (unsigned int k = 0; k < 3; k++)
            geo.Wmat(i,j) += geo.Rmat(i,k) * W(k,j);

#ifdef _GPU
      geo.x_cg_d = geo.x_cg;
      geo.vel_cg_d = geo.vel_cg;
      geo.Rmat_d = geo.Rmat;
      geo.Wmat_d = geo.Wmat;
#endif

#ifdef _BUILD_LIB
      if (input->overset)
        ZEFR->tg_update_transform(geo.Rmat.data(), geo.x_cg.data(), geo.nDims);
#endif
    }

    move(flow_time, true);
  }
  else
  {
#ifdef _BUILD_LIB
    if (input->overset)
      ZEFR->tg_point_connectivity();
#endif
  }
}

void FRSolver::create_elesObj(ELE_TYPE etype, unsigned int elesObjID, unsigned int startEle, unsigned int endEle)
{
  if (etype == QUAD)
  {
    if(!geo.nElesBT.count(TRI)) //note: nElesBT used here since TRI can be removed from ele_set during MPI preprocessing.
      elesObjs.push_back(std::make_shared<Quads>(&geo, input, elesObjID, startEle, endEle, order));
    else
    {
      std::cout << "Increased order of quads!" << std::endl;
      elesObjs.push_back(std::make_shared<Quads>(&geo, input, elesObjID, startEle, endEle, order+1));
    }
  }
  else if (etype == TRI)
  {
    if (input->viscous and !input->grad_via_div)
      ThrowException("Need to enable grad_via_div to use triangles for viscous problems!");

     elesObjs.push_back(std::make_shared<Tris>(&geo, input, elesObjID, startEle, endEle, order));
  }
  else if (etype == HEX)
     elesObjs.push_back(std::make_shared<Hexas>(&geo, input, elesObjID, startEle, endEle, order));
  else if (etype == TET)
  {
    if (input->viscous and !input->grad_via_div)
      ThrowException("Need to enable grad_via_div to use tetrahedra for viscous problems!");

     elesObjs.push_back(std::make_shared<Tets>(&geo, input, elesObjID, startEle, endEle, order));
  }
}

void FRSolver::orient_fpts()
{
  mdvector<double> fpt_coords_L({geo.nDims, geo.nGfpts}), fpt_coords_R({geo.nDims, geo.nGfpts});
  std::vector<double*> fpt_ptr_L(geo.nGfpts);
  std::vector<unsigned int> idxL(geo.nGfpts_int), idxR(geo.nGfpts_int), idxsort(geo.nGfpts_int);
 
  /* Gather all flux point coordinates */
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      unsigned int eleBT = ele + e->startEle;
      for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfptBT[e->etype](fpt,eleBT);
        int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,eleBT);

        for (unsigned int dim = 0; dim < geo.nDims; dim++)
        {
          if (slot == 0)
            fpt_coords_L(dim, gfpt) = e->coord_fpts(fpt, dim, ele);
          else
            fpt_coords_R(dim, gfpt) = e->coord_fpts(fpt, dim, ele);
        }

        if (slot == 0)
          fpt_ptr_L[gfpt] = &(e->coord_fpts(fpt, 0, ele));
      }
    }
  }


  for (unsigned int fpt = 0; fpt < geo.nGfpts_int; fpt++)
  {
    idxL[fpt] = fpt; idxR[fpt] = fpt; idxsort[fpt] = fpt;
  }

  /* Get consistent coupling via fuzzysort */
  for (unsigned int f = 0; f < geo.nGfpts_int/geo.nFptsPerFace; f++)
  {
    unsigned int shift = f * geo.nFptsPerFace;

    fuzzysort_ind(fpt_coords_L, idxL.data() + shift, geo.nFptsPerFace, geo.nDims);
    fuzzysort_ind(fpt_coords_R, idxR.data() + shift, geo.nFptsPerFace, geo.nDims);
  }

  /* Sort again to make left face index access coalesced memory */
  std::sort(idxsort.begin(), idxsort.end(), [&](unsigned int a, unsigned int b) {return fpt_ptr_L[a] < fpt_ptr_L[b];});

  // TODO: Can probably do this in fewer setups but this works.
  auto idxL_copy = idxL; auto idxR_copy = idxR;
  for (unsigned int fpt = 0; fpt < geo.nGfpts_int; fpt++)
  {
    idxL[fpt] = idxL_copy[idxsort[fpt]];
    idxR[fpt] = idxR_copy[idxsort[fpt]];
  }


  /* Invert the mapping */
  idxL_copy = idxL; idxR_copy = idxR;
  for (unsigned int fpt = 0; fpt < geo.nGfpts_int; fpt++)
  {
    idxL[idxL_copy[fpt]] = fpt;
    idxR[idxR_copy[fpt]] = fpt;
  }

  /* Reindex face flux points */
  for (auto etype : geo.ele_set)
  {
    auto fpt2gfptBT_copy = geo.fpt2gfptBT[etype];
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
    {
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEleBT[etype] * geo.nFptsPerFace; fpt++)
      {
        int slot = geo.fpt2gfpt_slotBT[etype](fpt,ele);
        int gfpt_old = fpt2gfptBT_copy(fpt, ele);

        if (gfpt_old >= geo.nGfpts_int) continue;

        if (slot == 0)
        {
          geo.fpt2gfptBT[etype](fpt, ele) = idxL[gfpt_old];
        }
        else
        {
          geo.fpt2gfptBT[etype](fpt, ele) = idxR[gfpt_old];
        }
      }
    }
  }

  auto face2fpts_copy = geo.face2fpts;
  for (int ff = 0; ff < geo.nFaces; ff++)
  {
    for (int fpt = 0; fpt < geo.nFptsPerFace; fpt++)
    {
      int gfpt_old = face2fpts_copy(fpt, ff);
      if (gfpt_old >= geo.nGfpts_int) continue;

      int gfpt_new = idxL[gfpt_old];
      geo.face2fpts(fpt, ff) = gfpt_new;
      geo.fpt2face[gfpt_new] = ff;
    }
  }

#ifdef _MPI
  /* For MPI, just use coupling from fuzzysort directly */
  for (auto &entry : geo.fpt_buffer_map)
  {
    auto &fpts = entry.second;
    for (unsigned int f = 0; f < fpts.size()/geo.nFptsPerFace; f++)
    {
      unsigned int shift = f * geo.nFptsPerFace;
      fuzzysort_ind(fpt_coords_L, fpts.data() + shift, geo.nFptsPerFace, geo.nDims);
    }
  }
#endif
}


void FRSolver::set_fpt_adjacency()
{
  /* Sizing fpt2fptN */
  if (geo.nDims == 2)
    geo.fpt2fptN.assign({geo.nFacesPerEleBT[QUAD] * geo.nFptsPerFace, geo.nEles}, -1);
  else
    geo.fpt2fptN.assign({geo.nFacesPerEleBT[HEX] * geo.nFptsPerFace, geo.nEles}, -1);

  /* Construct gfpt2fpt connectivity */
  mdvector<unsigned int> gfpt2fpt({2, geo.nGfpts}, -1);
  for (auto etype : geo.ele_set)
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEleBT[etype] * geo.nFptsPerFace; fpt++)
      {
        int gfpt = geo.fpt2gfptBT[etype](fpt,ele);
        int slot = geo.fpt2gfpt_slotBT[etype](fpt,ele);
        gfpt2fpt(slot, gfpt) = fpt;
      }

  /* Construct fpt2fptN connectivity */
  for (auto etype : geo.ele_set)
    for (unsigned int ele = 0; ele < geo.nElesBT[etype]; ele++)
      for (unsigned int fpt = 0; fpt < geo.nFacesPerEleBT[etype] * geo.nFptsPerFace; fpt++)
      {
        unsigned int eleID = geo.eleID[etype](ele);
        int gfpt = geo.fpt2gfptBT[etype](fpt,ele);
        int slot = geo.fpt2gfpt_slotBT[etype](fpt,ele);
        int notslot = (slot == 0) ? 1 : 0;
        geo.fpt2fptN(fpt, eleID) = gfpt2fpt(notslot, gfpt);
      }

#ifdef _MPI
  /* Send fpt2fptN connectivity on MPI faces */
  std::map<unsigned int, mdvector<unsigned int>> fpts_rbuffs, fpts_sbuffs;
  std::vector<MPI_Request> rreqs(geo.fpt_buffer_map.size());
  std::vector<MPI_Request> sreqs(geo.fpt_buffer_map.size());
  unsigned int idx = 0;
  for (const auto &entry : geo.fpt_buffer_map)
  {
    unsigned int rankN = entry.first;
    const auto &gfpts = entry.second;

    /* Stage nonblocking receives */
    fpts_rbuffs[rankN].assign({(unsigned int) gfpts.size()}, 0);
    MPI_Irecv(fpts_rbuffs[rankN].data(), (unsigned int) gfpts.size(), MPI_UNSIGNED, 
        rankN, 0, myComm, &rreqs[idx]);

    /* Pack buffer of fpt data */
    fpts_sbuffs[rankN].assign({(unsigned int) gfpts.size()}, 0);
    for (unsigned int i = 0; i < gfpts.size(); i++)
    {
      unsigned int gfpt = gfpts(i);
      unsigned int fpt = gfpt2fpt(0, gfpt);
      fpts_sbuffs[rankN](i) = fpt;
    }

    /* Send buffer to paired rank */
    MPI_Isend(fpts_sbuffs[rankN].data(), (unsigned int) gfpts.size(), MPI_UNSIGNED,
        rankN, 0, myComm, &sreqs[idx]);
    idx++;
  }
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

  /* Unpack buffer into fpt2fptN */
  for (const auto &entry : geo.fpt_buffer_map)
  {
    unsigned int rankN = entry.first;
    const auto &gfpts = entry.second;
    for (unsigned int i = 0; i < gfpts.size(); i++)
    {
      unsigned int gfpt = gfpts(i);
      unsigned int fpt = gfpt2fpt(0, gfpt);
      unsigned int faceID = geo.fpt2face[gfpt];
      unsigned int eleID = geo.face2eles(faceID, 0);
      geo.fpt2fptN(fpt, eleID) = fpts_rbuffs[rankN](i);
    }
  }
#endif
}

void FRSolver::setup_jacoN_views()
{
#ifdef _MPI
  /* Allocate memory for jacoN data on MPI faces */
  inv_jacoN_spts_mpibnd.assign({geo.nMpiFaces, eles->nDims, eles->nSpts, eles->nDims});
  jacoN_det_spts_mpibnd.assign({geo.nMpiFaces, eles->nSpts});

  /* Fill in inv_jacoN on MPI faces */
  /* Note: Consider creating a larger buffer if this takes too long */
  std::map<unsigned int, mdvector<double>> inv_jacoN_sbuffs;
  std::vector<MPI_Request> rreqs(geo.nMpiFaces);
  std::vector<MPI_Request> sreqs(geo.nMpiFaces);
  for (unsigned int mpiFace = 0; mpiFace < geo.nMpiFaces; mpiFace++)
  {
    unsigned int faceID = geo.mpiFaces[mpiFace];
    unsigned int eleID = geo.face2eles(faceID, 0);
    auto eles = elesObjs[ele2elesObj(eleID)];
    unsigned int rankN = geo.procR[mpiFace];

    /* Stage nonblocking receives */
    MPI_Irecv(&inv_jacoN_spts_mpibnd(mpiFace, 0, 0, 0), eles->nDims * eles->nSpts * eles->nDims, 
        MPI_DOUBLE, rankN, faceID, myComm, &rreqs[mpiFace]);

    /* Pack buffer of jacoN data */
    unsigned int faceNID = geo.faceID_R[mpiFace];
    unsigned int ele = eleID - geo.eleID[eles->etype](eles->startEle);
    inv_jacoN_sbuffs[faceNID].assign({eles->nDims, eles->nSpts, eles->nDims});
    for (unsigned int dim1 = 0; dim1 < eles->nDims; dim1++)
      for (unsigned int spt = 0; spt < eles->nSpts; spt++)
        for (unsigned int dim2 = 0; dim2 < eles->nDims; dim2++)
          inv_jacoN_sbuffs[faceNID](dim1, spt, dim2) = eles->inv_jaco_spts(dim1, spt, dim2, ele);

    /* Send buffer to paired rank */
    MPI_Isend(inv_jacoN_sbuffs[faceNID].data(), eles->nDims * eles->nSpts * eles->nDims, 
        MPI_DOUBLE, rankN, faceNID, myComm, &sreqs[mpiFace]);
  }
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

#ifdef _GPU
  /* Copy inv_jacoN to GPU */
  inv_jacoN_spts_mpibnd_d = inv_jacoN_spts_mpibnd;
#endif

  /* Fill in jacoN_det on MPI faces */
  std::map<unsigned int, mdvector<double>> jacoN_det_sbuffs;
  std::fill(rreqs.begin(), rreqs.end(), MPI_REQUEST_NULL);
  std::fill(sreqs.begin(), sreqs.end(), MPI_REQUEST_NULL);
  for (unsigned int mpiFace = 0; mpiFace < geo.nMpiFaces; mpiFace++)
  {
    unsigned int faceID = geo.mpiFaces[mpiFace];
    unsigned int eleID = geo.face2eles(faceID, 0);
    auto eles = elesObjs[ele2elesObj(eleID)];
    unsigned int rankN = geo.procR[mpiFace];

    /* Stage nonblocking receives */
    MPI_Irecv(&jacoN_det_spts_mpibnd(mpiFace, 0), eles->nSpts, MPI_DOUBLE, 
        rankN, faceID, myComm, &rreqs[mpiFace]);

    /* Pack buffer of jacoN data */
    unsigned int faceNID = geo.faceID_R[mpiFace];
    unsigned int ele = eleID - geo.eleID[eles->etype](eles->startEle);
    jacoN_det_sbuffs[faceNID].assign({eles->nSpts});
    for (unsigned int spt = 0; spt < eles->nSpts; spt++)
      jacoN_det_sbuffs[faceNID](spt) = eles->jaco_det_spts(spt, ele);

    /* Send buffer to paired rank */
    MPI_Isend(jacoN_det_sbuffs[faceNID].data(), eles->nSpts, MPI_DOUBLE,
        rankN, faceNID, myComm, &sreqs[mpiFace]);
  }
  MPI_Waitall(rreqs.size(), rreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(sreqs.size(), sreqs.data(), MPI_STATUSES_IGNORE);

#ifdef _GPU
  /* Copy jacoN_det to GPU */
  jacoN_det_spts_mpibnd_d = jacoN_det_spts_mpibnd;
#endif
#endif

  for (auto eles : elesObjs)
  {
    /* Setup jacoN views of neighboring element data */
    unsigned int base_stride = eles->nElesPad;
    mdvector<double*> inv_jacoN_base_ptrs({eles->nFaces*eles->nElesPad});
    mdvector<unsigned int> inv_jacoN_strides({3, eles->nFaces*eles->nElesPad});
    mdvector<double*> jacoN_det_base_ptrs({eles->nFaces*eles->nElesPad});
    mdvector<unsigned int> jacoN_det_strides({eles->nFaces*eles->nElesPad});

#ifdef _GPU
    mdvector<double*> inv_jacoN_base_ptrs_d({eles->nFaces*eles->nElesPad});
    mdvector<unsigned int> inv_jacoN_strides_d({3, eles->nFaces*eles->nElesPad});
    mdvector<double*> jacoN_det_base_ptrs_d({eles->nFaces*eles->nElesPad});
    mdvector<unsigned int> jacoN_det_strides_d({eles->nFaces*eles->nElesPad});
#endif

    /* Set pointers for jacoN */
    for (unsigned int ele = 0; ele < eles->nEles; ele++)
    {
      unsigned int eleID = geo.eleID[eles->etype](ele + eles->startEle);
      for (unsigned int face = 0; face < eles->nFaces; face++)
      {
#ifdef _MPI
        /* Element neighbor is on another rank */
        int faceID = geo.ele2face(eleID, face);
        int mpiFace = findFirst(geo.mpiFaces, faceID);
        if (mpiFace > -1)
        {
          inv_jacoN_base_ptrs(ele + face * base_stride) = &inv_jacoN_spts_mpibnd(mpiFace, 0, 0, 0);
          inv_jacoN_strides(0, ele + face * base_stride) = inv_jacoN_spts_mpibnd.get_stride(0);
          inv_jacoN_strides(1, ele + face * base_stride) = inv_jacoN_spts_mpibnd.get_stride(1);
          inv_jacoN_strides(2, ele + face * base_stride) = inv_jacoN_spts_mpibnd.get_stride(2);

          jacoN_det_base_ptrs(ele + face * base_stride) = &jacoN_det_spts_mpibnd(mpiFace, 0);
          jacoN_det_strides(ele + face * base_stride) = jacoN_det_spts_mpibnd.get_stride(0);

#ifdef _GPU
          inv_jacoN_base_ptrs_d(ele + face * base_stride) = inv_jacoN_spts_mpibnd_d.get_ptr(mpiFace, 0, 0, 0);
          inv_jacoN_strides_d(0, ele + face * base_stride) = inv_jacoN_spts_mpibnd_d.get_stride(0);
          inv_jacoN_strides_d(1, ele + face * base_stride) = inv_jacoN_spts_mpibnd_d.get_stride(1);
          inv_jacoN_strides_d(2, ele + face * base_stride) = inv_jacoN_spts_mpibnd_d.get_stride(2);

          jacoN_det_base_ptrs_d(ele + face * base_stride) = jacoN_det_spts_mpibnd_d.get_ptr(mpiFace, 0);
          jacoN_det_strides_d(ele + face * base_stride) = jacoN_det_spts_mpibnd_d.get_stride(0);
#endif
          continue;
        }
#endif

        /* Element neighbor is a boundary */
        int eleNID = geo.ele2eleN(face, eleID);
        if (eleNID == -1)
        {
          inv_jacoN_base_ptrs(ele + face * base_stride) = nullptr;
          inv_jacoN_strides(0, ele + face * base_stride) = 0;
          inv_jacoN_strides(1, ele + face * base_stride) = 0;
          inv_jacoN_strides(2, ele + face * base_stride) = 0;

          jacoN_det_base_ptrs(ele + face * base_stride) = nullptr;
          jacoN_det_strides(ele + face * base_stride) = 0;

#ifdef _GPU
          inv_jacoN_base_ptrs_d(ele + face * base_stride) = nullptr;
          inv_jacoN_strides_d(0, ele + face * base_stride) = 0;
          inv_jacoN_strides_d(1, ele + face * base_stride) = 0;
          inv_jacoN_strides_d(2, ele + face * base_stride) = 0;

          jacoN_det_base_ptrs_d(ele + face * base_stride) = nullptr;
          jacoN_det_strides_d(ele + face * base_stride) = 0;
#endif
        }

        /* Interior element neighbor */
        else
        {
          auto elesN = elesObjs[ele2elesObj(eleNID)];
          unsigned int eleN = eleNID - geo.eleID[elesN->etype](elesN->startEle);

          inv_jacoN_base_ptrs(ele + face * base_stride) = &elesN->inv_jaco_spts(0, 0, 0, eleN);
          inv_jacoN_strides(0, ele + face * base_stride) = elesN->inv_jaco_spts.get_stride(1);
          inv_jacoN_strides(1, ele + face * base_stride) = elesN->inv_jaco_spts.get_stride(2);
          inv_jacoN_strides(2, ele + face * base_stride) = elesN->inv_jaco_spts.get_stride(3);

          jacoN_det_base_ptrs(ele + face * base_stride) = &elesN->jaco_det_spts(0, eleN);
          jacoN_det_strides(ele + face * base_stride) = elesN->jaco_det_spts.get_stride(1);

#ifdef _GPU
          inv_jacoN_base_ptrs_d(ele + face * base_stride) = elesN->inv_jaco_spts_d.get_ptr(0, 0, 0, eleN);
          inv_jacoN_strides_d(0, ele + face * base_stride) = elesN->inv_jaco_spts_d.get_stride(1);
          inv_jacoN_strides_d(1, ele + face * base_stride) = elesN->inv_jaco_spts_d.get_stride(2);
          inv_jacoN_strides_d(2, ele + face * base_stride) = elesN->inv_jaco_spts_d.get_stride(3);

          jacoN_det_base_ptrs_d(ele + face * base_stride) = elesN->jaco_det_spts_d.get_ptr(0, eleN);
          jacoN_det_strides_d(ele + face * base_stride) = elesN->jaco_det_spts_d.get_stride(1);
#endif
        }
      }
    }

    /* Create jacoN views of neighboring element data */
    eles->inv_jacoN_spts.assign(inv_jacoN_base_ptrs, inv_jacoN_strides, base_stride);
    eles->jacoN_det_spts.assign(jacoN_det_base_ptrs, jacoN_det_strides, base_stride);

#ifdef _GPU
    eles->inv_jacoN_spts_d.assign(inv_jacoN_base_ptrs_d, inv_jacoN_strides_d, base_stride);
    eles->jacoN_det_spts_d.assign(jacoN_det_base_ptrs_d, jacoN_det_strides_d, base_stride);
#endif
  }
}


void FRSolver::setup_update()
{
  /* Setup variables for timestepping scheme */
  if (input->dt_scheme == "Euler")
  {
    input->nStages = 1;
    rk_beta.assign({input->nStages}, 1.0);
    rk_c.assign({input->nStages}, 0.0);
  }
  else if (input->dt_scheme == "RK44")
  {
    input->nStages = 4;
    
    rk_alpha.assign({input->nStages-1});
    rk_alpha(0) = 0.5; rk_alpha(1) = 0.5; rk_alpha(2) = 1.0;

    rk_c.assign({input->nStages});
    rk_c(1) = 0.5; rk_c(2) = 0.5; rk_c(3) = 1.0;

    rk_beta.assign({input->nStages});
    rk_beta(0) = 1./6.; rk_beta(1) = 1./3.; 
    rk_beta(2) = 1./3.; rk_beta(3) = 1./6.;
  }
  else if (input->dt_scheme == "RKj")
  {
    input->nStages = 4;
    rk_alpha.assign({input->nStages});
    /* Standard RK44 */
    //rk_alpha(0) = 1./4; rk_alpha(1) = 1./3.; 
    //rk_alpha(2) = 1./2.; rk_alpha(3) = 1.0;
    /* OptRK4 (r = 0.5) */
    rk_alpha(0) = 0.153; rk_alpha(1) = 0.442; 
    rk_alpha(2) = 0.930; rk_alpha(3) = 1.0;

    rk_c = rk_alpha;
  }
  else if (input->dt_scheme == "RK54")
  {
    input->nStages = 5;
    rk_alpha.assign({input->nStages - 1});
    rk_alpha(0) =   970286171893. / 4311952581923.;
    rk_alpha(1) =  6584761158862. / 12103376702013.;
    rk_alpha(2) =  2251764453980. / 15575788980749.;
    rk_alpha(3) = 26877169314380. / 34165994151039.;

    rk_beta.assign({input->nStages});
    rk_beta(0) =  1153189308089. / 22510343858157.;
    rk_beta(1) =  1772645290293. / 4653164025191.;
    rk_beta(2) = -1672844663538. / 4480602732383.;
    rk_beta(3) =  2114624349019. / 3568978502595.;
    rk_beta(4) =  5198255086312. / 14908931495163.;

    rk_bhat.assign({input->nStages});
    rk_bhat(0) =  1016888040809. / 7410784769900.;
    rk_bhat(1) = 11231460423587. / 58533540763752.;
    rk_bhat(2) = -1563879915014. / 6823010717585.;
    rk_bhat(3) =   606302364029. / 971179775848.;
    rk_bhat(4) =  1097981568119. / 3980877426909.;

    rk_c.assign({input->nStages});
    for (int i = 1; i < input->nStages; i++)
    {
      rk_c(i) = rk_alpha(i-1);

      for (int j = 0; j < i-1; j++)
        rk_c(i) += rk_beta(j);
    }
  }
  else if (input->dt_scheme == "Steady")
  {
    input->nStages = 1;
    rk_alpha.assign({input->nStages, input->nStages});
    rk_alpha(0, 0) = 1.0;
  }
  else if (input->dt_scheme == "DIRK34")
  {
    input->nStages = 3;
    double alp = 2.0 * std::cos(M_PI / 18.0) / std::sqrt(3.0);

    rk_alpha.assign({input->nStages, input->nStages},0);
    rk_alpha(0,0) = (1.0 + alp) / 2.0;
    rk_alpha(1,0) = -alp / 2.0;
    rk_alpha(1,1) = (1.0 + alp) / 2.0;
    rk_alpha(2,0) = 1.0 + alp;
    rk_alpha(2,1) = -(1.0 + 2.0*alp);
    rk_alpha(2,2) = (1.0 + alp) / 2.0;

    rk_beta.assign({input->nStages});
    rk_beta(0) = 1.0 / (6.0*alp*alp);
    rk_beta(1) = 1.0 - 1.0 / (3.0*alp*alp);
    rk_beta(2) = 1.0 / (6.0*alp*alp);

    rk_c.assign({input->nStages});
    rk_c(0) = (1.0 + alp) / 2.0;
    rk_c(1) = 1.0 / 2.0;
    rk_c(2) = (1.0 - alp) / 2.0;
  }
  else if (input->dt_scheme == "ESDIRK43")
  {
    input->nStages = 4; // Explicit first stage
    startStage = 1;

    double gamma = 1767732205903. / 4055673282236.;
    rk_alpha.assign({input->nStages, input->nStages},0);
    rk_alpha(1,0) = gamma;
    rk_alpha(1,1) = gamma;
    rk_alpha(2,2) = gamma;
    rk_alpha(3,3) = gamma;
    rk_alpha(2,0) =  2746238789719. / 10658868560708.;
    rk_alpha(2,1) =  -640167445237. / 6845629431997.;
    rk_alpha(3,0) =  1471266399579. / 7840856788654.;
    rk_alpha(3,1) = -4482444167858. / 7529755066697.;
    rk_alpha(3,2) = 11266239266428. / 11593286722821.;

    rk_beta.assign({input->nStages});
    for (unsigned int s = 0; s < input->nStages; s++)
      rk_beta(s) = rk_alpha(input->nStages-1,s);

    rk_bhat.assign({input->nStages});
    rk_bhat(0) =   2756255671327. / 12835298489170.;
    rk_bhat(1) = -10771552573575. / 22201958757719.;
    rk_bhat(2) =   9247589265047. / 10645013368117.;
    rk_bhat(3) =   2193209047091. / 5459859503100.;

    rk_c.assign({input->nStages},0);
    rk_c(1) = 2. * gamma;
    rk_c(2) = 3. / 5.;
    rk_c(3) = 1.;
  }
  else if (input->dt_scheme == "ESDIRK64")
  {
    input->nStages = 6; // Explicit first stage
    startStage = 1;

    double gamma = 1. / 4.;
    rk_alpha.assign({input->nStages, input->nStages},0);
    rk_alpha(1,0) = gamma;
    rk_alpha(1,1) = gamma;
    rk_alpha(2,2) = gamma;
    rk_alpha(3,3) = gamma;
    rk_alpha(4,4) = gamma;
    rk_alpha(5,5) = gamma;
    rk_alpha(2,0) =        8611. / 62500.;
    rk_alpha(2,1) =       -1743. / 31250;
    rk_alpha(3,0) =     5012029. / 34652500.;
    rk_alpha(3,1) =     -654441. / 2922500.;
    rk_alpha(3,2) =      174375. / 388108.;
    rk_alpha(4,0) = 15267082809. / 155376265600.;
    rk_alpha(4,1) =   -71443401. / 120774400.;
    rk_alpha(4,2) =   730878875. / 902184768.;
    rk_alpha(4,3) =     2285395. / 8070912.;
    rk_alpha(5,0) =       82889. / 524892.;
    rk_alpha(5,1) =           0.;
    rk_alpha(5,2) =       15625. / 83664.;
    rk_alpha(5,3) =       69875. / 102672.;
    rk_alpha(5,4) =       -2260. / 8211.;

    rk_beta.assign({input->nStages});
    for (unsigned int s = 0; s < input->nStages; s++)
      rk_beta(s) = rk_alpha(input->nStages-1,s);

    rk_bhat.assign({input->nStages});
    rk_bhat(0) = 4586570599. / 29645900160.;
    rk_bhat(1) =          0.;
    rk_bhat(2) =  178811875. / 945068544.;
    rk_bhat(3) =  814220225. / 1159782912.;
    rk_bhat(4) =   -3700637. / 11593932.;
    rk_bhat(5) =      61727. / 225920.;

    rk_c.assign({input->nStages},0);
    rk_c(1) = 2. * gamma;
    rk_c(2) = 83. / 250.;
    rk_c(3) = 31. / 50.;
    rk_c(4) = 17. / 20.;
    rk_c(5) = 1.;
  }
  else
  {
    ThrowException("dt_scheme not recognized!");
  }

  if (input->adapt_dt)
  {
    // Order of time stepping scheme
    double p;
    if (input->dt_scheme == "RK54")
      p = 4;
    else if (input->dt_scheme == "ESDIRK43")
      p = 3;
    else if (input->dt_scheme == "ESDIRK64")
      p = 4;
    else
      ThrowException("Embedded pairs not implemented for this dt_scheme!");

    expa = input->pi_alpha / p;
    expb = input->pi_beta / p;
    prev_err = 1.;
  }

  if (input->implicit_method)
  {
    if (input->iterative_method == JAC)
      nCounter = 1;
    else if (input->iterative_method == MCGS)
      nCounter = geo.nColors;
    if (input->backsweep)
      nCounter *= 2;
  }
}

void FRSolver::setup_output()
{
  /* Create output directory to store data files */
  if (input->rank == 0)
  {
    std::string cmd = "mkdir -p " + input->output_prefix;
    system(cmd.c_str());
  }

#ifdef _MPI
    MPI_Barrier(worldComm);
#endif
       
  for (auto e : elesObjs)
    e->setup_ppt_connectivity();
}

void FRSolver::restart(std::string restart_file, unsigned restart_iter)
{
  if (input->restart_type > 0) // append .pvtu / .vtu to case name
  {
    std::stringstream ss;

    ss << restart_file << "/" << restart_file;

    if (input->overset)
    {
      ss << "_Grid" << input->gridID;
    }

    ss << "_" << std::setw(9) << std::setfill('0') << restart_iter;

#ifdef _MPI
    ss << ".pvtu";
#else
    ss << ".vtu";
#endif

    restart_file = ss.str();
  }

  size_t pos;
#ifdef _MPI
  /* From .pvtu, form partition specific filename */
  pos = restart_file.rfind(".pvtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .pvtu file for parallel restart!");
  }

  restart_file = restart_file.substr(0, pos);

  std::stringstream ss;
  ss << std::setw(3) << std::setfill('0') << input->rank;

  restart_file += "_" + ss.str() + ".vtu";
#endif

  /* Open .vtu file */
  std::ifstream f(restart_file);
  pos = restart_file.rfind(".vtu");
  if (pos == std::string::npos)
  {
    ThrowException("Must provide .vtu file for restart!");
  }

  if (!f.is_open())
  {
    ThrowException("Could not open restart file " + restart_file + "!");
  }

  std::string param, line;
  double val;
  unsigned int order_restart;
  std::vector<mdvector<double>> U_restart(elesObjs.size());

  bool has_gv = false; // can do the same here for all 'extra' fields

  /* Load data from restart file */
  while (f >> param)
  {
    if (param == "TIME")
    {
      f >> restart_time;
      flow_time = restart_time;
    }
    if (param == "ITER")
    {
      f >> current_iter;
      this->restart_iter = current_iter;
      input->iter = current_iter;
      input->initIter = current_iter;
    }
    if (param == "ORDER")
    {
      f >> order_restart;
    }
    if (param == "X_CG")
    {
      geo.x_cg.assign({3});
      f >> geo.x_cg(0) >> geo.x_cg(1) >> geo.x_cg(2);
    }
    if (param == "V_CG")
    {
      geo.vel_cg.assign({3});
      f >> geo.vel_cg(0) >> geo.vel_cg(1) >> geo.vel_cg(2);
    }
    if (param == "ROT-Q")
    {
      geo.q.assign({4});
      f >> geo.q(0) >> geo.q(1) >> geo.q(2) >> geo.q(3);
    }
    if (param == "OMEGA")
    {
      geo.omega.assign({3});
      f >> geo.omega(0) >> geo.omega(1) >> geo.omega(2);
    }
    if (param == "IBLANK" && input->overset)
    {
      geo.iblank_cell.assign({geo.nEles});
      for (unsigned int ele = 0; ele < elesObjs[0]->nEles; ele++)
        f >> geo.iblank_cell(ele);
    }
    if (param == "grid_velocity")
    {
      has_gv = true;
    }

    if (param == "<AppendedData")
    {
      std::getline(f,line);
      f.ignore(1); 

      unsigned int nRpts;
      /* Setup extrapolation operator from equistant restart points */
      for (auto e : elesObjs)
      {
        if (e->etype == QUAD and geo.nElesBT.count(TRI)) // to deal with increased quad order with mixed grids
        {
          e->set_oppRestart(order_restart + 1, true);
        }
        else
          e->set_oppRestart(order_restart, true);

        nRpts = e->oppRestart.get_dim(1);
        U_restart[e->elesObjID].assign({nRpts, e->nVars, e->nElesPad});
      }

      unsigned int temp; 
      for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
      {
        binary_read(f, temp);
        for (auto e: elesObjs)
        {
          nRpts = e->oppRestart.get_dim(1);

          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            /// TODO: make sure this is setup correctly first [and implement everywhere iblank_cell is used
            //if (input->overset && geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) continue;
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

            for (unsigned int rpt = 0; rpt < nRpts; rpt++)
            {
              binary_read(f, U_restart[e->elesObjID](rpt, n, ele));
            }
          }
        }
      }

      /* Extrapolate values from restart points to solution points */
      for (auto e : elesObjs)
      {
        nRpts = e->oppRestart.get_dim(1);

        auto &A = e->oppRestart(0, 0);
        auto &B = U_restart[e->elesObjID](0, 0, 0);
        auto &C = e->U_spts(0, 0, 0);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nSpts, 
            e->nElesPad * e->nVars, nRpts, 1.0, &A, nRpts, &B, 
            e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);
      }

      /* Read grid velocity NOTE: if any additional fields exist after,
       * remove 'if input->motion' or else read will occur in wrong location */
      if (input->motion && has_gv)
      {
        for (auto e : elesObjs)
        {
          U_restart[e->elesObjID].assign({nRpts, 3, e->nEles});

          unsigned int temp;
          binary_read(f, temp);
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

            for (unsigned int rpt = 0; rpt < nRpts; rpt++)
              for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
                binary_read(f, U_restart[e->elesObjID](rpt, n, ele));
          }

          /* Extrapolate values from restart points to solution points */
          int m = e->nSpts;
          int k = nRpts;
          int n = e->nEles * e->nDims;
          auto &A = e->oppRestart(0, 0);
          auto &B = U_restart[e->elesObjID](0, 0, 0);
          auto &C = e->grid_vel_spts(0, 0, 0);

          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                      1.0, &A, k, &B, n, 0.0, &C, n);
        }
      }
    }
  }

  f.close();
}

#ifdef _GPU
void FRSolver::solver_data_to_device()
{
  /* Initial copy of data to GPU. Assignment operator will allocate data on device when first
   * used. */

  /* -- Geometry Data -- */
  geo.gfpt2bnd_d = geo.gfpt2bnd;
  for (auto etype : geo.ele_set)
  {
    geo.fpt2gfptBT_d[etype] = geo.fpt2gfptBT[etype];
    geo.fpt2gfpt_slotBT_d[etype] = geo.fpt2gfpt_slotBT[etype];

    if (input->motion)
    {
      geo.ele2nodesBT_d[etype] = geo.ele2nodesBT[etype];
    }

    if (input->implicit_method && input->viscous && !input->KPF_Jacobian)
      geo.eleID_d[etype] = geo.eleID[etype];
  }

  if (input->implicit_method && input->viscous && !input->KPF_Jacobian)
  {
    geo.ele2eleN_d = geo.ele2eleN;
    geo.face2faceN_d = geo.face2faceN;
    geo.fpt2fptN_d = geo.fpt2fptN;
  }
  geo.flip_beta_d = geo.flip_beta;

  /* -- Element Data -- */
  for (auto e : elesObjs)
  {
    e->oppE_d = e->oppE;
    e->oppD_d = e->oppD;
    e->oppD_fpts_d = e->oppD_fpts;
    e->oppDiv_d = e->oppDiv;
    e->oppDiv_fpts_d = e->oppDiv_fpts;

    e->U_ini_d = e->U_ini;
    e->dt_d = e->dt;
    e->U_spts_d = e->U_spts;
    e->U_fpts_d = e->U_fpts;
    e->Uavg_d = e->Uavg;
    e->weights_spts_d = e->weights_spts;
    e->weights_fpts_d = e->weights_fpts;
    e->Fcomm_d = e->Fcomm;
    e->F_spts_d = e->F_spts;
    e->divF_spts_d = e->divF_spts;
    e->coord_spts_d = e->coord_spts;
    e->inv_jaco_spts_d = e->inv_jaco_spts;
    e->jaco_det_spts_d = e->jaco_det_spts;
    e->vol_d = e->vol;

    if (input->grad_via_div)
    {
      e->norm_fpts_d = e->norm_fpts;
      e->dA_fpts_d = e->dA_fpts;
    }

    if (input->CFL_type == 2 || input->CFL_tau_type == 2)
      e->h_ref_d = e->h_ref;

    if (input->viscous)
    {
      e->dU_spts_d = e->dU_spts;
      e->Ucomm_d = e->Ucomm;
      e->dU_fpts_d = e->dU_fpts;
    }

    if (input->dt_scheme == "RK54")
      e->U_til_d = e->U_til;
    
    if (input->adapt_dt)
      e->rk_err_d = e->rk_err;

    //TODO: Temporary fix. Need to remove usage of jaco_spts_d from all kernels.
    if (input->motion)
    {
      e->jaco_spts_d = e->jaco_spts;
    }

    if (input->overset || input->motion)
    {
      e->nodes_d = e->nodes;
      e->coord_fpts_d = e->coord_fpts; /// TODO: use...
    }

    if (input->motion)
    {
      e->grid_vel_nodes_d = e->grid_vel_nodes;
      e->grid_vel_spts_d = e->grid_vel_spts;
      e->grid_vel_fpts_d = e->grid_vel_fpts;
      e->shape_spts_d = e->shape_spts;
      e->shape_fpts_d = e->shape_fpts;
      e->dshape_spts_d = e->dshape_spts;
      e->dshape_fpts_d = e->dshape_fpts;
      e->jaco_fpts_d = e->jaco_fpts;
      e->inv_jaco_fpts_d = e->inv_jaco_fpts;
      e->tnorm_d = e->tnorm;
      

      if (input->motion_type == RIGID_BODY)
      {
        if (input->viscous)
          e->inv_jaco_spts_init_d = e->inv_jaco_spts;
        e->jaco_spts_init_d = e->jaco_spts;
      }
    }

    if (input->implicit_method)
    {
      if (input->pseudo_time)
      {
        e->dtau_d = e->dtau;
        if (!input->remove_deltaU)
          e->U_iniNM_d = e->U_iniNM;
      }

      if (input->KPF_Jacobian)
      {
        e->oppD_spts1D_d = e->oppD_spts1D;
        e->oppDE_spts1D_d = e->oppDE_spts1D;
        e->oppDivE_spts1D_d = e->oppDivE_spts1D;
      }

      e->dFdU_spts_d = e->dFdU_spts;
      e->dFcdU_d = e->dFcdU;
      if (input->viscous)
      {
        e->dFddU_spts_d = e->dFddU_spts;
        e->dUcdU_d = e->dUcdU;
        e->dFcddU_d = e->dFcddU;
      }

      e->LHS_d = e->LHS;
      e->RHS_d = e->RHS;
      e->deltaU_d = e->deltaU;

      unsigned int N = e->nSpts * e->nVars;
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        e->LHS_ptrs(ele) = e->LHS_d.data() + ele * (N * N);
        e->RHS_ptrs(ele) = e->RHS_d.data() + ele * N;
        e->deltaU_ptrs(ele) = e->deltaU_d.data() + ele * N;
      }
      e->LHS_ptrs_d = e->LHS_ptrs;
      e->RHS_ptrs_d = e->RHS_ptrs;
      e->deltaU_ptrs_d = e->deltaU_ptrs;
      e->LHS_info_d = e->LHS_info;

      if (input->linear_solver == INV)
      {
        e->LHSinv_d = e->LHSinv;
        for (unsigned int ele = 0; ele < e->nEles; ele++)
          e->LHSinv_ptrs(ele) = e->LHSinv_d.data() + ele * (N * N);
        e->LHSinv_ptrs_d = e->LHSinv_ptrs;
      }
    }

    /* -- Averaging and Statistics -- */

    if (input->tavg)
    {
      e->tavg_acc_d = e->tavg_acc;
      e->tavg_prev_d = e->tavg_prev;
      e->tavg_curr_d = e->tavg_curr;
    }
  }

  if (input->overset || input->motion)
  {
    geo.coord_nodes_d = geo.coord_nodes;
  }

  if (input->motion)
  {
    geo.coords_init_d = geo.coords_init;
    geo.grid_vel_nodes_d = geo.grid_vel_nodes;

    /* Moving-grid parameters for convenience / ease of future additions
     * (add to input.hpp, then also here) */
    motion_vars.moveAx = input->moveAx;
    motion_vars.moveAy = input->moveAy;
    motion_vars.moveAz = input->moveAz;
    motion_vars.moveFx = input->moveFx;
    motion_vars.moveFy = input->moveFy;
    motion_vars.moveFz = input->moveFz;

    if (input->motion_type == RIGID_BODY)
    {
      faces->norm_init_d = faces->norm;

      nodes_ini_d = eles->nodes;
      nodes_til_d = eles->nodes;
      geo.x_cg_d = geo.x_cg;
      x_ini_d = geo.x_cg;
      x_til_d = geo.x_cg;
      geo.vel_cg_d = geo.vel_cg;
      v_ini_d = geo.vel_cg;
      v_til_d = geo.vel_cg;
      geo.q_d = geo.q;
      q_ini_d = geo.q;
      q_til_d = geo.q;
      geo.qdot_d = geo.qdot;
      qdot_ini_d = geo.qdot;
      qdot_til_d = geo.qdot;

      geo.Rmat_d = geo.Rmat;
      geo.Wmat_d = geo.Wmat;

      if (geo.nBndFaces != 0)
      {
        force_d.set_size({geo.nBndFaces, geo.nDims});
        moment_d.set_size({geo.nBndFaces, geo.nDims});
        device_fill(force_d, force_d.max_size(), 0.);
        device_fill(moment_d, moment_d.max_size(), 0.);
      }
    }
  }

  /* -- Face Data -- */
  faces->U_bnd_d = faces->U_bnd;
  faces->U_bnd_ldg_d = faces->U_bnd_ldg;
  faces->P_d = faces->P;
  faces->Ucomm_bnd_d = faces->Ucomm_bnd;
  faces->Fcomm_bnd_d = faces->Fcomm_bnd;
  faces->norm_d = faces->norm;
  faces->dA_d = faces->dA;
  faces->waveSp_d = faces->waveSp;
  faces->diffCo_d = faces->diffCo;
  faces->rus_bias_d = faces->rus_bias;
  faces->LDG_bias_d = faces->LDG_bias;

  if (input->viscous)
  {
    faces->dU_bnd_d = faces->dU_bnd;
  }

  if (input->motion)
  {
    faces->Vg_d = faces->Vg;
  }

  if (input->overset || input->motion)
    faces->coord_d = faces->coord;

  if (input->implicit_method)
  {
    faces->dFcdU_bnd_d = faces->dFcdU_bnd;
    faces->dUbdU_d = faces->dUbdU;

    if (input->viscous)
    {
      faces->dUcdU_bnd_d = faces->dUcdU_bnd;
      faces->dFcddU_bnd_d = faces->dFcddU_bnd;
      faces->ddUbddU_d = faces->ddUbddU;
    }
  }

  /* -- Additional data -- */
  /* Input parameters */
  input->V_fs_d = input->V_fs;
  input->V_wall_d = input->V_wall;
  input->norm_fs_d = input->norm_fs;
  input->AdvDiff_A_d = input->AdvDiff_A;

  rk_alpha_d = rk_alpha;
  rk_beta_d = rk_beta;
  if (input->adapt_dt)
    rk_bhat_d = rk_bhat;

#ifdef _MPI
  /* MPI data */
  for (auto &entry : geo.fpt_buffer_map) 
  {
    int pairedRank = entry.first;
    auto &fpts = entry.second;
    geo.fpt_buffer_map_d[pairedRank] = fpts;
    faces->U_sbuffs_d[pairedRank] = faces->U_sbuffs[pairedRank];
    faces->U_rbuffs_d[pairedRank] = faces->U_rbuffs[pairedRank];
  }

#endif
}
#endif

void FRSolver::compute_residual(unsigned int stage, int color)
{
#ifdef _BUILD_LIB
  // Record event for start of compute_res (solution up-to-date for calculation)
  if (input->overset)
    event_record_wait_pair(2, 0, 3);
#endif

  unsigned int startFpt = 0; unsigned int endFpt = geo.nGfpts;

#ifdef _MPI
  endFpt = geo.nGfpts_int + geo.nGfpts_bnd;
  unsigned int startFptMpi = endFpt;
#endif

  /* MCGS: Extrapolate solution on the previous color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[prev_color];
  else
    elesObjs = this->elesObjs;

  /* Extrapolate solution to flux points */
  for (auto e : elesObjs)
    e->extrapolate_U();

  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    for (auto e : elesObjs)
    {
      e->compute_Uavg();
      e->poly_squeeze();
    }
  }

  // Wait for completion of extrapolated solution before packing MPI buffers
  event_record_wait_pair(0, 0, 1);

  overset_u_send();

  /* MCGS: Compute, transform and extrapolate gradient on all colors */
  if (color > -1)
  {
    if (input->viscous)
      elesObjs = this->elesObjs;
    else
      elesObjs = elesObjsBC[color];
  }

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  if (input->viscous)
  {
    // gradient computation from divergence
    if (input->grad_via_div)
    {
      overset_u_recv();

      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        // Compute unit advection flux at solution points with wavespeed along dim
        for (auto e : elesObjs)
          e->compute_unit_advF(dim);

        // Compute solution point contributions to physical gradient (times jacobian determinant) along dim via divergence of F
        for (auto e : elesObjs)
          e->compute_dU_spts_via_divF(dim);
      }

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();

      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        // Convert common U to common normal advection flux
        for (auto e : elesObjs)
          e->common_U_to_F(dim);

        // Compute flux point contributions to physical gradient (times jacobian determinant) along dim via divergence of F
        for (auto e : elesObjs)
          e->compute_dU_fpts_via_divF(dim);
      }
    }
    else
    {
      /* Compute solution point contribution to (corrected) gradient of state variables at solution points */
      for (auto e : elesObjs)
        e->compute_dU_spts();
      
      overset_u_recv();

      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();

      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      /* Compute flux point contribution to (corrected) gradient of state variables at solution points */
      for (auto e : elesObjs)
        e->compute_dU_fpts();
      
    }
  }

  /* Compute flux at solution points */
  for (auto e : elesObjs)
    e->compute_F();

  if (input->viscous)
  {
    /* Interpolate gradient data to/from other grid(s) */
    overset_grad_send();

    /* Extrapolate physical solution gradient (computed during compute_F) to flux points */
    for (auto e : elesObjs)
      e->extrapolate_dU();

    // Wait for completion of extrapolated gradient before packing MPI buffers
    event_record_wait_pair(0, 0, 1);

#ifdef _BUILD_LIB
    if (input->overset)
      event_record_wait_pair(4, 0, 3);
#endif

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();
#endif

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* MCGS: Compute residual on current color */
    if (color > -1)
      elesObjs = elesObjsBC[color];
  }
  else
  {
    overset_u_recv();
  }

  /* Compute solution point contribution to divergence of flux */
  for (auto e : elesObjs)
    e->compute_divF_spts(stage);

  /* Unpack gradient data from other grid(s) */
  if (input->viscous)
    overset_grad_recv();

  /* Compute common interface flux at non-MPI flux points */
  faces->compute_common_F(startFpt, endFpt);

#ifdef _MPI
  if (!input->viscous)
  {
    /* Receive solution data */
    faces->recv_U_data();
  }
  else
  {
    /* Receive gradient data */
    faces->recv_dU_data();
  }

  /* Complete computation of fluxes */
  faces->compute_common_F(startFptMpi, geo.nGfpts);
#endif

  /* Compute flux point contribution to divergence of flux */
  for (auto e : elesObjs)
    e->compute_divF_fpts(stage);
  
  /* Add source term (if required) */
  if (input->source)
  {
    for (auto e : elesObjs)
      e->add_source(stage, flow_time);
  }

}

void FRSolver::compute_residual_start(unsigned int stage, int color)
{
#ifdef _BUILD_LIB
  // Record event for start of compute_res (solution up-to-date for calculation)
  if (input->overset)
    event_record_wait_pair(2, 0, 3);
#endif

  /* MCGS: Extrapolate solution on the previous color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[prev_color];
  else
    elesObjs = this->elesObjs;

  /* Extrapolate solution to flux points */
  for (auto e : elesObjs)
    e->extrapolate_U();

  /* If "squeeze" stabilization enabled, apply  it */
  if (input->squeeze)
  {
    for (auto e : elesObjs)
    {
      e->compute_Uavg();
      e->poly_squeeze();
    }
  }

  // Wait for completion of extrapolated solution before packing MPI buffers
  event_record_wait_pair(0, 0, 1);

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  if (input->viscous && !input->grad_via_div)
  {
    /* Compute solution point contribution to (corrected) gradient of state variables at solution points */
    for (auto e : elesObjs)
      e->compute_dU_spts();
  }
}

void FRSolver::compute_residual_mid(unsigned int stage, int color)
{
  unsigned int startFpt = 0; unsigned int endFpt = geo.nGfpts;

#ifdef _MPI
  endFpt = geo.nGfpts_int + geo.nGfpts_bnd;
  unsigned int startFptMpi = endFpt;
#endif

  /* MCGS: Extrapolate solution on the previous color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[prev_color];
  else
    elesObjs = this->elesObjs;

  /* MCGS: Compute, transform and extrapolate gradient on all colors */
  if (color > -1)
  {
    if (input->viscous)
      elesObjs = this->elesObjs;
    else
      elesObjs = elesObjsBC[color];
  }

#ifdef _MPI
  /* Commence sending U data to other processes */
  faces->send_U_data();
#endif

  /* Apply boundary conditions to state variables */
  faces->apply_bcs();

  if (input->viscous)
  {
    // gradient computation from divergence
    if (input->grad_via_div)
    {
      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        // Compute unit advection flux at solution points with wavespeed along dim
        for (auto e : elesObjs)
          e->compute_unit_advF(dim);

        // Compute solution point contributions to physical gradient (times jacobian determinant) along dim via divergence of F
        for (auto e : elesObjs)
          e->compute_dU_spts_via_divF(dim);
      }

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();

      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        // Convert common U to common normal advection flux
        for (auto e : elesObjs)
          e->common_U_to_F(dim);

        // Compute flux point contributions to physical gradient (times jacobian determinant) along dim via divergence of F
        for (auto e : elesObjs)
          e->compute_dU_fpts_via_divF(dim);
      }
    }
    else
    {
      /* Compute common interface solution and convective flux at non-MPI flux points */
      faces->compute_common_U(startFpt, endFpt);

#ifdef _MPI
      /* Receieve U data */
      faces->recv_U_data();

      /* Complete computation on remaining flux points */
      faces->compute_common_U(startFptMpi, geo.nGfpts);
#endif

      /* Compute flux point contribution to (corrected) gradient of state variables at solution points */
      for (auto e : elesObjs)
        e->compute_dU_fpts();
    }
  }

  /* Compute flux at solution points */
  for (auto e : elesObjs)
    e->compute_F();

  if (input->viscous)
  {
#ifdef _BUILD_LIB
    // Wait for completion of corrected *and transformed* gradient before sending
    if (input->overset)
      event_record_wait_pair(3, 0, 3);
#endif

    /* Extrapolate physical solution gradient (computed during compute_F) to flux points */
    for (auto e : elesObjs)
      e->extrapolate_dU();

    // Wait for completion of extrapolated gradient before packing MPI buffers
    event_record_wait_pair(0, 0, 1);

#ifdef _BUILD_LIB
    if (input->overset)
      event_record_wait_pair(4, 0, 3);
#endif

#ifdef _MPI
    /* Commence sending gradient data to other processes */
    faces->send_dU_data();
#endif

    /* Apply boundary conditions to the gradient */
    faces->apply_bcs_dU();

    /* MCGS: Compute residual on current color */
    if (color > -1)
      elesObjs = elesObjsBC[color];
  }

  /* Compute solution point contribution to divergence of flux */
  for (auto e : elesObjs)
    e->compute_divF_spts(stage);
}


void FRSolver::compute_residual_finish(unsigned int stage, int color)
{
  unsigned int startFpt = 0; unsigned int endFpt = geo.nGfpts;

#ifdef _MPI
  endFpt = geo.nGfpts_int + geo.nGfpts_bnd;
  unsigned int startFptMpi = endFpt;
#endif

  /* MCGS: Extrapolate solution on the previous color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[prev_color];
  else
    elesObjs = this->elesObjs;

#ifdef _BUILD_LIB
  // Wait for updated data on GPU before moving on to common_F
  if (input->overset)
    event_record_wait_pair(2, 3, 0);
#endif

  /* Compute common interface flux at non-MPI flux points */
  faces->compute_common_F(startFpt, endFpt);

#ifdef _MPI
  if (!input->viscous)
  {
    /* Receive solution data */
    faces->recv_U_data();
  }
  else
  {
    /* Receive gradient data */
    faces->recv_dU_data();
  }

  /* Complete computation of fluxes */
  faces->compute_common_F(startFptMpi, geo.nGfpts);
#endif

  /* Compute flux point contribution to divergence of flux */
  for (auto e : elesObjs)
    e->compute_divF_fpts(stage);

  /* Add source term (if required) */
  if (input->source)
  {
    for (auto e : elesObjs)
      e->add_source(stage, flow_time);
  }
}

void FRSolver::compute_LHS(unsigned int stage)
{
  /* Compute block diagonal components of residual Jacobian */
  PUSH_NVTX_RANGE("compute_dRdU",1);
  compute_dRdU();
  POP_NVTX_RANGE;

#ifdef _CPU
  /* Apply stage dt to LHS */
  if (!input->implicit_steady)
  {
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double dt = (input->dt_type != 2) ? e->dt(0) : e->dt(ele);
        for (unsigned int vari = 0; vari < e->nVars; vari++)
          for (unsigned int spti = 0; spti < e->nSpts; spti++)
            for (unsigned int varj = 0; varj < e->nVars; varj++)
              for (unsigned int sptj = 0; sptj < e->nSpts; sptj++)
              {
                e->LHS(ele, vari, spti, varj, sptj) *= rk_alpha(stage, stage) * dt;
                if (spti == sptj && vari == varj)
                  e->LHS(ele, vari, spti, varj, sptj) += 1;
              }
      }
  }

  /* Apply pseudo dt to LHS */
  if (input->pseudo_time)
  {
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double dtau = dtau_ratio * ((input->dtau_type != 2) ? e->dtau(0) : e->dtau(ele));
        for (unsigned int vari = 0; vari < e->nVars; vari++)
          for (unsigned int spti = 0; spti < e->nSpts; spti++)
            for (unsigned int varj = 0; varj < e->nVars; varj++)
              for (unsigned int sptj = 0; sptj < e->nSpts; sptj++)
              {
                e->LHS(ele, vari, spti, varj, sptj) *= dtau;
                if (spti == sptj && vari == varj)
                  e->LHS(ele, vari, spti, varj, sptj) += 1;
              }
      }
  }
#endif

#ifdef _GPU
  /* Apply stage dt to LHS */
  if (!input->implicit_steady)
  {
    for (auto e : elesObjs)
      apply_dt_LHS_wrapper(e->LHS_d, e->dt_d, rk_alpha(stage,stage), 
          input->dt_type, e->nSpts, e->nEles, e->nVars);
    check_error();
  }

  /* Apply pseudo dt to LHS */
  if (input->pseudo_time)
  {
    for (auto e : elesObjs)
      apply_dt_LHS_wrapper(e->LHS_d, e->dtau_d, dtau_ratio, 
          input->dtau_type, e->nSpts, e->nEles, e->nVars);
    check_error();
  }
#endif

  /* Write LHS */
  if (input->write_LHS)
    write_LHS(input->output_prefix);
}

void FRSolver::compute_dRdU()
{
  if (!input->FDA_Jacobian)
  {
    /* Apply boundary conditions for flux Jacobian */
    faces->apply_bcs_dFdU();

    /* Compute flux Jacobian at solution points */
    for (auto e : elesObjs)
      e->compute_dFdU();

    /* Compute common interface flux Jacobian */
    faces->compute_common_dFdU(0, geo.nGfpts);

#ifdef _GPU
    if (input->KPF_Jacobian)
    {
      if (input->viscous)
      {
        /* Compute common interface flux Jacobian (neighbor gradient contributions) */
        for (auto e : elesObjs)
          e->compute_KPF_dFcdU_gradN();
      }

      /* Zero out LHS */
      for (auto e : elesObjs)
        device_fill(e->LHS_d, e->LHS_d.max_size(), 0.);
    }
#endif

    /* Compute block diagonal components of flux divergence Jacobian */
    for (auto e : elesObjs)
      e->compute_local_dRdU();
  }

  else
  {
    /* Compute residual Jacobian using FDA */
    // Note: This is very expensive! Only for testing.
    std::cout << "Compute residual Jacobian using FDA..." << std::endl;
    std::vector<mdvector<double>> U0(elesObjs.size());
    for (auto e : elesObjs)
      U0[e->elesObjID] = e->U_spts;
    mdvector<double> divF;
    double eps = std::sqrt(std::numeric_limits<double>::epsilon());
    for (auto e : elesObjs)
    {
      divF = e->divF_spts;
      for (unsigned int spt = 0; spt < e->nSpts; spt++)
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            // Add eps to solution and compute divF
            e->U_spts = U0[e->elesObjID];
            e->U_spts(spt, var, ele) += eps;
            compute_residual(0);

            // Compute LHS implicit Jacobian
            for (unsigned int vari = 0; vari < e->nVars; vari++)
              for (unsigned int spti = 0; spti < e->nSpts; spti++)
                e->LHS(ele, vari, spti, var, spt) = (e->divF_spts(0, spti, vari, ele) - divF(0, spti, vari, ele)) / eps;
          }
      e->U_spts = U0[e->elesObjID];
      compute_residual(0);
    }
  }
}

void FRSolver::compute_LHS_LU()
{
#ifdef _CPU
  /* Perform LU using Eigen */
  for (auto e : elesObjs)
  {
    unsigned int N = e->nSpts * e->nVars;
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      Eigen::Map<MatrixXdRM> A(&e->LHS(ele, 0, 0, 0, 0), N, N);
      e->LU_ptrs[ele] = Eigen::PartialPivLU<Eigen::MatrixXd>(A);
    }
  }
#endif

#ifdef _GPU
  /* Perform in-place batched LU of A transpose using cuBLAS (col-maj) */
  for (auto e : elesObjs)
  {
    unsigned int N = e->nSpts * e->nVars;
    cublasDgetrfBatched_wrapper(N, e->LHS_ptrs_d.data(), N, nullptr, e->LHS_info_d.data(), e->nEles);
  }
  check_error();
#endif
}

void FRSolver::compute_LHS_inverse()
{
#ifdef _CPU
  /* Perform inversion using Eigen */
  for (auto e : elesObjs)
  {
    unsigned int N = e->nSpts * e->nVars;
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      Eigen::Map<MatrixXdRM> A(&e->LHS(ele, 0, 0, 0, 0), N, N);
      Eigen::Map<MatrixXdRM> Ainv(&e->LHSinv(ele, 0, 0, 0, 0), N, N);
      Ainv = A.inverse();
    }
  }
#endif

#ifdef _GPU
  /* Perform batched inversion on LU of A transpose using cuBLAS (col-maj) */
  for (auto e : elesObjs)
  {
    unsigned int N = e->nSpts * e->nVars;
    cublasDgetrfBatched_wrapper(N, e->LHS_ptrs_d.data(), N, nullptr, e->LHS_info_d.data(), e->nEles);
    cublasDgetriBatched_wrapper(N, (const double**) e->LHS_ptrs_d.data(), N, nullptr, 
        e->LHSinv_ptrs_d.data(), N, e->LHS_info_d.data(), e->nEles);
  }
  check_error();
#endif
}

void FRSolver::compute_LHS_SVD()
{
#ifdef _CPU
  /* Perform SVD using Eigen */
  for (auto e : elesObjs)
  {
    unsigned int N = e->nSpts * e->nVars;
    unsigned int rank = e->svd_rank;
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      /* Compute SVD */
      Eigen::Map<MatrixXdRM> A(&e->LHS(ele, 0, 0, 0, 0), N, N);
      e->SVD_ptrs[ele] = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      MatrixXdRM U = e->SVD_ptrs[ele].matrixU();
      MatrixXdRM V = e->SVD_ptrs[ele].matrixV();
      Eigen::VectorXd invS = e->SVD_ptrs[ele].singularValues();
      for (unsigned int i = 0; i < N; i++)
        invS(i) = 1.0 / invS(i);

      /* Copy Low Rank Approximation */
      Eigen::Map<MatrixXdRM> U_hat(&e->LHSU(ele, 0, 0), N, rank);
      Eigen::Map<MatrixXdRM> V_hat(&e->LHSV(ele, 0, 0), N, rank);
      for (unsigned int i = 0; i < N; i++)
        for (unsigned int r = 0; r < rank; r++)
        {
          unsigned int j = (N-rank) + r;
          U_hat(i, r) = U(i, j);
          V_hat(i, r) = V(i, j);
        }

      Eigen::Map<Eigen::VectorXd> invS_hat(&e->LHSinvS(ele, 0), rank);
      for (unsigned int r = 0; r < rank; r++)
      {
        unsigned int j = (N-rank) + r;
        invS_hat(r) = invS(j);
      }

      /* Compute diagonal of inverse diagonal matrices */
      Eigen::VectorXd AinvD(N);
      Eigen::VectorXd AinvD_hat(N);
      for (unsigned int i = 0; i < N; i++)
      {
        AinvD(i) = 0;
        for (unsigned int j = 0; j < N; j++)
          AinvD(i) += invS(j) * V(i, j) * U(i, j);

        AinvD_hat(i) = 0;
        for (unsigned int r = 0; r < rank; r++)
          AinvD_hat(i) += invS_hat(r) * V_hat(i, r) * U_hat(i, r);
      }

      /* Store inverse diagonal correction */
      Eigen::Map<Eigen::VectorXd> AinvD_corr(&e->LHSinvD(ele, 0), N);
      AinvD_corr = input->svd_omg * (AinvD - AinvD_hat);
    }
  }
#endif

#ifdef _GPU
  ThrowException("SVD solver not implemented on GPU!");
#endif
}

void FRSolver::compute_RHS(unsigned int stage, int color)
{
  /* MCGS: compute_RHS for a given color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[color];
  else
    elesObjs = this->elesObjs;

#ifdef _CPU
  for (auto e : elesObjs)
    for (unsigned int ele = 0; ele < e->nEles; ele++)
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          e->RHS(ele, var, spt) = -rk_alpha(stage,0) * e->divF_spts(0, spt, var, ele) / e->jaco_det_spts(spt, ele);
          for (unsigned int s = 1; s <= stage; s++)
            e->RHS(ele, var, spt) -= rk_alpha(stage, s) * e->divF_spts(s, spt, var, ele) / e->jaco_det_spts(spt, ele);
        }

  if (!input->implicit_steady)
  {
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double dt = (input->dt_type != 2) ? e->dt(0) : e->dt(ele);
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            e->RHS(ele, var, spt) *= dt;
      }

    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            e->RHS(ele, var, spt) -= (e->U_spts(spt, var, ele) - e->U_ini(spt, var, ele));
  }

  if (input->pseudo_time)
  {
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double dtau = dtau_ratio * ((input->dtau_type != 2) ? e->dtau(0) : e->dtau(ele));
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            e->RHS(ele, var, spt) *= dtau;
      }

    if (!input->remove_deltaU)
    {
      for (auto e : elesObjs)
        for (unsigned int ele = 0; ele < e->nEles; ele++)
          for (unsigned int var = 0; var < e->nVars; var++)
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
              e->RHS(ele, var, spt) -= (e->U_spts(spt, var, ele) - e->U_iniNM(spt, var, ele));
    }
  }
#endif

#ifdef _GPU
  /* Compute RHS in deltaU (in-place solve) */
  if (input->linear_solver == LU)
  {
    for (auto e : elesObjs)
      compute_RHS_wrapper(e->U_spts_d, e->U_iniNM_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, 
          e->dt_d, e->dtau_d, rk_alpha_d, e->deltaU_d, dtau_ratio, input->implicit_steady, 
          input->pseudo_time, input->remove_deltaU, input->dt_type, input->dtau_type, e->nSpts, 
          e->nEles, e->nVars, stage);
    check_error();
  }
  else
  {
    for (auto e : elesObjs)
      compute_RHS_wrapper(e->U_spts_d, e->U_iniNM_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, 
          e->dt_d, e->dtau_d, rk_alpha_d, e->RHS_d, dtau_ratio, input->implicit_steady, 
          input->pseudo_time, input->remove_deltaU, input->dt_type, input->dtau_type, e->nSpts, 
          e->nEles, e->nVars, stage);
    check_error();
  }
#endif
}

void FRSolver::compute_deltaU(int color)
{
  /* MCGS: compute_deltaU for a given color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[color];
  else
    elesObjs = this->elesObjs;

#ifdef _CPU
  /* Perform LU solve using Eigen */
  if (input->linear_solver == LU)
  {
    for (auto e : elesObjs)
    {
      unsigned int N = e->nSpts * e->nVars;
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        Eigen::Map<Eigen::VectorXd> x(&e->deltaU(ele, 0, 0), N);
        Eigen::Map<Eigen::VectorXd> b(&e->RHS(ele, 0, 0), N);
        x = e->LU_ptrs[ele].solve(b);
      }
    }
  }

  /* Perform matrix-vector of A inverse and RHS */
  else if (input->linear_solver == INV)
  {
    for (auto e : elesObjs)
    {
      unsigned int N = e->nSpts * e->nVars;
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        Eigen::Map<MatrixXdRM> Ainv(&e->LHSinv(ele, 0, 0, 0, 0), N, N);
        Eigen::Map<Eigen::VectorXd> x(&e->deltaU(ele, 0, 0), N);
        Eigen::Map<Eigen::VectorXd> b(&e->RHS(ele, 0, 0), N);
        x = Ainv*b;
      }
    }
  }

  /* Perform SVD solve */
  else if (input->linear_solver == SVD)
  {
    for (auto e : elesObjs)
    {
      unsigned int N = e->nSpts * e->nVars;
      unsigned int rank = e->svd_rank;
      mdvector<double> deltaU_temp({rank});
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        // deltaU = LHSinv_diag * RHS
        for (unsigned int vari = 0; vari < e->nVars; vari++)
          for (unsigned int spti = 0; spti < e->nSpts; spti++)
          {
            unsigned int i = e->nSpts * vari + spti;
            e->deltaU(ele, vari, spti) = e->LHSinvD(ele, i) * e->RHS(ele, vari, spti);
          }

        // deltaU_temp = U transpose * RHS  / s
        for (unsigned int r = 0; r < rank; r++)
        {
          double val = 0;
          for (unsigned int varj = 0; varj < e->nVars; varj++)
            for (unsigned int sptj = 0; sptj < e->nSpts; sptj++)
            {
              unsigned int j = e->nSpts * varj + sptj;
              val += e->LHSU(ele, j, r) * e->RHS(ele, varj, sptj);
            }
          deltaU_temp(r) = val * e->LHSinvS(ele, r);
        }

        // deltaU += V * deltaU_temp
        for (unsigned int vari = 0; vari < e->nVars; vari++)
          for (unsigned int spti = 0; spti < e->nSpts; spti++)
          {
            unsigned int i = e->nSpts * vari + spti;
            double val = 0;
            for (unsigned int r = 0; r < rank; r++)
              val += e->LHSV(ele, i, r) * deltaU_temp(r);
            e->deltaU(ele, vari, spti) += val;
          }
      }
    }
  }

  else
  {
    ThrowException("Linear solver not recognized!");
  }
#endif

#ifdef _GPU
  /* Perform in-place batched LU solve of A using cuBLAS */
  if (input->linear_solver == LU)
  {
    for (auto e : elesObjs)
    {
      int info;
      unsigned int N = e->nSpts * e->nVars;
      cublasDgetrsBatched_wrapper(N, 1, (const double**) e->LHS_ptrs_d.data(), N, nullptr, 
          e->deltaU_ptrs_d.data(), N, &info, e->nEles);

      if (info) ThrowException("cuBLAS batch LU solve failed!");
    }
  }

  /* Perform batched matrix-vector of A inverse and RHS */
  else if (input->linear_solver == INV)
  {
    for (auto e : elesObjs)
    {
      unsigned int N = e->nSpts * e->nVars;
      DgemvBatched_wrapper(N, N, 1.0, (const double**) e->LHSinv_ptrs_d.data(), N, 
          (const double**) e->RHS_ptrs_d.data(), 1, 0.0, e->deltaU_ptrs_d.data(), 1, e->nEles);
    }
  }

  else
  {
    ThrowException("Linear solver not recognized!");
  }
  check_error();
#endif
}

void FRSolver::compute_U(int color)
{
  /* MCGS: compute_U for a given color */
  std::vector<std::shared_ptr<Elements>> elesObjs;
  if (color > -1)
    elesObjs = elesObjsBC[color];
  else
    elesObjs = this->elesObjs;

#ifdef _CPU
  for (auto e : elesObjs)
    for (unsigned int ele = 0; ele < e->nEles; ele++)
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
          e->U_spts(spt, var, ele) += e->deltaU(ele, var, spt);
#endif

#ifdef _GPU
  for (auto e : elesObjs)
    compute_U_wrapper(e->U_spts_d, e->deltaU_d, e->nSpts, e->nEles, e->nVars);
  check_error();
#endif
}

void FRSolver::initialize_U()
{
  // initialization moved into elements
}

void FRSolver::setup_views()
{
  /* Setup face view of element solution data struture */
  // TODO: Might not want to allocate all these at once. Turn this into a function maybe?
  mdvector<double*> U_base_ptrs({2 * geo.nGfpts});
  mdvector<double*> U_ldg_base_ptrs({2 * geo.nGfpts});
  mdvector<double*> Fcomm_base_ptrs({2 * geo.nGfpts});
  mdvector<unsigned int> U_strides({2 * geo.nGfpts});
  mdvector<double*> Ucomm_base_ptrs;
  mdvector<double*> dU_base_ptrs;
  mdvector<unsigned int> dU_strides;

  if (input->viscous)
  {
    Ucomm_base_ptrs.assign({2 * geo.nGfpts});
    dU_base_ptrs.assign({2 * geo.nGfpts});
    dU_strides.assign({2, 2 * geo.nGfpts});
  }

#ifdef _GPU
  mdvector<double*> U_base_ptrs_d({2 * geo.nGfpts});
  mdvector<double*> U_ldg_base_ptrs_d({2 * geo.nGfpts});
  mdvector<double*> Fcomm_base_ptrs_d({2 * geo.nGfpts});
  mdvector<unsigned int> U_strides_d({2 * geo.nGfpts});
  mdvector<double*> Ucomm_base_ptrs_d;
  mdvector<double*> dU_base_ptrs_d;
  mdvector<unsigned int> dU_strides_d;

  if (input->viscous)
  {
    Ucomm_base_ptrs_d.assign({2 * geo.nGfpts});
    dU_base_ptrs_d.assign({2 * geo.nGfpts});
    dU_strides_d.assign({2, 2 * geo.nGfpts});
  }
#endif

  /* Setup face views of element data for implicit */
  mdvector<double*> dFcdU_base_ptrs;
  mdvector<unsigned int> dFcdU_strides;
  mdvector<double*> dUcdU_base_ptrs, dFcddU_base_ptrs;
  mdvector<unsigned int> dFcddU_strides;
  if (input->implicit_method)
  {
    dFcdU_base_ptrs.assign({2 * geo.nGfpts});
    dFcdU_strides.assign({2, 2 * geo.nGfpts});
    if (input->viscous)
    {
      dUcdU_base_ptrs.assign({2 * geo.nGfpts});
      dFcddU_base_ptrs.assign({2 * geo.nGfpts});
      dFcddU_strides.assign({4, 2 * geo.nGfpts});
    }
  }

#ifdef _GPU
  mdvector<double*> dFcdU_base_ptrs_d;
  mdvector<unsigned int> dFcdU_strides_d;
  mdvector<double*> dUcdU_base_ptrs_d, dFcddU_base_ptrs_d;
  mdvector<unsigned int> dFcddU_strides_d;
  if (input->implicit_method)
  {
    dFcdU_base_ptrs_d.assign({2 * geo.nGfpts});
    dFcdU_strides_d.assign({2, 2 * geo.nGfpts});
    if (input->viscous)
    {
      dUcdU_base_ptrs_d.assign({2 * geo.nGfpts});
      dFcddU_base_ptrs_d.assign({2 * geo.nGfpts});
      dFcddU_strides_d.assign({4, 2 * geo.nGfpts});
    }
  }
#endif

  /* Set pointers for internal faces */
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      unsigned int eleBT = ele + e->startEle;
      for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
      {
        int gfpt = geo.fpt2gfptBT[e->etype](fpt,eleBT);
        int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,eleBT);

        U_base_ptrs(gfpt + slot * geo.nGfpts) = &e->U_fpts(fpt, 0, ele);
        U_ldg_base_ptrs(gfpt + slot * geo.nGfpts) = &e->U_fpts(fpt, 0, ele);
        U_strides(gfpt + slot * geo.nGfpts) = e->U_fpts.get_stride(1);

        Fcomm_base_ptrs(gfpt + slot * geo.nGfpts) = &e->Fcomm(fpt, 0, ele);

        if (input->viscous) Ucomm_base_ptrs(gfpt + slot * geo.nGfpts) = &e->Ucomm(fpt, 0, ele);
#ifdef _GPU
        U_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_ptr(fpt, 0, ele);
        U_ldg_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_ptr(fpt, 0, ele);
        U_strides_d(gfpt + slot * geo.nGfpts) = e->U_fpts_d.get_stride(1);

        Fcomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->Fcomm_d.get_ptr(fpt, 0, ele);

        if (input->viscous) Ucomm_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->Ucomm_d.get_ptr(fpt, 0, ele);
#endif

        if (input->viscous)
        {
          dU_base_ptrs(gfpt + slot * geo.nGfpts) = &e->dU_fpts(0, fpt, 0, ele);
          dU_strides(0, gfpt + slot * geo.nGfpts) = e->dU_fpts.get_stride(1);
          dU_strides(1, gfpt + slot * geo.nGfpts) = e->dU_fpts.get_stride(3);

#ifdef _GPU
          dU_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->dU_fpts_d.get_ptr(0, fpt, 0, ele);
          dU_strides_d(0, gfpt + slot * geo.nGfpts) = e->dU_fpts_d.get_stride(1);
          dU_strides_d(1, gfpt + slot * geo.nGfpts) = e->dU_fpts_d.get_stride(3);
#endif
        }

        /* Set pointers for internal faces for implicit */
        if (input->implicit_method)
        {
          dFcdU_base_ptrs(gfpt + slot * geo.nGfpts) = &e->dFcdU(fpt, 0, 0, ele);
          dFcdU_strides(0, gfpt + slot * geo.nGfpts) = e->dFcdU.get_stride(1);
          dFcdU_strides(1, gfpt + slot * geo.nGfpts) = e->dFcdU.get_stride(2);
#ifdef _GPU
          dFcdU_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->dFcdU_d.get_ptr(fpt, 0, 0, ele);
          dFcdU_strides_d(0, gfpt + slot * geo.nGfpts) = e->dFcdU_d.get_stride(1);
          dFcdU_strides_d(1, gfpt + slot * geo.nGfpts) = e->dFcdU_d.get_stride(2);
#endif

          if (input->viscous)
          {
            dUcdU_base_ptrs(gfpt + slot * geo.nGfpts) = &e->dUcdU(fpt, 0, 0, ele);
            dFcddU_base_ptrs(gfpt + slot * geo.nGfpts) = &e->dFcddU(0, 0, fpt, 0, 0, ele);
            dFcddU_strides(0, gfpt + slot * geo.nGfpts) = e->dFcddU.get_stride(1);
            dFcddU_strides(1, gfpt + slot * geo.nGfpts) = e->dFcddU.get_stride(2);
            dFcddU_strides(2, gfpt + slot * geo.nGfpts) = e->dFcddU.get_stride(4);
            dFcddU_strides(3, gfpt + slot * geo.nGfpts) = e->dFcddU.get_stride(5);
#ifdef _GPU
            dUcdU_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->dUcdU_d.get_ptr(fpt, 0, 0, ele);
            dFcddU_base_ptrs_d(gfpt + slot * geo.nGfpts) = e->dFcddU_d.get_ptr(0, 0, fpt, 0, 0, ele);
            dFcddU_strides_d(0, gfpt + slot * geo.nGfpts) = e->dFcddU_d.get_stride(1);
            dFcddU_strides_d(1, gfpt + slot * geo.nGfpts) = e->dFcddU_d.get_stride(2);
            dFcddU_strides_d(2, gfpt + slot * geo.nGfpts) = e->dFcddU_d.get_stride(4);
            dFcddU_strides_d(3, gfpt + slot * geo.nGfpts) = e->dFcddU_d.get_stride(5);
#endif
          }
        }
      }
    }
  }

  /* Set pointers for remaining faces (includes boundary and MPI faces) */
  unsigned int i = 0;
  for (unsigned int gfpt = geo.nGfpts_int; gfpt < geo.nGfpts; gfpt++)
  {
    U_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd(0, i);
    
    if (gfpt < geo.nGfpts_int + geo.nGfpts_bnd)
      U_ldg_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd_ldg(0, i);
    else
      U_ldg_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->U_bnd(0, i); // point U_ldg to correct MPI data;

    U_strides(gfpt + 1 * geo.nGfpts) = faces->U_bnd.get_stride(1);

    Fcomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Fcomm_bnd(0, i);

    if (input->viscous) Ucomm_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->Ucomm_bnd(0, i);

#ifdef _GPU
    U_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_ptr(0, i);
    if (gfpt < geo.nGfpts_int + geo.nGfpts_bnd)
      U_ldg_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_ldg_d.get_ptr(0, i);
    else
      U_ldg_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_ptr(0, i); // point U_ldg to correct MPI data;

    U_strides_d(gfpt + 1 * geo.nGfpts) = faces->U_bnd_d.get_stride(1);

    Fcomm_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->Fcomm_bnd_d.get_ptr(0, i);

    if (input->viscous) Ucomm_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->Ucomm_bnd_d.get_ptr(0, i);
#endif

    if (input->viscous)
    {
      dU_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->dU_bnd(0, 0, i);
      dU_strides(0, gfpt + 1 * geo.nGfpts) = faces->dU_bnd.get_stride(1);
      dU_strides(1, gfpt + 1 * geo.nGfpts) = faces->dU_bnd.get_stride(2);
#ifdef _GPU
      dU_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->dU_bnd_d.get_ptr(0, 0, i);
      dU_strides_d(0, gfpt + 1 * geo.nGfpts) = faces->dU_bnd_d.get_stride(1);
      dU_strides_d(1, gfpt + 1 * geo.nGfpts) = faces->dU_bnd_d.get_stride(2);
#endif
    }

    /* Set pointers for remaining faces for implicit */
    if (input->implicit_method)
    {
      dFcdU_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->dFcdU_bnd(0, 0, i);
      dFcdU_strides(0, gfpt + 1 * geo.nGfpts) = faces->dFcdU_bnd.get_stride(1);
      dFcdU_strides(1, gfpt + 1 * geo.nGfpts) = faces->dFcdU_bnd.get_stride(2);
#ifdef _GPU
      dFcdU_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->dFcdU_bnd_d.get_ptr(0, 0, i);
      dFcdU_strides_d(0, gfpt + 1 * geo.nGfpts) = faces->dFcdU_bnd_d.get_stride(1);
      dFcdU_strides_d(1, gfpt + 1 * geo.nGfpts) = faces->dFcdU_bnd_d.get_stride(2);
#endif

      if (input->viscous)
      {
        dUcdU_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->dUcdU_bnd(0, 0, i);
        dFcddU_base_ptrs(gfpt + 1 * geo.nGfpts) = &faces->dFcddU_bnd(0, 0, 0, 0, i);
        dFcddU_strides(0, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd.get_stride(1);
        dFcddU_strides(1, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd.get_stride(2);
        dFcddU_strides(2, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd.get_stride(3);
        dFcddU_strides(3, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd.get_stride(4);
#ifdef _GPU
        dUcdU_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->dUcdU_bnd_d.get_ptr(0, 0, i);
        dFcddU_base_ptrs_d(gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd_d.get_ptr(0, 0, 0, 0, i);
        dFcddU_strides_d(0, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd_d.get_stride(1);
        dFcddU_strides_d(1, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd_d.get_stride(2);
        dFcddU_strides_d(2, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd_d.get_stride(3);
        dFcddU_strides_d(3, gfpt + 1 * geo.nGfpts) = faces->dFcddU_bnd_d.get_stride(4);
#endif
      }
    }

    i++;
  }

  /* Create views of element data for faces */
  faces->U.assign(U_base_ptrs, U_strides, geo.nGfpts);
  faces->U_ldg.assign(U_ldg_base_ptrs, U_strides, geo.nGfpts);
  faces->Fcomm.assign(Fcomm_base_ptrs, U_strides, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm.assign(Ucomm_base_ptrs, U_strides, geo.nGfpts);
    faces->dU.assign(dU_base_ptrs, dU_strides, geo.nGfpts);
  }

#ifdef _GPU
  faces->U_d.assign(U_base_ptrs_d, U_strides_d, geo.nGfpts);
  faces->U_ldg_d.assign(U_ldg_base_ptrs_d, U_strides_d, geo.nGfpts);
  faces->Fcomm_d.assign(Fcomm_base_ptrs_d, U_strides_d, geo.nGfpts);
  if (input->viscous)
  {
    faces->Ucomm_d.assign(Ucomm_base_ptrs_d, U_strides_d, geo.nGfpts);
    faces->dU_d.assign(dU_base_ptrs_d, dU_strides_d, geo.nGfpts);
  }
#endif

  /* Create views of element data for implicit */
  if (input->implicit_method)
  {
    faces->dFcdU.assign(dFcdU_base_ptrs, dFcdU_strides, geo.nGfpts);
    if (input->viscous)
    {
      faces->dUcdU.assign(dUcdU_base_ptrs, dFcdU_strides, geo.nGfpts);
      faces->dFcddU.assign(dFcddU_base_ptrs, dFcddU_strides, geo.nGfpts);
    }

#ifdef _GPU
    faces->dFcdU_d.assign(dFcdU_base_ptrs_d, dFcdU_strides_d, geo.nGfpts);
    if (input->viscous)
    {
      faces->dUcdU_d.assign(dUcdU_base_ptrs_d, dFcdU_strides_d, geo.nGfpts);
      faces->dFcddU_d.assign(dFcddU_base_ptrs_d, dFcddU_strides_d, geo.nGfpts);
    }
#endif
  }
}

/* Note: Source term in update() is used primarily for multigrid. To add a true source term, define
 * a source term in funcs.cpp and set source input flag to 1. */
#ifdef _CPU
void FRSolver::update(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif 
#ifdef _GPU
void FRSolver::update(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  // Determine whether to turn on implicit output
  if (input->implicit_method)
  {
    if (input->implicit_steady)
      report_NMconv_freq = input->report_NMconv_freq;
    else
    {
      unsigned int iter = (current_iter - restart_iter)+1;
      if (input->report_freq != 0 && (iter%input->report_freq == 0 || iter == input->n_steps || iter == 1))
        report_NMconv_freq = input->report_NMconv_freq;
      else
        report_NMconv_freq = 0;
    }
  }

  prev_time = flow_time;

  // Update grid to start of time step (if not already done so at previous step)
  move(flow_time, true);

  if (input->adapt_dt)
  {
    step_adaptive(sourceBT);
  }
  else
  {
    if (input->dt_scheme == "Steady")
      step_Steady(0, current_iter+1, sourceBT);
    else if (input->dt_scheme == "DIRK34" || input->dt_scheme == "ESDIRK43" || input->dt_scheme == "ESDIRK64")
      step_DIRK(sourceBT);
    else if (input->dt_scheme == "RK54")
      step_LSRK(sourceBT);
    else
      step_RK(sourceBT);

    flow_time = prev_time + elesObjs[0]->dt(0);
  }

  current_iter++;
}


#ifdef _CPU
void FRSolver::step_adaptive(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_adaptive(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  if (input->dt_scheme == "RK54")
    step_LSRK(sourceBT);
  else if (input->dt_scheme == "ESDIRK43" || input->dt_scheme == "ESDIRK64")
    step_DIRK(sourceBT);
  else
    ThrowException("Embedded pairs not implemented for this dt_scheme!");

  // Calculate error (infinity norm of RK error) and scaling factor for dt
  double max_err = 0;
#ifdef _CPU
  for (auto e : elesObjs)
  {
    for (uint spt = 0; spt < e->nSpts; spt++)
    {
      for (uint n = 0; n < e->nVars; n++)
      {
#pragma omp parallel for simd reduction(max:max_err)
        for (uint ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
          double err = std::abs(e->rk_err(spt, n, ele)) /
              (input->atol + input->rtol * std::max( std::abs(e->U_spts(spt, n, ele)), std::abs(e->U_ini(spt, n, ele)) ));
          max_err = std::max(max_err, err);
        }
      }
    }
  }


#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &max_err, 1, MPI_DOUBLE, MPI_MAX, worldComm);
#endif

  // Determine the time step scaling factor and the new time step
  double fac = pow(max_err, -expa) * pow(prev_err, expb);
  fac = std::min(input->maxfac, std::max(input->minfac, input->sfact*fac));

  for (auto e : elesObjs)
  {
    e->dt(0) *= fac;
    e->dt(0) = std::min(e->dt(0), input->max_dt);
  }
#endif

#ifdef _GPU
  max_err = 0.0;
  for (auto e : elesObjs)
  {
    double err = get_rk_error_wrapper(e->U_spts_d, e->U_ini_d, e->rk_err_d, e->nSpts, e->nEles,
        e->nVars, input->atol, input->rtol, worldComm, input->overset, geo.iblank_cell_d.data());

    err = std::isnan(err) ? INFINITY : err; // convert NaNs to "large" error

    max_err = std::max(max_err, err);
  }

  for (auto e : elesObjs)
  {
    set_adaptive_dt_wrapper(e->dt_d, e->dt(0), expa, expb, input->minfac, input->maxfac, 
        input->sfact, input->max_dt, max_err, prev_err);
  }
#endif

  if (elesObjs[0]->dt(0) < 1e-14)
    ThrowException("dt approaching 0 - quitting simulation");

  if (max_err < 1.)
  {
    // Accept the time step and continue on
    prev_err = max_err;
  }
  else
  {
    // Reject step - reset solution back to beginning of time step
    flow_time = prev_time;
#ifdef _CPU
    for (auto e : elesObjs)
      e->U_spts = e->U_ini;
#endif

    if (input->motion_type == RIGID_BODY)
    {
      geo.vel_cg = v_ini;
      geo.x_cg = x_ini;
      geo.q = q_ini;
      geo.qdot = qdot_ini;
    }

#ifdef _GPU
    for (auto e : elesObjs)
      device_copy(e->U_spts_d, e->U_ini_d, e->U_ini_d.max_size());

    if (input->motion_type == RIGID_BODY)
    {
      device_copy(geo.x_cg_d, x_ini_d, x_ini_d.max_size());
      device_copy(geo.vel_cg_d, v_ini_d, v_ini_d.max_size());
    }
#endif

    // Try again with new dt
    step_adaptive(sourceBT);
  }
}

#ifdef _CPU
void FRSolver::step_RK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_RK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
#ifdef _CPU
  if (input->nStages > 1)
  {
    for (auto e : elesObjs)
      e->U_ini = e->U_spts;
  }
#endif

#ifdef _GPU
  if (input->nStages > 1)
  {
    for (auto e : elesObjs)
      device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    check_error();
  }
#endif

  unsigned int nSteps = (input->dt_scheme == "RKj") ? input->nStages : input->nStages - 1;

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < nSteps; stage++)
  {
    flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

    move(flow_time);

    compute_residual(stage);

    rigid_body_update(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
            }
          }
        }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
    unsigned int last_stage = (input->dt_scheme == "RKj") ? input->nStages + 1 : input->nStages;

    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                          rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                          input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
    }
    check_error();
#endif
  }

  /* Final stage combining residuals for full Butcher table style RK timestepping*/
  if (input->dt_scheme != "RKj")
  {
    flow_time = prev_time + rk_c(input->nStages-1) * elesObjs[0]->dt(0);

    move(flow_time);

    compute_residual(input->nStages-1);

    rigid_body_update(input->nStages-1);

    if (input->nStages > 1)
    {
#ifdef _CPU
      for (auto e : elesObjs)
        e->U_spts = e->U_ini;
#endif
#ifdef _GPU
      for (auto e : elesObjs)
        device_copy(e->U_spts_d, e->U_ini_d, e->U_spts_d.max_size());
#endif
    }
    else if (input->dt_type != 0)
    {
      compute_element_dt();
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      for (unsigned int stage = 0; stage < input->nStages; stage++)
      {
        if (!sourceBT.count(e->etype))
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(stage, spt, n, ele);
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(stage, spt, n, ele);
                }
              }
            }
          }
        }
        else
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt,ele) *
                      (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                          rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                          input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
    }

    check_error();
#endif
  }

}

#ifdef _CPU
void FRSolver::step_LSRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_LSRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  /* NOTE: this implementation is not the 'true' low-storage implementation
   * since we are using an additional array 'U_til' instead of swapping
   * pointers at each stage */

  // Copy current solution into "U_ini" ['rold' in PyFR]
#ifdef _CPU
  for (auto e : elesObjs)
  {
    e->U_ini = e->U_spts;
    e->U_til = e->U_spts;
  }
  if (input->adapt_dt)
    for (auto e : elesObjs)
      e->rk_err.fill(0.0);
#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    device_copy(e->U_til_d, e->U_spts_d, e->U_spts_d.max_size());
    
    // Get current delta t [dt(0)] (updated on GPU)
    copy_from_device(e->dt.data(), e->dt_d.data(), 1);
  }

  if (input->adapt_dt)
    for (auto e : elesObjs)
      device_fill(e->rk_err_d, e->rk_err_d.max_size());

  check_error();
#endif

  if (input->motion_type == RIGID_BODY)
  {
    x_ini = geo.x_cg;        x_til = geo.x_cg;
    v_ini = geo.vel_cg;      v_til = geo.vel_cg;
    q_ini = geo.q;           q_til = geo.q;
    qdot_ini = geo.qdot;     qdot_til = geo.qdot;
  }

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  for (unsigned int stage = 0; stage < input->nStages; stage++)
  {
    PUSH_NVTX_RANGE("ONE_STAGE",6);
    flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

    move(flow_time); // Set grid to current evaluation time

    compute_residual(0);

    rigid_body_update(stage);

    double ai = (stage < rk_alpha.size()) ? rk_alpha(stage) : 0;
    double bi = rk_beta(stage);
    double bhi = rk_bhat(stage);

#ifdef _CPU
    // Update Error
    if (input->adapt_dt)
    {
      for (auto e : elesObjs)
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              e->rk_err(spt, n, ele) -= (bi - bhi) * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);
            }
          }
        }
      }
    }

    // Update solution registers
    for (auto e : elesObjs)
    {
      if (stage < input->nStages - 1)
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              e->U_spts(spt, n, ele) = e->U_til(spt, n, ele) - ai * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);

              e->U_til(spt, n, ele) = e->U_spts(spt, n, ele) - (bi - ai) * e->dt(0) /
                  e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);
            }
          }
        }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              e->U_spts(spt, n, ele) = e->U_til(spt, n, ele) - bi * e->dt(0) /
                  e->jaco_det_spts(spt, ele) * e->divF_spts(0, spt, n, ele);
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        LSRK_update_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d, e->divF_spts_d,
            e->jaco_det_spts_d, e->dt(0), ai, bi, bhi, e->nSpts, e->nEles,
            e->nVars, stage, input->nStages, input->adapt_dt, input->overset,
            geo.iblank_cell_d.data());
      }
      else
      {
        LSRK_update_source_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d,
            e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt(0), ai, bi, bhi,
            e->nSpts, e->nEles, e->nVars, stage, input->nStages, input->adapt_dt,
            input->overset, geo.iblank_cell_d.data());
      }
    }
    check_error();
#endif
    POP_NVTX_RANGE;
  }

  flow_time = prev_time + elesObjs[0]->dt(0);
}

#ifdef _CPU
void FRSolver::step_Steady(unsigned int stage, unsigned int iterNM, const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_Steady(unsigned int stage, unsigned int iterNM, const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  if (input->pseudo_time)
  {
    /* Store initial U */
    if (!input->remove_deltaU)
    {
#ifdef _CPU
      for (auto e : elesObjs)
        e->U_iniNM = e->U_spts;
#endif
#ifdef _GPU
      for (auto e : elesObjs)
        device_copy(e->U_iniNM_d, e->U_spts_d, e->U_spts_d.max_size());
      check_error();
#endif
    }

    /* Adapt dtau_ratio */
    if (input->adapt_dtau)
    {
      if (iterNM == 1)
        dtau_ratio = 1.0;
      else
      {
        dtau_ratio *= (1.0 + input->dtau_growth_rate);
        if (dtau_ratio > input->dtau_ratio_max)
          dtau_ratio = input->dtau_ratio_max;
      }
    }
  }

  /* Compute Jacobian */
  if (!input->freeze_Jacobian || stage == startStage || iterNM > 1)
  {
    PUSH_NVTX_RANGE("compute_Jacobian",0);

    /* Compute residual everywhere before computing Jacobian */
    compute_residual(stage);

    /* Estimate CFL based timestep */
    if (input->dt_type != 0 && !input->implicit_steady && stage == startStage && iterNM == 1)
      compute_element_dt();

    /* Estimate CFL based pseudo timestep */
    if (input->pseudo_time && input->dtau_type != 0)
      compute_element_dt(input->pseudo_time);

    /* Compute LHS */
    compute_LHS(stage);

    /* Setup linear solver */
    if (input->linear_solver == LU)
      compute_LHS_LU();
    else if (input->linear_solver == INV)
      compute_LHS_inverse();
    else if (input->linear_solver == SVD)
      compute_LHS_SVD();
    else
      ThrowException("Linear solver not recognized!");

    POP_NVTX_RANGE;
  }

  /* Set flow time of current stage */
  if (!input->implicit_steady && iterNM == 1)
    flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

  /* Block Iterative Method */
  PUSH_NVTX_RANGE("Block_iter_method",1);
  for (unsigned int iterBM = 1; (iterBM <= input->iterBM_max) && (res_max > input->res_tol); iterBM++)
  {
    /* Sweep through colors and backsweep */
    for (unsigned int counter = 0; counter < nCounter; counter++)
    {
      /* Set color */
      int color;
      if (input->iterative_method == JAC)
        color = -1;
      else if (input->iterative_method == MCGS)
      {
        color = counter;
        if (color >= geo.nColors)
          color = 2*geo.nColors-1 - counter;
      }
      else
        ThrowException("Iterative method not recognized!");

      /* Compute residual */
      compute_residual(stage, color);
      prev_color = color;

      /* Compute RHS */
      compute_RHS(stage, color);

      /* Write RHS */
      if (input->write_RHS)
        write_RHS(input->output_prefix);

      /* Solve system for deltaU */
      compute_deltaU(color);

      /* Add deltaU to solution */
      compute_U(color);
    }

    /* Write residual to convergence file */
    if (report_NMconv_freq != 0 && (iterNM%report_NMconv_freq == 0 || iterNM == input->iterNM_max || iterNM == 1) && 
      (iterBM%input->report_BMconv_freq == 0 || iterBM == input->iterBM_max || iterNM*iterBM == 1))
      report_RHS(stage, iterNM, iterBM);
  }
  POP_NVTX_RANGE;
}

#ifdef _CPU
void FRSolver::step_DIRK(const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_DIRK(const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
#ifdef _CPU
  for (auto e : elesObjs)
    e->U_ini = e->U_spts;

  if (input->adapt_dt)
    for (auto e : elesObjs)
      e->rk_err.fill(0.0);
#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    copy_from_device(e->dt.data(), e->dt_d.data(), 1);
  }

  if (input->adapt_dt)
    for (auto e : elesObjs)
      device_fill(e->rk_err_d, e->rk_err_d.max_size());
  check_error();
#endif

  /* ESDIRK: Explicit first stage */
  if (input->dt_scheme == "ESDIRK43" || input->dt_scheme == "ESDIRK64")
    compute_residual(0);

  /* Main stage loop */
  for (unsigned int stage = startStage; stage < input->nStages; stage++)
  {
    /* Newton's Method */
    for (unsigned int iterNM = 1; (iterNM <= input->iterNM_max) && (res_max > input->res_tol); iterNM++)
      step_Steady(stage, iterNM, sourceBT);
  }

#ifdef _CPU
  /* Update error */
  if (input->adapt_dt)
    for (unsigned int stage = 0; stage < input->nStages; stage++)
      for (auto e : elesObjs)
        for (unsigned int ele = 0; ele < e->nEles; ele++)
          for (unsigned int var = 0; var < e->nVars; var++)
            for (unsigned int spt = 0; spt < e->nSpts; spt++)
              e->rk_err(spt, var, ele) -= (rk_beta(stage) - rk_bhat(stage)) * e->dt(0) * 
                e->divF_spts(stage, spt, var, ele) / e->jaco_det_spts(spt, ele);

  for (auto e : elesObjs)
    e->U_spts = e->U_ini;

  /* Update solution */
  for (unsigned int stage = 0; stage < input->nStages; stage++)
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            if (input->dt_type != 2)
              e->U_spts(spt, var, ele) -= rk_beta(stage) * e->dt(0) * e->divF_spts(stage, spt, var, ele) / e->jaco_det_spts(spt, ele);
            else
              e->U_spts(spt, var, ele) -= rk_beta(stage) * e->dt(ele) * e->divF_spts(stage, spt, var, ele) / e->jaco_det_spts(spt, ele);
          }
#endif

#ifdef _GPU
  if (input->adapt_dt)
    for (auto e : elesObjs)
      RK_error_update_wrapper(e->rk_err_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d, rk_beta_d, 
          rk_bhat_d, e->nSpts, e->nEles, e->nVars, e->nDims, input->equation, input->nStages);
  
  for (auto e : elesObjs)
    device_copy(e->U_spts_d, e->U_ini_d, e->U_spts_d.max_size());

  for (auto e : elesObjs)
    DIRK_update_wrapper(e->U_spts_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d, rk_beta_d, input->dt_type, 
        e->nSpts, e->nEles, e->nVars, e->nDims, input->equation, input->nStages);

  check_error();
#endif

  flow_time = prev_time + elesObjs[0]->dt(0);
}

#ifdef _CPU
void FRSolver::step_RK_stage(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_RK_stage(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
#ifdef _CPU
  if (input->nStages > 1 && stage == 0)
  {
    for (auto e : elesObjs)
      e->U_ini = e->U_spts;
  }
#endif

#ifdef _GPU
  if (input->nStages > 1 && stage == 0)
  {
    for (auto e : elesObjs)
      device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    check_error();
  }
#endif

  unsigned int nSteps = (input->dt_scheme == "RKj") ? input->nStages : input->nStages - 1;

  if (stage < nSteps)
  {
    /* Main stage loop. Complete for Jameson-style RK timestepping */
    flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

    move(flow_time);

    compute_residual(stage);

    rigid_body_update(stage);

    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
            }
          }
        }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
    unsigned int last_stage = (input->dt_scheme == "RKj") ? input->nStages + 1 : input->nStages;

    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                          rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                          input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
    }
    check_error();
#endif
  }

  /* Final stage combining residuals for full Butcher table style RK timestepping*/
  if (stage == nSteps && input->dt_scheme != "RKj")
  {
    flow_time = prev_time + rk_c(input->nStages-1) * elesObjs[0]->dt(0);

    move(flow_time);

    compute_residual(input->nStages-1);

    rigid_body_update(input->nStages-1);

    if (input->nStages > 1)
    {
#ifdef _CPU
      for (auto e : elesObjs)
        e->U_spts = e->U_ini;
#endif
#ifdef _GPU
      for (auto e : elesObjs)
        device_copy(e->U_spts_d, e->U_ini_d, e->U_spts_d.max_size());
#endif
    }
    else if (input->dt_type != 0)
    {
      compute_element_dt();
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      for (unsigned int stage = 0; stage < input->nStages; stage++)
      {
        if (!sourceBT.count(e->etype))
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(stage, spt, n, ele);
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(stage, spt, n, ele);
                }
              }
            }
          }
        }
        else
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(stage) * e->dt(ele) / e->jaco_det_spts(spt,ele) *
                      (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                          rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                          input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
    }

    check_error();
#endif
  }

}

void FRSolver::step_RK_stage_start(int stage)
{
  if (input->nStages > 1 && stage == 0)
  {
#ifdef _CPU
    for (auto e : elesObjs)
      e->U_ini = e->U_spts;
#endif

#ifdef _GPU
    for (auto e : elesObjs)
      device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
    check_error();
#endif
  }

  /* Main stage loop. Complete for Jameson-style RK timestepping */
  flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

  compute_residual_start(stage);
}

void FRSolver::step_RK_stage_mid(int stage)
{
  compute_residual_mid(stage);
}

#ifdef _CPU
void FRSolver::step_RK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_RK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  unsigned int nSteps = (input->dt_scheme == "RKj") ? input->nStages : input->nStages - 1;

  flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

  compute_residual_finish(stage);

  rigid_body_update(stage);

  if (stage < nSteps)
  {
    /* If in first stage, compute stable timestep */
    if (stage == 0)
    {
      // TODO: Revisit this as it is kind of expensive.
      if (input->dt_type != 0)
      {
        compute_element_dt();
      }
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * e->divF_spts(stage, spt, n, ele);
              }
            }
          }
        }
      }
      else
      {
#pragma omp parallel for collapse(2)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          for (unsigned int n = 0; n < e->nVars; n++)
          {
#pragma omp simd
            for (unsigned int ele = 0; ele < e->nEles; ele++)
            {
              if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

              if (input->dt_type != 2)
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(0) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
              else
              {
                e->U_spts(spt, n, ele) = e->U_ini(spt, n, ele) - rk_alpha(stage) * e->dt(ele) /
                    e->jaco_det_spts(spt, ele) * (e->divF_spts(stage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    /* Increase last_stage if using RKj timestepping to bypass final stage branch in kernel. */
    unsigned int last_stage = (input->dt_scheme == "RKj") ? input->nStages + 1 : input->nStages;

    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
                                 rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_ini_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
                                 rk_alpha_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
                                 input->equation, stage, last_stage, false, input->overset, geo.iblank_cell_d.data());
      }
    }
    check_error();
#endif
  }

  /* Final stage combining residuals for full Butcher table style RK timestepping*/
  else if (stage == nSteps && input->dt_scheme != "RKj")
  {
    if (input->nStages > 1)
    {
#ifdef _CPU
      for (auto e : elesObjs)
        e->U_spts = e->U_ini;
#endif
#ifdef _GPU
      for (auto e : elesObjs)
        device_copy(e->U_spts_d, e->U_ini_d, e->U_spts_d.max_size());
#endif
    }
    else if (input->dt_type != 0)
    {
      compute_element_dt();
    }

#ifdef _CPU
    for (auto e : elesObjs)
    {
      for (unsigned int rkstage = 0; rkstage < input->nStages; rkstage++)
      {
        if (!sourceBT.count(e->etype))
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(rkstage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(rkstage, spt, n, ele);
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(rkstage) * e->dt(ele) / e->jaco_det_spts(spt, ele) *
                      e->divF_spts(rkstage, spt, n, ele);
                }
              }
            }
          }
        }
        else
        {
#pragma omp parallel for collapse(2)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
          {
            for (unsigned int n = 0; n < e->nVars; n++)
            {
#pragma omp simd
              for (unsigned int ele = 0; ele < e->nEles; ele++)
              {
                if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

                if (input->dt_type != 2)
                {
                  e->U_spts(spt, n, ele) -= rk_beta(rkstage) * e->dt(0) / e->jaco_det_spts(spt, ele) *
                      (e->divF_spts(rkstage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
                else
                {
                  e->U_spts(spt, n, ele) -= rk_beta(rkstage) * e->dt(ele) / e->jaco_det_spts(spt,ele) *
                      (e->divF_spts(rkstage, spt, n, ele) + sourceBT.at(e->etype)(spt, n, ele));
                }
              }
            }
          }
        }
      }
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      if (!sourceBT.count(e->etype))
      {
        RK_update_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, e->jaco_det_spts_d, e->dt_d,
            rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
            input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
      else
      {
        RK_update_source_wrapper(e->U_spts_d, e->U_spts_d, e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt_d,
            rk_beta_d, input->dt_type, e->nSpts, e->nEles, e->nVars, e->nDims,
            input->equation, 0, input->nStages, true, input->overset, geo.iblank_cell_d.data());
      }
    }

    check_error();
#endif

    current_iter++;
  }
}

#ifdef _CPU
void FRSolver::step_LSRK_stage_start(int stage)
#endif
#ifdef _GPU
void FRSolver::step_LSRK_stage_start(int stage)
#endif
{
  /* NOTE: this implementation is not the 'true' low-storage implementation
   * since we are using an additional array 'U_til' instead of swapping
   * pointers at each stage */

  if (stage == 0)
  {
    // Copy current solution into "U_ini" ['rold' in PyFR]
#ifdef _CPU
    for (auto e : elesObjs)
    {
      e->U_ini = e->U_spts;
      e->U_til = e->U_spts;
      e->rk_err.fill(0.0);
    }
#endif

#ifdef _GPU
    for (auto e : elesObjs)
    {
      device_copy(e->U_ini_d, e->U_spts_d, e->U_spts_d.max_size());
      device_copy(e->U_til_d, e->U_spts_d, e->U_spts_d.max_size());
      device_fill(e->rk_err_d, e->rk_err_d.max_size());

      // Get current delta t [dt(0)] (updated on GPU)
      copy_from_device(e->dt.data(), e->dt_d.data(), 1);
    }

    check_error();
#endif

    if (input->motion_type == RIGID_BODY)
    {
      x_ini = geo.x_cg;        x_til = geo.x_cg;
      v_ini = geo.vel_cg;      v_til = geo.vel_cg;
      q_ini = geo.q;           q_til = geo.q;
      qdot_ini = geo.qdot;     qdot_til = geo.qdot;
    }
  }

  PUSH_NVTX_RANGE("ONE_STAGE",6);
  flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

  compute_residual_start(0);
}

#ifdef _CPU
void FRSolver::step_LSRK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector<double>> &sourceBT)
#endif
#ifdef _GPU
void FRSolver::step_LSRK_stage_finish(int stage, const std::map<ELE_TYPE, mdvector_gpu<double>> &sourceBT)
#endif
{
  flow_time = prev_time + rk_c(stage) * elesObjs[0]->dt(0);

  compute_residual_finish(0);

  rigid_body_update(stage);

  double ai = (stage < rk_alpha.size()) ? rk_alpha(stage) : 0;
  double bi = rk_beta(stage);
  double bhi = rk_bhat(stage);

#ifdef _CPU
  // Update Error
  for (auto e : elesObjs)
  {
#pragma omp parallel for collapse(2)
    for (unsigned int spt = 0; spt < e->nSpts; spt++)
    {
      for (unsigned int n = 0; n < e->nVars; n++)
      {
#pragma omp simd
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          e->rk_err(spt, n, ele) -= (bi - bhi) * e->dt(0) /
              e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);
        }
      }
    }

    // Update solution registers
    if (stage < input->nStages - 1)
    {
#pragma omp parallel for collapse(2)
      for (unsigned int spt = 0; spt < e->nSpts; spt++)
      {
        for (unsigned int n = 0; n < e->nVars; n++)
        {
#pragma omp simd
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

            e->U_spts(spt, n, ele) = e->U_til(spt, n, ele) - ai * e->dt(0) /
                e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);

            e->U_til(spt, n, ele) = e->U_spts(spt, n, ele) - (bi - ai) * e->dt(0) /
                e->jaco_det_spts(spt,ele) * e->divF_spts(0, spt, n, ele);
          }
        }
      }
    }
    else
    {
#pragma omp parallel for collapse(2)
      for (unsigned int spt = 0; spt < e->nSpts; spt++)
      {
        for (unsigned int n = 0; n < e->nVars; n++)
        {
#pragma omp simd
          for (unsigned int ele = 0; ele < e->nEles; ele++)
          {
            if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

            e->U_spts(spt, n, ele) = e->U_til(spt, n, ele) - bi * e->dt(0) /
                e->jaco_det_spts(spt, ele) * e->divF_spts(0, spt, n, ele);
          }
        }
      }
    }
  }
#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    if (!sourceBT.count(e->etype))
    {
      LSRK_update_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d, e->divF_spts_d,
          e->jaco_det_spts_d, e->dt(0), ai, bi, bhi, e->nSpts, e->nEles,
          e->nVars, stage, input->nStages, input->overset, geo.iblank_cell_d.data());
    }
    else
    {
      LSRK_update_source_wrapper(e->U_spts_d, e->U_til_d, e->rk_err_d,
          e->divF_spts_d, sourceBT.at(e->etype), e->jaco_det_spts_d, e->dt(0), ai, bi, bhi,
          e->nSpts, e->nEles, e->nVars, stage, input->nStages, input->overset,
          geo.iblank_cell_d.data());
    }
  }
  check_error();
#endif
  POP_NVTX_RANGE;

  flow_time = prev_time + elesObjs[0]->dt(0);
}


void FRSolver::compute_element_dt(bool pseudo_time)
{
  double CFL;
  unsigned int CFL_type, dt_type;
  if (pseudo_time)
  {
    CFL = input->CFL_tau;
    CFL_type = input->CFL_tau_type;
    dt_type = input->dtau_type;
  }
  else
  {
    CFL = input->CFL;
    CFL_type = input->CFL_type;
    dt_type = input->dt_type;
  }

#ifdef _CPU
  /* CFL-estimate used by Liang, Lohner, and others. Factor of 2 to be 
   * consistent with 1D CFL estimates. */
  if (CFL_type == 1)
  {
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      { 
        unsigned int eleBT = ele + e->startEle;
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        double int_waveSp = 0.;  /* Edge/Face integrated wavespeed */

        for (unsigned int fpt = 0; fpt < e->nFpts; fpt++)
        {
          int gfpt = geo.fpt2gfptBT[e->etype](fpt,eleBT);
          int slot = geo.fpt2gfpt_slotBT[e->etype](fpt,eleBT);

          int_waveSp += e->weights_fpts(fpt % e->nFptsPerFace) * faces->waveSp(gfpt) * faces->dA(slot, gfpt); 
        }

        double *dt_ptr = pseudo_time ? &e->dtau(ele) : &e->dt(ele);
        *dt_ptr = 2.0 * CFL * get_cfl_limit_adv(order) * e->vol(ele) / int_waveSp;
      }
    }
  }

  /* CFL-estimate based on MacCormack for NS */
  else if (CFL_type == 2)
  {
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      { 
        unsigned int eleBT = ele + e->startEle;
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        /* Compute inverse of timestep in each face */
        std::vector<double> dtinv(2*e->nDims);
        for (unsigned int face = 0; face < 2*e->nDims; face++)
        {
          for (unsigned int fpt = face * e->nFptsPerFace; fpt < (face+1) * e->nFptsPerFace; fpt++)
          {
            int gfpt = geo.fpt2gfptBT[e->etype](fpt,eleBT);

            double dtinv_temp = faces->waveSp(gfpt) / (get_cfl_limit_adv(order) * e->h_ref(fpt, ele));
            if (input->viscous)
              dtinv_temp += faces->diffCo(gfpt) / (get_cfl_limit_diff(order, input->ldg_b) * e->h_ref(fpt, ele) * e->h_ref(fpt, ele));
            dtinv[face] = std::max(dtinv[face], dtinv_temp);
          }
        }

        /* Find maximum in each dimension */
        double *dt_ptr = pseudo_time ? &e->dtau(ele) : &e->dt(ele);
        if (e->nDims == 2)
        {
          dtinv[0] = std::max(dtinv[0], dtinv[2]);
          dtinv[1] = std::max(dtinv[1], dtinv[3]);

          *dt_ptr = CFL / (dtinv[0] + dtinv[1]);
        }
        else
        {
          dtinv[0] = std::max(dtinv[0],dtinv[1]);
          dtinv[1] = std::max(dtinv[2],dtinv[3]);
          dtinv[2] = std::max(dtinv[4],dtinv[5]);

          /// NOTE: this seems ultra-conservative.  Need additional scaling factor?
          *dt_ptr = CFL / (dtinv[0] + dtinv[1] + dtinv[2]); // * 32; = empirically-found factor for sphere
        }
      }
    }
  }

  /* Global minimum */
  if (dt_type == 1) 
  {
    double minDT = INFINITY;
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        double *dt_ptr = pseudo_time ? &e->dtau(ele) : &e->dt(ele);
        minDT = std::min(minDT, *dt_ptr);
      }
    }

#ifdef _MPI
    /// TODO: If interfacing with other explicit solver, work together here
    MPI_Allreduce(MPI_IN_PLACE, &minDT, 1, MPI_DOUBLE, MPI_MIN, worldComm);
#endif

    for (auto e : elesObjs)
    {
      double *dt_ptr = pseudo_time ? &e->dtau(0) : &e->dt(0);
      *dt_ptr = minDT;
    }
  }
#endif

#ifdef _GPU
  /* CFL estimate */
  for (auto e : elesObjs)
  {
    if (pseudo_time)
      compute_element_dt_wrapper(e->dtau_d, faces->waveSp_d, faces->diffCo_d, faces->dA_d, 
          geo.fpt2gfptBT_d[e->etype], geo.fpt2gfpt_slotBT_d[e->etype], e->weights_fpts_d, e->vol_d, 
          e->h_ref_d, e->nFptsPerFace, CFL, input->ldg_b, order, CFL_type, e->nFpts, e->nEles, 
          e->nDims, e->startEle, input->overset, geo.iblank_cell_d.data());
    else
      compute_element_dt_wrapper(e->dt_d, faces->waveSp_d, faces->diffCo_d, faces->dA_d, 
          geo.fpt2gfptBT_d[e->etype], geo.fpt2gfpt_slotBT_d[e->etype], e->weights_fpts_d, e->vol_d, 
          e->h_ref_d, e->nFptsPerFace, CFL, input->ldg_b, order, CFL_type, e->nFpts, e->nEles, 
          e->nDims, e->startEle, input->overset, geo.iblank_cell_d.data());
  }
  check_error();

  /* Global minimum */
  if (dt_type == 1) 
  {
    double minDT = INFINITY;
    for (auto e : elesObjs)
    {
      double dt;
      if (pseudo_time)
        dt = device_min(e->dtau_d, e->dtau_d.max_size());
      else
        dt = device_min(e->dt_d, e->dt_d.max_size());
      minDT = std::min(minDT, dt);
    }

#ifdef _MPI
    /// TODO: If interfacing with other explicit solver, work together here
    MPI_Allreduce(MPI_IN_PLACE, &minDT, 1, MPI_DOUBLE, MPI_MIN, myComm);
#endif

    for (auto e : elesObjs)
    {
      if (pseudo_time)
      {
        e->dtau(0) = minDT;
        copy_to_device(e->dtau_d.data(), e->dtau.data(), 1);
        //device_fill(e->dtau_d, 1, minDT);
      }
      else
      {
        e->dt(0) = minDT;
        copy_to_device(e->dt_d.data(), e->dt.data(), 1);
        //device_fill(e->dt_d, 1, minDT);
      }
    }
  }
#endif
}

void FRSolver::write_solution_pyfr(const std::string &_prefix)
{
  if (elesObjs.size() > 1)
    ThrowException("PyFR write not supported for mixed element grids.");

  auto e = elesObjs[0];

  /* --- Apply polynomial squeezing if requested --- */

  if (input->squeeze)
  {
    e->compute_Uavg();
    e->poly_squeeze();
  }

#ifdef _GPU
    e->U_spts = e->U_spts_d;
#endif

  std::string prefix = _prefix;

  if (input->overset) prefix += "-G" + std::to_string(input->gridID);

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  std::string filename = input->output_prefix + "/" + prefix + "-" + std::to_string(iter) + ".pyfrs";

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing data to file " << filename << std::endl;

  std::stringstream ss;

  // Create a dataspace and datatype for a std::string
  DataSpace dspace(H5S_SCALAR);
  hid_t string_type = H5Tcopy (H5T_C_S1);
  H5Tset_size (string_type, H5T_VARIABLE);

  // Setup config and stats strings

  /* --- Config String --- */
  ss.str(""); ss.clear();
  ss << "[constants]" << std::endl;
  ss << "gamma = 1.4" << std::endl;
  ss << std::endl;

  ss << "[solver]" << std::endl;
  ss << "system = ";
  if (input->equation == EulerNS) 
  {
    if (input->viscous)
      ss << "navier-stokes" << std::endl;
    else
      ss << "euler" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    if (input->viscous)
      ss << "advection-diffusion" << std::endl;
    else
      ss << "advection" << std::endl;
  }
  ss << "order = " << input->order << std::endl;
  if (input->overset)
  {
    ss << "overset = true" << std::endl;
  }
  ss << std::endl;

  if (geo.nDims == 2)
    ss << "[solver-elements-quad]" << std::endl;
  else
    ss << "[solver-elements-hex]" << std::endl;
  ss << "soln-pts = gauss-legendre" << std::endl;
  ss << std::endl;

  std::string config = ss.str();

  /* --- Stats String --- */
  ss.str(""); ss.clear();
  ss << "[data]" << std::endl;
  if (input->equation == EulerNS)
  {
    ss << "fields = rho,rhou,rhov,";
    if (geo.nDims == 3)
      ss << "rhoW,";
    ss << "E" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    ss << "fields = u" << std::endl;
  }
  ss << "prefix = soln" << std::endl;
  ss << std::endl;

  ss << "[solver-time-integrator]" << std::endl;
  ss << "tcurr = " << flow_time << std::endl;
  //ss << "wall-time = " << input->??"
  ss << std::endl;

  if (input->motion)
  {
    if (input->motion_type == RIGID_BODY || input->motion_type == CIRCULAR_TRANS)
    {
      ss << "[moving-grid]" << std::endl;

      ss << "x-cg =";
      for (int d = 0; d < geo.nDims; d++)
        ss << " " << std::scientific << std::setprecision(16) << geo.x_cg(d);
      ss << std::endl;

      ss << "v-cg =";
      for (int d = 0; d < geo.nDims; d++)
        ss << " " << std::scientific << std::setprecision(16) << geo.vel_cg(d);
      ss << std::endl;

      if (input->motion_type == RIGID_BODY)
      {
        ss << "rot-q =";
        for (int d = 0; d < 4; d++)
          ss << " " << std::scientific << std::setprecision(16) << geo.q(d);
        ss << std::endl;

        ss << "omega =";
        for (int d = 0; d < 3; d++)
          ss << " " << std::scientific << std::setprecision(16) << geo.omega(d);
        ss << std::endl;
      }

      ss << std::endl;
    }
  }

  std::string stats = ss.str();

  int nEles = e->nEles;
  int nElesPad = e->nElesPad;
  int nVars = e->nVars;
  int nSpts = e->nSpts;

#ifdef _MPI
  // Need to traspose data to match PyFR layout - [spts,vars,eles] in row-major format
  std::vector<std::vector<double>> data_p(input->nRanks);
  std::vector<std::vector<int>> iblank_p(input->nRanks);
  if (input->overset && input->rank == 0)
  {
    iblank_p[0].resize(nEles);
    for (int ele = 0; ele < nEles; ele++)
      iblank_p[0][ele] = geo.iblank_cell(ele);
  }

  /* --- Gather all the data onto Rank 0 for writing --- */

  std::vector<int> nEles_p(input->nRanks);
  MPI_Allgather(&nEles, 1, MPI_INT, nEles_p.data(), 1, MPI_INT, geo.myComm);

  for (int p = 1; p < input->nRanks; p++)
  {
    int nEles = nEles_p[p];
    int nElesPad = nEles;
#ifdef _GPU
    nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#endif
    int size = nElesPad * nSpts * nVars;

    if (input->rank == 0)
    {
      data_p[p].resize(size);
      MPI_Status status;
      MPI_Recv(data_p[p].data(), size, MPI_DOUBLE, p, 0, geo.myComm, &status);

      if (input->overset)
      {
        iblank_p[p].resize(nEles);
        MPI_Status status2;
        MPI_Recv(iblank_p[p].data(), nEles, MPI_INT, p, 0, geo.myComm, &status2);
      }
    }
    else
    {
      if (p == input->rank)
      {
        MPI_Send(e->U_spts.data(), size, MPI_DOUBLE, 0, 0, geo.myComm);

        if (input->overset)
        {
          MPI_Send(geo.iblank_cell.data(), nEles, MPI_INT, 0, 0, geo.myComm);
        }
      }
    }

    MPI_Barrier(geo.myComm);
  }

  /* --- Write Data to File (on Rank 0) --- */

  if (input->rank == 0)
  {
    H5File file(filename, H5F_ACC_TRUNC);

    // Write out all the data
    DataSet dset = file.createDataSet("config", string_type, dspace);
    dset.write(config, string_type, dspace);
    dset.close();

    dset = file.createDataSet("stats", string_type, dspace);
    dset.write(stats, string_type, dspace);
    dset.close();

    // Write mesh ID
    dset = file.createDataSet("mesh_uuid", string_type, dspace);
    dset.write(geo.mesh_uuid, string_type, dspace);
    dset.close();

    dspace.close();

    std::string sol_prefix = "soln_";
    sol_prefix += (geo.nDims == 2) ? "quad" : "hex";
    sol_prefix += "_p";
    for (int p = 0; p < input->nRanks; p++)
    {
      nEles = nEles_p[p];
#ifdef _GPU
      nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#else
      nElesPad = nEles;
#endif

      hsize_t dimsU[3] = {nSpts, nVars, nElesPad};
      hsize_t dimsF[3] = {nSpts, nVars, nEles};

      // Create a dataspace for the solution, using a hyperslab to ignore padding
      DataSpace dspaceU(3, dimsU);
#ifdef _GPU
      hsize_t count[3] = {1,1,1};
      hsize_t start[3] = {0,0,0};
      hsize_t stride[3] = {1,1,1};
      hsize_t block[3] = {nSpts, nVars, nEles};
      dspaceU.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
#endif

      // Create a dataspace for the actual dataset
      DataSpace dspaceF(3, dimsF);

      std::string solname = sol_prefix + std::to_string(p);
      dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceF);
      if (p == 0)
        dset.write(e->U_spts.data(), PredType::NATIVE_DOUBLE, dspaceU);
      else
        dset.write(data_p[p].data(), PredType::NATIVE_DOUBLE, dspaceU);

      dspaceU.close();
      dset.close();

      // Write out iblank as separate DataSet with same naming convention as soln
      if (input->overset)
      {
        hsize_t dims[1] = {nEles};
        DataSpace dspaceI(1, dims);
        std::string iname = "iblank_";
        iname += (geo.nDims == 2) ? "quad" : "hex";
        iname += "_p" + std::to_string(p);
        // NOTE: be aware of C++ 32-bit int vs. HDF5 8-bit int
        dset = file.createDataSet(iname, PredType::NATIVE_INT8, dspaceI);
        dset.write(iblank_p[p].data(), PredType::NATIVE_INT, dspaceI);
        dset.close();
      }
    }
  }
#else
  /* --- Write Data to File --- */

  H5File file(filename, H5F_ACC_TRUNC);

  DataSet dset = file.createDataSet("config", string_type, dspace);
  dset.write(config, string_type, dspace);
  dset.close();

  dset = file.createDataSet("stats", string_type, dspace);
  dset.write(stats, string_type, dspace);
  dset.close();

  // Write mesh ID
  dset = file.createDataSet("mesh_uuid", string_type, dspace);
  dset.write(geo.mesh_uuid, string_type, dspace);
  dset.close();

  dspace.close();

  hsize_t dimsU[3] = {nSpts, nVars, nElesPad};
  hsize_t dimsF[3] = {nSpts, nVars, nEles};

  // Create a dataspace for the solution, using a hyperslab to ignore padding
  DataSpace dspaceU(3, dimsU);
#ifdef _GPU
  hsize_t count[3] = {1,1,1};
  hsize_t start[3] = {0,0,0};
  hsize_t stride[3] = {1,1,1};
  hsize_t block[3] = {nSpts, nVars, nEles};
  dspaceU.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
#endif

  // Create a dataspace for the actual dataset
  DataSpace dspaceF(3, dimsF);

  std::string solname = "soln_";
  solname += (geo.nDims == 2) ? "quad" : "hex";
  solname += "_p" + std::to_string(input->rank);
  dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceF);
  dset.write(e->U_spts.data(), PredType::NATIVE_DOUBLE, dspaceU);
  dset.close();
#endif 

}

void FRSolver::restart_pyfr(std::string restart_file, unsigned restart_iter)
{
  if (elesObjs.size() > 1)
    ThrowException("PyFR restart not supported for mixed element grids.");

  auto e = elesObjs[0];

  std::string filename = restart_file;

  if (input->restart_type > 0) // append .pvtu / .vtu to case name
  {
    std::stringstream ss;

    ss << restart_file << "/" << restart_file;

    if (input->overset)
    {
      ss << "-G" << input->gridID;
    }

    ss << "-" << restart_iter;
    ss << ".pyfrs";

    filename = ss.str();
  }
  else
  {
    std::string str = filename;
    size_t ind = str.find("-");
    str.erase(str.begin(), str.begin()+ind+1);
    ind = str.find(".pyfrs");
    str.erase(str.begin()+ind,str.end());
    std::stringstream ss(str);
    ss >> restart_iter;
  }

  current_iter = restart_iter;
  input->iter = restart_iter;
  input->initIter = restart_iter;

  if (input->rank == 0)
    std::cout << "Reading data from file " << filename << std::endl;

  H5File file(filename, H5F_ACC_RDONLY);

  // Read the mesh ID string
  std::string mesh_uuid;

  DataSet dset = file.openDataSet("mesh_uuid");
  DataType dtype = dset.getDataType();
  DataSpace dspace(H5S_SCALAR);

  dset.read(mesh_uuid, dtype, dspace);
  dset.close();

  if (mesh_uuid != geo.mesh_uuid)
    ThrowException("Restart Error - Mesh and solution files do not mesh [mesh_uuid].");

  // Read the config string
  dset = file.openDataSet("config");
  dset.read(geo.config, dtype, dspace);
  dset.close();

  // Read the stats string
  dset = file.openDataSet("stats");
  dset.read(geo.stats, dtype, dspace);
  dset.close();

  // Read the solution data
  std::string solname = "soln_";
  solname += (geo.nDims == 2) ? "quad" : "hex";
  solname += "_p" + std::to_string(input->rank); /// TODO: write per rank in parallel...

  dset = file.openDataSet(solname);
  auto ds = dset.getSpace();

  hsize_t dims[3];
  int ds_rank = ds.getSimpleExtentDims(dims);

  if (ds_rank != 3)
    ThrowException("Improper DataSpace rank for solution data.");

  // Create a datatype for a std::string
  hid_t string_type = H5Tcopy (H5T_C_S1);
  H5Tset_size (string_type, H5T_VARIABLE);

  unsigned int nEles = e->nEles;
  unsigned int nVars = e->nVars;
#ifdef _GPU
  unsigned int nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#else
  unsigned int nElesPad = nEles;
#endif

  if (dims[2] != nEles || dims[1] != nVars)
    ThrowException("Size of solution data set does not match that from mesh.");

  unsigned int nSpts = dims[0];

  if (input->overset)
  {
    geo.iblank_cell.assign({geo.nEles}, 1);

    // Check for a dataset of iblank
    std::string iname = "iblank_";
    iname += (geo.nDims == 2) ? "quad" : "hex";
    iname += "_p" + std::to_string(input->rank);

    try
    {
      DataSet dsetI = file.openDataSet(iname);
      DataSpace dspaceI = dsetI.getSpace();
      hsize_t dimsI[1];
      dspaceI.getSimpleExtentDims(dimsI);

      if (dimsI[0] != geo.nEles) ThrowException("Error reading iblank data!");

      dsetI.read(geo.iblank_cell.data(), PredType::NATIVE_INT);
    }
    catch (...)
    {
      // No iblank dataset found; try looking for an iblank attribute:
      if (dset.attrExists("iblank"))
      {
        Attribute att = dset.openAttribute("iblank");
        DataSpace dspaceI = att.getSpace();
        hsize_t dim[1];
        dspaceI.getSimpleExtentDims(dim);

        if (geo.nEles <= MAX_H5_ATTR_SIZE && dim[0] != geo.nEles)
          ThrowException("Attribute error - expecting size of 'iblank' to be nEles");

        att.read(PredType::NATIVE_INT, geo.iblank_cell.data());
      }
    }
  }

  // Create dataspaces to handle reading of data into padded solution storage
  hsize_t dimsU[3] = {nSpts, nVars, nElesPad};
  hsize_t dimsF[3] = {nSpts, nVars, nEles};

  DataSpace dspaceU(3, dimsU);
  DataSpace dspaceF(3, dimsF);
#ifdef _GPU
  hsize_t count[3] = {1,1,1};
  hsize_t start[3] = {0,0,0};
  hsize_t stride[3] = {1,1,1};
  hsize_t block[3] = {nSpts, nVars, nEles};
  dspaceU.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
#endif

  if (nSpts == e->nSpts)
  {
    e->U_spts.assign({nSpts,nVars,nElesPad});
    dset.read(e->U_spts.data(), PredType::NATIVE_DOUBLE, dspaceU, dspaceF);
  }
  else
  {
    int restartOrder = (input->nDims == 2)
                     ? (std::sqrt(nSpts)-1) : (std::cbrt(nSpts)-1);

    // Read into temporary storage, in case changing polynomial order
    mdvector<double> U_restart({nSpts,nVars,nElesPad});
    dset.read(U_restart.data(), PredType::NATIVE_DOUBLE, dspaceU, dspaceF);

    // Setup extrapolation operator from restart points
    e->set_oppRestart(restartOrder);

    // Extrapolate values from restart points to solution points
    auto &A = e->oppRestart(0, 0);
    auto &B = U_restart(0, 0, 0);
    auto &C = e->U_spts(0, 0, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nSpts,
        e->nElesPad * e->nVars, nSpts, 1.0, &A, nSpts, &B,
        e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);
  }

  dset.close();

  // Process the config / stats string
  std::string str, key, tmp;
  std::stringstream ss;
  std::istringstream stats(geo.stats);

  while (std::getline(stats, str))
  {
    ss.str(str);  ss >> key;
    if (key == "tcurr")
    {
      // "tcurr = ####"
      ss >> tmp >> restart_time;
      flow_time = restart_time;
    }
    if (key == "x-cg")
    {
      geo.x_cg.assign({geo.nDims});
      ss >> tmp >> geo.x_cg(0) >> geo.x_cg(1) >> geo.x_cg(2);
    }
    if (key == "v-cg")
    {
      geo.vel_cg.assign({geo.nDims});
      ss >> tmp >> geo.vel_cg(0) >> geo.vel_cg(1) >> geo.vel_cg(2);
    }
    if (key == "rot-q")
    {
      geo.q.assign({4});
      ss >> tmp >> geo.q(0) >> geo.q(1) >> geo.q(2) >> geo.q(3);
    }
    if (key == "omega")
    {
      if (geo.omega.size() != 3)
      {
        geo.omega.assign({3});
        ss >> tmp >> geo.omega(0) >> geo.omega(1) >> geo.omega(2);

        geo.qdot.assign({4});
        for (int i = 0; i < 3; i++)
          geo.qdot(i+1) = 0.5*geo.omega(i);
      }
    }

    ss.str(""); ss.clear();
    key.clear();
  }

  if (input->motion_type == RIGID_BODY)
  {
    Quat q(geo.q(0), geo.q(1), geo.q(2), geo.q(3));
    Quat omega(0., geo.omega(0), geo.omega(1), geo.omega(2));
    Quat qdot = 0.5*omega*q;
    for (int i = 0; i < 4; i++)
      geo.qdot(i) = qdot[i];
  }
}

void FRSolver::write_solution(const std::string &_prefix)
{
#ifdef _GPU
  for (auto e : elesObjs)
    e->U_spts = e->U_spts_d;
#endif

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing data to file for case " << prefix << "..." << std::endl;

  if (input->overset) prefix += "_Grid" + std::to_string(input->gridID);

  std::stringstream ss;

#ifdef _MPI
  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_" << std::setw(9) << std::setfill('0');
    ss << iter << ".pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;

    std::vector<std::string> var;
    if (input->equation == AdvDiff)
    {
      var = {"u"};
    }
    else if (input->equation == EulerNS)
    {
      if (geo.nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

    }

    for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
    {
      f << "<PDataArray type=\"Float64\" Name=\"" << var[n] << "\"/>" << std::endl;
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
      f << std::endl;
    }
    if (input->motion)
    {
      f << "<PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"grid_velocity\"/>";
      f << std::endl;
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" />" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
      ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write partition solution to file in binary .vtu format */
  std::ofstream f(outputfile, std::ios::binary);

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" >" << std::endl;

  /* Write comments for solution order, iteration number and flowtime */
  f << "<!-- ORDER " << input->order << " -->" << std::endl;
  f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << iter << " -->" << std::endl;
  if (input->overset)
  {
    f << "<!-- IBLANK ";
    for (unsigned int ic = 0; ic < eles->nEles; ic++)
    {
      f << geo.iblank_cell(ic) << " ";
    }
    f << " -->" << std::endl;
  }

  if (input->motion)
  {
    for (auto e : elesObjs)
    {
      if (input->motion_type == RIGID_BODY || input->motion_type == CIRCULAR_TRANS)
      {
        f << "<!-- X_CG ";
        for (int d = 0; d < 3; d++)
          f << std::scientific << std::setprecision(16) << geo.x_cg(d) << " ";
        f << " -->" << std::endl;

        f << "<!-- V_CG ";
        for (int d = 0; d < 3; d++)
          f << std::scientific << std::setprecision(16) << geo.vel_cg(d) << " ";
        f << " -->" << std::endl;

        if (input->motion_type == RIGID_BODY)
        {
          f << "<!-- ROT-Q ";
          for (int d = 0; d < 4; d++)
            f << std::scientific << std::setprecision(16) << geo.q(d) << " ";
          f << " -->" << std::endl;

          f << "<!-- OMEGA ";
          for (int d = 0; d < 3; d++)
            f << std::scientific << std::setprecision(16) << geo.omega(d) << " ";
          f << " -->" << std::endl;
        }
      }

      e->update_plot_point_coords();
#ifdef _GPU
      e->grid_vel_nodes = e->grid_vel_nodes_d;
#endif
    }
  }

  std::vector<unsigned int> nElesBO(elesObjs.size());
  for (auto e : elesObjs)
    nElesBO[e->elesObjID] = e->nEles;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (auto e : elesObjs)
    {
      for (int ele = 0; ele < e->nEles; ele++)
        if (geo.iblank_cell(ele) != NORMAL) nElesBO[e->elesObjID]--;
//      if (geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) nElesBO[e->elesObjID]--;
    }
  }

  unsigned int nCells = 0;
  unsigned int nPts = 0;

  for (auto e : elesObjs)
  {
    nCells += e->nSubelements * nElesBO[e->elesObjID];
    nPts += e->nPpts * nElesBO[e->elesObjID];
  }

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPts << "\" ";
  f << "NumberOfCells=\"" << nCells << "\">";
  f << std::endl;


  size_t b_offset = 0;
  /* Write solution information */
  f << "<PointData>" << std::endl;

  std::vector<std::string> var;
  if (input->equation == AdvDiff)
  {
    var = {"u"};
  }
  else if(input->equation == EulerNS)
  {
    if (geo.nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};
  }

  for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
  {
    f << "<DataArray type=\"Float64\" Name=\"" << var[n] << "\" ";
    f << "format=\"appended\" offset=\"" << b_offset << "\"/>"<< std::endl;
    b_offset += sizeof(unsigned int);
    for (auto e : elesObjs)
      b_offset += (nElesBO[e->elesObjID] * e->nPpts * sizeof(double));
  }

  if (input->filt_on && input->sen_write)
  {
#ifdef _GPU
  for (auto e : elesObjs)
    filt.sensor[e->etype] = filt.sensor_d[e->etype];
#endif
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << filt.sensor[e->etype](ele) << " ";
        }
        f << std::endl;
      }
    }
    f << "</DataArray>" << std::endl;
  }

  if (input->motion)
  {
    eles->get_grid_velocity_ppts();

    f << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
    f << "format=\"appended\" offset=\"" << b_offset << "\"/>"<< std::endl;
    b_offset += sizeof(unsigned int);
    for (auto e : elesObjs)
      b_offset += (nElesBO[e->elesObjID] * e->nPpts * 3 * sizeof(double));
  }

  f << "</PointData>" << std::endl;

  /* Write plot point information (single precision) */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  f << "</Points>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (nElesBO[e->elesObjID] * e->nPpts * 3 * sizeof(float));

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (nElesBO[e->elesObjID] * e->nSubelements * e->nNodesPerSubelement * sizeof(unsigned int));

  f << "<DataArray type=\"UInt32\" Name=\"offsets\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (nElesBO[e->elesObjID] * e->nSubelements * sizeof(unsigned int));

  f << "<DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" ";
  f << "offset=\""<< b_offset << "\"/>" << std::endl;
  b_offset += sizeof(unsigned int);
  for (auto e : elesObjs)
    b_offset += (nElesBO[e->elesObjID] * e->nSubelements * sizeof(char));
  f << "</Cells>" << std::endl;

  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;

  /* Adding raw binary data as AppendedData*/
  f << "<AppendedData encoding=\"raw\">" << std::endl;
  f << "_"; // leading underscore

  /* Write solution data */
  /* Extrapolate solution to plot points */
  for (auto e : elesObjs)
  {
    auto &A = e->oppE_ppts(0, 0);
    auto &B = e->U_spts(0, 0, 0);
    auto &C = e->U_ppts(0, 0, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nPpts, 
        e->nElesPad * e->nVars, e->nSpts, 1.0, &A, e->nSpts, &B,
        e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);

    /* Apply squeezing if needed */
    if (input->squeeze)
    {
      e->compute_Uavg();

#ifdef _GPU
      e->Uavg = e->Uavg_d;
#endif

      e->poly_squeeze_ppts();
    }
  }

  unsigned int nBytes = 0;
  double dzero = 0.0;
  float fzero = 0.0f;

  /* Write out conservative variables */

  for (auto e : elesObjs)
    nBytes += nElesBO[e->elesObjID] * e->nPpts * sizeof(double);

  for (unsigned int n = 0; n < elesObjs[0]->nVars; n++)
  {
    binary_write(f, nBytes);
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          binary_write(f, e->U_ppts(ppt, n, ele));
        }
      }
    }
  }


  /* Write out grid velocity (for moving grids) */
  if (input->motion)
  {
    nBytes = 0;
    for (auto e : elesObjs)
      nBytes += nElesBO[e->elesObjID] * e->nPpts * 3 * sizeof(double);
    binary_write(f, nBytes);
    for (auto e : elesObjs)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          for (unsigned int dim = 0; dim < e->nDims; dim++)
            binary_write(f, e->grid_vel_ppts(ppt, dim, ele));

          if (geo.nDims == 2)
            binary_write(f, dzero);
        }
      }
    }
  }

  /* Write plot point coordinates */
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += nElesBO[e->elesObjID] * e->nPpts * 3 * sizeof(float);
  binary_write(f, nBytes);

  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
      {
        binary_write(f, (float) e->coord_ppts(ppt, 0, ele));
        binary_write(f, (float) e->coord_ppts(ppt, 1, ele));
        if (geo.nDims == 2)
          binary_write(f, fzero);
        else
          binary_write(f, (float) e->coord_ppts(ppt, 2, ele));
      }
    }
  }

  /* Write cell information */
  // Write connectivity
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += nElesBO[e->elesObjID] * e->nSubelements * e->nNodesPerSubelement * sizeof(unsigned int);
  binary_write(f, nBytes);

  int shift = 0; // To account for blanked elements
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        for (unsigned int i = 0; i < e->nNodesPerSubelement; i++)
        {
          binary_write(f, e->ppt_connect(i, subele) + shift);
        }
      }
      shift += e->nPpts;
    }
  }

  // Offsets
  nBytes = 0;
  for (auto e: elesObjs)
    nBytes += nElesBO[e->elesObjID] * e->nSubelements * sizeof(unsigned int);

  binary_write(f, nBytes);

  unsigned int offset = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        offset += e->nNodesPerSubelement;
        binary_write(f, offset);
      }
    }
  }

  // Types
  nBytes = 0;
  for (auto e : elesObjs)
    nBytes += nElesBO[e->elesObjID] * e->nSubelements * sizeof(char);
  binary_write(f, nBytes);

  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        if (e->etype == QUAD)
          binary_write(f, (char) 9);
        else if (e->etype == TRI)
          binary_write(f, (char) 5);
        else if (e->etype == TET)
          binary_write(f, (char) 10);
        else if (e->etype == HEX)
          binary_write(f, (char) 12);
      }
    }
  }

  f << std::endl;
  f << "</AppendedData>" << std::endl;

  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::write_color()
{
  if (input->rank == 0) std::cout << "Writing colors to file..." << std::endl;

  std::stringstream ss;
#ifdef _MPI

  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << input->output_prefix << "_color.pvtu";
   
    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;
    f << "<PDataArray type=\"Int32\" Name=\"color\" format=\"ascii\"/>";
    f << std::endl;
    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;



    for (unsigned int n = 0; n < input->nRanks; n++)
    { 
      ss.str("");
      ss << input->output_prefix << "_color_"; 
      ss << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_color_";
  ss << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << input->output_prefix << "_color";
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write partition color to file in .vtu format */
  std::ofstream f(outputfile);

  std::vector<unsigned int> nElesBO(elesObjs.size());
  for (auto e : elesObjs)
    nElesBO[e->elesObjID] = e->nEles;

  if (input->overset)
  {
    /* Remove blanked elements from total cell count */
    for (auto e : elesObjs)
    {
      for (int ele = 0; ele < e->nEles; ele++)
        if (geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) nElesBO[e->elesObjID]--;
    }
  }

  unsigned int nCells = 0;
  unsigned int nPts = 0;

  for (auto e : elesObjs)
  {
    nCells += e->nSubelements * nElesBO[e->elesObjID];
    nPts += e->nPpts * nElesBO[e->elesObjID];
  }

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;
  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPts << "\" ";
  f << "NumberOfCells=\"" << nCells << "\">";
  f << std::endl;
  
  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl; 

  for (auto e : elesObjs)
  {
    if (e->nDims == 2)
    {
      // TODO: Change order of ppt structures for better looping 
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << e->coord_ppts(ppt, 0, ele) << " ";
          f << e->coord_ppts(ppt, 1, ele) << " ";
          f << 0.0 << std::endl;
        }
      }
    }
    else
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
        for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
        {
          f << e->coord_ppts(ppt, 0, ele) << " ";
          f << e->coord_ppts(ppt, 1, ele) << " ";
          f << e->coord_ppts(ppt, 2, ele) << std::endl;
        }
      }
    }
  }
  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;
  int shift = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        for (unsigned int i = 0; i < e->nNodesPerSubelement; i++)
        {
          f << e->ppt_connect(i, subele) + shift << " ";
        }
        f << std::endl;
      }

      shift += e->nPpts;
    }
  }
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  unsigned int offset = 0;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        offset += e->nNodesPerSubelement;
        f << offset << " ";
      }
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      for (unsigned int subele = 0; subele < e->nSubelements; subele++)
      {
        if (e->etype == QUAD)
          f << 9 << " ";
        else if (e->etype == TRI)
          f << 5 << " ";
        else if (e->etype == HEX)
          f << 12 << " ";
      }
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write color information */
  f << "<PointData>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"color\" ";
  f << "format=\"ascii\">"<< std::endl;
  for (auto e : elesObjs)
  {
    for (unsigned int ele = e->startEle; ele < e->endEle; ele++)
    {
      if (input->overset && geo.iblank_cell(geo.eleID[e->etype](ele)) != NORMAL) continue;
      for (unsigned int ppt = 0; ppt < e->nPpts; ppt++)
      {
        f << std::scientific << std::setprecision(16) << geo.ele2colorBT[e->etype](ele);
        f  << " ";
      }
      f << std::endl;
    }
  }
  f << "</DataArray>" << std::endl;
  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}

void FRSolver::write_averages(const std::string &_prefix)
{
  if (elesObjs.size() > 1)
    ThrowException("PyFR write not supported for mixed element grids.");

  auto e = elesObjs[0];

#ifdef _GPU
    e->tavg_acc = e->tavg_acc_d;
#endif

  std::string prefix = _prefix + "-tavg";

  if (input->overset) prefix += "-G" + std::to_string(input->gridID);

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  std::string filename = input->output_prefix + "/" + prefix + "-" + std::to_string(iter) + ".pyfrs";

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing time averages to file " << filename << std::endl;

  std::stringstream ss;

  // Create a dataspace and datatype for a std::string
  DataSpace dspace(H5S_SCALAR);
  hid_t string_type = H5Tcopy (H5T_C_S1);
  H5Tset_size (string_type, H5T_VARIABLE);

  // Setup config and stats strings

  /* --- Config String --- */
  ss.str(""); ss.clear();
  ss << "[constants]" << std::endl;
  ss << "gamma = 1.4" << std::endl;
  ss << std::endl;

  ss << "[solver]" << std::endl;
  ss << "system = ";
  if (input->equation == EulerNS)
  {
    if (input->viscous)
      ss << "navier-stokes" << std::endl;
    else
      ss << "euler" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    if (input->viscous)
      ss << "advection-diffusion" << std::endl;
    else
      ss << "advection" << std::endl;
  }
  ss << "order = " << input->order << std::endl;
  if (input->overset)
  {
    ss << "overset = true" << std::endl;
  }
  ss << std::endl;

  if (geo.nDims == 2)
    ss << "[solver-elements-quad]" << std::endl;
  else
    ss << "[solver-elements-hex]" << std::endl;
  ss << "soln-pts = gauss-legendre" << std::endl;
  ss << std::endl;

  std::string config = ss.str();

  /* --- Stats String --- */
  ss.str(""); ss.clear();
  ss << "[data]" << std::endl;
  if (input->equation == EulerNS)
  {
    ss << "fields = avg-rho,avg-rhou,avg-rhov,";
    if (geo.nDims == 3)
      ss << "avg-rhoW,";
    ss << "avg-E,avg-vx,avg-vy,";
    if (geo.nDims == 3)
      ss << "avg-vz,";
    ss << "avg-P" << std::endl;
  }
  else if (input->equation == AdvDiff)
  {
    ss << "fields = u" << std::endl;
  }
  ss << "prefix = tavg" << std::endl;
  ss << std::endl;

  ss << "[solver-time-integrator]" << std::endl;
  ss << "tcurr = " << flow_time << std::endl;
  ss << std::endl;

  ss << "[tavg]" << std::endl;
  ss << "tstart = " << restart_time << std::endl;
  ss << "tend = " << flow_time << std::endl;
  //ss << "wall-time = " << input->??"
  ss << std::endl;

  std::string stats = ss.str();

  int nEles = e->nEles;
  int nElesPad = e->nElesPad;
  int nVars = e->nVars + e->nDims + 1;
  int nSpts = e->nSpts;

  // Normalize the accumulated data
  for (int spt = 0; spt < nSpts; spt++)
    for (int n = 0; n < nVars; n++)
      for (int ele = 0; ele < nEles; ele++)
        e->tavg_curr(spt,n,ele) = e->tavg_acc(spt,n,ele) / (flow_time - restart_time);

#ifdef _MPI
  // Need to traspose data to match PyFR layout - [spts,vars,eles] in row-major format
  std::vector<std::vector<double>> data_p(input->nRanks);
  std::vector<std::vector<int>> iblank_p(input->nRanks);
  if (input->overset && input->rank == 0)
  {
    iblank_p[0].resize(nEles);
    for (int ele = 0; ele < nEles; ele++)
      iblank_p[0][ele] = geo.iblank_cell(ele);
  }

  /* --- Gather all the data onto Rank 0 for writing --- */

  std::vector<int> nEles_p(input->nRanks);
  MPI_Allgather(&nEles, 1, MPI_INT, nEles_p.data(), 1, MPI_INT, geo.myComm);

  for (int p = 1; p < input->nRanks; p++)
  {
    int nEles = nEles_p[p];
    int nElesPad = nEles;
#ifdef _GPU
    nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#endif
    int size = nElesPad * nSpts * nVars;

    if (input->rank == 0)
    {
      data_p[p].resize(size);
      MPI_Status status;
      MPI_Recv(data_p[p].data(), size, MPI_DOUBLE, p, 0, geo.myComm, &status);

      if (input->overset)
      {
        iblank_p[p].resize(nEles);
        MPI_Status status2;
        MPI_Recv(iblank_p[p].data(), nEles, MPI_INT, p, 0, geo.myComm, &status2);
      }
    }
    else
    {
      if (p == input->rank)
      {
        MPI_Send(e->tavg_curr.data(), size, MPI_DOUBLE, 0, 0, geo.myComm);

        if (input->overset)
        {
          MPI_Send(geo.iblank_cell.data(), nEles, MPI_INT, 0, 0, geo.myComm);
        }
      }
    }

    MPI_Barrier(geo.myComm);
  }

  /* --- Write Data to File (on Rank 0) --- */

  if (input->rank == 0)
  {
    H5File file(filename, H5F_ACC_TRUNC);

    // Write out all the data
    DataSet dset = file.createDataSet("config", string_type, dspace);
    dset.write(config, string_type, dspace);
    dset.close();

    dset = file.createDataSet("stats", string_type, dspace);
    dset.write(stats, string_type, dspace);
    dset.close();

    // Write mesh ID
    dset = file.createDataSet("mesh_uuid", string_type, dspace);
    dset.write(geo.mesh_uuid, string_type, dspace);
    dset.close();

    dspace.close();

    std::string sol_prefix = "tavg_";
    sol_prefix += (geo.nDims == 2) ? "quad" : "hex";
    sol_prefix += "_p";
    for (int p = 0; p < input->nRanks; p++)
    {
      nEles = nEles_p[p];
#ifdef _GPU
      nElesPad = (nEles % 16 == 0) ?  nEles : nEles + (16 - nEles % 16);  // Padded for 128-byte alignment
#else
      nElesPad = nEles;
#endif

      hsize_t dimsU[3] = {nSpts, nVars, nElesPad};
      hsize_t dimsF[3] = {nSpts, nVars, nEles};

      // Create a dataspace for the solution, using a hyperslab to ignore padding
      DataSpace dspaceU(3, dimsU);
#ifdef _GPU
      hsize_t count[3] = {1,1,1};
      hsize_t start[3] = {0,0,0};
      hsize_t stride[3] = {1,1,1};
      hsize_t block[3] = {nSpts, nVars, nEles};
      dspaceU.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
#endif

      // Create a dataspace for the actual dataset
      DataSpace dspaceF(3, dimsF);

      std::string solname = sol_prefix + std::to_string(p);
      dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceF);
      if (p == 0)
        dset.write(e->tavg_curr.data(), PredType::NATIVE_DOUBLE, dspaceU);
      else
        dset.write(data_p[p].data(), PredType::NATIVE_DOUBLE, dspaceU);

      dspaceU.close();
      dset.close();

      // Write out iblank as separate DataSet with same naming convention as soln
      if (input->overset)
      {
        hsize_t dims[1] = {nEles};
        DataSpace dspaceI(1, dims);
        std::string iname = "iblank_";
        iname += (geo.nDims == 2) ? "quad" : "hex";
        iname += "_p" + std::to_string(p);
        // NOTE: be aware of C++ 32-bit int vs. HDF5 8-bit int
        dset = file.createDataSet(iname, PredType::NATIVE_INT8, dspaceI);
        dset.write(iblank_p[p].data(), PredType::NATIVE_INT, dspaceI);
        dset.close();
      }
    }
  }
#else
  /* --- Write Data to File --- */

  H5File file(filename, H5F_ACC_TRUNC);

  DataSet dset = file.createDataSet("config", string_type, dspace);
  dset.write(config, string_type, dspace);
  dset.close();

  dset = file.createDataSet("stats", string_type, dspace);
  dset.write(stats, string_type, dspace);
  dset.close();

  // Write mesh ID
  dset = file.createDataSet("mesh_uuid", string_type, dspace);
  dset.write(geo.mesh_uuid, string_type, dspace);
  dset.close();

  dspace.close();

  hsize_t dimsU[3] = {nSpts, nVars, nElesPad};
  hsize_t dimsF[3] = {nSpts, nVars, nEles};

  // Create a dataspace for the solution, using a hyperslab to ignore padding
  DataSpace dspaceU(3, dimsU);
#ifdef _GPU
  hsize_t count[3] = {1,1,1};
  hsize_t start[3] = {0,0,0};
  hsize_t stride[3] = {1,1,1};
  hsize_t block[3] = {nSpts, nVars, nEles};
  dspaceU.selectHyperslab(H5S_SELECT_SET, count, start, stride, block);
#endif

  // Create a dataspace for the actual dataset
  DataSpace dspaceF(3, dimsF);

  std::string solname = "tavg_";
  solname += (geo.nDims == 2) ? "quad" : "hex";
  solname += "_p" + std::to_string(input->rank);
  dset = file.createDataSet(solname, PredType::NATIVE_DOUBLE, dspaceF);
  dset.write(e->tavg_curr.data(), PredType::NATIVE_DOUBLE, dspaceU);
  dset.close();
#endif

}


void FRSolver::write_overset_boundary(const std::string &_prefix)
{
  if (!input->overset) ThrowException("Overset surface export must have overset grid.");

  auto e = elesObjs[0];

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing overset boundary surface data to " << prefix << "..." << std::endl;

  prefix += "_Grid" + std::to_string(input->gridID);

  // Prep the index lists [to grab data from a face of an ele]
  unsigned int nDims = geo.nDims;
  unsigned int nPts1D = order+3;
  unsigned int nPtsFace = nPts1D;
  if (nDims==3) nPtsFace *= nPts1D;
  unsigned int nSubCells = nPts1D - 1;
  if (nDims==3) nSubCells *= nSubCells;
  unsigned int nFacesEle = geo.nFacesPerEleBT[e->etype];

  mdvector<int> index_map({nFacesEle, nPtsFace});

  if (nDims == 2)
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                    // Bottom
      index_map(1,j) = nPts1D-1 + j*nPts1D;        // Right
      index_map(2,j) = nPts1D*nPts1D - 1 + j*(-1); // Top
      index_map(3,j) = 0 + j*nPts1D;               // Left
    }
  }
  else
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                      // Zmin / Bottom
      index_map(1,j) = nPts1D*nPtsFace - 1 + j*(-1); // Zmax / Top
      index_map(2,j) = 0 + j*nPts1D;                 // Xmin / Left
      index_map(3,j) = nPts1D - 1 + j*nPts1D;        // Xmax / Right
    }

    // Ymin / Front
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + j1*nPtsFace;
        index_map(4,J) = J2;
      }
    }

    // Ymax / Back
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + (j1+1)*nPtsFace - nPts1D;
        index_map(5,J) = J2;
      }
    }
  }

  // Write the ParaView file for each Gmsh boundary
  std::stringstream ss;

#ifdef _MPI
  /* Write .pvtu file on rank 0 if running in parallel */
  if (input->rank == 0)
  {
    ss << input->output_prefix << "/";
    ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0');
    ss << iter << ".pvtu";

    std::ofstream f(ss.str());
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    f << "<PPointData>" << std::endl;
    if (input->equation == AdvDiff)
    {
      f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
      f << std::endl;
    }
    else if (input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (e->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (unsigned int n = 0; n < e->nVars; n++)
      {
        f << "<PDataArray type=\"Float32\" Name=\"" << var[n];
        f << "\" format=\"ascii\"/>";
        f << std::endl;
      }
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
      f << std::endl;
    }

    if (input->motion)
    {
      f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" format=\"ascii\"/>";
      f << std::endl;
    }

    f << "</PPointData>" << std::endl;
    f << "<PPoints>" << std::endl;
    f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\"/>" << std::endl;
    f << "</PPoints>" << std::endl;

    for (unsigned int n = 0; n < input->nRanks; n++)
    {
      ss.str("");
      ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
      ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
      f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
    }

    f << "</PUnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;

    f.close();
  }
#endif

  ss.str("");
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_OVERSET_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".vtu";
#endif

  auto outputfile = ss.str();

  /* Write parition solution to file in .vtu format */
  std::ofstream f(outputfile);

  /* Write header */
  f << "<?xml version=\"1.0\"?>" << std::endl;
  f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
  f << "byte_order=\"LittleEndian\" ";
  f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

  /* Write comments for solution order, iteration number and flowtime */
  f << "<!-- ORDER " << input->order << " -->" << std::endl;
  f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
  f << "<!-- ITER " << iter << " -->" << std::endl;

  // Load list of eles (and sub-ele face indices) from which to load data
  std::vector<int> eleList, indList;
  unsigned int nFaces = 0;

  for (unsigned int ff = 0; ff < geo.nFaces; ff++)
  {
    if (geo.iblank_face(ff) == FRINGE)
    {
      int ic1 = geo.face2eles(ff,0);
      int ic2 = geo.face2eles(ff,1);
      if (geo.iblank_cell(ic1) == NORMAL)
      {
        eleList.push_back(ic1);
      }
      else if (ic2 > 0 && geo.iblank_cell(ic2) == NORMAL)
      {
        eleList.push_back(ic2);
      }

      for (int i = 0; i < nFacesEle; i++)
      {
        if (geo.ele2face(eleList.back(),i) == ff)
        {
          indList.push_back(i);
          break;
        }
      }

      nFaces++;
    }
  }

  // Write data to file

  f << "<UnstructuredGrid>" << std::endl;
  f << "<Piece NumberOfPoints=\"" << nPtsFace * nFaces << "\" ";
  f << "NumberOfCells=\"" << nSubCells * nFaces << "\">";
  f << std::endl;


  /* Write plot point coordinates */
  f << "<Points>" << std::endl;
  f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
  f << "format=\"ascii\">" << std::endl;

  if (e->nDims == 2)
  {
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << e->coord_ppts(ppt, 0, ele) << " ";
        f << e->coord_ppts(ppt, 1, ele) << " ";
        f << 0.0 << std::endl;
      }
    }
  }
  else
  {
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << e->coord_ppts(ppt, 0, ele) << " ";
        f << e->coord_ppts(ppt, 1, ele) << " ";
        f << e->coord_ppts(ppt, 2, ele) << std::endl;
      }
    }
  }

  f << "</DataArray>" << std::endl;
  f << "</Points>" << std::endl;

  /* Write cell information */
  f << "<Cells>" << std::endl;
  f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
  f << "format=\"ascii\">"<< std::endl;

  if (nDims == 2)
  {
    int count = 0;
    for (int face = 0; face < nFaces; face++)
    {
      for (int j = 0; j < nPts1D-1; j++)
      {
        f << count + j << " ";
        f << count + j + 1 << " ";
        f << std::endl;
      }
      count += nPtsFace;
    }
  }
  else
  {
    int count = 0;
    for (int face = 0; face < nFaces; face++)
    {
      for (int j = 0; j < nPts1D-1; j++)
      {
        for (int i = 0; i < nPts1D-1; i++)
        {
          f << count + j*nPts1D     + i   << " ";
          f << count + j*nPts1D     + i+1 << " ";
          f << count + (j+1)*nPts1D + i+1 << " ";
          f << count + (j+1)*nPts1D + i   << " ";
          f << std::endl;
        }
      }
      count += nPtsFace;
    }
  }

  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
  f << "format=\"ascii\">"<< std::endl;
  int nvPerFace = (nDims == 2) ? 2 : 4;
  int offset = nvPerFace;
  for (int face = 0; face < nFaces; face++)
  {
    for (int subele = 0; subele < nSubCells; subele++)
    {
      f << offset << " ";
      offset += nvPerFace;
    }
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;

  f << "<DataArray type=\"UInt8\" Name=\"types\" ";
  f << "format=\"ascii\">"<< std::endl;
  int nCells = nSubCells * nFaces;
  if (nDims == 2)
  {
    for (int cell = 0; cell < nCells; cell++)
      f << 3 << " ";
  }
  else
  {
    for (int cell = 0; cell < nCells; cell++)
      f << 9 << " ";
  }
  f << std::endl;
  f << "</DataArray>" << std::endl;
  f << "</Cells>" << std::endl;

  /* Write solution information */
  f << "<PointData>" << std::endl;

  if (input->equation == AdvDiff)
  {
    f << "<DataArray type=\"Float32\" Name=\"u\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        f << std::scientific << std::setprecision(16) << e->U_ppts(ppt, 0, ele);
        f  << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }
  else if(input->equation == EulerNS)
  {
    std::vector<std::string> var;
    if (e->nDims == 2)
      var = {"rho", "xmom", "ymom", "energy"};
    else
      var = {"rho", "xmom", "ymom", "zmom", "energy"};

    for (int n = 0; n < e->nVars; n++)
    {
      f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
      f << "format=\"ascii\">"<< std::endl;

      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << std::scientific << std::setprecision(16);
          f << e->U_ppts(ppt, n, ele);
          f << " ";
        }

        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
  }

  if (input->filt_on && input->sen_write)
  {
    f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
    for (int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      for (int pt = 0; pt < nPtsFace; pt++)
      {
        f << filt.sensor[e->etype](ele) << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  if (input->motion)
  {
    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
    f << "format=\"ascii\">"<< std::endl;
    for (unsigned int face = 0; face < nFaces; face++)
    {
      int ele = eleList[face];
      int ind = indList[face];
      for (unsigned int pt = 0; pt < nPtsFace; pt++)
      {
        int ppt = index_map(ind,pt);
        for (unsigned int dim = 0; dim < e->nDims; dim++)
        {
          f << std::scientific << std::setprecision(16);
          f << e->grid_vel_ppts(ppt, dim, ele);
          f  << " ";
        }
        if (e->nDims == 2) f << 0.0 << " ";
      }
      f << std::endl;
    }
    f << "</DataArray>" << std::endl;
  }

  f << "</PointData>" << std::endl;
  f << "</Piece>" << std::endl;
  f << "</UnstructuredGrid>" << std::endl;
  f << "</VTKFile>" << std::endl;
  f.close();
}


void FRSolver::write_surfaces(const std::string &_prefix)
{
  if (geo.ele_set.count(TET))
    ThrowException("Surface write not implemented for triangular faces.");

  if (input->implicit_method)
    ThrowException("Surface write not implemented for implicit methods.");

  auto e = elesObjs[0];

#ifdef _GPU
  e->U_spts = e->U_spts_d;
#endif

  std::string prefix = _prefix;

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  if (input->gridID == 0 && input->rank == 0)
    std::cout << "Writing surface data to " << prefix << "..." << std::endl;

  if (input->overset) prefix += "_Grid" + std::to_string(input->gridID);

  // Prep the index lists [to grab data from a face of an ele]
  unsigned int nDims = geo.nDims;
  unsigned int nPts1D = order+1;
  unsigned int nPtsFace = nPts1D;
  if (nDims==3) nPtsFace *= nPts1D;
  unsigned int nSubCells = nPts1D - 1;
  if (nDims==3) nSubCells *= nSubCells;
  unsigned int nFacesEle = geo.nFacesPerEleBT[e->etype];

  mdvector<int> index_map({nFacesEle, nPtsFace});

  if (nDims == 2)
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                    // Bottom
      index_map(1,j) = nPts1D-1 + j*nPts1D;        // Right
      index_map(2,j) = nPts1D*nPts1D - 1 + j*(-1); // Top
      index_map(3,j) = 0 + j*nPts1D;               // Left
    }
  }
  else
  {
    for (int j = 0; j < nPtsFace; j++)
    {
      index_map(0,j) = 0 + j*1;                      // Zmin / Bottom
      index_map(1,j) = nPts1D*nPtsFace - 1 + j*(-1); // Zmax / Top
      index_map(2,j) = 0 + j*nPts1D;                 // Xmin / Left
      index_map(3,j) = nPts1D - 1 + j*nPts1D;        // Xmax / Right
    }

    // Ymin / Front
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + j1*nPtsFace;
        index_map(4,J) = J2;
      }
    }

    // Ymax / Back
    for (int j1 = 0; j1 < nPts1D; j1++) {
      for (int j2 = 0; j2 < nPts1D; j2++) {
        int J  = j2 + j1*nPts1D;
        int J2 = j2 + (j1+1)*nPtsFace - nPts1D;
        index_map(5,J) = J2;
      }
    }
  }

  // General Solution Preprocessing Stuff

  if (input->motion)
  {
    e->update_plot_point_coords();
#ifdef _GPU
    e->grid_vel_nodes = e->grid_vel_nodes_d;
#endif
    e->get_grid_velocity_ppts();
  }

  /* Extrapolate solution to plot points */
  auto &A = e->oppE_ppts(0, 0);
  auto &B = e->U_spts(0, 0, 0);
  auto &C = e->U_ppts(0, 0, 0);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nPpts,
              e->nElesPad * e->nVars, e->nSpts, 1.0, &A, e->nSpts, &B,
              e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);

  /* Apply squeezing if needed */
  if (input->squeeze)
  {
    e->compute_Uavg();

#ifdef _GPU
    e->Uavg = e->Uavg_d;
#endif

    e->poly_squeeze_ppts();
  }

#ifdef _GPU
  if (input->filt_on && input->sen_write)
    filt.sensor[e->etype] = filt.sensor_d[e->etype];
#endif

  // Write the ParaView file for each Gmsh boundary
  for (int bnd = 0; bnd < geo.nBounds; bnd++)
  {
    std::stringstream ss;

#ifdef _MPI
    /* Write .pvtu file on rank 0 if running in parallel */
    if (input->rank == 0)
    {
      ss << input->output_prefix << "/";
      ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0');
      ss << iter << ".pvtu";

      std::ofstream f(ss.str());
      f << "<?xml version=\"1.0\"?>" << std::endl;
      f << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" ";
      f << "byte_order=\"LittleEndian\" ";
      f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

      f << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
      f << "<PPointData>" << std::endl;
      if (input->equation == AdvDiff)
      {
        f << "<PDataArray type=\"Float32\" Name=\"u\" format=\"ascii\"/>";
        f << std::endl;
      }
      else if (input->equation == EulerNS)
      {
        std::vector<std::string> var;
        if (e->nDims == 2)
          var = {"rho", "xmom", "ymom", "energy"};
        else
          var = {"rho", "xmom", "ymom", "zmom", "energy"};

        for (unsigned int n = 0; n < e->nVars; n++)
        {
          f << "<PDataArray type=\"Float32\" Name=\"" << var[n];
          f << "\" format=\"ascii\"/>";
          f << std::endl;
        }
      }

      if (input->filt_on && input->sen_write)
      {
        f << "<PDataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\"/>";
        f << std::endl;
      }

      if (input->motion)
      {
        f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" format=\"ascii\"/>";
        f << std::endl;
      }

      f << "</PPointData>" << std::endl;
      f << "<PPoints>" << std::endl;
      f << "<PDataArray type=\"Float32\" NumberOfComponents=\"3\" ";
      f << "format=\"ascii\"/>" << std::endl;
      f << "</PPoints>" << std::endl;

      for (unsigned int n = 0; n < input->nRanks; n++)
      {
        ss.str("");
        ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
        ss << "_" << std::setw(3) << std::setfill('0') << n << ".vtu";
        f << "<Piece Source=\"" << ss.str() << "\"/>" << std::endl;
      }

      f << "</PUnstructuredGrid>" << std::endl;
      f << "</VTKFile>" << std::endl;

      f.close();
    }
#endif

    ss.str("");
#ifdef _MPI
    ss << input->output_prefix << "/";
    ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
    ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".vtu";
#else
    ss << input->output_prefix << "/";
    ss << prefix << "_" << geo.bcNames[bnd] << "_" << std::setw(9) << std::setfill('0') << iter;
    ss << ".vtu";
#endif

    auto outputfile = ss.str();

    /* Write parition solution to file in .vtu format */
    std::ofstream f(outputfile);

    /* Write header */
    f << "<?xml version=\"1.0\"?>" << std::endl;
    f << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" ";
    f << "byte_order=\"LittleEndian\" ";
    f << "compressor=\"vtkZLibDataCompressor\">" << std::endl;

    /* Write comments for solution order, iteration number and flowtime */
    f << "<!-- ORDER " << input->order << " -->" << std::endl;
    f << "<!-- TIME " << std::scientific << std::setprecision(16) << flow_time << " -->" << std::endl;
    f << "<!-- ITER " << iter << " -->" << std::endl;

    // Load list of eles (and sub-ele face indices) from which to load data
    std::vector<int> eleList, indList;
    unsigned int nFaces = 0;

    for (auto &ff : geo.boundFaces[bnd])
    {
      // Load data from each face on boundary
      int ele = geo.face2eles(ff,0);

      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

      int j = -1;
      for (int i = 0; i < nFacesEle; i++)
      {
        if (geo.ele2face(ele, i) == ff)
        {
          j = i;
          break;
        }
      }
      if (j < 0) ThrowException("write_surfaces: Error with ele/face connectivity!");

      eleList.push_back(ele);
      indList.push_back(j);
      nFaces++;
    }

    // Write data to file

    f << "<UnstructuredGrid>" << std::endl;
    f << "<Piece NumberOfPoints=\"" << nPtsFace * nFaces << "\" ";
    f << "NumberOfCells=\"" << nSubCells * nFaces << "\">";
    f << std::endl;


    /* Write plot point coordinates */
    f << "<Points>" << std::endl;
    f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" ";
    f << "format=\"ascii\">" << std::endl;

    if (e->nDims == 2)
    {
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << e->coord_ppts(ppt, 0, ele) << " ";
          f << e->coord_ppts(ppt, 1, ele) << " ";
          f << 0.0 << std::endl;
        }
      }
    }
    else
    {
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << e->coord_ppts(ppt, 0, ele) << " ";
          f << e->coord_ppts(ppt, 1, ele) << " ";
          f << e->coord_ppts(ppt, 2, ele) << std::endl;
        }
      }
    }

    f << "</DataArray>" << std::endl;
    f << "</Points>" << std::endl;

    /* Write cell information */
    f << "<Cells>" << std::endl;
    f << "<DataArray type=\"Int32\" Name=\"connectivity\" ";
    f << "format=\"ascii\">"<< std::endl;

    if (nDims == 2)
    {
      int count = 0;
      for (int face = 0; face < nFaces; face++)
      {
        for (int j = 0; j < nPts1D-1; j++)
        {
          f << count + j << " ";
          f << count + j + 1 << " ";
          f << std::endl;
        }
        count += nPtsFace;
      }
    }
    else
    {
      int count = 0;
      for (int face = 0; face < nFaces; face++)
      {
        for (int j = 0; j < nPts1D-1; j++)
        {
          for (int i = 0; i < nPts1D-1; i++)
          {
            f << count + j*nPts1D     + i   << " ";
            f << count + j*nPts1D     + i+1 << " ";
            f << count + (j+1)*nPts1D + i+1 << " ";
            f << count + (j+1)*nPts1D + i   << " ";
            f << std::endl;
          }
        }
        count += nPtsFace;
      }
    }

    f << "</DataArray>" << std::endl;

    f << "<DataArray type=\"Int32\" Name=\"offsets\" ";
    f << "format=\"ascii\">"<< std::endl;
    int nvPerFace = (nDims == 2) ? 2 : 4;
    int offset = nvPerFace;
    for (int face = 0; face < nFaces; face++)
    {
      for (int subele = 0; subele < nSubCells; subele++)
      {
        f << offset << " ";
        offset += nvPerFace;
      }
    }
    f << std::endl;
    f << "</DataArray>" << std::endl;

    f << "<DataArray type=\"UInt8\" Name=\"types\" ";
    f << "format=\"ascii\">"<< std::endl;
    int nCells = nSubCells * nFaces;
    if (nDims == 2)
    {
      for (int cell = 0; cell < nCells; cell++)
        f << 3 << " ";
    }
    else
    {
      for (int cell = 0; cell < nCells; cell++)
        f << 9 << " ";
    }
    f << std::endl;
    f << "</DataArray>" << std::endl;
    f << "</Cells>" << std::endl;

    /* Write solution information */
    f << "<PointData>" << std::endl;

    if (input->equation == AdvDiff)
    {
      f << "<DataArray type=\"Float32\" Name=\"u\" ";
      f << "format=\"ascii\">"<< std::endl;
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          f << std::scientific << std::setprecision(16) << e->U_ppts(ppt, 0, ele);
          f  << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }
    else if(input->equation == EulerNS)
    {
      std::vector<std::string> var;
      if (e->nDims == 2)
        var = {"rho", "xmom", "ymom", "energy"};
      else
        var = {"rho", "xmom", "ymom", "zmom", "energy"};

      for (int n = 0; n < e->nVars; n++)
      {
        f << "<DataArray type=\"Float32\" Name=\"" << var[n] << "\" ";
        f << "format=\"ascii\">"<< std::endl;

        for (int face = 0; face < nFaces; face++)
        {
          int ele = eleList[face];
          int ind = indList[face];
          for (int pt = 0; pt < nPtsFace; pt++)
          {
            int ppt = index_map(ind,pt);
            f << std::scientific << std::setprecision(16);
            f << e->U_ppts(ppt, n, ele);
            f << " ";
          }

          f << std::endl;
        }
        f << "</DataArray>" << std::endl;
      }
    }

    if (input->filt_on && input->sen_write)
    {
      f << "<DataArray type=\"Float32\" Name=\"sensor\" format=\"ascii\">"<< std::endl;
      for (int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        for (int pt = 0; pt < nPtsFace; pt++)
        {
          f << filt.sensor[e->etype](ele) << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }

    if (input->motion)
    {
      f << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" Name=\"grid_velocity\" ";
      f << "format=\"ascii\">"<< std::endl;
      for (unsigned int face = 0; face < nFaces; face++)
      {
        int ele = eleList[face];
        int ind = indList[face];
        for (unsigned int pt = 0; pt < nPtsFace; pt++)
        {
          int ppt = index_map(ind,pt);
          for (unsigned int dim = 0; dim < e->nDims; dim++)
          {
            f << std::scientific << std::setprecision(16);
            f << e->grid_vel_ppts(ppt, dim, ele);
            f  << " ";
          }
          if (e->nDims == 2) f << 0.0 << " ";
        }
        f << std::endl;
      }
      f << "</DataArray>" << std::endl;
    }

    f << "</PointData>" << std::endl;
    f << "</Piece>" << std::endl;
    f << "</UnstructuredGrid>" << std::endl;
    f << "</VTKFile>" << std::endl;
    f.close();
  }

  if (input->overset && input->plot_overset)
  {
    write_overset_boundary(_prefix);
  }
}

void FRSolver::write_LHS(const std::string &_prefix)
{
#if !defined (_MPI)
  auto e = elesObjs[0];
  if (input->implicit_method)
  {
    if (input->rank == 0) std::cout << "Writing LHS to file..." << std::endl;

    unsigned int iter = current_iter;
    if (input->p_multi)
      iter = iter / input->mg_steps[0];

    std::string prefix = _prefix;
    std::stringstream ss;
    ss << input->output_prefix << "/";
    ss << prefix << "_LHS_" << std::setw(9) << std::setfill('0');
    ss << iter << ".dat";

    H5File file(ss.str(), H5F_ACC_TRUNC);
    unsigned int N = e->nSpts * e->nVars;
    hsize_t dims[3] = {e->nEles, N, N};
    DataSpace dspaceU(3, dims);

    std::string name = "LHS_" + std::to_string(iter);
    DataSet dset = file.createDataSet(name, PredType::NATIVE_DOUBLE, dspaceU);
    dset.write(e->LHS.data(), PredType::NATIVE_DOUBLE, dspaceU);

    dspaceU.close();
    dset.close();
    file.close();
  }
#endif
}

void FRSolver::write_RHS(const std::string &_prefix)
{
#if !defined (_MPI)
  auto e = elesObjs[0];
  if (input->implicit_method)
  {
    if (input->rank == 0) std::cout << "Writing RHS to file..." << std::endl;
#ifdef _GPU
    for (auto e : elesObjs)
      e->RHS = e->RHS_d;
#endif

    unsigned int iter = current_iter;
    if (input->p_multi)
      iter = iter / input->mg_steps[0];

    std::string prefix = _prefix;
    std::stringstream ss;
    ss << input->output_prefix << "/";
    ss << prefix << "_RHS_" << std::setw(9) << std::setfill('0');
    ss << iter << ".dat";

    H5File file(ss.str(), H5F_ACC_TRUNC);
    unsigned int N = e->nSpts * e->nVars;
    hsize_t dims[2] = {e->nEles, N};
    DataSpace dspaceU(2, dims);

    std::string name = "RHS_" + std::to_string(iter);
    DataSet dset = file.createDataSet(name, PredType::NATIVE_DOUBLE, dspaceU);
    dset.write(e->RHS.data(), PredType::NATIVE_DOUBLE, dspaceU);

    dspaceU.close();
    dset.close();
    file.close();
  }
#endif
}

void FRSolver::report_residuals(std::ofstream &f, std::chrono::high_resolution_clock::time_point t1)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If running on GPU, copy out divergence */
#ifdef _GPU
  for (auto e : elesObjs)
  {
    e->divF_spts = e->divF_spts_d;
    e->dt = e->dt_d;
  }
#endif

  std::vector<double> res(elesObjs[0]->nVars,0.0);
  for (auto e : elesObjs)
  {
    for (unsigned int spt = 0; spt < e->nSpts; spt++)
    {
      for (unsigned int n = 0; n < e->nVars; n++)
      {
        for (unsigned int ele = 0; ele < e->nEles; ele++)
        {
          if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;

          if (input->res_type == 0)
            res[n] = std::max(res[n], std::abs(e->divF_spts(0, spt, n, ele)
                                               / e->jaco_det_spts(spt, ele)));

          else if (input->res_type == 1)
            res[n] += std::abs(e->divF_spts(0, spt, n, ele)
                               / e->jaco_det_spts(spt, ele));

          else if (input->res_type == 2)
            res[n] += e->divF_spts(0, spt, n, ele) * e->divF_spts(0, spt, n, ele)
                / (e->jaco_det_spts(spt, ele) * e->jaco_det_spts(spt, ele));

          if (std::isnan(res[n]))
          {
            std::cout << "NaN residual encountered at ele " << ele << ", spt " << spt << ", var " << n << std::endl;
            for (int i = 0; i < std::min(8,(int)e->nNodes); i++)
            {
              if (geo.nDims == 3)
                printf("%f %f %f\n",e->nodes(i,0,ele),e->nodes(i,1,ele),e->nodes(i,2,ele));
              else
                printf("%f %f\n",e->nodes(i,0,ele),e->nodes(i,1,ele));
            }
            ThrowException("NaN in residual");
          }
        }
      }
    }
  }

  unsigned int nDoF = 0;
  for (auto e : elesObjs)
    nDoF += (e->nSpts * e->nEles);

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
#endif

  double minDT = INFINITY; double maxDT = 0.0; 

  if (input->dt_type == 2)
  {
    for (auto e : elesObjs)
    {
      minDT = std::min(minDT, *std::min_element(e->dt.data(), e->dt.data() + e->nEles));
      maxDT = std::max(maxDT, *std::max_element(e->dt.data(), e->dt.data() + e->nEles));
    }

#ifdef _MPI
    if (input->rank == 0)
    {
      MPI_Reduce(MPI_IN_PLACE, &minDT, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(MPI_IN_PLACE, &maxDT, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
    else
    {
      MPI_Reduce(&minDT, &minDT, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(&maxDT, &maxDT, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
#endif
  }

  /* Print residual to terminal (normalized by number of solution points) */
  if (input->rank == 0) 
  {
    if (input->res_type == 2)
    {
      for (auto &val : res)  
        val = std::sqrt(val);
    }

    if (input->overset)
      std::cout << "G" << std::setw(4) << std::left << input->gridID;

    std::cout << std::setw(6) << std::left << iter << " ";

    for (auto val : res)
      std::cout << std::scientific << std::setprecision(6) << std::setw(15) << std::left << val / nDoF << " ";

    if (input->dt_type == 2)
    {

      std::cout << "dt: " <<  minDT << " (min) ";
      std::cout << maxDT << " (max)";
    }
    else
    {
      std::cout << "dt: " << elesObjs[0]->dt(0);
    }

    std::cout << std::endl;
    
    /* Write to history file */
    auto t2 = std::chrono::high_resolution_clock::now();
    auto current_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);

    f << iter << " " << std::scientific << flow_time << " " << current_runtime.count() << " ";

    for (auto val : res)
      f << val / nDoF << " ";
    f << std::endl;

    /* Store maximum residual */
    res_max = res[0] / nDoF;
  }

#ifdef _MPI
  /* Broadcast maximum residual */
  MPI_Bcast(&res_max, 1, MPI_DOUBLE, 0, myComm);
#endif
}

void FRSolver::report_RHS(unsigned int stage, unsigned int iterNM, unsigned int iterBM)
{
  /* Recompute RHS without pseudo time step */
#ifdef _CPU
  for (auto e : elesObjs)
    for (unsigned int ele = 0; ele < e->nEles; ele++)
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          e->RHS(ele, var, spt) = -rk_alpha(stage,0) * e->divF_spts(0, spt, var, ele) / e->jaco_det_spts(spt, ele);
          for (unsigned int s = 1; s <= stage; s++)
            e->RHS(ele, var, spt) -= rk_alpha(stage, s) * e->divF_spts(s, spt, var, ele) / e->jaco_det_spts(spt, ele);
        }

  if (!input->implicit_steady)
  {
    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        double dt = (input->dt_type != 2) ? e->dt(0) : e->dt(ele);
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            e->RHS(ele, var, spt) *= dt;
      }

    for (auto e : elesObjs)
      for (unsigned int ele = 0; ele < e->nEles; ele++)
        for (unsigned int var = 0; var < e->nVars; var++)
          for (unsigned int spt = 0; spt < e->nSpts; spt++)
            e->RHS(ele, var, spt) -= (e->U_spts(spt, var, ele) - e->U_ini(spt, var, ele));
  }
#endif
#ifdef _GPU
  for (auto e : elesObjs)
    compute_RHS_wrapper(e->U_spts_d, e->U_iniNM_d, e->U_ini_d, e->divF_spts_d, e->jaco_det_spts_d, 
        e->dt_d, e->dtau_d, rk_alpha_d, e->RHS_d, dtau_ratio, input->implicit_steady, 
        0, input->remove_deltaU, input->dt_type, input->dtau_type, e->nSpts, 
        e->nEles, e->nVars, stage);
  check_error();
#endif

  /* If running on GPU, copy out RHS and dtau */
#ifdef _GPU
  for (auto e : elesObjs)
    e->RHS = e->RHS_d;

  if (input->pseudo_time)
    for (auto e : elesObjs)
      e->dtau = e->dtau_d;
#endif

  std::vector<double> res(elesObjs[0]->nVars, 0.0);
  for (auto e : elesObjs)
    for (unsigned int ele = 0; ele < e->nEles; ele++)
      for (unsigned int var = 0; var < e->nVars; var++)
        for (unsigned int spt = 0; spt < e->nSpts; spt++)
        {
          if (input->res_type == 0)
            res[var] = std::max(res[var], std::abs(e->RHS(ele, var, spt)));

          else if (input->res_type == 1)
            res[var] += std::abs(e->RHS(ele, var, spt));

          else if (input->res_type == 2)
            res[var] += (e->RHS(ele, var, spt) * e->RHS(ele, var, spt));

          if (std::isnan(res[var]))
          {
            std::cout << "NaN residual encountered at ele " << ele << ", spt " << spt << ", var " << var << std::endl;
            for (int i = 0; i < std::min(8,(int)e->nNodes); i++)
            {
              if (geo.nDims == 3)
                printf("%f %f %f\n",e->nodes(i,0,ele),e->nodes(i,1,ele),e->nodes(i,2,ele));
              else
                printf("%f %f\n",e->nodes(i,0,ele),e->nodes(i,1,ele));
            }
            ThrowException("NaN in residual");
          }
        }

  unsigned int nDoF = 0;
  for (auto e : elesObjs)
    nDoF += (e->nSpts * e->nEles);

#ifdef _MPI
  MPI_Op oper = MPI_SUM;
  if (input->res_type == 0)
    oper = MPI_MAX;

  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(res.data(), res.data(), elesObjs[0]->nVars, MPI_DOUBLE, oper, 0, myComm);
    MPI_Reduce(&nDoF, &nDoF, 1, MPI_INT, MPI_SUM, 0, myComm);
  }
#endif

  double minDTAU = INFINITY; double maxDTAU = 0.0; 
  if (input->pseudo_time && input->dtau_type == 2)
  {
    for (auto e : elesObjs)
    {
      minDTAU = std::min(minDTAU, *std::min_element(e->dtau.data(), e->dtau.data() + e->nEles));
      maxDTAU = std::max(maxDTAU, *std::max_element(e->dtau.data(), e->dtau.data() + e->nEles));
    }

#ifdef _MPI
    if (input->rank == 0)
    {
      MPI_Reduce(MPI_IN_PLACE, &minDTAU, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(MPI_IN_PLACE, &maxDTAU, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
    else
    {
      MPI_Reduce(&minDTAU, &minDTAU, 1, MPI_DOUBLE, MPI_MIN, 0, myComm);
      MPI_Reduce(&maxDTAU, &maxDTAU, 1, MPI_DOUBLE, MPI_MAX, 0, myComm);
    }
#endif
  }

  /* Print residual to terminal (normalized by number of solution points) */
  if (input->rank == 0) 
  {
    if (input->res_type == 2)
      std::transform(res.begin(), res.end(), res.begin(), (double(*)(double)) std::sqrt);

    if (stage == startStage && iterNM * iterBM == 1)
    {
      std::cout << std::endl;
      if (!input->implicit_steady)
        std::cout << "Stage ";
      std::cout << "IterNM IterBM ";
      if (input->equation == AdvDiff)
        std::cout << "Res[U]" << std::endl;
      else if (input->equation == EulerNS)
      {
        std::cout << "Res[Rho]        Res[xMom]       Res[yMom]       ";
        if (geo.nDims == 3)
          std::cout << "Res[zMom]       ";
        std::cout << "Res[Energy]" << std::endl;
      }
    }

    if (!input->implicit_steady)
      std::cout << std::setw(5) << std::left << stage << " ";
    std::cout << std::setw(6) << std::left << iterNM << " ";
    std::cout << std::setw(6) << std::left << iterBM << " ";
    for (auto val : res)
      std::cout << std::scientific << std::setprecision(6) << std::setw(15) << std::left << val / nDoF << " ";

    if (input->pseudo_time)
    {
      if (input->dtau_type == 2)
      {
        std::cout << "dtau: " << dtau_ratio * minDTAU << " (min) ";
        std::cout << dtau_ratio * maxDTAU << " (max)";
      }
      else
        std::cout << "dtau: " << dtau_ratio * elesObjs[0]->dtau(0);
    }
    std::cout << std::endl;
    
    /* Write to convergence file */
    if (!input->implicit_steady)
      conv_file << current_iter+1 << " " << stage << " ";
    auto timer2 = std::chrono::high_resolution_clock::now();
    auto current_runtime = std::chrono::duration_cast<std::chrono::duration<double>>(timer2-this->conv_timer);
    conv_file << iterNM << " " << iterBM << " " << std::scientific << current_runtime.count() << " ";
    for (auto val : res)
      conv_file << val / nDoF << " ";
    conv_file << std::endl;

    /* Store maximum residual */
    res_max = res[0] / nDoF;
  }

#ifdef _MPI
  /* Broadcast maximum residual */
  MPI_Bcast(&res_max, 1, MPI_DOUBLE, 0, myComm);
#endif
}

void FRSolver::report_forces(std::ofstream &f)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If using GPU, copy out solution, gradient and pressure */
#ifdef _GPU
  for (auto e : elesObjs)
  {
    e->U_fpts = e->U_fpts_d;
    if (input->viscous)
      e->dU_fpts = e->dU_fpts_d;
  }
  faces->P = faces->P_d;

  if (input->motion)
    faces->norm = faces->norm_d;
#endif

  std::string prefix = input->output_prefix;
  if (input->overset) prefix += "_Grid" + std::to_string(input->gridID);

  std::stringstream ss;
#ifdef _MPI
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << "_" << std::setw(3) << std::setfill('0') << input->rank << ".cp";
#else
  ss << input->output_prefix << "/";
  ss << prefix << "_" << std::setw(9) << std::setfill('0') << iter;
  ss << ".cp";
#endif

  auto cpfile = ss.str();
  std::ofstream g(cpfile);

  std::array<double, 3> force_conv = {0,0,0};
  std::array<double, 3> force_visc = {0,0,0};
  compute_forces(force_conv, force_visc, &g);

  /* Convert dimensional forces into non-dimensional coefficients
   * NOTE: Reference area assumed to be 1. Divide by A for 'true' Cl, Cd */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < geo.nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  for (int i = 0; i < 3; i++)
  {
    force_conv[i] *= fac;
    force_visc[i] *= fac;
  }

  /* Compute lift, drag, and side force coefficients */
  double CL_conv, CD_conv, CL_visc, CD_visc, CQ_conv, CQ_visc;

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, force_conv.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, force_visc.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(force_conv.data(), force_conv.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(force_visc.data(), force_visc.data(), geo.nDims, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
#endif

  /* Get angle of attack (and sideslip) */
  double alpha = std::atan2(input->V_fs(1), input->V_fs(0));
  double cosa = std::cos(alpha);
  double sina = std::sin(alpha);

  double beta = 0.0;
  if (geo.nDims == 3)
    beta = std::atan2(input->V_fs(2), input->V_fs(0)*cosa + input->V_fs(1)*sina);
  double cosb = std::cos(beta);
  double sinb = std::sin(beta);

  if (input->rank == 0)
  {
    if (geo.nDims == 2)
    {
      CL_conv = -force_conv[0] * sina + force_conv[1] * cosa;
      CD_conv = force_conv[0] * cosa + force_conv[1] * sina;
      CL_visc = -force_visc[0] * sina + force_visc[1] * cosa;
      CD_visc = force_visc[0] * cosa + force_visc[1] * sina;
    }
    else if (geo.nDims == 3)
    {
      CD_conv = (force_conv[0] * cosa - force_conv[1] * sina) * cosb
          + force_conv[2] * sina * sinb;
      CD_visc = (force_visc[0] * cosa - force_visc[1] * sina) * cosb
          + force_visc[2] * sina * sinb;

      CL_conv = -force_conv[0] * sina + force_conv[1] * cosa;
      CL_visc = -force_visc[0] * sina + force_visc[1] * cosa;

      CQ_conv = -(force_conv[0] * cosa + force_conv[1] * sina) * sinb
          + force_conv[2] * cosb;
      CQ_visc = -(force_visc[0] * cosa + force_visc[1] * sina) * sinb
          + force_visc[2] * cosb;
    }

    std::cout << "CL_conv = " << CL_conv << " CD_conv = " << CD_conv;

    f << iter << " " << std::scientific << std::setprecision(16) << flow_time << " ";
    f << CL_conv << " " << CD_conv;

    if (geo.nDims == 3)
    {
      std::cout << " CQ_conv = " << CQ_conv;
      f << " " << CQ_conv;
    }

    if (input->viscous)
    {
      std::cout << " CL_visc = " << CL_visc << " CD_visc = " << CD_visc;
      f << " " << CL_visc << " " << CD_visc;
      if (geo.nDims == 3)
      {
        std::cout << " CQ_visc = " << CQ_visc;
        f << " " << CQ_visc;
      }
    }

    std::cout << std::endl;
    f << std::endl;
  }
}

void FRSolver::report_error(std::ofstream &f)
{
  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  /* If using GPU, copy out solution */
#ifdef _GPU
  for (auto e : elesObjs)
  {
    e->U_spts = e->U_spts_d;
    if (input->viscous)
      e->dU_spts = e->dU_spts_d;
  }
#endif

  std::vector<double> l2_error(2,0.0);
  double vol = 0;

  for (auto e : elesObjs)
  {
    /* Extrapolate solution to quadrature points */
    auto &A = e->oppE_qpts(0, 0);
    auto &B = e->U_spts(0, 0, 0);
    auto &C = e->U_qpts(0, 0, 0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nQpts, 
        e->nElesPad * e->nVars, e->nSpts, 1.0, &A, e->nSpts, &B,
        e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);

    /* Extrapolate derivatives to quadrature points */
    if (input->viscous)
    {
      for (unsigned int dim = 0; dim < e->nDims; dim++)
      {
        auto &A = e->oppE_qpts(0, 0);
        auto &B = e->dU_spts(dim, 0, 0, 0);
        auto &C = e->dU_qpts(dim, 0, 0, 0);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e->nQpts,
                    e->nElesPad * e->nVars, e->nSpts, 1.0, &A, e->nSpts, &B,
                    e->nElesPad * e->nVars, 0.0, &C, e->nElesPad * e->nVars);

      }
    }

    unsigned int n = input->err_field;
    std::vector<double> dU_true(geo.nDims, 0.0), dU_error(geo.nDims, 0.0);

    for (unsigned int ele = 0; ele < e->nEles; ele++)
    {
      if (input->overset && geo.iblank_cell(ele) != NORMAL) continue;
      for (unsigned int qpt = 0; qpt < e->nQpts; qpt++)
      {
        double U_true = 0.0;

        double x = e->coord_qpts(qpt, 0, ele);
        double y = e->coord_qpts(qpt, 1, ele);
        double z = (geo.nDims == 2) ? 0.0 : e->coord_qpts(qpt, 2, ele);

        /* Compute true solution and derivatives */
        U_true = compute_U_true(x, y, z, flow_time, n, input);

        if (input->viscous)
        {
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            dU_true[dim] = compute_dU_true(x, y, z, flow_time, n, dim, input);
        }

        /* Compute errors */
        double U_error;
        U_error = U_true - e->U_qpts(qpt, n, ele);
        if (input->viscous)
        {
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            dU_error[dim] = dU_true[dim] - e->dU_qpts(dim, qpt, n, ele); 
        }
        vol = 1;

        l2_error[0] += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele) * U_error * U_error; 
        if (geo.nDims == 2)
        {
          l2_error[1] += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele) * (U_error * U_error +
              dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1]); 
        }
        else
        {
          l2_error[1] += e->weights_qpts(qpt) * e->jaco_det_qpts(qpt, ele) * (U_error * U_error +
              dU_error[0] * dU_error[0] + dU_error[1] * dU_error[1] + dU_error[2] * dU_error[2]); 
        }
      }
    }
  }

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(l2_error.data(), l2_error.data(), 2, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }

#endif


  /* Print to terminal */
  if (input->rank == 0)
  {
    std::cout << "l2_error: ";
    for (auto &val : l2_error)
      std::cout << std::scientific << std::sqrt(val / vol) << " ";
    std::cout << std::endl;

    /* Write to file */
    f << iter << " " << std::scientific << std::setprecision(16) << flow_time << " ";

    for (auto &val : l2_error)
      f << std::sqrt(val / vol) << " ";
    f << std::endl;
  }

}

void FRSolver::compute_forces(std::array<double,3> &force_conv, std::array<double,3> &force_visc, std::ofstream *cp_file = NULL)
{
  double taun[3] = {0,0,0};

  /* Factor for forming non-dimensional coefficients */
  double Vsq = 0.0;
  for (unsigned int dim = 0; dim < geo.nDims; dim++)
    Vsq += input->V_fs(dim) * input->V_fs(dim);

  double fac = 1.0 / (0.5 * input->rho_fs * Vsq);

  bool write_cp = (cp_file != NULL && cp_file->is_open());

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);
    unsigned int idx = count % geo.nFptsPerFace;

    if (bnd_id == SLIP_WALL || bnd_id == ISOTHERMAL_NOSLIP || bnd_id == ISOTHERMAL_NOSLIP_MOVING || 
        bnd_id == ADIABATIC_NOSLIP || bnd_id == ADIABATIC_NOSLIP_MOVING) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(0, fpt);

      if (write_cp)
      {
        /* Write CP distrubtion to file */
        double CP = (PL - input->P_fs) * fac;
        for(unsigned int dim = 0; dim < geo.nDims; dim++)
          *cp_file << std::scientific << faces->coord(dim, fpt) << " ";
        *cp_file << std::scientific << CP << std::endl;
      }

      /* Sum inviscid force contributions */
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
      {
        //TODO: need to fix quadrature weights for mixed element cases!
        force_conv[dim] += elesObjs[0]->weights_fpts(idx) * PL *
          faces->norm(dim, fpt) * faces->dA(0, fpt);
      }

      if (input->viscous)
      {
        if (geo.nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(0, 0, fpt);
          double momx = faces->U(0, 1, fpt);
          double momy = faces->U(0, 2, fpt);
          double e = faces->U(0, 3, fpt);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(0, 0, 0, fpt);
          double momx_dx = faces->dU(0, 0, 1, fpt);
          double momy_dx = faces->dU(0, 0, 2, fpt);

          double rho_dy = faces->dU(0, 1, 0, fpt);
          double momx_dy = faces->dU(0, 1, 1, fpt);
          double momy_dy = faces->dU(0, 1, 2, fpt);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(0, fpt) + tauxy * faces->norm(1, fpt);
          taun[1] = tauxy * faces->norm(0, fpt) + tauyy * faces->norm(1, fpt);

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force_visc[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
              faces->dA(0, fpt);

        }
        else if (geo.nDims == 3)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(0, 0, fpt);
          double momx = faces->U(0, 1, fpt);
          double momy = faces->U(0, 2, fpt);
          double momz = faces->U(0, 3, fpt);
          double e = faces->U(0, 4, fpt);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

           /* Gradients */
          double rho_dx = faces->dU(0, 0, 0, fpt);
          double momx_dx = faces->dU(0, 0, 1, fpt);
          double momy_dx = faces->dU(0, 0, 2, fpt);
          double momz_dx = faces->dU(0, 0, 3, fpt);

          double rho_dy = faces->dU(0, 1, 0, fpt);
          double momx_dy = faces->dU(0, 1, 1, fpt);
          double momy_dy = faces->dU(0, 1, 2, fpt);
          double momz_dy = faces->dU(0, 1, 3, fpt);

          double rho_dz = faces->dU(0, 2, 0, fpt);
          double momx_dz = faces->dU(0, 2, 1, fpt);
          double momy_dz = faces->dU(0, 2, 2, fpt);
          double momz_dz = faces->dU(0, 2, 3, fpt);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;
          double du_dz = (momx_dz - rho_dz * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;
          double dv_dz = (momy_dz - rho_dz * v) / rho;

          double dw_dx = (momz_dx - rho_dx * w) / rho;
          double dw_dy = (momz_dy - rho_dy * w) / rho;
          double dw_dz = (momz_dz - rho_dz * w) / rho;

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(0, fpt) + tauxy * faces->norm(1, fpt) + tauxz * faces->norm(2, fpt);
          taun[1] = tauxy * faces->norm(0, fpt) + tauyy * faces->norm(1, fpt) + tauyz * faces->norm(2, fpt);
          taun[2] = tauxz * faces->norm(0, fpt) + tauyz * faces->norm(1, fpt) + tauzz * faces->norm(2, fpt);

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force_visc[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
              faces->dA(0, fpt);
        }

      }
      count++;
    }
  }
}

void FRSolver::compute_moments(std::array<double,3> &tot_force, std::array<double,3> &tot_moment)
{
  /*! ---- TAKING ALL MOMENTS ABOUT 'geo.x_cg' ---- */
  tot_force.fill(0.0);
  tot_moment.fill(0.0);

  double taun[3];
  double force[3] = {0,0,0}; //! NOTE: need z-component initialized to '0' for 2D

  int c1[3] = {1,2,0}; // Cross-product index maps
  int c2[3] = {2,0,1};

  unsigned int count = 0;
  /* Loop over boundary faces */
  for (unsigned int fpt = geo.nGfpts_int; fpt < geo.nGfpts_int + geo.nGfpts_bnd; fpt++)
  {
    /* Get boundary ID */
    unsigned int bnd_id = geo.gfpt2bnd(fpt - geo.nGfpts_int);
    unsigned int idx = count % geo.nFptsPerFace;

    if (bnd_id == SLIP_WALL || bnd_id == ISOTHERMAL_NOSLIP || bnd_id == ISOTHERMAL_NOSLIP_MOVING || 
        bnd_id == ADIABATIC_NOSLIP || bnd_id == ADIABATIC_NOSLIP_MOVING) /* On wall boundary */
    {
      /* Get pressure */
      double PL = faces->P(0, fpt);

      /* Sum inviscid force contributions */
      //TODO: need to fix quadrature weights for mixed element cases!
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
        force[dim] = elesObjs[0]->weights_fpts(idx) * PL *
          faces->norm(dim, fpt) * faces->dA(0, fpt);

      if (input->viscous)
      {
        if (geo.nDims == 2)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(0, 0, fpt);
          double momx = faces->U(0, 1, fpt);
          double momy = faces->U(0, 2, fpt);
          double e = faces->U(0, 3, fpt);

          double u = momx / rho;
          double v = momy / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v);

          /* Gradients */
          double rho_dx = faces->dU(0, 0, 0, fpt);
          double momx_dx = faces->dU(0, 0, 1, fpt);
          double momy_dx = faces->dU(0, 0, 2, fpt);

          double rho_dy = faces->dU(0, 1, 0, fpt);
          double momx_dy = faces->dU(0, 1, 1, fpt);
          double momy_dy = faces->dU(0, 1, 2, fpt);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;

          double diag = (du_dx + dv_dy) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauyy = 2.0 * mu * (dv_dy - diag);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(0, fpt) + tauxy * faces->norm(1, fpt);
          taun[1] = tauxy * faces->norm(0, fpt) + tauyy * faces->norm(1, fpt);

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] * faces->dA(0, fpt);
        }
        else if (geo.nDims == 3)
        {
          /* Setting variables for convenience */
          /* States */
          double rho = faces->U(0, 0, 0);
          double momx = faces->U(0, 1, 0);
          double momy = faces->U(0, 2, 0);
          double momz = faces->U(0, 3, 0);
          double e = faces->U(0, 4, 0);

          double u = momx / rho;
          double v = momy / rho;
          double w = momz / rho;
          double e_int = e / rho - 0.5 * (u*u + v*v + w*w);

           /* Gradients */
          double rho_dx = faces->dU(0, 0, 0, fpt);
          double momx_dx = faces->dU(0, 0, 1, fpt);
          double momy_dx = faces->dU(0, 0, 2, fpt);
          double momz_dx = faces->dU(0, 0, 3, fpt);

          double rho_dy = faces->dU(0, 1, 0, fpt);
          double momx_dy = faces->dU(0, 1, 1, fpt);
          double momy_dy = faces->dU(0, 1, 2, fpt);
          double momz_dy = faces->dU(0, 1, 3, fpt);

          double rho_dz = faces->dU(0, 2, 0, fpt);
          double momx_dz = faces->dU(0, 2, 1, fpt);
          double momy_dz = faces->dU(0, 2, 2, fpt);
          double momz_dz = faces->dU(0, 2, 3, fpt);

          /* Set viscosity */
          double mu;
          if (input->fix_vis)
          {
            mu = input->mu;
          }
          /* If desired, use Sutherland's law */
          else
          {
            double rt_ratio = (input->gamma - 1.0) * e_int / (input->rt);
            mu = input->mu * std::pow(rt_ratio,1.5) * (1. + input->c_sth) / (rt_ratio +
                input->c_sth);
          }

          double du_dx = (momx_dx - rho_dx * u) / rho;
          double du_dy = (momx_dy - rho_dy * u) / rho;
          double du_dz = (momx_dz - rho_dz * u) / rho;

          double dv_dx = (momy_dx - rho_dx * v) / rho;
          double dv_dy = (momy_dy - rho_dy * v) / rho;
          double dv_dz = (momy_dz - rho_dz * v) / rho;

          double dw_dx = (momz_dx - rho_dx * w) / rho;
          double dw_dy = (momz_dy - rho_dy * w) / rho;
          double dw_dz = (momz_dz - rho_dz * w) / rho;

          double diag = (du_dx + dv_dy + dw_dz) / 3.0;

          double tauxx = 2.0 * mu * (du_dx - diag);
          double tauyy = 2.0 * mu * (dv_dy - diag);
          double tauzz = 2.0 * mu * (dw_dz - diag);
          double tauxy = mu * (du_dy + dv_dx);
          double tauxz = mu * (du_dz + dw_dx);
          double tauyz = mu * (dv_dz + dw_dy);

          /* Get viscous normal stress */
          taun[0] = tauxx * faces->norm(0, fpt) + tauxy * faces->norm(1, fpt) + tauxz * faces->norm(2, fpt);
          taun[1] = tauxy * faces->norm(0, fpt) + tauyy * faces->norm(1, fpt) + tauyz * faces->norm(2, fpt);
          taun[2] = tauxz * faces->norm(0, fpt) + tauyz * faces->norm(1, fpt) + tauzz * faces->norm(2, fpt);

          //TODO: need to fix quadrature weights for mixed element cases!
          for (unsigned int dim = 0; dim < geo.nDims; dim++)
            force[dim] -= elesObjs[0]->weights_fpts(idx) * taun[dim] *
              faces->dA(0, fpt);
        }

      }

      // Add fpt's contribution to total force and moment
      for (unsigned int d = 0; d < geo.nDims; d++)
        tot_force[d] += force[d];

      if (geo.nDims == 3)
      {
        for (unsigned int d = 0; d < geo.nDims; d++)
          tot_moment[d] += (faces->coord(c1[d], fpt) - geo.x_cg(c1[d])) * force[c2[d]]
              - (faces->coord(c2[d], fpt) - geo.x_cg(c2[d])) * force[c1[d]];
      }
      else
      {
        // Only a 'z' component in 2D
        tot_moment[2] += (faces->coord(0, fpt) - geo.x_cg(0)) * force[1]
            - (faces->coord(1, fpt) - geo.x_cg(1)) * force[0];
      }

      count++;
    }
  }
}

void FRSolver::report_turbulent_stats(std::ofstream &f)
{
#ifdef _GPU
  for (auto e : elesObjs)
  {
    e->U_spts = e->U_spts_d;
    e->dU_spts = e->dU_spts_d;
  }
#endif

  unsigned int iter = current_iter;
  if (input->p_multi)
    iter = iter / input->mg_steps[0];

  double keng = 0; double enst = 0; double vol = 0;

  for (auto e : elesObjs)
  {
    for (unsigned int ele = 0; ele < e->nEles; ele++)
      vol += e->vol(ele);

    for (unsigned int spt = 0; spt < e->nSpts; spt++)
    {
      for (unsigned int ele = 0; ele < e->nEles; ele++)
      {
        /* Setting variables for convenience */
        /* States */
        double rho = e->U_spts(spt, 0, ele);
        double momx = e->U_spts(spt, 1, ele);
        double momy = e->U_spts(spt, 2, ele);
        double momz = e->U_spts(spt, 3, ele);

        double u = momx / rho;
        double v = momy / rho;
        double w = momz / rho;

         /* Gradients */
        double rho_dx = e->dU_spts(0, spt, 0, ele);
        double momx_dx = e->dU_spts(0, spt, 1, ele);
        double momy_dx = e->dU_spts(0, spt, 2, ele);
        double momz_dx = e->dU_spts(0, spt, 3, ele);

        double rho_dy = e->dU_spts(1, spt, 0, ele);
        double momx_dy = e->dU_spts(1, spt, 1, ele);
        double momy_dy = e->dU_spts(1, spt, 2, ele);
        double momz_dy = e->dU_spts(1, spt, 3, ele);

        double rho_dz = e->dU_spts(2, spt, 0, ele);
        double momx_dz = e->dU_spts(2, spt, 1, ele);
        double momy_dz = e->dU_spts(2, spt, 2, ele);
        double momz_dz = e->dU_spts(2, spt, 3, ele);

        double du_dx = (momx_dx - rho_dx * u) / rho;
        double du_dy = (momx_dy - rho_dy * u) / rho;
        double du_dz = (momx_dz - rho_dz * u) / rho;

        double dv_dx = (momy_dx - rho_dx * v) / rho;
        double dv_dy = (momy_dy - rho_dy * v) / rho;
        double dv_dz = (momy_dz - rho_dz * v) / rho;

        double dw_dx = (momz_dx - rho_dx * w) / rho;
        double dw_dy = (momz_dy - rho_dy * w) / rho;
        double dw_dz = (momz_dz - rho_dz * w) / rho;

        double elekeng = 0.5 * rho * (u*u + v*v + w*w);
        double eleenst = 0.5 * rho * ((dw_dy - dv_dz) * (dw_dy - dv_dz) + 
                                      (du_dz - dw_dx) * (du_dz - dw_dx) + 
                                      (dv_dx - du_dy) * (dv_dx - du_dy));

        keng += e->jaco_det_spts(spt, ele) * e->weights_spts(spt) * elekeng;
        enst += e->jaco_det_spts(spt, ele) * e->weights_spts(spt) * eleenst;
      }
    }
  }

#ifdef _MPI
  if (input->rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, &vol, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &keng, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(MPI_IN_PLACE, &enst, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }
  else
  {
    MPI_Reduce(&vol, &vol, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(&keng, &keng, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
    MPI_Reduce(&enst, &enst, 1, MPI_DOUBLE, MPI_SUM, 0, myComm);
  }

#endif

  if (input->rank == 0)
  {
    keng /= vol * input->rho_fs * input->v_mag_fs * input->v_mag_fs;
    enst /= vol * input->rho_fs * input->v_mag_fs * input->v_mag_fs;

    /* Print to terminal */
    std::cout << std::scientific <<  "ke: " << keng << " enstrophy: " << enst << std::endl;

    /* Write to file */
    f << iter << " " << std::scientific << std::setprecision(16) << flow_time << " ";
    f << keng << " " << enst << std::endl;
  }

}

void FRSolver::accumulate_time_averages(void)
{
#ifdef _CPU
  double fac = 0.5*(flow_time - tavg_prev_time);

  for (auto e : elesObjs)
  {
    unsigned int nVars = e->nVars;
    unsigned int nDims = e->nDims;
    for (unsigned spt = 0; spt < e->nSpts; spt++)
    {
      for (unsigned ele = 0; ele < e->nEles; ele++)
      {
        double rho = e->U_spts(spt,0,ele);
        double rhoU = e->U_spts(spt,1,ele);
        double rhoV = e->U_spts(spt,2,ele);
        double rhoW = e->U_spts(spt,3,ele);
        double rhoE = e->U_spts(spt,nDims+1,ele);

        double u = rhoU / rho;
        double v = rhoV / rho;
        double w = rhoW / rho;

        double P = (input->gamma - 1.) * (rhoE - 0.5* rho * (u*u + v*v + w*w));

        // Collect conservative variables
        for (unsigned int i = 0; i < nVars; i++)
          e->tavg_curr(spt, i, ele) = e->U_spts(spt, i, ele);

        // Collect velocities
        e->tavg_curr(spt, nVars+0, ele) = u;
        e->tavg_curr(spt, nVars+1, ele) = v;
        e->tavg_curr(spt, nVars+2, ele) = w;

        // Pressure
        e->tavg_curr(spt, nVars+nDims, ele) = P;

        // Accumulate
        for (int i = 0; i < nVars+nDims+1; i++)
          e->tavg_acc(spt, i, ele) += fac * (e->tavg_prev(spt, i, ele) + e->tavg_curr(spt, i, ele));
      }
    }

    e->tavg_prev = e->tavg_curr;
  }

#endif

#ifdef _GPU
  for (auto e : elesObjs)
  {
    accumulate_time_averages_wrapper(e->tavg_acc_d,e->tavg_prev_d,e->tavg_curr_d,e->U_spts_d,
        tavg_prev_time,flow_time,input->gamma,e->nSpts,e->nVars,e->nDims,e->nEles);

    cudaDeviceSynchronize();
    check_error();
    device_copy(e->tavg_prev_d, e->tavg_curr_d, e->tavg_curr_d.max_size());
  }
#endif

  tavg_prev_time = flow_time;
}

void FRSolver::filter_solution()
{
  if (input->filt_on)
  {
    filt.apply_sensor();
    filt.apply_expfilter();
  }
}

void FRSolver::init_grid_motion(double time)
{
  if (!input->motion) return;

#ifdef _CPU
  move_grid(input, geo, time);
#endif

#ifdef _GPU
  if (input->motion_type != RIGID_BODY)
  {
    move_grid_wrapper(geo.coord_nodes_d, geo.coords_init_d, geo.grid_vel_nodes_d,
                      motion_vars, geo.nNodes, geo.nDims, input->motion_type, time, geo.gridID);
    check_error();
  }
#endif

  for (auto e : elesObjs)
    e->move(faces);

  grid_time = time;
}

void FRSolver::move(double time, bool update_iblank)
{
  if (!input->motion) return;
  if (time == grid_time && !(update_iblank && input->overset)) return; // Already set

  if (update_iblank && input->overset)
  {
#ifdef _BUILD_LIB
    // Guess grid position at end of time step and perform blanking procedure
    auto xcg = geo.x_cg;
    auto Rmat = geo.Rmat;
    double dt = elesObjs[0]->dt(0);

    for (unsigned int d = 0; d < geo.nDims; d++)
      geo.x_cg(d) += dt * geo.vel_cg(d);

    if (input->motion_type == RIGID_BODY)
    {
      Quat q(geo.q(0),geo.q(1),geo.q(2),geo.q(3));
      Quat qdot(geo.qdot(0),geo.qdot(1),geo.qdot(2),geo.qdot(3));

      q.normalize();
      for (unsigned int i = 0; i < 4; i++)
        q[i] += qdot[i] * dt;

      geo.Rmat = getRotationMatrix(q);

#ifdef _CPU
      // Update grid position based on rigid-body motion: CG offset + rotation
      for (unsigned int i = 0; i < geo.nNodes; i++)
        for (unsigned int d = 0; d < geo.nDims; d++)
          geo.coord_nodes(i,d) = geo.x_cg(d);

      auto &A = geo.coords_init;
      auto &B = geo.Rmat;  /// TODO: double-check orientation
      auto &C = geo.coord_nodes;

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geo.nNodes, geo.nDims, geo.nDims,
          1.0, A.data(), A.ldim(), B.data(), B.ldim(), 1.0, C.data(), C.ldim());
#endif

#ifdef _GPU
      geo.x_cg_d = geo.x_cg;
      geo.Rmat_d = geo.Rmat;

      update_nodes_rigid_wrapper(geo.coords_init_d, geo.coord_nodes_d, geo.Rmat_d,
          geo.x_cg_d, geo.nNodes, geo.nDims);

      copy_coords_ele_wrapper(eles->nodes_d, geo.coord_nodes_d,
          geo.ele2nodesBT_d[HEX], eles->nNodes, eles->nNodes, eles->nDims);
#endif
    }
    else
    {
#ifdef _CPU
    for (unsigned int nd = 0; nd < geo.nNodes; nd++)
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
        geo.coord_nodes(nd,dim) += geo.grid_vel_nodes(nd,dim) * dt;
#endif

#ifdef _GPU
    geo.x_cg_d = geo.x_cg;

    estimate_point_positions_nodes_wrapper(geo.coord_nodes_d,
        geo.grid_vel_nodes_d,dt,geo.nNodes,geo.nDims);
#endif
    }

#ifdef _GPU
    geo.coord_nodes = geo.coord_nodes_d;
#endif

    ZEFR->tg_update_transform(geo.Rmat.data(), geo.x_cg.data(), geo.nDims);

    PUSH_NVTX_RANGE("TG-UNBLANK-1",4);
    ZEFR->unblank_1();
    POP_NVTX_RANGE;

    // Reset position & orientation to original values
    geo.x_cg = xcg;
    geo.Rmat = Rmat;
#ifdef _GPU
    geo.x_cg_d = geo.x_cg;
    geo.Rmat_d = geo.Rmat;
#endif
#endif // _BUILD_LIB
  }

  if (input->motion_type != RIGID_BODY)
  {
#ifdef _CPU
    move_grid(input, geo, time);
#endif
#ifdef _GPU
    move_grid_wrapper(geo.coord_nodes_d, geo.coords_init_d, geo.grid_vel_nodes_d,
                      motion_vars, geo.nNodes, geo.nDims, input->motion_type, time, geo.gridID);
    check_error();
#endif
  }

  for (auto e : elesObjs)
    e->move(faces);

#ifdef _BUILD_LIB
  // Update the overset connectivity to the new grid positions
  if (input->overset)
  {
    if (input->motion_type == CIRCULAR_TRANS || input->motion_type == RIGID_BODY)
    {
      if (input->motion_type == CIRCULAR_TRANS)
      {
        geo.x_cg.assign({3},0.0);

        if (input->gridID == 0)
        {
          geo.x_cg(0) = input->moveAx*sin(2.*pi*input->moveFx*time);
          geo.x_cg(1) = input->moveAy*(1-cos(2.*pi*input->moveFy*time));
          if (geo.nDims == 3)
            geo.x_cg(2) = -input->moveAz*sin(2.*pi*input->moveFz*time);
        }
      }

      // Update Tioga's offset vector for iblank setting
      ZEFR->tg_update_transform(geo.Rmat.data(), geo.x_cg.data(), geo.nDims);
    }

#ifdef _GPU
    geo.coord_nodes = geo.coord_nodes_d;
#endif

    if (update_iblank)
    {
      // Grid reset to current flow time; re-do blanking, find any unblanked
      // elements, and perform the unblank interpolation on them
      PUSH_NVTX_RANGE("TG-UNBLANK-2",5);
      ZEFR->unblank_2(faces->nVars);
      POP_NVTX_RANGE;

      ZEFR->update_iblank_gpu();
    }

    PUSH_NVTX_RANGE("TG-PT-CONN",3);
    ZEFR->tg_point_connectivity();
    POP_NVTX_RANGE;
  }
#endif

  grid_time = time;
}

void FRSolver::move_grid_next(double time)
{
  if (!input->motion || !input->overset) return;

#ifdef _BUILD_LIB
  // Guess grid position at end of time step and perform blanking procedure
  geo.tmp_x_cg = geo.x_cg;
  geo.tmp_Rmat = geo.Rmat;
  double dt = elesObjs[0]->dt(0);

  for (unsigned int d = 0; d < geo.nDims; d++)
    geo.x_cg(d) += dt * geo.vel_cg(d);

  if (input->motion_type == RIGID_BODY)
  {
    Quat q(geo.q(0),geo.q(1),geo.q(2),geo.q(3));
    Quat qdot(geo.qdot(0),geo.qdot(1),geo.qdot(2),geo.qdot(3));

    q.normalize();
    for (unsigned int i = 0; i < 4; i++)
      q[i] += qdot[i] * dt;

    geo.Rmat = getRotationMatrix(q);

#ifdef _CPU
    // Update grid position based on rigid-body motion: CG offset + rotation
    for (unsigned int i = 0; i < geo.nNodes; i++)
      for (unsigned int d = 0; d < geo.nDims; d++)
        geo.coord_nodes(i,d) = geo.x_cg(d);

    auto &A = geo.coords_init;
    auto &B = geo.Rmat;  /// TODO: double-check orientation
    auto &C = geo.coord_nodes;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, geo.nNodes, geo.nDims, geo.nDims,
                1.0, A.data(), A.ldim(), B.data(), B.ldim(), 1.0, C.data(), C.ldim());
#endif

#ifdef _GPU
    geo.x_cg_d = geo.x_cg;
    geo.Rmat_d = geo.Rmat;

    update_nodes_rigid_wrapper(geo.coords_init_d, geo.coord_nodes_d, geo.Rmat_d,
                               geo.x_cg_d, geo.nNodes, geo.nDims);

    copy_coords_ele_wrapper(eles->nodes_d, geo.coord_nodes_d,
                            geo.ele2nodesBT_d[HEX], eles->nNodes, eles->nNodes, eles->nDims);
#endif
  }
  else
  {
#ifdef _CPU
    for (unsigned int nd = 0; nd < geo.nNodes; nd++)
      for (unsigned int dim = 0; dim < geo.nDims; dim++)
        geo.coord_nodes(nd,dim) += geo.grid_vel_nodes(nd,dim) * dt;
#endif

#ifdef _GPU
    geo.x_cg_d = geo.x_cg;

    estimate_point_positions_nodes_wrapper(geo.coord_nodes_d,
                                           geo.grid_vel_nodes_d,dt,geo.nNodes,geo.nDims);
#endif
  }

#ifdef _GPU
  geo.coord_nodes = geo.coord_nodes_d;
#endif
#endif // _BUILD_LIB
}


void FRSolver::move_grid_now(double time)
{
  if (!input->motion) return;

  // Reset position & orientation to original values
  geo.x_cg = geo.tmp_x_cg;
  geo.Rmat = geo.tmp_Rmat;
#ifdef _GPU
  geo.x_cg_d = geo.x_cg;
  geo.Rmat_d = geo.Rmat;
#endif

  if (input->motion_type != RIGID_BODY)
  {
#ifdef _CPU
    move_grid(input, geo, time);
#endif
#ifdef _GPU
    move_grid_wrapper(geo.coord_nodes_d, geo.coords_init_d, geo.grid_vel_nodes_d,
                      motion_vars, geo.nNodes, geo.nDims, input->motion_type, time, geo.gridID);
    check_error();
#endif
  }

  for (auto e : elesObjs)
    e->move(faces);

#ifdef _BUILD_LIB
  // Update the overset connectivity to the new grid positions
  if (input->overset)
  {
    if (input->motion_type == CIRCULAR_TRANS || input->motion_type == RIGID_BODY)
    {
      if (input->motion_type == CIRCULAR_TRANS)
      {
        geo.x_cg.assign({3},0.0);

        if (input->gridID == 0)
        {
          geo.x_cg(0) = input->moveAx*sin(2.*pi*input->moveFx*time);
          geo.x_cg(1) = input->moveAy*(1-cos(2.*pi*input->moveFy*time));
          if (geo.nDims == 3)
            geo.x_cg(2) = -input->moveAz*sin(2.*pi*input->moveFz*time);
        }
        geo.tmp_x_cg = geo.x_cg;
      }
    }

#ifdef _GPU
    geo.coord_nodes = geo.coord_nodes_d;
#endif
  }
#endif

  grid_time = time;
}

void FRSolver::rigid_body_update(unsigned int stage)
{
  if (input->motion_type != RIGID_BODY) return;

  // ---- Compute forces & moments in the global coordinate system ----

  std::array<double,3> force = {0,0,0}, torque = {0,0,0};

  if (input->full_6dof && input->gridID == 0) // Don't apply to overset bkgd grids
  {
#ifdef _CPU
    compute_moments(force,torque);
#endif
#ifdef _GPU
    compute_moments_wrapper(force,torque,faces->U_d,faces->dU_d,faces->P_d,faces->coord_d,geo.x_cg_d,faces->norm_d,
        faces->dA_d,geo.gfpt2bnd_d,eles->weights_fpts_d,force_d,moment_d,input->gamma,input->rt,
        input->c_sth,input->mu,input->viscous,input->fix_vis,eles->nVars,eles->nDims,geo.nGfpts_int,
        geo.nBndFaces,geo.nFptsPerFace);
#endif

#ifdef _MPI
    // Update translational motion in sync across grids, but not rotation
    MPI_Allreduce(MPI_IN_PLACE, force.data(), 3, MPI_DOUBLE, MPI_SUM, worldComm);
    MPI_Allreduce(MPI_IN_PLACE, torque.data(), 3, MPI_DOUBLE, MPI_SUM, myComm);
#endif
  }

  // Add in force from gravity
  force[2] -= input->g*geo.mass;

  int c1[3] = {1,2,0}; // Cross-product index maps
  int c2[3] = {2,0,1};

  double bi = rk_beta(stage);

  Quat q(geo.q(0),geo.q(1),geo.q(2),geo.q(3));
  q.normalize();
  Quat qdot(geo.qdot(0),geo.qdot(1),geo.qdot(2),geo.qdot(3));
  Quat omega;
  omega = 2.*(q.conj())*qdot; // body-frame omega

  // Transform moments from global coords to body coords
  Quat tau(0.0, torque[0],torque[1],torque[2]);
  tau = q.conj()*tau*q;

  // ---- q_ddot = .5*(qdot*w + q*[Jinv*(tau - w x J*w)]) ----
  Quat Jw;
  for (unsigned i = 0; i < 3; i++)
    for (unsigned j = 0; j < 3; j++)
      Jw[i+1] += geo.Jmat(i,j) * omega[j+1];

  Quat tmp1 = tau - (2.*q.conj()*qdot).cross(Jw);
  Quat wdot;
  for (unsigned i = 0; i < 3; i++)
    for (unsigned j = 0; j < 3; j++)
      wdot[i+1] += geo.Jinv(i,j) * tmp1[j+1];

  Quat q_res = .5*q*omega; // Omega is in body coords
  Quat qdot_res = .5*(qdot*omega + q*wdot);
  double v_res[3] = {force[0]/geo.mass, force[1]/geo.mass, force[2]/geo.mass};

  auto tmp_x_cg = geo.x_cg;

  // ---- Update x, v, q, qdot ----

  if (input->dt_scheme == "RK54")
  {
    if (stage < input->nStages - 1)
    {
      double ai = rk_alpha(stage);
      for (unsigned int d = 0; d < eles->nDims; d++)
      {
        geo.x_cg(d) = x_til(d) + ai*eles->dt(0) * geo.vel_cg(d);
        x_til(d) = geo.x_cg(d) + (bi-ai)*eles->dt(0) * geo.vel_cg(d);

        geo.vel_cg(d) = v_til(d) + ai*eles->dt(0) * v_res[d];
        v_til(d) = geo.vel_cg(d) + (bi-ai)*eles->dt(0) * v_res[d];
      }

      for (unsigned int i = 0; i < 4; i++)
      {
        geo.q(i) = q_til(i) + ai*eles->dt(0) * q_res[i];
        q_til(i) = geo.q(i) + (bi-ai)*eles->dt(0) * q_res[i];

        geo.qdot(i) = qdot_til(i) + ai*eles->dt(0) * qdot_res[i];
        qdot_til(i) = geo.qdot(i) + (bi-ai)*eles->dt(0) * qdot_res[i];
      }
    }
    else
    {
      for (unsigned int d = 0; d < geo.nDims; d++)
      {
        geo.x_cg(d) = x_til(d) + bi*eles->dt(0) * geo.vel_cg(d);
        geo.vel_cg(d) = v_til(d) + bi*eles->dt(0) * v_res[d];
      }

      for (unsigned int i = 0; i < 4; i++)
      {
        geo.q(i) = q_til(i) + bi*eles->dt(0) * q_res[i];
        geo.qdot(i) = qdot_til(i) + bi*eles->dt(0) * qdot_res[i];
      }
    }
  }
  else if (!input->implicit_method)
  {
    if (stage == 0 && input->nStages > 1)
    {
      x_til = geo.x_cg;
      v_til = geo.vel_cg;
      q_til = geo.q;
      qdot_til = geo.qdot;
    }

    for (unsigned int d = 0; d < eles->nDims; d++)
    {
      geo.x_res(stage,d) = geo.vel_cg(d);
      geo.v_res(stage,d) = v_res[d];
    }

    for (unsigned int i = 0; i < 4; i++)
    {
      geo.q_res(stage,i) = q_res[i];
      geo.qdot_res(stage,i) = qdot_res[i];
    }

    if (stage < input->nStages - 1)
    {
      double ai = rk_alpha(stage);
      for (unsigned int d = 0; d < eles->nDims; d++)
      {
        geo.x_cg(d) = x_til(d) + ai*eles->dt(0) * geo.x_res(stage,d);
        geo.vel_cg(d) = v_til(d) + ai*eles->dt(0) * geo.v_res(stage,d);
      }

      for (unsigned int i = 0; i < 4; i++)
      {
        geo.q(i) = q_til(i) + ai*eles->dt(0) * geo.q_res(stage,i);
        geo.qdot(i) = qdot_til(i) + ai*eles->dt(0) * geo.qdot_res(stage,i);
      }
    }
    else
    {
      // Last-stage update
      if (input->nStages > 1)
      {
        geo.x_cg = x_til;
        geo.vel_cg = v_til;
        geo.q = q_til;
        geo.qdot = qdot_til;
      }

      for (int step = 0; step < input->nStages; step++)
      {
        double bi = rk_beta(step);
        for (unsigned int d = 0; d < geo.nDims; d++)
        {
          geo.x_cg(d) += bi*eles->dt(0) * geo.x_res(step,d);
          geo.vel_cg(d) += bi*eles->dt(0) * geo.v_res(step,d);
        }

        for (unsigned int i = 0; i < 4; i++)
        {
          geo.q(i) += bi*eles->dt(0) * geo.q_res(step,i);
          geo.qdot(i) += bi*eles->dt(0) * geo.qdot_res(step,i);
        }
      }
    }
  }

  // ---- Update derived quantities (omega; nodal positions & velocities) ----

  for (unsigned int i = 0; i < 4; i++)
  {
    q[i] = geo.q(i);
    qdot[i] = geo.qdot(i);
  }

  // Normalize q's for later use
  double qnorm = q.norm();
  q.normalize();
  for (unsigned int i = 0; i < 4; i++)
    geo.q(i) /= qnorm;

  omega = 2*qdot*q.conj(); // Global-frame omega
  for (unsigned int i = 0; i < geo.nDims; i++)
    geo.omega(i) = omega[i+1];

  geo.Rmat = getRotationMatrix(q);

  // v_g = q * (omega_b cross x_b) * q_conj = Rmat * W_b * x_b = Wmat * x_b
  // W is 'Spin' of omega (cross-product in matrix form)
  omega = 2*q.conj()*qdot;
  mdvector<double> W({3,3}, 0.);
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

  geo.tmp_x_cg = geo.x_cg;
  geo.tmp_Rmat = geo.Rmat;
#ifdef _GPU
  geo.x_cg_d = geo.x_cg;
  geo.vel_cg_d = geo.vel_cg;
  geo.Rmat_d = geo.Rmat;
  geo.Wmat_d = geo.Wmat;
#endif
}

void FRSolver::overset_u_send(void)
{
#ifdef _BUILD_LIB
  if (input->overset)
    ZEFR->overset_interp_send(faces->nVars, 0);
#endif
}

void FRSolver::overset_u_recv(void)
{
#ifdef _BUILD_LIB
  if (input->overset)
    ZEFR->overset_interp_recv(faces->nVars, 0);
#endif
}

void FRSolver::overset_grad_send(void)
{
#ifdef _BUILD_LIB
  if (input->overset)
  {
    // Wait for completion of corrected *and transformed* gradient
    event_record_wait_pair(3, 0, 3);

    ZEFR->overset_interp_send(faces->nVars, 1);
  }
#endif
}

void FRSolver::overset_grad_recv(void)
{
#ifdef _BUILD_LIB
  if (input->overset)
  {
    ZEFR->overset_interp_recv(faces->nVars, 1);

    // Wait for updated data on GPU before moving on to common_F
    event_record_wait_pair(2, 3, 0);
  }
#endif
}

#ifdef _GPU
void FRSolver::report_gpu_mem_usage()
{
  size_t free, total, used;
  cudaMemGetInfo(&free, &total);
  used = total - free;

#ifndef _MPI
  std::cout << "GPU Memory Usage: " << used/1e6 << " MB used of " << total/1e6 << " MB available" << std::endl;
#else
  size_t used_max, used_min;

  if (input->rank == 0)
  {
    MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, myComm);
    MPI_Reduce(&used, &used_min, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, myComm);

    if (input->overset)
      std::cout << "Grid " << input->gridID << ": ";
    std::cout << "GPU Memory Usage: " << (used_min/1e6) << " (min) - " << (used_max/1e6) << " (max) MB used of " << total/1e6;
    std::cout << " MB available per GPU" << std::endl;
  }
  else
  {
    MPI_Reduce(&used, &used_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, myComm);
    MPI_Reduce(&used, &used_min, 1, MPI_UNSIGNED_LONG, MPI_MIN, 0, myComm);
  }
#endif
}
#endif

void FRSolver::set_conv_file(std::chrono::high_resolution_clock::time_point t1)
{
  if (input->restart)
    this->conv_file.open(input->output_prefix + "/" + input->output_prefix + "_conv.dat", std::ios::app);
  else
    this->conv_file.open(input->output_prefix + "/" + input->output_prefix + "_conv.dat");
  this->conv_timer = t1;
}

double FRSolver::get_current_time(void)
{
  return flow_time;
}
