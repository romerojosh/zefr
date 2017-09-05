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

#ifndef _zefr_hpp
#define _zefr_hpp

#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _MPI
#include "mpi.h"
#endif

class Elements;
class Faces;
class FRSolver;
class PMGrid;

#include "input.hpp"
#include "macros.hpp"
#include "multigrid.hpp"
#include "solver.hpp"

class Zefr
{
friend class FRSolver;
public:

  /*! Assign MPI communicator and overset-grid ID
   *
   * For MPI cases, Python will call MPI_Init, so we should NOT do that here
   * Instead, Python will give us our own communicator for internal use
   * (1 communicator per grid)
   */
#ifdef _MPI
  Zefr(MPI_Comm comm_in = MPI_COMM_WORLD, int n_grids = 1, int grid_id = 0, MPI_Comm comm_world = MPI_COMM_WORLD);
#else
  Zefr(void);
#endif

  //! Read input file and set basic run parameters
  void read_input(const char *inputfile);

  //! Re-apply parameters in InputStruct that may have been changed in Python
  void init_inputs(void);

  //! Perform preprocessing and prepare to run case
  void setup_solver(void);

  //! Read the restart file(s) and (for moving grids) update the geo accordingly
  void restart_solution(void);

  //! Run one full time step, including any filtering or multigrid operations
  void do_step(void);

  //! Call "do_step()" n times
  void do_n_steps(int n);

  //! Run one full time step, including any filtering or multigrid operations
  void do_rk_stage(int iter, int stage);

  //! Do the first part of an RK stage, up to extrapolate_u
  void do_rk_stage_start(int iter, int stage);
  //! Perform residual computation up to corrected gradient for overset interp
  void do_rk_stage_mid(int iter, int stage);
  //! Finish residual computation & RK stage after overset gradient interp
  void do_rk_stage_finish(int iter, int stage);

  // Functions to write data to file and/or terminal
  void write_residual(void);
  void write_solution(void);
  void write_forces(void);
  void write_error(void);

  void write_wall_time(void);

  // Other Misc. Functions
  InputStruct &get_input(void) { return input; }

  DataStruct &get_data(void) { return simData; }

  void get_forces(void);

  /* ==== Overset-Related Functions ==== */

  // Geometry Access Functions
  void get_basic_geo_data(int &btag, int& gType, int &nnodes, double *&xyz, int *&iblank,
                          int &nwall, int &nover, int *&wallNodes, int *&overNodes,
                          int &nCellTypes, int &nvert_cell, int &nCells_type,
                          int *&c2v);

  void get_extra_geo_data(int &nFaceTypes, int& nvert_face, int& nFaces_type,
                          int*& f2v, int*& f2c, int*& c2f, int*& iblank_face,
                          int*& iblank_cell, int& nOver, int*& overFaces,
                          int& nWall, int*& wallFaces, int& nMpiFaces,
                          int*& mpiFaces, int*& procR, int*& faceIdR,
                          double*& grid_vel, double*& offset, double*& Rmat);

  void get_gpu_geo_data(double*& coord_nodes, double*& coord_eles,
                        int*& iblank_cell, int*& iblank_face);

  // Solution-data access functions
  double get_u_spt(int ele, int spt, int var);
  double get_grad_spt(int ele, int spt, int dim, int var);
  double *get_u_spts(int &ele_stride, int &spt_stride, int &var_stride);
  double *get_du_spts(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride);
  double *get_u_spts_d(int &ele_stride, int &spt_stride, int &var_stride);
  double *get_du_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride);

  // Callback Functions for TIOGA
  void get_nodes_per_cell(int& nNodes);
  void get_nodes_per_face(int& nNodes);
  void get_receptor_nodes(int cellID, int& nNodes, double* xyz);
  void get_face_nodes(int faceID, int& nNodes, double* xyz);
  void donor_inclusion_test(int cellID, double* xyz, int& passFlag, double* rst);
  void donor_frac(int cellID, int& nweights, int* inode,
                  double* weights, double* rst, int buffsize);
  double& get_u_fpt(int face, int fpt, int var);
  double& get_grad_fpt(int face, int fpt, int dim, int var);

  // GPU-related callback functions
  void update_iblank_gpu(void);
  void donor_data_from_device(int *donorIDs, int nDonors, int gradFlag = 0);
  void fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag = 0, double* data = NULL);
  void unblank_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double* data);

  void get_face_nodes_gpu(int* faceIDs, int nFaces, int* nPtsFace, double *xyz);
  void get_cell_nodes_gpu(int* cellIDs, int nCells, int* nPtsCell, double *xyz);

  int get_n_weights(int cellID);
  void donor_frac_gpu(int* cellIDs, int nFringe, double* rst, double* weights);

  /// TODO: Reconsider organization
  void set_dataUpdate_callback(void (*dataUpdate)(int, double*, int));

  void set_tioga_callbacks(void (*preprocess)(void), void (*connect)(void),
                           void (*point_connect)(void), void (*iter_iblanks)(double, int),
                           void (*unblank_part_1)(void), void (*unblank_part_2)(int),
                           void (*dataUpdate_send)(int, int), void (*dataUpdate_recv)(int, int));

  void set_rigid_body_callbacks(void (*setTransform)(double*, double*, int));

  //! Move grid to an estimate of the position at t^{n+1} for overset unblanking
  void move_grid_next(double time);

  //! Move grid to given flow time
  void move_grid(double time);

  //! Move grid to the flow time specified by the iteration and RK stage
  void move_grid(int iter, int stage);

  void* get_tg_stream_handle(void);
  void* get_tg_event_handle(void);

private:
  // Generic data about the run
  int rank = 0, nRanks = 1;
  int grank = 0;   //! Global MPI rank
  int myGrid = 0;  //! For overset: which grid this rank belongs to
  int nGrids = 1;  //! For overset: # of grids in entire system
  int gridType = 1; //! For overset: with Direct Cut: background (0) or geometry (1)

  // Basic ZEFR Solver Objects
  std::shared_ptr<FRSolver> solver;
  InputStruct input;
  DataStruct simData;
  std::shared_ptr<PMGrid> pmg;
  GeoStruct *geo;

  // Files to write history output
  std::chrono::high_resolution_clock::time_point t_start;

  std::ofstream hist_file;
  std::ofstream force_file;
  std::ofstream error_file;

#ifdef _MPI
  void mpi_init(MPI_Comm comm_in, MPI_Comm comm_world, int n_grids = 1, int grid_id = 0);
#endif

  // Again, to simplify MPI vs. no-MPI compilation, this will be either an
  // MPI_Comm or an int
  _mpi_comm myComm;
  _mpi_comm worldComm;

  //! Callback function to TIOGA to perform overset interpolation
  void (*overset_interp)(int nVars, double* U_spts, int gradFlag);

  void (*overset_interp_send)(int nVars, int gradFlag);
  void (*overset_interp_recv)(int nVars, int gradFlag);

  //! Callback function to TIOGA to pre-process the grids
  void (*tg_preprocess) (void);

  //! Callback function to TIOGA to process connectivity
  void (*tg_process_connectivity) (void);

  //! Callback function to TIOGA to process only point connectivity (don't update blanking)
  void (*tg_point_connectivity) (void);

  //! Callback to TIOGA to update blanking for current iteration
  void (*tg_set_iter_iblanks) (double dt, int nVars);

  //! Callback to set a new rotation matrix & offset for TIOGA's ADT class
  void (*tg_update_transform)(double* Rmat, double* offset, int ndim);

  void (*unblank_1)(void);
  void (*unblank_2)(int);
};

#endif /* _zefr_hpp */
