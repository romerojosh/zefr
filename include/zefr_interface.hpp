#ifndef _zefr_interface_hpp
#define _zefr_interface_hpp

#include "zefr.hpp"

struct BasicGeo
{
  int btag;         //! Body tag (aka grid ID)
  int gridType;     //! Grid type for Direct Cut (background or geometry)
  int nnodes;       //! # of mesh nodes
  double *xyz;      //! Physical positions of all mesh nodes
  int *iblank;      //! Nodal iblank values [to be set externally]
  int nwall;        //! # of wall-boundary nodes
  int nover;        //! # of overset-boundary nodes
  int *wallNodes;   //! List of wall-boundary nodes
  int *overNodes;   //! List of overset-boundary nodes
  int nCellTypes;   //! # of different cell types (hex, tet, prism, etc.)
  int *nvert_cell;  //! # of nodes per cell for each cell type
  int *nface_cell;  //! # of faces per cell for each cell type
  int *nCells_type; //! # of cells for each cell type
  int **c2v;         //! Cell-to-vertex connectivity (all cell types)
  int nCellsTot;    //! Total # of cells in grid (all types)
};

struct ExtraGeo
{
  int nFaceTypes;   //! # of different face types (quad or tri) [1 for now]
  int *faceTypes;   //! List of face types present (line,tri,quad)
  int *cellTypes;   //! List of cell types present (tri,quad,tet,pri,pyr,hex)
  int *nvert_face;  //! # of nodes per face type
  int *nFaces_type; //! # of faces for each face type
  int nFacesTot;    //! Total # of faces in grid (all types)
  int **f2v;        //! Face-to-vertex connectivity (one face type)
  int *f2c;         //! Face-to-cell connectivity
  int **c2f;        //! Cell-to-face connectivity
  int *iblank_cell; //! Cell iblank values
  int *iblank_face; //! Face iblank values
  int nOverFaces;   //! # of explicitly-defined overset faces
  int nWallFaces;   //! # of solid wall boundary faces
  int nMpiFaces;    //! # of MPI faces
  int *overFaces;   //! List of explicitly-defined overset faces
  int *wallFaces;   //! List of solid wal  boundary faces
  int *mpiFaces;    //! List of MPI face ID's on this rank
  int *procR;       //! Opposite rank for each MPI face
  int *mpiFidR;     //! Face ID of MPI face on opposite rank
  double *grid_vel; //! Grid velocity at mesh nodes
  double *offset;
  double *Rmat;
};

struct GpuGeo
{
  double *coord_nodes; //! x,y,z positions of each node in grid
  double *coord_eles;  //! x,y,z positions of each node for each element in grid
  int estride, dstride, nstride; //! Strides for element, node, and dim within coord_eles
  int *iblank_cell;
  int *iblank_face;
};

struct CallbackFuncs
{
  void (*get_nodes_per_cell)(int* cellID, int* nNodes);
  void (*get_nodes_per_face)(int* faceID, int* nNodes);
  void (*get_receptor_nodes)(int* cellID, int* nNodes, double* xyz);
  void (*get_face_nodes)(int* faceID, int* nNodes, double* xyz);
  void (*get_q_index_face)(int* faceID, int *fpt, int* ind, int* stride);
  void (*donor_inclusion_test)(int* cellID, double* xyz, int* passFlag,
                               double* rst);
  void (*donor_frac)(int* cellID, double* xyz, int* nweights, int* inode,
                     double* weights, double* rst, int* buffsize);
  void (*convert_to_modal)(int *cellID, int *nSpts, double *q_in, int *npts,
                           int *index_out, double *q_out);
  double* (*get_q_spts)(int &ele_stride, int &spt_stride, int &var_stride, int etype);
  double* (*get_dq_spts)(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype);
  double* (*get_q_spts_d)(int &ele_stride, int &spt_stride, int &var_stride, int etype);
  double* (*get_dq_spts_d)(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype);
  double (*get_q_spt)(int cellID, int spt, int var);
  double (*get_grad_spt)(int cellID, int spt, int dim, int var);
  double& (*get_q_fpt)(int faceID, int fpt, int var);
  double& (*get_grad_fpt)(int faceID, int fpt, int dim, int var);
  void (*fringe_data_to_device)(int* fringeIDs, int nFringe, int gradFlag, double *data);
  void (*unblank_data_to_device)(int* cellIDs, int nCells, int gradFlag, double *data);
  void (*get_face_nodes_gpu)(int* fringeIDs, int nFringe, int* nPtsFace, double* xyz);
  void (*get_cell_nodes_gpu)(int* cellIDs, int nCells, int* nPtsCell, double* xyz);
  int (*get_n_weights)(int cellID);
  void (*donor_frac_gpu)(int* cellIDs, int nFringe, double* rst, double* weights);
};

namespace zefr {

#ifdef _MPI
void initialize(MPI_Comm comm_in, const char* inputFile, int nGrids=1, int gridID=0, MPI_Comm world_comm = MPI_COMM_WORLD);
#else
void initialize(char* input_file);
#endif

void set_zefr_object(Zefr *_ZEFR);

Zefr* get_zefr_object(void);

void finalize(void);

static inline bool use_gpus(void)
{
  #ifdef _GPU
  return true;
  #else
  return false;
  #endif
}

/* ==== Access functions for mesh data ==== */

BasicGeo get_basic_geo_data(void);

ExtraGeo get_extra_geo_data(void);

GpuGeo get_gpu_geo_data(void);

CallbackFuncs get_callback_funcs(void);

/* ==== Access functions for solution data ==== */

double get_q_spt(int ele, int spt, int var);
double get_grad_spt(int ele, int spt, int dim, int var);
double *get_q_spts(int &ele_stride, int &spt_stride, int &var_stride, int etype = 0);
double *get_dq_spts(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype = 0);
double *get_q_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int etype = 0);
double *get_dq_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype = 0);

/* ==== Callback Function Wrappers ==== */

void get_nodes_per_cell(int* cellID, int* nNodes);
void get_nodes_per_face(int* faceID, int* nNodes);
void get_receptor_nodes(int* cellID, int* nNodes, double* xyz);
void get_face_nodes(int* faceID, int* nNodes, double* xyz);
void get_face_nodes_gpu(int* faceIDs, int nFringe, int* nPtsFace, double* xyz);
void get_cell_nodes_gpu(int* cellIDs, int nCells, int* nPtsCell, double* xyz);
void donor_inclusion_test(int* cellID, double* xyz, int* passFlag, double* rst);
void donor_frac(int* cellID, double* xyz, int* nweights, int* inode,
                double* weights, double* rst, int* buffsize);
void convert_to_modal(int *cellID, int *nSpts, double *q_in, int *npts,
                      int *index_out, double *q_out);

double& get_q_fpt(int face, int fpt, int var);
double& get_grad_fpt(int face, int fpt, int dim, int var);

//! For runs using GPUs - copy updated fringe data to device
void fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double* data);

//! For runs with GPUs - copy unblanked element data to device
void unblank_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double *data);

int get_n_weights(int cellID);
void donor_frac_gpu(int* cellIDs, int nFringe, double* rst, double* weights);

void signal_handler(int signum);

} /* namespace zefr */
#endif /* _zefr_interface_hpp */
