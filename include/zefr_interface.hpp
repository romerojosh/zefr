#ifndef _zefr_interface_hpp
#define _zefr_interface_hpp

#include "zefr.hpp"

struct BasicGeo
{
  int btag;         //! Body tag (aka grid ID)
  int nnodes;       //! # of mesh nodes
  double *xyz;      //! Physical positions of all mesh nodes
  int *iblank;      //! Nodal iblank values [to be set externally]
  int nwall;        //! # of wall-boundary nodes
  int nover;        //! # of overset-boundary nodes
  int *wallNodes;   //! List of wall-boundary nodes
  int *overNodes;   //! List of overset-boundary nodes
  int nCellTypes;   //! # of different cell types (hex, tet, prism, etc.) [1 for now]
  int nvert_cell;  //! # of nodes per cell for each cell type
  int nCells_type; //! # of cells for each cell type
  int *c2v;         //! Cell-to-vertex connectivity (one cell type)
};

struct ExtraGeo
{
  int nFaceTypes;   //! # of different face types (quad or tri) [1 for now]
  int nvert_face;  //! # of nodes per face
  int nFaces_type; //! # of faces for each face type
  int *f2v;         //! Face-to-vertex connectivity (one face type)
  int *f2c;         //! Face-to-cell connectivity
  int *c2f;         //! Cell-to-face connectivity
  int *iblank_cell; //! Cell iblank values
  int *iblank_face; //! Face iblank values
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
};

#ifdef _MPI
void initialize(MPI_Comm comm_in, char* inputFile, int nGrids=1, int gridID=0);
#else
void initialize(char* input_file);
#endif

void set_zefr_object(zefr *_ZEFR);

zefr* get_zefr_object(void);

void finalize(void);

/* ==== Access functions for mesh data ==== */

BasicGeo get_basic_geo_data(void);

ExtraGeo get_extra_geo_data(void);

CallbackFuncs get_callback_funcs(void);

/* ==== Access functions for solution data ==== */

double *get_q_spts(void);
double *get_q_fpts(void);

/* ==== Callback Function Wrappers ==== */

void get_nodes_per_cell(int* cellID, int* nNodes);
void get_nodes_per_face(int* faceID, int* nNodes);
void get_receptor_nodes(int* cellID, int* nNodes, double* xyz);
void get_face_nodes(int* faceID, int* nNodes, double* xyz);
void get_q_index_face(int* faceID, int *fpt, int* ind, int* stride);
void donor_inclusion_test(int* cellID, double* xyz, int* passFlag, double* rst);
void donor_frac(int* cellID, double* xyz, int* nweights, int* inode,
                double* weights, double* rst, int* buffsize);
void convert_to_modal(int *cellID, int *nSpts, double *q_in, int *npts,
                      int *index_out, double *q_out);

#endif /* _zefr_interface_hpp */
