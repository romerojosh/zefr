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
  int nCellTypes;   //! # of different cell types (hex, tet, prism, etc.)
  int *nvert_cell;  //! # of nodes per cell for each cell type
  int *nCells_type; //! # of cells for each cell type
  int **c2v;        //! Cell-to-vertex connectivity (per cell type)
};

struct ExtraGeo
{
  int nFaceTypes;
  int *nvert_face;
  int *nFaces_type;
  int **f2v;
  int *f2c;
  int *c2f;
  int *iblank_cell;
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
};

zefr *ZEFR;

void initialize(MPI_Comm comm_in);

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
