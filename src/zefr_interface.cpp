#include "zefr_interface.hpp"

zefr *ZEFR = NULL;

#ifdef _MPI
void initialize(MPI_Comm comm_in, char *inputFile, int nGrids, int gridID)
{
  if (!ZEFR) ZEFR = new zefr(comm_in, nGrids, gridID);

  ZEFR->read_input(inputFile);
}
#else
void initialize(char *input_file)
{
  if (!ZEFR) ZEFR = new zefr();

  ZEFR->read_input(input_file);
}
#endif

void set_zefr_object(zefr *_ZEFR)
{
  delete ZEFR;

  ZEFR = _ZEFR;
}

zefr* get_zefr_object(void)
{
  return ZEFR;
}

void finalize(void)
{
  delete ZEFR;
}

/* ---- Data-Acess Functions ---- */

BasicGeo get_basic_geo_data(void)
{
  BasicGeo geo;

  ZEFR->get_basic_geo_data(geo.btag,geo.nnodes,geo.xyz,geo.iblank,geo.nwall,
                           geo.nover,geo.wallNodes,geo.overNodes,geo.nCellTypes,
                           geo.nvert_cell,geo.nCells_type,geo.c2v);

  return geo;
}

double* get_q_spts(void)
{
  return ZEFR->get_u_spts();
}

double* get_q_fpts(void)
{
  return ZEFR->get_u_fpts();
}

ExtraGeo get_extra_geo_data(void)
{
  ExtraGeo geo;

  ZEFR->get_extra_geo_data(geo.nFaceTypes,geo.nvert_face,geo.nFaces_type,
                           geo.f2v,geo.f2c,geo.c2f,geo.iblank_face,
                           geo.iblank_cell);

  return geo;
}

CallbackFuncs get_callback_funcs(void)
{
  CallbackFuncs call;

  call.get_nodes_per_cell = get_nodes_per_cell;
  call.get_nodes_per_face = get_nodes_per_face;
  call.get_receptor_nodes = get_receptor_nodes;
  call.get_face_nodes = get_face_nodes;
  call.get_q_index_face = get_q_index_face;
  call.donor_inclusion_test = donor_inclusion_test;
  call.donor_frac = donor_frac;
  call.convert_to_modal = convert_to_modal;
}

/* ---- TIOGA Callback Functions ---- */

void get_nodes_per_cell(int* cellID, int* nNodes)
{
  ZEFR->get_nodes_per_cell(*nNodes);
}

void get_nodes_per_face(int* faceID, int* nNodes)
{
  ZEFR->get_nodes_per_face(*nNodes);
}

void get_receptor_nodes(int* cellID, int* nNodes, double* xyz)
{
  ZEFR->get_receptor_nodes(*cellID, *nNodes, xyz);
}

void get_face_nodes(int* faceID, int* nNodes, double* xyz)
{
  ZEFR->get_face_nodes(*faceID, *nNodes, xyz);
}

void get_q_index_face(int* faceID, int *fpt, int* ind, int* stride)
{
  ZEFR->get_q_index_face(*faceID, *fpt, *ind, *stride);
}

void donor_inclusion_test(int* cellID, double* xyz, int* passFlag, double* rst)
{
  ZEFR->donor_inclusion_test(*cellID, xyz, *passFlag, rst);
}

void donor_frac(int* cellID, double* xyz, int* nweights, int* inode,
                double* weights, double* rst, int* buffsize)
{
  ZEFR->donor_frac(*cellID, *nweights, inode, weights, rst, *buffsize);
}

void convert_to_modal(int* cellID, int* nSpts, double* q_in, int* npts, int* index_out, double* q_out)
{
  //assert(*nSpts == *npts);
  *index_out = (*cellID) * (*nSpts);
  for (int spt = 0; spt < (*nSpts); spt++)
    q_out[spt] = q_in[spt];
}
