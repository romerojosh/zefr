#include "zefr_interface.hpp"


void initialize(MPI_Comm comm_in)
{
  ZEFR = new zefr();
}

void finalize(void)
{
  delete zefr;
}

BasicGeo get_basic_geo_data(void)
{
  BasicGeo geo;

  ZEFR->get_basic_geo_data(geo.btag,geo.nnodes,geo.xyz,geo.iblank,geo.nwall,
                           geo.nover,geo.wallNodes,geo.overNodes,geo.nCellTypes,
                           geo.nvert_cell,geo.nCells_type,geo.c2v);

  return geo;
}


ExtraGeo get_extra_geo_data(void)
{
  ExtraGeo geo;

  ZEFR->get_extra_geo_data(geo.f2v);

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
}

void get_nodes_per_cell(int* cellID, int* nNodes)
{
  ZEFR->get_nodes_per_cell(*nNodes);
}

void get_nodes_per_face(int* faceID, int* nNodes)
{
  ZEFR->get_nodes_per_face(*faceID, *nNodes);
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
