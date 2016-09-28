#include "zefr_interface.hpp"

Zefr *ZEFR = NULL;

namespace zefr {

#ifdef _MPI
void initialize(MPI_Comm comm_in, char *inputFile, int nGrids, int gridID)
{
  if (!ZEFR) ZEFR = new Zefr(comm_in, nGrids, gridID);

  ZEFR->read_input(inputFile);
}
#else
void initialize(char *input_file)
{
  if (!ZEFR) ZEFR = new Zefr();

  ZEFR->read_input(input_file);
}
#endif

void set_zefr_object(Zefr *_ZEFR)
{
  delete ZEFR;

  ZEFR = _ZEFR;
}

Zefr* get_zefr_object(void)
{
  return ZEFR;
}

void finalize(void)
{
  ZEFR->write_wall_time();
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

double get_q_spt(int ele, int spt, int var)
{
  return ZEFR->get_u_spt(ele,spt,var);
}

double get_grad_spt(int ele, int spt, int dim, int var)
{
  return ZEFR->get_grad_spt(ele,spt,dim,var);
}

double& get_q_fpt(int face, int fpt, int var)
{
  return ZEFR->get_u_fpt(face,fpt,var);
}

double& get_grad_fpt(int face, int fpt, int dim, int var)
{
  return ZEFR->get_grad_fpt(face,fpt,dim,var);
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
                           geo.iblank_cell,geo.nOverFaces,geo.overFaces,
                           geo.nMpiFaces,geo.mpiFaces,geo.procR,geo.mpiFidR);

  return geo;
}

CallbackFuncs get_callback_funcs(void)
{
  CallbackFuncs call;

  call.get_nodes_per_cell = get_nodes_per_cell;
  call.get_nodes_per_face = get_nodes_per_face;
  call.get_receptor_nodes = get_receptor_nodes;
  call.get_face_nodes = get_face_nodes;
  call.donor_inclusion_test = donor_inclusion_test;
  call.donor_frac = donor_frac;
  call.convert_to_modal = convert_to_modal;
  call.get_q_spt = get_q_spt;
  call.get_grad_spt = get_grad_spt;
  call.get_q_fpt = get_q_fpt;
  call.get_grad_fpt = get_grad_fpt;

  /* GPU-specific functions */
  call.donor_data_from_device = donor_data_from_device;
  call.fringe_data_to_device = fringe_data_to_device;

  return call;
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

void donor_data_from_device(int *donorIDs, int nDonors, int gradFlag)
{
#ifdef _GPU
  ZEFR->donor_data_from_device(donorIDs, nDonors, gradFlag);
#endif
}

void fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag)
{
#ifdef _GPU
  ZEFR->fringe_data_to_device(fringeIDs, nFringe, gradFlag);
#endif
}

} /* namespace zefr */
