#include <csignal>

#include "zefr_interface.hpp"

Zefr *ZEFR = NULL;

namespace zefr {

#ifdef _MPI
void initialize(MPI_Comm comm_in, const char *inputFile, int nGrids, int gridID, MPI_Comm world_comm)
{
  if (!ZEFR) ZEFR = new Zefr(comm_in, nGrids, gridID, world_comm);

  ZEFR->read_input(inputFile);

  auto input = ZEFR->get_input();

  if (input.catch_signals)
  {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
  }
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

  ZEFR->get_basic_geo_data(geo.btag,geo.gridType,geo.nnodes,geo.xyz,geo.iblank,geo.nwall,
                           geo.nover,geo.wallNodes,geo.overNodes,geo.nCellTypes,
                           geo.nvert_cell,geo.nface_cell,geo.nCells_type,geo.c2v);

  geo.nCellsTot = 0;
  for (int i = 0; i < geo.nCellTypes; i++)
    geo.nCellsTot += geo.nCells_type[i];

  return geo;
}

double& get_q_spt(int ele, int spt, int var)
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

double* get_q_spts(int &ele_stride, int &spt_stride, int &var_stride, int etype)
{
  return ZEFR->get_u_spts(ele_stride, spt_stride, var_stride, etype);
}

double* get_dq_spts(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype)
{
  return ZEFR->get_du_spts(ele_stride, spt_stride, var_stride, dim_stride, etype);
}

double* get_q_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int etype)
{
  return ZEFR->get_u_spts_d(ele_stride, spt_stride, var_stride, etype);
}

double* get_dq_spts_d(int &ele_stride, int &spt_stride, int &var_stride, int &dim_stride, int etype)
{
  return ZEFR->get_du_spts_d(ele_stride, spt_stride, var_stride, dim_stride, etype);
}

ExtraGeo get_extra_geo_data(void)
{
  ExtraGeo geo;

  ZEFR->get_extra_geo_data(geo.nFaceTypes,geo.faceTypes,geo.cellTypes,
      geo.nvert_face,geo.nFaces_type,geo.f2v,geo.f2c,geo.c2f,geo.iblank_face,
      geo.iblank_cell,geo.nOverFaces,geo.overFaces,geo.nWallFaces,geo.wallFaces,
      geo.nMpiFaces,geo.mpiFaces,geo.procR,geo.mpiFidR,geo.grid_vel,geo.offset,geo.Rmat);

  geo.nFacesTot = 0;
  for (int i = 0; i < geo.nFaceTypes; i++)
    geo.nFacesTot += geo.nFaces_type[i];

  return geo;
}

GpuGeo get_gpu_geo_data(void)
{
  GpuGeo geo;

  ZEFR->get_gpu_geo_data(geo.coord_nodes,geo.coord_eles,geo.iblank_cell,geo.iblank_face);

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
  call.get_q_spts = get_q_spts;
  call.get_dq_spts = get_dq_spts;

  /* GPU-specific functions */
  call.fringe_data_to_device = fringe_data_to_device;
  call.unblank_data_to_device = unblank_data_to_device;
  call.get_q_spts_d = get_q_spts_d;
  call.get_dq_spts_d = get_dq_spts_d;

  call.get_face_nodes_gpu = get_face_nodes_gpu;
  call.get_cell_nodes_gpu = get_cell_nodes_gpu;

  call.get_n_weights = get_n_weights;
  call.donor_frac_gpu = donor_frac_gpu;

  return call;
}

/* ---- TIOGA Callback Functions ---- */

void get_nodes_per_cell(int* cellID, int* nNodes)
{
  ZEFR->get_nodes_per_cell(*cellID, *nNodes);
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

void get_face_nodes_gpu(int* faceIDs, int nFringe, int* nPtsFace, double* xyz)
{
  ZEFR->get_face_nodes_gpu(faceIDs, nFringe, nPtsFace, xyz);
}

void get_cell_nodes_gpu(int* cellIDs, int nCells, int* nPtsCell, double* xyz)
{
  ZEFR->get_cell_nodes_gpu(cellIDs, nCells, nPtsCell, xyz);
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

int get_n_weights(int cellID)
{
  return ZEFR->get_n_weights(cellID);
}

void donor_frac_gpu(int* cellIDs, int nFringe, double* rst, double* weights)
{
  ZEFR->donor_frac_gpu(cellIDs, nFringe, rst, weights);
}

void convert_to_modal(int* cellID, int* nSpts, double* q_in, int* npts, int* index_out, double* q_out)
{
  //assert(*nSpts == *npts);
  *index_out = (*cellID) * (*nSpts);
  for (int spt = 0; spt < (*nSpts); spt++)
    q_out[spt] = q_in[spt];
}

void fringe_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double *data)
{
#ifdef _GPU
  ZEFR->fringe_data_to_device(fringeIDs, nFringe, gradFlag, data);
#endif
}

void unblank_data_to_device(int *fringeIDs, int nFringe, int gradFlag, double *data)
{
#ifdef _GPU
  ZEFR->unblank_data_to_device(fringeIDs, nFringe, gradFlag, data);
#endif
}

void signal_handler(int signum)
{
  if (signum == SIGINT)
    std::cout << "Received signal SIGINT - dumping solution to file" << std::endl;
  else if (signum == SIGTERM)
    std::cout << "Received signal SIGTERM - dumping solution to file and quitting" << std::endl;
  else
    exit(signum);

  ZEFR->write_solution();

  if (signum == SIGTERM)
    exit(signum);
}

} /* namespace zefr */
