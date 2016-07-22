#include <valgrind/callgrind.h>

#include "zefr_interface.hpp"
#include "tiogaInterface.h"

#include "mpi.h"

int main(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);
CALLGRIND_STOP_INSTRUMENTATION;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  int gridID = 0;
  int nGrids = 2;
  if (size == 8)
  {
    gridID = (rank >= 3);
  }
  else if (size == 3)
  {
    gridID = (rank > 0);
  }
  else if (size == 1)
  {
    gridID = 1;
    nGrids = 1;
  }
  else
  {
    gridID = rank%nGrids; //(rank > (size/2));
  }

  gridID = rank>0;
//  // 2-sphere test case
//  nGrids = 3;
//  if (rank == 0) gridID = 0;
//  if (rank == 1) gridID = 1;
//  if (rank > 1) gridID = 2;

  bool sphereTest = true;
  if (sphereTest)
  {
    if (nGrids == 2)
      gridID = (rank>0);
    else if (nGrids == 3)
      gridID = (rank>0);
  }

  MPI_Comm gridComm;

  bool oneGrid = false;
  if (oneGrid)
  {
    nGrids = 1; gridID = 0;
    gridComm = MPI_COMM_WORLD;
  }
  else
  {
    MPI_Comm_split(MPI_COMM_WORLD, gridID, rank, &gridComm);
  }
  cout << "Rank " << rank << ", GridID = " << gridID << ", nproc = " << size << endl;

  // Setup the ZEFR solver object
  char inputFile[] = "input_sphere";
  zefr::initialize(gridComm, inputFile, nGrids, gridID);

  Zefr *z = zefr::get_zefr_object();
  z->setup_solver();

  BasicGeo geo = zefr::get_basic_geo_data();
  ExtraGeo geoAB = zefr::get_extra_geo_data();
  CallbackFuncs cbs = zefr::get_callback_funcs();
  InputStruct &inp = z->get_input();

  if (nGrids > 1) inp.overset = 1;

  double *U_spts = zefr::get_q_spts();
  double *U_fpts = zefr::get_q_fpts();

  Timer tg_time;
  tg_time.startTimer();

  // Setup the TIOGA connectivity object
  tioga_init_(MPI_COMM_WORLD);

  tioga_registergrid_data_(geo.btag, geo.nnodes, geo.xyz, geo.iblank,
      geo.nwall, geo.nover, geo.wallNodes, geo.overNodes,
      geo.nCellTypes, geo.nvert_cell, geo.nCells_type, geo.c2v);

  tioga_setcelliblank_(geoAB.iblank_cell);

  tioga_register_face_data_(geoAB.f2c,geoAB.c2f,geoAB.iblank_face,
      geoAB.nOverFaces,geoAB.nMpiFaces,geoAB.overFaces,geoAB.mpiFaces,
      geoAB.procR,geoAB.mpiFidR,geoAB.nFaceTypes,geoAB.nvert_face,
      geoAB.nFaces_type,geoAB.f2v);

  tioga_set_highorder_callback_(cbs.get_nodes_per_cell,
      cbs.get_receptor_nodes, cbs.donor_inclusion_test,
      cbs.donor_frac, cbs.convert_to_modal);

  tioga_set_ab_callback_(cbs.get_nodes_per_face, cbs.get_face_nodes,
      cbs.get_q_index_face, cbs.get_q_spt);


  if (nGrids > 1)
  {
    tioga_preprocess_grids_();
    tioga_performconnectivity_();
  }
  tg_time.stopTimer();

  // Output initial solution and grid
  z->write_solution();

  MPI_Barrier(MPI_COMM_WORLD);
  CALLGRIND_START_INSTRUMENTATION;

  // Run the solver loop now
  Timer runTime("Compute Time: ");
  Timer tgTime("Interp Time: ");
  Timer mpiTime("MPI Wait Time: ");
  inp.waitTimer = mpiTime;

  for (int iter = 1; iter <= inp.n_steps; iter++)
  {
    tgTime.startTimer();
    if (nGrids > 1)
      tioga_dataupdate_ab(5,U_spts,U_fpts);
    tgTime.stopTimer();

    runTime.startTimer();
    z->do_step();
    runTime.stopTimer();

    if (iter%inp.report_freq == 0 or iter == 1 or iter == inp.n_steps)
      z->write_residual();

    if (iter%inp.write_freq == 0 or iter == 0 or iter == inp.n_steps)
      z->write_solution();

//    if (inp.force_freq > 0 and (iter%inp.force_freq == 0 or iter == inp.n_steps))
//      z->write_forces();

//    if (inp.error_freq > 0 and (iter%inp.error_freq == 0 or iter == inp.n_steps))
//      z->write_error();
  }
  CALLGRIND_STOP_INSTRUMENTATION;

  z->write_solution();

//  if (rank == 0)
//  {
    std::cout << "Preprocessing/Connectivity Time: ";
    tg_time.showTime(2);

    tgTime.showTime(2);
    runTime.showTime(2);
//  }

    inp.waitTimer.showTime();

  zefr::finalize();
  tioga_delete_();

  MPI_Finalize();

  return 0;
}
