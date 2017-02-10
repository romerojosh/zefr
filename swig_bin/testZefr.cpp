#include "zefr_interface.hpp"
#include "tiogaInterface.h"

#include "mpi.h"
#include <valgrind/callgrind.h>

int main(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  int gridID = 0;
  int nGrids = 2;

  //gridID = rank%nGrids; //(rank > (size/2));
  gridID = rank>0;

  /* Basic sphere test case */
  //nGrids = 2;
  //gridID = (rank >= size / nGrids);
  //if (rank <= 1) gridID = 0;
  //if (rank > 1) gridID = 1;

//  nGrids = 1;
//  gridID = 0;

  /* 2-sphere test case */
//  nGrids = 3;
//  if (rank == 0) gridID = 0;
//  if (rank == 1) gridID = 1;
//  if (rank > 1) gridID = 2;

  /* 2-Grid TSTO test case */
//  nGrids = 2;
//  if (rank <= 1) gridID = 0;
//  if (rank > 1) gridID = 1;

  /* 3-Grid TSTO test case */
//  nGrids = 3;
//  if (rank == 0) gridID = 0;
//  if (rank == 1) gridID = 1;
//  if (rank > 1) gridID = 2;

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

  // Setup the ZEFR solver object
  if (argc == 1)
  {
    char inputFile[] = "input_sphere";
    zefr::initialize(gridComm, inputFile, nGrids, gridID);
  }
  else
  {
    zefr::initialize(gridComm, argv[1], nGrids, gridID);
  }

  Zefr *z = zefr::get_zefr_object();
  InputStruct &inp = z->get_input();

  // Setup the TIOGA connectivity object
  tioga_init_(MPI_COMM_WORLD);

  if (nGrids > 1) inp.overset = 1;

  if (inp.motion && (inp.motion_type == RIGID_BODY || inp.motion_type == CIRCULAR_TRANS))
    z->set_rigid_body_callbacks(tioga_set_transform);

  /* NOTE: tioga_dataUpdate is now being called from within ZEFR, in order to
   * accomodate both multi-stage RK time stepping + viscous cases with gradient
   * data interpolation.  Likewise with moving grids and connectivity update */
  z->set_tioga_callbacks(tioga_preprocess_grids_, tioga_performconnectivity_,
      tioga_dataupdate_ab_send, tioga_dataupdate_ab_recv);

  z->setup_solver();

  BasicGeo geo = zefr::get_basic_geo_data();
  ExtraGeo geoAB = zefr::get_extra_geo_data();
  CallbackFuncs cbs = zefr::get_callback_funcs();

  Timer tg_time;
  tg_time.startTimer();

  if (rank == 0)
    std::cout << "Setting TIOGA callback functions..." << std::endl;

  tioga_registergrid_data_(geo.btag, geo.nnodes, geo.xyz, geo.iblank,
      geo.nwall, geo.nover, geo.wallNodes, geo.overNodes,
      geo.nCellTypes, geo.nvert_cell, geo.nCells_type, geo.c2v);

  tioga_setcelliblank_(geoAB.iblank_cell);

  tioga_register_face_data_(geoAB.f2c,geoAB.c2f,geoAB.iblank_face,
      geoAB.nOverFaces,geoAB.nMpiFaces,geoAB.overFaces,geoAB.mpiFaces,
      geoAB.procR,geoAB.mpiFidR,geoAB.nFaceTypes,geoAB.nvert_face,
      geoAB.nFaces_type,geoAB.f2v);

  tioga_set_highorder_callback_(cbs.get_nodes_per_cell, cbs.get_receptor_nodes,
      cbs.donor_inclusion_test, cbs.donor_frac, cbs.convert_to_modal);

  tioga_set_ab_callback_(cbs.get_nodes_per_face, cbs.get_face_nodes,
      cbs.get_q_spt, cbs.get_q_fpt, cbs.get_grad_spt, cbs.get_grad_fpt,
      cbs.get_q_spts, cbs.get_dq_spts);

  // If code was compiled to use GPUs, need additional callbacks
  if (zefr::use_gpus())
  {
    tioga_set_ab_callback_gpu_(cbs.donor_data_from_device,  cbs.fringe_data_to_device,
        cbs.unblank_data_to_device, cbs.get_q_spts_d, cbs.get_dq_spts_d);
#ifdef _GPU
    tioga_set_stream_handle(z->get_tg_stream_handle(), z->get_tg_event_handle());
#endif
  }

  if (nGrids > 1)
  {
    tioga_preprocess_grids_();
    tioga_performconnectivity_();
  }
  tg_time.stopTimer();

  // setup cell/face iblank data for use on GPU
  if (nGrids > 1 && zefr::use_gpus())
    z->update_iblank_gpu();

  if (inp.restart)
    z->restart_solution();

  // Output initial solution and grid
  if (!inp.restart)
    z->write_solution();

  MPI_Barrier(MPI_COMM_WORLD);

  // Run the solver loop now
  Timer runTime("ZEFR Compute Time: ");
  Timer tgTime("TIOGA Interp Time: ");
  inp.waitTimer.setPrefix("ZEFR MPI Time: ");

  CALLGRIND_START_INSTRUMENTATION;

  for (int iter = inp.initIter+1; iter <= inp.n_steps; iter++)
  {
    runTime.startTimer();
    z->do_step();
    runTime.stopTimer();

    if (inp.report_freq > 0 and (iter%inp.report_freq == 0 or iter == inp.initIter+1 or iter == inp.n_steps))
      z->write_residual();

    if (inp.write_freq > 0 and (iter%inp.write_freq == 0 or iter == inp.n_steps))
      z->write_solution();

    if (inp.force_freq > 0 and (iter%inp.force_freq == 0 or iter == inp.n_steps))
      z->write_forces();

    if (inp.error_freq > 0 and (iter%inp.error_freq == 0 or iter == inp.n_steps))
      z->write_error();
  }

  CALLGRIND_STOP_INSTRUMENTATION;

//  z->write_solution();

  std::cout << "Preprocessing/Connectivity Time: ";
  tg_time.showTime(2);

  MPI_Barrier(MPI_COMM_WORLD);
  tgTime.showTime(2);
  MPI_Barrier(MPI_COMM_WORLD);
  runTime.showTime(2);
  MPI_Barrier(MPI_COMM_WORLD);

  inp.waitTimer.showTime();

  zefr::finalize();
  tioga_delete_();

  MPI_Finalize();

  return 0;
}
