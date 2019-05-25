#include "zefr_interface.hpp"
#include "tiogaInterface.h"

#include "mpi.h"

void initialize_overset(Zefr *z, InputStruct &inp);
void setup_overset_data(Zefr *z, InputStruct &inp);
void setup_grid0_unblank(Zefr* z, InputStruct& inp);

int main(int argc, char *argv[])
{
  MPI_Init(&argc,&argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  int gridID = 0;
  int nGrids = 1;
  std::string inputFile;

  if (argc > 1)
    inputFile = argv[1];
  else
  {
    if (rank == 0)
    {
      std::cout << "\nUnblank all elements in the overset Grid 0 using data from other grids.\n" << std::endl;
      std::cout << "This is a single-use-case function to 'unblank' a single background" << std::endl;
      std::cout << "grid using interpolated data from all overlapping grids [primarily" << std::endl;
      std::cout << "for post-processing the overset Taylor-Green test case]." << std::endl;
      std::cout << "\nUsage:\n  " << argv[0] << " input_file <nproc_grid1> <nproc_grid2> ...\n" << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  if (argc > 2)
  {
    nGrids = argc - 2;
    for (int i = 0, sum = 0; i < nGrids && sum <= rank; i++)
    {
      sum += atoi(argv[i+2]);
      gridID = i;
    }
  }

  MPI_Comm gridComm;
  if (nGrids > 1)
    MPI_Comm_split(MPI_COMM_WORLD, gridID, rank, &gridComm);
  else
    gridComm = MPI_COMM_WORLD;

  // Setup the ZEFR solver object
  zefr::initialize(gridComm, inputFile.c_str(), nGrids, gridID, MPI_COMM_WORLD);

  Zefr *z = zefr::get_zefr_object();
  InputStruct &inp = z->get_input();

  if (nGrids < 2 || !inp.restart)
  {
    if (rank == 0)
    {
      std::cout << "Expecting to be given at least 2 grids and restart solution data." << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  initialize_overset(z, inp);

  z->setup_solver();

  setup_overset_data(z, inp);

  z->restart_solution();

  setup_grid0_unblank(z, inp);

  z->do_unblank();

  // setup cell/face iblank data for use on GPU
  if (nGrids > 1 && zefr::use_gpus())
    z->update_iblank_gpu();

  // Output initial solution and grid
  if (!inp.restart)
    z->write_solution();

  MPI_Barrier(MPI_COMM_WORLD);

  // Run the solver loop now
  for (int iter = inp.initIter+1; iter <= inp.n_steps; iter++)
  {
    if (!inp.adapt_dt)
    {
      // Can use the new-style method [no callbacks within ZEFR]
      for (int stage = 0; stage < inp.nStages; stage++)
      {
        z->do_rk_stage_start(iter,stage);

        if (nGrids > 1)
          tioga_dataupdate_ab(5, 0);

        z->do_rk_stage_mid(iter,stage);

        z->do_rk_stage_finish(iter,stage);
      }
    }
    else
    {
      // Adaptive time stepping -> stick to previous method
      z->do_step();
    }

    if (inp.tavg)
    {
      bool do_accum = (iter%inp.tavg_freq == 0 or iter == inp.initIter+1);
      bool do_write = (iter%inp.write_tavg_freq == 0 || iter == inp.n_steps);

      if (do_accum || do_write)
        z->update_averages();

      if (do_write)
        z->write_averages();
    }

    if (inp.report_freq > 0 and (iter%inp.report_freq == 0 or iter == inp.initIter+1 or iter == inp.n_steps))
      z->write_residual();

    if (inp.write_freq > 0 and (iter%inp.write_freq == 0 or iter == inp.n_steps))
      z->write_solution();

    if (inp.force_freq > 0 and (iter%inp.force_freq == 0 or iter == inp.n_steps))
      z->write_forces();

    if (inp.error_freq > 0 and (iter%inp.error_freq == 0 or iter == inp.n_steps))
      z->write_error();
  }

  zefr::finalize();

  if (nGrids > 1)
    tioga_delete_();

  MPI_Finalize();

  return 0;
}

void initialize_overset(Zefr* z, InputStruct& inp)
{
  // Setup the TIOGA connectivity object
  tioga_init_(MPI_COMM_WORLD);

  inp.overset = 1;

  /*! NOTE: The following 2 functions are not required [in fact, discouraged]
   *  when using the new HELIOS-compatible Python layer */

  if (inp.motion_type == RIGID_BODY || inp.motion_type == CIRCULAR_TRANS)
    z->set_rigid_body_callbacks(tioga_set_transform);

  /* NOTE: tioga_dataUpdate is being called from within ZEFR, to accomodate
   * multi-stage RK time stepping */
  z->set_tioga_callbacks(tioga_do_point_connectivity, tioga_unblank_part_1,
      tioga_unblank_part_2, tioga_dataupdate_ab_send, tioga_dataupdate_ab_recv);
}

void setup_overset_data(Zefr* z, InputStruct& inp)
{
  if (inp.grank == 0)
    std::cout << "Setting TIOGA callback functions..." << std::endl;

  BasicGeo geo = zefr::get_basic_geo_data();
  ExtraGeo geoAB = zefr::get_extra_geo_data();
  GpuGeo geoGpu = zefr::get_gpu_geo_data();
  CallbackFuncs cbs = zefr::get_callback_funcs();

  tioga_registergrid_data_(geo.btag, geo.nnodes, geo.xyz, geo.iblank,
                           geo.nwall, geo.nover, geo.wallNodes, geo.overNodes,
                           geo.nCellTypes, geo.nvert_cell, geo.nface_cell,
                           geo.nCells_type, geo.c2v);

  tioga_setcelliblank_(geoAB.iblank_cell);

  tioga_register_face_data_(geo.gridType,geoAB.f2c,geoAB.c2f,geoAB.iblank_face,
                            geoAB.nOverFaces,geoAB.nWallFaces,geoAB.nMpiFaces,
                            geoAB.overFaces,geoAB.wallFaces,geoAB.mpiFaces,
                            geoAB.procR,geoAB.mpiFidR,geoAB.nFaceTypes,
                            geoAB.nvert_face,geoAB.nFaces_type,geoAB.f2v);

  tioga_set_highorder_callback_(cbs.get_nodes_per_cell, cbs.get_receptor_nodes,
                                cbs.donor_inclusion_test, cbs.donor_frac, cbs.convert_to_modal);

  tioga_set_ab_callback_(cbs.get_nodes_per_face, cbs.get_face_nodes,
                         cbs.get_q_spt, cbs.get_q_fpt, cbs.get_grad_spt, cbs.get_grad_fpt,
                         cbs.get_q_spts, cbs.get_dq_spts);

  if (inp.motion)
    tioga_register_moving_grid_data(geoAB.grid_vel, geoAB.offset, geoAB.Rmat);

  // If code was compiled to use GPUs, need additional callbacks
  if (zefr::use_gpus())
  {
    tioga_set_ab_callback_gpu_(cbs.fringe_data_to_device, cbs.unblank_data_to_device,
                               cbs.get_q_spts_d, cbs.get_dq_spts_d,
                               cbs.get_face_nodes_gpu, cbs.get_cell_nodes_gpu,
                               cbs.get_n_weights, cbs.donor_frac_gpu);

    tioga_set_device_geo_data(geoGpu.coord_nodes,geoGpu.coord_eles,geoGpu.iblank_cell,
                              geoGpu.iblank_face);

    tioga_set_stream_handle(z->get_tg_stream_handle(), z->get_tg_event_handle());
  }

  tioga_preprocess_grids_();
  tioga_performconnectivity_();
}

void setup_grid0_unblank(Zefr* z, InputStruct& inp)
{
  if (inp.grank == 0)
    std::cout << "Setting all Grid 0 elements to 'normal'" << std::endl;

  BasicGeo geo = zefr::get_basic_geo_data();
  ExtraGeo geoAB = zefr::get_extra_geo_data();

  if (inp.gridID == 0)
  {
    for (int i = 0; i < geo.nCellsTot; i++)
      geoAB.iblank_cell[i] = 1;
  }
}
