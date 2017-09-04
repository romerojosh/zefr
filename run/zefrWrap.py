import sys
import os
TIOGA_DIR = '/home/jcrabill/tioga/bin/'
ZEFR_DIR = '/home/jcrabill/zefr/bin/'
sys.path.append(TIOGA_DIR)
sys.path.append(ZEFR_DIR)
#TIOGA_DIR = '/home/jacob/tioga/bin/'
#ZEFR_DIR = '/home/jacob/zefr/bin/'
#sys.path.append(TIOGA_DIR)
#sys.path.append(ZEFR_DIR)

from mpi4py import MPI
import numpy as np
import zefr

Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
nproc = Comm.Get_size()

# To run an overset case, set the number of grids here,
# and specify the grid-to-rank association
nGrids = 2
inputFile = "input_sphere"

if len(sys.argv) == 2:
    inputFile = sys.argv[1]
elif len(sys.argv) > 2:
    inputFile = sys.argv[1]
    nGrids = len(sys.argv) - 2
    nRanksGrid = []
    rankSum = 0
    for i in range(0,nGrids):
        rankSum += int(sys.argv[i+2])
        if rankSum > rank:
            GridID = i;
            break;

if nGrids > 1:
    import tioga as tg
    gridComm = Comm.Split(GridID,rank)
else:
    GridID = 0
    gridComm = Comm

gridRank = gridComm.Get_rank()
gridSize = gridComm.Get_size()

# -------------- Setup the ZEFR solver object; process the grids ----------------
zefr.initialize(gridComm,inputFile,nGrids,GridID)

z = zefr.get_zefr_object()
inp = z.get_input()
data = z.get_data()

if nGrids > 1:
    tg.tioga_init_(Comm)

    inp.overset = 1

    z.set_tioga_callbacks(tg.tioga_preprocess_grids_, 
        tg.tioga_performconnectivity_, tg.tioga_do_point_connectivity,
        tg.tioga_set_iter_iblanks, tg.tioga_unblank_part_1, tg.tioga_unblank_part_2,
        tg.tioga_dataupdate_ab_send, tg.tioga_dataupdate_ab_recv)

    if inp.motion_type == zefr.RIGID_BODY or inp.motion_type == zefr.CIRCULAR_TRANS:
        z.set_rigid_body_callbacks(tg.tioga_set_transform)

z.setup_solver()

# Setup the TIOGA object; prepare to receive grid data
if nGrids > 1:
    # Get geo data from ZEFR and pass into TIOGA
    geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
    geoAB = zefr.get_extra_geo_data() # Geo data for AB method
    cbs = zefr.get_callback_funcs()   # Callback functions for high-order/AB method
    
    tg.tioga_registergrid_data_(geo.btag, geo.nnodes, geo.xyz, geo.iblank,
        geo.nwall, geo.nover, geo.wallNodes, geo.overNodes,
        geo.nCellTypes, geo.nvert_cell, geo.nCells_type, geo.c2v)
    
    tg.tioga_setcelliblank_(geoAB.iblank_cell)
    
    tg.tioga_register_face_data_(geo.gridType, geoAB.f2c, geoAB.c2f,
        geoAB.iblank_face, geoAB.nOverFaces, geoAB.nWallFaces, 
        geoAB.nMpiFaces, geoAB.overFaces, geoAB.wallFaces,
        geoAB.mpiFaces, geoAB.procR, geoAB.mpiFidR, geoAB.nFaceTypes, 
        geoAB.nvert_face, geoAB.nFaces_type, geoAB.f2v);
  
    tg.tioga_set_highorder_callback_(cbs.get_nodes_per_cell,
        cbs.get_receptor_nodes, cbs.donor_inclusion_test,
        cbs.donor_frac, cbs.convert_to_modal)
    
    tg.tioga_set_ab_callback_(cbs.get_nodes_per_face, cbs.get_face_nodes,
        cbs.get_q_spt, cbs.get_q_fpt, cbs.get_grad_spt, cbs.get_grad_fpt,
        cbs.get_q_spts, cbs.get_dq_spts)

    if inp.motion:
        tg.tioga_register_moving_grid_data(geoAB.grid_vel)
    
    # Callback functions for TIOGA to access zefr's data on the device
    if zefr.use_gpus():
        if gridRank == 0:
            print("Grid "+str(GridID)+": Setting GPU callback functions")
        tg.tioga_set_ab_callback_gpu_(cbs.donor_data_from_device,
            cbs.fringe_data_to_device, cbs.unblank_data_to_device, 
            cbs.get_q_spts_d, cbs.get_dq_spts_d)
        tg.tioga_set_stream_handle(z.get_tg_stream_handle(), z.get_tg_event_handle())

    # Perform overset connectivity / hole blanking
    if gridRank == 0:
        print("Grid "+str(GridID)+": Beginning connectivity...")
    tgTime = zefr.Timer("TIOGA Time: ")
    tgTime.startTimer()

    tg.tioga_preprocess_grids_()
    tg.tioga_performconnectivity_()

    if gridRank == 0:
        print("Grid "+str(GridID)+": Connectivity complete.")
    tgTime.stopTimer()
    tgTime.showTime()

    if zefr.use_gpus():
        z.update_iblank_gpu()

# ----------------------- TESTING HELIOS COMPATIBILITY LAYER ----------------------- 

ncells = geo.nCells_type
print('ncells = ',ncells)

geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
geoAB = zefr.get_extra_geo_data() # Geo data for AB method
xyz = zefr.ptrToArray(geo.xyz, geo.nnodes, 3)
xyz = zefr.ptrToArray(geo.xyz, geo.nnodes, 3)
overFaces = zefr.ptrToArray(geoAB.overFaces, geoAB.nOverFaces)
c2f = zefr.ptrToArray(geoAB.c2f, ncells, 6)
print('c2f', type(c2f), c2f.dtype, c2f.shape)
print('xyz', type(xyz), xyz.dtype, xyz.shape)
print('overFaces', type(overFaces), overFaces.shape)
print(c2f)
print(xyz)
print(overFaces)

# Test the other way around: numpy array to C pointer
#xyzptr = zefr.arrayToDblPtr(xyz)
#print('xyzptr',type(xyzptr),xyzptr)
#print('org_xyzptr',type(geo.xyz),geo.xyz)

#c2fptr = zefr.arrayToIntPtr(c2f)
#print('c2fptr',type(c2fptr),c2fptr)
#print('org_c2fptr',type(geoAB.c2f),geoAB.c2f)

# Test reshaping
np.reshape(c2f, (ncells,6))
print(c2f.shape)

# ------------------------------- Run the solver -------------------------------
z.write_solution()

Comm.Barrier

for iter in range(1,inp.n_steps+1):
    z.do_step()

    if iter%inp.report_freq == 0 or iter==1 or iter==inp.n_steps:
        z.write_residual()
    if iter%inp.write_freq == 0 or iter==0 or iter==inp.n_steps:
        z.write_solution()
    if inp.force_freq > 0:
        if iter%inp.force_freq == 0 or iter==inp.n_steps:
            z.write_forces()
    if inp.error_freq > 0:
        if iter%inp.error_freq == 0 or iter==inp.n_steps:
            z.write_error()

# --------------------------- Finalize - free memory ---------------------------
if gridRank == 0:
    print("Finishing run...")
zefr.finalize()
if nGrids > 1:
  tg.tioga_delete_()
