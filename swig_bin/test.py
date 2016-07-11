import sys
import os
TIOGA_DIR = os.getcwd() + '/TIOGA/'
sys.path.append(TIOGA_DIR)

from mpi4py import MPI
import zefr
import tioga as tg

Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
nproc = Comm.Get_size()

GridID = (rank%2) # Simple grid splitting for 2 grids
#GridID = 0

gridComm = Comm.Split(GridID,rank)
gridRank = gridComm.Get_rank()
gridSize = gridComm.Get_size()

# Setup the ZEFR solver object; process the grids
inputFile = "input_sphere"
zefr.initialize(gridComm,inputFile,2,GridID)
z = zefr.get_zefr_object()
z.setup_solver()

# Setup the TIOGA object; prepare to receive grid data
tg.tioga_init_(Comm)

# Get geo data from ZEFR and pass into TIOGA
geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
geoAB = zefr.get_extra_geo_data() # Geo data for AB method
cbs = zefr.get_callback_funcs()   # Callback functions for high-order/AB method
U_spts = zefr.get_q_spts()
U_fpts = zefr.get_q_fpts()

tg.tioga_registergrid_data_(geo.btag, geo.nnodes, geo.xyz, geo.iblank,
        geo.nwall, geo.nover, geo.wallNodes, geo.overNodes,
        geo.nCellTypes, geo.nvert_cell, geo.nCells_type, geo.c2v)

tg.tioga_setcelliblank_(geoAB.iblank_cell)

tg.tioga_register_face_data_(geoAB.f2c,geoAB.c2f,geoAB.iblank_face,
        geoAB.nOverFaces,geoAB.nMpiFaces,geoAB.overFaces,geoAB.mpiFaces,
        geoAB.procR,geoAB.mpiFidR,geoAB.nFaceTypes,geoAB.nvert_face,
        geoAB.nFaces_type,geoAB.f2v);

tg.tioga_set_highorder_callback_(cbs.get_nodes_per_cell,
        cbs.get_receptor_nodes, cbs.donor_inclusion_test,
        cbs.donor_frac, cbs.convert_to_modal)

tg.tioga_set_ab_callback_(cbs.get_nodes_per_face, cbs.get_face_nodes,
        cbs.get_q_index_face, cbs.get_q_spt)

# Perform overset connectivity / hole blanking
print "Beginning connectivity"
tg.tioga_preprocess_grids_()
tg.tioga_performconnectivity_()
print "Connectivity done."

Comm.Barrier()

# Run the solver
z.write_solution()
z.write_residual()

Comm.Barrier()

for iter in range(1,1000):
    tg.tioga_dataupdate_ab(5,U_spts,U_fpts)
    z.do_step()
    if iter%200 == 0:
        z.write_residual()
        z.write_solution()

# Finalize - free memory
print "Finishing run..."
zefr.finalize()
tg.tioga_delete_()
