# Simplified Python interface to ZEFR 
# For use with external domain connectivity package
# 
# Written Jun 2017 by Jacob Crabill
# Aerospace Computing Lab, Stanford University
import sys
import os
import numpy
from mpi4py import MPI
#from numpy import *
import numpy as np

import zefr

class zefrSolver:

    def __init__(self, inputFile, gridID = 0, nGrids = 1):
        self.inputFile = inputFile
        self.name = 'zefr'

        if not os.path.isfile(inputFile):
          print("ERROR: Input file not found ["+inputFile+"]")
          return None

        self.worldComm = MPI.COMM_WORLD;
        self.worldComm.Barrier()

        # Create a sub-communicatory specific to our grid
        self.grank = self.worldComm.Get_rank()
        self.nproc = self.worldComm.Get_size()

        if nGrids > 1:
            self.gridComm = Comm.Split(gridID,self.rank)

        # Initialize the ZEFR solver object
        zefr.initialize(self.gridComm,self.input,nGrids,gridID,self.worldComm)
        
        # Get the ZEFR solver object for more direct access
        self.z = zefr.get_zefr_object()

        # Get the struct of input/solver parameters
        self.inp = z.get_input()

    def set_callbacks(self, tg_preprocess, tg_performconn, tg_pointconn,
            tg_iter_iblank, tg_ab_send_data, tg_ab_recv_data, tg_set_transform):
        
        self.z.set_tioga_callbacks(tg_preprocess, tg_performconn, tg_pointconn,
                tg_iter_iblank, tg_ab_send_data, tg_ab_recv_data)

        if self.inp.motion_type == zefr.RIGID_BODY or self.inp.motion_type == zefr.CIRCULAR_TRANS:
            self.z.set_rigid_body_callbacks(tg_set_transform)


    def get_callbacks(self,callbacks):
        # Get geo data from ZEFR and pass into TIOGA
        geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
        geoAB = zefr.get_extra_geo_data() # Geo data for AB method
        cbs = zefr.get_callback_funcs()   # Callback functions for high-order/AB method
    
        # NOTE: Python/SWIG can give you pointers to the underlying struct objects
        # via the *.this attribute (i.e. ptr = geoAB.this)

        # TODO: get more details on how to pass these out to TIOGA / PUNDIT
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
            cbs.get_q_spt, cbs.get_q_fpt, cbs.get_grad_spt, cbs.get_grad_fpt,
            cbs.get_q_spts, cbs.get_dq_spts)

        if self.inp.motion:
            tg.tioga_register_moving_grid_data(geoAB.grid_vel)
    
        # Callback functions for TIOGA to access ZEFR's data on the device
        if zefr.use_gpus():
            if gridRank == 0:
                print("Grid "+str(GridID)+": Setting GPU callback functions")
            tg.tioga_set_ab_callback_gpu_(cbs.donor_data_from_device,
                cbs.fringe_data_to_device, cbs.unblank_data_to_device, 
                cbs.get_q_spts_d, cbs.get_dq_spts_d)
            tg.tioga_set_stream_handle(z.get_tg_stream_handle(), z.get_tg_event_handle())

    def init_solver(self,callbacks):
        # Need to finish some TIOGA / domain-connectivity setup before fully
        # initializing the ZEFR solver object
        if nGrids > 1:
            tg.tioga_init_(Comm)

            self.inp.overset = 1

            self.z.set_tioga_callbacks(tg.tioga_preprocess_grids_, 
                tg.tioga_performconnectivity_, tg.tioga_do_point_connectivity,
                tg.tioga_set_iter_iblanks, tg.tioga_dataupdate_ab_send, tg.tioga_dataupdate_ab_recv)

            if self.inp.motion_type == zefr.RIGID_BODY or self.inp.motion_type == zefr.CIRCULAR_TRANS:
                self.z.set_rigid_body_callbacks(tg.tioga_set_transform)

        self.z.setup_solver()

        # This should probably go somewhere else...? (In terms of encapsulation/organization)
        tg.tioga_preprocess_grids_()
        tg.tioga_performconnectivity_()

        if (self.inp.restart)
            self.z.restart_solution()

        if zefr.use_gpus():
            z.update_iblank_gpu()

    def restart_solution(self):
        if (self.inp.restart)
            self.z.restart_solution()

    def doTimeStep(self):
        self.z.do_step()

    def printResiduals(self):
        self.z.write_residual()

    def writeSolution(self):
        self.z.write_solution()

    def printForces(self):
        self.z.write_forces()

    def printTestCaseError(self):
        if self.inp.error_freq > 0:
            self.z.write_error()



