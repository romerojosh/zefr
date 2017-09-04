#! /usr/bin/env python
#
# File: zefrInterface.py
# Authors: Jacob Crabill
# Last Modified: 9/02/2017


__version__ = "1.00"

# ==================================================================
# Standard Python modules

import sys
import os
import copy
import string
import types

#Extension modules
#sys.path.append(os.path.abspath(os.getenv('PYTHONPATH')))
#sys.path.append(os.path.abspath(os.getenv('PYTHONPATH')+'/numpy'))

import numpy as np

CONVERT_DIR = '/home/jcrabill/zefr/external/'
sys.path.append(CONVERT_DIR)

from convert import *

# Try to import the MPI.COMM_WORLD.module
try:
    from mpi4py import MPI
    _parallel = True

except ImportError:
    _parallel = False


try:
    import zefr
except ImportError:
    print("Import Error: ZEFR solver module")

class zefrSolver:

    def __init__(self,startfile,gridID,nGrids):
        if not os.path.isfile(startfile):
          print('Error: ZEFR input file',startfile,'not found in path')
          raise

        self.nGrids = nGrids
        self.ID = gridID
        self.inpfile=startfile

        Comm = MPI.COMM_WORLD
        rank = Comm.Get_rank()
        nproc = Comm.Get_size()

        self.gridComm = Comm.Split(gridID,rank)
        self.gridRank = self.gridComm.Get_rank()
        self.gridSize = self.gridComm.Get_size()

        # Create the directory where all ZEFR IO will occur
        if self.gridRank == 0 and os.path.isdir('zefr') == False:
          os.mkdir('zefr')

        zefr.initialize(self.gridComm,startfile,nGrids,gridID)

        # scaling for solver variable [i.e. non-dimensionalization]
        self.scale = np.array([1.,1.,1.,1.,1.],'d')

        self.z = zefr.get_zefr_object()
        self.inp = self.z.get_input()
        self.simdata = self.z.get_data()
        self.name = 'zefr'

    # Run one full time step
    def runStep(self,iter,nstages):
        os.chdir('zefr')

        #self.z.do_step(nstages)
        for i in range(0,nstages):
            runSubSteps(iter,i,nstages)

        os.chdir('..')

    # Run one RK stage
    def runSubStepStart(self,iter,stage):
        self.z.do_rk_stage_start(iter,stage)

    def runSubStepFinish(self,iter,stage):
        self.z.do_rk_stage_finish(iter,stage)

    def sifInitialize(self,properties,conditions):
        os.chdir('zefr') 

        self.useGpu = properties['use-gpu']

        rinf = properties['rinf']
        ainf = properties['ainf']

        gridScale = conditions['meshRefLength']
        Reref = conditions['reyNumber']
        Lref = conditions['reyRefLength']
        Mref = conditions['refMach']

        self.inp.rho_fs = rinf
        self.inp.L_fs = conditions['reyRefLength']

        # All conditions come in as SI units - take care of non-dim here
        self.dt = ainf * conditions['dt'] / gridScale
        mach_fs = conditions['Mach']
        self.restart = 'no'
        if conditions['from_restart'] == 'yes':
            self.inp.restart = 'true'
            self.inp.restartIter = conditions['restartstep']

        Re = (Reref/Lref) * gridScale / Mref * (mach_fs)
        self.forceDim = 0.5 * rinf * (Mref*ainf*gridScale)**2
        self.momDim = self.forceDim * gridScale

        # Have ZEFR do any additional setup on input parameters
        #zefr.sifinit(dt,Re,mach_fs)
            
        self.scale = np.array([1./rinf,
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf*ainf),1.],'d')

        os.chdir('..')

    def getForcesAndMoments(self):
        self.z.get_forces()
        bodyForce = np.array(ptrToArray(self.simdata.forces))
        bodyForce[:3] = bodyForce[:3]*self.forceDim
        bodyForce[3:] = bodyForce[3:]*self.momDim
        return bodyForce
                        
    # sifInit called first to setup solver input
    def initData(self):
        os.chdir('zefr')

        # TODO: modify ZEFR so callbacks aren't needed yet? 
        # TODO: Or should I get callbacks here or in sifInit?

        # Setup the ZEFR solver 
        self.z.setup_solver()

        # Get all relevant geometry and solver data
        geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
        geoAB = zefr.get_extra_geo_data() # Geo data for AB method
        cbs = zefr.get_callback_funcs()   # Callback functions for high-order/AB method
        simdata = self.simdata

        # -----------------------------------------------------------------------
        # Turn all pointers from ZEFR into (single-element) lists of numpy arrays
        # (For 2D/3D arrays, can call reshape() after the fact)
        # -----------------------------------------------------------------------
        ndims = 3
        nfields = simdata.nfields
        nspts = simdata.nspts
        ncells = geo.nCells_type
        nfaces = geoAB.nFaces_type
        nnodes = geo.nnodes
        nvert = geo.nvert_cell
        nvertf = geoAB.nvert_face
        nfaces = geoAB.nFaces_type

        # Wrap all geometry data
        # NOTE: numpy arrays are row-major (last dim contiguous)
        xyz = [ptrToArray(geo.xyz, nnodes, ndims)]
        c2v = [ptrToArray(geo.c2v, ncells, nvert)]
        wallNodes = [ptrToArray(geo.wallNodes, geo.nwall)]
        overNodes = [ptrToArray(geo.overNodes, geo.nover)]

        iblank = [ptrToArray(geo.iblank, nnodes)]
        iblank_cell = [ptrToArray(geoAB.iblank_cell, ncells)]
        iblank_face = [ptrToArray(geoAB.iblank_face, nfaces)]
        f2v = [ptrToArray(geoAB.f2v, nfaces, nvertf)]
        c2f = [ptrToArray(geoAB.c2f, ncells, (2**ndims))]
        f2c = [ptrToArray(geoAB.f2c, nfaces, 2)]

        overFaces = [ptrToArray(geoAB.overFaces, geoAB.nOverFaces)]
        wallFaces = [ptrToArray(geoAB.wallFaces, geoAB.nWallFaces)]
        mpiFaces = [ptrToArray(geoAB.mpiFaces, geoAB.nMpiFaces)]
        procR = [ptrToArray(geoAB.procR, geoAB.nMpiFaces)]
        mpiFidR = [ptrToArray(geoAB.mpiFidR, geoAB.nMpiFaces)]

        # Wrap relevant solution data
        # NOTE: for GPU, solution is padded to 128 bytes using nElesPad
        q = [ptrToArray(simdata.u_spts, nspts,nfields,ncells)]

        dq = []
        if self.inp.viscous:
          dq.append(ptrToArray(simdata.dq_spts, ndims,nspts,nfields,ncells))

        self.gridData = {'gridtype' : 'unstructured',
                         'gridCutType' : geo.gridType,
                         'tetConn' : 'None',
                         'pyraConn' : 'None',
                         'prismConn' : 'None',
                         'hexaConn' : c2v,
                         'bodyTag' : [geo.btag],
                         'wallnode' : wallNodes,
                         'obcnode' : overNodes,
                         'grid-coordinates' : xyz,
                         'q-variables' : q,
                         'dq-variables' : dq,
                         'iblanking' : iblank,
                         'scaling' : [self.scale],
                         'face2cell' : f2c,
                         'cell2face' : c2f,
                         'iblank-face' : iblank_face,
                         'iblank-cell' : iblank_cell,
                         'overset-faces' : overFaces,
                         'wall-faces' : wallFaces,
                         'mpi-faces' : mpiFaces,
                         'mpi-right-proc' : procR,
                         'mpi-right-id' : mpiFidR,
                         'nvert-face' : [geoAB.nvert_face],
                         'faceConn' : f2v}
                         #'iblkHasNBHole':0,
                         #'istor':'row'}
                         #'fsicoord':self.fsicoord,
                         #'fsiforce':self.fsiforce,
                         #'fsitag':self.fsitag}  

        if self.inp.motion:
            gridVel = [ptrToArray(geoAB.grid_vel, nnodes,ndims)]
            offset = [ptrToArray(geoAB.offset, ndims)]
            Rmat = [ptrToArray(geoAB.Rmat, ndims,ndims)]

            self.gridData.update({'gridVel':grid_vel,
                'rigidOffset':offset,
                'rigidRotMat':Rmat})

        self.callbacks = {}

        #self.oldCallbacks = {'nodesPerCell': cbs.get_nodes_per_cell,
        self.callbacks.update({'nodesPerCell': cbs.get_nodes_per_cell,
            'receptorNodes': cbs.get_receptor_nodes,
            'donorInclusionTest': cbs.donor_inclusion_test,
            'donorFrac': cbs.donor_frac,
            'convertToModal': cbs.convert_to_modal})

        self.callbacks.update({'nodesPerFace': cbs.get_nodes_per_face,
            'faceNodes': cbs.get_face_nodes,
            'get_q_spt': cbs.get_q_spt,
            'get_q_fpt': cbs.get_q_fpt,
            'get_dq_spt': cbs.get_grad_spt,
            'get_dq_fpt': cbs.get_grad_fpt,
            'get_q_spts': cbs.get_q_spts,
            'get_dq_spts': cbs.get_dq_spts})

        if self.useGpu:
            self.callbacks.update({'donorDataDevice': cbs.donor_data_from_device,
                'fringeDataToDevice': cbs.fringe_data_to_device,
                'unblankToDevice': cbs.unblank_data_to_device,
                'faceNodesGPU': cbs.get_face_nodes_gpu,
                'cellNodesGPU': cbs.get_cell_nodes_gpu,
                'q_spts_d': cbs.get_q_spts_d,
                'dq_spts_d': cbs.get_dq_spts_d,
                'nWeightsGPU': cbs.get_n_weights,
                'weightsGPU': cbs.donor_frac_gpu})

            geoGpu = zefr.get_gpu_geo_data();
            self.gridData.update({'nodesGPU': geoGpu.coord_nodes,
                'eleCoordsGPU': geoGpu.coord_eles,
                'iblankCellGPU': geoGpu.iblank_cell,
                'iblankFaceGPU': geoGpu.iblank_face})

            cuStream = self.z.get_tg_stream_handle()
            cuEvent = self.z.get_tg_event_handle()
            self.gridData.update({'cuStream': cuStream, 'cuEvent': cuEvent})

        os.chdir('..')

        return (self.gridData, self.callbacks)
    
    def reportResidual(self,istep):
        os.chdir('zefr')
        self.z.report_residual()
        os.chdir('..')

    def writePlotData(self,istep):
        os.chdir('zefr')
        self.z.write_residual()
        os.chdir('..')

    def writeRestartData(self,istep):
        os.chdir('zefr')
        self.z.write_solution()
        os.chdir('..')

    # HELIOS can provide grid velocities
    def setGridSpeeds_maneuver(self,MeshMotionData):
        self.vx = -MeshMotionData['aircraftTranslation'][0]/self.ainf
        self.vy = -MeshMotionData['aircraftTranslation'][1]/self.ainf
        self.vz = -MeshMotionData['aircraftTranslation'][2]/self.ainf
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("gust speed from turns :",self.vx,self.vy,self.vz)

    def deformPart1(self,time,iter):
        self.z.move_grid_next(time+self.dt)

    def deformPart2(self,time,iter):
        self.z.move_grid(time)

    def moveGrid(self,iter,stage):
        self.z.move_grid(iter,stage)

    def updateBlankingGpu(self):
        self.z.update_iblank_gpu()

    def finish(self,step):
        istep = 0
        self.writeRestartData(step)
