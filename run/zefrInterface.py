#! /usr/bin/env python
#
# File: zefrInterface.py
# Authors: Jacob Crabill
# Last Modified: 6/4/2018


__version__ = "1.00"

# ==================================================================
# Standard Python modules

import sys
import os
import copy
import string
import types

import numpy as np

ZEFR_DIR = '/p/home/jcrabill/zefr/'

# For 'convert' module: pointer/array conversion utilities
sys.path.append(ZEFR_DIR + '/external/')
# For ZEFR module
sys.path.append(ZEFR_DIR + '/bin/')

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
    quit()

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

        zefr.initialize(self.gridComm,startfile,nGrids,gridID,Comm)

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

    def runSubStepMid(self,iter,stage):
        self.z.do_rk_stage_mid(iter,stage)

    def runSubStepFinish(self,iter,stage):
        self.z.do_rk_stage_finish(iter,stage)

    def sifInitialize(self,properties,conditions):
        os.chdir('zefr') 

        self.useGpu = properties['use-gpu']
        self.nStages = int(properties['nstages'])

        self.inp.nStages = self.nStages
        self.inp.viscous = bool(properties['viscous'])
        self.inp.motion = bool(properties['moving-grid'])
        if not self.inp.motion:
            self.inp.motion_type = 0

        self.inp.write_freq = int(properties['plot-freq'])
        self.inp.report_freq = int(properties['report-freq'])
        self.inp.force_freq = int(properties['force-freq'])

        self.inp.restart = False
        if properties['from_restart'] == 'yes':
            self.inp.restart = True;
            self.inp.restart_iter = int(properties['restartstep'])

        rinf = properties['rinf']
        ainf = properties['ainf']

        gridScale = conditions['meshRefLength']
        Reref = conditions['reyNumber']
        Lref = conditions['reyRefLength']
        mach_fs = conditions['Mach']

        mu = conditions['viscosity']
        gamma = conditions['gamma']
        prandtl = conditions['prandtl']

        self.inp.dt = conditions['dt']
        self.inp.adapt_dt = (properties['adapt-dt'] == 1)
        if 'max-dt' in properties:
            self.inp.max_dt = properties['max-dt']

        # All conditions come in as SI units - take care of non-dim here
        ainf *= gridScale
        dt = ainf * conditions['dt'] / gridScale

        v_fs = mach_fs * ainf
        self.forceDim = 1. / (0.5 * rinf * (v_fs*gridScale)**2)
        self.momDim = 1. / (self.forceDim * gridScale)

        # Update ZEFR's input parameters with non-dimensionalized quantities
        self.inp.dt = dt
        self.inp.rho_fs = rinf
        self.inp.mach_fs = mach_fs
        self.inp.v_mag_fs = v_fs
        self.inp.Re_fs = Reref
        self.inp.L_fs = Lref

        self.inp.mu = rinf * v_fs * Lref / Reref
        self.inp.gamma = gamma
        self.inp.prandtl = prandtl

        # Have ZEFR do any additional setup on input parameters
        self.z.init_inputs()
            
        self.scale = np.array([1./rinf,
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf*ainf),1.],'d')

        os.chdir('..')
                        
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
        simdata = self.simdata            # Access to forces, solution/gradient data

        # -----------------------------------------------------------------------
        # Turn all pointers from ZEFR into (single-element) lists of numpy arrays
        # (For 2D/3D arrays, can call reshape() after the fact)
        # -----------------------------------------------------------------------
        ndims = 3
        nfields = simdata.nfields
        nspts = simdata.nspts
        netypes = geo.nCellTypes
        ncells = geo.nCells_type
        ncellsTot = geo.nCellsTot
        nface = geo.nface_cell
        nftypes = geoAB.nFaceTypes
        nfaces = geoAB.nFaces_type
        nfacesTot = geoAB.nFacesTot
        nnodes = geo.nnodes
        nvert = geo.nvert_cell
        nvertf = geoAB.nvert_face
        nfaces = geoAB.nFaces_type

        # Wrap all geometry data
        # NOTE: numpy arrays are row-major (last dim contiguous)
        xyz = ptrToArray(geo.xyz, nnodes, ndims)

        nCells    = ptrToArray(ncells, netypes)
        nFaces    = ptrToArray(nfaces, nftypes)
        nVertCell = ptrToArray(nvert, netypes)
        nFaceCell = ptrToArray(geo.nface_cell, netypes)

        c2v = ptrToArray(geo.c2v, netypes)
        c2f = ptrToArray(geoAB.c2f, netypes)

        tet2v, pyr2v, pri2v, hex2v = [], [], [], []  # Elements to nodes conn
        tet2f, pyr2f, pri2f, hex2f = [], [], [], []  # Elements to faces conn

        for i in range(0,netypes):
            if nVertCell[i] == 4:
                tet2v = ptrToArray(i, geo.c2v, ncells, nvert)
                tet2f = ptrToArray(i, geoAB.c2f, ncells, nface)
            elif nVertCell[i] == 5:
                pyr2v = ptrToArray(i, geo.c2v, ncells, nvert)
                pyr2f = ptrToArray(i, geoAB.c2f, ncells, nface)
            elif nVertCell[i] == 6:
                pri2v = ptrToArray(i, geo.c2v, ncells, nvert)
                pri2f = ptrToArray(i, geoAB.c2f, ncells, nface)
            elif nVertCell[i] == 8:
                hex2v = ptrToArray(i, geo.c2v, ncells, nvert)
                hex2f = ptrToArray(i, geoAB.c2f, ncells, nface)

        wallNodes = ptrToArray(geo.wallNodes, geo.nwall)
        overNodes = ptrToArray(geo.overNodes, geo.nover)

        iblank = ptrToArray(geo.iblank, nnodes)
        iblank_cell = ptrToArray(geoAB.iblank_cell, ncellsTot)
        iblank_face = ptrToArray(geoAB.iblank_face, nfacesTot)

        f2v = ptrToArray(geoAB.f2v, nftypes)
        f2c = ptrToArray(geoAB.f2c, nfacesTot, 2)

        tri2v, quad2v = [], []  # Faces to nodes conn
        for i in range(0,nftypes):
            if nVertCell[i] == 3:
                tet2v = ptrToArray(i, geoAB.f2v, nfaces, nvertf)
            elif nVertCell[i] == 4:
                quad2v = ptrToArray(i, geoAB.f2v, nfaces, nvertf)

        overFaces = ptrToArray(geoAB.overFaces, geoAB.nOverFaces)
        wallFaces = ptrToArray(geoAB.wallFaces, geoAB.nWallFaces)
        mpiFaces = ptrToArray(geoAB.mpiFaces, geoAB.nMpiFaces)
        procR = ptrToArray(geoAB.procR, geoAB.nMpiFaces)
        mpiFidR = ptrToArray(geoAB.mpiFidR, geoAB.nMpiFaces)

        # Wrap relevant solution data
        # NOTE: for GPU, solution is padded to 128 bytes using nElesPad
        q = netypes*[[]]
        dq = netypes*[[]]
        for i in range(0,netypes):
            q[i] = ptrToArray(i, simdata.u_spts, nspts,nfields,ncells)
            if self.inp.viscous:
                dq[i] = ptrToArray(i, simdata.du_spts, ndims,nspts,nfields,ncells)

        self.gridData = {'gridtype' : 'unstructured',
                         'gridCutType' : [geo.gridType],
                         'c2v'       : [c2v],
                         'tetConn'   : [tet2v],
                         'pyraConn'  : [pyr2v],
                         'prismConn' : [pri2v],
                         'hexaConn'  : [hex2v],
                         'bodyTag' : [geo.btag],
                         'nCellTypes' : [netypes],
                         'nCellsType' : [nCells],
                         'nVertCell' : [nVertCell],
                         'nFaceCell' : [nFaceCell],
                         'wallnode' : [wallNodes],
                         'obcnode' : [overNodes],
                         'grid-coordinates' : [xyz],
                         'q-variables' : [q],
                         'dq-variables' : [dq],
                         'iblanking' : [iblank],
                         'scaling' : [self.scale],
                         'nFaceTypes' : [nftypes],
                         'nfaces' : [nFaces],
                         'face2cell' : [f2c],
                         'tetFaces'  : [tet2f],
                         'pyrFaces'  : [pyr2f],
                         'priFaces'  : [pri2f],
                         'hexFaces'  : [hex2f],
                         'cell2face' : [c2f],
                         'iblank-face' : [iblank_face],
                         'iblank-cell' : [iblank_cell],
                         'overset-faces' : [overFaces],
                         'wall-faces' : [wallFaces],
                         'mpi-faces' : [mpiFaces],
                         'mpi-right-proc' : [procR],
                         'mpi-right-id' : [mpiFidR],
                         'nvert-face' : [geoAB.nvert_face],
                         'triConn'  : [tri2v],
                         'quadConn' : [quad2v],
                         'faceConn' : [f2v]}
                         #'iblkHasNBHole':0,
                         #'istor':'row'}
                         #'fsicoord':self.fsicoord,
                         #'fsiforce':self.fsiforce,
                         #'fsitag':self.fsitag}  

        if self.inp.motion:
            gridVel = [ptrToArray(geoAB.grid_vel, nnodes,ndims)]
            offset = [ptrToArray(geoAB.offset, ndims)]
            Rmat = [ptrToArray(geoAB.Rmat, ndims,ndims)]

            self.gridData.update({'gridVel':gridVel,
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
            self.callbacks.update({'fringeDataToDevice': cbs.fringe_data_to_device,
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
    
    def setCallbacks(self,cbs):
        self.z.set_rigid_body_callbacks(cbs['tg-set-transform'])

    def restart(self,iter):
        os.chdir('zefr')
        self.z.restart_solution()
        os.chdir('..')

    def reportResidual(self,istep):
        os.chdir('zefr')
        self.z.write_residual()
        os.chdir('..')

    def writePlotData(self,istep):
        os.chdir('zefr')
        self.z.write_solution()
        os.chdir('..')

    def writeRestartData(self,istep):
        os.chdir('zefr')
        self.z.write_solution()
        os.chdir('..')

    def computeForces(self,istep):
        os.chdir('zefr')
        self.z.write_forces()
        os.chdir('..')

        # Apply scaling to forces & moments
        bodyForce = ptrToArray(self.simdata.forces, 6)

        F = np.array(bodyForce[0:3]*self.forceDim)
        M = np.array(bodyForce[3:6]*self.momDim)

        return F, M

    # HELIOS can provide grid velocities
    def setGridSpeeds_maneuver(self,MeshMotionData):
        self.vx = -MeshMotionData['aircraftTranslation'][0]/self.ainf
        self.vy = -MeshMotionData['aircraftTranslation'][1]/self.ainf
        self.vz = -MeshMotionData['aircraftTranslation'][2]/self.ainf
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("gust speed from turns :",self.vx,self.vy,self.vz)

    def adaptDT(self):
        return self.z.adapt_dt()

    def deformPart1(self,time,iter):
        self.z.move_grid_next(time+self.inp.dt)

    def deformPart2(self,time,iter):
        self.z.move_grid(time)

    def moveGrid(self,iter,stage):
        self.z.move_grid(iter,stage)

    def updateBlankingGpu(self):
        self.z.update_iblank_gpu()

    def finish(self,step):
        if step % self.inp.write_freq != 0:
            self.writeRestartData(step)
