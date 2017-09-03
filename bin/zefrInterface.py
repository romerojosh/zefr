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
        self.gridRank = gridComm.Get_rank()
        self.gridSize = gridComm.Get_size()

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

        os.chdir('..')

    # Run one full time step
    def runStep(self,iter,nstages):
        os.chdir('zefr')

        #self.z.do_step(nstages)
        for i in range(0,nstages):
            runSubSteps(iter,i,nstages)

        os.chdir('..')

    # Run one RK stage
    def runSubSteps(self,iter,stage):
        os.chdir('zefr')

        self.z.do_rk_stage(iter,stage)

        os.chdir('..')

    def sifInitialize(self,properties,conditions):
        os.chdir('zefr') 

        rinf = properties['rinf']
        ainf = properties['ainf']

        gridScale = conditions['meshRefLength']
        Reref = conditions['reyNumber']
        Lref = conditions['reyRefLength']
        Mref = conditions['refMach']

        self.inp.rho_fs = rinf
        self.inp.L_fs = conditions['reyRefLength']

        # All conditions come in as SI units - take care of non-dim here
        dt = ainf * conditions['dt'] / gridScale
        mach_fs = conditions['Mach']
        self.restart = 'no'
        if conditions['from_restart'] == 'yes':
            self.inp.restart = 'true'
            self.inp.restartIter = conditions['restartstep']

        Re = (Reref/Lref) * gridScale / Mref * (mach_fs)
        self.forceDim = 0.5 * rinf * (refMach*ainf*gridScale)**2
        self.momDim = self.forceDim * gridScale

        # Have ZEFR do any additional setup on input parameters
        zefr.sifinit(dt,Re,mach_fs)
            
        self.scale = np.array([1./rinf,
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf),
                               1./(rinf*ainf*ainf),1.],'d')

        os.chdir('..')

    def getForcesAndMoments(self):
        self.z.get_forces()
        bodyForce = np.array(dptrToArray(self.simdata.forces))
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
        nvertf = getAB.nvert_face
        nfaces = geoAB.nFaces_type

        # Wrap all geometry data
        xyz = [zefr.dptrToArray(geo.xyz, nnodes*ndims)]
        c2v = [zefr.iptrToArray(geo.c2v, ncells*nvert)]
        wallNodes = [zefr.iptrToArray(geo.wallNodes, geo.nwall)]
        overNodes = [zefr.iptrToArray(geo.overNodes, geo.nover)]

        iblank = [zefr.iptrToArray(geo.iblank, nnodes)]
        iblank_cell = [zefr.iptrToArray(geoAB.iblank_cell, ncells)]
        iblank_face = [zefr.iptrToArray(geoAB.iblank_face, nfaces)]
        f2v = [zefr.iptrToArray(geoAB.f2v, nfaces*nvertf)]
        c2f = [zefr.iptrToArray(geoAB.c2f, ncells*(2**ndims))]
        f2c = [zefr.iptrToArray(geoAB.f2c, nfaces*2)]

        overFaces = [zefr.iptrToArray(geoAB.overFaces, geoAB.nOverFaces)]
        wallFaces = [zefr.iptrToArray(geoAB.wallFaces, geoAB.nWallFaces)]
        mpiFaces = [zefr.iptrToArray(geoAB.mpiFaces, geoAB.nMpiFaces)]
        procR = [zefr.iptrToArray(geoAB.procR, geo.nMpiFaces)]
        mpiFidR = [zefr.iptrToArray(geoAB.mpiFidR, geo.nMpiFaces)]

        # Wrap relevant solution data
        q = [zefr.dptrToArray(simdata.u_spts, ncells*nspts*nfields)]

        dq = []
        if self.inp.viscous:
          dq.append(zefr.dptrToArray(simdata.dq_spts, ncells*nspts*ndims*nfields))

        self.gridData = {'gridtype' : 'unstructured',
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
                         'ofaces' : overFaces,
                         'wfaces' : wallFaces,
                         'mfaces' : mpiFaces,
                         'right-proc' : procR,
                         'right-id' : mpiFidR,
                         'nvert-face' : [geoAB.nvert_face],
                         'faceConn' : f2v}
                         #'iblkHasNBHole':0,
                         #'istor':'row'}
                         #'fsicoord':self.fsicoord,
                         #'fsiforce':self.fsiforce,
                         #'fsitag':self.fsitag}  

        if self.inp.motion:
            gridVel = [zefr.dptrToArray(geoAB.grid_vel, nnodes*ndims)]
            offset = [zefr.dptrToArray(geoAB.offset, ndims)]
            Rmat = [zefr.dptrToArray(geoAB.Rmat, ndims*ndims)]

            self.gridData.update({'gridVel',grid_vel,
                'rigidOffset',offset,
                'rigidRotMat',Rmat})

        self.oldCallbacks = {'nodesPerCell',cbs.get_nodes_per_cell,
            'receptorNodes',cbs.get_receptor_nodes,
            'donorInclusionTest',cbs.donor_inclusion_test,
            'donorFrac',cbs.donor_frac,
            'convertToModal',cbs.convert_to_modal}

        self.callbacks = {'nodesPerFace',cbs.get_nodse_per_face,
            'faceNodes',cbs.get_face_nodes,
            'get_q_spt',cbs.get_q_spt,
            'get_q_fpt',cbs.get_q_fpt,
            'get_grad_spt',cbs.get_grad_spt,
            'get_grad_fpt',cbs.get_grad_fpt,
            'q_spts',cbs.get_q_spts,
            'dq_spts',cbs.get_dq_spts}

        if self.useGPU:
            self.callbacks.update({'donorDataDevice',cbs.donor_data_from_device,
                'fringeDataToDevice',cbs.fringe_data_to_device,
                'unblankToDevice',cbs.unblank_data_to_device,
                'faceNodesGPU',cbs.get_face_nodes_gpu,
                'cellNodesGPU',cbs.get_cell_nodes_gpu,
                'q_spts_d',cbs.get_q_spts_d,
                'dq_spts_d',cbs.get_dq_spts_d})

            geoGpu = zefr.get_gpu_geo_data();
            self.gridData.update({'nodesGPU',geoGpu.coord_nodes,
                'eleCoordsGPU',geoGpu.coord_eles,
                'iblankCellGPU',geoGpu.iblank_cell,
                'iblankFaceGPU',geoGpu.iblank_face})

        os.chdir('..')
    
    def writePlotData(self,istep):
        os.chdir('zefr')
        self.z.write_solution()
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

    def deformPart1(self,iter,stage):
        self.z.move_grid(iter,stage)
        pass

    def deformPart2(self,iter,stage):
        self.z.move_grid(iter,stage)
        pass

    def finish(self,step):
        istep = 0
        self.writeRestartData(step)
