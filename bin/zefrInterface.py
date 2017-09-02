#! /usr/bin/env python
#
# File: turnssolution.py
# Authors: Jayanarayanan Sitaraman
# Last Modified: 1/31/06


__version__ = "1.03"

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

import numpy

# Try to import the MPI.COMM_WORLD.module
try:
    from mpi4py import MPI
    _parallel = True

except ImportError:
    _parallel = False


try:
    import zefr
except ImportError:
    print "Import Error: ZEFR solver module"

class zefrSolver:

    def __init__(self,startfile,gridID):
        self.nGrids = 2 # TODO
        self.ID = gridID
        self.inpfile=startfile

        Comm = MPI.COMM_WORLD
        rank = Comm.Get_rank()
        nproc = Comm.Get_size()

        self.gridComm = Comm.Split(gridID,rank)
        self.gridRank = gridComm.Get_rank()
        self.gridSize = gridComm.Get_size()

        zefr.initialize(self.gridComm,startfile,self.nGrids,gridID)

        # scaling for solver variable [i.e. non-dimensionalization]
        self.scale=numpy.array([1.,1.,1.,1.,1.],'d')

        self.z = zefr.get_zefr_object()
        self.inp = self.z.get_input()
        self.name='zefr'

    def runStep(self,nstep,ncyc):
        os.chdir('zefr') # All ZEFR data stored in a sub-folder
        # One Runge-Kutta time step here
        #self.z.do_step()

        for i in range(0,ncyc):
            runSubSteps(ntime,i,ncyc)
            os.chdir('..')

    def runSubSteps(self,ntime,ncyc):
        os.chdir('zefr')
        # Note:
        # ntime = iter [global time step #]
        # ncyc = stage [RK stage #]
        # Run one RK stage here
        self.z.do_rk_stage(ntime,ncyc)

        os.chdir('..')
        #return self.rsp,self.loads

    def sifInitialize(self,properties,conditions):
        os.chdir('zefr') 
        rinf=properties['rinf']
        ainf=properties['ainf']
        self.ainf=ainf
        # All conditions come in as SI units - take care of non-dim here
        if conditions['timeacc']=='yes':
            dt=ainf*conditions['dt']/conditions['meshRefLength']
        else:
            dt=1. 
        alpha=conditions['alpha']
        beta=conditions['beta']
        mach_fs=conditions['Mach']
        self.restart='no'
        if conditions['from_restart']=='yes':
            self.restart='yes'
            self.restartStep=conditions['restartstep']

        gridScale=conditions['meshRefLength']
        Re=conditions['reyNumber']/conditions['reyRefLength']*gridScale/conditions['refMach']*(mach_fs)
        self.forceDim=0.5*rinf*(refMach*ainf*gridScale)**2
        self.momDim=self.forceDim*gridScale

        turns.sifinit(dt,Re,mach_fs,alpha,beta)
            
        self.scale=numpy.array([1./rinf,
                                1./(rinf*ainf),
                                1./(rinf*ainf),
                                1./(rinf*ainf),
                                1./(rinf*ainf*ainf),1.],'d')

        os.chdir('..')

    def getForcesAndMoments(self):
        bodyForce=turns.params_global.bodyForce.copy()
        bodyForce[:3]=bodyForce[:3]*self.forceDim
        bodyForce[3:]=bodyForce[3:]*self.momDim
        return bodyForce
                        
    # sifInit called first to setup solver input
    def initData(self):
        os.chdir('turns')
        turns.initdata()

        [xNB,iNB,qNB,index]=self.getMeshPoints()
                
        # Get geo data from ZEFR and pass into TIOGA
        geo = zefr.get_basic_geo_data()   # Basic geometry/connectivity data
        geoAB = zefr.get_extra_geo_data() # Geo data for AB method
        cbs = zefr.get_callback_funcs()   # Callback functions for high-order/AB method

        # TODO: implement converter functions [cptr to 1D numpy array]
        # [For 2D/3D arrays, can call reshape() after the fact]
        size = geo.nnodes*geo.ndims
        #dims = [geo.nnodes, geo.ndims]
        xyz = zefr.convert_to_npdarray(geo.xyz, size) #, dims) 
        size = geo.ncells*geo.nvert
        c2v = zefr.convert_to_npiarray(geo.c2v, size) 

        # NOTE: ALL of these arrays should be turned into numpy arrays
        self.gridData={'gridtype':'unstructured',
                       'tetConn':'None',
                       'pyraConn':'None',
                       'prismConn':'None',
                       'hexaConn':geo.c2v,
                       'bodyTag':self.btag,
                       'wallnode':geo.wallNodes,
                       'obcnode':geo.overNodes,
                       'grid-coordinates':geo.xyz,
                       'q-variables':self.q,
                       'iblanking':geo.iblank,
                       'scaling':self.scale,
                       'face2cell',geoAB.f2c,
                       'cell2face',geoAB.c2f,
                       'iblank-face',geoAB.iblank_face,
                       'iblank-cell',geoAB.iblank_cell,
                       'ofaces',geoAB.overFaces,
                       'mfaces',geoAB.mpiFaces,
                       'right-proc',geoAB.procR,
                       'right-id',geoAB.mpiFidR,
                       'nvert-face',geoAB.nvert_face,
                       'faceConn',geoAB.f2v}
                       #'iblkHasNBHole':0,
                       #'istor':'row'}
                       #'fsicoord':self.fsicoord,
                       #'fsiforce':self.fsiforce,
                       #'fsitag':self.fsitag}  

        if self.inp.motion:
            self.gridData.append({'gridVel',geoAB.grid_vel,
                'rigidOffset',geoAB.offset,
                'rigidRotMat',geoAB.Rmat})

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
            self.callbacks.append({'donorDataDevice',cbs.donor_data_from_device,
                'fringeDataToDevice',cbs.fringe_data_to_device,
                'unblankToDevice',cbs.unblank_data_to_device,
                'faceNodesGPU',cbs.get_face_nodes_gpu,
                'cellNodesGPU',cbs.get_cell_nodes_gpu,
                'q_spts_d',cbs.get_q_spts_d,
                'dq_spts_d',cbs.get_dq_spts_d})

            GpuGeo geoGpu = zefr::get_gpu_geo_data();
            self.gridData.append({'nodesGPU',geoGpu.coord_nodes,
                'eleCoordsGPU',geoGpu.coord_eles,
                'iblankCellGPU',geoGpu.iblank_cell,
                'iblankFaceGPU',geoGpu.iblank_cell})

        os.chdir('..')
    
    def writePlotData(self,istep):
        self.writeData(istep)

    def writeRestartData(self,istep):
        os.chdir('turns')
        os.chdir('..')

    # HELIOS can provide grid velocities
    def setGridSpeeds_maneuver(self,MeshMotionData):
        self.vx=-MeshMotionData['aircraftTranslation'][0]/self.ainf
        self.vy=-MeshMotionData['aircraftTranslation'][1]/self.ainf
        self.vz=-MeshMotionData['aircraftTranslation'][2]/self.ainf
        if MPI.COMM_WORLD.Get_rank()==0:
            print "gust speed from turns :",self.vx,self.vy,self.vz

    def deformPart1(self):
        if self.nopart1==0:
            turns.warpmeshpart1()
        else:
            self.nopart1=0
            if self.use_unstruct=='true':
                turns.setiblankstoone(self.iblank[0])
                    
    def deformPart2(self):
        if self.use_unstruct=='true':
            turns.setcoorddata(self.x[0])
            turns.warpmeshpart2()

    def finish(self,step):
        istep=0
        self.writeRestartData(step)
        # Jay add
            
    # FOR REFERENCE
    def initUnstruct(self):
        #
        # create unstructred grid data structure
        #
        turns.convertunstruct()
        self.tetConn=[]
        self.pyraConn=[]
        self.prismConn=[]
        self.x=[]
        self.q=[]
        self.iblank=[]
        self.hexaConn=[]
        self.wbcnode=[]
        self.obcnode=[]
        self.bodytags=[]
        self.fsicoord=[]
        self.fsitag=[]
        self.fsiforce=[]
        self.tetConn.append(numpy.array([[],[],[],[]],'i'))
        self.pyraConn.append(numpy.array([[],[],[],[],[]],'i'))
        self.prismConn.append(numpy.array([[],[],[],[],[],[]],'i'))

        i=1
        nnodes,ncells,nwbc,nobc,gridtype = turns.getdimturns(i)

        if (gridtype==0):
            if (nwbc > 0 ) :
                x,ndc8,obc,wbc,bodytags = turns.getpart(i,nwbc,nobc,nnodes,ncells)
                self.wbcnode.append(wbc)
                self.obcnode.append(obc)
            else:
                x,ndc8,obc,bodytags = turns.getpart1(i,nobc,nnodes,ncells)
                self.wbcnode.append(numpy.array([],'i'))                
                self.obcnode.append(obc)                
                        
        q,iblank=turns.getiblanksq(i,nnodes)
        turns.getqdata(q)
        self.q.append(q)
        self.iblank.append(iblank)
        self.x.append(x)
        self.hexaConn.append(ndc8)
        self.bodytags.append(bodytags)
        self.fsicoord.append(turns.fsimod.xxout)
        self.fsiforce.append(turns.fsimod.fxout)
        self.fsitag.append(turns.fsimod.ftag)
        nfsi=turns.fsimod.xxout.shape[0]/3
        fsiout=numpy.reshape(turns.fsimod.xxout,(nfsi,3))
        #print "myrank, bodytag=",MPI.COMM_WORLD.Get_rank(),bodytags[0]
        #turnsio.write_arrayT('fsi_coord'+str(MPI.COMM_WORLD.Get_rank()),fsiout)
                                     

        self.unstData={'gridtype':'unstructured',
                       'tetConn':self.tetConn,
                       'pyraConn':self.pyraConn,
                       'prismConn':self.prismConn,
                       'hexaConn':self.hexaConn,
                       'bodyTag':self.bodytags,
                       'wallnode':self.wbcnode,
                       'obcnode':self.obcnode,
                       'grid-coordinates':self.x,
                       'q-variables':self.q,
                       'iblanking':self.iblank,
                       'scaling':self.scale,
                       'iblkHasNBHole':0,
                       'istor':'row',
                       'fsicoord':self.fsicoord,
                       'fsiforce':self.fsiforce,
                       'fsitag':self.fsitag}
