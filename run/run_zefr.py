import sys
import os
import yaml

TIOGA_DIR = '/home/jcrabill/tioga/bin/'
ZEFR_DIR = '/home/jcrabill/zefr/bin/'
sys.path.append(TIOGA_DIR)
sys.path.append(ZEFR_DIR)

from mpi4py import MPI
import numpy as np

from zefrInterface import zefrSolver
import tioga as tg

Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
nproc = Comm.get_size()

# ------------------------------------------------------------
# Parse our input
# ------------------------------------------------------------

# Split our run into the various grids
if len(sys.argv) == 2:
    inputFile = sys.argv[1]
elif len(sys.argv) > 2:
    nGrids = len(sys.argv) - 2
    nRanksGrid = []
    rankSum = 0
    for i in range(0,nGrids):
        rankSum += int(sys.argv[i+2])
        if rankSum > rank:
            gridID = i;
            break;

if nGrids > 1:
    gridComm = Comm.Split(gridID,rank)
else:
    gridID = 0
    gridComm = Comm

gridRank = gridComm.Get_rank()
gridSize = gridComm.Get_size()

# Read in overall simulation parameters
parameters = {}
with open(inputFile) as f:
    for line in f:
        line = line.strip().split()
        if len(line) == 2 and not str(line[0]).startswith('#'):
            parameters[line[0]] = line[1]

expected_conditions = ['meshRefLength','reyNumber','reyRefLength','refMach',
    'reyRefLength','dt','Mach','from_restart']

conditions = {}
for cond in expected_conditions:
    try:
        conditions[cond] = parameters[cond]
    except:
        print('Condition',cond,'not given in',inputFile)

# ------------------------------------------------------------
# Begin setting up TIOGA and the ZEFR solvers
# ------------------------------------------------------------

try:
  zefrInput = parameters['zefrInput']
except:
  print('ZEFR input file ("zefrInput") not given in',inputFile)

ZEFR = zefrSolver(zefrInput,gridID,nGrids)

# TODO: refactor ZEFR so that TIOGA can be called only here :(

nSteps = parameters['nsteps']
nStages = parameters['nstages']

repFreq = parameters['report-freq']
plotFreq = parameters['plot-freq']
forceFreq = parameters['force-freq']

# TODO: pass callbacks into TIOGA; do preprocessing

ZEFR.initData()

# ------------------------------------------------------------
# Run the simulation
# ------------------------------------------------------------
for i in range(0,nSteps):
    # Do unblanking here 
    # (move to t+dt, hole cut, move to t, hole cut, union on iblank)
    #ZEFR.moveGrid(i,nStages-1)
    ZEFR.deformPart1(i,nStages-1)
    tg.unblankPart1()
    ZEFR.deformPart2(i,0)
    tg.unblankPart2() # Set final iblank & do point connectivity

    for j in range(0,nStages):
        # Move grids
        if j != 0:
            ZEFR.moveGrid(i,j)
            tg.doPointConnectivity()

        # Interpolate solution
        tg.interpolate_u()

        # Calculate first part of residual, up to corrected gradient
        ZEFR.runSubStep(i,j)
  
        # Interpolated gradient
        tg.interpolate_du()

        # Finish residual calculation and RK stage advancement
        # (Should include rigid_body_update() if doing 6DOF from ZEFR)
        ZEFR.runSubStep(i,j)

    if i % repFreq == 0:
        ZEFR.reportResidual()
    if i % plotFreq == 0:
        ZEFR.writePlotData()
    if i % forceFreq == 0:
        forces = ZEFR.getForcesAndMoments()
        if rank == 0:
            print('Iter {0}: Forces {1}'.format(i,forces))
