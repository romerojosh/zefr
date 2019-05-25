#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script integrates kinetic energy and enstrophy over a grid for post-
processing the Taylor-Green test case
It may also serve as an example of how to integrate arbitrary functions of the
solution on a grid
"""

import glob
import os
import sys

import numpy as np
from math import sqrt, pi

import mpi4py.rc
mpi4py.rc.initialize = False

# Add the PyFR submodule to our path
sys.path.append(os.path.join(os.path.dirname(__file__), 'PyFR'))

from pyfr.inifile import Inifile
from pyfr.quadrules import get_quadrule
from pyfr.shapes import HexShape, PriShape, TetShape
from pyfr.readers.native import NativeReader
from pyfr.solvers.euler.elements import EulerElements

u0 = .1*sqrt(1.4)
fac = (u0**2)*(2*pi)**3

def tg_int(meshf, solnfs):
    cfg = Inifile(solnfs[0]['config'])
    shapedata = {}

    # Per-element type data
    for ename, ebasis in [('hex', HexShape), ('tet', TetShape)]:
        sptname = 'spt_' + ename + '_p0'

        # Check the element type is present in the mesh
        if sptname not in meshf:
            continue

        # Construct the elements instance
        eles = EulerElements(ebasis, meshf[sptname], cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4

        # Weights
        rname = cfg.get('solver-elements-' + ename, 'soln-pts')
        wts = get_quadrule(ename, rname, eles.nupts).wts

        shapedata[ename] = (smat, rcpdjac, gradop, wts)

    # Iterate over the solutions
    for solnf in solnfs:
        stats = Inifile(solnf['stats'])

        # Simulation time
        t = stats.getfloat('solver-time-integrator', 'tcurr')

        # Kinetic energy and enstrophy
        keng = 0
        enst = 0

        # Iterate over the element types
        for ename, (smat, rcpdjac, gradop, wts) in shapedata.items():
            # Extract the solution
            soln = solnf['soln_' + ename + '_p0']

            # Dimensions
            nupts, nvars = soln.shape[:2]

            # Evaluate the transformed gradient of the solution
            gradsoln = np.dot(gradop, soln.reshape(nupts, -1))
            gradsoln = gradsoln.reshape(3, nupts, nvars, -1)

            # Untransform
            gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln)

            # Jacobians
            jacs = 1/rcpdjac

            # Variables
            rho, rhovx, rhovy, rhovz, E = np.rollaxis(soln, 1)
            vx, vy, vz = rhovx / rho, rhovy / rho, rhovz / rho

            drhox, drhoy, drhoz = gradsoln[0]
            drhovxx, drhovxy, drhovxz = gradsoln[1]
            drhovyx, drhovyy, drhovyz = gradsoln[2]
            drhovzx, drhovzy, drhovzz = gradsoln[3]

            dvxx = (drhovxx - vx*drhox) / rho;
            dvxy = (drhovxy - vx*drhoy) / rho;
            dvxz = (drhovxz - vx*drhoz) / rho;
            dvyx = (drhovyx - vy*drhox) / rho;
            dvyy = (drhovyy - vy*drhoy) / rho;
            dvyz = (drhovyz - vy*drhoz) / rho;
            dvzx = (drhovzx - vz*drhox) / rho;
            dvzy = (drhovzy - vz*drhoy) / rho;
            dvzz = (drhovzz - vz*drhoz) / rho;

            # Kinetic energy
            elekeng = 0.5*(rhovx*rhovx + rhovy*rhovy + rhovz*rhovz)/rho

            # Enstrophy
            eleenst = 0.5*rho*((dvzy - dvyz)**2 +
                               (dvxz - dvzx)**2 +
                               (dvyx - dvxy)**2)

            # Do the quadrature
            keng += np.sum(wts[:, None]*jacs*elekeng)
            enst += np.sum(wts[:, None]*jacs*eleenst)

        yield (t*u0, keng/fac, enst/fac)


if __name__ == '__main__':
    # Open the mesh and solutions
    meshf = NativeReader(sys.argv[1])
    solnfs = [NativeReader(arg) for arg in glob.glob(sys.argv[2])]

    for l in tg_int(meshf, solnfs):
        print(','.join(str(m) for m in l))
        sys.stdout.flush()
