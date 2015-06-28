ZEFR: A high-order CFD solver using the direct Flux Reconstruction method in 2D
======================================================
Objective: The goal of this project is to create a simple, high-performance 
high-order CFD solver using the Flux Reconstruction methodology for two
dimensional, and eventually, three dimensional problems.

Current Features:
------------------
- Supported equations: Advection-Diffusion, Euler, and Navier-Stokes
- Supported meshes:  2D quadrilateral, triangular, and mixed grids in GMSH format. 
Element boundary representations up to 2nd order supported.
- Output: Paraview using .vtk legacy ASCII files
- Supports fully featured compilation on both CPU (multithreaded with OpenMP) 
and Nvidia GPUs
- P-multigrid for convergence acceleration of steady-state problems

Installing and Running ZEFR:
------------
Building ZEFR is very straightforward. First, you must create a config file (see 
examples in the configfiles/ directory). To compile, simply type:
make CONFIG=<your config file> 
in the top directory. This will build the ZEFR executable in a created bin/ 
directory.

To run the code, simply run the executable with an input file as an argument:
./zefr <input file>
See the input files in the testcases/ directory for examples of working cases.


Code Structure:
-------------
The program is structured using three main classes: the FRSolver class, the Elements 
class, and the Faces class. Additionally, there are two main structs used in the 
code: an InputStruct structure which contains input file data, and a GeoStruct 
structure which contains most general geometry data.

In terms of organization, the FRSolver class is at the highest level. The public 
facing methods of this class allow instantiation of a solver, advancing in time, 
and methods to output relevant diagnostics (residuals, forces, Paraview files, and
error output). The private methods of this class orchestrate the entire Flux 
Reconstruction algorithm including interactions between Faces and Elements objects. 

Continuing on this, the FRSolver class contains a single instance of both the 
Elements and Faces classes:
- The Elements class is an abstract class that contains
most operations on element-local data. The element-local data is comprised of two major types: 
data at element solution points (typically denoted with "\_spts" in the 
code) and data at element flux points (typically denoted with "\_fpts" in the code).
These data structures are indexed by local solution point indices by element. 

-The Faces class is an abstract class that contains most operations on global data at 
flux points. The global flux point data is comprised solely of data at flux points 
(sometimes denoted using "\_gfpts" in the code but not always. Anything in the faces class
belongs to this set of data). These data structures are indexed by a global flux
point index and slot (0 corresponding to the left element state and 1 corresponding
to the right element state). Note that this data is generally simply a copy
of the element-local flux point data into a better organized, contiguous data
structure.


 


