# ZEFR: A high-order CFD solver using the Direct Flux Reconstruction Method

**Objective**: The goal of this project is to create a simple, high-performance 
high-order CFD solver using the Direct Flux Reconstruction methodology for two
dimensional, and eventually, three dimensional problems.

## Current Features:
- Can solve the Advection-Diffusion, Euler, and Navier-Stokes equations in 2D
- Supports 2D quadrilateral, triangular, and mixed meshes in GMSH format. Element boundary representations up to 2nd order supported.
- Paraview output using binary XML .vtu files
- Supports fully featured compilation on both CPU (multithreaded with OpenMP) 
and Nvidia GPUs
- P-multigrid for convergence acceleration of steady-state problems

## Installing and Running ZEFR:
Building ZEFR is very straightforward. First, you must create a config file (see 
examples in the configfiles/ directory). To compile, simply type the following in the top directory:

```
make CONFIG=<your config file> 
```
This will build the ZEFR executable in a created bin/ 
directory.

To run the code, simply run the executable with an input file as an argument:

```
./zefr <input file>
```
See the input files in the testcases/ directory for examples of working cases.

### Mesh Utilities

ZEFR includes a dependence on PyFR for mesh conversion and post-processing.
Use the 'meshtools' program under 'external' to convert grids to the PyFR format
and to export solution data to ParaView's .vtu/.pvtu file format for visualization.

Numpy, h5py, and pytools are required to use the meshtools script.

# Code Structure:
The program is structured using three main classes: the **FRSolver** class, the **Elements** class, and the **Faces** class. 


Additionally, there are two main structs used in the 
code: an **InputStruct** structure which contains input file data, and a **GeoStruct** 
structure which contains most general geometry data.

In terms of organization, the **FRSolver** class is at the highest level. The public 
facing methods of this class allow instantiation of a solver, advancing in time, 
and methods to output relevant diagnostics (residuals, forces, Paraview files, and
error output). The private methods of this class orchestrate the entire Flux 
Reconstruction algorithm and interactions between **Faces** and **Elements** objects. 

The FRSolver class contains a single instance of both the 
**Elements** and **Faces** classes:

- The **Elements** class is an abstract class that contains most operations on *element-local* data (for example, computing flux gradients at the solution points). The element-local data is comprised of two major types: data at element solution points (typically denoted with "\_spts" in the code) and data at element flux points (typically denoted with "\_fpts" in the code). These data structures are indexed by local solution point indices by element. 

- The **Faces** class is an abstract class that contains most operations on *global data* at 
flux points (for example, any Riemann solve). The global flux point data is comprised solely of data at flux points 
(sometimes denoted using "\_gfpts" in the code but not always. Anything in the faces class
belongs to this set of data). These data structures are indexed by a global flux point index and slot (0 corresponding to the left element state and 1 corresponding to the right element state). Note that this data is generally simply a copy of the element-local flux point data into a better organized, contiguous data
structure.

The multigrid functionality is encapsulated in the **PMgrid** class and is the only additional class. 

A good starting point to figuring out how the code works is the *compute_residual()* method of the **FRSolver** class. It is the main caller method for a single FR iteration.

## Development Guidelines:
If you would like to develop code for ZEFR, please follow these guidelines:

- Do initial development in branches which will be pulled into master by request.
- Use consistent formatting (2 space tabs, brackets on separate lines, etc.) 
- Try to generate both the CPU and GPU implementation and test for consistency. The code uses custom multidimensional vector objects (**mdvector** and **mdvector_gpu**) which should ease some of the translation of regular C++ code to CUDA kernels. You can simply pass **mdvector_gpu** objects into your kernels by value and you can use them in an identical manner to the **mdvector** objects in the serial code. To create a functional kernel, it should be nearly a straight copy and paste job.
- Most importantly, have fun! 

## Additional Remarks:
This code is still in the development phase and as such, documentation within the source is a bit sparse (though much of the code should be straightforward). I will be adding more documentation, at least of the class methods and variables soon. 

## License:
This code is currently released under the GNU General Public License Version 3.0. See LICENSE for details.
 


