CXX = g++
CU = nvcc
LFLAGS =-lcudart -L=/cm/shared/apps/cuda65/toolkit/6.5.14/lib64


main:
	$(CXX) -c -std=c++11 -D_GPU scratch.cpp  -I../include
	$(CU) -c -std=c++11 -arch=sm_20 -D_GPU ../src/solver_kernels.cu -I../include
	$(CXX) $(LFLAGS) scratch.o solver_kernels.o  -o main
