ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif


CXX = g++
CU = nvcc
FLAGS += -std=c++11
CXXFLAGS = -Ofast -Wall -Wextra -Wconversion -Wno-unknown-pragmas
CUFLAGS = -arch=sm_20 -O3 -use_fast_math --default-stream per-thread

# Setting OpenMP flags
ifeq ($(strip $(OPENMP)),YES)
	CXXFLAGS += -fopenmp
	CUFLAGS += -Xcompiler -fopenmp
	FLAGS += -D_OMP
endif

# Setting BLAS flags
ifeq ($(strip $(BLAS)),STANDARD)
	LIBS = -L$(BLAS_DIR)/lib -lblas
endif

ifeq ($(strip $(BLAS)),OPENBLAS)
	LIBS = -L$(BLAS_DIR)/lib -lopenblas
endif

ifeq ($(strip $(BLAS)),$(strip ATLAS))
	LIBS = -L$(BLAS_DIR)/lib -latlas
endif

INCS = -I$(strip $(BLAS_DIR))/include 
INCS += -I$(strip $(BLAS_DIR))/include/openblas

# Setting MPI/METIS flags
ifeq ($(strip $(MPI)),YES)
	CXX = mpicxx
	FLAGS += -D_MPI
	INCS += -I$(strip $(METIS_DIR))/include -I$(strip $(MPI_INC_DIR))
	LIBS += -L$(strip $(METIS_DIR))/lib -lmetis 
endif

# Setting Architecture flags
ifeq ($(strip $(ARCH)),CPU)
	FLAGS += -D_CPU
endif

ifeq ($(strip $(ARCH)),GPU)
	FLAGS += -D_GPU
	LIBS += -L$(strip $(CUDA_DIR))/lib64 -lcudart -lcublas
	INCS += -I$(strip $(CUDA_DIR))/include
endif

# Including external template libraries
INCS += -I external/tnt/
INCS += -I external/jama/
INCS += -I$(strip $(AUX_DIR))/

TARGET = zefr
OBJS = bin/elements.o bin/faces.o bin/funcs.o bin/geometry.o bin/hexas.o bin/input.o bin/multigrid.o bin/points.o bin/polynomials.o bin/quads.o bin/solver.o bin/filter.o  bin/zefr.o 

ifeq ($(strip $(ARCH)),GPU)
	OBJS += bin/elements_kernels.o bin/faces_kernels.o bin/solver_kernels.o  bin/filter_kernels.o
endif

INCS += -I include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) -o bin/$(TARGET) $(OBJS) $(LIBS) $(CXXFLAGS)

bin/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p bin
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

ifeq ($(strip $(ARCH)),GPU)
bin/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS) -D_NO_TNT
endif

clean:
	rm bin/$(TARGET) bin/*.o

