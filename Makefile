ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif


CXX = g++
CU = nvcc
FLAGS += -std=c++11
CXXFLAGS = -Ofast -Wall -Wextra -Wconversion -fopenmp 
CUFLAGS = -arch=sm_20 -O3 -use_fast_math -Xcompiler -fopenmp

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

ifeq ($(strip $(ARCH)),CPU)
	FLAGS += -D_CPU
endif

ifeq ($(strip $(ARCH)),GPU)
	FLAGS += -D_GPU
	LIBS += -L$(strip $(CUDA_DIR))/lib64 -lcudart -lcublas
endif

TARGET = zefr
OBJS = bin/elements.o bin/faces.o bin/funcs.o bin/geometry.o bin/input.o bin/multigrid.o bin/points.o bin/polynomials.o bin/quads.o bin/shape.o bin/solver.o bin/zefr.o 

ifeq ($(strip $(ARCH)),GPU)
	OBJS += bin/elements_kernels.o bin/faces_kernels.o bin/solver_kernels.o 
endif

INCS += -I include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) -o bin/$(TARGET) $(OBJS) $(LIBS) $(CXXFLAGS)

bin/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p bin
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

ifeq ($(strip $(ARCH)),GPU)
bin/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS)
endif

clean:
	rm bin/$(TARGET) bin/*.o
