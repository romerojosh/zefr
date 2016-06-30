ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif


CXX = g++
AR = ar -rvs
CU = nvcc
FLAGS += -std=c++11
CXXFLAGS = -Ofast -Wno-unknown-pragmas
CUFLAGS = -arch=sm_20 -O3 -use_fast_math --default-stream per-thread

WARN_ON = -Wall -Wextra -Wconversion
WARN_OFF = -Wno-narrowing -Wno-unused-result -Wno-narrowing 

ifeq ($(strip $(WARNINGS)),YES)
	CXXFLAGS += $(WARN_ON)
else
	CXXFLAGS += $(WARN_OFF)
endif

# Setting OpenMP flags
ifeq ($(strip $(OPENMP)),YES)
	CXXFLAGS += -fopenmp
	CUFLAGS += -Xcompiler -fopenmp
	FLAGS += -D_OMP
endif

# Setting BLAS flags
ifeq ($(strip $(BLAS)),STANDARD)
	LIBS = -L$(BLAS_LIB_DIR) -lblas
endif

ifeq ($(strip $(BLAS)),OPENBLAS)
	LIBS = -L$(BLAS_LIB_DIR)/lib -lopenblas
endif

ifeq ($(strip $(BLAS)),$(strip ATLAS))
	LIBS = -L$(BLAS_LIB_DIR) -latlas -lcblas
endif

INCS = -I$(strip $(BLAS_INC_DIR))
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
INCS += -I$(CURDIR)/external/tnt/
INCS += -I$(CURDIR)/external/jama/
INCS += -I$(strip $(AUX_DIR))/

TARGET = zefr
OBJS = bin/elements.o bin/faces.o bin/funcs.o bin/geometry.o bin/hexas.o bin/input.o bin/multigrid.o bin/points.o bin/polynomials.o bin/quads.o bin/solver.o bin/filter.o  bin/zefr.o 

ifeq ($(strip $(ARCH)),GPU)
	OBJS += bin/elements_kernels.o bin/faces_kernels.o bin/solver_kernels.o  bin/filter_kernels.o
endif

INCS += -I$(CURDIR)/include

$(TARGET): $(OBJS)
	@$(CXX) $(INCS) -o bin/$(TARGET) $(OBJS) $(LIBS) $(CXXFLAGS)

# Option to build libzefr.so for use with SWIG / Python wrapping
lib: $(OBJS)
	$(CXX) -shared $(INCS) $(FLAGS) $(CXXFLAGS) -o bin/libzefr.so $(OBJS) $(LIBS)
#	@$(AR) bin/lib$(TARGET).a $(OBJS)

# Build the library & run SWIG/Python scripts
swig: FLAGS += -D_BUILD_LIB
swig: CXXFLAGS += -fPIC
swig: lib
	@cp bin/libzefr.so swig/
	@$(MAKE) -C swig FLAGS='$(FLAGS)' CXXFLAGS='$(CXXFLAGS)' INCS='$(INCS)' LIBS='$(LIBS)'

#	cd swig && make FLAGS='$(FLAGS)' CXXFLAGS='$(CXXFLAGS)' INCS='$(INCS)' LIBS='$(LIBS)'

bin/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p bin
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

ifeq ($(strip $(ARCH)),GPU)
bin/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS) -D_NO_TNT
endif

clean:
	@rm -f bin/$(TARGET) bin/*.o bin/*.a swig/*.so swig/*.pyc swig/zefr.py

