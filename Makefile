ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif

CXX = g++
CC = gcc
ifeq ($(strip $(INTEL)),YES)
	CXX = icpc
	CC = icc
endif
AR = ar -rvs
ifeq ($(CU),)
  CU = nvcc
endif
ifeq ($(strip $(SWIG_BIN)),)
	SWIG = swig -c++ -python
else
	SWIG = $(SWIG_BIN)/swig -c++ -python
endif

CXXFLAGS = -std=c++11 -Wno-unknown-pragmas #-fstack-protector-all
CCFLAGS = -std=c99 -w
CUFLAGS = -std=c++11 --default-stream per-thread $(EXTRA_CUFLAGS)
WARN_ON = -Wall -Wextra -Wconversion
WARN_OFF = -Wno-narrowing -Wno-unused-result -Wno-narrowing -Wno-literal-suffix

RELEASE_FLAGS = -Ofast
FLAGS = $(AUX_FLAGS)

ifeq ($(strip $(WARNINGS)),YES)
	CXXFLAGS += $(WARN_ON)
else
	CXXFLAGS += $(WARN_OFF) 
	CUFLAGS += -Xcompiler=-Wno-narrowing,-Wno-unused-result,-Wno-narrowing,-Wno-literal-suffix -Xcudafe "--diag_suppress=subscript_out_of_range"
endif

ifeq ($(strip $(DEBUG_LEVEL)),1)
	CCFLAGS += -g -O3 -D_NVTX
	CXXFLAGS += -g -O3 -D_NVTX
	CUFLAGS += -g -O3 -D_NVTX
else 
ifeq ($(strip $(DEBUG_LEVEL)),2)
	CCFLAGS += -g -O0 #-D_NVTX
	CXXFLAGS += -g -O0 #-D_NVTX
	CUFLAGS += -g -G -O0 #-D_NVTX
else
	CCFLAGS += $(RELEASE_FLAGS)
	CXXFLAGS += $(RELEASE_FLAGS)
	CUFLAGS += -O3 -use_fast_math
endif
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
	LIBS = -L$(BLAS_LIB_DIR) -lopenblas
endif

ifeq ($(strip $(BLAS)),$(strip ATLAS))
	LIBS = -L$(BLAS_LIB_DIR) -latlas -lcblas
endif

INCS = -I$(strip $(BLAS_INC_DIR))

# Setting MPI/METIS flags
ifeq ($(strip $(MPI)),YES)
ifeq ($(strip $(INTEL)),YES)
	CXX = mpiicpc -cxx=icpc
else
	CXX = mpicxx
endif
	FLAGS += -D_MPI
	INCS += -I$(strip $(MPI_INC_DIR))/ -I$(strip $(METIS_INC_DIR))/
ifneq ($(MPI_LIB_DIR),)
	LIBS += -L$(strip $(MPI_LIB_DIR))/ -lmpi -Wl,-rpath=$(MPI_LIB_DIR)
endif
	LIBS += -L$(strip $(METIS_LIB_DIR))/ -lmetis -Wl,-rpath=$(strip $(METIS_LIB_DIR))
endif

# Setting HDF5 flags
ifneq ($(strip $(HDF5)),NO)
	INCS += -I$(strip $(HDF5_INC_DIR))/
	LIBS += -L$(strip $(HDF5_LIB_DIR))/ -lhdf5 -lhdf5_cpp -Wl,-rpath=$(strip $(HDF5_LIB_DIR))/
endif

# Setting Architecture flags
ifeq ($(strip $(ARCH)),CPU)
	FLAGS += -D_CPU
endif

ifeq ($(strip $(ARCH)),GPU)
	FLAGS += -D_GPU
	LIBS += -L$(strip $(CUDA_DIR))/lib64 -lcudart -lcublas -lnvToolsExt -Wl,-rpath=$(strip $(CUDA_DIR))/lib64
	INCS += -I$(strip $(CUDA_DIR))/include

  # CUDA-aware MPI capability
ifeq ($(strip $(CUDA_AWARE)),YES)
	CXXFLAGS += -D_CUDA_AWARE
endif

ifeq ($(CU),)
  CUFLAGS += -arch=sm_20
else
	
ifeq ($(strip $(SM)),FERMI)
  CUFLAGS += -arch=sm_20 -D_FERMI
endif

ifeq ($(strip $(SM)),KEPLER)
  CUFLAGS += -arch=sm_35 -D_KEPLER
endif

ifeq ($(strip $(SM)),GEFORCE)
  CUFLAGS += -arch=sm_30 -D_GEFORCE
endif

endif
endif

# Including external template libraries
FLAGS += -DEIGEN_MPL2_ONLY
INCS += -I$(CURDIR)/external/
INCS += -I$(strip $(AUX_DIR))/

SRCDIR = $(CURDIR)/src
BINDIR = $(CURDIR)/bin
SWIGDIR = $(CURDIR)/swig_bin

TARGET = zefr
OBJS = $(BINDIR)/elements.o $(BINDIR)/faces.o $(BINDIR)/funcs.o $(BINDIR)/geometry.o $(BINDIR)/hexas.o $(BINDIR)/input.o $(BINDIR)/multigrid.o $(BINDIR)/points.o $(BINDIR)/polynomials.o $(BINDIR)/quads.o $(BINDIR)/solver.o $(BINDIR)/tets.o $(BINDIR)/tris.o $(BINDIR)/filter.o  $(BINDIR)/zefr.o 
#$(BINDIR)/gimmik_cpu.o

ifeq ($(strip $(ARCH)),GPU)
	OBJS += $(BINDIR)/elements_kernels.o $(BINDIR)/faces_kernels.o $(BINDIR)/solver_kernels.o  $(BINDIR)/filter_kernels.o $(BINDIR)/gimmik_gpu.o
endif

SOBJS = $(OBJS) $(BINDIR)/zefr_interface.o
SWIG_OBJ = $(BINDIR)/zefr_swig.o
SWIG_TARGET = $(BINDIR)/_zefr.so
SWIG_INCS = -I$(strip $(PYTHON_INC_DIR))/ -I$(strip $(MPI4PY_INC_DIR))/
SWIG_LIBS = 
WRAP_TARGET = $(BINDIR)/zefrWrap

INCS += -I$(CURDIR)/include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) $(OBJS) $(LIBS) $(CXXFLAGS) -o $(BINDIR)/$(TARGET)

# Build Zefr as a Python extension module (shared library) using SWIG
.PHONY: swig
swig: FLAGS += -D_BUILD_LIB
swig: CXXFLAGS += -I$(TIOGA_INC_DIR)/ -fPIC $(SWIG_INCS)
swig: INCS += -I$(TIOGA_INC_DIR)/
swig: CUFLAGS += -Xcompiler -fPIC
swig: $(SOBJS) $(SWIG_OBJ)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) $(SWIG_INCS) -shared -o $(SWIG_TARGET) $(SOBJS) $(SWIG_OBJ) $(LIBS) $(SWIG_LIBS)
	
# Build Zefr as a static library
.PHONY: static
static: FLAGS += -D_BUILD_LIB
static: CXXFLAGS += -fPIC
static: CUFLAGS += -Xcompiler -fPIC
static: $(SOBJS)
	$(AR) $(BINDIR)/libzefr.a $(SOBJS)

# Build Zefr as a shared library
.PHONY: shared
shared: FLAGS += -D_BUILD_LIB
shared: CXXFLAGS += -fPIC
shared: CUFLAGS += -Xcompiler -fPIC
shared: $(SOBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) -shared -o $(BINDIR)/libzefr.so $(SOBJS) $(LIBS)

# Compile the zefrWrap wrapper program using dynamic linking
.PHONY: wrap
wrap: INCS += -I$(TIOGA_INC_DIR)/
wrap: shared
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/zefrWrap.cpp -o $(WRAP_TARGET) -L$(TIOGA_LIB_DIR)/ -L$(BINDIR)/ -lzefr -ltioga -Wl,-rpath=$(BINDIR)/ -Wl,-rpath=$(TIOGA_LIB_DIR)/ $(LIBS)

# Compile the zefrWrap wrapper program using static linking
.PHONY: wrap_static
wrap_static: INCS += -I$(TIOGA_INC_DIR)/
wrap_static: static
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/zefrWrap.cpp $(BINDIR)/libzefr.a $(TIOGA_LIB_DIR)/libtioga.a -o $(WRAP_TARGET) $(LIBS)

# Implicit Rules
$(BINDIR)/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p bin
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

$(BINDIR)/%.o: src/%.c  include/*.hpp include/*.h
	@mkdir -p bin
	$(CC) $(INCS) -c -o $@ $< $(FLAGS) $(CCFLAGS)

ifeq ($(strip $(ARCH)),GPU)
$(BINDIR)/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS)
endif

$(BINDIR)/%_swig.cpp: include/%.i include/*.hpp
	$(SWIG) $(FLAGS) $(INCS) $(SWIG_INCS) -o $@ $<

$(BINDIR)/%_swig.o: $(BINDIR)/%_swig.cpp
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

clean:
	rm -f $(BINDIR)/$(TARGET) $(BINDIR)/*.o $(BINDIR)/*.a $(BINDIR)/*.so $(BINDIR)/zefr.pyc $(BINDIR)/zefr.py

