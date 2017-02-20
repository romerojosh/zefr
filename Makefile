ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif

CXX = g++
AR = ar -rvs
ifeq ($(CU),)
  CU = nvcc
endif

CXXFLAGS = -std=c++11 -Wno-unknown-pragmas #-fstack-protector-all
CUFLAGS = -std=c++11 --default-stream per-thread
WARN_ON = -Wall -Wextra -Wconversion
WARN_OFF = -Wno-narrowing -Wno-unused-result -Wno-narrowing -Wno-literal-suffix

RELEASE_FLAGS = -Ofast

ifeq ($(strip $(WARNINGS)),YES)
	CXXFLAGS += $(WARN_ON)
else
	CXXFLAGS += $(WARN_OFF) 
	CUFLAGS += -Xcompiler=-Wno-narrowing,-Wno-unused-result,-Wno-narrowing,-Wno-literal-suffix -Xcudafe "--diag_suppress=subscript_out_of_range"
endif

ifeq ($(strip $(DEBUG_LEVEL)),1)
	CXXFLAGS += -g -O3 -D_NVTX
	CUFLAGS += -g -O3 -D_NVTX
else 
ifeq ($(strip $(DEBUG_LEVEL)),2)
	CXXFLAGS += -g -O0 #-D_NVTX
	CUFLAGS += -g -O0 #-D_NVTX
else
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
	CXX = mpicxx
	FLAGS += -D_MPI
	INCS += -I$(strip $(METIS_INC_DIR))/ -I$(strip $(MPI_INC_DIR))/
	LIBS += -L$(strip $(METIS_LIB_DIR))/ -lmetis -Wl,-rpath=$(strip $(METIS_LIB_DIR))
ifneq ($(MPI_LIB_DIR),)
	LIBS += -L$(MPI_LIB_DIR)/ -lmpi -Wl,-rpath=$(MPI_LIB_DIR)
endif
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

ifeq ($(CU),)
  CUFLAGS += -arch=sm_20
else
	
ifeq ($(strip $(SM)),FERMI)
  CUFLAGS += -arch=sm_20
endif

ifeq ($(strip $(SM)),KEPLER)
  CUFLAGS += -arch=sm_35
endif

endif
endif

# Including external template libraries
INCS += -I$(CURDIR)/external/tnt/
INCS += -I$(CURDIR)/external/jama/
INCS += -I$(strip $(AUX_DIR))/

SRCDIR = $(CURDIR)/src
BINDIR = $(CURDIR)/bin
SWIGDIR = $(CURDIR)/swig_bin

TARGET = zefr
OBJS = $(BINDIR)/elements.o $(BINDIR)/faces.o $(BINDIR)/funcs.o $(BINDIR)/geometry.o $(BINDIR)/hexas.o $(BINDIR)/input.o $(BINDIR)/multigrid.o $(BINDIR)/points.o $(BINDIR)/polynomials.o $(BINDIR)/quads.o $(BINDIR)/solver.o $(BINDIR)/tris.o $(BINDIR)/filter.o  $(BINDIR)/zefr.o 

ifeq ($(strip $(ARCH)),GPU)
	OBJS += $(BINDIR)/elements_kernels.o $(BINDIR)/faces_kernels.o $(BINDIR)/solver_kernels.o  $(BINDIR)/filter_kernels.o
endif

SOBJS = $(OBJS) $(BINDIR)/zefr_interface.o

INCS += -I$(CURDIR)/include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) $(OBJS) $(LIBS) $(CXXFLAGS) -o $(BINDIR)/$(TARGET)

# Build Zefr as a Python extension module (shared library) using SWIG
.PHONY: swig
swig: FLAGS += -D_BUILD_LIB
swig: CXXFLAGS += -I$(TIOGA_INC_DIR)/ -fPIC
swig: INCS += -I$(TIOGA_INC_DIR)/
swig: CUFLAGS += -Xcompiler -fPIC
swig: $(SOBJS)
	@$(MAKE) -C $(SWIGDIR) CXX='$(CXX)' CU='$(CU)' SOBJS='$(SOBJS)' BINDIR='$(BINDIR)' FLAGS='$(FLAGS)' CXXFLAGS='$(CXXFLAGS)' INCS='$(INCS)' LIBS='$(LIBS)' PYTHON_INC_DIR='$(PYTHON_INC_DIR)' MPI4PY_INC_DIR='$(MPI4PY_INC_DIR)' SWIG_BIN='$(SWIG_BIN)'

# Build Zefr as a static library
.PHONY: static
static: FLAGS += -D_BUILD_LIB
static: $(SOBJS)
	$(AR) $(BINDIR)/libzefr.a $(SOBJS)

# Build Zefr as a shared library
.PHONY: lib
lib: FLAGS += -D_BUILD_LIB
lib: CXXFLAGS += -fPIC
lib: CUFLAGS += -Xcompiler -fPIC
lib: $(SOBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(INCS) -shared -o $(BINDIR)/libzefr.so $(SOBJS) $(LIBS)

# Compile the testZefr wrapper program using dynamic linking
.PHONY: test
test: INCS += -I$(SWIGDIR)/ -I$(TIOGA_INC_DIR)/
test: lib
	cp $(BINDIR)/libzefr.so $(SWIGDIR)/lib/
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCS) $(SWIGDIR)/testZefr.cpp -o $(SWIGDIR)/testZefr -L$(TIOGA_LIB_DIR)/ -L$(SWIGDIR)/lib -lzefr -ltioga -Wl,-rpath=$(SWIGDIR)/lib -Wl,-rpath=$(TIOGA_LIB_DIR) $(LIBS)

# Compile the testZefr wrapper program using static linking
.PHONY: test_static
test_static: INCS += -I$(SWIGDIR)/ -I$(TIOGA_INC_DIR)/
test_static: static
	cp $(BINDIR)/libzefr.a $(SWIGDIR)/lib/
	$(CXX) $(CXXFLAGS) $(FLAGS) $(INCS) $(SWIGDIR)/testZefr.cpp $(SWIGDIR)/lib/libzefr.a $(TIOGA_LIB_DIR)/libtioga.a -o $(SWIGDIR)/testZefr $(LIBS)

# Implicit Rules
$(BINDIR)/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p bin
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS)

ifeq ($(strip $(ARCH)),GPU)
$(BINDIR)/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS) -D_NO_TNT
endif

clean:
	rm -f $(BINDIR)/$(TARGET) $(BINDIR)/*.o $(BINDIR)/*.a $(SWIGDIR)/*.so $(SWIGDIR)/*.o $(SWIGDIR)/zefr.pyc $(SWIGDIR)/zefr.py

