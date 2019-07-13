ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CONFIG),)
$(error ERROR: CONFIG not specified.)
endif
endif

include $(CONFIG)

ifeq ($(CXX),)
	CXX = g++
endif

ifeq ($(CC),)
	CC = gcc
endif

CU = nvcc

AR = ar -rvs

ifeq ($(strip $(SWIG_BIN)),)
	SWIG = swig -c++ -python
else
	SWIG = $(SWIG_BIN)/swig -c++ -python
endif

CXXFLAGS = -std=c++11 -fPIC -Wno-unknown-pragmas -fsigned-char #-fstack-protector-all
CCFLAGS = -std=c99 -w -fPIC -fsigned-char
CUFLAGS = -std=c++11 --default-stream per-thread $(EXTRA_CUFLAGS) -Xcompiler -fPIC -Xcompiler -fsigned-char
DFLAGS =

WARN_ON = -Wall -Wextra -Wconversion
WARN_OFF = -Wno-narrowing
ifneq ($(strip $(CC)),icc)
WARN_OFF += -Wno-unused-result -Wno-literal-suffix
endif

RELEASE_FLAGS = -Ofast
FLAGS = $(AUX_FLAGS)

ifeq ($(strip $(WARNINGS)),YES)
	CXXFLAGS += $(WARN_ON)
else
	CXXFLAGS += $(WARN_OFF)
	CUFLAGS += -Xcompiler=-Wno-narrowing -Xcudafe "--diag_suppress=subscript_out_of_range"
	# If compiling on Ubuntu 16.04 with default GCC, this might be needed:
	CUFLAGS += -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES
endif

ifeq ($(strip $(DEBUG_LEVEL)),1)
	CCFLAGS += -g -O3
	CXXFLAGS += -g -O3
	CUFLAGS += -g -O3
else
ifeq ($(strip $(DEBUG_LEVEL)),2)
	CCFLAGS += -g -O0
	CXXFLAGS += -g -O0
	CUFLAGS += -g -G -O0
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
	CCFLAGS += -fopenmp
endif

ifeq ($(strip $(USE_GIMMIK)),NO)
	DFLAGS += -D_NO_GIMMIK
endif

# Setting BLAS flags
ifeq ($(strip $(BLAS_DIST)),STANDARD)
	LIBS = -L$(BLAS_ROOT)/lib -lblas
endif

ifeq ($(strip $(BLAS_DIST)),OPENBLAS)
	LIBS = -L$(BLAS_ROOT)/lib -lopenblas
endif

ifeq ($(strip $(BLAS_DIST)),MKL)
	LIBS = -L$(BLAS_ROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
	DFLAGS += -D_MKL_BLAS
endif

ifeq ($(strip $(BLAS_DIST)),ESSL)
	LIBS = -L$(BLAS_ROOT)/lib64 -lessl -lm
	DFLAGS += -D_ESSL_BLAS
endif


INCS = -I$(strip $(BLAS_ROOT))/include

# Setting MPI/METIS flags
ifeq ($(strip $(USE_MPI)),YES)
	DFLAGS += -D_MPI
	INCS += -I$(strip $(MPI_ROOT))/include -I$(strip $(METIS_ROOT))/include
	LIBS += -L$(strip $(METIS_ROOT))/lib -lmetis -Wl,-rpath=$(strip $(METIS_ROOT))/lib

ifeq ($(strip $(MPI_DIST)),OMPI)
	LIBS += -L$(strip $(MPI_ROOT))/lib -lmpi -Wl,-rpath=$(MPI_ROOT)/lib
endif

ifeq ($(strip $(MPI_DIST)),INTEL)
	CXXFLAGS += -traceback -fp-model fast=2
	CFLAGS += -traceback -fp-model fast=2
	LIBS += -L$(strip $(MPI_ROOT))/lib -lmpich_intel -Wl,-rpath=$(MPI_ROOT)/lib
endif

ifeq ($(strip $(MPI_DIST)),IBM)
	LIBS += -L$(strip $(MPI_ROOT))/lib -lmpi_ibm -Wl,-rpath=$(MPI_ROOT)/lib
endif

# CUDA-aware MPI capability
ifeq ($(strip $(MPI_CUDA_AWARE)),YES)
	DFLAGS += -D_CUDA_AWARE
endif

endif

# Setting HDF5 flags
INCS += -I$(strip $(HDF5_ROOT))/include
LIBS += -L$(strip $(HDF5_ROOT))/lib -lhdf5 -lhdf5_cpp -Wl,-rpath=$(strip $(HDF5_ROOT))/lib

# Setting Architecture flags
ifeq ($(strip $(TARGET_ARCH)),CPU)
	DFLAGS += -D_CPU
endif

ifeq ($(strip $(TARGET_ARCH)),GPU)
	DFLAGS += -D_GPU
ifeq ($(strip $(CUDA_NVTX)),YES)
        DFLAGS += -D_NVTX
endif
	LIBS += -L$(strip $(CUDA_ROOT))/lib64 -lcudart -lcublas -lnvToolsExt -Wl,-rpath=$(strip $(CUDA_ROOT))/lib64
	INCS += -I$(strip $(CUDA_ROOT))/include
	CUFLAGS += -arch=sm_$(CUDA_CC)
endif

# Including external template libraries
FLAGS += -DEIGEN_MPL2_ONLY
INCS += -I$(CURDIR)/external/
INCS += -I$(CURDIR)/external/Eigen/
INCS += -I$(strip $(AUX_DIR))/

SRCDIR = $(CURDIR)/src
ifeq ($(strip $(BUILDDIR)),)
	BINDIR = $(CURDIR)/bin
else
	BINDIR = $(CURDIR)/$(BUILDDIR)
endif

SWIGDIR = $(CURDIR)/swig_bin
ifeq ($(strip $(TIOGA_ROOT)),)
	TIOGA_ROOT = $(CURDIR)/external/tioga/
endif
ifneq ($(strip $(TIOGA_CONFIG)),)
    TG_CONFIG = $(strip $(TIOGA_ROOT))/$(strip $(TIOGA_CONFIG))
endif

TARGET = zefr
OBJS = $(BINDIR)/elements.o $(BINDIR)/faces.o $(BINDIR)/funcs.o $(BINDIR)/geometry.o $(BINDIR)/hexas.o $(BINDIR)/input.o $(BINDIR)/multigrid.o $(BINDIR)/points.o $(BINDIR)/polynomials.o $(BINDIR)/prisms.o $(BINDIR)/quads.o $(BINDIR)/solver.o $(BINDIR)/tets.o $(BINDIR)/tris.o $(BINDIR)/filter.o  $(BINDIR)/zefr.o

ifeq ($(strip $(TARGET_ARCH)),CPU)
	OBJS += $(BINDIR)/gimmik_cpu.o
endif
ifeq ($(strip $(TARGET_ARCH)),GPU)
  OBJS += $(BINDIR)/aux_kernels.o $(BINDIR)/elements_kernels.o $(BINDIR)/faces_kernels.o $(BINDIR)/solver_kernels.o  $(BINDIR)/filter_kernels.o $(BINDIR)/gimmik_gpu.o
endif

SOBJS = $(OBJS) $(BINDIR)/zefr_interface.o
SWIG_OBJ = $(BINDIR)/zefr_swig.o
SWIG_TARGET = $(BINDIR)/_zefr.so
CONVERT_OBJ = external/convert_swig.o
CONVERT_TARGET = external/_convert.so
SWIG_INCS = -I$(strip $(PYTHON_INC_DIR))/ -I$(strip $(MPI4PY_INC_DIR))/ -I$(strip $(NUMPY_INC_DIR))/
SWIG_LIBS =
WRAP_TARGET = $(BINDIR)/zefrWrap
INTERP_TARGET = $(BINDIR)/gridInterp

INCS += -I$(CURDIR)/include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) $(OBJS) $(LIBS) $(CXXFLAGS) $(DFLAGS) -o $(BINDIR)/$(TARGET)

# Build Zefr as a Python extension module (shared library) using SWIG
.PHONY: swig
swig: FLAGS += -D_BUILD_LIB
swig: CXXFLAGS += -I$(strip $(TIOGA_ROOT))/include/ $(SWIG_INCS)
swig: INCS += -I$(strip $(TIOGA_ROOT))/include/
swig: $(SOBJS) $(SWIG_OBJ) $(CONVERT_OBJ)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DFLAGS) $(INCS) $(SWIG_INCS) -shared -o $(SWIG_TARGET) $(SOBJS) $(SWIG_OBJ) $(LIBS) $(SWIG_LIBS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DFLAGS) $(INCS) $(SWIG_INCS) -shared -o $(CONVERT_TARGET) $(CONVERT_OBJ)

# Build Zefr as a static library
.PHONY: static
static: FLAGS += -D_BUILD_LIB
static: $(SOBJS)
	$(AR) $(BINDIR)/libzefr.a $(SOBJS)

# Build Zefr as a shared library
.PHONY: shared
shared: FLAGS += -D_BUILD_LIB
shared: $(SOBJS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DFLAGS) $(INCS) -shared -o $(BINDIR)/libzefr.so $(SOBJS) $(LIBS)

# Compile the zefrWrap wrapper program using dynamic linking
.PHONY: wrap
wrap: INCS += -I$(strip $(TIOGA_ROOT))/ -I$(strip $(TIOGA_ROOT))/include
wrap: shared
	echo $(TIOGA_CONFIG)
	$(MAKE) -C external/tioga/ shared CONFIG=$(TIOGA_CONFIG)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/zefrWrap.cpp -o $(WRAP_TARGET) -L$(strip $(TIOGA_ROOT))/bin/ -L$(BINDIR)/ -lzefr -Wl,-rpath=$(BINDIR)/ -ltioga -Wl,-rpath=$(CURDIR)/$(strip $(TIOGA_ROOT))/bin/ $(LIBS)

# Compile the zefrWrap wrapper program using static linking
.PHONY: wrap_static
wrap_static: INCS += -I$(strip $(TIOGA_ROOT))/ -I$(strip $(TIOGA_ROOT))/include
wrap_static: static
	$(MAKE) -C external/tioga/ lib CONFIG=$(TIOGA_CONFIG)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/zefrWrap.cpp $(BINDIR)/libzefr.a $(strip $(TIOGA_ROOT))/bin/libtioga.a -o $(WRAP_TARGET) $(LIBS)

# Compile a wrapper that interpolates overset data to a single grid [dynamic linking]
.PHONY: interp
interp: INCS += -I$(strip $(TIOGA_ROOT))/ -I$(strip $(TIOGA_ROOT))/include
interp: shared
	$(MAKE) -C external/tioga/ shared CONFIG=$(TIOGA_CONFIG)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/gridInterp.cpp -o $(INTERP_TARGET) -L$(strip $(TIOGA_ROOT))/bin/ -L$(BINDIR)/ -lzefr -Wl,-rpath=$(BINDIR)/ -ltioga -Wl,-rpath=$(CURDIR)/$(strip $(TIOGA_ROOT))/bin/ $(LIBS)

# Compile a wrapper that interpolates overset data to a single grid [static linking]
.PHONY: interp_static
interp_static: INCS += -I$(strip $(TIOGA_ROOT))/ -I$(strip $(TIOGA_ROOT))/include
interp_static: static
	$(MAKE) -C external/tioga/ lib CONFIG=$(TIOGA_CONFIG)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(FLAGS) $(INCS) $(SRCDIR)/gridInterp.cpp $(BINDIR)/libzefr.a $(strip $(TIOGA_ROOT))/bin/libtioga.a -o $(INTERP_TARGET) $(LIBS)

# Implicit Rules
$(BINDIR)/%.o: src/%.cpp  include/*.hpp include/*.h
	@mkdir -p $(BINDIR)
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS) $(DFLAGS)

$(BINDIR)/%.o: src/%.c  include/*.hpp include/*.h
	@mkdir -p $(BINDIR)
	$(CC) $(INCS) -c -o $@ $< $(FLAGS) $(CCFLAGS) $(DFLAGS)

ifeq ($(strip $(TARGET_ARCH)),GPU)
$(BINDIR)/%.o: src/%.cu include/*.hpp include/*.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS) $(DFLAGS)
endif

$(BINDIR)/%_swig.cpp: include/%.i include/*.hpp
	$(SWIG) $(FLAGS) $(DFLAGS) $(INCS) $(SWIG_INCS) -o $@ $<

$(BINDIR)/%_swig.o: $(BINDIR)/%_swig.cpp
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS) $(DFLAGS)

external/%_swig.cpp: external/%.i
	$(SWIG) $(FLAGS) $(INCS) $(SWIG_INCS) -o $@ $<

external/%_swig.o: external/%_swig.cpp
	$(CXX) $(INCS) -c -o $@ $< $(FLAGS) $(CXXFLAGS) $(DFLAGS)

.PHONY: clean
clean:
	mv $(BINDIR)/gimmik_cpu.o $(BINDIR)/gimmik_cpu.o.bkup 2>/dev/null || :
	rm -f $(BINDIR)/$(TARGET) $(BINDIR)/*.o $(BINDIR)/*.a $(BINDIR)/*.so $(BINDIR)/zefr.pyc $(BINDIR)/zefr.py
	rm -f external/convert_swig.o external/_convert.so external/convert.py
	mv $(BINDIR)/gimmik_cpu.o.bkup $(BINDIR)/gimmik_cpu.o 2>/dev/null || :

.PHONY: cleanall
cleanall:
	rm -f $(BINDIR)/$(TARGET) $(BINDIR)/*.o $(BINDIR)/*.a $(BINDIR)/*.so $(BINDIR)/zefr.pyc $(BINDIR)/zefr.py
	rm -f external/convert_swig.o external/_convert.so external/convert.py

# Specific dependancy-tracking build rules
$(BINDIR)/gimmik_cpu.o: $(SRCDIR)/gimmik_cpu.c include/gimmik.h
ifeq ($(strip $(USE_GIMMIK)),YES)
	@echo ""
	@echo "*****************************************************"
	@echo "Compiling gimmik_cpu.o: This may take a long time..."
	@echo "*****************************************************"
	@echo ""
endif
	$(CC) $(INCS) -c -o $@ $< $(FLAGS) $(CCFLAGS) $(DFLAGS)

ifeq ($(strip $(TARGET_ARCH)),GPU)
$(BINDIR)/gimmik_gpu.o: $(SRCDIR)/gimmik_gpu.cu include/gimmik.h
	$(CU) $(INCS) -c -o $@ $< $(FLAGS) $(CUFLAGS) $(DFLAGS)
endif
