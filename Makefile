ifeq ($(CONFIG),)
include configfiles/default.config
else
include $(CONFIG)
endif

CXX = g++
AR = ar -rvs
CU = nvcc
CXXFLAGS = -std=c++11 -Ofast -Wno-unknown-pragmas
CUFLAGS = -arch=sm_20 -O3 -use_fast_math --default-stream per-thread

WARN_ON = -Wall -Wextra -Wconversion
WARN_OFF = -Wno-narrowing -Wno-unused-result -Wno-narrowing -Wno-literal-suffix

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

SRCDIR = $(CURDIR)/src
BINDIR = $(CURDIR)/bin
SWIGDIR = $(CURDIR)/swig_bin

TARGET = zefr
OBJS = $(BINDIR)/elements.o $(BINDIR)/faces.o $(BINDIR)/funcs.o $(BINDIR)/geometry.o $(BINDIR)/hexas.o $(BINDIR)/input.o $(BINDIR)/multigrid.o $(BINDIR)/points.o $(BINDIR)/polynomials.o $(BINDIR)/quads.o $(BINDIR)/solver.o $(BINDIR)/filter.o  $(BINDIR)/zefr.o 

ifeq ($(strip $(ARCH)),GPU)
	OBJS += $(BINDIR)/elements_kernels.o $(BINDIR)/faces_kernels.o $(BINDIR)/solver_kernels.o  $(BINDIR)/filter_kernels.o
endif

SOBJS = $(OBJS) $(BINDIR)/zefr_interface.o

INCS += -I$(CURDIR)/include

$(TARGET): $(OBJS)
	$(CXX) $(INCS) $(OBJS) $(LIBS) $(CXXFLAGS) -o $(BINDIR)/$(TARGET)

# Build the Python extension module (shared library) using SWIG
.PHONY: swig
swig: FLAGS += -D_BUILD_LIB
swig: CXXFLAGS += -fPIC
swig: $(SOBJS)
	@$(MAKE) -C $(SWIGDIR) CXX='$(CXX)' CU='$(CU)' SOBJS='$(SOBJS)' BINDIR='$(BINDIR)' FLAGS='$(FLAGS)' CXXFLAGS='$(CXXFLAGS)' INCS='$(INCS)' LIBS='$(LIBS)'

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

